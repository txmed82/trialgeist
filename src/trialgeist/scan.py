"""End-to-end scan: extract claims -> exact engines -> self-verified report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .certificates import line_count, run_certificate
from .claims import ClaimSet, check_all
from .verdict import Finding, Tier

TIER_ORDER = [Tier.PROVEN_IMPOSSIBLE, Tier.STATISTICALLY_EXTREME, Tier.REGISTRY_DISCORDANT, Tier.INCONCLUSIVE, Tier.CONSISTENT]


@dataclass
class Report:
    source: str
    findings: list[Finding]
    notes: list[str] = field(default_factory=list)
    n_claims: int = 0

    def by_tier(self, tier: Tier) -> list[Finding]:
        return [f for f in self.findings if f.tier is tier]

    @property
    def proven(self) -> list[Finding]:
        return self.by_tier(Tier.PROVEN_IMPOSSIBLE)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            out.append(f)
    return out


def scan_pdf(pdf: str | Path, nct: str | None = None, km_eps_pt: float = 1.0, registry_fetch=None) -> Report:
    from .extract.km_vector import find_km_plots, plot_to_claims
    from .extract.pdf_text import extract_baseline, extract_claims

    pdf = str(pdf)
    claims = extract_claims(pdf) + extract_baseline(pdf)
    plots, notes = find_km_plots(pdf)
    for p in plots:
        claims += plot_to_claims(p, eps_pt=km_eps_pt)
    findings = check_all(ClaimSet(source=pdf, claims=claims))
    notes.append(f"{len(plots)} vector KM plot(s) with numbers at risk analysed")
    if nct:
        from .registry.diff import registry_findings
        from .extract.paper_outcomes import paper_from_pdf

        try:
            findings += registry_findings(nct, paper_from_pdf(pdf), fetch=registry_fetch)
        except Exception as e:  # network or parse failure must never become a finding
            notes.append(f"registry check skipped: {type(e).__name__}: {e}")
    return Report(pdf, _dedupe(findings), notes, len(claims))


def scan_xml(xml: str | Path, nct: str | None = None, registry_fetch=None) -> Report:
    """PMC JATS XML: exact table cells (no KM geometry in XML)."""
    from .extract.jats import extract_claims_xml, paper_text_xml

    xml = str(xml)
    claims = extract_claims_xml(xml)
    findings = check_all(ClaimSet(source=xml, claims=claims))
    notes = ["JATS XML: tables parsed from markup; figures (KM curves) need the PDF"]
    if nct:
        from .extract.paper_outcomes import paper_from_text
        from .registry.diff import registry_findings

        try:
            findings += registry_findings(nct, paper_from_text(paper_text_xml(xml)), fetch=registry_fetch)
        except Exception as e:
            notes.append(f"registry check skipped: {type(e).__name__}: {e}")
    return Report(xml, _dedupe(findings), notes, len(claims))


def scan_any(path: str | Path, nct: str | None = None, **kw) -> Report:
    p = str(path).lower()
    if p.endswith((".xml", ".nxml")):
        return scan_xml(path, nct=nct)
    if p.endswith(".json"):
        return scan_claims(path)
    return scan_pdf(path, nct=nct, **kw)


def scan_claims(path: str | Path) -> Report:
    cs = ClaimSet.model_validate_json(Path(path).read_text())
    return Report(str(path), _dedupe(check_all(cs)), [], len(cs.claims))


def self_verify(report: Report, workdir: Path) -> None:
    """Run every certificate; a proof whose certificate does not reproduce is withdrawn."""
    for f in report.proven:
        if not f.certificate:
            f.tier = Tier.INCONCLUSIVE
            report.notes.append(f"{f.id}: no certificate, withdrawn")
            continue
        ok, dt, out = run_certificate(f.certificate, workdir, f.id, timeout=30)
        f.stats["certificate_seconds"] = round(dt, 3)
        f.stats["certificate_lines"] = line_count(f.certificate)
        if not ok:
            f.tier = Tier.INCONCLUSIVE
            report.notes.append(f"{f.id}: certificate did not reproduce, withdrawn: {out[-300:]}")


def write_report(report: Report, out: str | Path, verify: bool = True) -> Path:
    out = Path(out)
    certdir = out / "certificates"
    certdir.mkdir(parents=True, exist_ok=True)
    if verify:
        self_verify(report, certdir)
    for f in report.proven:
        (certdir / f"certificate_{f.id}.py").write_text(f.certificate or "")
    (out / "report.json").write_text(json.dumps({
        "source": report.source,
        "n_claims": report.n_claims,
        "notes": report.notes,
        "counts": {t.value: len(report.by_tier(t)) for t in TIER_ORDER},
        "findings": [f.model_dump(mode="json") for f in report.findings],
    }, indent=2))
    (out / "report.md").write_text(render_markdown(report))
    return out


def render_markdown(report: Report) -> str:
    lines = [f"# TRIALGEIST report: {Path(report.source).name}", ""]
    lines.append("| Tier | Findings |\n|---|---|")
    for t in TIER_ORDER:
        lines.append(f"| {t.value} | {len(report.by_tier(t))} |")
    lines.append(f"\n{report.n_claims} printed claims were extracted and checked.\n")
    for t in TIER_ORDER[:3]:
        fs = report.by_tier(t)
        if not fs:
            continue
        lines.append(f"## {t.value}\n")
        for f in fs:
            lines.append(f"### {f.title}\n\n{f.detail}\n")
            if f.assumptions:
                lines.append("Assumptions: " + "; ".join(f.assumptions) + "\n")
            for p in f.provenance:
                loc = f"page {p.page + 1}" if p.page is not None else p.source
                bb = f", bbox {tuple(round(v, 1) for v in p.bbox)}" if p.bbox else ""
                lines.append(f"- Source: {loc}{bb}: `{(p.text or '')[:120]}`")
            if f.certificate:
                lines.append(f"- Certificate: `certificates/certificate_{f.id}.py` (run with `python3`)")
            lines.append("")
    if report.notes:
        lines.append("## Notes\n")
        lines += [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"
