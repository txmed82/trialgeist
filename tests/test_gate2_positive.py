"""Gate 2: real retracted trials with publicly documented numerical impossibilities.

research/positive_controls.json holds, for each control, the paper, the
public source documenting the problem, and the digits exactly as printed.
TRIALGEIST must (1) prove each documented impossibility from those digits and
emit a certificate that re-derives it in < 2 s, and (2) when the paper's
open-access PDF is in tests/corpus/positive/<id>/paper.pdf, rediscover the
same contradiction end to end from the PDF itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from trialgeist.certificates import line_count, run_certificate
from trialgeist.claims import Claim, check_claim
from trialgeist.verdict import Tier

ROOT = Path(__file__).parents[1]
DATA = ROOT / "research" / "positive_controls.json"
CONTROLS = json.loads(DATA.read_text())["controls"] if DATA.exists() else []
ADAPTER = TypeAdapter(Claim)


PROVE = [c for c in CONTROLS if c["expected"] == "PROVEN_IMPOSSIBLE"]
REFUSE = [c for c in CONTROLS if c["expected"] != "PROVEN_IMPOSSIBLE"]


def test_gate2_has_controls():
    papers = {c["paper"].get("doi") or c["paper"]["citation"] for c in PROVE}
    assert len(PROVE) >= 5 and len(papers) >= 3, (len(PROVE), len(papers))


@pytest.mark.parametrize("ctrl", PROVE, ids=[c["id"] for c in PROVE])
def test_gate2_documented_impossibility_is_proven(ctrl, tmp_path):
    findings = check_claim(ADAPTER.validate_python(ctrl["claim"]))
    proven = [f for f in findings if f.tier is Tier.PROVEN_IMPOSSIBLE]
    assert proven, [(f.tier.value, f.detail) for f in findings]
    for f in proven:
        assert line_count(f.certificate) < 80
        ok, dt, out = run_certificate(f.certificate, tmp_path, f.id)
        assert ok and dt < 2.0, out


@pytest.mark.parametrize("ctrl", REFUSE, ids=[c["id"] for c in REFUSE])
def test_gate2_conservative_refusals(ctrl):
    """Documented problems that some rounding convention explains must NOT be called proofs."""
    findings = check_claim(ADAPTER.validate_python(ctrl["claim"]))
    assert not any(f.tier is Tier.PROVEN_IMPOSSIBLE for f in findings)


POS = ROOT / "tests" / "corpus" / "positive"
# end-to-end: paper file (PDF or PMC XML) -> extraction -> the same printed digits proven impossible
E2E = [c for c in PROVE if c["paper"].get("pmcid") and c.get("e2e", True)
       and any((POS / c["paper"]["pmcid"] / f).exists() for f in ("paper.pdf", "paper.nxml"))]


def _scan(pmcid: str):
    from trialgeist.scan import scan_pdf, scan_xml

    d, out = POS / pmcid, []
    if (d / "paper.pdf").exists():
        out += scan_pdf(d / "paper.pdf").proven
    if (d / "paper.nxml").exists():
        out += scan_xml(d / "paper.nxml").proven
    return out


@pytest.mark.parametrize("ctrl", E2E, ids=[c["id"] for c in E2E])
def test_gate2_rediscovered_end_to_end(ctrl, tmp_path):
    claim = ctrl["claim"]
    want = [str(claim[k]) for k in ("percent", "mean", "overall_mean", "median") if k in claim]
    want += list(claim.get("percents", []))
    hits = [f for f in _scan(ctrl["paper"]["pmcid"])
            if any(w in (f.provenance[0].text if f.provenance else "") or w in str(f.inputs) for w in want)]
    assert hits, f"{ctrl['id']} not rediscovered from the paper file"
    ok, _, out = run_certificate(hits[0].certificate, tmp_path, hits[0].id)
    assert ok, out
