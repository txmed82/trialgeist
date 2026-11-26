"""Engine 4: ClinicalTrials.gov historical time-machine diff.

Data sources (see research/ctgov_history_api.md, verified 2026-11-26):

* ``/api/v2/studies/{nct}``      current record (authoritative dates)
* ``/api/int/studies/{nct}/history``       version index (needs ncbi_sid cookie,
  obtained from any v2 call on the same client)
* ``/api/int/studies/{nct}/history/{v}``   one historical snapshot

Findings are ``REGISTRY_DISCORDANT``: dated, linkable evidence, never an
accusation of misconduct.  Every finding names the registry versions,
their posting dates and the exact text compared.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "registry"
BASE = "https://clinicaltrials.gov"
NCT_RE = re.compile(r"^NCT\d{8}$")

Fetch = Callable[[str], Any]  # path -> parsed JSON


# ------------------------------------------------------------------ fetching


def http_fetcher(cache_dir: Path | None = None, timeout: float = 30.0) -> Fetch:
    import time

    import httpx

    client = httpx.Client(base_url=BASE, follow_redirects=True, timeout=timeout,
                          headers={"User-Agent": "trialgeist/0.1 (+https://github.com/; forensic research)"})
    primed = False

    def fetch(path: str):
        nonlocal primed
        key = cache_dir / (path.strip("/").replace("/", "__") + ".json") if cache_dir else None
        if key and key.exists():
            return json.loads(key.read_text())
        if not primed and path.startswith("/api/int/"):
            client.get("/api/v2/version")  # sets the ncbi_sid cookie the WAF wants
            primed = True
        for attempt in range(5):
            r = client.get(path)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if key:
                key.parent.mkdir(parents=True, exist_ok=True)
                key.write_text(json.dumps(data))
            time.sleep(0.35)  # <= 3 req/s
            return data
        raise RuntimeError(f"ClinicalTrials.gov kept failing for {path}")

    return fetch


def fixture_fetcher(directory: Path) -> Fetch:
    """Offline fetcher over saved fixtures named NCT..._{current,history,vN}.json."""

    def fetch(path: str):
        m = re.match(r"/api/v2/studies/(NCT\d+)$", path)
        if m:
            return json.loads((directory / f"{m.group(1)}_current.json").read_text())
        m = re.match(r"/api/int/studies/(NCT\d+)/history$", path)
        if m:
            return json.loads((directory / f"{m.group(1)}_history.json").read_text())
        m = re.match(r"/api/int/studies/(NCT\d+)/history/(\d+)$", path)
        if m:
            return json.loads((directory / f"{m.group(1)}_v{m.group(2)}.json").read_text())
        raise FileNotFoundError(path)

    return fetch


# ------------------------------------------------------------------ registry model


def _unesc(x):
    if isinstance(x, str):
        return html.unescape(x)
    if isinstance(x, list):
        return [_unesc(v) for v in x]
    if isinstance(x, dict):
        return {k: _unesc(v) for k, v in x.items()}
    return x


def parse_date(s: str | None, end_of_period: bool = False) -> date | None:
    if not s:
        return None
    parts = [int(p) for p in s.split("-")]
    if len(parts) == 3:
        return date(*parts)
    if len(parts) == 2:
        y, m = parts
        if not end_of_period:
            return date(y, m, 1)
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        return date.fromordinal(nxt.toordinal() - 1)
    return date(parts[0], 12 if end_of_period else 1, 31 if end_of_period else 1)


@dataclass
class Outcome:
    measure: str
    time_frame: str = ""
    description: str = ""

    @property
    def text(self) -> str:
        return f"{self.measure} [{self.time_frame}]"


@dataclass
class Snapshot:
    version: int | None  # None = current record
    posted: date | None
    primary: list[Outcome]
    secondary: list[Outcome]
    enrollment: int | None
    enrollment_type: str | None
    start: str | None
    primary_completion: str | None

    @property
    def url(self) -> str:
        return ""


def snapshot_from_protocol(ps: dict, version: int | None, posted: date | None) -> Snapshot:
    ps = _unesc(ps)
    om = ps.get("outcomesModule") or {}
    conv = lambda xs: [Outcome(o.get("measure", ""), o.get("timeFrame", ""), o.get("description", "")) for o in xs or []]
    ei = (ps.get("designModule") or {}).get("enrollmentInfo") or {}
    sm = ps.get("statusModule") or {}
    return Snapshot(version, posted, conv(om.get("primaryOutcomes")), conv(om.get("secondaryOutcomes")),
                    ei.get("count"), ei.get("type"), (sm.get("startDateStruct") or {}).get("date"),
                    (sm.get("primaryCompletionDateStruct") or {}).get("date"))


def select_version(changes: list[dict], target: date) -> tuple[int, bool]:
    """Max version posted on or before target; (v0, True) if none (retrospective / pre-archive)."""
    ok = [c["version"] for c in changes if parse_date(c["date"]) <= target]
    if ok:
        return max(ok), False
    return min(c["version"] for c in changes), True


@dataclass
class RegistryView:
    nct: str
    current: Snapshot
    at_start: Snapshot
    at_primary_completion: Snapshot
    start_flag: bool
    pc_flag: bool
    changes: list[dict]
    outcome_change_versions: list[tuple[int, str]] = field(default_factory=list)


def load_registry(nct: str, fetch: Fetch) -> RegistryView:
    if not NCT_RE.match(nct):
        raise ValueError(f"not an NCT id: {nct}")
    cur = fetch(f"/api/v2/studies/{nct}")
    current = snapshot_from_protocol(cur["protocolSection"], None, None)
    hist = fetch(f"/api/int/studies/{nct}/history")
    changes = sorted(hist["changes"], key=lambda c: c["version"])
    start = parse_date(current.start)
    pc = parse_date(current.primary_completion, end_of_period=True)
    if start is None or pc is None:
        raise ValueError("current record lacks start or primary completion date")
    v_s, f_s = select_version(changes, start)
    v_p, f_p = select_version(changes, pc)
    posted = {c["version"]: parse_date(c["date"]) for c in changes}

    def snap(v):
        return snapshot_from_protocol(fetch(f"/api/int/studies/{nct}/history/{v}")["study"]["protocolSection"], v, posted[v])

    at_s = snap(v_s)
    at_p = at_s if v_p == v_s else snap(v_p)
    outcome_changes = [(c["version"], c["date"]) for c in changes
                       if "Outcome Measures" in c.get("moduleLabels", []) and v_s < c["version"] <= v_p]
    return RegistryView(nct, current, at_s, at_p, f_s, f_p, changes, outcome_changes)


# ------------------------------------------------------------------ text matching

STOP = set("""a an the of to in on at for and or with by from as is are was were be been this that these those
patients patient participants subjects subject proportion percentage number rate rates change score measured
assessed defined using day days week weeks month months year years time baseline""".split())
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.4375, "year": 365.25, "hour": 1 / 24}
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "twelve": 12, "eighteen": 18, "twenty-four": 24, "thirty": 30}
DUR_RE = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(WORDNUM) + r")[\s-]*(hour|day|week|month|year)s?\b", re.I)


SYNONYMS = [
    (r"\ball[- ]cause mortality\b|\bdeath from any cause\b|\bany-cause death\b", " death "),
    (r"\bmortality\b|\bdied\b|\bdeaths?\b", " death "),
    (r"\bglycated h(a)?emoglobin\b|\bh(b)?a1c\b|\bglycosylated h(a)?emoglobin\b", " hba1c "),
    (r"\bmajor adverse cardiovascular events?\b|\bmace\b", " mace "),
    (r"\bmodified rankin( scale)?\b|\bmrs\b", " mrs "),
    (r"\bprogression[- ]free survival\b|\bpfs\b", " pfs "),
    (r"\boverall survival\b", " osurv "),
    (r"\boswestry disability index\b|\bodi\b", " odi "),
    (r"\bnumeric(al)? rating scale\b|\bnrs\b", " nrs "),
    (r"\bsurgical[- ]site infections?\b|\bssi\b", " ssi "),
    (r"\bventilator[- ]free days?\b", " vfd "),
]


def normalise(s: str) -> str:
    s = " " + s.lower() + " "
    for rx, rep in SYNONYMS:
        s = re.sub(rx, rep, s)
    return s


def tokens(s: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]+", normalise(s))
    out = set()
    for w in words:
        if w in STOP or len(w) < 3:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def durations_days(s: str) -> set[float]:
    out = set()
    for num, unit in DUR_RE.findall(s):
        v = float(num) if num[0].isdigit() else float(WORDNUM[num.lower()])
        out.add(round(v * UNIT_DAYS[unit.lower()], 1))
    return out


# ------------------------------------------------------------------ paper side


@dataclass
class PaperClaims:
    primary_outcomes: list[str] = field(default_factory=list)  # full sentence(s) naming the primary outcome
    randomized_n: int | None = None
    provenance: list[Provenance] = field(default_factory=list)


# ------------------------------------------------------------------ diff


MATCH = 0.6  # token overlap (of the shorter side) to call two outcome texts the same
DIFFERENT = 0.25  # overlap below which a paper outcome is clearly not the registered one


def _link(nct: str, v: int | None) -> str:
    return f"{BASE}/study/{nct}" + ("" if v is None else f"?tab=history&a={v + 1}")


def registry_diff(view: RegistryView, paper: PaperClaims) -> list[Finding]:
    out: list[Finding] = []
    s, p = view.at_start, view.at_primary_completion
    base_inputs = {"nct": view.nct, "version_at_start": s.version, "posted_at_start": str(s.posted),
                   "version_at_primary_completion": p.version, "posted_at_primary_completion": str(p.posted),
                   "start_date": view.current.start, "primary_completion_date": view.current.primary_completion}
    caveats = []
    if view.start_flag:
        caveats.append("no registry version existed on the start date (retrospective registration or pre-2005 archive); "
                       "the first posted version was used")
    prov = [Provenance(source="ctgov", text=_link(view.nct, s.version), note=f"version {s.version} posted {s.posted}")]
    if p.version != s.version:
        prov.append(Provenance(source="ctgov", text=_link(view.nct, p.version), note=f"version {p.version} posted {p.posted}"))

    # 1. registered primary outcome changed after enrollment began
    if s.primary and p.primary:
        moved = [o for o in s.primary if max((similarity(o.measure, q.measure) for q in p.primary), default=0) < MATCH]
        if moved:
            out.append(make_finding(
                engine=ENGINE, check="primary_changed_after_start", tier=Tier.REGISTRY_DISCORDANT,
                title=f"{view.nct}: registered primary outcome changed after enrollment began",
                detail=(f"At the start of enrollment (version {s.version}, posted {s.posted}) the primary outcome was: "
                        + "; ".join(o.text for o in s.primary)
                        + f". By primary completion (version {p.version}, posted {p.posted}) it was: "
                        + "; ".join(o.text for o in p.primary) + "."
                        + (f" Outcome-measure edits were posted in versions {[v for v, _ in view.outcome_change_versions]}." if view.outcome_change_versions else "")),
                inputs={**base_inputs, "primary_at_start": [o.text for o in s.primary], "primary_at_completion": [o.text for o in p.primary]},
                assumptions=caveats + ["a change may be legitimate and publicly justified; this is evidence of a change, not of misconduct"],
                provenance=prov))

    # 2. paper's primary outcome vs registered primaries/secondaries at start
    for sent in paper.primary_outcomes:
        reg_p = [(max(similarity(sent, o.measure), similarity(sent, o.measure + " " + o.description)), o) for o in s.primary]
        reg_s = [(similarity(sent, o.measure), o) for o in s.secondary]
        best_p = max(reg_p, key=lambda x: x[0], default=(0.0, None))
        best_s = max(reg_s, key=lambda x: x[0], default=(0.0, None))
        if not s.primary:
            continue
        if best_p[0] < MATCH:
            swapped = best_s[0] >= MATCH and best_s[0] >= best_p[0] + 0.3
            if not swapped and best_p[0] >= DIFFERENT:
                continue  # partial overlap: paraphrase is likelier than a new endpoint, stay silent
            out.append(make_finding(
                engine=ENGINE, check="secondary_reported_as_primary" if swapped else "unregistered_primary",
                tier=Tier.REGISTRY_DISCORDANT,
                title=(f"{view.nct}: paper's primary outcome was a registered secondary outcome" if swapped
                       else f"{view.nct}: paper's primary outcome does not match the registered primary"),
                detail=(f"Paper: \"{sent[:300]}\". Registered primary at enrollment start (version {s.version}, {s.posted}): "
                        + "; ".join(o.text for o in s.primary)
                        + (f". Closest registered secondary: {best_s[1].text}." if swapped and best_s[1] else ".")),
                inputs={**base_inputs, "paper_primary": sent, "overlap_primary": round(best_p[0], 2), "overlap_secondary": round(best_s[0], 2)},
                assumptions=caveats + ["outcome texts are compared by content-word overlap; paraphrase can cause misses, not false matches"],
                provenance=prov + paper.provenance))
            continue
        # 3. time-frame mismatch for the matched primary
        o = best_p[1]
        dp, dr = durations_days(sent), durations_days(o.time_frame + " " + o.measure)
        if dp and dr and not (dp & dr) and max(dr) >= 2 * max(dp):
            out.append(make_finding(
                engine=ENGINE, check="primary_timeframe_changed", tier=Tier.REGISTRY_DISCORDANT,
                title=f"{view.nct}: primary outcome reported at a different time horizon than registered",
                detail=(f"Paper: \"{sent[:300]}\" (horizon(s) {sorted(dp)} days). Registered at start (version {s.version}): "
                        f"{o.text} (horizon(s) {sorted(dr)} days)."),
                inputs={**base_inputs, "paper_primary": sent, "registered": o.text},
                assumptions=caveats + ["time horizons are parsed from explicit durations in the text"],
                provenance=prov + paper.provenance))

    # 4. sample size
    if paper.randomized_n and s.enrollment:
        ratio = paper.randomized_n / s.enrollment
        if ratio < 0.75 or ratio > 1.33:
            out.append(make_finding(
                engine=ENGINE, check="sample_size_changed", tier=Tier.REGISTRY_DISCORDANT,
                title=f"{view.nct}: randomized {paper.randomized_n} vs {s.enrollment} planned at enrollment start",
                detail=(f"Registry version {s.version} (posted {s.posted}) planned enrollment {s.enrollment} "
                        f"({s.enrollment_type or 'type not given'}); the paper reports {paper.randomized_n} randomized "
                        f"({ratio:.0%} of plan). Final registry enrollment: {view.current.enrollment} ({view.current.enrollment_type})."),
                inputs={**base_inputs, "paper_n": paper.randomized_n, "planned_n": s.enrollment, "final_registry_n": view.current.enrollment},
                assumptions=caveats + ["a justified, documented change (e.g. interim analysis, pandemic) is not misconduct"],
                provenance=prov + paper.provenance))
    return out


def registry_findings(nct: str, paper: PaperClaims, fetch: Fetch | None = None, cache_dir: Path | None = None) -> list[Finding]:
    fetch = fetch or http_fetcher(cache_dir or Path.home() / ".cache" / "trialgeist" / "ctgov")
    return registry_diff(load_registry(nct, fetch), paper)
