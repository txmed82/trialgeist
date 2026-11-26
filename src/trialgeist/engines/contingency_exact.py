"""Engine 1d: exact 2x2 contingency compatibility.

Arms of size n1, n2 with event counts a, c (printed exactly, or as
percentages of the arm).  Every integer table consistent with the printed
counts/percentages is enumerated and tested against:

* crude risk ratio / odds ratio / risk difference point estimates (only when
  the paper says the estimate is crude/unadjusted; orientation may be either way)
* a printed p-value under every plausible test (Fisher exact: min-likelihood,
  doubled, mid-p; Pearson chi-square with and without Yates; unpooled Wald z;
  each optionally one-sided).

Fisher p-values are exact rationals.  Normal/chi-square tail probabilities use
``math.erfc`` with a 1e-9 relative + 1e-12 absolute acceptance margin, far above
double rounding error, so a borderline floating evaluation is never a proof.
"""

from __future__ import annotations

from typing import Any

import math
import re
from fractions import Fraction
from math import comb

from ..certificates import HULL_SRC, render
from ..numbers import Interval, parse_reported, proof_interval
from ..verdict import Finding, Provenance, Tier, make_finding
from .counts import _k_range

ENGINE = "summary_feasibility"
REL, ABS = 1e-9, 1e-12
MAX_TABLES = 5000

_P_RE = re.compile(r"^\s*[pP]?\s*(?P<op><=|>=|≤|≥|<|>|=)?\s*(?P<num>.+?)\s*$")


def parse_pvalue(text: str) -> Interval:
    """'0.03' -> rounding hull; '<0.001' -> [0, 0.001]; '>0.99' -> [0.99, 1]."""
    m = _P_RE.match(text.replace("·", "."))
    if not m:
        raise ValueError(text)
    op, num = m.group("op"), m.group("num")
    r = parse_reported(num)
    if op in ("<", "≤", "<="):
        return Interval(Fraction(0), r.value * (1 + Fraction(1, 10**9)))
    if op in (">", "≥", ">="):
        return Interval(r.value * (1 - Fraction(1, 10**9)), Fraction(1))
    iv = proof_interval(r, nonnegative=True)
    return Interval(iv.lo, min(iv.hi, Fraction(1)) if iv.lo <= 1 else iv.hi)


def fisher_pvalues(a: int, n1: int, c: int, n2: int) -> dict[str, Fraction]:
    m = a + c  # total events
    lo, hi = max(0, m - n2), min(n1, m)
    tot = comb(n1 + n2, m)
    probs = {x: Fraction(comb(n1, x) * comb(n2, m - x), tot) for x in range(lo, hi + 1)}
    po = probs[a]
    left = sum((p for x, p in probs.items() if x <= a), Fraction(0))
    right = sum((p for x, p in probs.items() if x >= a), Fraction(0))
    minlike = sum((p for p in probs.values() if p <= po), Fraction(0))
    minlike_tol = sum((p for p in probs.values() if p <= po * (1 + Fraction(1, 10**7))), Fraction(0))
    return {
        "fisher_minlike": minlike,
        "fisher_minlike_tol": minlike_tol,
        "fisher_doubled": min(Fraction(1), 2 * min(left, right)),
        "fisher_midp": min(Fraction(1), 2 * (min(left, right) - po / 2)),
        "fisher_one_sided": min(left, right),
    }


def normal_pvalues(a: int, n1: int, c: int, n2: int) -> dict[str, float]:
    out: dict[str, float] = {}
    n = n1 + n2
    m, f = a + c, n - a - c
    if m == 0 or f == 0:
        return out
    b, d = n1 - a, n2 - c
    x2 = Fraction(n * (a * d - b * c) ** 2, n1 * n2 * m * f)
    out["chi2"] = math.erfc(math.sqrt(float(x2) / 2))
    yates_num = max(abs(a * d - b * c) - Fraction(n, 2), Fraction(0)) ** 2
    out["chi2_yates"] = math.erfc(math.sqrt(float(n * yates_num / (n1 * n2 * m * f)) / 2))
    p1, p2 = Fraction(a, n1), Fraction(c, n2)
    se2 = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    if se2 > 0:
        z = abs(float(p1 - p2)) / math.sqrt(float(se2))
        out["wald_z"] = math.erfc(z / math.sqrt(2))
    for k in list(out):
        out[k + "_one_sided"] = out[k] / 2
    return out


def _in(iv: Interval, x) -> bool:
    if isinstance(x, Fraction):
        return iv.lo <= x <= iv.hi
    lo, hi = float(iv.lo), float(iv.hi)
    return lo * (1 - REL) - ABS <= x <= hi * (1 + REL) + ABS


def _estimates(a, n1, c, n2) -> dict[str, Fraction | None]:
    b, d = n1 - a, n2 - c
    return {
        "rr": Fraction(a * n2, n1 * c) if a and c else None,
        "or": Fraction(a * d, b * c) if a and b and c and d else None,
        "rd": Fraction(a, n1) - Fraction(c, n2),
    }


def contingency_check(
    n1: int,
    n2: int,
    events1: int | None = None,
    events2: int | None = None,
    pct1: str | None = None,
    pct2: str | None = None,
    rr: str | None = None,
    odds_ratio: str | None = None,
    rd_percent: str | None = None,
    estimate_is_crude: bool = False,
    p_value: str | None = None,
    tests: tuple[str, ...] | None = None,
    label: str = "",
    provenance: list[Provenance] | None = None,
) -> Finding:
    def cands(n, ev, pct):
        if ev is not None:
            return [ev] if 0 <= ev <= n else []
        if pct is not None:
            iv = proof_interval(pct, nonnegative=True)
            rg = _k_range(n, iv.lo, iv.hi)
            return list(range(rg[0], rg[1] + 1)) if rg else []
        return list(range(0, n + 1))

    A, C = cands(n1, events1, pct1), cands(n2, events2, pct2)
    inputs = {"label": label, "n1": n1, "n2": n2, "events1": events1, "events2": events2, "pct1": pct1, "pct2": pct2,
              "rr": rr, "or": odds_ratio, "rd_percent": rd_percent, "estimate_is_crude": estimate_is_crude,
              "p_value": p_value, "tests": list(tests) if tests else None}
    assumptions = [f"denominators are the full arms (n1={n1}, n2={n2})"]
    est_ivs: dict[str, Interval] = {}
    if estimate_is_crude:
        if rr:
            est_ivs["rr"] = proof_interval(rr, nonnegative=True)
        if odds_ratio:
            est_ivs["or"] = proof_interval(odds_ratio, nonnegative=True)
        if rd_percent:
            est_ivs["rd"] = proof_interval(rd_percent).scale(Fraction(1, 100))
        if est_ivs:
            assumptions.append("printed estimates are crude (unadjusted, unstratified); either arm may be the reference")
    p_iv = parse_pvalue(p_value) if p_value else None
    if p_iv:
        assumptions.append("p-value comes from one of: " + ", ".join(tests or ["Fisher (min-lik, doubled, mid-p)", "chi-square +/- Yates", "Wald z", "or one-sided versions"]))
    common: dict[str, Any] = dict(engine=ENGINE, check="contingency", inputs=inputs, assumptions=assumptions, provenance=provenance)
    if not est_ivs and not p_iv:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title="Nothing to test", detail="No crude estimate or p-value.")
    if len(A) * len(C) > MAX_TABLES:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title="Too many candidate tables", detail=f"{len(A) * len(C)} tables")
    if not A or not C:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title="Percent not reproducible by any count",
                            detail="Handled by the percent engine.")
    for a in A:
        for c in C:
            est = _estimates(a, n1, c, n2)
            ok = True
            for k, iv in est_ivs.items():
                v = est[k]
                if v is None:
                    continue  # undefined with zero cells: never used as evidence
                inv = (-v if k == "rd" else 1 / v) if v != 0 else v
                if not (_in(iv, v) or _in(iv, inv)):
                    ok = False
                    break
            if ok and p_iv:
                ps: dict = {**fisher_pvalues(a, n1, c, n2), **normal_pvalues(a, n1, c, n2)}
                if tests:
                    ps = {k: v for k, v in ps.items() if any(k.startswith(t) for t in tests)}
                ok = any(_in(p_iv, v) for v in ps.values())
            if ok:
                return make_finding(**common, tier=Tier.CONSISTENT, title=f"2x2 consistent: {label}",
                                    detail=f"Table a={a}/{n1}, c={c}/{n2} reproduces the printed statistics.",
                                    witness={"a": a, "c": c})
    claim = (f"No 2x2 table with arms {n1}/{n2} and the printed event counts/percentages reproduces "
             + ", ".join(f"{k.upper()} {inputs['rr' if k == 'rr' else 'or' if k == 'or' else 'rd_percent']}" for k in est_ivs)
             + (f" p={p_value}" if p_iv else "") + ".")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"2x2 table impossible ({label})", detail=claim,
                     stats={"tables_checked": len(A) * len(C)})
    f.certificate = _certificate(f.id, inputs, assumptions, claim, A, C)
    return f


def _certificate(cid, inputs, assumptions, claim, A, C) -> str:
    body = f'''import math
from fractions import Fraction as F
from math import comb
N1, N2, A, C = {inputs["n1"]}, {inputs["n2"]}, {A!r}, {C!r}  # candidate event counts from printed values
EST = {{k: v for k, v in (("rr", {inputs["rr"]!r}), ("or", {inputs["or"]!r}), ("rd", {inputs["rd_percent"]!r})) if v and {inputs["estimate_is_crude"]}}}
P, TESTS = {inputs["p_value"]!r}, {inputs["tests"]!r}
{HULL_SRC}
def p_iv(t):
    t = t.replace("\\u00b7", ".").lstrip("Pp= "); op = t[0] if t[0] in "<>\\u2264\\u2265" else ""
    v = F(t.lstrip("<>=\\u2264\\u2265 ")); e = F(1, 10**9)
    return (F(0), v * (1 + e)) if op in "<\\u2264" and op else (v * (1 - e), F(1)) if op else hull(t, True)
def inside(iv, x):
    if isinstance(x, F): return iv[0] <= x <= iv[1]
    return float(iv[0]) * (1 - 1e-9) - 1e-12 <= x <= float(iv[1]) * (1 + 1e-9) + 1e-12
def pvals(a, c):
    b, d, n, m = N1 - a, N2 - c, N1 + N2, a + c; f = n - m
    pr = {{x: F(comb(N1, x) * comb(N2, m - x), comb(n, m)) for x in range(max(0, m - N2), min(N1, m) + 1)}}
    lft = sum(p for x, p in pr.items() if x <= a); rgt = sum(p for x, p in pr.items() if x >= a); po = pr[a]
    out = {{"fisher_minlike": sum(p for p in pr.values() if p <= po), "fisher_minlike_tol": sum(p for p in pr.values() if p <= po * (1 + F(1, 10**7))),
           "fisher_doubled": min(F(1), 2 * min(lft, rgt)), "fisher_midp": min(F(1), 2 * (min(lft, rgt) - po / 2)), "fisher_one_sided": min(lft, rgt)}}
    if m and f:
        nn = {{"chi2": math.erfc(math.sqrt(float(F(n * (a * d - b * c) ** 2, N1 * N2 * m * f)) / 2)),
              "chi2_yates": math.erfc(math.sqrt(float(n * max(abs(a * d - b * c) - F(n, 2), F(0)) ** 2 / (N1 * N2 * m * f)) / 2))}}
        se2 = F(a, N1) * F(b, N1) / N1 + F(c, N2) * F(d, N2) / N2
        if se2: nn["wald_z"] = math.erfc(abs(float(F(a, N1) - F(c, N2))) / math.sqrt(float(se2)) / math.sqrt(2))
        out.update(nn); out.update({{k + "_one_sided": v / 2 for k, v in nn.items()}})
    return {{k: v for k, v in out.items() if not TESTS or any(k.startswith(t) for t in TESTS)}}
for a in A:
    for c in C:
        b, d = N1 - a, N2 - c
        est = {{"rr": F(a * N2, N1 * c) if a and c else None, "or": F(a * d, b * c) if a and b and c and d else None, "rd": F(a, N1) - F(c, N2)}}
        ok = True
        for k, txt in EST.items():
            v = est[k]
            if v is None: continue
            iv = hull(txt, k != "rd"); iv = (iv[0] / 100, iv[1] / 100) if k == "rd" else iv
            ok = ok and (inside(iv, v) or inside(iv, -v if k == "rd" else (1 / v if v else v)))
        if ok and P: ok = any(inside(p_iv(P), v) for v in pvals(a, c).values())
        if ok: print(f"NOT VERIFIED: a={{a}}, c={{c}} fits"); raise SystemExit(1)
print(f"VERIFIED: none of {{len(A) * len(C)}} candidate tables reproduces the printed statistics.")
'''
    return render(cid, "CONTINGENCY_2x2", claim, inputs, assumptions, body)
