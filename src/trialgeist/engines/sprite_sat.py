"""Engine 1c: SPRITE-SAT, exact joint feasibility of (N, mean, SD, min, max) on a
bounded integer scale.

For each admissible integer sum S (GRIM) and each SD convention, the sum of
squares Q is confined to an integer window.  Then we ask CP-SAT whether
non-negative integer counts c_v over the scale {L..U} exist with

    sum c_v = N,  sum v*c_v = S,  Qlo <= sum v^2 c_v <= Qhi,
    c_min >= 1 and c_v = 0 below min (if the observed min is printed), same for max.

This is the exact problem (no relaxation), so CONSISTENT comes with a concrete
witness data set, and PROVEN_IMPOSSIBLE is re-checked independently with Z3 in
the certificate.
"""

from __future__ import annotations

from typing import Any

from fractions import Fraction
from math import ceil, floor

from ortools.sat.python import cp_model

from ..certificates import HULL_SRC, render
from ..numbers import parse_reported, proof_interval
from ..verdict import Finding, Provenance, Tier, make_finding
from .grim_grimmer import _q_window, grim_check, q_max, q_min, sum_range
from ..numbers import Interval

ENGINE = "summary_feasibility"
MAX_SOLVES = 20_000


def _units(x, g: Fraction) -> int:
    f = Fraction(str(x)) / g
    if f.denominator != 1:
        raise ValueError(f"{x} is not a multiple of granularity {g}")
    return int(f)


def solve_counts(n: int, s: int, qlo: int, qhi: int, lo: int, hi: int, omin: int | None, omax: int | None,
                 time_limit: float = 10.0) -> list[int] | None | str:
    m = cp_model.CpModel()
    vals = list(range(lo, hi + 1))
    c = [m.NewIntVar(0, n, f"c{v}") for v in vals]
    m.Add(sum(c) == n)
    m.Add(sum(v * x for v, x in zip(vals, c)) == s)
    q = sum(v * v * x for v, x in zip(vals, c))
    m.Add(q >= qlo)
    m.Add(q <= qhi)
    if omin is not None:
        m.Add(c[omin - lo] >= 1)
        for v in range(lo, omin):
            m.Add(c[v - lo] == 0)
    if omax is not None:
        m.Add(c[omax - lo] >= 1)
        for v in range(omax + 1, hi + 1):
            m.Add(c[v - lo] == 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [solver.Value(x) for x in c]
    if st == cp_model.INFEASIBLE:
        return None
    return "unknown"


def sprite_check(
    n: int,
    mean: str,
    sd: str,
    scale_min,
    scale_max,
    granularity: str | Fraction = Fraction(1),
    observed_min=None,
    observed_max=None,
    label: str = "",
    provenance: list[Provenance] | None = None,
) -> Finding:
    if n < 2:
        raise ValueError("an SD needs N >= 2")
    rm, rs = parse_reported(mean), parse_reported(sd)
    g = Fraction(str(granularity)) if not isinstance(granularity, Fraction) else granularity
    lo, hi = _units(scale_min, g), _units(scale_max, g)
    omin = None if observed_min is None else _units(observed_min, g)
    omax = None if observed_max is None else _units(observed_max, g)
    if omin is not None and omin < lo or omax is not None and omax > hi:
        raise ValueError("observed range outside the scale")
    if omin is not None:
        lo = max(lo, omin)
    if omax is not None:
        hi = min(hi, omax)
    inputs = {"label": label, "N": n, "mean": rm.text, "sd": rs.text, "granularity": str(g),
              "scale_min": str(scale_min), "scale_max": str(scale_max),
              "observed_min": None if observed_min is None else str(observed_min),
              "observed_max": None if observed_max is None else str(observed_max)}
    assumptions = [
        f"all N={n} participants contribute",
        f"every observation is a multiple of {g} within [{scale_min}, {scale_max}]",
        "SD is the sample (N-1) or population (N) SD; both are tried",
    ]
    if omin is not None or omax is not None:
        assumptions.append("printed min/max are the observed extremes of this same group")
    common: dict[str, Any] = dict(engine=ENGINE, check="sprite_sat", inputs=inputs, assumptions=assumptions, provenance=provenance)

    mh, sh = proof_interval(rm), proof_interval(rs, nonnegative=True)
    rng = sum_range(n, mh, g, lo * g, hi * g)
    if rng is None:
        return grim_check(n, rm, g, lo * g, hi * g, label, provenance)
    var_iv = Interval(sh.lo**2, sh.hi**2)
    solves = 0
    unknown = False
    for sample in (True, False):
        for s in range(rng[0], rng[1] + 1):
            qlo, qhi, _, _ = _q_window(n, s, var_iv, g, sample)
            qlo, qhi = max(qlo, q_min(n, s)), min(qhi, q_max(n, s, lo, hi))
            if qlo > qhi or (qlo == qhi and (qlo - s) % 2):
                continue
            solves += 1
            if solves > MAX_SOLVES:
                return make_finding(**common, tier=Tier.INCONCLUSIVE, title="SPRITE-SAT search too large",
                                    detail=f"More than {MAX_SOLVES} candidate sums.")
            res = solve_counts(n, s, qlo, qhi, lo, hi, omin, omax)
            if res == "unknown":
                unknown = True
                continue
            if res is not None:
                dist = {str(g * (lo + i)): k for i, k in enumerate(res) if k}
                return make_finding(**common, tier=Tier.CONSISTENT,
                                    title=f"SPRITE-SAT consistent: {label or rm.text}",
                                    detail=f"A data set exists ({'N-1' if sample else 'N'} SD).",
                                    witness={"S": s, "counts": dist, "variance": "sample" if sample else "population"},
                                    stats={"solves": solves})
    if unknown:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title="SPRITE-SAT timed out", detail="Solver returned UNKNOWN.")
    claim = (f"No data set of N={n} values on {{{scale_min}..{scale_max}}} (step {g})"
             + (f" with observed range [{observed_min}, {observed_max}]" if (omin is not None or omax is not None) else "")
             + f" has mean {rm.text} and SD {rs.text} under either SD formula.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE,
                     title=f"SPRITE-SAT impossible: {rm.text} ({rs.text}), N={n}, scale {scale_min}-{scale_max}" + (f" ({label})" if label else ""),
                     detail=claim, stats={"solves": solves})
    f.certificate = sprite_certificate(f.id, inputs, assumptions, claim, n, g, lo, hi, omin, omax)
    return f


def sprite_certificate(cid, inputs, assumptions, claim, n, g, lo, hi, omin, omax) -> str:
    body = f'''from fractions import Fraction as F
from math import ceil, floor
from z3 import Int, Solver, Sum, sat  # pip install z3-solver
N, MEAN, SD, G = {n}, {inputs["mean"]!r}, {inputs["sd"]!r}, F({str(g)!r})
LO, HI, OMIN, OMAX = {lo}, {hi}, {omin!r}, {omax!r}  # in units of G; LO/HI already clipped to observed range
{HULL_SRC}
mlo, mhi = hull(MEAN); slo, shi = hull(SD, nonneg=True)
smin, smax = max(ceil(mlo * N / G), N * LO), min(floor(mhi * N / G), N * HI)
vals = range(LO, HI + 1)
for denom, name in ((N * (N - 1), "N-1"), (N * N, "N")):
    for s in range(smin, smax + 1):
        qlo = ceil((s * s + denom * slo * slo / (G * G)) / N)  # N*Q - S^2 = denom * var / G^2
        qhi = floor((s * s + denom * shi * shi / (G * G)) / N)
        if qlo > qhi: continue
        c = {{v: Int(f"c{{v}}") for v in vals}}; sv = Solver()
        sv.add(*[x >= 0 for x in c.values()], Sum(list(c.values())) == N)
        sv.add(Sum([v * c[v] for v in vals]) == s)
        q = Sum([v * v * c[v] for v in vals]); sv.add(q >= qlo, q <= qhi)
        if OMIN is not None: sv.add(c[OMIN] >= 1)
        if OMAX is not None: sv.add(c[OMAX] >= 1)
        if sv.check() == sat:
            print(f"NOT VERIFIED: {{name}} S={{s}} {{sv.model()}}"); raise SystemExit(1)
        print(f"  {{name}} S={{s}}: Q in [{{qlo}}, {{qhi}}] -> UNSAT")
print(f"VERIFIED: no integer data set on [{{LO}}, {{HI}}] x {{G}} matches mean {{MEAN}}, SD {{SD}}, N={{N}}.")
'''
    return render(cid, "SPRITE-SAT", claim, inputs, assumptions, body)
