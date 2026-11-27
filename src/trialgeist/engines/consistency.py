"""Cross-table consistency checks that need no integrality.

* ``mean_partition_check``: an overall mean over N = sum(n_i) participants
  must equal the n-weighted average of the subgroup means that partition it.
* ``median_range_check``: a median must lie inside the printed range.

Both are sound for real-valued data; the assumption (subgroups partition the
overall group; median and range describe the same sample) is written into
the finding and certificate.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from ..certificates import HULL_SRC, render
from ..numbers import parse_reported, proof_interval
from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "consistency"


def mean_partition_check(overall_mean: str, parts: list[tuple[int, str]], label: str = "",
                         provenance: list[Provenance] | None = None, weights_exact: bool = True) -> Finding:
    """weights_exact=False: per-row missing data makes the weights unknown, so the
    pooled mean is only known to be SOME convex combination of the subgroup means,
    i.e. it lies between the smallest and largest of them."""
    if not weights_exact:
        return _mean_hull_check(overall_mean, [m for _, m in parts], label, provenance)
    ro = parse_reported(overall_mean)
    rp = [(n, parse_reported(m)) for n, m in parts]
    N = sum(n for n, _ in rp)
    oi = proof_interval(ro)
    lo = sum(n * proof_interval(r).lo for n, r in rp) / N
    hi = sum(n * proof_interval(r).hi for n, r in rp) / N
    inputs = {"label": label, "overall_mean": ro.text, "parts": [[n, r.text] for n, r in rp]}
    assumptions = ["the subgroups are disjoint and together make up exactly the overall group",
                   "every mean is computed over all members of its group (no missing values)"]
    common: dict[str, Any] = dict(engine=ENGINE, check="mean_partition", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if not (oi.hi < lo or oi.lo > hi):
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Pooled mean consistent ({label or ro.text})",
                            detail=f"weighted subgroup means give [{float(lo):.4f}, {float(hi):.4f}]")
    claim = (f"Subgroup means {', '.join(f'{r.text} (n={n})' for n, r in rp)} force the overall mean into "
             f"[{float(lo):.4f}, {float(hi):.4f}], but it is printed as {ro.text}.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Overall mean {ro.text} contradicts its subgroups" +
                     (f" ({label})" if label else ""), detail=claim)
    body = f'''from fractions import Fraction as F
OVERALL, PARTS = {ro.text!r}, {[[n, r.text] for n, r in rp]!r}  # (n, printed mean) per subgroup
{HULL_SRC}
N = sum(n for n, _ in PARTS)
lo = sum(n * hull(m)[0] for n, m in PARTS) / N  # smallest possible pooled mean
hi = sum(n * hull(m)[1] for n, m in PARTS) / N  # largest possible pooled mean
olo, ohi = hull(OVERALL)
print(f"  pooled mean must lie in [{{float(lo):.6f}}, {{float(hi):.6f}}]; printed {{OVERALL}} means [{{float(olo):.6f}}, {{float(ohi):.6f}}]")
if not (ohi < lo or olo > hi): print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: the overall mean cannot be the weighted average of its subgroup means.")
'''
    f.certificate = render(f.id, "MEAN_PARTITION", claim, inputs, assumptions, body)
    return f


_PRINTS_SRC = """def prints(x, r):  # exact x prints as r under some convention: rounding of any tie rule, or truncation
    v, d = F(r), len(r.split(".")[1]) if "." in r else 0
    u = F(1, 10 ** d)
    if v - u / 2 <= x <= v + u / 2: return True
    if r.lstrip().startswith(("-", "\u2212")): return v - u < x <= v
    return v <= x < v + u or (v == 0 and -u < x <= 0)"""


def _prints(x: Fraction, r) -> bool:
    from ..numbers import convention_interval, PROOF_CONVENTIONS

    return any(convention_interval(r, c).contains(x) for c in PROOF_CONVENTIONS)


def median_range_check(median: str, rmin: str, rmax: str, label: str = "", granularity: str | None = None,
                       provenance: list[Provenance] | None = None, bounds: str = "range") -> Finding:
    """min <= median <= max.  With integer-valued data (granularity g), the
    median is a multiple of g/2 and min/max are multiples of g, which are exact
    values: no cascade or float slack can move them, so each candidate is
    tested against every rounding convention directly."""
    m, a, b = parse_reported(median), parse_reported(rmin), parse_reported(rmax)
    inputs = {"label": label, "median": m.text, "min": a.text, "max": b.text, "granularity": granularity}
    what = "interquartile range (Q1-Q3)" if bounds == "iqr" else "range"
    assumptions = [f"median and {what} describe the same sample" + (" (Q1 <= median <= Q3 under every quantile definition)" if bounds == "iqr" else "")]
    if granularity:
        g = Fraction(granularity)
        assumptions.append(f"every observation is a multiple of {granularity} (so the median is a multiple of {g / 2})")

        def grid(r, step):
            k0 = int((r.value - 2 * r.unit) / step) - 1
            return [k * step for k in range(k0, k0 + int(4 * r.unit / step) + 3) if _prints(k * step, r)]

        meds, mins, maxs = grid(m, g / 2), grid(a, g), grid(b, g)
        feasible = any(lo <= md <= hi for md in meds for lo in mins for hi in maxs)
        why = (f"medians printing as {m.text}: {[str(x) for x in meds]}; minima printing as {a.text}: "
               f"{[str(x) for x in mins]}; maxima printing as {b.text}: {[str(x) for x in maxs]}")
    else:
        mi, ai, bi = proof_interval(m), proof_interval(a), proof_interval(b)
        feasible = mi.hi >= ai.lo and mi.lo <= bi.hi and ai.lo <= bi.hi
        why = ""
    common: dict[str, Any] = dict(engine=ENGINE, check="median_range", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if feasible:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Median inside range ({label or m.text})", detail=why)
    claim = f"A median of {m.text} cannot lie outside its own {what} {a.text} to {b.text}." + (f" ({why})" if why else "")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Median {m.text} outside range {a.text}-{b.text}" +
                     (f" ({label})" if label else ""), detail=claim)
    if granularity:
        body = f'''from fractions import Fraction as F
MED, MIN, MAX, G = {m.text!r}, {a.text!r}, {b.text!r}, F({granularity!r})
{_PRINTS_SRC}
def grid(r, step):  # exact grid values printing as r (half-up/down/even, truncation)
    c = F(r) / step
    return [k * step for k in range(int(c) - 4 * int(1 / step + 1) - 2, int(c) + 4 * int(1 / step + 1) + 3) if prints(k * step, r)]
meds, mins, maxs = grid(MED, G / 2), grid(MIN, G), grid(MAX, G)
print("  medians:", [str(x) for x in meds], " minima:", [str(x) for x in mins], " maxima:", [str(x) for x in maxs])
if any(lo <= md <= hi for md in meds for lo in mins for hi in maxs): print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: no median printing as", MED, "lies between a minimum printing as", MIN, "and a maximum printing as", MAX)
'''
    else:
        body = f'''from fractions import Fraction as F
MED, MIN, MAX = {m.text!r}, {a.text!r}, {b.text!r}
{HULL_SRC}
(mlo, mhi), (alo, _), (_, bhi) = hull(MED), hull(MIN), hull(MAX)
print(f"  median <= {{float(mhi)}}, minimum >= {{float(alo)}}, maximum <= {{float(bhi)}}")
if mhi >= alo and mlo <= bhi and alo <= bhi: print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: min <= median <= max is violated.")
'''
    f.certificate = render(f.id, "MEDIAN_RANGE", claim, inputs, assumptions, body)
    return f


def _mean_hull_check(overall_mean: str, means: list[str], label: str, provenance) -> Finding:
    ro, rs = parse_reported(overall_mean), [parse_reported(m) for m in means]
    oi = proof_interval(ro)
    lo, hi = min(proof_interval(r).lo for r in rs), max(proof_interval(r).hi for r in rs)
    inputs = {"label": label, "overall_mean": ro.text, "subgroup_means": [r.text for r in rs]}
    assumptions = ["the subgroups are disjoint and together make up the overall group (any amount of missing data allowed)"]
    common: dict[str, Any] = dict(engine=ENGINE, check="mean_hull", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if not (oi.hi < lo or oi.lo > hi):
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Overall mean between subgroup means ({label or ro.text})",
                            detail=f"[{float(lo):.4f}, {float(hi):.4f}]")
    claim = (f"A pooled mean is a weighted average of its subgroup means ({', '.join(r.text for r in rs)}), so it must "
             f"lie in [{float(lo):.4f}, {float(hi):.4f}] whatever the group sizes; it is printed as {ro.text}.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Overall mean {ro.text} outside its subgroup means" +
                     (f" ({label})" if label else ""), detail=claim)
    body = f'''from fractions import Fraction as F
OVERALL, MEANS = {ro.text!r}, {[r.text for r in rs]!r}
{HULL_SRC}
lo, hi = min(hull(m)[0] for m in MEANS), max(hull(m)[1] for m in MEANS)  # any weights: pooled mean in [lo, hi]
olo, ohi = hull(OVERALL)
print(f"  subgroup means allow [{{float(lo):.6f}}, {{float(hi):.6f}}]; printed {{OVERALL}} means [{{float(olo):.6f}}, {{float(ohi):.6f}}]")
if not (ohi < lo or olo > hi): print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: no weighting of the subgroup means gives the printed overall mean.")
'''
    f.certificate = render(f.id, "MEAN_HULL", claim, inputs, assumptions, body)
    return f


def arm_total_check(total: int, parts: list[int], total_n: int, arm_ns: list[int], label: str = "",
                    provenance: list[Provenance] | None = None) -> Finding:
    """A Total column whose header N is the sum of the arm Ns is the union of the
    arms, so every row count in it is the sum of the arm counts.  Exact integers."""
    inputs = {"label": label, "total_count": total, "arm_counts": list(parts), "total_n": total_n, "arm_ns": list(arm_ns)}
    assumptions = [f"the Total column (N={total_n}) is the union of the arm columns (N={' + '.join(map(str, arm_ns))})",
                   "the row counts the same characteristic in every column"]
    common: dict[str, Any] = dict(engine=ENGINE, check="arm_total", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if sum(parts) == total:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Total column adds up ({label})", detail="")
    claim = f"Arm counts {' + '.join(map(str, parts))} = {sum(parts)}, but the Total column prints {total}."
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Total {total} is not the sum of its arms ({sum(parts)})" +
                     (f" ({label})" if label else ""), detail=claim)
    body = f'''TOTAL_N, ARM_NS, TOTAL, ARMS = {total_n}, {list(arm_ns)}, {total}, {list(parts)}
print(f"  header Ns: {{' + '.join(map(str, ARM_NS))}} = {{sum(ARM_NS)}} (Total column N = {{TOTAL_N}})")
print(f"  row counts: {{' + '.join(map(str, ARMS))}} = {{sum(ARMS)}} (Total column prints {{TOTAL}})")
if sum(ARM_NS) != TOTAL_N or sum(ARMS) == TOTAL: print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: the Total column is the union of the arms, yet its count is not the sum of theirs.")
'''
    f.certificate = render(f.id, "ARM_TOTAL", claim, inputs, assumptions, body)
    return f


def common_denominator_check(counts: list[int], percents: list[str], d_max: int | None = None, label: str = "",
                             provenance: list[Provenance] | None = None) -> Finding:
    """Rows of one variable, each printed as n (%), share one denominator D >=
    max(counts) (<= d_max when the column N is confirmed).  The rows need not be
    mutually exclusive (multi-response lists are fine).  Each cell can be fine
    on its own while no single D reproduces all of them."""
    from math import ceil, floor

    rs = [parse_reported(p) for p in percents]
    ivs = [proof_interval(r, nonnegative=True) for r in rs]
    d_lo = max(1, max(counts))
    d_hi = d_max
    for k, iv in zip(counts, ivs):  # any positive cell bounds D from above
        if k > 0 and iv.lo > 0:
            b = floor(100 * k / iv.lo)
            d_hi = b if d_hi is None else min(d_hi, b)
    inputs = {"label": label, "counts": list(counts), "percents": [r.text for r in rs], "d_max": d_max}
    assumptions = ["the rows describe one variable in one column, so their percentages share one whole-number denominator D"
                   + (f" of at most the column N ({d_max})" if d_max else "")]
    common: dict[str, Any] = dict(engine=ENGINE, check="common_denominator", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if d_hi is None or d_hi - d_lo > 200_000:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title=f"Common denominator unbounded ({label})", detail="")

    def fits(d: int) -> bool:
        return all(iv.lo * d <= 100 * k <= iv.hi * d for k, iv in zip(counts, ivs))

    if any(fits(d) for d in range(d_lo, d_hi + 1)):
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Common denominator exists ({label})", detail="")
    claim = (f"No whole-number denominator D in [{d_lo}, {d_hi}] reproduces all of "
             f"{', '.join(f'{k} ({r.text}%)' for k, r in zip(counts, rs))} at once.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title="Category percentages share no denominator" +
                     (f" ({label})" if label else ""), detail=claim)
    body = f'''from fractions import Fraction as F
from math import floor
COUNTS, PCTS, D_MAX = {list(counts)}, {[r.text for r in rs]!r}, {d_max!r}
{HULL_SRC}
H = [hull(p, nonneg=True) for p in PCTS]
d_lo, d_hi = max(1, max(COUNTS)), D_MAX
for k, (lo, hi) in zip(COUNTS, H):
    if k > 0 and lo > 0: d_hi = floor(100 * k / lo) if d_hi is None else min(d_hi, floor(100 * k / lo))
ok = [d for d in range(d_lo, d_hi + 1) if all(lo * d <= 100 * k <= hi * d for k, (lo, hi) in zip(COUNTS, H))]
print(f"  denominators tried: {{d_lo}}..{{d_hi}}; reproducing all cells: {{ok[:5]}}")
if ok: print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: the category percentages cannot share a denominator.")
'''
    f.certificate = render(f.id, "COMMON_DENOMINATOR", claim, inputs, assumptions, body)
    return f


__all__ = ["arm_total_check", "common_denominator_check", "mean_partition_check", "median_range_check", "Fraction"]
