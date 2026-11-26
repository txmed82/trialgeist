"""Engine 1b: percentage-to-integer feasibility and count conservation.

* ``percent_check``: a printed percentage, optionally with its printed count.
  When a count is printed, every denominator D in [n, N] is tried before a
  mismatch is called impossible, so ordinary missing data never trips it.
* ``count_sum_check``: mutually exclusive, exhaustive category counts must add
  to the stated total (exact integers).
* ``percent_partition_check``: exhaustive categories reported only as
  percentages must admit integer counts k_i (k_i/N inside each printed
  percentage) that add to N.
"""

from __future__ import annotations

from typing import Any

from fractions import Fraction
from math import ceil, floor

from ..certificates import HULL_SRC, render
from ..numbers import parse_reported, proof_interval
from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "summary_feasibility"
HUNDRED = Fraction(100)


def _k_range(d: int, lo: Fraction, hi: Fraction) -> tuple[int, int] | None:
    a, b = max(0, ceil(lo * d / HUNDRED)), min(d, floor(hi * d / HUNDRED))
    return (a, b) if a <= b else None


def percent_check(
    total_n: int,
    percent: str,
    count: int | None = None,
    label: str = "",
    denominator_exact: bool = False,
    provenance: list[Provenance] | None = None,
) -> Finding:
    """Check a printed percentage (0-100 scale) against N and optional count."""
    r = parse_reported(percent)
    lo, hi = (lambda iv: (iv.lo, iv.hi))(proof_interval(r, nonnegative=True))
    inputs = {"label": label, "N": total_n, "percent": r.text, "count": count, "denominator_exact": denominator_exact}
    common: dict[str, Any] = dict(engine=ENGINE, check="percent", inputs=inputs, provenance=provenance)

    if count is not None:
        if count > total_n:
            claim = f"Count {count} exceeds the group size N={total_n}."
            f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Count exceeds N: {count} > {total_n}", detail=claim,
                             assumptions=[f"N={total_n} is the size of the group the count is drawn from"])
            f.certificate = _count_gt_cert(f.id, inputs, f.assumptions, claim)
            return f
        if count == 0:
            ok = lo <= 0
        else:
            dmin, dmax = (total_n, total_n) if denominator_exact else (count, total_n)
            # D with count/D*100 in [lo, hi]  <=>  D in [100*count/hi, 100*count/lo]
            d_lo = ceil(HUNDRED * count / hi) if hi > 0 else None
            d_hi = floor(HUNDRED * count / lo) if lo > 0 else total_n
            ok = d_lo is not None and max(d_lo, dmin) <= min(d_hi, dmax)
        if ok:
            return make_finding(**common, tier=Tier.CONSISTENT, title=f"Percent consistent: {count} ({r.text}%)",
                                detail="Some admissible denominator reproduces the printed percentage.")
        assumptions = [f"the percentage is {count}/D x 100 for a denominator D <= N={total_n} (D=N if denominator_exact)"]
        claim = (
            f"{count}/D x 100 never prints as {r.text}% for any denominator D in "
            f"[{total_n if denominator_exact else count}, {total_n}]."
        )
        f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Impossible percentage: {count} ({r.text}%) of N={total_n}",
                         detail=claim, assumptions=assumptions)
        f.certificate = _pct_count_cert(f.id, inputs, assumptions, claim)
        return f

    # Percentage only.
    if _k_range(total_n, lo, hi) is not None:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Percent consistent: {r.text}% of {total_n}",
                            detail="Some integer count reproduces the printed percentage.")
    if not denominator_exact:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title=f"Percent {r.text}% not k/{total_n}",
                            detail="No integer k gives k/N; the denominator may exclude missing data, so this is not a proof.")
    assumptions = [f"denominator is exactly N={total_n}"]
    claim = f"No integer k in [0, {total_n}] has k/{total_n} x 100 printing as {r.text}%."
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Impossible percentage: {r.text}% of N={total_n}", detail=claim,
                     assumptions=assumptions)
    f.certificate = _pct_only_cert(f.id, inputs, assumptions, claim)
    return f


def _count_gt_cert(cid, inputs, assumptions, claim) -> str:
    body = f'''N, COUNT = {inputs["N"]}, {inputs["count"]}
if COUNT <= N: print("NOT VERIFIED"); raise SystemExit(1)
print(f"VERIFIED: count {{COUNT}} > group size {{N}}.")
'''
    return render(cid, "COUNT_EXCEEDS_N", claim, inputs, assumptions, body)


def _pct_count_cert(cid, inputs, assumptions, claim) -> str:
    body = f'''from fractions import Fraction as F
N, PCT, K, EXACT = {inputs["N"]}, {inputs["percent"]!r}, {inputs["count"]}, {inputs["denominator_exact"]}
{HULL_SRC}
lo, hi = hull(PCT, nonneg=True)
ds = [N] if EXACT else range(max(K, 1), N + 1)
hits = [d for d in ds if lo <= F(100 * K, d) <= hi]
for d in (list(ds)[:1] + list(ds)[-1:]):
    print(f"  {{K}}/{{d}} = {{float(F(100 * K, d)):.6f}}%")
if hits: print(f"NOT VERIFIED: D={{hits[0]}} works"); raise SystemExit(1)
print(f"VERIFIED: {{K}}/D x 100 lies outside [{{float(lo):.8f}}, {{float(hi):.8f}}] for every D tried.")
'''
    return render(cid, "PERCENT_COUNT", claim, inputs, assumptions, body)


def _pct_only_cert(cid, inputs, assumptions, claim) -> str:
    body = f'''from fractions import Fraction as F
N, PCT = {inputs["N"]}, {inputs["percent"]!r}
{HULL_SRC}
lo, hi = hull(PCT, nonneg=True)
hits = [k for k in range(N + 1) if lo <= F(100 * k, N) <= hi]
if hits: print(f"NOT VERIFIED: k={{hits[0]}}"); raise SystemExit(1)
print(f"VERIFIED: no k/{{N}} x 100 in [{{float(lo):.8f}}, {{float(hi):.8f}}].")
'''
    return render(cid, "PERCENT_ONLY", claim, inputs, assumptions, body)


def count_sum_check(total: int, counts: list[int], label: str = "", provenance: list[Provenance] | None = None,
                    exhaustive: bool = True) -> Finding:
    """Mutually exclusive categories must add to the total (or not exceed it if not exhaustive)."""
    s = sum(counts)
    inputs = {"label": label, "total": total, "counts": list(counts), "exhaustive": exhaustive}
    bad = s != total if exhaustive else s > total
    common: dict[str, Any] = dict(engine=ENGINE, check="count_sum", inputs=inputs, provenance=provenance)
    if not bad:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Counts add up: {label}", detail=f"sum {s} vs total {total}")
    assumptions = ["categories are mutually exclusive" + (" and exhaustive" if exhaustive else "")]
    claim = f"Category counts {counts} sum to {s}, but the stated total is {total}."
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Counts do not add up: {s} vs {total} ({label})",
                     detail=claim, assumptions=assumptions)
    body = f'''TOTAL, COUNTS, EXHAUSTIVE = {total}, {list(counts)}, {exhaustive}
s = sum(COUNTS)
print(f"  sum({{COUNTS}}) = {{s}}; stated total = {{TOTAL}}")
if (s != TOTAL) if EXHAUSTIVE else (s > TOTAL): print("VERIFIED: counts are inconsistent with the total.")
else: print("NOT VERIFIED"); raise SystemExit(1)
'''
    f.certificate = render(f.id, "COUNT_SUM", claim, inputs, assumptions, body)
    return f


def percent_partition_check(total_n: int, percents: list[str], label: str = "",
                            provenance: list[Provenance] | None = None) -> Finding:
    """Exhaustive categories given only as percentages of exactly N."""
    rs = [parse_reported(p) for p in percents]
    ranges = []
    for r in rs:
        iv = proof_interval(r, nonnegative=True)
        ranges.append(_k_range(total_n, iv.lo, iv.hi))
    inputs = {"label": label, "N": total_n, "percents": [r.text for r in rs]}
    common: dict[str, Any] = dict(engine=ENGINE, check="percent_partition", inputs=inputs, provenance=provenance)
    assumptions = [f"categories are mutually exclusive and exhaustive, each a share of exactly N={total_n}"]
    ok = all(rg is not None for rg in ranges) and sum(a for a, _ in ranges) <= total_n <= sum(b for _, b in ranges)
    if ok:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"Percentages partition N: {label}",
                            detail="Integer counts exist that reproduce every percentage and sum to N.")
    claim = f"No integer counts reproduce percentages {[r.text for r in rs]} of N={total_n} and add up to N."
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"Percentages cannot partition N={total_n} ({label})",
                     detail=claim, assumptions=assumptions)
    body = f'''from fractions import Fraction as F
from math import ceil, floor
N, PCTS = {total_n}, {[r.text for r in rs]!r}
{HULL_SRC}
lo_sum = hi_sum = 0
for p in PCTS:
    lo, hi = hull(p, nonneg=True)
    a, b = max(0, ceil(lo * N / 100)), min(N, floor(hi * N / 100))
    print(f"  {{p}}% -> k in [{{a}}, {{b}}]")
    if a > b: print(f"VERIFIED: {{p}}% is not k/{{N}} for any integer k."); raise SystemExit(0)
    lo_sum, hi_sum = lo_sum + a, hi_sum + b
if lo_sum <= N <= hi_sum: print("NOT VERIFIED"); raise SystemExit(1)
print(f"VERIFIED: admissible counts sum to [{{lo_sum}}, {{hi_sum}}], which excludes N={{N}}.")
'''
    f.certificate = render(f.id, "PERCENT_PARTITION", claim, inputs, assumptions, body)
    return f
