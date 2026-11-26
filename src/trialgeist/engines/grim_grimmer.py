"""Engine 1a: GRIM (mean) and GRIMMER (SD) feasibility with exact arithmetic.

Data model: every observation is ``g * y`` with ``y`` an integer (``g`` is the
granularity: 1 for counts and Likert items, 1/k for the mean of k integer
items, 0.5 for half-point scales).  With ``S = sum(y)`` and ``Q = sum(y^2)``:

* mean   = g * S / N
* var    = g^2 * (N*Q - S^2) / (N * (N - 1))    (sample, "N-1")
* var    = g^2 * (N*Q - S^2) / N^2              (population, "N")

Necessary conditions used for proofs (each is a relaxation, never stronger
than reality):

* S is an integer inside the mean's proof interval (GRIM).
* Q is an integer with Q = S (mod 2) because y^2 = y (mod 2) (GRIMMER parity).
* Q >= Qmin(N, S): the most even integer split of S (exact minimum).
* Q <= Qmax(N, S, L, U) on a bounded scale: extreme-point split (exact max).
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil, floor

from ..certificates import HULL_SRC, render
from ..numbers import Convention, Interval, PROOF_CONVENTIONS, Reported, convention_interval, parse_reported, proof_interval
from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "summary_feasibility"
MAX_S_CANDIDATES = 2_000_000


def _frac(x) -> Fraction | None:
    if x is None:
        return None
    return Fraction(str(x)) if not isinstance(x, Fraction) else x


def sum_range(n: int, mean_iv: Interval, g: Fraction, lo_bound: Fraction | None = None, hi_bound: Fraction | None = None) -> tuple[int, int] | None:
    """Integer S (in units of g) with g*S/N inside ``mean_iv`` (respecting open ends)."""
    a, b = mean_iv.lo * n / g, mean_iv.hi * n / g
    smin, smax = ceil(a), floor(b)
    if not mean_iv.lo_closed and smin == a:
        smin += 1
    if not mean_iv.hi_closed and smax == b:
        smax -= 1
    if lo_bound is not None:
        smin = max(smin, ceil(lo_bound * n / g))
    if hi_bound is not None:
        smax = min(smax, floor(hi_bound * n / g))
    return (smin, smax) if smin <= smax else None


def q_min(n: int, s: int) -> int:
    q, r = divmod(s, n)
    return r * (q + 1) ** 2 + (n - r) * q**2


def q_max(n: int, s: int, lo: int, hi: int) -> int:
    """Max sum of squares of n integers in [lo, hi] summing to s."""
    if not (n * lo <= s <= n * hi):
        raise ValueError("sum outside box")
    w = hi - lo
    if w == 0:
        return n * lo * lo
    k, rem = divmod(s - n * lo, w)
    if k >= n:
        return n * hi * hi
    return k * hi * hi + (lo + rem) ** 2 + (n - k - 1) * lo * lo


def _q_window(n: int, s: int, var_iv: Interval, g: Fraction, sample: bool) -> tuple[int, int, int, int] | None:
    """Integer Q window (respecting open ends) for one S; returns (qlo, qhi, a, b) raw bounds."""
    denom = n * (n - 1) if sample else n * n
    vlo, vhi = var_iv.lo / (g * g), var_iv.hi / (g * g)
    a = (s * s + denom * vlo) / n
    b = (s * s + denom * vhi) / n
    qlo, qhi = ceil(a), floor(b)
    if not var_iv.lo_closed and qlo == a:
        qlo += 1
    if not var_iv.hi_closed and qhi == b:
        qhi -= 1
    return qlo, qhi, a, b


def _first_q(qlo: int, qhi: int, parity: int, floor_q: int, ceil_q: int | None) -> int | None:
    lo = max(qlo, floor_q)
    hi = qhi if ceil_q is None else min(qhi, ceil_q)
    if lo > hi:
        return None
    q = lo + ((lo - parity) % 2)
    return q if q <= hi else None


def grimmer_search(
    n: int,
    mean_iv: Interval,
    sd_iv: Interval,
    g: Fraction,
    sample: bool,
    lo_bound: int | None = None,
    hi_bound: int | None = None,
) -> dict | None:
    """Return a witness {S, Q} satisfying all necessary conditions, or None."""
    if sample and n < 2:
        return None
    rng = sum_range(n, mean_iv, g, None if lo_bound is None else lo_bound * g, None if hi_bound is None else hi_bound * g)
    if rng is None:
        return None
    var_iv = Interval(max(sd_iv.lo, Fraction(0)) ** 2, sd_iv.hi**2, sd_iv.lo_closed or sd_iv.lo <= 0, sd_iv.hi_closed)
    smin, smax = rng
    if smax - smin > MAX_S_CANDIDATES:
        raise OverflowError("too many candidate sums; not checkable")
    for s in range(smin, smax + 1):
        qlo, qhi, _, _ = _q_window(n, s, var_iv, g, sample)
        floor_q = q_min(n, s)
        ceil_q = q_max(n, s, lo_bound, hi_bound) if lo_bound is not None and hi_bound is not None else None
        q = _first_q(qlo, qhi, s % 2, floor_q, ceil_q)
        if q is not None:
            return {"S": s, "Q": q, "variance": "sample" if sample else "population"}
    return None


# --------------------------------------------------------------------- GRIM


def grim_check(
    n: int,
    mean: str | Reported,
    granularity: str | Fraction = Fraction(1),
    scale_min: str | int | None = None,
    scale_max: str | int | None = None,
    label: str = "",
    provenance: list[Provenance] | None = None,
    extra_assumptions: list[str] | None = None,
) -> Finding:
    r = parse_reported(mean)
    g = _frac(granularity)
    lo_b, hi_b = _frac(scale_min), _frac(scale_max)
    inputs = {"label": label, "N": n, "mean": r.text, "granularity": str(g), "scale_min": scale_min, "scale_max": scale_max}
    assumptions = [
        f"all N={n} participants contribute to the mean",
        f"each observation is an integer multiple of {g}",
    ] + (extra_assumptions or [])
    if lo_b is not None or hi_b is not None:
        assumptions.append(f"observations lie in [{scale_min}, {scale_max}]")
    per = {}
    for c in PROOF_CONVENTIONS:
        per[c.value] = sum_range(n, convention_interval(r, c), g, lo_b, hi_b) is not None
    hull = proof_interval(r)
    rng = sum_range(n, hull, g, lo_b, hi_b)
    if rng is not None:
        s = rng[0]
        return make_finding(
            engine=ENGINE, check="grim", tier=Tier.CONSISTENT,
            title=f"GRIM consistent: {label or r.text} (N={n})",
            detail=f"S={s} gives mean {g * s}/{n} = {float(g * s / n):.6g}, which prints as {r.text}.",
            inputs=inputs, assumptions=assumptions, per_convention=per, witness={"S": s}, provenance=provenance,
        )
    # Neighbouring achievable means for the human-readable explanation.
    s0 = floor(hull.lo * n / g)
    near = [(s0 + k, g * (s0 + k) / n) for k in (0, 1)]
    near_txt = ", ".join(f"{g * s}/{n}={float(m):.6f}" for s, m in near)
    claim = (
        f"No integer total gives a mean printing as {r.text} with N={n} (step {g}); "
        f"the nearest achievable means are {near_txt}."
    )
    f = make_finding(
        engine=ENGINE, check="grim", tier=Tier.PROVEN_IMPOSSIBLE,
        title=f"GRIM impossible: mean {r.text} with N={n}" + (f" ({label})" if label else ""),
        detail=claim, inputs=inputs, assumptions=assumptions, per_convention=per, provenance=provenance,
    )
    f.certificate = grim_certificate(f.id, inputs, assumptions, claim)
    return f


def grim_certificate(cid: str, inputs: dict, assumptions: list[str], claim: str) -> str:
    body = f'''from fractions import Fraction as F
from math import ceil, floor
N, MEAN, G = {inputs["N"]}, {inputs["mean"]!r}, F({str(inputs["granularity"])!r})
LO_B, HI_B = {inputs["scale_min"]!r}, {inputs["scale_max"]!r}
{HULL_SRC}
lo, hi = hull(MEAN)
smin, smax = ceil(lo * N / G), floor(hi * N / G)  # mean = G*S/N, S integer
if LO_B is not None: smin = max(smin, ceil(F(str(LO_B)) * N / G))
if HI_B is not None: smax = min(smax, floor(F(str(HI_B)) * N / G))
for s in (floor(lo * N / G), floor(lo * N / G) + 1):
    print(f"  S={{s}}: mean={{G * s / N}} = {{float(G * s / N):.8f}}")
if smin <= smax:
    print(f"NOT VERIFIED: S={{smin}} gives mean {{float(G * smin / N):.8f}}"); raise SystemExit(1)
print(f"VERIFIED: printed mean {{MEAN}} needs S/N in [{{float(lo):.10f}}, {{float(hi):.10f}}] "
      f"but no integer S exists for N={{N}}.")
'''
    return render(cid, "GRIM", claim, inputs, assumptions, body)


# ------------------------------------------------------------------ GRIMMER


def grimmer_check(
    n: int,
    mean: str | Reported,
    sd: str | Reported,
    granularity: str | Fraction = Fraction(1),
    scale_min: int | None = None,
    scale_max: int | None = None,
    label: str = "",
    provenance: list[Provenance] | None = None,
    extra_assumptions: list[str] | None = None,
) -> Finding:
    """Mean+SD integer feasibility. Bounds (if given) are in data units and must be multiples of g."""
    if n < 2:
        raise ValueError("an SD needs N >= 2")
    rm, rs = parse_reported(mean), parse_reported(sd)
    g = _frac(granularity)
    lo_u = hi_u = None
    if scale_min is not None and scale_max is not None:
        lo_f, hi_f = _frac(scale_min) / g, _frac(scale_max) / g
        if lo_f.denominator != 1 or hi_f.denominator != 1:
            raise ValueError("scale bounds must be multiples of the granularity")
        lo_u, hi_u = int(lo_f), int(hi_f)
    inputs = {"label": label, "N": n, "mean": rm.text, "sd": rs.text, "granularity": str(g), "scale_min": scale_min, "scale_max": scale_max}
    assumptions = [
        f"all N={n} participants contribute to mean and SD",
        f"each observation is an integer multiple of {g}",
        "SD is either the sample (N-1) or population (N) SD; both are tried",
    ] + (extra_assumptions or [])
    if lo_u is not None:
        assumptions.append(f"observations lie in [{scale_min}, {scale_max}]")
    per: dict[str, bool] = {}
    for c in PROOF_CONVENTIONS:
        for sample in (True, False):
            w = grimmer_search(n, convention_interval(rm, c), convention_interval(rs, c), g, sample, lo_u, hi_u)
            per[f"{c.value}/{'n-1' if sample else 'n'}"] = w is not None
    mh, sh = proof_interval(rm), proof_interval(rs, nonnegative=True)
    if sum_range(n, mh, g, None if lo_u is None else lo_u * g, None if hi_u is None else hi_u * g) is None:
        # The mean alone is impossible: report that as the primary GRIM finding instead.
        return grim_check(n, rm, g, scale_min, scale_max, label, provenance, extra_assumptions)
    witness = None
    for sample in (True, False):
        witness = grimmer_search(n, mh, sh, g, sample, lo_u, hi_u)
        if witness:
            break
    if witness:
        return make_finding(
            engine=ENGINE, check="grimmer", tier=Tier.CONSISTENT,
            title=f"GRIMMER consistent: {label or rm.text + ' (' + rs.text + ')'}",
            detail=f"Integer totals S={witness['S']}, Q={witness['Q']} satisfy the necessary conditions ({witness['variance']} SD).",
            inputs=inputs, assumptions=assumptions, per_convention=per, witness=witness, provenance=provenance,
        )
    claim = (
        f"No integer data set of N={n} values (step {g}) has a mean printing as {rm.text} and an SD printing as "
        f"{rs.text}, under either the N-1 or N SD formula."
    )
    f = make_finding(
        engine=ENGINE, check="grimmer", tier=Tier.PROVEN_IMPOSSIBLE,
        title=f"GRIMMER impossible: {rm.text} ({rs.text}) with N={n}" + (f" ({label})" if label else ""),
        detail=claim, inputs=inputs, assumptions=assumptions, per_convention=per, provenance=provenance,
    )
    f.certificate = grimmer_certificate(f.id, inputs, assumptions, claim, lo_u, hi_u)
    return f


def grimmer_certificate(cid: str, inputs: dict, assumptions: list[str], claim: str, lo_u, hi_u) -> str:
    body = f'''from fractions import Fraction as F
from math import ceil, floor
N, MEAN, SD, G = {inputs["N"]}, {inputs["mean"]!r}, {inputs["sd"]!r}, F({str(inputs["granularity"])!r})
LO, HI = {lo_u!r}, {hi_u!r}  # bounds on y = x/G (None = unbounded)
{HULL_SRC}
def qmin(s):  # most even integer split of s minimises sum of squares
    q, r = divmod(s, N); return r * (q + 1) ** 2 + (N - r) * q * q
def qmax(s):  # extreme split maximises sum of squares inside [LO, HI]
    if LO is None: return None
    k, rem = divmod(s - N * LO, HI - LO) if HI > LO else (N, 0)
    return N * HI * HI if k >= N else k * HI * HI + (LO + rem) ** 2 + (N - k - 1) * LO * LO
mlo, mhi = hull(MEAN); slo, shi = hull(SD, nonneg=True)
smin, smax = ceil(mlo * N / G), floor(mhi * N / G)
if LO is not None: smin, smax = max(smin, N * LO), min(smax, N * HI)
for denom, name in ((N * (N - 1), "n-1"), (N * N, "n")):
    for s in range(smin, smax + 1):
        a = (s * s + denom * slo * slo / (G * G)) / N  # N*Q - S^2 = denom*var/G^2
        b = (s * s + denom * shi * shi / (G * G)) / N
        lo, hi = max(ceil(a), qmin(s)), floor(b)
        if qmax(s) is not None: hi = min(hi, qmax(s))
        q = lo + ((lo - s) % 2)  # sum of squares has the parity of the sum
        print(f"  {{name}} S={{s}}: Q in [{{lo}}, {{hi}}] with Q%2={{s % 2}} -> {{'ok' if q <= hi else 'empty'}}")
        if q <= hi:
            print(f"NOT VERIFIED: S={{s}}, Q={{q}} ({{name}})"); raise SystemExit(1)
print(f"VERIFIED: no integer (S, Q) fits mean {{MEAN}} and SD {{SD}} with N={{N}} under N-1 or N.")
'''
    return render(cid, "GRIMMER", claim, inputs, assumptions, body)
