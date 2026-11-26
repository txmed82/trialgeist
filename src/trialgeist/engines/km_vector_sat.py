"""Engine 2: exact Kaplan-Meier step / numbers-at-risk feasibility (KM-SAT).

Input per arm: the printed numbers-at-risk table (checkpoint times and integer
counts) and the visible vertical drops of the curve (time, plotted level just
before and just after the drop), with a survival tolerance ``eps`` and a time
tolerance ``eps_t``.

Sound relaxation (PLAN.md decision 4).  A visible drop k is a *cluster* of one
or more event times with possible censoring in between.  With n_k the number
at risk just before the cluster, D_k >= 1 its events and a_k the number at risk
right after its last event, the cluster's survival factor F_k obeys

    a_k / (a_k + D_k)  <=  F_k  <=  (n_k - D_k) / n_k

(telescoping products; equality on the right when there is one event time
and no censoring).  The plotted levels give F_k in [r_lo, r_hi] with
r_lo = (after-eps)/(before+eps), r_hi = (after+eps)/(before-eps), rounded
outward to 1e-6.  Between clusters the at-risk count can only fall (censoring,
or events hidden under the other arm's curve).  Within the checkpoint interval
[tau_m, tau_m+1) the chain starts at most at n(tau_m) and ends at least at
n(tau_m+1).  Checkpoints with a drop within eps_t of them are merged away, so
"at risk at t" vs "at risk just after t" conventions cannot create a proof.

Drops no larger than 2*eps may be plotting jitter: they are optional (D may be 0).
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor

from ortools.sat.python import cp_model

from ..certificates import render
from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "km_sat"
QUANT = 10**6


def F(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))


@dataclass
class KMStep:
    t: Fraction
    before: Fraction
    after: Fraction


@dataclass
class KMArm:
    label: str
    risk_times: list[Fraction]
    at_risk: list[int]
    steps: list[KMStep]
    eps: Fraction = Fraction(1, 500)
    eps_t: Fraction = Fraction(0)
    total_events: int | None = None
    provenance: list[Provenance] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.risk_times = [F(t) for t in self.risk_times]
        self.steps = [KMStep(F(s.t), F(s.before), F(s.after)) for s in self.steps]
        self.eps, self.eps_t = F(self.eps), F(self.eps_t)
        if len(self.risk_times) != len(self.at_risk):
            raise ValueError("risk table times/counts length mismatch")


def ratio_bounds(before: Fraction, after: Fraction, eps: Fraction) -> tuple[Fraction, Fraction, bool]:
    """Outward-rounded [r_lo, r_hi] for the cluster factor, and whether the step is optional."""
    lo = max(Fraction(0), after - eps) / (before + eps)
    hi = Fraction(1) if before - eps <= 0 else min(Fraction(1), (after + eps) / (before - eps))
    lo_q = Fraction(floor(lo * QUANT), QUANT)
    hi_q = Fraction(min(QUANT, ceil(hi * QUANT)), QUANT)
    optional = before - after <= 2 * eps
    return lo_q, hi_q, optional


@dataclass
class Segment:
    """One checkpoint interval after merging ambiguous checkpoints."""

    t0: Fraction
    t1: Fraction | None
    n_start: int
    n_end: int  # lower bound on at-risk at the end (0 for the tail)
    steps: list[tuple[int, Fraction, Fraction, bool]]  # (global step index, r_lo, r_hi, optional)


def segments(arm: KMArm) -> list[Segment]:
    times, counts = arm.risk_times, arm.at_risk
    keep = [0]
    for m in range(1, len(times)):
        if any(abs(s.t - times[m]) <= arm.eps_t for s in arm.steps):
            continue  # drop at (or near) a checkpoint: at-risk convention ambiguous, merge
        keep.append(m)
    segs: list[Segment] = []
    for j, m in enumerate(keep):
        t0 = times[m]
        nxt = keep[j + 1] if j + 1 < len(keep) else None
        t1 = times[nxt] if nxt is not None else None
        n_end = counts[nxt] if nxt is not None else 0
        st = []
        for i, s in enumerate(arm.steps):
            in_seg = s.t >= t0 - (arm.eps_t if j == 0 else 0) and (t1 is None or s.t < t1)
            if j == 0 and s.t < t0:
                in_seg = True
            if in_seg:
                lo, hi, opt = ratio_bounds(s.before, s.after, arm.eps)
                st.append((i, lo, hi, opt))
        segs.append(Segment(t0, t1, counts[m], n_end, st))
    return segs


def _build(model: cp_model.CpModel, seg: Segment, tag: str):
    """Add one segment's chain; returns list of D variables."""
    N = max(seg.n_start, 0)
    prev = None
    Ds = []
    for k, (_, lo, hi, opt) in enumerate(seg.steps):
        n = model.NewIntVar(0, N, f"{tag}n{k}")
        D = model.NewIntVar(0, N, f"{tag}D{k}")
        a = model.NewIntVar(0, N, f"{tag}a{k}")
        model.Add(n <= (seg.n_start if prev is None else prev))
        model.Add(a <= n - D)
        pl, ph = int(lo * QUANT), int(hi * QUANT)
        if opt:
            u = model.NewBoolVar(f"{tag}u{k}")
            model.Add(D >= 1).OnlyEnforceIf(u)
            model.Add(D == 0).OnlyEnforceIf(u.Not())
            model.Add(QUANT * (n - D) >= pl * n).OnlyEnforceIf(u)
            model.Add((QUANT - ph) * a <= ph * D).OnlyEnforceIf(u)
        else:
            model.Add(D >= 1)
            model.Add(QUANT * (n - D) >= pl * n)
            model.Add((QUANT - ph) * a <= ph * D)
        Ds.append(D)
        prev = a
    end = seg.n_start if prev is None else prev
    model.Add(end >= seg.n_end) if not isinstance(end, int) else None
    return Ds, end


def _feasible(segs: list[Segment], total_events: int | None) -> bool | None:
    model = cp_model.CpModel()
    allD = []
    for i, seg in enumerate(segs):
        if not seg.steps and seg.n_start < seg.n_end:
            return False
        Ds, _ = _build(model, seg, f"s{i}_")
        allD += Ds
    if total_events is not None and allD:
        model.Add(sum(allD) <= total_events)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_workers = 1
    st = solver.Solve(model)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return True
    if st == cp_model.INFEASIBLE:
        return False
    return None


def km_check(arm: KMArm) -> Finding:
    segs = segments(arm)
    inputs = {
        "label": arm.label,
        "risk_times": [str(t) for t in arm.risk_times],
        "at_risk": list(arm.at_risk),
        "steps": [[str(s.t), str(s.before), str(s.after)] for s in arm.steps],
        "eps": str(arm.eps),
        "eps_t": str(arm.eps_t),
        "total_events": arm.total_events,
    }
    assumptions = [
        f"plotted survival levels are within +/-{float(arm.eps):.4g} of the true KM estimate",
        "numbers at risk are counted from the same population as the curve, with no delayed entry",
        "each visible drop contains at least one event (drops <= 2*eps may be jitter)",
    ]
    if arm.total_events is not None:
        assumptions.append(f"the arm has at most {arm.total_events} events in total")
    common: dict[str, Any] = dict(engine=ENGINE, check="km_sat", inputs=inputs, assumptions=assumptions, provenance=arm.provenance)
    whole = _feasible(segs, arm.total_events)
    if whole is None:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title=f"KM-SAT timeout ({arm.label})", detail="solver UNKNOWN")
    if whole:
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"KM curve consistent with risk table ({arm.label})",
                            detail=f"{len(arm.steps)} drops across {len(segs)} intervals admit integer events and censoring.")
    # Find the smallest self-contained contradiction for the certificate.
    bad: list[Segment] = []
    total = None
    for seg in segs:
        if _feasible([seg], None) is False:
            bad = [seg]
            break
    if not bad:
        bad, total = segs, arm.total_events
    seg_desc = "; ".join(
        f"[{float(s.t0):g}, {'end' if s.t1 is None else f'{float(s.t1):g}'}) at risk {s.n_start}->{s.n_end}, {len(s.steps)} drops"
        for s in bad
    )
    claim = (f"No integer sequence of events and censorings reproduces the plotted drops of '{arm.label}' "
             f"together with its numbers at risk: {seg_desc}"
             + (f" and at most {total} events" if total is not None else "") + ".")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"KM curve impossible given numbers at risk ({arm.label})",
                     detail=claim, stats={"segments": len(segs), "drops": len(arm.steps)})
    f.certificate = km_certificate(f.id, arm, bad, total, assumptions, claim)
    return f


def km_certificate(cid: str, arm: KMArm, segs: list[Segment], total: int | None, assumptions, claim) -> str:
    data = [
        {"n_start": s.n_start, "n_end": s.n_end,
         "steps": [[str(arm.steps[i].before), str(arm.steps[i].after)] for i, _, _, _ in s.steps]}
        for s in segs
    ]
    small_inputs = {"label": arm.label, "eps": str(arm.eps), "segments": data, "total_events": total}
    body = f'''from fractions import Fraction as F
from math import ceil, floor
from z3 import Int, Solver, And, Or, Sum, unsat  # pip install z3-solver
EPS, Q, TOTAL = F({str(arm.eps)!r}), {QUANT}, {total!r}
SEGS = {data!r}  # per interval: at-risk at start, at-risk lower bound at end, drops (before, after)
sv, allD = Solver(), []
for j, seg in enumerate(SEGS):
    N, prev = seg["n_start"], seg["n_start"]
    for k, (b, a_) in enumerate(seg["steps"]):
        b, a_ = F(b), F(a_)
        lo = floor(max(F(0), a_ - EPS) / (b + EPS) * Q)  # factor >= lo/Q, rounded outward
        hi = Q if b - EPS <= 0 else min(Q, ceil((a_ + EPS) / (b - EPS) * Q))
        n, D, a = Int(f"n{{j}}_{{k}}"), Int(f"D{{j}}_{{k}}"), Int(f"a{{j}}_{{k}}")
        sv.add(n >= 0, a >= 0, D >= 0, n <= prev, a <= n - D)  # censoring only removes people
        ratio = And(D >= 1, Q * (n - D) >= lo * n, (Q - hi) * a <= hi * D)  # (n-D)/n >= F >= a/(a+D)
        sv.add(Or(ratio, D == 0) if b - a_ <= 2 * EPS else ratio)  # tiny drops may be jitter
        allD.append(D); prev = a
    sv.add(prev >= seg["n_end"])
    print(f"  interval {{j}}: at risk {{N}} -> {{seg['n_end']}}, {{len(seg['steps'])}} drops")
if TOTAL is not None and allD: sv.add(Sum(allD) <= TOTAL)
if sv.check() != unsat:
    print("NOT VERIFIED:", sv.check()); raise SystemExit(1)
print("VERIFIED: no integer events/censoring reproduce these drops and numbers at risk (Z3 UNSAT).")
'''
    return render(cid, "KM-SAT", claim, small_inputs, assumptions, body)
