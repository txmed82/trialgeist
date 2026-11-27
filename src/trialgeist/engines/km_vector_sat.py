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
    dt: Fraction = Fraction(0)  # extra time tolerance for this drop alone (seen only after an occlusion)


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
        self.steps = [KMStep(F(s.t), F(s.before), F(s.after), F(s.dt)) for s in self.steps]
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
        if any(abs(s.t - times[m]) <= arm.eps_t + s.dt for s in arm.steps):
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
            in_seg = s.t >= t0 - (arm.eps_t + s.dt if j == 0 else 0) and (t1 is None or s.t < t1)
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


def _analyse(arm: KMArm) -> tuple[bool | None, list[Segment], int | None]:
    """(feasible?, minimal failing segments, total-events bound used)."""
    segs = segments(arm)
    whole = _feasible(segs, arm.total_events)
    if whole is not False:
        return whole, [], None
    for seg in segs:
        if _feasible([seg], None) is False:
            return False, [seg], None
    return False, segs, arm.total_events


def _inputs(arm: KMArm) -> dict:
    return {
        "label": arm.label,
        "risk_times": [str(t) for t in arm.risk_times],
        "at_risk": list(arm.at_risk),
        "steps": [[str(s.t), str(s.before), str(s.after)] + ([str(s.dt)] if s.dt else []) for s in arm.steps],
        "eps": str(arm.eps),
        "eps_t": str(arm.eps_t),
        "total_events": arm.total_events,
    }


def _assumptions(arm: KMArm) -> list[str]:
    a = [
        f"plotted survival levels are within +/-{float(arm.eps):.4g} of the true KM estimate",
        f"plotted drop times are within +/-{float(arm.eps_t):.4g} of the true event times",
        "numbers at risk are counted from the same population as the curve, with no delayed entry",
        "the curve is an unweighted, unadjusted Kaplan-Meier estimate",
        "each visible drop contains at least one event (drops <= 2*eps may be jitter)",
    ]
    wide = [s for s in arm.steps if s.dt]
    if wide:
        a.append(f"{len(wide)} drop(s) seen only after an occlusion may lie up to a further "
                 f"{float(max(s.dt for s in wide)):.4g} away (per drop, as listed in the inputs)")
    if arm.total_events is not None:
        a.append(f"the arm has at most {arm.total_events} events in total")
    return a


def _seg_desc(segs: list[Segment]) -> str:
    return "; ".join(
        f"[{float(s.t0):g}, {'end' if s.t1 is None else f'{float(s.t1):g}'}) at risk {s.n_start}->{s.n_end}, {len(s.steps)} drops"
        for s in segs
    )


def km_check(arm: KMArm) -> Finding:
    return km_check_any_row(arm.label, arm.steps, arm.eps, arm.eps_t,
                            [(arm.risk_times, arm.at_risk, arm.total_events)], arm.provenance)


def km_check_any_row(label: str, steps: list[KMStep], eps, eps_t, rows: list[tuple[list, list[int], int | None]],
                     provenance: list[Provenance] | None = None) -> Finding:
    """PROVEN only if the curve is incompatible with EVERY candidate risk row.

    PDF extraction cannot always tell which numbers-at-risk row belongs to
    which curve; trying every row keeps a mismatch from becoming a proof.
    """
    arms = [KMArm(label, list(t), list(c), steps, eps, eps_t, tot, provenance or []) for t, c, tot in rows]
    inputs = {"label": label, "alternatives": [_inputs(a) for a in arms]} if len(arms) > 1 else _inputs(arms[0])
    assumptions = _assumptions(arms[0])
    if len(arms) > 1:
        assumptions.append(f"the curve belongs to one of the {len(arms)} printed numbers-at-risk rows (all are tried)")
    common: dict[str, Any] = dict(engine=ENGINE, check="km_sat", inputs=inputs, assumptions=assumptions, provenance=provenance)
    results = [_analyse(a) for a in arms]
    if any(r[0] is True for r in results):
        ok = [i for i, r in enumerate(results) if r[0] is True]
        return make_finding(**common, tier=Tier.CONSISTENT, title=f"KM curve consistent with numbers at risk ({label})",
                            detail=f"{len(steps)} drops admit integer events/censoring for risk row(s) {ok}.")
    if any(r[0] is None for r in results):
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title=f"KM-SAT timeout ({label})", detail="solver UNKNOWN")
    alts = [(a, r[1], r[2]) for a, r in zip(arms, results)]
    desc = " | ".join(_seg_desc(segs) + (f", at most {tot} events" if tot is not None else "") for _, segs, tot in alts)
    claim = (f"No integer sequence of events and censorings reproduces the plotted drops of '{label}' together with "
             f"{'any of the printed numbers-at-risk rows' if len(arms) > 1 else 'its numbers at risk'}: {desc}.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE, title=f"KM curve impossible given numbers at risk ({label})",
                     detail=claim, stats={"drops": len(steps), "alternatives": len(arms)})
    f.certificate = km_certificate(f.id, alts, assumptions, claim, label)
    return f


def km_certificate(cid: str, alts, assumptions, claim, label) -> str:
    data = []
    for arm, segs, total in alts:
        data.append({"eps": str(arm.eps), "total": total, "segs": [
            {"n_start": s.n_start, "n_end": s.n_end,
             "steps": [[str(arm.steps[i].before), str(arm.steps[i].after)] for i, _, _, _ in s.steps]} for s in segs]})
    small_inputs = {"label": label, "alternatives": len(data)}
    body = f'''from fractions import Fraction as F
from math import ceil, floor
from z3 import Int, Solver, And, Or, Sum, unsat  # pip install z3-solver
Q = {QUANT}
ALTS = {data!r}  # per candidate risk row: tolerance, event cap, intervals (at risk start/end, drops)
for ai, alt in enumerate(ALTS):
    EPS, sv, allD = F(alt["eps"]), Solver(), []
    for j, seg in enumerate(alt["segs"]):
        prev = seg["n_start"]
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
        print(f"  row {{ai}} interval {{j}}: at risk {{seg['n_start']}} -> {{seg['n_end']}}, {{len(seg['steps'])}} drops")
    if alt["total"] is not None and allD: sv.add(Sum(allD) <= alt["total"])
    if sv.check() != unsat:
        print(f"NOT VERIFIED: risk row {{ai}} is satisfiable"); raise SystemExit(1)
print("VERIFIED: no integer events/censoring reproduce these drops with any candidate risk row (Z3 UNSAT).")
'''
    return render(cid, "KM-SAT", claim, small_inputs, assumptions, body)
