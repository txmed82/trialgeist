"""Engine: survival read at the numbers-at-risk checkpoints (KM checkpoint bound).

A digitiser cannot always rebuild every drop of a dense Kaplan-Meier curve (hundreds of
patients per arm, overlapping arms, confidence bands, censor marks).  It can still read the
curve's *level* at a few times.  The levels alone constrain the numbers at risk.

Bound.  With n(tau) the number at risk at tau, under either convention
#{T > tau} <= n(tau) <= #{T >= tau}, and S the unweighted Kaplan-Meier estimate, for any
checkpoints tau_i < tau_j:

    S(tau_j-) / S(tau_i)  >=  n(tau_j) / n(tau_i)

Proof: with event times t_1 < ... < t_k in (tau_i, tau_j), at-risk counts r_m and deaths d_m,
r_{m+1} <= r_m - d_m (people only leave the risk set), so
prod (r_m - d_m)/r_m >= (r_k - d_k)/r_1 = #{T > t_k} / #{T >= t_1} >= n(tau_j) / n(tau_i).
(No events in between: the ratio is 1 and n can only fall.)  In words: the curve cannot
fall by a larger fraction than the share of patients who left the risk set.

Readings.  Each reading (t, v) says the plotted curve at nominal time t has survival v, the
true time lying within eps_t of t and the true level within eps of v.  S is non-increasing,
so any reading whose true time is >= tau_i bounds S(tau_i) from below, and any reading whose
true time is < tau_j bounds S(tau_j-) from above:

    L_i = max(v - eps : t - eps_t >= tau_i)        U_j = min(v + eps + 1/n_j : t + eps_t < tau_j, 1)

The 1/n_j in U_j covers curves drawn by joining event points with sloping lines instead of
steps: there a point between two events is drawn below the true step by at most that event's
drop, S(t_k) * 1/n(t_k) <= 1/n_j (one event; n(t_k) >= n_j as t_k < tau_j).

A contradiction is  L_i > 0  and  U_j * n_i < n_j * L_i.  Every candidate risk row must be
contradicted for a proof (the row a curve belongs to is not always known).  A row whose
counts ever increase is malformed and never contradicted (it cannot be trusted either way).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import Any

from ..certificates import render
from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "km_checkpoint"


def F(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))


Reading = tuple[Fraction, Fraction]  # (nominal time, plotted survival)
Row = tuple[list[Fraction], list[int]]  # (checkpoint times, numbers at risk)


def violation(readings: list[Reading], eps: Fraction, eps_t: Fraction, row: Row) -> dict | None:
    """The strongest contradicted checkpoint pair of one row, or None."""
    times, counts = row
    if len(times) != len(counts) or any(b > a for a, b in zip(counts, counts[1:])) or any(n < 0 for n in counts):
        return None
    best = None
    for i in range(len(times)):
        ni = counts[i]
        if ni <= 0:
            continue
        lows = [(v - eps, (t, v)) for t, v in readings if t - eps_t >= times[i]]
        if not lows:
            continue
        li, ri = max(lows)
        if li <= 0:
            continue
        for j in range(i + 1, len(times)):
            nj = counts[j]
            if nj <= 0:
                continue
            highs = [(v + eps + Fraction(1, nj), (t, v)) for t, v in readings if t + eps_t < times[j]]
            uj, rj = min(highs) if highs else (Fraction(1), None)
            uj = min(uj, Fraction(1))
            if uj * ni < nj * li:
                margin = nj * li - uj * ni
                if best is None or margin * best["ni"] > best["margin"] * ni:
                    best = {"i": i, "j": j, "tau_i": times[i], "tau_j": times[j], "ni": ni, "nj": nj,
                            "read_i": ri, "read_j": rj, "L": li, "U": uj, "margin": margin}
    return best


def _assumptions(eps, eps_t, n_rows: int, joint: bool) -> list[str]:
    a = [
        f"plotted survival levels are within +/-{float(eps):.4g} of the true Kaplan-Meier estimate",
        f"plotted times are within +/-{float(eps_t):.4g} of the true times",
        "the curve is an unweighted, unadjusted Kaplan-Meier estimate of the same patients counted in the "
        "numbers at risk, with no delayed entry",
        "numbers at risk count the patients still under observation at each time (either convention: "
        "event-free at, or just after, the time)",
        "the curve is drawn as steps, or by joining single-event points (each upper reading is widened by one "
        "event, 1/n at risk, for the latter)",
    ]
    if joint:
        a.append(f"each of the {n_rows} printed numbers-at-risk rows belongs to exactly one of the digitised curves "
                 "(every one-to-one assignment is tried)")
    elif n_rows > 1:
        a.append(f"the curve belongs to one of the {n_rows} printed numbers-at-risk rows (all are tried)")
    return a


def _rows_inputs(rows: list[Row]) -> list[dict]:
    return [{"risk_times": [str(t) for t in ts], "at_risk": list(ns)} for ts, ns in rows]


def _pair_desc(v: dict) -> str:
    return (f"at risk {v['ni']} at {float(v['tau_i']):g} and {v['nj']} at {float(v['tau_j']):g} allow a fall to no less "
            f"than {float(v['nj']) / v['ni']:.3f} of the earlier level, but the curve reads >= {float(v['L']):.3f} "
            f"after {float(v['tau_i']):g} and <= {float(v['U']):.3f} before {float(v['tau_j']):g} "
            f"(ratio <= {float(v['U'] / v['L']):.3f})")


def km_checkpoint_check(label: str, readings: list, eps, eps_t, rows: list,
                        provenance: list[Provenance] | None = None) -> Finding:
    rd = sorted((F(t), F(v)) for t, v in readings)
    eps, eps_t = F(eps), F(eps_t)
    rws = [([F(t) for t in ts], [int(n) for n in ns]) for ts, ns in rows]
    inputs: dict[str, Any] = {"label": label, "eps": str(eps), "eps_t": str(eps_t), "rows": _rows_inputs(rws),
                              "readings": len(rd)}
    assumptions = _assumptions(eps, eps_t, len(rws), False)
    common: dict[str, Any] = dict(engine=ENGINE, check="km_checkpoint", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    vs = [violation(rd, eps, eps_t, r) for r in rws]
    if not rd or not rws or any(v is None for v in vs):
        ok = [k for k, v in enumerate(vs) if v is None]
        return make_finding(**common, tier=Tier.CONSISTENT,
                            title=f"KM curve consistent with numbers at risk at the checkpoints ({label})",
                            detail=f"{len(rd)} level readings fit risk row(s) {ok} at every pair of checkpoints.")
    desc = " | ".join(f"row {k}: {_pair_desc(v)}" for k, v in enumerate(vs))
    claim = (f"The plotted levels of '{label}' fall faster than {'any of the printed numbers-at-risk rows' if len(rws) > 1 else 'its numbers at risk'} "
             f"allow (a Kaplan-Meier curve cannot fall by a larger fraction than the share of patients who left "
             f"the risk set): {desc}.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE,
                     title=f"KM curve falls faster than its numbers at risk allow ({label})", detail=claim,
                     stats={"readings": len(rd), "alternatives": len(rws)})
    f.certificate = checkpoint_certificate(f.id, [[(0, k, v)] for k, v in enumerate(vs)], eps, eps_t, rws,
                                           assumptions, claim, label)
    return f


def km_checkpoint_joint_check(labels: list[str], curves: list[tuple[list, Any, Any]], rows: list,
                              provenance: list[Provenance] | None = None) -> Finding:
    """PROVEN when no one-to-one assignment of the printed rows to the curves fits: under every
    assignment some curve falls faster than its row allows."""
    cs = [(sorted((F(t), F(v)) for t, v in rd), F(e), F(et)) for rd, e, et in curves]
    rws = [([F(t) for t in ts], [int(n) for n in ns]) for ts, ns in rows]
    label = " / ".join(labels)
    eps = max(e for _, e, _ in cs)
    eps_t = max(et for _, _, et in cs)
    inputs: dict[str, Any] = {"curves": labels, "rows": _rows_inputs(rws),
                              "eps": [str(e) for _, e, _ in cs], "eps_t": [str(et) for _, _, et in cs]}
    assumptions = _assumptions(eps, eps_t, len(rws), True)
    common: dict[str, Any] = dict(engine=ENGINE, check="km_checkpoint_joint", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance)
    if len(cs) != len(rws) or not 2 <= len(cs) <= 4:
        return make_finding(**common, tier=Tier.INCONCLUSIVE, title=f"KM checkpoint assignment not checked ({label})",
                            detail="the number of curves and of numbers-at-risk rows differ")
    v = [[violation(rd, e, et, r) for r in rws] for rd, e, et in cs]
    witnesses = []
    for perm in permutations(range(len(rws))):
        hit = next(((c, perm[c]) for c in range(len(cs)) if v[c][perm[c]] is not None), None)
        if hit is None:
            return make_finding(**common, tier=Tier.CONSISTENT,
                                title=f"KM curves consistent with the numbers at risk under some assignment ({label})",
                                detail=f"assignment curve -> row {list(perm)} fits at every pair of checkpoints.")
        witnesses.append(hit)
    used = sorted(set(witnesses))
    desc = "; ".join(f"curve {labels[c]} with row {r}: {_pair_desc(v[c][r])}" for c, r in used)
    claim = (f"No assignment of the {len(rws)} printed numbers-at-risk rows to the curves {label} fits: under every "
             f"assignment some curve falls faster than its row allows. {desc}.")
    f = make_finding(**common, tier=Tier.PROVEN_IMPOSSIBLE,
                     title=f"KM curves fit no assignment of the numbers at risk ({label})", detail=claim,
                     stats={"curves": len(cs), "assignments": len(witnesses)})
    groups = [[(c, r, v[c][r])] for c, r in used]
    f.certificate = checkpoint_certificate(f.id, groups, None, None, rws, assumptions, claim, label,
                                           curves_eps=[(e, et) for _, e, et in cs], n_curves=len(cs))
    return f


def checkpoint_certificate(cid, groups, eps, eps_t, rows, assumptions, claim, label, curves_eps=None, n_curves=1) -> str:
    """groups: one entry per contradicted (curve, row) used; each a list with one (curve, row, violation)."""
    pairs = []
    for g in groups:
        c, r, v = g[0]
        e, et = (curves_eps[c] if curves_eps else (eps, eps_t))
        pairs.append({"curve": c, "row": r, "eps": str(e), "eps_t": str(et),
                      "tau_i": str(v["tau_i"]), "n_i": v["ni"], "tau_j": str(v["tau_j"]), "n_j": v["nj"],
                      "read_i": [str(v["read_i"][0]), str(v["read_i"][1])],
                      "read_j": None if v["read_j"] is None else [str(v["read_j"][0]), str(v["read_j"][1])]})
    row_data = [{"times": [str(t) for t in ts], "at_risk": ns} for ts, ns in rows]
    small_inputs = {"label": label, "rows": len(rows), "curves": n_curves}
    body = f'''from fractions import Fraction as F
from itertools import permutations
ROWS = {row_data!r}
PAIRS = {pairs!r}  # contradicted (curve, row) pairs: checkpoints, counts and the two plotted readings used
JOINT = {curves_eps is not None!r}  # True: rows are assigned one-to-one to {n_curves} curves
bad = set()
for p in PAIRS:
    row, eps, et = ROWS[p["row"]], F(p["eps"]), F(p["eps_t"])
    ti, tj, ni, nj = F(p["tau_i"]), F(p["tau_j"]), p["n_i"], p["n_j"]
    assert row["at_risk"][row["times"].index(p["tau_i"])] == ni and row["at_risk"][row["times"].index(p["tau_j"])] == nj
    assert all(b <= a for a, b in zip(row["at_risk"], row["at_risk"][1:])), "row must not increase"
    t, v = map(F, p["read_i"])
    assert t - et >= ti  # this reading's true time is >= tau_i, so S(tau_i) >= its level
    L = v - eps
    if p["read_j"] is None:
        U = F(1)  # S <= 1
    else:
        t, v = map(F, p["read_j"])
        assert t + et < tj  # true time < tau_j, so S(tau_j-) <= its level
        U = min(v + eps + F(1, nj), F(1))  # + one event: covers curves drawn by joining event points
    # KM: S(tau_j-)/S(tau_i) >= n_j/n_i (the curve cannot fall faster than people leave the risk set)
    ok = L > 0 and U * ni < nj * L
    print(f"  curve {{p['curve']}} row {{p['row']}}: S({{ti}}) >= {{float(L):.4f}}, S({{tj}}-) <= {{float(U):.4f}}, "
          f"ratio <= {{float(U / L):.4f}} < {{nj}}/{{ni}} = {{nj / ni:.4f}}: {{ok}}")
    if ok:
        bad.add((p["curve"], p["row"]))
if JOINT:
    k = {n_curves}
    ok = all(any((c, perm[c]) in bad for c in range(k)) for perm in permutations(range(len(ROWS))))
else:
    ok = all((0, r) in bad for r in range(len(ROWS)))
if not ok:
    print("NOT VERIFIED"); raise SystemExit(1)
print("VERIFIED: the plotted levels fall faster than " + ("every assignment of" if JOINT else "every candidate") +
      " numbers-at-risk row allows (exact rational arithmetic).")
'''
    return render(cid, "KM checkpoint", claim, small_inputs, assumptions, body)
