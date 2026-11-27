"""A0: KM-SAT soundness on simulated trials and detection of corrupted curves."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from trialgeist.engines.km_vector_sat import KMArm, KMStep, km_check
from trialgeist.verdict import Tier


def simulate(rnd: random.Random, n: int, horizon: int = 36, tick: Fraction = Fraction(1, 10), at_risk_gt: bool = False):
    """Return (times, status) with times on a grid (creates ties)."""
    lam = rnd.uniform(0.01, 0.08)
    obs = []
    for _ in range(n):
        t = rnd.expovariate(lam)
        c = rnd.uniform(0, horizon * 1.2)
        tt = min(t, c, horizon)
        tt = Fraction(round(tt / float(tick))) * tick
        obs.append((tt, t <= min(c, horizon)))
    return obs


def km_steps(obs):
    times = sorted({t for t, e in obs if e})
    s = Fraction(1)
    steps = []
    for t in times:
        n = sum(1 for u, _ in obs if u >= t)
        d = sum(1 for u, e in obs if e and u == t)
        new = s * (n - d) / n
        steps.append((t, s, new))
        s = new
    return steps


def to_arm(rnd, obs, checkpoints, eps=Fraction(1, 500), merge_dt=Fraction(0), gt=False, jitter=True, label="arm"):
    raw = km_steps(obs)
    # visual merging of near-coincident drops
    vis = []
    for t, b, a in raw:
        if vis and t - vis[-1][0] <= merge_dt:
            vis[-1] = (vis[-1][0], vis[-1][1], a)
        else:
            vis.append((t, b, a))

    def noisy(x):
        if not jitter:
            return x
        y = x + Fraction(rnd.uniform(-float(eps) / 2, float(eps) / 2)).limit_denominator(10**8)
        return Fraction(f"{float(y):.5f}")

    steps = [KMStep(t, noisy(b), noisy(a)) for t, b, a in vis]
    at_risk = [sum(1 for u, _ in obs if (u > tau if gt and tau > 0 else u >= tau)) for tau in checkpoints]
    total = sum(1 for _, e in obs if e)
    # eps_t is the horizontal resolution: it must cover the time span of any visually merged drop
    eps_t = max(Fraction(1, 20), merge_dt)
    return KMArm(label, list(checkpoints), at_risk, steps, eps=eps, eps_t=eps_t, total_events=total), raw


@pytest.mark.parametrize("seed", range(6))
def test_km_never_flags_clean_curves(seed):
    rnd = random.Random(seed)
    for _ in range(12):
        n = rnd.randint(15, 400)
        obs = simulate(rnd, n)
        cps = [Fraction(x) for x in range(0, 37, rnd.choice([3, 6, 12]))]
        arm, _ = to_arm(rnd, obs, cps, merge_dt=rnd.choice([Fraction(0), Fraction(3, 10), Fraction(1)]), gt=rnd.random() < 0.3)
        f = km_check(arm)
        assert f.tier is Tier.CONSISTENT, (seed, n, f.detail)


def _first_interval_with(arm, pred):
    for m in range(len(arm.risk_times) - 1):
        lo, hi = arm.risk_times[m], arm.risk_times[m + 1]
        idx = [i for i, s in enumerate(arm.steps) if lo + Fraction(1, 10) < s.t < hi - Fraction(1, 10)]
        if pred(m, idx):
            return m, idx
    return None


def test_km_detects_too_small_drop_and_certificate(tmp_path):
    from trialgeist.certificates import line_count, run_certificate

    rnd = random.Random(42)
    obs = simulate(rnd, 120)
    cps = [Fraction(x) for x in range(0, 37, 6)]
    arm, _ = to_arm(rnd, obs, cps, jitter=False)
    hit = _first_interval_with(arm, lambda m, idx: idx and arm.steps[idx[0]].before / arm.at_risk[m] > Fraction(1, 50))
    assert hit
    m, idx = hit
    s = arm.steps[idx[0]]
    # a drop of one third of the minimum possible KM drop, still clearly above jitter (2*eps)
    s.after = s.before - max(s.before / arm.at_risk[m] / 3, 3 * arm.eps)
    assert s.before - s.after < s.before / arm.at_risk[m]
    f = km_check(arm)
    assert f.tier is Tier.PROVEN_IMPOSSIBLE, f.detail
    assert line_count(f.certificate) < 80
    ok, dt, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok and dt < 2.0, out


def test_km_detects_risk_table_edit():
    rnd = random.Random(7)
    obs = simulate(rnd, 200)
    cps = [Fraction(x) for x in range(0, 37, 6)]
    arm, _ = to_arm(rnd, obs, cps, jitter=False)
    big = lambda i: arm.steps[i].before - arm.steps[i].after > 2 * arm.eps  # not jitter-optional
    clear = lambda tau: all(abs(st.t - tau) > arm.eps_t for st in arm.steps)  # boundary not merged away
    hit = _first_interval_with(arm, lambda m, idx: sum(map(big, idx)) >= 2 and clear(arm.risk_times[m]) and clear(arm.risk_times[m + 1]))
    m, idx = hit
    idx = [i for i in idx if big(i)]
    # claim almost nobody left the risk set although several drops (>=1 event each) occurred
    arm.at_risk[m + 1] = arm.at_risk[m] - len(idx) + 1
    assert km_check(arm).tier is Tier.PROVEN_IMPOSSIBLE


def test_km_increasing_risk_table_is_impossible():
    arm = KMArm("x", [0, 6, 12], [100, 80, 85], [KMStep(Fraction(2), Fraction(1), Fraction(99, 100))])
    assert km_check(arm).tier is Tier.PROVEN_IMPOSSIBLE



def test_per_drop_time_tolerance_merges_the_checkpoint_it_straddles():
    """A drop seen only after an occlusion may lie anywhere in it: when that window covers a
    risk-table time, the two intervals must be merged, never assigned to one side."""
    from fractions import Fraction as F

    from trialgeist.engines.km_vector_sat import KMArm, KMStep, segments

    steps = [KMStep(F(2), F(1), F(9, 10)), KMStep(F(5), F(9, 10), F(8, 10)), KMStep(F(7), F(8, 10), F(7, 10))]
    tight = KMArm("a", [0, 6, 12], [10, 8, 5], steps, eps=F(1, 100), eps_t=F(1, 10))
    assert len(segments(tight)) == 3
    late = [steps[0], KMStep(F(5), F(9, 10), F(8, 10), dt=F(3, 2)), steps[2]]
    wide = KMArm("a", [0, 6, 12], [10, 8, 5], late, eps=F(1, 100), eps_t=F(1, 10))
    segs = segments(wide)
    assert len(segs) == 2 and segs[0].t1 == 12  # the checkpoint at 6 is merged away


def test_drops_before_a_late_first_checkpoint_are_not_held_to_its_count():
    """A numbers-at-risk row that starts after time 0 (a table from month 3, or a row whose first
    column was lost): drops before its first time had more patients at risk than that count."""
    from trialgeist.engines.km_vector_sat import KMStep, km_check_any_row

    # 24 patients: deaths at 1, 2, 3, 4 (1/24, 1/23, 1/22, 1/21), then 5 at risk at 50, 1 at 100
    s, steps, n = Fraction(1), [], 24
    for t in (1, 2, 3, 4):
        a = s * (n - 1) / n
        steps.append(KMStep(Fraction(t), s, a))
        s, n = a, n - 1
    f = km_check_any_row("late row", steps, Fraction(1, 500), Fraction(1, 2), [([50, 100], [5, 1], None)])
    assert f.tier is not Tier.PROVEN_IMPOSSIBLE, f.detail
