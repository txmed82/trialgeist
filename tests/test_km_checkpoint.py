"""KM checkpoint bound: sound on true Kaplan-Meier data under both at-risk conventions and
adversarial reading noise; catches swapped / inflated numbers-at-risk rows; certificates run."""

from __future__ import annotations

import random
from fractions import Fraction as Fr

from hypothesis import given, settings
from hypothesis import strategies as st

from trialgeist.certificates import line_count, run_certificate
from trialgeist.engines.km_checkpoint import km_checkpoint_check, km_checkpoint_joint_check, violation
from trialgeist.verdict import Tier


def km_data(rnd: random.Random, n: int, lam: float, cens: float, horizon: int = 36, ties: bool = False):
    obs = []
    for _ in range(n):
        t, c = rnd.expovariate(lam), rnd.uniform(0, horizon / cens) if cens else horizon * 10
        if ties:  # discrete follow-up: many events and censorings at the same times as checkpoints
            t, c = float(round(t)), float(round(c))
        obs.append((min(t, c, horizon), t <= min(c, horizon)))
    return obs


def S_at(obs, t, left=False):
    """Exact KM S(t) (or S(t-) when left)."""
    s = Fr(1)
    for u in sorted({u for u, e in obs if e}):
        if u > t or (left and u == t):
            break
        n = sum(1 for v, _ in obs if v >= u)
        d = sum(1 for v, e in obs if v == u and e)
        s *= Fr(n - d, n)
    return s


def risk(obs, taus, strict):
    return [sum(1 for u, _ in obs if (u > tau if strict else u >= tau)) for tau in taus]


def readings(rnd, obs, eps, eps_t, horizon=36, k=60):
    """Adversarial readings: plotted at nominal t, true time anywhere within eps_t, level off by up to eps."""
    out = []
    for _ in range(k):
        true_t = Fr(rnd.randint(0, horizon * 20), 20)
        nominal = true_t + rnd.choice([-eps_t, eps_t, Fr(0), eps_t / 2])
        v = S_at(obs, true_t) + rnd.choice([-eps, eps, Fr(0)])
        out.append((nominal, v))
    return out


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 10**9), n=st.integers(5, 300), cens=st.sampled_from([0, 0.5, 1, 3]),
       strict=st.booleans(), ties=st.booleans())
def test_true_km_data_is_never_proven_impossible(seed, n, cens, strict, ties):
    rnd = random.Random(seed)
    obs = km_data(rnd, n, rnd.uniform(0.01, 0.2), cens, ties=ties)
    taus = [Fr(t) for t in range(0, 37, 6)]
    eps, eps_t = Fr(rnd.choice([0, 1, 2, 5]), 1000), Fr(rnd.choice([0, 1, 3]), 10)  # eps 0: the bound itself is tight
    row = (taus, risk(obs, taus, strict))
    assert violation(readings(rnd, obs, eps, eps_t), eps, eps_t, row) is None
    # the exact left/right limits right at the checkpoints are the tightest legal readings
    rd = [(tau + eps_t, S_at(obs, tau)) for tau in taus] + [(tau - eps_t - Fr(1, 10**6), S_at(obs, tau, left=True)) for tau in taus]
    assert violation(rd, eps, eps_t, row) is None


def _arms(seed=1):
    rnd = random.Random(seed)
    good = km_data(rnd, 400, 0.006, 1)   # S(36) ~ 0.8
    poor = km_data(rnd, 400, 0.05, 1)    # S(36) ~ 0.2
    taus = [Fr(t) for t in range(0, 37, 6)]
    rd = lambda o: [(Fr(t, 4), S_at(o, Fr(t, 4))) for t in range(0, 145)]  # noqa: E731
    return taus, good, poor, rd(good), rd(poor)


def test_rows_swapped_between_arms_are_caught_by_the_joint_check():
    taus, good, poor, rg, rp = _arms()
    eps, eps_t = Fr(1, 100), Fr(1, 2)
    rows = [(taus, risk(good, taus, False)), (taus, risk(poor, taus, False))]
    # per-curve with both candidate rows: the poor curve fits its own row, so no proof
    f = km_checkpoint_check("poor", rp, eps, eps_t, rows)
    assert f.tier is Tier.CONSISTENT
    # but with the rows printed under the wrong arms the curves fit no assignment ...
    f = km_checkpoint_check("poor", rp, eps, eps_t, rows[:1])
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    ok = km_checkpoint_joint_check(["good", "poor"], [(rg, eps, eps_t), (rp, eps, eps_t)], rows)
    assert ok.tier is Tier.CONSISTENT
    # ... and when the poor arm's row is inflated, no one-to-one assignment fits
    inflated = (taus, [min(c, r) for c, r in zip(risk(poor, taus, False), [400, 390, 380, 370, 360, 350, 340])])
    inflated = (taus, [max(a, b) for a, b in zip(inflated[1], risk(good, taus, False))])
    bad = km_checkpoint_joint_check(["good", "poor"], [(rg, eps, eps_t), (rp, eps, eps_t)], [rows[0], inflated])
    assert bad.tier is Tier.PROVEN_IMPOSSIBLE


def test_certificates_verify_and_are_short(tmp_path):
    taus, good, poor, rg, rp = _arms()
    eps, eps_t = Fr(1, 100), Fr(1, 2)
    f = km_checkpoint_check("poor", rp, eps, eps_t, [(taus, risk(good, taus, False))])
    ok, dt, out = run_certificate(f.certificate, tmp_path, "single")
    assert ok, out
    assert line_count(f.certificate) < 80 and dt < 2
    inflated = (taus, [max(a, b) for a, b in zip(risk(good, taus, False), [400, 390, 380, 370, 360, 350, 340])])
    j = km_checkpoint_joint_check(["good", "poor"], [(rg, eps, eps_t), (rp, eps, eps_t)], [(taus, risk(good, taus, False)), inflated])
    assert j.tier is Tier.PROVEN_IMPOSSIBLE
    ok, dt, out = run_certificate(j.certificate, tmp_path, "joint")
    assert ok, out
    assert line_count(j.certificate) < 80


def test_certificate_refuses_when_the_numbers_are_changed(tmp_path):
    taus, good, poor, rg, rp = _arms()
    f = km_checkpoint_check("poor", rp, Fr(1, 100), Fr(1, 2), [(taus, risk(good, taus, False))])
    assert f.certificate is not None
    # tamper: claim the tolerance was ten times wider, the contradiction must no longer reproduce
    src = f.certificate.replace("'eps': '1/100'", "'eps': '1/10'")
    assert src != f.certificate
    ok, _, out = run_certificate(src, tmp_path, "tampered")
    assert not ok and "NOT VERIFIED" in out


def test_increasing_rows_and_empty_inputs_are_never_proofs():
    taus = [Fr(0), Fr(6), Fr(12)]
    rd = [(Fr(1), Fr(1, 2)), (Fr(11), Fr(1, 10))]
    assert violation(rd, Fr(1, 100), Fr(0), (taus, [100, 120, 90])) is None  # malformed row
    assert km_checkpoint_check("x", [], "0.01", "0", [(taus, [100, 99, 98])]).tier is Tier.CONSISTENT
    assert km_checkpoint_joint_check(["a"], [(rd, "0.01", "0")], [(taus, [100, 99, 98]), (taus, [1, 1, 1])]).tier is Tier.INCONCLUSIVE
    assert km_checkpoint_joint_check(["a", "b"], [(rd, "0.01", "0")] * 2, [(taus, [100, 99, 98])]).tier is Tier.INCONCLUSIVE


def test_joint_check_with_an_unread_arm(tmp_path):
    """Three printed rows, two curves read (the third arm refused): each read curve takes a
    distinct row.  True data never proves; two curves that fit no distinct pair of rows do."""
    rnd = random.Random(7)
    arms = [km_data(rnd, 300, lam, 0.01) for lam in (0.01, 0.03, 0.08)]
    taus = [Fr(t) for t in (0, 6, 12, 18, 24, 30, 36)]
    rows = [(taus, risk(a, taus, False)) for a in arms]
    eps, eps_t = Fr(1, 200), Fr(1, 10)
    rd = [readings(rnd, a, eps, eps_t) for a in arms]
    for pick in ((0, 1), (1, 2), (0, 2), (2, 0)):
        f = km_checkpoint_joint_check([str(i) for i in pick], [(rd[i], eps, eps_t) for i in pick], rows)
        assert f.tier is Tier.CONSISTENT, f.detail
    # both read curves fall far faster than any printed row allows: only one could have the
    # steepest row, so the other contradicts whatever it is given
    steep = [(Fr(t), Fr(1, 1000) if t > 1 else Fr(1)) for t in range(0, 37)]
    f = km_checkpoint_joint_check(["x", "y"], [(steep, eps, eps_t), (steep, eps, eps_t)], rows)
    assert f.tier is Tier.PROVEN_IMPOSSIBLE, f.detail
    ok, _, out = run_certificate(f.certificate, tmp_path, "unread")
    assert ok, out
