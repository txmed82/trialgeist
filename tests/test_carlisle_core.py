"""A0: Carlisle-Stouffer engine calibration and guardrails."""

from __future__ import annotations

import random

from trialgeist.engines.carlisle import CategoricalVar, ContinuousVar, Design, carlisle_check, detect_design
from trialgeist.verdict import Tier

VARS = [("Age, yr", 62, 11), ("BMI", 28, 5), ("Systolic BP", 135, 18), ("Creatinine", 1.0, 0.3), ("Hemoglobin", 13.5, 1.6),
        ("Heart rate", 80, 12), ("Albumin", 3.9, 0.5), ("Sodium", 139, 3)]


def table(rnd, n1, n2, fabricate=None, reps=1, d=1):
    cont = []
    for name, mu, sd in [(f"{nm} {i}" if i else nm, mu, sd) for i in range(reps) for nm, mu, sd in VARS]:
        ms, ss = [], []
        for arm, n in enumerate((n1, n2)):
            xs = [rnd.gauss(mu, sd) for _ in range(n)]
            m = sum(xs) / n
            s = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
            if fabricate == "similar":
                m, s = mu + rnd.uniform(-0.002, 0.002) * sd, sd
            if fabricate == "different":
                m = mu + (0.6 * sd if arm == 0 else -0.6 * sd)
            ms.append(f"{m:.{d}f}")
            ss.append(f"{s:.{d}f}")
        cont.append(ContinuousVar(name, ms, ss, [n1, n2]))
    cat = []
    for name, p in [("Male", 0.55), ("Diabetes", 0.3)]:
        k1, k2 = sum(rnd.random() < p for _ in range(n1)), sum(rnd.random() < p for _ in range(n2))
        cat.append(CategoricalVar(name, [[k1, n1 - k1], [k2, n2 - k2]]))
    return cont, cat


def test_null_trials_are_not_extreme():
    rnd = random.Random(0)
    flagged = 0
    for _ in range(150):
        cont, cat = table(rnd, rnd.randint(40, 400), rnd.randint(40, 400), reps=rnd.choice([1, 3]), d=rnd.choice([0, 1, 2]))
        f = carlisle_check(cont, cat, Design("block"))
        assert f.tier is not Tier.PROVEN_IMPOSSIBLE
        flagged += f.tier is Tier.STATISTICALLY_EXTREME
    assert flagged == 0


def test_fabricated_tables_are_extreme():
    rnd = random.Random(1)
    sim = carlisle_check(*table(rnd, 300, 300, "similar", reps=3, d=2), Design("simple"))
    diff = carlisle_check(*table(rnd, 300, 300, "different"), Design("simple"))
    assert sim.tier is Tier.STATISTICALLY_EXTREME and "similar" in sim.title
    assert diff.tier is Tier.STATISTICALLY_EXTREME and "different" in diff.title
    assert sim.certificate and "calibrated" in sim.certificate


def test_cluster_and_strata_guardrails():
    rnd = random.Random(2)
    cont, cat = table(rnd, 300, 300, "similar")
    assert carlisle_check(cont, cat, Design("cluster")).tier is Tier.INCONCLUSIVE
    d = detect_design("Patients were randomly assigned in permuted blocks of 4, stratified by age and diabetes status.")
    assert d.randomization == "stratified" and any("age" in s for s in d.strata)
    f = carlisle_check(cont, cat, d)
    assert "Age, yr" in f.inputs["dropped_as_strata"]
