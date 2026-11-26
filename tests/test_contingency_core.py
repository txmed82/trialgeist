"""A0: 2x2 contingency engine against scipy's independent implementations."""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
from scipy.stats import chi2_contingency, fisher_exact

from trialgeist.engines.contingency_exact import contingency_check, fisher_pvalues, parse_pvalue
from trialgeist.verdict import Tier


def fmt_p(p: float, rnd: random.Random) -> str:
    if p < 0.001:
        return "<0.001"
    style = rnd.choice(["2", "3", "sig2"])
    if style == "2" and p >= 0.01:
        return f"{p:.2f}"
    if style == "sig2":
        return f"{p:.2g}" if "e" not in f"{p:.2g}" else f"{p:.4f}"
    return f"{p:.3f}"


def test_fisher_matches_scipy():
    rnd = random.Random(1)
    for _ in range(200):
        n1, n2 = rnd.randint(1, 60), rnd.randint(1, 60)
        a, c = rnd.randint(0, n1), rnd.randint(0, n2)
        ours = fisher_pvalues(a, n1, c, n2)["fisher_minlike_tol"]
        theirs = fisher_exact([[a, n1 - a], [c, n2 - c]]).pvalue
        assert abs(float(ours) - theirs) < 1e-9


def test_parse_pvalue():
    assert parse_pvalue("<0.001").hi > Fraction(1, 1000) > parse_pvalue("<0.001").lo
    iv = parse_pvalue("P = 0·04")
    assert iv.lo < Fraction(4, 100) < iv.hi


@pytest.mark.parametrize("seed", range(4))
def test_contingency_never_flags_real_tables(seed):
    rnd = random.Random(seed)
    for _ in range(120):
        n1, n2 = rnd.randint(5, 300), rnd.randint(5, 300)
        a, c = rnd.randint(1, n1 - 1), rnd.randint(1, n2 - 1)
        table = [[a, n1 - a], [c, n2 - c]]
        test = rnd.choice(["fisher", "chi2", "chi2_yates"])
        if test == "fisher":
            p = fisher_exact(table).pvalue
        else:
            p = chi2_contingency(table, correction=(test == "chi2_yates")).pvalue
        rr = (a / n1) / (c / n2)
        orr = (a * (n2 - c)) / ((n1 - a) * c)
        d = rnd.choice([1, 2])
        f = contingency_check(
            n1, n2, pct1=f"{100 * a / n1:.{d - 1}f}", pct2=f"{100 * c / n2:.{d - 1}f}",
            rr=f"{rr:.2f}", odds_ratio=f"{orr:.2f}", estimate_is_crude=True, p_value=fmt_p(p, rnd),
        )
        assert f.tier is not Tier.PROVEN_IMPOSSIBLE, (table, test, f.inputs)


def test_contingency_impossible_rr_and_certificate(tmp_path):
    from trialgeist.certificates import line_count, run_certificate

    # 10/50 vs 20/50: crude RR 0.50, cannot print as 0.62 under any orientation
    f = contingency_check(50, 50, events1=10, events2=20, rr="0.62", estimate_is_crude=True)
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    assert line_count(f.certificate) < 80
    ok, dt, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok and dt < 2.0, out
    # Adjusted estimates are never used as evidence
    assert contingency_check(50, 50, events1=10, events2=20, rr="0.62").tier is Tier.INCONCLUSIVE


def test_contingency_impossible_pvalue(tmp_path):
    from trialgeist.certificates import run_certificate

    # 30/100 vs 30/100 is identical: no test gives p=0.01
    f = contingency_check(100, 100, events1=30, events2=30, p_value="0.01")
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    ok, _, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok, out
