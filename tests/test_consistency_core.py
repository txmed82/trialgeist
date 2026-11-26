"""A0 soundness for the cross-table consistency engines: real data never yields a proof."""

from __future__ import annotations

import random
import statistics
from decimal import ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from fractions import Fraction as F

import pytest

from trialgeist.engines.consistency import _PRINTS_SRC, _prints, mean_partition_check, median_range_check
from trialgeist.engines.counts import percent_partition_any_check
from trialgeist.numbers import parse_reported
from trialgeist.verdict import Tier

MODES = (ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_DOWN)


def fmt(x: F, d: int, mode) -> str:
    return str((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal(1).scaleb(-d), rounding=mode))


def test_median_range_never_flags_real_samples():
    rnd = random.Random(1)
    for _ in range(1500):
        xs = [rnd.randint(0, 40) for _ in range(rnd.randint(1, 15))]
        med = F(statistics.median_low(xs) + statistics.median_high(xs), 2)
        d = rnd.choice([0, 0, 1])
        f = median_range_check(fmt(med, d, rnd.choice(MODES)), str(min(xs)), str(max(xs)), granularity="1")
        assert f.tier is Tier.CONSISTENT, (xs, f.detail)


def test_certificate_prints_matches_engine():
    ns: dict = {"F": F}
    exec(_PRINTS_SRC, ns)
    for num in range(-60, 60):
        for r in ["13", "13.5", "-2", "0", "2.3", "-0.5", "0.0", "7.25"]:
            assert ns["prints"](F(num, 4), r) == _prints(F(num, 4), parse_reported(r)), (num, r)


@pytest.mark.parametrize("seed", range(3))
def test_mean_partition_never_flags_real_data(seed):
    rnd = random.Random(seed)
    for _ in range(400):
        groups = [[F(rnd.randint(0, 4000), 100) for _ in range(rnd.randint(2, 40))] for _ in range(rnd.randint(2, 3))]
        d, mode = rnd.choice([1, 2]), rnd.choice(MODES)
        allv = [x for g in groups for x in g]
        overall = fmt(sum(allv) / len(allv), d, mode)
        parts = [(len(g), fmt(sum(g) / len(g), d, rnd.choice(MODES))) for g in groups]
        assert mean_partition_check(overall, parts).tier is Tier.CONSISTENT


def test_partition_any_denominator_never_flags_real_counts():
    rnd = random.Random(7)
    for _ in range(600):
        n = rnd.randint(5, 120)
        d_true = rnd.randint(max(1, n - 10), n)  # some missing data
        cuts = sorted(rnd.randint(0, d_true) for _ in range(rnd.randint(1, 4)))
        counts = [b - a for a, b in zip([0] + cuts, cuts + [d_true])]
        dec, mode = rnd.choice([0, 1]), rnd.choice(MODES)
        pcts = [fmt(F(100 * c, d_true), dec, mode) for c in counts]
        assert percent_partition_any_check(n, pcts).tier is Tier.CONSISTENT, (n, d_true, counts, pcts)


def test_partition_any_detects_documented_case():
    f = percent_partition_any_check(70, ["45.7", "34.8", "10.1", "9.4"])
    assert f.tier is Tier.PROVEN_IMPOSSIBLE and f.certificate
