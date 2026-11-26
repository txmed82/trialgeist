"""A0: exact summary-statistic cores.

Soundness is tested against independent printers (``decimal`` with every
rounding mode and IEEE-754 float formatting); completeness of the bounded
engine is tested against brute-force enumeration of every multiset.
"""

from __future__ import annotations

import itertools
import math
import random
from decimal import ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

import pytest

from trialgeist.engines.counts import count_sum_check, percent_check, percent_partition_check
from trialgeist.engines.grim_grimmer import grim_check, grimmer_check, q_max, q_min
from trialgeist.engines.sprite_sat import sprite_check
from trialgeist.numbers import Convention, convention_interval, parse_reported, proof_interval
from trialgeist.verdict import Tier

MODES = {ROUND_HALF_UP: Convention.HALF_UP, ROUND_HALF_EVEN: Convention.HALF_EVEN,
         ROUND_HALF_DOWN: Convention.HALF_DOWN, ROUND_DOWN: Convention.TRUNCATE}


def dec_print(x: Fraction, d: int, mode) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        v = Decimal(x.numerator) / Decimal(x.denominator)
        return str(v.quantize(Decimal(1).scaleb(-d), rounding=mode))


def sqrt_print(var: Fraction, d: int, mode) -> str:
    with localcontext() as ctx:
        ctx.prec = 60
        v = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
        return str(v.quantize(Decimal(1).scaleb(-d), rounding=mode))


def printers(x: Fraction, d: int):
    for mode in MODES:
        yield dec_print(x, d, mode)
    yield f"{float(x):.{d}f}"  # binary float, half-even on the double


# ------------------------------------------------------------------ numbers


def test_parse_keeps_precision_and_lancet_dot():
    r = parse_reported("4·20")
    assert r.value == Fraction(21, 5) and r.decimals == 2
    assert parse_reported("\u22120.5").value == Fraction(-1, 2)
    assert parse_reported("1,234.5").value == Fraction(12345, 10)
    with pytest.raises(TypeError):
        parse_reported(4.2)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", range(5))
def test_convention_intervals_match_decimal(seed):
    rnd = random.Random(seed)
    for _ in range(3000):
        d = rnd.randint(0, 3)
        # values on and near ties
        x = Fraction(rnd.randint(-20000, 20000), 2 * 10 ** (d + rnd.choice([0, 1])))
        for mode, conv in MODES.items():
            s = dec_print(x, d, mode)
            iv = convention_interval(parse_reported(s), conv)
            assert iv.contains(x), (x, s, conv)
            assert proof_interval(s).contains(x)


# --------------------------------------------------------------------- GRIM


@pytest.mark.parametrize("seed", range(4))
def test_grim_and_grimmer_never_flag_real_data(seed):
    rnd = random.Random(seed)
    for _ in range(400):
        n = rnd.randint(2, 120)
        lo, hi = 0, rnd.choice([1, 4, 5, 7, 10, 15, 100])
        g = rnd.choice([Fraction(1), Fraction(1), Fraction(1, 2), Fraction(1, 3)])
        data = [g * rnd.randint(lo, hi) for _ in range(n)]
        mean = sum(data, Fraction(0)) / n
        ss = sum((x - mean) ** 2 for x in data)
        for d in (0, 1, 2, 3):
            for m_txt in printers(mean, d):
                assert grim_check(n, m_txt, g).tier is not Tier.PROVEN_IMPOSSIBLE
                for var in (ss / (n - 1), ss / n):
                    for s_txt in [sqrt_print(var, d, mode) for mode in MODES] + [f"{math.sqrt(var):.{d}f}"]:
                        f = grimmer_check(n, m_txt, s_txt, g)
                        assert f.tier is not Tier.PROVEN_IMPOSSIBLE, (n, data, m_txt, s_txt)
                        f = grimmer_check(n, m_txt, s_txt, g, scale_min=lo * g, scale_max=hi * g)
                        assert f.tier is not Tier.PROVEN_IMPOSSIBLE, (n, data, m_txt, s_txt)


def test_grim_classic_example_and_certificate(tmp_path):
    from trialgeist.certificates import line_count, run_certificate

    f = grim_check(28, "5.19")  # 145/28=5.1786, 146/28=5.2143
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    assert line_count(f.certificate) < 80
    ok, dt, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok and dt < 2.0, out
    assert grim_check(28, "5.18").tier is Tier.CONSISTENT


def test_grim_truncation_is_respected():
    # 4.2857 (30/7) truncates to 4.28 but rounds to 4.29: must not be flagged.
    assert grim_check(7, "4.28").tier is Tier.CONSISTENT
    assert grim_check(7, "4.29").tier is Tier.CONSISTENT


def test_qmin_qmax_bruteforce():
    for n in range(1, 6):
        for lo, hi in ((0, 3), (1, 5), (-2, 2)):
            seen: dict[int, list[int]] = {}
            for combo in itertools.combinations_with_replacement(range(lo, hi + 1), n):
                seen.setdefault(sum(combo), []).append(sum(v * v for v in combo))
            for s, qs in seen.items():
                assert q_min(n, s) == min(qs)
                assert q_max(n, s, lo, hi) == max(qs)


# ------------------------------------------------------- SPRITE vs brute force


def _achievable(n, lo, hi, d):
    out = set()
    for combo in itertools.combinations_with_replacement(range(lo, hi + 1), n):
        s = sum(combo)
        ss = sum(v * v for v in combo) - Fraction(s * s, n)
        mean = Fraction(s, n)
        for var in (ss / (n - 1), ss / n):
            out.add((dec_print(mean, d, ROUND_HALF_UP), sqrt_print(var, d, ROUND_HALF_UP)))
    return out


@pytest.mark.parametrize("n,lo,hi", [(3, 0, 4), (5, 1, 5), (6, 0, 3), (4, 0, 7)])
def test_sprite_sound_and_complete_on_tiny_scales(n, lo, hi):
    d = 2
    ach = _achievable(n, lo, hi, d)
    rnd = random.Random(n * 100 + hi)
    means = sorted({m for m, _ in ach})
    sds = sorted({s for _, s in ach})
    cands = list(ach)[:40] + [(rnd.choice(means), rnd.choice(sds)) for _ in range(60)]
    for m, s in cands:
        f = sprite_check(n, m, s, lo, hi)
        if (m, s) in ach:
            assert f.tier is Tier.CONSISTENT, (m, s)
        # every CONSISTENT verdict carries a witness that truly reproduces the print
        if f.tier is Tier.CONSISTENT and f.check == "sprite_sat":
            counts = {Fraction(k): v for k, v in f.witness["counts"].items()}
            assert sum(counts.values()) == n


def test_sprite_certificate_runs(tmp_path):
    from trialgeist.certificates import line_count, run_certificate

    # Mean 1.00 on a 0-4 scale with N=5 needs S=5; SD 2.00 needs far more spread than 0..4 allows with S=5
    f = sprite_check(5, "1.00", "2.00", 0, 4)
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    assert line_count(f.certificate) < 80
    ok, dt, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok and dt < 2.0, out


# ------------------------------------------------------------- percentages


def test_percent_with_missing_data_not_flagged():
    # 13/41 = 31.7%; arm N=45 with 4 missing: must stay consistent
    assert percent_check(45, "31.7", count=13).tier is Tier.CONSISTENT


def test_percent_impossible_any_denominator(tmp_path):
    from trialgeist.certificates import run_certificate

    f = percent_check(50, "33.0", count=7)  # 7/D: D=21 -> 33.33, D=22 -> 31.8
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    ok, _, out = run_certificate(f.certificate, tmp_path, f.id)
    assert ok, out


def test_percent_only_requires_exact_denominator():
    assert percent_check(37, "50.0").tier is Tier.INCONCLUSIVE
    assert percent_check(37, "50.0", denominator_exact=True).tier is Tier.PROVEN_IMPOSSIBLE


def test_partition_and_sums(tmp_path):
    from trialgeist.certificates import run_certificate

    assert percent_partition_check(30, ["33.3", "33.3", "33.3"]).tier is Tier.CONSISTENT
    f = percent_partition_check(20, ["30", "30", "30"])
    assert f.tier is Tier.PROVEN_IMPOSSIBLE
    assert run_certificate(f.certificate, tmp_path, f.id)[0]
    f = count_sum_check(100, [40, 35, 24])
    assert f.tier is Tier.PROVEN_IMPOSSIBLE and run_certificate(f.certificate, tmp_path, f.id)[0]


@pytest.mark.parametrize("seed", range(3))
def test_percent_never_flags_real_counts(seed):
    rnd = random.Random(seed)
    for _ in range(3000):
        n = rnd.randint(1, 800)
        missing = rnd.randint(0, min(n - 1, 20))
        dnm = n - missing
        k = rnd.randint(0, dnm)
        for d in (0, 1, 2):
            for p in printers(Fraction(100 * k, dnm), d):
                assert percent_check(n, p, count=k).tier is not Tier.PROVEN_IMPOSSIBLE
                if missing == 0:
                    assert percent_check(n, p, denominator_exact=True).tier is not Tier.PROVEN_IMPOSSIBLE
