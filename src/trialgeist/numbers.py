"""Exact parsing of printed decimal numbers and rounding-interval arithmetic.

A printed value such as ``"4.23"`` does not name a number, it names the set of
true values that some rounding convention would print as ``"4.23"``.  Every
engine reasons about those sets with exact :class:`fractions.Fraction` bounds.

Soundness policy (see PLAN.md, decision 1): proofs use the closed hull of the
union of all supported conventions, widened by a tiny relative slack so that a
binary floating-point pipeline sitting exactly on a rounding boundary can never
manufacture a false contradiction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

__all__ = [
    "Reported",
    "Interval",
    "Convention",
    "PROOF_CONVENTIONS",
    "FLOAT_SLACK",
    "parse_reported",
    "convention_interval",
    "proof_interval",
    "ceil_frac",
    "floor_frac",
]

#: Relative widening applied to every proof interval (covers IEEE-754 double
#: error in the producer's software by ~7 orders of magnitude).
FLOAT_SLACK = Fraction(1, 10**9)

_MINUS_CHARS = "-\u2212\u2012\u2013\ufe63\uff0d"
_DECIMAL_CHARS = ".\u00b7\u2027\u22c5"  # Lancet prints decimals with a middle dot
_NUM_RE = re.compile(
    r"^\s*(?P<sign>[" + re.escape(_MINUS_CHARS) + r"+])?\s*"
    r"(?P<int>\d+)?(?:[" + re.escape(_DECIMAL_CHARS) + r"](?P<frac>\d+))?\s*%?\s*$"
)


def floor_frac(x: Fraction) -> int:
    return math.floor(x)


def ceil_frac(x: Fraction) -> int:
    return math.ceil(x)


@dataclass(frozen=True)
class Reported:
    """A number exactly as printed: value plus printed precision."""

    text: str
    value: Fraction
    decimals: int
    negative_zero: bool = False

    @property
    def unit(self) -> Fraction:
        return Fraction(1, 10**self.decimals)

    @property
    def half_unit(self) -> Fraction:
        return Fraction(1, 2 * 10**self.decimals)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.text


def parse_reported(text: str | int | float | Fraction | Reported, decimal_comma: bool = False) -> Reported:
    """Parse a printed number keeping its printed decimal places.

    ``float`` input is refused on purpose: ``4.20`` would silently lose its
    trailing zero.  Pass the string as printed.
    """
    if isinstance(text, Reported):
        return text
    if isinstance(text, bool) or isinstance(text, float):
        raise TypeError("pass printed numbers as strings so trailing zeros survive")
    if isinstance(text, int):
        return Reported(str(text), Fraction(text), 0)
    if isinstance(text, Fraction):
        if text.denominator != 1:
            raise TypeError("Fraction input must be an integer; pass the printed string")
        return Reported(str(text), text, 0)
    raw = str(text)
    s = raw.replace("\u2009", "").replace("\u00a0", "").replace(" ", "")
    if decimal_comma:
        s = s.replace(",", ".")
    else:
        # thousands separators like 1,234 or 1 234
        if re.fullmatch(r"[^\d]*\d{1,3}(,\d{3})+([.\u00b7]\d+)?%?", s):
            s = s.replace(",", "")
    m = _NUM_RE.match(s)
    if not m or (m.group("int") is None and m.group("frac") is None):
        raise ValueError(f"not a printed decimal number: {raw!r}")
    ip = m.group("int") or "0"
    fp = m.group("frac") or ""
    neg = m.group("sign") is not None and m.group("sign") in _MINUS_CHARS
    mag = Fraction(int(ip + fp), 10 ** len(fp))
    value = -mag if neg else mag
    return Reported(raw.strip(), value, len(fp), negative_zero=neg and mag == 0)


class Convention(str, Enum):
    HALF_UP = "half_up"  # ties away from zero (SAS, SPSS, Stata display, Excel ROUND)
    HALF_UP_POSINF = "half_up_toward_pos_inf"  # floor(x + 0.5) style
    HALF_EVEN = "half_even"  # banker's rounding (R, Python, IEEE)
    HALF_DOWN = "half_down"  # ties toward zero
    TRUNCATE = "truncate"  # chop digits toward zero


#: Conventions that must ALL be infeasible before anything is called impossible.
PROOF_CONVENTIONS: tuple[Convention, ...] = tuple(Convention)


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction
    lo_closed: bool = True
    hi_closed: bool = True

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("empty interval")

    def contains(self, x: Fraction) -> bool:
        lo_ok = x > self.lo or (self.lo_closed and x == self.lo)
        hi_ok = x < self.hi or (self.hi_closed and x == self.hi)
        return lo_ok and hi_ok

    def scale(self, k: Fraction) -> "Interval":
        if k <= 0:
            raise ValueError("scale factor must be positive")
        return Interval(self.lo * k, self.hi * k, self.lo_closed, self.hi_closed)

    def as_tuple(self) -> tuple[str, str]:
        return (str(self.lo), str(self.hi))


def convention_interval(r: Reported, conv: Convention) -> Interval:
    """Exact set of true values printed as ``r`` under ``conv``."""
    v, h = r.value, r.half_unit
    neg = v < 0 or r.negative_zero
    zero = v == 0
    if conv is Convention.HALF_UP_POSINF:
        return Interval(v - h, v + h, True, False)
    if conv is Convention.HALF_UP:
        if zero:
            return Interval(-h, h, False, False)
        return Interval(v - h, v + h, False, True) if neg else Interval(v - h, v + h, True, False)
    if conv is Convention.HALF_DOWN:
        if zero:
            return Interval(-h, h, True, True)
        return Interval(v - h, v + h, True, False) if neg else Interval(v - h, v + h, False, True)
    if conv is Convention.HALF_EVEN:
        last_digit_even = (abs(v) / r.unit) % 2 == 0
        return Interval(v - h, v + h, last_digit_even, last_digit_even)
    if conv is Convention.TRUNCATE:
        if zero:
            if r.negative_zero:
                return Interval(-2 * h, Fraction(0), False, True)
            return Interval(-2 * h, 2 * h, False, False)
        return Interval(v - 2 * h, v, False, True) if neg else Interval(v, v + 2 * h, True, False)
    raise ValueError(conv)


def proof_interval(
    r: Reported | str,
    conventions: tuple[Convention, ...] = PROOF_CONVENTIONS,
    slack: Fraction = FLOAT_SLACK,
    nonnegative: bool = False,
) -> Interval:
    """Closed hull of all conventions, widened by relative float slack.

    The union of the supported conventions is always a single interval, so the
    hull adds nothing spurious; closing it and widening it only makes proofs
    harder, never easier.
    """
    r = parse_reported(r)
    ivs = [convention_interval(r, c) for c in conventions]
    lo = min(i.lo for i in ivs)
    hi = max(i.hi for i in ivs)
    pad = slack * (max(abs(lo), abs(hi)) + r.unit)
    lo, hi = lo - pad, hi + pad
    if nonnegative and lo < 0:
        lo = Fraction(0)
    return Interval(lo, hi, True, True)
