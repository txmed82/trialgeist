"""Typed extraction schema ("claims") and the router that sends each claim to
the exact engine that can test it.

Claims are what an extractor (deterministic PDF parser, JATS XML parser, a
human, or an untrusted LLM proposal that has been re-located in the source)
asserts the paper printed.  Numbers stay strings so printed precision survives.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .engines.contingency_exact import contingency_check
from .engines.counts import count_sum_check, percent_check, percent_partition_check
from .engines.grim_grimmer import bhatia_davis_check, grim_check, grimmer_check
from .engines.km_vector_sat import KMStep, km_check_any_row
from .engines.sprite_sat import sprite_check
from .verdict import Tier, Finding, Provenance


class _Claim(BaseModel):
    label: str = ""
    provenance: list[Provenance] = Field(default_factory=list)


class MeanClaim(_Claim):
    kind: Literal["mean"] = "mean"
    n: int
    mean: str
    sd: str | None = None
    granularity: str = "1"
    scale_min: str | None = None
    scale_max: str | None = None
    observed_min: str | None = None
    observed_max: str | None = None
    n_min: int | None = None  # lower bound on contributors when missing data is possible
    n_exact: bool = True  # False: only checks valid for every N' in [n_min, n] are used


class PercentClaim(_Claim):
    kind: Literal["percent"] = "percent"
    n: int | None = None  # group size (upper bound on the denominator)
    percent: str
    count: int | None = None
    denominator_exact: bool = False


class CountSumClaim(_Claim):
    kind: Literal["count_sum"] = "count_sum"
    total: int
    counts: list[int]
    exhaustive: bool = True


class PercentPartitionClaim(_Claim):
    kind: Literal["percent_partition"] = "percent_partition"
    n: int
    percents: list[str]
    n_exact: bool = True  # False: shares of one unknown denominator D <= n


class TwoByTwoClaim(_Claim):
    kind: Literal["two_by_two"] = "two_by_two"
    n1: int
    n2: int
    events1: int | None = None
    events2: int | None = None
    pct1: str | None = None
    pct2: str | None = None
    rr: str | None = None
    odds_ratio: str | None = None
    rd_percent: str | None = None
    estimate_is_crude: bool = False
    p_value: str | None = None
    tests: list[str] | None = None


class KMStepIn(BaseModel):
    t: str
    before: str
    after: str
    dt: str = "0"  # extra time tolerance for this drop only


class KMRiskRow(BaseModel):
    label: str = ""
    risk_times: list[str]
    at_risk: list[int]


class KMArmClaim(_Claim):
    kind: Literal["km_arm"] = "km_arm"
    risk_times: list[str] = Field(default_factory=list)
    at_risk: list[int] = Field(default_factory=list)
    alternatives: list[KMRiskRow] = Field(default_factory=list)  # candidate risk rows when pairing is unknown
    steps: list[KMStepIn]
    eps: str = "0.002"
    eps_t: str = "0"
    total_events: int | None = None
    ocr_numbers: bool = False  # numbers at risk read by OCR from an image: never more than a lead


class KMCheckpointClaim(_Claim):
    """Survival levels read off a curve (not its individual drops) against numbers at risk."""

    kind: Literal["km_checkpoint"] = "km_checkpoint"
    readings: list[tuple[str, str]]  # (time, plotted survival)
    eps: str
    eps_t: str
    alternatives: list[KMRiskRow]
    ocr_numbers: bool = False


class KMCheckpointCurve(BaseModel):
    label: str
    readings: list[tuple[str, str]]
    eps: str
    eps_t: str


class KMCheckpointJointClaim(_Claim):
    """Curves of one panel against its rows, each row belonging to exactly one curve."""

    kind: Literal["km_checkpoint_joint"] = "km_checkpoint_joint"
    curves: list[KMCheckpointCurve]
    rows: list[KMRiskRow]
    ocr_numbers: bool = False


class BaselineVar(BaseModel):
    name: str
    means: list[str] | None = None
    sds: list[str] | None = None
    ns: list[int] | None = None
    counts: list[list[int]] | None = None  # categorical: per arm, per category


class BaselineTableClaim(_Claim):
    kind: Literal["baseline_table"] = "baseline_table"
    variables: list[BaselineVar]
    randomization: str = "unknown"
    strata: list[str] = Field(default_factory=list)


class MeanPartitionClaim(_Claim):
    kind: Literal["mean_partition"] = "mean_partition"
    overall_mean: str
    parts: list[tuple[int, str]]  # (n, printed mean) of disjoint subgroups making up the whole
    weights_exact: bool = True  # False: per-row missing data, pooled mean only within the subgroup means


class MedianRangeClaim(_Claim):
    kind: Literal["median_range"] = "median_range"
    median: str
    min: str
    max: str
    granularity: str | None = None
    bounds: Literal["range", "iqr"] = "range"


class ArmTotalClaim(_Claim):
    kind: Literal["arm_total"] = "arm_total"
    total: int
    counts: list[int]
    total_n: int
    arm_ns: list[int]


class CommonDenominatorClaim(_Claim):
    kind: Literal["common_denominator"] = "common_denominator"
    counts: list[int]
    percents: list[str]
    d_max: int | None = None


Claim = Annotated[
    Union[MeanClaim, PercentClaim, CountSumClaim, PercentPartitionClaim, TwoByTwoClaim, KMArmClaim, BaselineTableClaim,
          MeanPartitionClaim, MedianRangeClaim, ArmTotalClaim, CommonDenominatorClaim, KMCheckpointClaim,
          KMCheckpointJointClaim],
    Field(discriminator="kind"),
]


class ClaimSet(BaseModel):
    source: str = ""
    claims: list[Claim]


def _with_missing(c: MeanClaim) -> list[Finding]:
    """When N is only an upper bound, a proof must hold for every N' in [n_min, n]."""
    lo = c.n_min or 2
    out: list[Finding] = []
    if c.sd is not None and c.scale_min is not None and c.scale_max is not None:
        out.append(bhatia_davis_check(lo, c.mean, c.sd, c.scale_min, c.scale_max, label=c.label, provenance=c.provenance))
    return out


def _ocr_lead(f: Finding) -> Finding:
    """Proven arithmetic on OCR-read numbers at risk is a lead: OCR may have misread a digit."""
    if f.tier is Tier.PROVEN_IMPOSSIBLE:
        f.tier, f.certificate = Tier.INCONCLUSIVE, None
        f.title = "Lead (OCR-read numbers at risk): " + f.title
        f.detail = ("The curve and the numbers at risk as read by OCR contradict each other; check every "
                    "at-risk number against the figure before relying on this. " + f.detail)
        f.assumptions.append("numbers at risk were read by OCR from a raster image and are not verified")
    return f


def check_claim(c) -> list[Finding]:
    if isinstance(c, MeanClaim):
        if not c.n_exact:
            return _with_missing(c)
        if c.sd is None:
            return [grim_check(c.n, c.mean, Fraction(c.granularity), c.scale_min, c.scale_max, c.label, c.provenance)]
        out = []
        if c.scale_min is not None and c.scale_max is not None:
            out.append(bhatia_davis_check(c.n, c.mean, c.sd, c.scale_min, c.scale_max, label=c.label, provenance=c.provenance))
            out.append(sprite_check(c.n, c.mean, c.sd, c.scale_min, c.scale_max, Fraction(c.granularity),
                                    c.observed_min, c.observed_max, c.label, c.provenance))
        else:
            out.append(grimmer_check(c.n, c.mean, c.sd, Fraction(c.granularity), label=c.label, provenance=c.provenance))
        return out
    if isinstance(c, PercentClaim):
        if c.n is None:
            from .engines.counts import any_denominator_check

            return [any_denominator_check(c.count, c.percent, c.label, c.provenance)] if c.count is not None else []
        return [percent_check(c.n, c.percent, c.count, c.label, c.denominator_exact, c.provenance)]
    if isinstance(c, CountSumClaim):
        return [count_sum_check(c.total, c.counts, c.label, c.provenance, c.exhaustive)]
    if isinstance(c, PercentPartitionClaim):
        if not c.n_exact:
            from .engines.counts import percent_partition_any_check

            return [percent_partition_any_check(c.n, c.percents, c.label, c.provenance)]
        return [percent_partition_check(c.n, c.percents, c.label, c.provenance)]
    if isinstance(c, TwoByTwoClaim):
        return [contingency_check(c.n1, c.n2, c.events1, c.events2, c.pct1, c.pct2, c.rr, c.odds_ratio, c.rd_percent,
                                  c.estimate_is_crude, c.p_value, tuple(c.tests) if c.tests else None, c.label, c.provenance)]
    if isinstance(c, KMArmClaim):
        steps = [KMStep(Fraction(s.t), Fraction(s.before), Fraction(s.after), Fraction(s.dt)) for s in c.steps]
        rows = [(r.risk_times, r.at_risk) for r in c.alternatives] or [(c.risk_times, c.at_risk)]
        rows3 = [([Fraction(t) for t in ts], list(ns), c.total_events) for ts, ns in rows]
        f = km_check_any_row(c.label, steps, Fraction(c.eps), Fraction(c.eps_t), rows3, c.provenance)
        return [_ocr_lead(f) if c.ocr_numbers else f]
    if isinstance(c, (KMCheckpointClaim, KMCheckpointJointClaim)):
        from .engines.km_checkpoint import km_checkpoint_check, km_checkpoint_joint_check

        if isinstance(c, KMCheckpointClaim):
            f = km_checkpoint_check(c.label, c.readings, c.eps, c.eps_t,
                                    [(r.risk_times, r.at_risk) for r in c.alternatives], c.provenance)
        else:
            f = km_checkpoint_joint_check([k.label for k in c.curves], [(k.readings, k.eps, k.eps_t) for k in c.curves],
                                          [(r.risk_times, r.at_risk) for r in c.rows], c.provenance)
        return [_ocr_lead(f) if c.ocr_numbers else f]
    if isinstance(c, MeanPartitionClaim):
        from .engines.consistency import mean_partition_check

        return [mean_partition_check(c.overall_mean, list(c.parts), c.label, c.provenance, c.weights_exact)]
    if isinstance(c, ArmTotalClaim):
        from .engines.consistency import arm_total_check

        return [arm_total_check(c.total, c.counts, c.total_n, c.arm_ns, c.label, c.provenance)]
    if isinstance(c, CommonDenominatorClaim):
        from .engines.consistency import common_denominator_check

        return [common_denominator_check(c.counts, c.percents, c.d_max, c.label, c.provenance)]
    if isinstance(c, MedianRangeClaim):
        from .engines.consistency import median_range_check

        return [median_range_check(c.median, c.min, c.max, c.label, c.granularity, c.provenance, c.bounds)]
    if isinstance(c, BaselineTableClaim):
        from .engines.carlisle import CategoricalVar, ContinuousVar, Design, carlisle_check

        cont = [ContinuousVar(v.name, v.means, v.sds, v.ns) for v in c.variables if v.means and v.sds and v.ns]
        cat = [CategoricalVar(v.name, v.counts) for v in c.variables if v.counts]
        return [carlisle_check(cont, cat, Design(c.randomization, c.strata), c.label, c.provenance)]
    raise TypeError(type(c))


def check_all(cs: ClaimSet) -> list[Finding]:
    out: list[Finding] = []
    for c in cs.claims:
        out.extend(check_claim(c))
    return out
