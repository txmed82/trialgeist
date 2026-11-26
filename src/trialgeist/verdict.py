"""Typed findings, the three-tier verdict taxonomy and provenance records."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, Field


class Tier(str, Enum):
    PROVEN_IMPOSSIBLE = "PROVEN_IMPOSSIBLE"
    STATISTICALLY_EXTREME = "STATISTICALLY_EXTREME"
    REGISTRY_DISCORDANT = "REGISTRY_DISCORDANT"
    CONSISTENT = "CONSISTENT"
    INCONCLUSIVE = "INCONCLUSIVE"


class Provenance(BaseModel):
    """Where an input number came from, so a human can check it in seconds."""

    source: str = "manual"  # pdf | xml | ctgov | manual | synthetic
    page: int | None = None  # 0-based page index
    bbox: tuple[float, float, float, float] | None = None  # PDF points, top-left origin
    text: str | None = None  # verbatim source text
    note: str | None = None


class Finding(BaseModel):
    id: str
    engine: str
    check: str
    tier: Tier
    title: str
    detail: str
    inputs: dict[str, Any]
    assumptions: list[str] = Field(default_factory=list)
    per_convention: dict[str, bool] = Field(default_factory=dict)  # convention -> feasible?
    witness: dict[str, Any] | None = None  # a feasible assignment when CONSISTENT
    provenance: list[Provenance] = Field(default_factory=list)
    certificate: str | None = None  # standalone python source (PROVEN_IMPOSSIBLE only)
    stats: dict[str, Any] = Field(default_factory=dict)

    @property
    def proven(self) -> bool:
        return self.tier is Tier.PROVEN_IMPOSSIBLE


def _jsonable(x: Any) -> Any:
    if isinstance(x, Fraction):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Enum):
        return x.value
    return x


def finding_id(engine: str, check: str, inputs: dict[str, Any]) -> str:
    blob = json.dumps({"e": engine, "c": check, "i": _jsonable(inputs)}, sort_keys=True, default=str)
    return f"{check}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def make_finding(
    *,
    engine: str,
    check: str,
    tier: Tier,
    title: str,
    detail: str,
    inputs: dict[str, Any],
    assumptions: list[str] | None = None,
    per_convention: dict[str, bool] | None = None,
    witness: dict[str, Any] | None = None,
    provenance: list[Provenance] | None = None,
    certificate: str | None = None,
    stats: dict[str, Any] | None = None,
) -> Finding:
    inputs = _jsonable(inputs)
    return Finding(
        id=finding_id(engine, check, inputs),
        engine=engine,
        check=check,
        tier=tier,
        title=title,
        detail=detail,
        inputs=inputs,
        assumptions=assumptions or [],
        per_convention=per_convention or {},
        witness=_jsonable(witness) if witness is not None else None,
        provenance=provenance or [],
        certificate=certificate,
        stats=_jsonable(stats or {}),
    )
