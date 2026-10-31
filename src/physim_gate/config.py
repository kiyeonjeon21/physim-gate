"""Gate configuration (``gate.yaml``): which metrics to gate and how strictly.

Per-metric settings override ``defaults``. See ``docs/comparison-semantics.md`` for how each
setting affects the verdict.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from physim_gate.schema import (
    Direction,
    InputError,
    SchemaError,
    format_validation_error,
    load_yaml,
)

CONFIG_VERSION = 1


class Statistic(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"


class BootstrapMethod(StrEnum):
    EXPANDED = "expanded_percentile"
    PERCENTILE = "percentile"


class Pairing(StrEnum):
    AUTO = "auto"
    PAIRED = "paired"
    UNPAIRED = "unpaired"


class OnUncertain(StrEnum):
    FAIL = "fail"
    PASS = "pass"


class Correction(StrEnum):
    NONE = "none"
    BONFERRONI = "bonferroni"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


Confidence = Annotated[float, Field(gt=0.5, lt=1.0)]
MinSamples = Annotated[int, Field(ge=2)]
MinClusters = Annotated[int, Field(ge=2)]
LabelName = Annotated[str, Field(min_length=1)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Resamples = Annotated[int, Field(ge=1000, le=1_000_000)]


class Defaults(_Model):
    statistic: Statistic = Statistic.MEAN
    confidence: Confidence = 0.95
    min_samples: MinSamples = 10
    pairing: Pairing = Pairing.AUTO
    max_missing_fraction: Fraction = 0.0
    bootstrap_resamples: Resamples = 10_000
    bootstrap_seed: int = 0
    bootstrap_method: BootstrapMethod = BootstrapMethod.EXPANDED
    cluster_by: LabelName | None = None
    min_clusters: MinClusters = 3


class Policy(_Model):
    on_inconclusive: OnUncertain = OnUncertain.FAIL
    on_invalid: OnUncertain = OnUncertain.FAIL
    correction: Correction = Correction.NONE
    require_same_scenario: bool = True


class MetricGate(_Model):
    """How one metric is gated. Unset optional fields fall back to ``defaults``."""

    max_absolute_degradation: Annotated[float, Field(ge=0)] | None = None
    max_relative_degradation: Annotated[float, Field(ge=0)] | None = None
    direction: Direction | None = None
    required: bool = True
    statistic: Statistic | None = None
    confidence: Confidence | None = None
    min_samples: MinSamples | None = None
    pairing: Pairing | None = None
    max_missing_fraction: Fraction | None = None
    cluster_by: LabelName | None = None
    min_clusters: MinClusters | None = None

    @model_validator(mode="after")
    def _needs_threshold(self) -> Self:
        if self.max_absolute_degradation is None and self.max_relative_degradation is None:
            raise ValueError(
                "set max_absolute_degradation and/or max_relative_degradation "
                "(use 0 to tolerate no degradation at all)"
            )
        return self


class ResolvedMetricGate(_Model):
    """A :class:`MetricGate` with every default filled in."""

    name: str
    max_absolute_degradation: float | None
    max_relative_degradation: float | None
    direction: Direction | None
    required: bool
    statistic: Statistic
    confidence: float
    min_samples: int
    pairing: Pairing
    max_missing_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_method: BootstrapMethod
    cluster_by: str | None
    min_clusters: int


class GateConfig(_Model):
    version: Literal[1] = 1
    defaults: Defaults = Field(default_factory=Defaults)
    policy: Policy = Field(default_factory=Policy)
    metrics: Annotated[dict[str, MetricGate], Field(min_length=1)]

    @model_validator(mode="after")
    def _clusters_need_the_mean(self) -> Self:
        for name in self.metrics:
            gate = self.resolve(name)
            if gate.cluster_by is not None and gate.statistic is not Statistic.MEAN:
                raise ValueError(
                    f"metric {name!r}: cluster_by compares the mean of each cluster, so it needs "
                    "statistic: mean"
                )
        return self

    def with_default_cluster_by(self, label: str) -> GateConfig:
        """This config with ``defaults.cluster_by`` set to ``label``, unless it is already set."""
        if self.defaults.cluster_by is not None:
            return self
        defaults = self.defaults.model_copy(update={"cluster_by": label})
        return GateConfig.model_validate({**self.model_dump(), "defaults": defaults.model_dump()})

    def adjusted_confidence(self, name: str) -> float:
        """The confidence of one metric's interval after any multiple-testing correction."""
        confidence = self.resolve(name).confidence
        if self.policy.correction is Correction.BONFERRONI:
            return 1.0 - (1.0 - confidence) / len(self.metrics)
        return confidence

    def resolve(self, name: str) -> ResolvedMetricGate:
        gate = self.metrics[name]
        d = self.defaults
        return ResolvedMetricGate(
            name=name,
            max_absolute_degradation=gate.max_absolute_degradation,
            max_relative_degradation=gate.max_relative_degradation,
            direction=gate.direction,
            required=gate.required,
            statistic=gate.statistic or d.statistic,
            confidence=gate.confidence or d.confidence,
            min_samples=gate.min_samples or d.min_samples,
            pairing=gate.pairing or d.pairing,
            max_missing_fraction=(
                d.max_missing_fraction
                if gate.max_missing_fraction is None
                else gate.max_missing_fraction
            ),
            bootstrap_resamples=d.bootstrap_resamples,
            bootstrap_seed=d.bootstrap_seed,
            bootstrap_method=d.bootstrap_method,
            cluster_by=gate.cluster_by or d.cluster_by,
            min_clusters=gate.min_clusters or d.min_clusters,
        )


def config_from_data(data: Any, origin: str = "<config>") -> GateConfig:
    if not isinstance(data, dict):
        raise SchemaError(f"{origin}: expected a mapping at the top level")
    try:
        return GateConfig.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(
            f"{origin}: not a valid gate config", format_validation_error(exc)
        ) from exc


def load_config(path: Path) -> GateConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path}: cannot read file ({exc.strerror or exc})") from exc
    try:
        data = load_yaml(text, str(path))
    except yaml.YAMLError as exc:
        raise SchemaError(f"{path}: invalid YAML: {exc}") from exc
    return config_from_data(data, str(path))
