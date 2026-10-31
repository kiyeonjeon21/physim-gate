"""Versioned comparison result (``physim-gate.comparison``) and the verdict vocabulary.

A comparison result is self-contained: it embeds the gate config, both runs' provenance and the
analyzed samples, so every report can be regenerated from it without the original inputs.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physim_gate.config import GateConfig, Statistic
from physim_gate.schema import (
    Direction,
    Domain,
    Environment,
    Identifier,
    InputError,
    RunStatus,
    SchemaError,
    Simulator,
    Source,
    format_validation_error,
    parse_json_text,
)

COMPARISON_KIND = "physim-gate.comparison"
COMPARISON_SCHEMA_MAJOR = 1
COMPARISON_SCHEMA_VERSION = "1.1"


class ExitCode(IntEnum):
    """Process exit codes. Stable across 0.x releases."""

    OK = 0
    """Gate passed, or the command succeeded."""
    REGRESSION = 1
    """A blocking metric regressed (``compare``), or the input is invalid (``validate``)."""
    USAGE = 2
    """Bad arguments, unreadable input, or an invalid config or result document."""
    UNCERTAIN = 3
    """No regression shown, but a blocking metric was inconclusive or invalid."""
    INTERNAL = 4
    """Unexpected internal error (a bug, please report it)."""


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


class Evidence(StrEnum):
    """What the statistics support, independent of policy."""

    REGRESSION = "regression"
    """The whole confidence interval of the degradation lies above the allowed margin."""
    NON_INFERIOR = "non_inferior"
    """The whole confidence interval lies at or below the allowed margin."""
    INCONCLUSIVE = "inconclusive"
    """The confidence interval straddles the margin."""
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    """Fewer observations than ``min_samples``. No interval is computed."""
    NOT_EVALUATED = "not_evaluated"
    """The data could not be compared (see the reasons)."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class SampleSummary(_Model):
    """The observations of one metric in one run that entered the comparison."""

    n: int
    n_trials: int
    n_missing: int
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    min: float | None = None
    max: float | None = None
    values: list[float] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)


class PairingInfo(_Model):
    mode: Literal["paired", "unpaired"]
    n_pairs: int | None = None
    unmatched_baseline: list[str] = Field(default_factory=list)
    unmatched_candidate: list[str] = Field(default_factory=list)
    note: str | None = None


class ClusterInfo(_Model):
    """How trials were grouped into clusters (``cluster_by``) and each cluster's mean.

    With clusters the unit of comparison is the cluster, not the trial: the interval comes from
    the spread of the cluster means, so variation between clusters (training seeds, say) is
    counted. ``mode`` says whether clusters were matched by their label value.
    """

    label: str
    mode: Literal["paired", "unpaired"]
    baseline_means: dict[str, float]
    candidate_means: dict[str, float]
    baseline_trials: dict[str, int]
    candidate_trials: dict[str, int]
    unmatched_baseline: list[str] = Field(default_factory=list)
    unmatched_candidate: list[str] = Field(default_factory=list)
    note: str | None = None


class Numerical(_Model):
    """Point estimates. ``degradation > 0`` always means the candidate is worse."""

    baseline: float
    candidate: float
    delta: float
    degradation: float
    relative_degradation: float | None
    margin: float
    margin_basis: str
    exceeds_margin: bool


class StatisticalEvidence(_Model):
    outcome: Evidence
    method: str | None = None
    confidence: float | None = None
    adjusted_confidence: float | None = None
    resamples: int | None = None
    seed: str | None = None
    degradation_ci: tuple[float, float] | None = None
    excess_ci: tuple[float, float] | None = None


class MetricComparison(_Model):
    name: str
    unit: str | None = None
    direction: Direction | None = None
    required: bool
    statistic: Statistic
    min_samples: int
    pairing: PairingInfo | None = None
    clusters: ClusterInfo | None = None
    baseline: SampleSummary | None = None
    candidate: SampleSummary | None = None
    numerical: Numerical | None = None
    evidence: StatisticalEvidence
    verdict: Verdict
    blocking: bool
    reasons: list[str]


class RunSummary(_Model):
    run_id: str
    created_at: datetime | None = None
    domain: Domain
    simulator: Simulator
    scenario: Identifier
    policy: Identifier | None = None
    source: Source
    environment: Environment
    seeds: list[int]
    status: RunStatus
    n_trials: int
    n_trials_not_ok: int
    labels: dict[str, str] = Field(default_factory=dict)


class InputRef(_Model):
    path: str | None = None
    sha256: str | None = None


class Inputs(_Model):
    """The files compared. A side read from several files lists them in ``*_runs``; its
    ``sha256`` then digests the files' digests in order."""

    baseline: InputRef = Field(default_factory=InputRef)
    candidate: InputRef = Field(default_factory=InputRef)
    config: InputRef = Field(default_factory=InputRef)
    adapter: str | None = None
    baseline_runs: list[InputRef] = Field(default_factory=list)
    candidate_runs: list[InputRef] = Field(default_factory=list)


class Tool(_Model):
    name: str = "physim-gate"
    version: str


class ComparisonResult(_Model):
    kind: Literal["physim-gate.comparison"] = "physim-gate.comparison"
    schema_version: str = COMPARISON_SCHEMA_VERSION
    tool: Tool
    generated_at: datetime
    inputs: Inputs
    config: GateConfig
    baseline: RunSummary
    candidate: RunSummary
    run_issues: list[str]
    run_blocking_issues: list[str]
    metrics: list[MetricComparison]
    untracked_metrics: list[str]
    verdict: Verdict
    exit_code: int
    summary: str


def load_comparison(path: Path) -> ComparisonResult:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path}: cannot read file ({exc.strerror or exc})") from exc
    return comparison_from_data(parse_json_text(text, str(path)), str(path))


def comparison_from_data(data: Any, origin: str = "<data>") -> ComparisonResult:
    if not isinstance(data, dict):
        raise SchemaError(f"{origin}: expected a JSON object at the top level")
    if data.get("kind") != COMPARISON_KIND:
        raise SchemaError(
            f"{origin}: not a comparison result (missing kind {COMPARISON_KIND!r}); "
            "produce one with 'physim-gate compare --output'"
        )
    version = str(data.get("schema_version", ""))
    if version.split(".")[0] != str(COMPARISON_SCHEMA_MAJOR):
        raise SchemaError(
            f"{origin}: unsupported comparison schema version {version!r}; "
            f"this release reads {COMPARISON_SCHEMA_MAJOR}.x"
        )
    try:
        return ComparisonResult.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(
            f"{origin}: not a valid comparison result", format_validation_error(exc)
        ) from exc
