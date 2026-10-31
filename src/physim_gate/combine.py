"""Combine several run results of one side (one per training seed, say) into a single run.

Every trial gets the label :data:`RUN_LABEL` naming the run it came from, so the comparison can
treat each run as a cluster (``cluster_by: physim.run``). Pooling the trials of several runs as
if they were independent would ignore the variation between runs; see
``docs/comparison-semantics.md``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from physim_gate.schema import (
    Environment,
    InputError,
    MetricSpec,
    RunResult,
    RunStatus,
    Source,
    Trial,
)

RUN_LABEL = "physim.run"
"""Trial label set by :func:`combine_runs`: the name of the run the trial came from."""


def cluster_names(paths: Sequence[Path], runs: Sequence[RunResult]) -> list[str]:
    """Distinct, readable names for the runs of one side.

    The run ids when they differ. Otherwise the paths without their extension, shortened by
    the parts every path shares: directories in front, and any trailing part (so
    ``runs/s0/eval_info.json`` and ``runs/s1/eval_info.json`` become ``s0`` and ``s1``).
    """
    ids = [run.run_id for run in runs]
    if len(set(ids)) == len(ids):
        return ids
    stems = [str(path.with_suffix("")) if path.suffix else str(path) for path in paths]
    if len(set(stems)) != len(stems):
        raise InputError(f"the same file is given twice: {', '.join(str(p) for p in paths)}")
    return _shorten(stems)


_SEPARATORS = re.compile(r"(?<=[/_.-])")


def _shorten(names: list[str]) -> list[str]:
    """Drop the leading directories and the trailing pieces that every name shares."""
    pieces = [_SEPARATORS.split(name) for name in names]
    # Trailing pieces, but never the last distinguishing one.
    tail = 0
    while all(len(p) > tail + 1 for p in pieces) and len({p[-1 - tail] for p in pieces}) == 1:
        tail += 1
    trimmed = [p[: len(p) - tail] if tail else p for p in pieces]
    # Leading whole directories only, so a name never starts mid-word.
    head = 0
    while (
        all(len(p) > head + 1 for p in trimmed)
        and len({p[head] for p in trimmed}) == 1
        and trimmed[0][head].endswith("/")
    ):
        head += 1
    shortened = ["".join(p[head:]).rstrip("/_.-") for p in trimmed]
    if len(set(shortened)) != len(shortened) or not all(shortened):
        return names
    return shortened


def combine_runs(side: str, names: Sequence[str], runs: Sequence[RunResult]) -> RunResult:
    """One run holding the trials of ``runs``, each labeled with its run's name.

    The runs must describe the same scenario, simulator and domain; ``side`` ("baseline" or
    "candidate") names them in errors.
    """
    if len(names) != len(runs) or not runs:
        raise ValueError("combine_runs needs one name per run, and at least one run")
    first = runs[0]
    for field in ("scenario", "simulator", "domain"):
        described = [_describe(getattr(run, field)) for run in runs]
        if len(set(described)) > 1:
            details = ", ".join(f"{n}: {d}" for n, d in zip(names, described, strict=True))
            raise InputError(
                f"the {side} runs differ in {field} ({details}); combine only runs of the "
                "same scenario, simulator and domain"
            )

    metrics: dict[str, MetricSpec] = {}
    for name, run in zip(names, runs, strict=True):
        for metric, spec in run.metrics.items():
            known = metrics.setdefault(metric, spec)
            if (known.direction, known.unit) != (spec.direction, spec.unit):
                raise InputError(
                    f"the {side} runs declare metric {metric!r} differently ({name}: "
                    f"{spec.direction.value}, unit {spec.unit!r}; earlier: "
                    f"{known.direction.value}, unit {known.unit!r})"
                )

    trials: list[Trial] = []
    for name, run in zip(names, runs, strict=True):
        for trial in run.trials:
            if RUN_LABEL in trial.labels:
                raise InputError(
                    f"{side} run {name}: trial {trial.trial_id!r} already has the label "
                    f"{RUN_LABEL!r}, which PhySim Gate sets when it combines runs"
                )
            trials.append(
                trial.model_copy(
                    update={
                        "trial_id": f"{name}/{trial.trial_id}",
                        "labels": {**trial.labels, RUN_LABEL: name},
                    }
                )
            )

    statuses = {run.status for run in runs}
    if statuses == {RunStatus.FAILED}:
        status = RunStatus.FAILED
    elif statuses == {RunStatus.COMPLETED}:
        status = RunStatus.COMPLETED
    else:
        status = RunStatus.PARTIAL

    def shared(values: list[object]) -> bool:
        return all(v == values[0] for v in values)

    return RunResult(
        run_id="+".join(names),
        domain=first.domain,
        simulator=first.simulator,
        scenario=first.scenario,
        policy=first.policy if shared([r.policy for r in runs]) else None,
        seeds=sorted({seed for run in runs for seed in run.seeds}),
        source=first.source if shared([r.source for r in runs]) else Source(),
        environment=(first.environment if shared([r.environment for r in runs]) else Environment()),
        status=status,
        metrics=metrics,
        trials=trials,
        artifacts=[a for run in runs for a in run.artifacts],
        labels={k: v for k, v in first.labels.items() if all(r.labels.get(k) == v for r in runs)},
    )


def _describe(value: object) -> str:
    if isinstance(value, BaseModel):
        fields = value.model_dump(exclude_none=True)
        return " ".join(f"{k} {v}" for k, v in fields.items() if v not in ({}, [])) or "-"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
