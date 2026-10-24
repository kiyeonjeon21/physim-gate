"""The baseline-versus-candidate comparison engine.

For every gated metric the engine produces three layers that are kept separate on purpose:

1. *Numerical*: point estimates and whether the observed degradation exceeds the margin.
2. *Statistical evidence*: a bootstrap interval of the degradation, classified against the margin.
3. *Policy verdict*: PASS / FAIL / INCONCLUSIVE / INVALID, and whether it blocks the gate.

See ``docs/comparison-semantics.md`` for the full decision rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from physim_gate import __version__
from physim_gate.comparison import (
    ComparisonResult,
    Evidence,
    ExitCode,
    Inputs,
    MetricComparison,
    Numerical,
    PairingInfo,
    RunSummary,
    SampleSummary,
    StatisticalEvidence,
    Tool,
    Verdict,
)
from physim_gate.config import (
    BootstrapMethod,
    GateConfig,
    OnUncertain,
    Pairing,
    ResolvedMetricGate,
    Statistic,
)
from physim_gate.provenance import utc_now
from physim_gate.schema import (
    FAILURE_RATE_METRIC,
    Direction,
    RunResult,
    RunStatus,
    TrialStatus,
)
from physim_gate.stats import (
    BootstrapResult,
    bootstrap_degradation,
    is_binary,
    mean,
    median,
    newcombe_difference,
    stat_fn,
    stdev,
)

_MAX_LISTED_IDS = 20


class _Invalid(Exception):
    """Internal: the metric cannot be compared; the message becomes the reason."""


@dataclass
class _Samples:
    baseline: list[float]
    candidate: list[float]
    baseline_ids: list[str]
    candidate_ids: list[str]
    pairing: PairingInfo


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    config: GateConfig,
    *,
    inputs: Inputs | None = None,
    generated_at: datetime | None = None,
) -> ComparisonResult:
    """Compare two runs under a gate config and return a self-contained result."""
    run_issues, run_blocking = _check_runs(baseline, candidate, config, inputs)

    metrics = [
        _compare_metric(config.resolve(name), baseline, candidate, config)
        for name in config.metrics
    ]

    declared = (set(baseline.metrics) | set(candidate.metrics)) - set(config.metrics)
    verdict, summary = _overall(metrics, run_blocking)
    return ComparisonResult(
        tool=Tool(version=__version__),
        generated_at=generated_at or utc_now(),
        inputs=inputs or Inputs(),
        config=config,
        baseline=_summarize_run(baseline),
        candidate=_summarize_run(candidate),
        run_issues=run_issues,
        run_blocking_issues=run_blocking,
        metrics=metrics,
        untracked_metrics=sorted(declared),
        verdict=verdict,
        exit_code=int(exit_code_for(verdict)),
        summary=summary,
    )


def exit_code_for(verdict: Verdict) -> ExitCode:
    if verdict is Verdict.PASS:
        return ExitCode.OK
    if verdict is Verdict.FAIL:
        return ExitCode.REGRESSION
    return ExitCode.UNCERTAIN


def _check_runs(
    baseline: RunResult, candidate: RunResult, config: GateConfig, inputs: Inputs | None
) -> tuple[list[str], list[str]]:
    """Run-level comparability checks.

    Blocking issues make the overall verdict INVALID whatever ``policy.on_invalid`` says: that
    setting tolerates metrics that cannot be compared, not runs that cannot be compared.
    """
    issues: list[str] = []
    blocking: list[str] = []
    same_scenario = config.policy.require_same_scenario

    if baseline.scenario.id != candidate.scenario.id:
        message = (
            f"scenario differs: baseline {baseline.scenario.id!r}, "
            f"candidate {candidate.scenario.id!r}"
        )
        if same_scenario:
            blocking.append(message + " (policy.require_same_scenario is true)")
        else:
            issues.append(message)
    elif baseline.scenario.version != candidate.scenario.version:
        message = (
            f"scenario version differs: {baseline.scenario.version!r} vs "
            f"{candidate.scenario.version!r}"
        )
        if same_scenario and None not in (baseline.scenario.version, candidate.scenario.version):
            blocking.append(message + " (policy.require_same_scenario is true)")
        else:
            issues.append(message)
    if baseline.domain != candidate.domain:
        issues.append(f"domain differs: {baseline.domain.value} vs {candidate.domain.value}")
    if baseline.simulator.name != candidate.simulator.name:
        issues.append(
            f"simulator differs: {baseline.simulator.name!r} vs {candidate.simulator.name!r}"
        )
    elif baseline.simulator.version != candidate.simulator.version:
        issues.append(
            f"simulator version differs: {baseline.simulator.version!r} vs "
            f"{candidate.simulator.version!r}; differences may come from the simulator"
        )
    if baseline.run_id == candidate.run_id:
        issues.append(f"baseline and candidate share run_id {baseline.run_id!r}")
    if (
        inputs is not None
        and inputs.baseline.sha256 is not None
        and inputs.baseline.sha256 == inputs.candidate.sha256
    ):
        issues.append(
            "baseline and candidate inputs have identical content; expected only from a "
            "deterministic simulator and a change that does not affect it"
        )

    gates_failures = FAILURE_RATE_METRIC in config.metrics
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        if run.status is RunStatus.FAILED:
            blocking.append(f"{label} run status is 'failed'")
        elif run.status is not RunStatus.COMPLETED:
            issues.append(f"{label} run status is {run.status.value!r}")
        not_ok = sum(1 for t in run.trials if t.status is not TrialStatus.OK)
        if run.trials and not_ok == len(run.trials):
            blocking.append(
                f"every {label} trial has a status other than 'ok' ({not_ok} of {not_ok})"
            )
        elif not_ok and not gates_failures:
            issues.append(
                f"{not_ok} of {len(run.trials)} {label} trials have a status other than 'ok'; "
                f"gate {FAILURE_RATE_METRIC} to catch more failures"
            )

    seeds = _seed_conflicts(baseline, candidate)
    if seeds:
        examples = ", ".join(f"{k}: {b} vs {c}" for k, b, c in seeds[:3])
        message = (
            f"{len(seeds)} trial ids have different seeds in the two runs ({examples}), so "
            "they do not name the same conditions"
        )
        unpaired = all(config.resolve(name).pairing is Pairing.UNPAIRED for name in config.metrics)
        if unpaired:
            issues.append(message + "; every metric is compared unpaired")
        else:
            blocking.append(message)
    return issues, blocking


def _seed_conflicts(baseline: RunResult, candidate: RunResult) -> list[tuple[str, int, int]]:
    """Shared trial ids whose recorded seeds differ, in baseline order."""
    candidate_seeds = {t.trial_id: t.seed for t in candidate.trials if t.seed is not None}
    return [
        (t.trial_id, t.seed, other)
        for t in baseline.trials
        if t.seed is not None
        and (other := candidate_seeds.get(t.trial_id)) is not None
        and other != t.seed
    ]


def _compare_metric(
    gate: ResolvedMetricGate,
    baseline: RunResult,
    candidate: RunResult,
    config: GateConfig,
) -> MetricComparison:
    adjusted_confidence = config.adjusted_confidence(gate.name)
    base = {
        "name": gate.name,
        "required": gate.required,
        "statistic": gate.statistic,
        "min_samples": gate.min_samples,
    }
    try:
        unit, direction = _metric_identity(gate, baseline, candidate)
        base |= {"unit": unit, "direction": direction}
        samples = _collect_samples(gate, baseline, candidate)
    except _Invalid as exc:
        return _finish(
            base,
            config,
            evidence=StatisticalEvidence(outcome=Evidence.NOT_EVALUATED),
            verdict=Verdict.INVALID,
            reasons=[str(exc)],
        )

    base |= {
        "pairing": samples.pairing,
        "baseline": _summarize(samples.baseline, samples.baseline_ids, baseline, gate.name),
        "candidate": _summarize(samples.candidate, samples.candidate_ids, candidate, gate.name),
    }
    reasons: list[str] = []
    if samples.pairing.note:
        reasons.append(samples.pairing.note)

    fn = stat_fn(gate.statistic)
    b_stat = fn(samples.baseline)
    c_stat = fn(samples.candidate)
    sign = 1.0 if direction is Direction.LOWER_IS_BETTER else -1.0
    degradation = sign * (c_stat - b_stat)

    try:
        margin_fn, margin_basis = _margin(gate, b_stat, samples.baseline)
    except _Invalid as exc:
        return _finish(
            base,
            config,
            evidence=StatisticalEvidence(outcome=Evidence.NOT_EVALUATED),
            verdict=Verdict.INVALID,
            reasons=[*reasons, str(exc)],
        )
    margin = margin_fn(b_stat)
    numerical = Numerical(
        baseline=b_stat,
        candidate=c_stat,
        delta=c_stat - b_stat,
        degradation=degradation,
        relative_degradation=degradation / abs(b_stat) if b_stat != 0 else None,
        margin=margin,
        margin_basis=margin_basis,
        exceeds_margin=degradation > margin,
    )
    base["numerical"] = numerical

    paired = samples.pairing.mode == "paired"
    n_effective = (
        len(samples.baseline) if paired else min(len(samples.baseline), len(samples.candidate))
    )
    if n_effective < gate.min_samples:
        unit_word = "matched pairs" if paired else "observations in the smaller sample"
        reasons.append(
            f"insufficient samples: {n_effective} {unit_word}, min_samples is {gate.min_samples}; "
            "no confidence interval computed"
        )
        return _finish(
            base,
            config,
            evidence=StatisticalEvidence(outcome=Evidence.INSUFFICIENT_SAMPLES),
            verdict=Verdict.INCONCLUSIVE,
            reasons=reasons,
        )

    bootstrap_seed = f"{gate.bootstrap_seed}:{gate.name}"
    seed: str | None = bootstrap_seed
    resamples: int | None = gate.bootstrap_resamples
    binary = gate.statistic is Statistic.MEAN and is_binary(samples.baseline + samples.candidate)
    if binary:
        # 0/1 metrics: a percentile bootstrap collapses to a zero-width interval when a run has no
        # events, which would "prove" non-inferiority from 0 of 30 trials. Use a score interval.
        low, high = newcombe_difference(
            samples.baseline, samples.candidate, paired=paired, confidence=adjusted_confidence
        )
        degradation_ci = (low, high) if sign > 0 else (-high, -low)
        excess = (degradation_ci[0] - margin, degradation_ci[1] - margin)
        boot = BootstrapResult(degradation=degradation_ci, excess=excess)
        method = "paired_newcombe" if paired else "unpaired_newcombe"
        seed = resamples = None
    else:
        boot = bootstrap_degradation(
            samples.baseline,
            samples.candidate,
            paired=paired,
            statistic=gate.statistic,
            sign=sign,
            margin=margin_fn,
            confidence=adjusted_confidence,
            resamples=gate.bootstrap_resamples,
            seed=bootstrap_seed,
            method=gate.bootstrap_method,
        )
        method = "paired_bootstrap" if paired else "unpaired_bootstrap"
        if gate.bootstrap_method is BootstrapMethod.EXPANDED:
            method += "_expanded"
    excess_low, excess_high = boot.excess
    pct = _pct(adjusted_confidence)
    if excess_low > 0:
        outcome, verdict = Evidence.REGRESSION, Verdict.FAIL
        reasons.append(
            f"regression: the {pct} interval of the degradation lies entirely above the "
            f"allowed margin ({_fmt(margin)}{_unit(unit)})"
        )
    elif excess_high <= 0:
        outcome, verdict = Evidence.NON_INFERIOR, Verdict.PASS
        reasons.append(
            f"no regression: the {pct} interval of the degradation lies at or below the "
            f"allowed margin ({_fmt(margin)}{_unit(unit)})"
        )
    else:
        outcome, verdict = Evidence.INCONCLUSIVE, Verdict.INCONCLUSIVE
        observed = "exceeds" if numerical.exceeds_margin else "is within"
        reasons.append(
            f"inconclusive: the observed degradation {observed} the margin, but the {pct} "
            "interval straddles it; more trials would narrow the interval"
        )

    evidence = StatisticalEvidence(
        outcome=outcome,
        method=method,
        confidence=gate.confidence,
        adjusted_confidence=adjusted_confidence,
        resamples=resamples,
        seed=seed,
        degradation_ci=boot.degradation,
        excess_ci=boot.excess,
    )
    return _finish(base, config, evidence=evidence, verdict=verdict, reasons=reasons)


def _finish(
    base: dict[str, object],
    config: GateConfig,
    *,
    evidence: StatisticalEvidence,
    verdict: Verdict,
    reasons: list[str],
) -> MetricComparison:
    required = bool(base["required"])
    if verdict is Verdict.FAIL:
        blocking = required
    elif verdict is Verdict.INCONCLUSIVE:
        blocking = required and config.policy.on_inconclusive is OnUncertain.FAIL
    elif verdict is Verdict.INVALID:
        blocking = required and config.policy.on_invalid is OnUncertain.FAIL
    else:
        blocking = False
    if not required and verdict is not Verdict.PASS:
        reasons = [*reasons, "metric is not required, so this does not block the gate"]
    return MetricComparison.model_validate(
        {**base, "evidence": evidence, "verdict": verdict, "blocking": blocking, "reasons": reasons}
    )


def _metric_identity(
    gate: ResolvedMetricGate, baseline: RunResult, candidate: RunResult
) -> tuple[str, Direction]:
    b_spec = baseline.metric_spec(gate.name)
    c_spec = candidate.metric_spec(gate.name)
    missing = [label for label, s in (("baseline", b_spec), ("candidate", c_spec)) if s is None]
    if missing or b_spec is None or c_spec is None:
        raise _Invalid(f"metric is not declared in the {' and '.join(missing)} run")
    if b_spec.direction != c_spec.direction:
        raise _Invalid(
            f"direction differs between runs: baseline {b_spec.direction.value}, "
            f"candidate {c_spec.direction.value}"
        )
    if b_spec.unit != c_spec.unit:
        raise _Invalid(
            f"unit differs between runs: baseline {b_spec.unit!r}, candidate {c_spec.unit!r}"
        )
    if gate.direction is not None and gate.direction != b_spec.direction:
        raise _Invalid(
            f"config says {gate.direction.value} but the runs declare {b_spec.direction.value}"
        )
    return b_spec.unit, b_spec.direction


def _collect_samples(
    gate: ResolvedMetricGate, baseline: RunResult, candidate: RunResult
) -> _Samples:
    b_values = baseline.metric_values(gate.name)
    c_values = candidate.metric_values(gate.name)

    b_keys, c_keys = set(b_values), set(c_values)
    # When the runs share trial ids, they claim to cover the same set of conditions, so a
    # trial that only one run reports is missing from the other. Otherwise a run could drop
    # its crashed or bad trials and look better than one that reports them as null. Runs with
    # disjoint ids are independent samples, and nothing can be said to be missing.
    overlapping = bool(b_keys & c_keys)
    if overlapping and gate.name == FAILURE_RATE_METRIC:
        # A trial that a run left out did not complete in that run: count it as a failure.
        # This keeps physim.failure_rate a working safeguard when max_missing_fraction is
        # raised, which is the documented way to tolerate missing values.
        b_values = {**b_values, **dict.fromkeys(c_keys - b_keys, 1.0)}
        c_values = {**c_values, **dict.fromkeys(b_keys - c_keys, 1.0)}
        b_keys, c_keys = set(b_values), set(c_values)
    for label, values, absent in (
        ("baseline", b_values, len(c_keys - b_keys) if overlapping else 0),
        ("candidate", c_values, len(b_keys - c_keys) if overlapping else 0),
    ):
        if not values:
            raise _Invalid(f"{label} run has no trials")
        n_null = sum(1 for v in values.values() if v is None)
        n_missing = n_null + absent
        expected = len(values) + absent
        fraction = n_missing / expected
        if fraction > gate.max_missing_fraction:
            other = "candidate" if label == "baseline" else "baseline"
            detail = f"{n_null} null values"
            if absent:
                detail += f", {absent} trial ids reported only by the {other}"
            raise _Invalid(
                f"{label}: {n_missing} of {expected} trials missing ({detail}; "
                f"{fraction:.0%}), above max_missing_fraction "
                f"({gate.max_missing_fraction:.0%}). Report every trial, with values for failed "
                f"trials too, or raise max_missing_fraction and gate {FAILURE_RATE_METRIC} so "
                "failures stay visible"
            )

    if gate.pairing is Pairing.UNPAIRED:
        mode, note = "unpaired", None
    elif gate.pairing is Pairing.PAIRED:
        if not b_keys & c_keys:
            raise _Invalid("pairing is 'paired' but no trial_id appears in both runs")
        mode, note = "paired", None
    elif b_keys == c_keys:
        mode, note = "paired", None
    else:
        mode = "unpaired"
        if b_keys & c_keys:
            note = (
                "trial_ids differ between runs, so trials are compared as independent samples "
                "(set pairing: paired to match on the shared ids instead)"
            )
        else:
            note = "the runs share no trial_id, so trials are compared as independent samples"

    if mode == "unpaired":
        b_obs = [(k, v) for k, v in b_values.items() if v is not None]
        c_obs = [(k, v) for k, v in c_values.items() if v is not None]
        pairing = PairingInfo(mode="unpaired", note=note)
    else:
        shared = [k for k in b_values if k in c_keys]
        complete = [k for k in shared if b_values[k] is not None and c_values[k] is not None]
        b_obs = [(k, v) for k in complete if (v := b_values[k]) is not None]
        c_obs = [(k, v) for k in complete if (v := c_values[k]) is not None]
        unmatched_b = sorted(b_keys - c_keys)
        unmatched_c = sorted(c_keys - b_keys)
        dropped = len(shared) - len(complete)
        notes = []
        if unmatched_b or unmatched_c:
            notes.append(
                f"{len(unmatched_b)} baseline and {len(unmatched_c)} candidate trials have no "
                "match and are excluded"
            )
        if dropped:
            notes.append(f"{dropped} matched pairs dropped because a value is missing")
        pairing = PairingInfo(
            mode="paired",
            n_pairs=len(complete),
            unmatched_baseline=unmatched_b[:_MAX_LISTED_IDS],
            unmatched_candidate=unmatched_c[:_MAX_LISTED_IDS],
            note="; ".join(notes) or None,
        )

    if not b_obs or not c_obs:
        raise _Invalid("no usable observations in at least one run")
    return _Samples(
        baseline=[v for _, v in b_obs],
        candidate=[v for _, v in c_obs],
        baseline_ids=[k for k, _ in b_obs],
        candidate_ids=[k for k, _ in c_obs],
        pairing=pairing,
    )


def _margin(
    gate: ResolvedMetricGate, b_stat: float, baseline: list[float]
) -> tuple[Callable[[float], float], str]:
    absolute = gate.max_absolute_degradation
    relative = gate.max_relative_degradation
    # A relative margin needs a ratio scale. When the baseline is 0, or its values have both
    # signs (a signed error, say), a fraction of |baseline| is 0 or arbitrarily close to it.
    if b_stat == 0:
        undefined = "the baseline is 0"
    elif min(baseline) < 0 < max(baseline):
        undefined = "the baseline values have both signs"
    else:
        undefined = None
    if relative is not None and undefined is not None:
        if absolute is None:
            raise _Invalid(
                f"max_relative_degradation is undefined because {undefined}; "
                "add max_absolute_degradation for this metric"
            )
        return (
            lambda _b: absolute,
            f"absolute (the relative threshold is undefined because {undefined})",
        )
    if relative is None:
        assert absolute is not None
        return (lambda _b: absolute), "absolute"
    if absolute is None:
        return (lambda b: relative * abs(b)), f"relative ({relative:.4g} x |baseline|)"
    return (
        lambda b: min(absolute, relative * abs(b)),
        f"stricter of absolute ({absolute:.4g}) and relative ({relative:.4g} x |baseline|)",
    )


def _summarize(values: list[float], ids: list[str], run: RunResult, name: str) -> SampleSummary:
    all_values = run.metric_values(name)
    return SampleSummary(
        n=len(values),
        n_trials=len(all_values),
        n_missing=sum(1 for v in all_values.values() if v is None),
        mean=mean(values) if values else None,
        median=median(values) if values else None,
        stdev=stdev(values),
        min=min(values) if values else None,
        max=max(values) if values else None,
        values=values,
        trial_ids=ids,
    )


def _summarize_run(run: RunResult) -> RunSummary:
    return RunSummary(
        run_id=run.run_id,
        created_at=run.created_at,
        domain=run.domain,
        simulator=run.simulator,
        scenario=run.scenario,
        policy=run.policy,
        source=run.source,
        environment=run.environment,
        seeds=run.seeds,
        status=run.status,
        n_trials=len(run.trials),
        n_trials_not_ok=sum(1 for t in run.trials if t.status is not TrialStatus.OK),
        labels=run.labels,
    )


def _overall(metrics: list[MetricComparison], run_blocking: list[str]) -> tuple[Verdict, str]:
    blocking = [m for m in metrics if m.blocking]
    failed = [m.name for m in blocking if m.verdict is Verdict.FAIL]
    invalid = [m.name for m in blocking if m.verdict is Verdict.INVALID]
    uncertain = [m.name for m in blocking if m.verdict is Verdict.INCONCLUSIVE]
    total = len(metrics)
    if run_blocking:
        return Verdict.INVALID, "runs are not comparable: " + "; ".join(run_blocking)
    if all(m.verdict is Verdict.INVALID for m in metrics):
        # Whatever policy.on_invalid and required say, a gate that compared nothing has
        # shown nothing.
        return Verdict.INVALID, (
            f"no gated metric could be compared: {', '.join(m.name for m in metrics)}"
        )
    if failed:
        return Verdict.FAIL, (
            f"regression detected in {len(failed)} of {total} gated metrics: {', '.join(failed)}"
        )
    if invalid:
        return Verdict.INVALID, (
            f"{len(invalid)} of {total} gated metrics could not be compared: {', '.join(invalid)}"
        )
    if uncertain:
        return Verdict.INCONCLUSIVE, (
            f"no regression shown, but the evidence is inconclusive for {len(uncertain)} of "
            f"{total} gated metrics: {', '.join(uncertain)}"
        )
    warned = [m.name for m in metrics if m.verdict is not Verdict.PASS]
    if not any(m.required for m in metrics):
        detail = f"findings for: {', '.join(warned)}" if warned else "every metric passed"
        return Verdict.PASS, f"no metric is required, so nothing blocks the gate; {detail}"
    if warned:
        return Verdict.PASS, (
            f"no blocking regression; non-blocking findings for: {', '.join(warned)}"
        )
    return Verdict.PASS, f"all {total} gated metrics show no regression beyond their margins"


def _pct(confidence: float) -> str:
    return f"{confidence * 100:.4g}%"


def _unit(unit: str) -> str:
    return f" {unit}" if unit else ""


def _fmt(value: float) -> str:
    return f"{value:.4g}"
