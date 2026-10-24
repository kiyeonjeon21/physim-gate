from __future__ import annotations

from typing import Any

import pytest

from physim_gate.comparison import Evidence, Verdict
from physim_gate.config import GateConfig
from physim_gate.engine import compare_runs
from physim_gate.schema import FAILURE_RATE_METRIC, RunResult
from tests.conftest import make_config, make_run

N = 40


def ramp(center: float, spread: float = 1.0, n: int = N) -> list[float]:
    """Deterministic, symmetric spread of n values around center."""
    return [center + spread * ((i % 10) - 4.5) / 4.5 for i in range(n)]


def gate(**metric: Any) -> GateConfig:
    return make_config({"m": metric})


def compare(
    base: list[float | None],
    cand: list[float | None],
    *,
    direction: str = "higher_is_better",
    config: GateConfig | None = None,
    base_ids: list[str] | None = None,
    cand_ids: list[str] | None = None,
) -> Any:
    b = make_run({"m": base}, run_id="b", directions={"m": direction}, trial_ids=base_ids)
    c = make_run({"m": cand}, run_id="c", directions={"m": direction}, trial_ids=cand_ids)
    result = compare_runs(b, c, config or gate(max_absolute_degradation=0.5))
    return result.metrics[0], result


# --- direction semantics -------------------------------------------------------------------


def test_higher_is_better_drop_is_a_regression() -> None:
    metric, result = compare(ramp(10), ramp(8))
    assert metric.numerical.degradation == pytest.approx(2.0)
    assert metric.evidence.outcome is Evidence.REGRESSION
    assert metric.verdict is Verdict.FAIL
    assert result.verdict is Verdict.FAIL
    assert result.exit_code == 1


def test_higher_is_better_increase_is_fine() -> None:
    metric, result = compare(ramp(10), ramp(12))
    assert metric.numerical.degradation == pytest.approx(-2.0)
    assert metric.verdict is Verdict.PASS
    assert result.exit_code == 0


def test_lower_is_better_increase_is_a_regression() -> None:
    metric, _ = compare(ramp(10), ramp(12), direction="lower_is_better")
    assert metric.numerical.degradation == pytest.approx(2.0)
    assert metric.verdict is Verdict.FAIL


def test_lower_is_better_decrease_is_fine() -> None:
    metric, _ = compare(ramp(10), ramp(8), direction="lower_is_better")
    assert metric.verdict is Verdict.PASS


def test_identical_deterministic_runs_pass_with_zero_width_interval() -> None:
    values = ramp(3.0)
    metric, _ = compare(values, values, config=gate(max_absolute_degradation=0))
    assert metric.evidence.degradation_ci == (0.0, 0.0)
    assert metric.verdict is Verdict.PASS


# --- thresholds ----------------------------------------------------------------------------


def test_absolute_margin_tolerates_small_degradation() -> None:
    # Constant shift of 0.3 with a 0.5 margin: degraded, but within tolerance.
    metric, _ = compare(ramp(10), ramp(9.7))
    assert metric.numerical.exceeds_margin is False
    assert metric.verdict is Verdict.PASS


def test_relative_margin_scales_with_baseline() -> None:
    config = gate(max_relative_degradation=0.10)
    metric, _ = compare(ramp(100), ramp(95), config=config)
    assert metric.numerical.margin == pytest.approx(10.0)
    assert metric.numerical.relative_degradation == pytest.approx(0.05)
    assert metric.verdict is Verdict.PASS
    metric, _ = compare(ramp(100), ramp(85), config=config)
    assert metric.verdict is Verdict.FAIL


def test_stricter_of_absolute_and_relative_applies() -> None:
    config = gate(max_absolute_degradation=2.0, max_relative_degradation=0.10)
    metric, _ = compare(ramp(100), ramp(97), config=config)
    assert metric.numerical.margin == pytest.approx(2.0)
    assert "stricter" in metric.numerical.margin_basis
    assert metric.verdict is Verdict.FAIL


def test_relative_only_margin_with_zero_baseline_is_invalid() -> None:
    zeros = [0.0] * N
    metric, result = compare(zeros, zeros, config=gate(max_relative_degradation=0.1))
    assert metric.verdict is Verdict.INVALID
    assert "undefined because the baseline is 0" in metric.reasons[-1]
    assert result.verdict is Verdict.INVALID
    assert result.exit_code == 3


def test_zero_baseline_falls_back_to_absolute_margin() -> None:
    zeros = [0.0] * N
    config = gate(max_absolute_degradation=0.1, max_relative_degradation=0.1)
    metric, _ = compare(zeros, zeros, direction="lower_is_better", config=config)
    assert metric.verdict is Verdict.PASS
    assert metric.numerical.relative_degradation is None
    assert "undefined because the baseline is 0" in metric.numerical.margin_basis


def test_zero_baseline_with_regression_on_lower_is_better() -> None:
    zeros = [0.0] * N
    ones = [1.0] * N
    metric, _ = compare(
        zeros, ones, direction="lower_is_better", config=gate(max_absolute_degradation=0.1)
    )
    assert metric.verdict is Verdict.FAIL


# --- statistical evidence ------------------------------------------------------------------


def test_noisy_small_shift_is_inconclusive_not_fail() -> None:
    base = ramp(10, spread=3)
    cand = [v - 0.6 for v in ramp(10, spread=3)][::-1]  # unpaired-looking noise
    metric, result = compare(
        base, cand, config=gate(max_absolute_degradation=0.5, pairing="unpaired")
    )
    assert metric.numerical.exceeds_margin is True
    assert metric.evidence.outcome is Evidence.INCONCLUSIVE
    assert metric.verdict is Verdict.INCONCLUSIVE
    assert result.exit_code == 3


def test_inconclusive_can_be_allowed_by_policy() -> None:
    config = make_config(
        {"m": {"max_absolute_degradation": 0.5, "pairing": "unpaired"}},
        policy={"on_inconclusive": "pass"},
    )
    base = ramp(10, spread=3)
    cand = [v - 0.6 for v in ramp(10, spread=3)][::-1]
    metric, result = compare(base, cand, config=config)
    assert metric.verdict is Verdict.INCONCLUSIVE
    assert metric.blocking is False
    assert result.verdict is Verdict.PASS


def test_insufficient_samples() -> None:
    metric, result = compare(ramp(10, n=5), ramp(5, n=5))
    assert metric.evidence.outcome is Evidence.INSUFFICIENT_SAMPLES
    assert metric.evidence.degradation_ci is None
    assert metric.numerical.exceeds_margin is True
    assert metric.verdict is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_min_samples_is_configurable() -> None:
    metric, _ = compare(
        ramp(10, n=5), ramp(5, n=5), config=gate(max_absolute_degradation=0.5, min_samples=3)
    )
    assert metric.verdict is Verdict.FAIL


def test_bootstrap_is_deterministic_and_seed_dependent() -> None:
    base, cand = ramp(10, 3), [v + 0.1 for v in ramp(10, 3)][::-1]
    first, _ = compare(base, cand)
    second, _ = compare(base, cand)
    assert first.evidence.degradation_ci == second.evidence.degradation_ci
    other = make_config({"m": {"max_absolute_degradation": 0.5}}, defaults={"bootstrap_seed": 99})
    third, _ = compare(base, cand, config=other)
    assert third.evidence.degradation_ci != first.evidence.degradation_ci


def test_bonferroni_widens_intervals() -> None:
    b = make_run({"a": ramp(10, 3), "b": ramp(10, 3)}, run_id="b")
    c = make_run({"a": ramp(10, 3)[::-1], "b": ramp(10, 3)[::-1]}, run_id="c")
    metrics = {"a": {"max_absolute_degradation": 1}, "b": {"max_absolute_degradation": 1}}
    plain = compare_runs(b, c, make_config(metrics)).metrics[0]
    corrected = compare_runs(
        b, c, make_config(metrics, policy={"correction": "bonferroni"})
    ).metrics[0]
    assert corrected.evidence.adjusted_confidence == pytest.approx(0.975)
    assert plain.evidence.degradation_ci is not None
    assert corrected.evidence.degradation_ci is not None
    plain_width = plain.evidence.degradation_ci[1] - plain.evidence.degradation_ci[0]
    corrected_width = corrected.evidence.degradation_ci[1] - corrected.evidence.degradation_ci[0]
    assert corrected_width > plain_width


def test_median_statistic() -> None:
    base = [1.0] * 30 + [100.0] * 10
    cand = [1.0] * 30 + [1000.0] * 10
    config = gate(max_absolute_degradation=0.5, statistic="median")
    metric, _ = compare(base, cand, direction="lower_is_better", config=config)
    assert metric.numerical.degradation == 0
    assert metric.verdict is Verdict.PASS


# --- pairing -------------------------------------------------------------------------------


def test_auto_pairs_when_trial_ids_match() -> None:
    metric, _ = compare(ramp(10), ramp(9.9))
    assert metric.pairing.mode == "paired"
    assert metric.pairing.n_pairs == N
    assert metric.evidence.method == "paired_bootstrap_expanded"


def test_pairing_reduces_uncertainty_for_shared_conditions() -> None:
    # Large between-trial variation that is shared by both runs, small consistent shift.
    base = [float(i) for i in range(N)]
    cand = [v - 0.2 for v in base]
    paired, _ = compare(base, cand, config=gate(max_absolute_degradation=0.5))
    unpaired, _ = compare(base, cand, config=gate(max_absolute_degradation=0.5, pairing="unpaired"))
    assert paired.verdict is Verdict.PASS
    assert unpaired.verdict is Verdict.INCONCLUSIVE


def test_auto_falls_back_to_unpaired_when_ids_differ() -> None:
    ids_b = [f"b{i}" for i in range(N)]
    ids_c = [f"c{i}" for i in range(N)]
    metric, _ = compare(ramp(10), ramp(10), base_ids=ids_b, cand_ids=ids_c)
    assert metric.pairing.mode == "unpaired"
    assert metric.pairing.note == (
        "the runs share no trial_id, so trials are compared as independent samples"
    )
    assert metric.evidence.method == "unpaired_bootstrap_expanded"


def test_auto_suggests_explicit_pairing_only_when_some_ids_are_shared() -> None:
    ids_b = [f"t{i}" for i in range(N)]
    ids_c = [f"t{i}" for i in range(5, N + 5)]
    # Each run lacks 5 of the other's ids; allow that explicitly.
    config = gate(max_absolute_degradation=1, max_missing_fraction=0.2)
    metric, _ = compare(ramp(10), ramp(10), base_ids=ids_b, cand_ids=ids_c, config=config)
    assert metric.pairing is not None
    assert metric.pairing.mode == "unpaired"
    assert "set pairing: paired to match on the shared ids" in (metric.pairing.note or "")


def test_unpaired_samples_may_have_different_sizes() -> None:
    # Distinct trial ids: independent samples, so neither run is missing anything.
    metric, _ = compare(
        ramp(10, n=40),
        ramp(10, n=25),
        cand_ids=[f"other{i}" for i in range(25)],
        config=gate(max_absolute_degradation=1),
    )
    assert metric.pairing.mode == "unpaired"
    assert (metric.baseline.n, metric.candidate.n) == (40, 25)
    assert metric.verdict is Verdict.PASS


def test_explicit_pairing_uses_intersection_and_reports_unmatched() -> None:
    ids_b = [f"t{i}" for i in range(N)]
    ids_c = [f"t{i}" for i in range(5, N + 5)]
    # Each run lacks 5 of the other's ids; allow that explicitly.
    config = gate(max_absolute_degradation=1, pairing="paired", max_missing_fraction=0.2)
    metric, _ = compare(ramp(10), ramp(10), base_ids=ids_b, cand_ids=ids_c, config=config)
    assert metric.pairing.mode == "paired"
    assert metric.pairing.n_pairs == N - 5
    assert metric.pairing.unmatched_baseline == ["t0", "t1", "t2", "t3", "t4"]
    assert len(metric.pairing.unmatched_candidate) == 5
    assert "no match" in (metric.pairing.note or "")


def test_explicit_pairing_without_overlap_is_invalid() -> None:
    config = gate(max_absolute_degradation=1, pairing="paired")
    metric, _ = compare(
        ramp(10),
        ramp(10),
        base_ids=[f"a{i}" for i in range(N)],
        cand_ids=[f"b{i}" for i in range(N)],
        config=config,
    )
    assert metric.verdict is Verdict.INVALID


# --- missing and invalid data --------------------------------------------------------------


def test_missing_values_are_invalid_by_default() -> None:
    cand: list[float | None] = list(ramp(10))
    cand[3] = None
    metric, result = compare(ramp(10), cand)
    assert metric.verdict is Verdict.INVALID
    assert metric.direction is not None  # identity is still reported
    assert "1 of 40 trials missing (1 null values" in metric.reasons[0]
    assert result.exit_code == 3


def test_missing_values_tolerated_when_configured() -> None:
    cand: list[float | None] = list(ramp(10))
    cand[3] = None
    metric, _ = compare(
        ramp(10), cand, config=gate(max_absolute_degradation=0.5, max_missing_fraction=0.1)
    )
    assert metric.verdict is Verdict.PASS
    assert metric.pairing.n_pairs == N - 1
    assert "dropped" in (metric.pairing.note or "")
    assert metric.candidate.n_missing == 1


def test_metric_missing_from_candidate_is_invalid() -> None:
    b = make_run({"m": ramp(1)}, run_id="b")
    c = make_run({"other": ramp(1)}, run_id="c")
    result = compare_runs(b, c, gate(max_absolute_degradation=1))
    assert result.metrics[0].verdict is Verdict.INVALID
    assert result.metrics[0].reasons == ["metric is not declared in the candidate run"]
    assert result.untracked_metrics == ["other"]


@pytest.mark.parametrize(
    ("b_kwargs", "c_kwargs", "fragment"),
    [
        (
            {"directions": {"m": "higher_is_better"}},
            {"directions": {"m": "lower_is_better"}},
            "direction differs",
        ),
        ({"units": {"m": "m"}}, {"units": {"m": "cm"}}, "unit differs"),
    ],
)
def test_incompatible_metric_definitions(
    b_kwargs: dict[str, Any], c_kwargs: dict[str, Any], fragment: str
) -> None:
    b = make_run({"m": ramp(1)}, run_id="b", **b_kwargs)
    c = make_run({"m": ramp(1)}, run_id="c", **c_kwargs)
    result = compare_runs(b, c, gate(max_absolute_degradation=1))
    assert fragment in result.metrics[0].reasons[0]


def test_config_direction_must_match_runs() -> None:
    metric, _ = compare(
        ramp(1), ramp(1), config=gate(max_absolute_degradation=1, direction="lower_is_better")
    )
    assert metric.verdict is Verdict.INVALID
    assert "config says lower_is_better" in metric.reasons[0]


def test_empty_runs_are_invalid() -> None:
    b = make_run({"m": []}, run_id="b")
    c = make_run({"m": []}, run_id="c")
    metric = compare_runs(b, c, gate(max_absolute_degradation=1)).metrics[0]
    assert metric.verdict is Verdict.INVALID
    assert "no trials" in metric.reasons[0]


def test_invalid_can_be_allowed_by_policy() -> None:
    b = make_run({"m": ramp(1), "ok": ramp(1)}, run_id="b")
    c = make_run({"x": ramp(1), "ok": ramp(1)}, run_id="c")
    metrics = {"m": {"max_absolute_degradation": 1}, "ok": {"max_absolute_degradation": 1}}
    result = compare_runs(b, c, make_config(metrics, policy={"on_invalid": "pass"}))
    assert result.verdict is Verdict.PASS
    assert result.metrics[0].verdict is Verdict.INVALID
    assert result.metrics[0].blocking is False


# --- failure rate, optional metrics, run-level checks --------------------------------------


def test_failure_rate_metric_gates_crashes() -> None:
    b = make_run({"m": ramp(1)}, run_id="b", statuses=["ok"] * N)
    c = make_run({"m": ramp(1)}, run_id="c", statuses=["ok"] * 20 + ["error"] * 20)
    config = make_config({FAILURE_RATE_METRIC: {"max_absolute_degradation": 0.05}})
    metric = compare_runs(b, c, config).metrics[0]
    assert metric.numerical is not None
    assert metric.numerical.candidate == pytest.approx(0.5)
    assert metric.verdict is Verdict.FAIL


def test_optional_metric_regression_does_not_block() -> None:
    config = gate(max_absolute_degradation=0.5, required=False)
    metric, result = compare(ramp(10), ramp(5), config=config)
    assert metric.verdict is Verdict.FAIL
    assert metric.blocking is False
    assert result.verdict is Verdict.PASS
    assert "no metric is required" in result.summary


def test_scenario_mismatch_is_invalid_by_default() -> None:
    b = make_run({"m": ramp(1)}, run_id="b", scenario="left_turn")
    c = make_run({"m": ramp(1)}, run_id="c", scenario="right_turn")
    result = compare_runs(b, c, gate(max_absolute_degradation=1))
    assert result.verdict is Verdict.INVALID
    assert "scenario differs" in result.run_blocking_issues[0]
    relaxed = make_config(
        {"m": {"max_absolute_degradation": 1}}, policy={"require_same_scenario": False}
    )
    result = compare_runs(b, c, relaxed)
    assert result.verdict is Verdict.PASS
    assert "scenario differs" in result.run_issues[0]


def test_run_level_notes() -> None:
    b = make_run({"m": ramp(1)}, run_id="same", simulator={"name": "sim", "version": "1"})
    c = make_run(
        {"m": ramp(1)}, run_id="same", simulator={"name": "sim", "version": "2"}, status="partial"
    )
    notes = compare_runs(b, c, gate(max_absolute_degradation=1)).run_issues
    assert any("simulator version differs" in n for n in notes)
    assert any("share run_id" in n for n in notes)
    assert any("candidate run status is 'partial'" in n for n in notes)


def test_several_metrics_get_individual_verdicts() -> None:
    b = make_run({"good": ramp(10), "bad": ramp(10), "few": ramp(10, n=N)}, run_id="b")
    c = make_run({"good": ramp(10), "bad": ramp(5), "few": [None] * N}, run_id="c")
    config = make_config(
        {
            "good": {"max_absolute_degradation": 0.5},
            "bad": {"max_absolute_degradation": 0.5},
            "few": {"max_absolute_degradation": 0.5},
        }
    )
    result = compare_runs(b, c, config)
    verdicts = {m.name: m.verdict for m in result.metrics}
    assert verdicts == {"good": Verdict.PASS, "bad": Verdict.FAIL, "few": Verdict.INVALID}
    assert result.verdict is Verdict.FAIL
    assert result.summary.startswith("regression detected in 1 of 3")


def test_comparison_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    b: RunResult = make_run({"m": ramp(10, 2)}, run_id="b")
    c: RunResult = make_run({"m": ramp(9.8, 2)[::-1]}, run_id="c")
    config = gate(max_absolute_degradation=0.5)
    first = compare_runs(b, c, config).model_dump_json()
    assert first == compare_runs(b, c, config).model_dump_json()
    assert '"generated_at":"2023-11-14T22:13:20Z"' in first


# --- binary metrics (regression tests for the zero-event false PASS) -----------------------


def test_zero_events_in_both_runs_do_not_prove_non_inferiority() -> None:
    # Before v0.1.1 the bootstrap gave a zero-width interval here and the gate said PASS,
    # although 0 of 30 is compatible with a true collision rate of about 11%.
    zeros = [0.0] * 30
    metric, result = compare(
        zeros, zeros, direction="lower_is_better", config=gate(max_absolute_degradation=0.02)
    )
    assert metric.evidence.method == "paired_newcombe"
    assert metric.evidence.degradation_ci is not None
    assert metric.evidence.degradation_ci[1] > 0.02
    assert metric.verdict is Verdict.INCONCLUSIVE
    assert result.exit_code == 3


def test_candidate_without_events_is_not_a_pass_with_few_trials() -> None:
    base = [1.0] + [0.0] * 29
    cand = [0.0] * 30
    config = gate(max_absolute_degradation=0.02, pairing="unpaired")
    metric, _ = compare(base, cand, direction="lower_is_better", config=config)
    assert metric.evidence.method == "unpaired_newcombe"
    assert metric.verdict is Verdict.INCONCLUSIVE


def test_binary_regression_is_still_detected() -> None:
    base = [0.0] * 200
    cand = [1.0] * 40 + [0.0] * 160
    metric, _ = compare(
        base, cand, direction="lower_is_better", config=gate(max_absolute_degradation=0.02)
    )
    assert metric.verdict is Verdict.FAIL
    assert metric.evidence.resamples is None
    assert metric.evidence.seed is None


def test_binary_higher_is_better_sign() -> None:
    base = [1.0] * 190 + [0.0] * 10
    cand = [1.0] * 150 + [0.0] * 50
    metric, _ = compare(base, cand, config=gate(max_absolute_degradation=0.05, pairing="unpaired"))
    assert metric.numerical.degradation == pytest.approx(0.2)
    ci = metric.evidence.degradation_ci
    assert ci is not None
    assert ci[0] > 0.05
    assert metric.verdict is Verdict.FAIL


def test_binary_median_keeps_the_bootstrap() -> None:
    values = [1.0, 0.0] * 20
    metric, _ = compare(
        values, values, config=gate(max_absolute_degradation=0.5, statistic="median")
    )
    assert metric.evidence.method == "paired_bootstrap_expanded"


def test_percentile_bootstrap_remains_available() -> None:
    values = [float(i % 7) for i in range(12)]
    base = make_run({"m": values}, run_id="b")
    cand = make_run(
        {"m": [v + 0.2 + 0.3 * ((3 * i) % 5 - 2) for i, v in enumerate(values)]}, run_id="c"
    )
    gate = {"m": {"max_absolute_degradation": 1.0}}
    expanded = compare_runs(base, cand, make_config(gate)).metrics[0]
    plain = compare_runs(
        base, cand, make_config(gate, defaults={"bootstrap_method": "percentile"})
    ).metrics[0]
    assert plain.evidence.method == "paired_bootstrap"
    assert expanded.evidence.method == "paired_bootstrap_expanded"
    assert plain.evidence.degradation_ci is not None
    assert expanded.evidence.degradation_ci is not None
    # The expanded interval is wider at small n.
    plain_width = plain.evidence.degradation_ci[1] - plain.evidence.degradation_ci[0]
    expanded_width = expanded.evidence.degradation_ci[1] - expanded.evidence.degradation_ci[0]
    assert expanded_width > plain_width
