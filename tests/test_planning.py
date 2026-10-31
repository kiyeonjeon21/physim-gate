from __future__ import annotations

import itertools
import json
import random
from math import comb
from pathlib import Path

import pytest
from typer.testing import CliRunner

from physim_gate.adapters.base import get_adapter
from physim_gate.cli import app
from physim_gate.combine import combine_runs
from physim_gate.comparison import ComparisonResult, Verdict
from physim_gate.config import GateConfig, load_config
from physim_gate.engine import compare_runs
from physim_gate.planning import (
    Goal,
    PairCells,
    PlanningError,
    _binomial,
    binary_verdict_probabilities,
    paired_binary_verdict_probabilities,
    plan_binary,
    plan_clusters,
    plan_continuous,
    plan_from_comparison,
)
from physim_gate.schema import Direction, RunResult, dump_json
from physim_gate.stats import newcombe_counts
from tests.conftest import make_run


def brute_force(n: int, qb: float, qc: float, margin: float) -> tuple[float, float]:
    p_pass = p_fail = 0.0
    for xb, xc in itertools.product(range(n + 1), repeat=2):
        weight = (
            comb(n, xb)
            * qb**xb
            * (1 - qb) ** (n - xb)
            * comb(n, xc)
            * qc**xc
            * (1 - qc) ** (n - xc)
        )
        low, high = newcombe_counts(xb, n, xc, n, 0.95)
        if high <= margin:
            p_pass += weight
        elif low > margin:
            p_fail += weight
    return p_pass, p_fail


@pytest.mark.parametrize("n", [10, 23])
def test_exact_probabilities_match_brute_force(n: int) -> None:
    for qb, qc, margin in itertools.product(
        [0.0, 0.03, 0.2, 0.5, 0.9], [0.0, 0.1, 0.45, 1.0], [0.0, 0.05, 0.2]
    ):
        result = binary_verdict_probabilities(n, qb, qc, margin, 0.95)
        expected = brute_force(n, qb, qc, margin)
        assert result.pass_ == pytest.approx(expected[0], abs=1e-12)
        assert result.fail == pytest.approx(expected[1], abs=1e-12)
        assert result.pass_ + result.fail + result.inconclusive == pytest.approx(1.0)


def test_binary_plan_matches_the_engine() -> None:
    """At the planned size, the real engine PASSes about as often as the plan says."""
    plan = plan_binary(0.7, 0.15)
    assert plan.goal is Goal.PASS
    assert plan.trials is not None
    assert plan.probabilities is not None
    n = plan.trials
    config = GateConfig.model_validate(
        {
            "version": 1,
            "defaults": {"pairing": "unpaired"},
            "metrics": {"success": {"max_absolute_degradation": 0.15}},
        }
    )
    rng = random.Random(7)
    reps = 400
    passes = 0
    for _ in range(reps):
        runs = [
            make_run(
                {"success": [1.0 if rng.random() < 0.7 else 0.0 for _ in range(n)]},
                run_id=name,
                directions={"success": "higher_is_better"},
            )
            for name in ("base", "cand")
        ]
        result = compare_runs(runs[0], runs[1], config)
        passes += result.metrics[0].verdict.value == "PASS"
    # 400 repetitions: the standard error of the observed rate is about 0.02.
    assert passes / reps == pytest.approx(plan.probabilities.pass_, abs=0.07)


def test_binary_plans() -> None:
    plan = plan_binary(0.9, 0.05)
    assert plan.trials == 573
    assert plan.probabilities is not None
    assert plan.probabilities.pass_ >= 0.8
    smaller = binary_verdict_probabilities(572, 0.1, 0.1, 0.05, 0.95)
    assert smaller.pass_ < 0.8

    regression = plan_binary(0.02, 0.01, direction=Direction.LOWER_IS_BETTER, true_degradation=0.03)
    assert regression.goal is Goal.FAIL
    assert regression.probabilities is not None
    assert regression.probabilities.fail >= 0.8

    # Higher-is-better and lower-is-better describe the same question from both sides.
    mirrored = plan_binary(0.1, 0.05, direction=Direction.LOWER_IS_BETTER)
    assert mirrored.trials == plan.trials

    assert plan_binary(0.5, 0.6, min_samples=10).trials == 10
    assert plan_binary(0.5, 0.001, max_trials=1000).trials is None


def test_continuous_plans() -> None:
    unpaired = plan_continuous(2.0, 0.5)
    paired = plan_continuous(2.0, 0.5, paired=True)
    assert unpaired.trials == 252
    assert paired.trials == 126
    assert unpaired.probabilities is not None
    assert unpaired.probabilities.pass_ == pytest.approx(0.8, abs=0.01)
    regression = plan_continuous(1.0, 0.1, true_degradation=0.6)
    assert regression.goal is Goal.FAIL
    assert regression.trials == 63
    assert plan_continuous(1.0, 0.0001, max_trials=1000).trials is None
    assert any("below 30 trials" in note for note in plan_continuous(1.0, 2.0).notes)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: plan_binary(0.9, 0.05, true_degradation=0.05), "equals the margin"),
        (lambda: plan_binary(0.9, -0.1), "margin must be >= 0"),
        (lambda: plan_binary(1.2, 0.1), "rate must be between 0 and 1"),
        (lambda: plan_binary(0.05, 0.1, true_degradation=0.2), "leaves"),
        (lambda: plan_binary(0.5, 0.1, confidence=0.4), "confidence"),
        (lambda: plan_binary(0.5, 0.1, power=1.0), "power"),
        (lambda: plan_continuous(0.0, 0.1), "stdev must be > 0"),
    ],
)
def test_planning_errors(call: object, message: str) -> None:
    with pytest.raises(PlanningError, match=message):
        call()  # type: ignore[operator]


def test_plan_cli() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plan", "--rate", "0.9", "--margin", "0.05"])
    assert result.exit_code == 0, result.output
    assert "573 trials per run" in result.output
    assert "PASS 80.1%" in result.output

    result = runner.invoke(app, ["plan", "-s", "2", "-m", "0.5", "--paired", "-f", "json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["trials"] == 126
    assert document["goal"] == "pass"
    assert set(document["probabilities"]) == {"pass", "inconclusive", "fail"}

    result = runner.invoke(app, ["plan", "-r", "0.5", "-m", "0.001", "--min-samples", "10"])
    assert result.exit_code == 0, result.output

    for args, message in [
        (["plan", "-m", "0.1"], "exactly one of --rate"),
        (["plan", "-m", "0.1", "-r", "0.5", "-s", "1"], "exactly one of --rate"),
        (["plan", "-m", "0.1", "-r", "0.5", "-t", "0.1"], "equals the margin"),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 2
        assert message in result.output


# Paired 0/1 planning and planning from a pilot comparison.

PUSHT = Path(__file__).parent.parent / "examples" / "lerobot-pusht"


def test_binomial_sampler_has_the_right_mean_and_variance() -> None:
    rng = random.Random(1)
    for n, p in [(50, 0.02), (400, 0.3), (400, 0.85), (10, 0.0), (10, 1.0)]:
        draws = [_binomial(rng, n, p) for _ in range(4000)]
        mean = sum(draws) / len(draws)
        variance = sum((d - mean) ** 2 for d in draws) / len(draws)
        assert mean == pytest.approx(n * p, abs=0.05 * n * max(p, 1 - p) ** 0.5 + 1e-9)
        assert variance == pytest.approx(n * p * (1 - p), rel=0.1, abs=1e-9)


def test_pair_cells() -> None:
    cells = PairCells.from_discordance(0.1, 0.08, 0.02)
    assert (cells.base_only, cells.cand_only) == pytest.approx((0.03, 0.05))
    assert cells.both == pytest.approx(0.07)
    assert cells.degradation == pytest.approx(0.02)
    assert cells.with_degradation(0.0).degradation == pytest.approx(0.0)
    with pytest.raises(PlanningError, match=r"needs a discordance of at least 0\.1"):
        PairCells.from_discordance(0.1, 0.05, 0.1)
    with pytest.raises(PlanningError, match="impossible"):
        PairCells.from_discordance(0.01, 0.2, 0.0)
    with pytest.raises(PlanningError, match=r"discordance must be in \(0, 1\]"):
        PairCells.from_discordance(0.1, 0.0, 0.0)


def test_paired_simulation_matches_the_engine() -> None:
    """At a fixed size, the paired simulation and the real engine PASS about equally often."""
    cells = PairCells.from_discordance(0.1, 0.08, 0.0)
    n = 150
    predicted = paired_binary_verdict_probabilities(n, cells, 0.05, 0.95)
    config = GateConfig.model_validate(
        {"version": 1, "metrics": {"success": {"max_absolute_degradation": 0.05}}}
    )
    rng = random.Random(5)
    reps, passes = 600, 0
    thresholds = (cells.both, cells.both + cells.base_only, 1 - cells.neither)
    for _ in range(reps):
        base, cand = [], []
        for _ in range(n):
            u = rng.random()
            bad_b, bad_c = (
                (1, 1) if u < thresholds[0]
                else (1, 0) if u < thresholds[1]
                else (0, 1) if u < thresholds[2]
                else (0, 0)
            )  # fmt: skip
            base.append(1.0 - bad_b)
            cand.append(1.0 - bad_c)
        runs = [
            make_run({"success": values}, run_id=name, directions={"success": "higher_is_better"})
            for name, values in (("b", base), ("c", cand))
        ]
        passes += compare_runs(runs[0], runs[1], config).metrics[0].verdict.value == "PASS"
    # Standard errors: about 0.02 for the engine estimate, 0.008 for the simulation.
    assert passes / reps == pytest.approx(predicted.pass_, abs=0.07)


def test_pairing_reduces_trials_when_runs_agree() -> None:
    unpaired = plan_binary(0.9, 0.05)
    paired = plan_binary(0.9, 0.05, discordance=0.04)
    anti = plan_binary(0.9, 0.05, discordance=0.2)
    assert paired.trials is not None
    assert unpaired.trials is not None
    assert anti.trials is not None
    assert paired.trials < unpaired.trials / 2
    # More disagreement than independent runs (2 * 0.1 * 0.9 = 0.18) means negative correlation.
    assert anti.trials > unpaired.trials
    assert "independent runs would give 0.18" in " ".join(paired.notes)


def _pusht_comparison() -> ComparisonResult:
    adapter = get_adapter("lerobot")
    options = {"start_seed": "1000"}
    return compare_runs(
        adapter.load(PUSHT / "baseline", options),
        adapter.load(PUSHT / "candidate-n-action-steps-8", options),
        load_config(PUSHT / "gate.yaml"),
    )


def test_plan_from_a_pilot_comparison() -> None:
    comparison = _pusht_comparison()
    plans = {item.metric: item for item in plan_from_comparison(comparison)}
    success = plans["success"].plan
    assert success is not None
    assert success.goal is Goal.FAIL
    assert success.true_degradation == pytest.approx(0.10)
    assert success.method.startswith("simulated")
    assert success.trials is not None
    assert 250 < success.trials < 400
    max_reward = plans["max_reward"].plan
    assert max_reward is not None
    assert max_reward.method == "normal approximation (paired)"
    assert max_reward.trials is not None
    assert max_reward.trials < success.trials

    only = plan_from_comparison(comparison, metrics=["success"], true_degradation=0.03)
    assert [item.metric for item in only] == ["success"]
    assert only[0].plan is not None
    assert only[0].plan.goal is Goal.PASS
    with pytest.raises(PlanningError, match="no metric nope"):
        plan_from_comparison(comparison, metrics=["nope"])


def test_plan_from_pilot_explains_unplannable_metrics() -> None:
    values = [1.0] * 20
    base = make_run({"ok": values, "gone": values, "flat": [2.0] * 20}, run_id="b")
    cand = make_run({"ok": values, "gone": [None] * 20, "flat": [2.0] * 20}, run_id="c")
    config = GateConfig.model_validate(
        {
            "version": 1,
            "metrics": {
                "ok": {"max_absolute_degradation": 0.05},
                "gone": {"max_absolute_degradation": 0.05},
                "flat": {"max_absolute_degradation": 0.05},
            },
        }
    )
    plans = {i.metric: i for i in plan_from_comparison(compare_runs(base, cand, config))}
    assert plans["ok"].reason is not None
    assert "every pair agreed" in plans["ok"].reason
    assert plans["gone"].reason == "INVALID: the pilot has no usable values"
    assert plans["flat"].reason is not None
    assert "do not vary" in plans["flat"].reason


def test_plan_cli_with_pilot_and_pairing(tmp_path: Path) -> None:
    runner = CliRunner()
    pilot = tmp_path / "comparison.json"
    pilot.write_text(dump_json(_pusht_comparison()))
    result = runner.invoke(app, ["plan", "--from", str(pilot)], env={"COLUMNS": "80"})
    assert result.exit_code == 0, result.output
    assert "success" in result.output
    assert "trials per run" in result.output
    assert not [line for line in result.output.splitlines() if line.endswith(" ")]
    as_json = json.loads(runner.invoke(app, ["plan", "--from", str(pilot), "-f", "json"]).stdout)
    assert [item["metric"] for item in as_json] == ["success", "max_reward", "sum_reward"]
    assert as_json[0]["plan"]["goal"] == "fail"

    paired = runner.invoke(app, ["plan", "-r", "0.9", "-m", "0.05", "--discordance", "0.04"])
    assert paired.exit_code == 0, paired.output
    assert "simulated" in paired.output

    for args, message in [
        (["plan", "--from", str(pilot), "-m", "0.1"], "--margin cannot be combined with --from"),
        (["plan", "-r", "0.9", "-m", "0.05", "--metric", "x"], "--metric applies to --from"),
        (["plan", "-r", "0.9"], "give --margin"),
        (["plan", "-r", "0.9", "-m", "0.05", "--paired"], "paired 0/1 runs need --discordance"),
        (["plan", "-s", "1", "-m", "0.5", "--discordance", "0.1"], "--discordance applies to"),
        (["plan", "--from", str(tmp_path / "missing.json")], "cannot read file"),
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 2, (args, result.output)
        assert message in result.output, (args, result.output)


@pytest.mark.parametrize("pairing", ["paired", "unpaired"])
def test_plan_from_an_underpowered_pilot_keeps_method_and_confidence(pairing: str) -> None:
    """A pilot too small to decide is when planning is needed most.

    Its plan must use the 0/1 method and the gate's confidence (after correction), exactly like
    a pilot with the same proportions and enough trials.
    """
    base = [1.0] * 6 + [0.0, 1.0]
    cand = [1.0] * 6 + [1.0, 0.0]
    config = GateConfig.model_validate(
        {
            "version": 1,
            "defaults": {"confidence": 0.99, "pairing": pairing},
            "policy": {"correction": "bonferroni"},
            "metrics": {
                "success": {"max_absolute_degradation": 0.2},
                "other": {"max_absolute_degradation": 1.0},
            },
        }
    )

    def plan(repeat: int) -> object:
        pilot = compare_runs(
            make_run({"success": base * repeat, "other": base * repeat}, run_id="b",
                     directions={"success": "higher_is_better"}),
            make_run({"success": cand * repeat, "other": cand * repeat}, run_id="c",
                     directions={"success": "higher_is_better"}),
            config,
        )  # fmt: skip
        verdict = pilot.metrics[0].evidence.outcome.value
        [item] = plan_from_comparison(pilot, metrics=["success"])
        assert item.plan is not None, item.reason
        return verdict, item.plan

    small_verdict, small = plan(1)
    large_verdict, large = plan(4)
    assert small_verdict == "insufficient_samples"
    assert large_verdict != "insufficient_samples"
    assert small.confidence == large.confidence == pytest.approx(1 - 0.01 / 2)  # type: ignore[attr-defined]
    assert small.method == large.method  # type: ignore[attr-defined]
    assert "normal approximation" not in small.method  # type: ignore[attr-defined]
    assert small.trials == large.trials  # type: ignore[attr-defined]


def test_cluster_plan_matches_the_engine() -> None:
    """The planned number of clusters reaches the goal about as often as planned."""
    plan = plan_clusters(1.0, 0.5, true_degradation=1.6, min_clusters=3)
    assert plan.unit == "clusters"
    k = plan.trials
    assert k is not None
    assert plan.probabilities is not None
    rng = random.Random(7)
    config = GateConfig.model_validate(
        {"version": 1, "metrics": {"m": {"max_absolute_degradation": 0.5, "cluster_by": "seed"}}}
    )
    fails = 0
    reps = 400
    for rep in range(reps):
        runs = []
        for side, center in (("b", 10.0), ("c", 10.0 - 1.6)):
            trials = [
                {
                    "trial_id": f"{side}{i}",
                    "metrics": {"m": rng.gauss(center, 1.0)},
                    "labels": {"seed": f"{side}{i}"},
                }
                for i in range(k)
            ]
            runs.append(
                RunResult.model_validate(
                    {
                        "run_id": f"{side}{rep}",
                        "simulator": {"name": "plan-check"},
                        "scenario": {"id": "s"},
                        "metrics": {"m": {"direction": "higher_is_better"}},
                        "trials": trials,
                    }
                )
            )
        if compare_runs(runs[0], runs[1], config).metrics[0].verdict is Verdict.FAIL:
            fails += 1
    assert fails / reps == pytest.approx(plan.probabilities.fail, abs=0.06)


def test_plan_from_a_clustered_pilot_counts_runs(tmp_path: Path) -> None:
    def run(run_id: str, center: float) -> RunResult:
        return make_run({"m": [center - 0.1, center, center + 0.1] * 5}, run_id=run_id)

    base = combine_runs(
        "baseline", ["s0", "s1", "s2"], [run(f"b{i}", 10 + i / 5) for i in range(3)]
    )
    cand = combine_runs(
        "candidate", ["t0", "t1", "t2"], [run(f"c{i}", 9.8 + i / 3) for i in range(3)]
    )
    config = GateConfig.model_validate(
        {"version": 1, "metrics": {"m": {"max_absolute_degradation": 0.5}}}
    ).with_default_cluster_by("physim.run")
    (item,) = plan_from_comparison(compare_runs(base, cand, config))
    assert item.plan is not None
    assert item.plan.unit == "runs"
    assert item.plan.kind == "clusters"
