"""Comparisons over clusters of trials (cluster_by), and combining several run files."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from physim_gate.cli import app
from physim_gate.combine import RUN_LABEL, cluster_names, combine_runs
from physim_gate.comparison import Evidence, MetricComparison, Verdict
from physim_gate.config import GateConfig
from physim_gate.engine import compare_runs
from physim_gate.reporting import render_html, render_markdown
from physim_gate.reporting.format import count
from physim_gate.schema import InputError, RunResult, SchemaError, dump_json
from physim_gate.stats import _t_quantile, t_interval_difference
from tests.conftest import make_config, make_run

runner = CliRunner()


def clustered(
    groups: Mapping[str, Sequence[float]],
    *,
    run_id: str = "run",
    label: str = "seed",
    direction: str = "higher_is_better",
) -> RunResult:
    """A run whose trials carry ``label``: one group of values per cluster."""
    return RunResult.model_validate(
        {
            "run_id": run_id,
            "simulator": {"name": "unit-test"},
            "scenario": {"id": "scenario"},
            "metrics": {"m": {"direction": direction}},
            "trials": [
                {
                    "trial_id": f"{run_id}-{cluster}-{i}",
                    "metrics": {"m": v},
                    "labels": {label: cluster},
                }
                for cluster, values in groups.items()
                for i, v in enumerate(values)
            ],
        }
    )


def gate(**metric: Any) -> GateConfig:
    return make_config({"m": {"max_absolute_degradation": 0.5, "cluster_by": "seed", **metric}})


def only(baseline: RunResult, candidate: RunResult, config: GateConfig) -> MetricComparison:
    return compare_runs(baseline, candidate, config).metrics[0]


BASE = {
    "s0": [10.0, 10.2, 9.8],
    "s1": [10.5, 10.4, 10.6],
    "s2": [9.6, 9.5, 9.7],
    "s3": [10.1, 10.0],
}


def test_unpaired_clusters_use_welch_t_over_cluster_means() -> None:
    cand = {"s4": [9.0, 9.2], "s5": [9.9, 9.7], "s6": [9.4, 9.5, 9.3]}
    metric = only(clustered(BASE, run_id="b"), clustered(cand, run_id="c"), gate())
    b_means = [statistics.mean(v) for v in BASE.values()]
    c_means = [statistics.mean(v) for v in cand.values()]
    expected = t_interval_difference(b_means, c_means, paired=False, confidence=0.95)
    assert metric.evidence.method == "unpaired_cluster_t"
    # higher is better, so the degradation is baseline - candidate
    assert metric.evidence.degradation_ci == pytest.approx((-expected.high, -expected.low))
    assert metric.numerical is not None
    assert metric.numerical.baseline == pytest.approx(statistics.mean(b_means))
    assert metric.clusters is not None
    assert metric.clusters.mode == "unpaired"
    assert metric.clusters.baseline_trials == {"s0": 3, "s1": 3, "s2": 3, "s3": 2}
    assert metric.pairing is None
    assert metric.reasons[0].startswith("compared by seed: the means of 4 baseline and 3 candidate")


def test_clusters_with_the_same_names_are_paired() -> None:
    shifts = {"s0": 0.1, "s1": 0.3, "s2": -0.1, "s3": 0.2}
    cand = {k: [v - shifts[k] for v in values] for k, values in BASE.items()}
    metric = only(clustered(BASE, run_id="b"), clustered(cand, run_id="c"), gate())
    assert metric.evidence.method == "paired_cluster_t"
    assert metric.clusters is not None
    assert metric.clusters.mode == "paired"
    assert count(metric) == "4 clusters"
    b_means = list(metric.clusters.baseline_means.values())
    c_means = list(metric.clusters.candidate_means.values())
    expected = t_interval_difference(b_means, c_means, paired=True, confidence=0.95)
    assert metric.evidence.degradation_ci == pytest.approx((-expected.high, -expected.low))


def test_cluster_means_that_do_not_vary_fall_back_to_the_trials() -> None:
    # Every cluster lost exactly 0.1 (up to rounding): the spread of the differences is no
    # evidence of certainty.
    cand = {k: [v - 0.1 for v in values] for k, values in BASE.items()}
    metric = only(clustered(BASE, run_id="b"), clustered(cand, run_id="c"), gate())
    assert any("cluster means do not vary" in r for r in metric.reasons)
    assert metric.evidence.method == "unpaired_bootstrap_expanded"
    assert metric.evidence.degradation_ci is not None
    low, high = metric.evidence.degradation_ci
    assert high - low > 0.1


def test_paired_clusters_report_unmatched_ones() -> None:
    cand = {"s0": [9.9], "s1": [10.3], "s2": [9.6], "s9": [1.0]}
    config = gate(pairing="paired")
    metric = only(clustered(BASE, run_id="b"), clustered(cand, run_id="c"), config)
    assert metric.clusters is not None
    assert metric.clusters.unmatched_baseline == ["s3"]
    assert metric.clusters.unmatched_candidate == ["s9"]
    assert list(metric.clusters.baseline_means) == ["s0", "s1", "s2"]
    assert "1 baseline and 1 candidate clusters have no match" in (metric.clusters.note or "")


def test_too_few_clusters_is_inconclusive_not_a_trial_level_verdict() -> None:
    # Two clusters of 500 identical trials each would give a confident trial-level answer.
    base = {"s0": [10.0] * 500, "s1": [10.0] * 500}
    cand = {"s2": [5.0] * 500, "s3": [5.0] * 500}
    metric = only(clustered(base, run_id="b"), clustered(cand, run_id="c"), gate())
    assert metric.verdict is Verdict.INCONCLUSIVE
    assert metric.evidence.outcome is Evidence.INSUFFICIENT_SAMPLES
    assert (
        "insufficient clusters: 2 clusters in the smaller run (seed), min_clusters is 3"
        in (metric.reasons[-1])
    )
    two = only(clustered(base, run_id="b"), clustered(cand, run_id="c"), gate(min_clusters=2))
    assert two.verdict is Verdict.FAIL
    assert any("cluster means do not vary" in r for r in two.reasons)


def test_between_cluster_spread_widens_the_interval() -> None:
    """The point of clustering: trials within a seed are not independent evidence."""
    base = {f"s{i}": [10.0 + shift] * 50 for i, shift in enumerate((-1.0, 0.0, 1.0, 0.5))}
    cand = {f"c{i}": [9.6 + shift] * 50 for i, shift in enumerate((-1.0, 0.0, 1.0, 0.5))}
    b, c = clustered(base, run_id="b"), clustered(cand, run_id="c")
    pooled = only(b, c, make_config({"m": {"max_absolute_degradation": 0.1}}))
    clustered_metric = only(b, c, gate(max_absolute_degradation=0.1))
    assert pooled.verdict is Verdict.FAIL
    assert clustered_metric.verdict is Verdict.INCONCLUSIVE
    assert any("more clusters would narrow" in r for r in clustered_metric.reasons)


def test_trials_without_the_cluster_label_make_the_metric_invalid() -> None:
    plain = make_run({"m": [1.0, 2.0, 3.0]}, run_id="b")
    metric = only(plain, clustered(BASE, run_id="c"), gate())
    assert metric.verdict is Verdict.INVALID
    assert "3 of 3 baseline trials have no such label (for example 't0')" in metric.reasons[0]


def test_lower_is_better_and_relative_margins_use_the_cluster_means() -> None:
    base = {"s0": [1.0, 1.2], "s1": [1.1, 0.9], "s2": [1.05, 1.0]}
    cand = {"s3": [1.5, 1.6], "s4": [1.45, 1.55], "s5": [1.6, 1.5]}
    config = make_config({"m": {"max_relative_degradation": 0.1, "cluster_by": "seed"}})
    metric = only(
        clustered(base, run_id="b", direction="lower_is_better"),
        clustered(cand, run_id="c", direction="lower_is_better"),
        config,
    )
    assert metric.numerical is not None
    assert metric.numerical.margin == pytest.approx(0.1 * metric.numerical.baseline)
    assert metric.evidence.degradation_ci is not None
    assert metric.evidence.degradation_ci[0] > 0
    assert metric.verdict is Verdict.FAIL


def test_cluster_by_needs_the_mean() -> None:
    with pytest.raises(ValueError, match="cluster_by compares the mean"):
        make_config(
            {"m": {"max_absolute_degradation": 1, "cluster_by": "seed", "statistic": "median"}}
        )


def test_default_cluster_by_does_not_override_the_config() -> None:
    config = make_config({"m": {"max_absolute_degradation": 1}})
    assert config.with_default_cluster_by(RUN_LABEL).resolve("m").cluster_by == RUN_LABEL
    own = make_config({"m": {"max_absolute_degradation": 1}}, defaults={"cluster_by": "seed"})
    assert own.with_default_cluster_by(RUN_LABEL).resolve("m").cluster_by == "seed"


def test_t_quantile_handles_few_degrees_of_freedom_at_strict_levels() -> None:
    # t[0.99999, 1] = tan(pi * (0.99999 - 0.5))
    import math

    assert _t_quantile(0.99999, 1) == pytest.approx(math.tan(math.pi * 0.49999), rel=1e-9)


# combining run files


def test_cluster_names_prefer_run_ids_then_shortened_paths() -> None:
    runs = [make_run({"m": [1.0]}, run_id=r) for r in ("a", "b")]
    assert cluster_names([Path("x.json"), Path("y.json")], runs) == ["a", "b"]
    same = [make_run({"m": [1.0]}, run_id="eval") for _ in range(2)]
    paths = [Path("runs/clean-s0/eval_info.json"), Path("runs/clean-s1/eval_info.json")]
    assert cluster_names(paths, same) == ["clean-s0", "clean-s1"]
    flat = [
        Path("clean-s0__eval-10000__eval_info.json"),
        Path("clean-s1__eval-10000__eval_info.json"),
    ]
    assert cluster_names(flat, same) == ["clean-s0", "clean-s1"]
    with pytest.raises(InputError, match="the same file is given twice"):
        cluster_names([Path("a.json"), Path("a.json")], same)


def test_combine_labels_and_prefixes_every_trial() -> None:
    runs = [make_run({"m": [1.0, 2.0]}, run_id=r) for r in ("s0", "s1")]
    combined = combine_runs("baseline", ["s0", "s1"], runs)
    assert [t.trial_id for t in combined.trials] == ["s0/t0", "s0/t1", "s1/t0", "s1/t1"]
    assert {t.labels[RUN_LABEL] for t in combined.trials} == {"s0", "s1"}
    assert combined.run_id == "s0+s1"


def test_combine_rejects_runs_of_different_scenarios() -> None:
    runs = [
        make_run({"m": [1.0]}, run_id="a"),
        make_run({"m": [1.0]}, run_id="b", scenario="other"),
    ]
    with pytest.raises(InputError, match="the baseline runs differ in scenario"):
        combine_runs("baseline", ["a", "b"], runs)


def test_combine_rejects_a_reserved_label() -> None:
    run = make_run({"m": [1.0]}, run_id="a")
    labeled = run.model_copy(
        update={"trials": [run.trials[0].model_copy(update={"labels": {RUN_LABEL: "x"}})]}
    )
    with pytest.raises(InputError, match="already has the label"):
        combine_runs("candidate", ["a", "b"], [labeled, run])


def test_combine_rejects_conflicting_metric_declarations() -> None:
    a = make_run({"m": [1.0]}, run_id="a")
    b = make_run({"m": [1.0]}, run_id="b", directions={"m": "lower_is_better"})
    with pytest.raises(SchemaError, match="declare metric 'm' differently"):
        combine_runs("baseline", ["a", "b"], [a, b])


# command line


def write_runs(tmp_path: Path, side: str, means: Sequence[float]) -> list[str]:
    paths = []
    for i, center in enumerate(means):
        run = make_run({"m": [center - 0.1, center, center + 0.1] * 10}, run_id=f"{side}-s{i}")
        path = tmp_path / f"{side}-s{i}.json"
        path.write_text(dump_json(run))
        paths.append(str(path))
    return paths


def test_compare_several_runs_per_side_clusters_by_run(tmp_path: Path) -> None:
    base = write_runs(tmp_path, "base", [10.0, 10.4, 9.7, 10.1])
    cand = write_runs(tmp_path, "cand", [9.0, 9.3, 8.8])
    config = tmp_path / "gate.yaml"
    config.write_text("version: 1\nmetrics:\n  m:\n    max_absolute_degradation: 0.1\n")
    args = ["compare", "-g", str(config), "-f", "json"]
    for path in base:
        args += ["-b", path]
    for path in cand:
        args += ["-c", path]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    metric = data["metrics"][0]
    assert data["schema_version"] == "1.1"
    assert data["config"]["defaults"]["cluster_by"] == RUN_LABEL
    assert metric["evidence"]["method"] == "unpaired_cluster_t"
    assert list(metric["clusters"]["baseline_means"]) == [
        "base-s0",
        "base-s1",
        "base-s2",
        "base-s3",
    ]
    assert [ref["path"] for ref in data["inputs"]["baseline_runs"]] == base
    assert data["inputs"]["baseline"]["path"] is None
    assert len(data["inputs"]["baseline"]["sha256"]) == 64


def test_compare_one_run_per_side_is_unchanged(tmp_path: Path) -> None:
    base = write_runs(tmp_path, "base", [10.0])
    cand = write_runs(tmp_path, "cand", [9.0])
    config = tmp_path / "gate.yaml"
    config.write_text("version: 1\nmetrics:\n  m:\n    max_absolute_degradation: 0.1\n")
    result = runner.invoke(
        app, ["compare", "-b", *base, "-c", *cand, "-g", str(config), "-f", "json"]
    )
    data = json.loads(result.output)
    assert data["config"]["defaults"]["cluster_by"] is None
    assert data["metrics"][0]["clusters"] is None
    assert data["inputs"]["baseline_runs"] == []


def test_a_stored_baseline_cannot_face_several_candidates(tmp_path: Path) -> None:
    cand = write_runs(tmp_path, "cand", [9.0, 9.1])
    config = tmp_path / "gate.yaml"
    config.write_text("version: 1\nmetrics:\n  m:\n    max_absolute_degradation: 0.1\n")
    args = ["compare", "-b", "@main", "-c", cand[0], "-c", cand[1], "-g", str(config)]
    result = runner.invoke(app, [*args, "--store", str(tmp_path / "store")])
    assert result.exit_code == 2
    assert "holds one run per baseline" in result.output


def test_a_file_on_both_sides_is_an_error(tmp_path: Path) -> None:
    base = write_runs(tmp_path, "base", [10.0, 10.1])
    config = tmp_path / "gate.yaml"
    config.write_text("version: 1\nmetrics:\n  m:\n    max_absolute_degradation: 0.1\n")
    args = ["compare", "-b", base[0], "-b", base[1], "-c", base[1], "-g", str(config)]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "baseline and candidate are the same file" in result.output


# reports


def test_reports_show_runs_and_their_means(tmp_path: Path) -> None:
    base = write_runs(tmp_path, "base", [10.0, 10.4, 9.7])
    cand = write_runs(tmp_path, "cand", [9.0, 9.3, 8.8])
    config = tmp_path / "gate.yaml"
    config.write_text("version: 1\nmetrics:\n  m:\n    max_absolute_degradation: 0.1\n")
    out = tmp_path / "comparison.json"
    args = ["compare", "-g", str(config), "-o", str(out)]
    for b, c in zip(base, cand, strict=True):
        args += ["-b", b, "-c", c]
    runner.invoke(app, args)
    from physim_gate.comparison import load_comparison

    result = load_comparison(out)
    markdown = render_markdown(result)
    assert "| 3/3\u00a0runs |" in markdown
    assert "or the runs or clusters whose means were compared" in markdown
    html = render_html(result)
    assert "Means of the runs compared (3 baseline, 3 candidate)" in html
    assert "baseline 1/3" in html
    assert "3 vs 3 runs" in html
