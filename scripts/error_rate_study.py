"""Monte Carlo study of the gate's decision error rates, using the real comparison engine.

For each scenario the true degradation is known, so the share of PASS, FAIL and INCONCLUSIVE
verdicts measures how often the gate is wrong. At the boundary (true degradation exactly equal
to the margin) both PASS and FAIL are errors, and each should occur in at most about
(1 - confidence) / 2 = 2.5% of runs.

    python scripts/error_rate_study.py            # 400 repetitions per scenario (minutes)
    python scripts/error_rate_study.py --reps 1000
    python scripts/error_rate_study.py --method percentile --match normal --sizes 10 20 30
    python scripts/error_rate_study.py --match runs     # several training runs per side

Results are deterministic for a given --reps. The study uses 2,000 bootstrap resamples per
comparison (the default is 10,000) to keep the runtime reasonable.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable
from typing import Any

from physim_gate.config import GateConfig
from physim_gate.engine import compare_runs
from physim_gate.schema import RunResult

Generator = Callable[[random.Random, int], tuple[list[float], list[float]]]
ClusterGenerator = Callable[[random.Random, int], tuple[list[list[float]], list[list[float]]]]


def _run(values: list[float], direction: str, run_id: str, prefix: str = "t") -> RunResult:
    return RunResult.model_validate(
        {
            "run_id": run_id,
            "simulator": {"name": "monte-carlo"},
            "scenario": {"id": "study"},
            "metrics": {"m": {"direction": direction}},
            "trials": [
                {"trial_id": f"{prefix}{i}", "metrics": {"m": v}} for i, v in enumerate(values)
            ],
        }
    )


def normal(shift: float, rho: float = 0.0) -> Generator:
    """Shared condition component with correlation rho between baseline and candidate."""

    def generate(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
        shared = [rng.gauss(0, 1) for _ in range(n)]
        own = math.sqrt(1 - rho)
        base = [10 + math.sqrt(rho) * s + own * rng.gauss(0, 1) for s in shared]
        cand = [10 + shift + math.sqrt(rho) * s + own * rng.gauss(0, 1) for s in shared]
        return base, cand

    return generate


def normal_large_baseline(shift: float, n_baseline: int = 1000) -> Generator:
    """A large stored baseline against a small candidate: n is the candidate's size."""

    def generate(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
        base = [10 + rng.gauss(0, 1) for _ in range(n_baseline)]
        cand = [10 + shift + rng.gauss(0, 1) for _ in range(n)]
        return base, cand

    return generate


def lognormal(scale: float) -> Generator:
    def generate(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
        base = [math.exp(rng.gauss(0, 1)) for _ in range(n)]
        cand = [scale * math.exp(rng.gauss(0, 1)) for _ in range(n)]
        return base, cand

    return generate


def bernoulli(p_base: float, p_cand: float, rho: float = 0.0) -> Generator:
    """0/1 outcomes; with rho > 0 the pairs share a latent uniform draw."""

    def generate(rng: random.Random, n: int) -> tuple[list[float], list[float]]:
        base, cand = [], []
        for _ in range(n):
            u = rng.random()
            ub = u if rng.random() < rho else rng.random()
            uc = u if rng.random() < rho else rng.random()
            base.append(1.0 if ub < p_base else 0.0)
            cand.append(1.0 if uc < p_cand else 0.0)
        return base, cand

    return generate


def rates(
    generate: Generator,
    n: int,
    direction: str,
    gate: dict[str, Any],
    reps: int,
    method: str = "expanded_percentile",
) -> dict[str, float]:
    config = GateConfig.model_validate(
        {
            "version": 1,
            "defaults": {
                "bootstrap_resamples": 2000,
                "min_samples": 2,
                "bootstrap_method": method,
            },
            "metrics": {"m": gate},
        }
    )
    counts = dict.fromkeys(("PASS", "FAIL", "INCONCLUSIVE", "INVALID"), 0)
    for rep in range(reps):
        base, cand = generate(random.Random(f"rep-{rep}"), n)
        # Samples of different sizes are independent; distinct ids keep them from counting as
        # missing trials of each other.
        prefixes = ("t", "t") if len(base) == len(cand) else ("b", "c")
        result = compare_runs(
            _run(base, direction, "b", prefixes[0]), _run(cand, direction, "c", prefixes[1]), config
        )
        counts[result.metrics[0].verdict.value] += 1
    return {verdict: count / reps for verdict, count in counts.items()}


def runs_normal(shift: float, seed_sd: float, trials: int = 50) -> ClusterGenerator:
    """Training runs whose means vary with sd ``seed_sd``; trials share episode effects."""

    def generate(rng: random.Random, k: int) -> tuple[list[list[float]], list[list[float]]]:
        episodes = [rng.gauss(0, 1) for _ in range(trials)]
        sides = []
        for offset in (0.0, shift):
            sides.append(
                [
                    [10 + offset + b + e + rng.gauss(0, 0.5) for e in episodes]
                    for b in (rng.gauss(0, seed_sd) for _ in range(k))
                ]
            )
        return sides[0], sides[1]

    return generate


def runs_bernoulli(p: float, shift: float, seed_sd: float, trials: int = 100) -> ClusterGenerator:
    """0/1 outcomes of training runs whose success rates vary with sd ``seed_sd``."""

    def generate(rng: random.Random, k: int) -> tuple[list[list[float]], list[list[float]]]:
        sides = []
        for rate in (p, p + shift):
            runs = []
            for _ in range(k):
                q = rate + rng.gauss(0, seed_sd)
                runs.append([1.0 if rng.random() < q else 0.0 for _ in range(trials)])
            sides.append(runs)
        return sides[0], sides[1]

    return generate


def _clustered_run(runs: list[list[float]], direction: str, run_id: str) -> RunResult:
    return RunResult.model_validate(
        {
            "run_id": run_id,
            "simulator": {"name": "monte-carlo"},
            "scenario": {"id": "study"},
            "metrics": {"m": {"direction": direction}},
            "trials": [
                {
                    "trial_id": f"{run_id}{k}-{i}",
                    "metrics": {"m": v},
                    "labels": {"seed": f"{run_id}{k}"},
                }
                for k, values in enumerate(runs)
                for i, v in enumerate(values)
            ],
        }
    )


def cluster_rates(
    generate: ClusterGenerator,
    k: int,
    direction: str,
    gate: dict[str, Any],
    reps: int,
    method: str = "expanded_percentile",
) -> dict[str, float]:
    """Like :func:`rates`, with ``k`` training runs per side."""
    config = GateConfig.model_validate(
        {
            "version": 1,
            "defaults": {"bootstrap_resamples": 2000, "bootstrap_method": method},
            "metrics": {"m": gate},
        }
    )
    counts = dict.fromkeys(("PASS", "FAIL", "INCONCLUSIVE", "INVALID"), 0)
    for rep in range(reps):
        base, cand = generate(random.Random(f"rep-{rep}"), k)
        result = compare_runs(
            _clustered_run(base, direction, "b"), _clustered_run(cand, direction, "c"), config
        )
        counts[result.metrics[0].verdict.value] += 1
    return {verdict: count / reps for verdict, count in counts.items()}


LOWER, HIGHER = "lower_is_better", "higher_is_better"
UNPAIRED = {"pairing": "unpaired"}

SCENARIOS: list[tuple[str, Generator, str, dict[str, Any], tuple[int, ...]]] = [
    ("normal, unpaired, at margin", normal(0.5), LOWER,
     {"max_absolute_degradation": 0.5, **UNPAIRED}, (10, 30, 100)),
    ("normal, unpaired, 1000 baseline trials, at margin", normal_large_baseline(0.5), LOWER,
     {"max_absolute_degradation": 0.5, **UNPAIRED}, (10, 30, 100)),
    ("normal, paired rho=0.8, at margin", normal(0.5, 0.8), LOWER,
     {"max_absolute_degradation": 0.5}, (10, 30, 100)),
    ("normal, unpaired, no change", normal(0.0), LOWER,
     {"max_absolute_degradation": 0.5, **UNPAIRED}, (10, 30, 100)),
    ("normal, unpaired, 2x margin", normal(1.0), LOWER,
     {"max_absolute_degradation": 0.5, **UNPAIRED}, (10, 30, 100)),
    ("lognormal mean +10%, relative margin 10%", lognormal(1.10), LOWER,
     {"max_relative_degradation": 0.10, **UNPAIRED}, (10, 30, 100)),
    ("success 0.90 -> 0.85, margin 0.05, unpaired", bernoulli(0.90, 0.85), HIGHER,
     {"max_absolute_degradation": 0.05, **UNPAIRED}, (30, 100, 300)),
    ("success 0.90 -> 0.85, margin 0.05, paired", bernoulli(0.90, 0.85, 0.8), HIGHER,
     {"max_absolute_degradation": 0.05}, (30, 100, 300)),
    ("collision 0.01 -> 0.03, margin 0.02, unpaired", bernoulli(0.01, 0.03), LOWER,
     {"max_absolute_degradation": 0.02, **UNPAIRED}, (30, 80, 300)),
    ("collision 0.01 -> 0.06, margin 0.02, unpaired", bernoulli(0.01, 0.06), LOWER,
     {"max_absolute_degradation": 0.02, **UNPAIRED}, (30, 80, 300)),
    ("collision 0.01 -> 0.06, margin 0.02, paired", bernoulli(0.01, 0.06, 0.8), LOWER,
     {"max_absolute_degradation": 0.02}, (30, 80, 300)),
]  # fmt: skip


# Several training runs per side, at the margin: pooling every trial ignores the spread between
# runs; cluster_by: seed compares run means. n is the number of runs per side.
POOLED = {"pairing": "unpaired"}
BY_SEED = {"cluster_by": "seed"}
CLUSTER_SCENARIOS: list[tuple[str, ClusterGenerator, str, dict[str, Any], tuple[int, ...]]] = [
    ("runs, normal, run sd 0.5, at margin, trials pooled", runs_normal(0.5, 0.5), LOWER,
     {"max_absolute_degradation": 0.5, **POOLED}, (3, 5, 10)),
    ("runs, normal, run sd 0.5, at margin, cluster_by", runs_normal(0.5, 0.5), LOWER,
     {"max_absolute_degradation": 0.5, **BY_SEED}, (3, 5, 10)),
    ("runs, normal, no run spread, at margin, cluster_by", runs_normal(0.5, 0.0), LOWER,
     {"max_absolute_degradation": 0.5, **BY_SEED}, (3, 5, 10)),
    ("runs, success 0.42, run sd 0.03, at margin 0.02, trials pooled",
     runs_bernoulli(0.42, -0.02, 0.03), HIGHER,
     {"max_absolute_degradation": 0.02, **POOLED}, (3, 5, 10)),
    ("runs, success 0.42, run sd 0.03, at margin 0.02, cluster_by",
     runs_bernoulli(0.42, -0.02, 0.03), HIGHER,
     {"max_absolute_degradation": 0.02, **BY_SEED}, (3, 5, 10)),
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument(
        "--method",
        choices=("expanded_percentile", "percentile"),
        default="expanded_percentile",
        help="bootstrap interval",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="*", help="override the trial counts of continuous scenarios"
    )
    parser.add_argument("--match", default="", help="only scenarios whose name contains this")
    args = parser.parse_args()
    print(
        f"| scenario | n | PASS | FAIL | INCONCLUSIVE |  "
        f"({args.reps} repetitions each, {args.method} bootstrap)"
    )
    print("|---|---|---|---|---|")
    for name, generate, direction, gate, sizes in SCENARIOS:
        if args.match not in name:
            continue
        if args.sizes and not name.startswith(("success", "collision")):
            sizes = tuple(args.sizes)
        for n in sizes:
            r = rates(generate, n, direction, gate, args.reps, args.method)
            print(
                f"| {name} | {n} | {r['PASS']:.3f} | {r['FAIL']:.3f} | {r['INCONCLUSIVE']:.3f} |",
                flush=True,
            )
    for name, cluster_generate, direction, gate, sizes in CLUSTER_SCENARIOS:
        if args.match not in name:
            continue
        for k in sizes:
            r = cluster_rates(cluster_generate, k, direction, gate, args.reps, args.method)
            print(
                f"| {name} | {k} | {r['PASS']:.3f} | {r['FAIL']:.3f} | {r['INCONCLUSIVE']:.3f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
