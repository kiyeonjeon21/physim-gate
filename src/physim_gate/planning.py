"""Sample-size planning: how many trials per run does a gate need to reach a decision?

The gate PASSes a metric only when the whole confidence interval of the degradation lies at or
below the margin, and FAILs it only when the whole interval lies above. With too few trials the
interval straddles the margin and the verdict is INCONCLUSIVE. Planning answers: given what you
expect the true degradation to be, how many trials make the intended verdict likely?

* 0/1 metrics (success, collision, ...) are evaluated exactly. Every possible pair of event
  counts is weighted by its binomial probability and decided with the same unpaired Newcombe
  score interval the comparison engine uses, so the reported verdict probabilities are exact
  for unpaired comparisons.
* Paired 0/1 metrics depend on how often the two runs disagree on the same condition (the
  discordance). Their probabilities are simulated with the engine's paired Newcombe interval.
* :func:`plan_from_comparison` takes everything from a pilot comparison: rates, discordance,
  spreads, pairing, margins and the observed degradation.
* Continuous metrics use the normal approximation
  ``n = ((z_conf + z_power) * sd_diff / |margin - degradation|) ** 2``. The engine uses a
  bootstrap, so this is an estimate, and it is least reliable for small n and skewed data.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from physim_gate.combine import RUN_LABEL
from physim_gate.comparison import ComparisonResult, MetricComparison
from physim_gate.config import Statistic
from physim_gate.schema import Direction
from physim_gate.stats import _t_quantile, is_binary, newcombe_counts, newcombe_paired_counts

#: Simulated comparisons per candidate size for paired 0/1 metrics.
PAIRED_SIMULATIONS = 4000

#: Tail mass ignored when enumerating binomial outcomes.
_TAIL = 1e-13


class Goal(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class VerdictProbabilities:
    pass_: float
    fail: float
    inconclusive: float

    def of(self, goal: Goal) -> float:
        return self.pass_ if goal is Goal.PASS else self.fail


@dataclass(frozen=True)
class Plan:
    kind: str
    method: str
    goal: Goal
    margin: float
    true_degradation: float
    confidence: float
    power: float
    #: Trials per run (or matched pairs). ``None`` when more than ``max_trials`` are needed.
    trials: int | None
    max_trials: int
    #: Verdict probabilities at ``trials``: exact for 0/1 metrics, approximate otherwise.
    probabilities: VerdictProbabilities | None
    notes: list[str] = field(default_factory=list)
    #: What ``trials`` counts: "trials", or "runs" / "clusters" for clustered comparisons.
    unit: str = "trials"


class PlanningError(ValueError):
    """The planning question has no answer (for example degradation equal to the margin)."""


def _goal(margin: float, true_degradation: float) -> Goal:
    if margin < 0:
        raise PlanningError("margin must be >= 0")
    if math.isclose(true_degradation, margin, rel_tol=0, abs_tol=1e-12):
        raise PlanningError(
            "the true degradation equals the margin: at the boundary neither PASS nor FAIL is "
            "expected, whatever the number of trials"
        )
    return Goal.PASS if true_degradation < margin else Goal.FAIL


def _z(probability: float) -> float:
    return statistics.NormalDist().inv_cdf(probability)


def _check_levels(confidence: float, power: float) -> None:
    if not 0.5 < confidence < 1:
        raise PlanningError("confidence must be between 0.5 and 1")
    if not 0 < power < 1:
        raise PlanningError("power must be between 0 and 1")


class _Binomial:
    """Binomial(n, p) probabilities over the range that carries all but ``_TAIL`` of the mass."""

    def __init__(self, n: int, p: float) -> None:
        self.n = n
        if p <= 0 or p >= 1:
            self.lo = self.hi = 0 if p <= 0 else n
            self.pmf = [1.0]
        else:
            sd = math.sqrt(n * p * (1 - p))
            width = 9 * sd + 10
            self.lo = max(0, math.floor(n * p - width))
            self.hi = min(n, math.ceil(n * p + width))
            log_p, log_q = math.log(p), math.log1p(-p)
            base = math.lgamma(n + 1)
            self.pmf = [
                math.exp(
                    base - math.lgamma(x + 1) - math.lgamma(n - x + 1) + x * log_p + (n - x) * log_q
                )
                for x in range(self.lo, self.hi + 1)
            ]
        self._cdf: list[float] = []
        total = 0.0
        for value in self.pmf:
            total += value
            self._cdf.append(total)

    def cdf(self, x: int) -> float:
        """P(X <= x)."""
        if x < self.lo:
            return 0.0
        if x >= self.hi:
            return 1.0
        return self._cdf[x - self.lo]

    def support(self) -> range:
        return range(self.lo, self.hi + 1)

    def prob(self, x: int) -> float:
        return self.pmf[x - self.lo]


def binary_verdict_probabilities(
    trials: int,
    baseline_bad_rate: float,
    candidate_bad_rate: float,
    margin: float,
    confidence: float,
) -> VerdictProbabilities:
    """Exact verdict probabilities for an unpaired 0/1 metric where events are bad.

    Degradation is ``candidate rate - baseline rate`` of the bad event. The interval for each
    outcome is the engine's unpaired Newcombe interval. Its upper and lower bounds increase with
    the candidate count and decrease with the baseline count, so for each baseline count the
    decisive candidate counts form a prefix (PASS) and a suffix (FAIL), found with a sweep.
    """
    n = trials
    base = _Binomial(n, baseline_bad_rate)
    cand = _Binomial(n, candidate_bad_rate)

    def upper(xb: int, xc: int) -> float:
        return newcombe_counts(xb, n, xc, n, confidence)[1]

    def lower(xb: int, xc: int) -> float:
        return newcombe_counts(xb, n, xc, n, confidence)[0]

    p_pass = p_fail = 0.0
    # PASS needs upper(xb, xc) <= margin; the largest such xc never decreases as xb grows.
    pass_edge = -1
    # FAIL needs lower(xb, xc) > margin; the smallest such xc never decreases as xb grows.
    fail_edge = 0
    for xb in base.support():
        while pass_edge + 1 <= n and upper(xb, pass_edge + 1) <= margin:
            pass_edge += 1
        while fail_edge <= n and not lower(xb, fail_edge) > margin:
            fail_edge += 1
        weight = base.prob(xb)
        p_pass += weight * (cand.cdf(pass_edge) if pass_edge >= 0 else 0.0)
        p_fail += weight * (1.0 - cand.cdf(fail_edge - 1))
    p_pass, p_fail = min(max(p_pass, 0.0), 1.0), min(max(p_fail, 0.0), 1.0)
    return VerdictProbabilities(
        pass_=p_pass, fail=p_fail, inconclusive=max(0.0, 1.0 - p_pass - p_fail)
    )


def _smallest_n(
    probability: Callable[[int], float], power: float, start: int, max_trials: int
) -> int | None:
    """Smallest n >= start with probability(n) >= power, assuming it rises with n."""
    if probability(start) >= power:
        return start
    low, high = start, start
    while probability(high) < power:
        if high >= max_trials:
            return None
        low, high = high, min(max_trials, high * 2)
    while high - low > 1:
        mid = (low + high) // 2
        if probability(mid) >= power:
            high = mid
        else:
            low = mid
    return high


def plan_binary(
    rate: float,
    margin: float,
    *,
    direction: Direction = Direction.HIGHER_IS_BETTER,
    true_degradation: float = 0.0,
    confidence: float = 0.95,
    power: float = 0.8,
    min_samples: int = 10,
    max_trials: int = 100_000,
    discordance: float | None = None,
) -> Plan:
    """Trials per run for a 0/1 metric with baseline ``rate`` (fraction of 1s).

    With ``discordance`` (the probability that baseline and candidate disagree on the same trial
    condition) the runs are planned as paired, by simulation. Without it they are unpaired and
    the probabilities are exact.
    """
    _check_levels(confidence, power)
    if not 0 <= rate <= 1:
        raise PlanningError("rate must be between 0 and 1")
    goal = _goal(margin, true_degradation)
    bad_base = rate if direction is Direction.LOWER_IS_BETTER else 1.0 - rate
    bad_cand = bad_base + true_degradation
    if not 0 <= bad_cand <= 1:
        raise PlanningError(
            f"a true degradation of {true_degradation:g} from a rate of {rate:g} leaves [0, 1]"
        )
    if discordance is not None:
        cells = PairCells.from_discordance(bad_base, discordance, true_degradation)
        return _plan_paired_binary(
            cells, margin, goal, true_degradation, confidence, power, min_samples, max_trials
        )

    cache: dict[int, VerdictProbabilities] = {}

    def probabilities(n: int) -> VerdictProbabilities:
        if n not in cache:
            cache[n] = binary_verdict_probabilities(n, bad_base, bad_cand, margin, confidence)
        return cache[n]

    trials = _smallest_n(lambda n: probabilities(n).of(goal), power, min_samples, max_trials)
    notes = [
        "exact for unpaired runs; runs that share trial conditions (paired) usually need fewer",
        "power of 0/1 metrics rises unevenly with n, so a few more trials than shown add margin",
    ]
    return Plan(
        kind="binary",
        method="exact (unpaired Newcombe score interval)",
        goal=goal,
        margin=margin,
        true_degradation=true_degradation,
        confidence=confidence,
        power=power,
        trials=trials,
        max_trials=max_trials,
        probabilities=probabilities(trials) if trials is not None else None,
        notes=notes,
    )


def plan_continuous(
    stdev: float,
    margin: float,
    *,
    paired: bool = False,
    true_degradation: float = 0.0,
    confidence: float = 0.95,
    power: float = 0.8,
    min_samples: int = 10,
    max_trials: int = 100_000,
) -> Plan:
    """Trials per run for a continuous metric.

    ``stdev`` is the standard deviation of single-trial values (unpaired) or of the per-pair
    differences between candidate and baseline (``paired=True``).
    """
    _check_levels(confidence, power)
    if not stdev > 0:
        raise PlanningError("stdev must be > 0")
    goal = _goal(margin, true_degradation)
    sd_diff = stdev if paired else stdev * math.sqrt(2)
    z_conf = _z(1 - (1 - confidence) / 2)
    gap = abs(margin - true_degradation)
    needed = math.ceil(((z_conf + _z(power)) * sd_diff / gap) ** 2)
    required = max(needed, min_samples)
    trials = required if required <= max_trials else None

    probabilities = None
    if trials is not None:
        se = sd_diff / math.sqrt(trials)
        normal = statistics.NormalDist()
        p_pass = normal.cdf((margin - true_degradation) / se - z_conf)
        p_fail = normal.cdf((true_degradation - margin) / se - z_conf)
        probabilities = VerdictProbabilities(
            pass_=p_pass, fail=p_fail, inconclusive=max(0.0, 1.0 - p_pass - p_fail)
        )
    notes = ["the engine's bootstrap may need somewhat more trials"]
    if trials is not None and trials < 30:
        notes.append("below 30 trials the bootstrap interval is too narrow, see the error rates")
    return Plan(
        kind="continuous",
        method="normal approximation (paired)" if paired else "normal approximation (unpaired)",
        goal=goal,
        margin=margin,
        true_degradation=true_degradation,
        confidence=confidence,
        power=power,
        trials=trials,
        max_trials=max_trials,
        probabilities=probabilities,
        notes=notes,
    )


def plan_clusters(
    stdev: float,
    margin: float,
    *,
    paired: bool = False,
    true_degradation: float = 0.0,
    confidence: float = 0.95,
    power: float = 0.8,
    min_clusters: int = 3,
    max_clusters: int = 10_000,
    unit: str = "clusters",
) -> Plan:
    """Clusters (runs, training seeds) per side for a clustered comparison.

    ``stdev`` is the standard deviation of cluster means (unpaired) or of the per-cluster
    differences (``paired=True``). The engine's t interval has ``k - 1`` degrees of freedom
    for ``k`` matched clusters and about ``2k - 2`` for two sides of ``k``, so the critical
    value is a t quantile; the spread of the estimate is treated as normal.
    """
    _check_levels(confidence, power)
    if not stdev > 0:
        raise PlanningError("stdev must be > 0")
    goal = _goal(margin, true_degradation)
    sd_diff = stdev if paired else stdev * math.sqrt(2)
    normal = statistics.NormalDist()

    def probabilities(k: int) -> VerdictProbabilities:
        se = sd_diff / math.sqrt(k)
        critical = _t_quantile(1 - (1 - confidence) / 2, k - 1 if paired else 2 * k - 2)
        p_pass = normal.cdf((margin - true_degradation) / se - critical)
        p_fail = normal.cdf((true_degradation - margin) / se - critical)
        return VerdictProbabilities(
            pass_=p_pass, fail=p_fail, inconclusive=max(0.0, 1.0 - p_pass - p_fail)
        )

    start = max(min_clusters, 2)
    clusters = _smallest_n(lambda k: probabilities(k).of(goal), power, start, max_clusters)
    return Plan(
        kind="clusters",
        method=f"t approximation over cluster means ({'paired' if paired else 'unpaired'})",
        goal=goal,
        margin=margin,
        true_degradation=true_degradation,
        confidence=confidence,
        power=power,
        trials=clusters,
        max_trials=max_clusters,
        probabilities=probabilities(clusters) if clusters is not None else None,
        notes=[f"counts {unit} per side; the trials within each only shrink its spread"],
        unit=unit,
    )


@dataclass(frozen=True)
class PairCells:
    """Probabilities of the four outcomes of a pair, counting the *bad* event (degradation up).

    ``both``: bad in baseline and candidate; ``base_only``: bad only in the baseline;
    ``cand_only``: bad only in the candidate; ``neither``: bad in neither.
    """

    both: float
    base_only: float
    cand_only: float
    neither: float

    @property
    def degradation(self) -> float:
        return self.cand_only - self.base_only

    @property
    def discordance(self) -> float:
        return self.base_only + self.cand_only

    @classmethod
    def from_discordance(cls, bad_base: float, discordance: float, degradation: float) -> PairCells:
        if not 0 < discordance <= 1:
            raise PlanningError("discordance must be in (0, 1]")
        if abs(degradation) > discordance:
            raise PlanningError(
                f"a true degradation of {degradation:g} needs a discordance of at least "
                f"{abs(degradation):g}: only pairs that disagree can move the rate"
            )
        base_only = (discordance - degradation) / 2
        cand_only = (discordance + degradation) / 2
        both = bad_base - base_only
        neither = 1.0 - both - base_only - cand_only
        if both < -1e-12 or neither < -1e-12:
            raise PlanningError(
                f"a discordance of {discordance:g} is impossible with a rate of "
                f"{bad_base:g} and a degradation of {degradation:g}"
            )
        return cls(max(both, 0.0), base_only, cand_only, max(neither, 0.0))

    def with_degradation(self, degradation: float) -> PairCells:
        """The same rate and discordance, with the disagreements split to give ``degradation``."""
        return PairCells.from_discordance(self.both + self.base_only, self.discordance, degradation)


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """A Binomial(n, p) draw by counting geometric waiting times, O(n * min(p, 1 - p))."""
    if p <= 0 or n <= 0:
        return 0
    if p >= 1:
        return n
    if p > 0.5:
        return n - _binomial(rng, n, 1 - p)
    log_q = math.log1p(-p)
    count = position = 0
    while True:
        position += int(math.log(1.0 - rng.random()) / log_q) + 1
        if position > n:
            return count
        count += 1


def paired_binary_verdict_probabilities(
    trials: int,
    cells: PairCells,
    margin: float,
    confidence: float,
    *,
    simulations: int = PAIRED_SIMULATIONS,
    seed: str = "physim-gate-plan",
) -> VerdictProbabilities:
    """Simulated verdict probabilities for a paired 0/1 metric (bad events, lower is better)."""
    rng = random.Random(f"{seed}:{trials}")
    p_pass = p_fail = 0
    for _ in range(simulations):
        remaining = trials
        base_only = _binomial(rng, remaining, cells.base_only)
        remaining -= base_only
        rest = 1.0 - cells.base_only
        cand_only = _binomial(rng, remaining, cells.cand_only / rest if rest > 0 else 0.0)
        remaining -= cand_only
        rest -= cells.cand_only
        both = _binomial(rng, remaining, cells.both / rest if rest > 0 else 0.0)
        neither = remaining - both
        low, high = newcombe_paired_counts(
            both, base_only, cand_only, neither, confidence=confidence
        )
        if high <= margin:
            p_pass += 1
        elif low > margin:
            p_fail += 1
    return VerdictProbabilities(
        pass_=p_pass / simulations,
        fail=p_fail / simulations,
        inconclusive=(simulations - p_pass - p_fail) / simulations,
    )


def _independent_discordance(cells: PairCells) -> float:
    base, cand = cells.both + cells.base_only, cells.both + cells.cand_only
    return base * (1 - cand) + cand * (1 - base)


def _plan_paired_binary(
    cells: PairCells,
    margin: float,
    goal: Goal,
    true_degradation: float,
    confidence: float,
    power: float,
    min_samples: int,
    max_trials: int,
) -> Plan:
    cache: dict[int, VerdictProbabilities] = {}

    def probabilities(n: int) -> VerdictProbabilities:
        if n not in cache:
            cache[n] = paired_binary_verdict_probabilities(n, cells, margin, confidence)
        return cache[n]

    trials = _smallest_n(lambda n: probabilities(n).of(goal), power, min_samples, max_trials)
    se = math.sqrt(power * (1 - power) / PAIRED_SIMULATIONS)
    return Plan(
        kind="binary",
        method=f"simulated, {PAIRED_SIMULATIONS:,} runs per size (paired Newcombe interval)",
        goal=goal,
        margin=margin,
        true_degradation=true_degradation,
        confidence=confidence,
        power=power,
        trials=trials,
        max_trials=max_trials,
        probabilities=probabilities(trials) if trials is not None else None,
        notes=[
            f"discordance {cells.discordance:.3g}: the share of pairs whose outcomes differ "
            f"(independent runs would give {_independent_discordance(cells):.3g}; less means "
            "pairing helps)",
            f"simulated probabilities are accurate to about +/- {se * 100:.1f} points",
        ],
    )


@dataclass(frozen=True)
class MetricPlan:
    """The plan for one metric of a pilot comparison, or why there is none."""

    metric: str
    plan: Plan | None
    reason: str | None = None


def plan_from_comparison(
    result: ComparisonResult,
    *,
    metrics: list[str] | None = None,
    true_degradation: float | None = None,
    power: float = 0.8,
    max_trials: int = 100_000,
) -> list[MetricPlan]:
    """Plan each gated metric of a pilot comparison from its own data.

    The observed degradation is the assumed true degradation unless ``true_degradation`` is
    given. Rates, spreads, pairing, discordance, margins and confidence come from the pilot.
    A relative margin is used at its pilot value.
    """
    known = {m.name for m in result.metrics}
    unknown = sorted(set(metrics or []) - known)
    if unknown:
        raise PlanningError(
            f"no metric {', '.join(unknown)} in the comparison; it has {', '.join(sorted(known))}"
        )
    plans = []
    for metric in result.metrics:
        if metrics and metric.name not in metrics:
            continue
        try:
            plan = _plan_metric(result, metric, true_degradation, power, max_trials)
            plans.append(MetricPlan(metric.name, plan))
        except PlanningError as exc:
            plans.append(MetricPlan(metric.name, None, str(exc)))
    return plans


def _plan_metric(
    result: ComparisonResult,
    metric: MetricComparison,
    true_degradation: float | None,
    power: float,
    max_trials: int,
) -> Plan:
    n, base, cand = metric.numerical, metric.baseline, metric.candidate
    if n is None or base is None or cand is None or metric.direction is None:
        raise PlanningError(f"{metric.verdict.value}: the pilot has no usable values")
    degradation = n.degradation if true_degradation is None else true_degradation
    if metric.clusters is not None:
        return _plan_clustered(result, metric, degradation, power, max_trials)
    # A pilot with fewer than min_samples observations has no interval, so its evidence records
    # neither the method nor the confidence. Derive both the way the engine would have.
    confidence = metric.evidence.adjusted_confidence or result.config.adjusted_confidence(
        metric.name
    )
    paired = metric.pairing is not None and metric.pairing.mode == "paired"
    method = metric.evidence.method
    if method is None:
        binary = metric.statistic is Statistic.MEAN and is_binary(base.values + cand.values)
    else:
        binary = method.endswith("newcombe")
    lower = metric.direction is Direction.LOWER_IS_BETTER

    if binary:
        if not paired:
            return plan_binary(
                base.mean or 0.0,
                n.margin,
                direction=metric.direction,
                true_degradation=degradation,
                confidence=confidence,
                power=power,
                min_samples=metric.min_samples,
                max_trials=max_trials,
            )
        bad = [
            (b if lower else 1.0 - b, c if lower else 1.0 - c)
            for b, c in zip(base.values, cand.values, strict=True)
        ]
        size = len(bad)
        cells = PairCells(
            both=sum(1 for b, c in bad if b and c) / size,
            base_only=sum(1 for b, c in bad if b and not c) / size,
            cand_only=sum(1 for b, c in bad if c and not b) / size,
            neither=sum(1 for b, c in bad if not b and not c) / size,
        )
        if cells.discordance == 0:
            raise PlanningError(
                "every pair agreed in the pilot, so the paired interval cannot be planned from it; "
                "use 'plan --rate ... --discordance ...' with an assumed discordance"
            )
        if true_degradation is not None:
            cells = cells.with_degradation(true_degradation)
        _check_levels(confidence, power)
        goal = _goal(n.margin, cells.degradation)
        return _plan_paired_binary(
            cells,
            n.margin,
            goal,
            cells.degradation,
            confidence,
            power,
            metric.min_samples,
            max_trials,
        )

    if paired:
        differences = [c - b for b, c in zip(base.values, cand.values, strict=True)]
        spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
    else:
        spread = (
            math.sqrt((statistics.variance(base.values) + statistics.variance(cand.values)) / 2)
            if len(base.values) > 1 and len(cand.values) > 1
            else 0.0
        )
    if spread == 0:
        raise PlanningError("the pilot values do not vary, so their spread cannot be estimated")
    plan = plan_continuous(
        spread,
        n.margin,
        paired=paired,
        true_degradation=degradation,
        confidence=confidence,
        power=power,
        min_samples=metric.min_samples,
        max_trials=max_trials,
    )
    if metric.statistic is Statistic.MEDIAN:
        plan.notes.append("the plan assumes the mean; medians usually need more trials")
    return plan


def _plan_clustered(
    result: ComparisonResult,
    metric: MetricComparison,
    degradation: float,
    power: float,
    max_clusters: int,
) -> Plan:
    clusters, n = metric.clusters, metric.numerical
    assert clusters is not None
    assert n is not None
    b_means = list(clusters.baseline_means.values())
    c_means = list(clusters.candidate_means.values())
    paired = clusters.mode == "paired"
    if paired:
        differences = [c - b for b, c in zip(b_means, c_means, strict=True)]
        spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
    elif len(b_means) > 1 and len(c_means) > 1:
        spread = math.sqrt((statistics.variance(b_means) + statistics.variance(c_means)) / 2)
    else:
        raise PlanningError(
            "the pilot needs at least 2 clusters per side to estimate the spread between them"
        )
    if spread == 0:
        raise PlanningError(
            "the pilot's cluster means do not vary, so their spread cannot be estimated"
        )
    confidence = metric.evidence.adjusted_confidence or result.config.adjusted_confidence(
        metric.name
    )
    gate = result.config.resolve(metric.name)
    unit = "runs" if clusters.label == RUN_LABEL else "clusters"
    return plan_clusters(
        spread,
        n.margin,
        paired=paired,
        true_degradation=degradation,
        confidence=confidence,
        power=power,
        min_clusters=gate.min_clusters,
        max_clusters=max_clusters,
        unit=unit,
    )
