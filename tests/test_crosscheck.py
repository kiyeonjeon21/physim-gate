"""Cross-checks of the reimplemented statistics against scipy and statsmodels.

PhySim Gate keeps its runtime dependencies small, so it does not call these libraries. These
tests run where they are installed (the ``dev`` extra) and skip otherwise.
"""

from __future__ import annotations

import itertools
import random

import pytest

from physim_gate.config import BootstrapMethod, Statistic
from physim_gate.stats import (
    _expanded_alpha,
    _t_quantile,
    bootstrap_degradation,
    newcombe_counts,
    t_interval_difference,
)

np = pytest.importorskip("numpy")
scipy_stats = pytest.importorskip("scipy.stats")


def _samples(seed: int, nb: int, nc: int) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    return (
        [rng.lognormvariate(0, 1) for _ in range(nb)],
        [rng.lognormvariate(0.3, 1) for _ in range(nc)],
    )


@pytest.mark.parametrize("paired", [False, True])
def test_percentile_bootstrap_matches_scipy(paired: bool) -> None:
    baseline, candidate = _samples(3, 12 if paired else 15, 12)
    ours = bootstrap_degradation(
        baseline,
        candidate,
        paired=paired,
        statistic=Statistic.MEAN,
        sign=1.0,
        margin=lambda _: 0.0,
        confidence=0.95,
        resamples=100_000,
        seed="crosscheck",
        method=BootstrapMethod.PERCENTILE,
    )
    reference = scipy_stats.bootstrap(
        (np.array(baseline), np.array(candidate)),
        lambda b, c, axis=-1: np.mean(c, axis=axis) - np.mean(b, axis=axis),
        method="percentile",
        paired=paired,
        n_resamples=100_000,
        confidence_level=0.95,
        random_state=1,
        vectorized=True,
    ).confidence_interval
    # Both are Monte Carlo estimates; with 100,000 resamples the endpoints agree to about 1%
    # of the interval width.
    width = reference.high - reference.low
    assert ours.degradation[0] == pytest.approx(reference.low, abs=0.02 * width)
    assert ours.degradation[1] == pytest.approx(reference.high, abs=0.02 * width)


def test_expanded_percentile_levels_match_scipy_t_quantiles() -> None:
    for df in (1, 2, 5, 9, 29, 100, 1000):
        assert _t_quantile(0.975, df) == pytest.approx(scipy_stats.t.ppf(0.975, df), rel=1e-10)
    for df in (2.5, 9.37, 117.2):
        assert _t_quantile(0.975, df) == pytest.approx(scipy_stats.t.ppf(0.975, df), rel=1e-10)
    for n, df in ((10, 9), (10, 18), (30, 29), (200, 398)):
        expected = 2 * scipy_stats.norm.cdf(-np.sqrt(n / (n - 1)) * scipy_stats.t.ppf(0.975, df))
        assert _expanded_alpha(0.05, n / (n - 1), df) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(("nb", "nc"), [(3, 3), (5, 3), (10, 4)])
def test_cluster_t_intervals_match_scipy(nb: int, nc: int) -> None:
    baseline, candidate = _samples(nb * 10 + nc, nb, nc)
    welch = scipy_stats.ttest_ind(candidate, baseline, equal_var=False).confidence_interval(0.95)
    ours = t_interval_difference(baseline, candidate, paired=False, confidence=0.95)
    assert (ours.low, ours.high) == pytest.approx((welch.low, welch.high), rel=1e-9)
    if nb == nc:
        paired = scipy_stats.ttest_rel(candidate, baseline).confidence_interval(0.9)
        ours = t_interval_difference(baseline, candidate, paired=True, confidence=0.9)
        assert (ours.low, ours.high) == pytest.approx((paired.low, paired.high), rel=1e-9)


def test_unpaired_newcombe_matches_statsmodels() -> None:
    proportion = pytest.importorskip("statsmodels.stats.proportion")
    for xb, nb, xc, nc in itertools.product([0, 3, 9], [10, 48], [0, 5, 20], [20, 56]):
        if xb > nb or xc > nc:
            continue
        low, high = proportion.confint_proportions_2indep(
            xc, nc, xb, nb, method="newcomb", compare="diff", alpha=0.05
        )
        ours = newcombe_counts(xb, nb, xc, nc, 0.95)
        assert ours == pytest.approx((low, high), abs=1e-12)
