"""Small, dependency-free statistics used by the comparison engine.

Everything here is deterministic for a given seed, so a comparison can be reproduced exactly
from its inputs. The bootstrap uses Python's Mersenne Twister seeded with a string, which is
stable across platforms and Python versions.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from physim_gate.config import BootstrapMethod, Statistic

StatFn = Callable[[Sequence[float]], float]


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def stat_fn(statistic: Statistic) -> StatFn:
    return mean if statistic is Statistic.MEAN else median


def stdev(values: Sequence[float]) -> float | None:
    """Sample standard deviation, or ``None`` with fewer than two values."""
    return statistics.stdev(values) if len(values) >= 2 else None


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman and Fan type 7) of pre-sorted values."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


Interval = tuple[float, float]


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile bootstrap intervals for a degradation estimate.

    ``degradation`` is the interval of ``sign * (stat(candidate) - stat(baseline))``, so positive
    values are worse. ``excess`` is the interval of the degradation minus the allowed margin, with
    the margin recomputed in each replicate when it depends on the baseline (relative thresholds).
    The gate decides on ``excess``: the entire interval above zero is evidence of a regression,
    the entire interval at or below zero is evidence of non-inferiority.
    """

    degradation: Interval
    excess: Interval


def bootstrap_degradation(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    paired: bool,
    statistic: Statistic,
    sign: float,
    margin: Callable[[float], float],
    confidence: float,
    resamples: int,
    seed: str,
    method: BootstrapMethod = BootstrapMethod.EXPANDED,
) -> BootstrapResult:
    """Bootstrap the degradation of ``candidate`` relative to ``baseline``.

    Paired: trial indices are resampled jointly, preserving the pairing. Unpaired: each sample is
    resampled independently. The interval is two-sided at ``confidence``. Deciding on one side
    of it gives a one-sided error rate of about ``(1 - confidence) / 2``.

    ``method`` picks the interval: plain percentiles, or Hesterberg's expanded percentiles
    (the default), which widen the interval for small samples; see :func:`_expanded_alpha`.
    """
    if paired and len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal length")
    if not baseline or not candidate:
        raise ValueError("bootstrap needs non-empty samples")

    rng = random.Random(seed)
    fn = stat_fn(statistic)
    degradations: list[float] = []
    excesses: list[float] = []

    if paired:
        indices = range(len(baseline))
        n = len(baseline)
        for _ in range(resamples):
            pick = rng.choices(indices, k=n)
            b = fn([baseline[i] for i in pick])
            c = fn([candidate[i] for i in pick])
            d = sign * (c - b)
            degradations.append(d)
            excesses.append(d - margin(b))
    else:
        nb, nc = len(baseline), len(candidate)
        for _ in range(resamples):
            b = fn(rng.choices(baseline, k=nb))
            c = fn(rng.choices(candidate, k=nc))
            d = sign * (c - b)
            degradations.append(d)
            excesses.append(d - margin(b))

    alpha = 1.0 - confidence
    degradations.sort()
    excesses.sort()
    levels = {"degradation": (alpha / 2, 1 - alpha / 2), "excess": (alpha / 2, 1 - alpha / 2)}
    if method is BootstrapMethod.EXPANDED:
        if paired:
            n = len(baseline)
            inflation, df = (n / (n - 1), n - 1) if n > 1 else (1.0, 0.0)
        else:
            inflation, df = _welch_expansion(baseline, candidate)
        expanded = _expanded_alpha(alpha, inflation, df)
        levels = dict.fromkeys(levels, (expanded / 2, 1 - expanded / 2))
    return BootstrapResult(
        degradation=(
            quantile(degradations, levels["degradation"][0]),
            quantile(degradations, levels["degradation"][1]),
        ),
        excess=(quantile(excesses, levels["excess"][0]), quantile(excesses, levels["excess"][1])),
    )


def _expanded_alpha(alpha: float, inflation: float, df: float) -> float:
    """Hesterberg's expanded percentile interval (2015, The American Statistician 69:371).

    The percentile interval is too narrow for small samples, partly because the bootstrap uses
    the plug-in variance (divisor n) and partly because it ignores the extra tail weight that a
    t distribution would add. Both are undone by using the percentiles
    ``Phi(+-sqrt(inflation) * t[alpha/2, df])`` instead of ``alpha / 2``, where ``inflation`` is
    the ratio of the unbiased to the plug-in variance of the statistic. For one sample (or
    pairs) that is ``n / (n - 1)`` with ``df = n - 1``; for two independent samples see
    :func:`_welch_expansion`.
    """
    if inflation <= 0 or not df >= 1:
        return alpha
    t = _t_quantile(1 - alpha / 2, df)
    return 2 * statistics.NormalDist().cdf(-math.sqrt(inflation) * t)


def _welch_expansion(baseline: Sequence[float], candidate: Sequence[float]) -> tuple[float, float]:
    """Variance inflation and Welch-Satterthwaite degrees of freedom for two samples.

    The variance of the difference is ``s_b^2 / n_b + s_c^2 / n_c``, while the bootstrap sees
    the plug-in version ``s_b^2 (n_b - 1) / n_b^2 + s_c^2 (n_c - 1) / n_c^2``. Their ratio is the
    inflation. The degrees of freedom follow the larger variance term, so a small candidate
    against a large baseline gets about ``n_c - 1`` instead of ``n_b + n_c - 2``. For two
    samples of equal size and variance this is exactly ``n / (n - 1)`` and ``2n - 2``.
    """
    nb, nc = len(baseline), len(candidate)
    if nb < 2 or nc < 2:
        return 1.0, 0.0
    vb = statistics.variance(baseline) / nb
    vc = statistics.variance(candidate) / nc
    plug_in = vb * (nb - 1) / nb + vc * (nc - 1) / nc
    if plug_in <= 0:
        # Both samples are constant: the interval has zero width whatever the expansion.
        return 1.0, 0.0
    df = (vb + vc) ** 2 / (vb**2 / (nb - 1) + vc**2 / (nc - 1))
    return (vb + vc) / plug_in, df


def _t_quantile(probability: float, df: float) -> float:
    """Student t quantile by bisection on the CDF (regularized incomplete beta)."""

    def cdf(x: float) -> float:
        beta = _regularized_beta(df / (df + x * x), df / 2, 0.5)
        return 1 - beta / 2 if x >= 0 else beta / 2

    low, high = 0.0, 1.0
    while cdf(high) < probability:
        # Few degrees of freedom and a strict level (Bonferroni) put the quantile far out:
        # t[0.99999, 1] is about 31831.
        low, high = high, high * 2
    for _ in range(200):
        mid = (low + high) / 2
        if cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _regularized_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) by Lentz's continued fraction (Numerical Recipes, section 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_beta(1 - x, b, a)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 300):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            result *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return math.exp(log_front) * result / a


@dataclass(frozen=True)
class TInterval:
    """A t interval of ``mean(candidate) - mean(baseline)``, with its standard error and df."""

    low: float
    high: float
    standard_error: float
    df: float


def t_interval_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    paired: bool,
    confidence: float,
) -> TInterval:
    """Student t interval of the difference in means: paired t, or Welch's t when unpaired.

    The engine uses it over cluster means (one value per training seed, say), where the
    number of values is small and a bootstrap would be unreliable.
    """
    if paired:
        if len(baseline) != len(candidate):
            raise ValueError("paired samples must have equal length")
        n = len(baseline)
        if n < 2:
            raise ValueError("a paired t interval needs at least 2 pairs")
        differences = [c - b for b, c in zip(baseline, candidate, strict=True)]
        center = mean(differences)
        se = statistics.stdev(differences) / math.sqrt(n)
        df: float = n - 1
    else:
        nb, nc = len(baseline), len(candidate)
        if nb < 2 or nc < 2:
            raise ValueError("a Welch t interval needs at least 2 values per sample")
        vb = statistics.variance(baseline) / nb
        vc = statistics.variance(candidate) / nc
        center = mean(candidate) - mean(baseline)
        se = math.sqrt(vb + vc)
        df = (vb + vc) ** 2 / (vb**2 / (nb - 1) + vc**2 / (nc - 1)) if se > 0 else nb + nc - 2
    half = _t_quantile(1 - (1 - confidence) / 2, df) * se
    return TInterval(low=center - half, high=center + half, standard_error=se, df=df)


def is_binary(values: Sequence[float]) -> bool:
    """True when every value is exactly 0 or 1 (success flags, collision indicators, ...)."""
    return all(v in (0.0, 1.0) for v in values)


def wilson_interval(successes: int, n: int, z: float) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("wilson_interval needs n > 0")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    paired: bool,
    confidence: float,
) -> Interval:
    """Score interval for ``mean(candidate) - mean(baseline)`` of 0/1 data.

    Unpaired: Newcombe's hybrid score interval (Newcombe 1998, Stat Med 17:873, method 10).
    Paired: Newcombe's interval for paired proportions (Newcombe 1998, Stat Med 17:2635,
    method 10), which combines the Wilson intervals of both rates with the continuity-corrected
    correlation of the pairs.

    Unlike a percentile bootstrap, these intervals stay wide when a sample has no events at all,
    so "0 collisions in 30 trials" is not mistaken for a collision rate known to be zero.
    """
    if not baseline or not candidate:
        raise ValueError("newcombe_difference needs non-empty samples")
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    nb, nc = len(baseline), len(candidate)
    xb, xc = round(sum(baseline)), round(sum(candidate))

    if paired:
        if nb != nc:
            raise ValueError("paired samples must have equal length")
        both = sum(1 for b, c in zip(baseline, candidate, strict=True) if b == 1 and c == 1)
        return newcombe_paired_counts(
            both, xb - both, xc - both, nb - xb - xc + both, confidence=confidence
        )
    return _newcombe_combine(xb, nb, xc, nc, z, 0.0)


def newcombe_paired_counts(
    both: int, base_only: int, cand_only: int, neither: int, *, confidence: float
) -> Interval:
    """Paired Newcombe interval (method 10) from the four cells of the 2x2 table of pairs.

    ``both`` counts pairs where baseline and candidate are 1, ``base_only`` pairs where only the
    baseline is 1, and so on. The interval is for ``rate(candidate) - rate(baseline)``.
    """
    n = both + base_only + cand_only + neither
    if n <= 0:
        raise ValueError("newcombe_paired_counts needs at least one pair")
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    xb, xc = both + base_only, both + cand_only
    phi = 0.0
    denominator = math.sqrt(xb * (n - xb) * xc * (n - xc))
    if denominator > 0:
        cross = float(both * neither - base_only * cand_only)
        if cross > 0:
            cross = max(cross - n / 2, 0.0)
        phi = cross / denominator
    return _newcombe_combine(xb, n, xc, n, z, phi)


def newcombe_counts(xb: int, nb: int, xc: int, nc: int, confidence: float) -> Interval:
    """Unpaired Newcombe hybrid score interval from event counts (see newcombe_difference)."""
    if nb <= 0 or nc <= 0:
        raise ValueError("newcombe_counts needs nb > 0 and nc > 0")
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    return _newcombe_combine(xb, nb, xc, nc, z, 0.0)


def _newcombe_combine(xb: int, nb: int, xc: int, nc: int, z: float, phi: float) -> Interval:
    pb, pc = xb / nb, xc / nc
    lb, ub = wilson_interval(xb, nb, z)
    lc, uc = wilson_interval(xc, nc, z)
    difference = pc - pb
    lower_sq = (pc - lc) ** 2 - 2 * phi * (pc - lc) * (ub - pb) + (ub - pb) ** 2
    upper_sq = (uc - pc) ** 2 - 2 * phi * (uc - pc) * (pb - lb) + (pb - lb) ** 2
    return (
        max(-1.0, difference - math.sqrt(max(lower_sq, 0.0))),
        min(1.0, difference + math.sqrt(max(upper_sq, 0.0))),
    )
