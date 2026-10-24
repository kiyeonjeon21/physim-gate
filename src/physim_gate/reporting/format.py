"""Number formatting shared by the reporters."""

from __future__ import annotations

import math

from physim_gate.comparison import MetricComparison


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}g}"


def signed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    text = num(value, digits)
    return text if value <= 0 or text.startswith("+") else f"+{text}"


def interval(bounds: tuple[float, float] | None, digits: int = 4) -> str:
    """Both bounds in one notation and at one precision, so they read as a pair.

    The precision follows the larger bound, to ``digits`` significant digits; trailing zeros
    are dropped only where both bounds have them.
    """
    if bounds is None:
        return "-"
    low, high = bounds
    largest = max(abs(low), abs(high))
    if largest == 0 or largest >= 1e5 or largest < 1e-3:
        return f"[{signed(low, digits)}, {signed(high, digits)}]"
    decimals = max(0, digits - 1 - math.floor(math.log10(largest)))
    texts = [f"{value:+.{decimals}f}" if value != 0 else "0" for value in (low, high)]
    while decimals > 0 and all(t == "0" or t.endswith("0") for t in texts):
        decimals -= 1
        texts = [t if t == "0" else t[:-1] for t in texts]
    if decimals == 0:
        texts = [t.rstrip(".") for t in texts]
    return f"[{texts[0]}, {texts[1]}]"


def unit_suffix(metric: MetricComparison) -> str:
    return f" {metric.unit}" if metric.unit else ""


def direction_label(metric: MetricComparison) -> str:
    if metric.direction is None:
        return "-"
    return "higher is better" if metric.direction.value == "higher_is_better" else "lower is better"


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:+.1f}%"


_METHOD_LABELS = {
    "paired_bootstrap": "paired bootstrap",
    "unpaired_bootstrap": "unpaired bootstrap",
    "paired_bootstrap_expanded": "paired bootstrap, expanded percentile",
    "unpaired_bootstrap_expanded": "unpaired bootstrap, expanded percentile",
    "paired_newcombe": "paired score interval (Newcombe)",
    "unpaired_newcombe": "unpaired score interval (Newcombe)",
}


def method_label(metric: MetricComparison) -> str | None:
    """Human-readable name of the interval method, with the resample count for bootstraps."""
    method = metric.evidence.method
    if method is None:
        return None
    label = _METHOD_LABELS.get(method, method.replace("_", " "))
    if metric.evidence.resamples is not None:
        label += f", {metric.evidence.resamples} resamples"
    return label


def interval_kind(metric: MetricComparison) -> str:
    method = metric.evidence.method or ""
    return "score interval" if method.endswith("newcombe") else "bootstrap interval"
