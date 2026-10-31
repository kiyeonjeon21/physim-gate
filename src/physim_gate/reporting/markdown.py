"""GitHub-flavored Markdown summary, sized for a pull-request comment or a job summary.

The first line is a hidden marker so a workflow can find and update its previous comment
instead of adding a new one on every push. GitHub rejects comments over 65,536 characters, so
the output degrades in steps when a comparison has many metrics: first the per-metric reasons
are dropped, then passing metrics are folded into a count.
"""

from __future__ import annotations

from collections.abc import Iterable

from physim_gate.comparison import ComparisonResult, MetricComparison, RunSummary, Verdict
from physim_gate.reporting.format import count, interval, method_label, num, signed

MARKER = "<!-- physim-gate -->"
#: Keeps short values on one line in narrow tables such as pull-request comments.
NBSP = "\u00a0"
#: GitHub's limit for comment bodies is 65,536 characters; leave room for surrounding text.
MAX_CHARS = 60_000

_ICON = {
    Verdict.PASS: "✅",
    Verdict.FAIL: "❌",
    Verdict.INCONCLUSIVE: "⚠️",
    Verdict.INVALID: "⛔",
}


def cell(text: str) -> str:
    """Escape what would break a table cell or start Markdown formatting."""
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


def sentence(text: str) -> str:
    """Capitalize the first letter, for summaries that start lowercase in terminal output."""
    return text[:1].upper() + text[1:]


def code_span(text: str) -> str:
    """An inline code span. GitHub renders its content literally, except that a pipe still
    ends a table cell unless escaped."""
    quoted = text.replace("`", "'").replace("|", "\\|").replace("\n", " ")
    return f"`{quoted}`"


def _verdict(metric: MetricComparison) -> str:
    label = f"{_ICON[metric.verdict]}{NBSP}{metric.verdict.value}"
    if metric.verdict is not Verdict.PASS and not metric.blocking:
        label += " (non-blocking)"
    return label


def _metric_name(metric: MetricComparison) -> str:
    arrow = ""
    if metric.direction is not None:
        arrow = NBSP + ("↑" if metric.direction.value == "higher_is_better" else "↓")
    unit = f" ({cell(metric.unit)})" if metric.unit else ""
    optional = " optional" if not metric.required else ""
    return f"{code_span(metric.name)}{arrow}{unit}{optional}"


def _row(metric: MetricComparison) -> str:
    n = metric.numerical
    cells = [
        _metric_name(metric),
        num(n.baseline) if n else "-",
        num(n.candidate) if n else "-",
        signed(n.degradation) if n else "-",
        interval(metric.evidence.degradation_ci).replace(", ", "," + NBSP),
        num(n.margin) if n else "-",
        count(metric).replace(" ", NBSP),
        _verdict(metric),
    ]
    return "| " + " | ".join(cells) + " |"


def _run(label: str, run: RunSummary) -> str:
    parts = [f"{run.n_trials} trials"]
    if run.n_trials_not_ok:
        parts.append(f"{run.n_trials_not_ok} not ok")
    if run.policy is not None:
        version = f"@{run.policy.version}" if run.policy.version else ""
        parts.append(f"policy {code_span(run.policy.id + version)}")
    if run.source.commit:
        parts.append(
            f"commit {code_span(run.source.commit[:12])}" + (" (dirty)" if run.source.dirty else "")
        )
    return f"**{label}** {code_span(run.run_id)}: " + ", ".join(parts)


def _simulator(run: RunSummary) -> str:
    return run.simulator.name + (f" {run.simulator.version}" if run.simulator.version else "")


def _details(metrics: Iterable[MetricComparison]) -> list[str]:
    lines: list[str] = []
    for metric in metrics:
        lines.append(f"**{cell(metric.name)}**: {_verdict(metric)}")
        method = method_label(metric)
        if method is not None:
            lines.append(f"- method: {method}")
        lines += [f"- {cell(reason)}" for reason in metric.reasons]
        lines.append("")
    return lines


def _render(
    result: ComparisonResult, *, reasons: bool, fold_passing: bool, marker: bool, heading: str
) -> str:
    sims = sorted({_simulator(result.baseline), _simulator(result.candidate)})
    lines = [
        *([MARKER] if marker else []),
        f"{heading} {_ICON[result.verdict]} PhySim Gate: {result.verdict.value}",
        "",
        cell(sentence(result.summary)),
        "",
        f"Scenario {code_span(result.candidate.scenario.id)} ({result.candidate.domain.value}), "
        f"simulator {', '.join(code_span(s) for s in sims)}.  ",
        _run("Baseline", result.baseline) + "  ",
        _run("Candidate", result.candidate),
        "",
    ]
    shown = list(result.metrics)
    folded = 0
    if fold_passing:
        shown = [m for m in result.metrics if m.verdict is not Verdict.PASS]
        folded = len(result.metrics) - len(shown)
    if shown:
        lines += [
            "| Metric | Baseline | Candidate | Degradation | Interval | Margin | n | Verdict |",
            "|---|--:|--:|--:|--:|--:|--:|---|",
            *(_row(m) for m in shown),
            "",
        ]
    if folded:
        lines += [f"{folded} passing metric(s) not listed; see the HTML report.", ""]
    legend = (
        "<sub>Degradation > 0 means worse. Interval: confidence interval of the degradation. "
        "n: matched pairs, or baseline/candidate observations"
    )
    if any(m.clusters is not None for m in result.metrics):
        legend += ", or the runs or clusters whose means were compared"
    lines.append(legend + ".</sub>")
    lines.append("")

    issues = [f"- **blocking:** {cell(i)}" for i in result.run_blocking_issues]
    issues += [f"- {cell(i)}" for i in result.run_issues]
    if issues:
        lines += ["**Run notes**", "", *issues, ""]

    explained = [m for m in result.metrics if m.verdict is not Verdict.PASS]
    if reasons and (explained or result.metrics):
        lines += [
            "<details><summary>Why each metric got its verdict</summary>",
            "",
            *_details(explained or result.metrics),
            "</details>",
            "",
        ]
    if result.untracked_metrics:
        lines += [f"Not gated: {', '.join(code_span(m) for m in result.untracked_metrics)}", ""]
    lines.append(f"<sub>physim-gate {result.tool.version}, exit code {result.exit_code}</sub>")
    return "\n".join(lines) + "\n"


def render_markdown(
    result: ComparisonResult,
    *,
    max_chars: int = MAX_CHARS,
    marker: bool = True,
    heading: str = "###",
) -> str:
    """Render the comparison as Markdown no longer than ``max_chars`` where possible.

    ``marker`` adds the hidden comment that identifies the summary; ``heading`` is the Markdown
    heading of the title (a suite nests scenario summaries one level deeper).
    """
    for reasons, fold_passing in ((True, False), (False, False), (False, True)):
        text = _render(
            result, reasons=reasons, fold_passing=fold_passing, marker=marker, heading=heading
        )
        if len(text) <= max_chars:
            break
    return text
