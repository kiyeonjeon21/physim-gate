"""Human-readable terminal summary (rich). Honors ``NO_COLOR``."""

from __future__ import annotations

import os
import sys
import textwrap

from rich.console import Console
from rich.measure import Measurement
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from physim_gate.comparison import ComparisonResult, MetricComparison, RunSummary, Verdict
from physim_gate.reporting.format import count, interval, num, signed, unit_suffix

VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.INCONCLUSIVE: "bold yellow",
    Verdict.INVALID: "bold magenta",
}


class _Console(Console):
    def on_broken_pipe(self) -> None:
        """Stop writing when the reader went away (``... | head``) without exiting.

        Rich's default raises ``SystemExit(1)``, which would replace the verdict's exit code
        with 1 ("regression"). The process must keep the exit code of the actual result.
        """
        self.quiet = True
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, (sys.stderr if self.stderr else sys.stdout).fileno())
        os.close(devnull)


def make_console(*, stderr: bool = False) -> Console:
    """A console that does not squeeze tables into 80 columns in CI logs.

    Rich falls back to 80 columns when output is not a terminal. CI log viewers are wider, so
    use 120 there unless ``COLUMNS`` says otherwise.
    """
    stream = sys.stderr if stderr else sys.stdout
    interactive = stream.isatty()
    width = None if interactive or "COLUMNS" in os.environ else 120
    return _Console(stderr=stderr, highlight=False, width=width, soft_wrap=stderr)


def _run_line(label: str, run: RunSummary) -> Text:
    text = Text(f"{label:<10}", style="dim")
    text.append(run.run_id, style="bold")
    parts = [f"{run.n_trials} trials"]
    if run.n_trials_not_ok:
        parts.append(f"{run.n_trials_not_ok} not ok")
    if run.policy is not None:
        version = f"@{run.policy.version}" if run.policy.version else ""
        parts.append(f"policy {run.policy.id}{version}")
    if run.source.commit:
        parts.append(f"commit {run.source.commit[:12]}" + ("+dirty" if run.source.dirty else ""))
    text.append("  " + " | ".join(parts), style="dim")
    return text


def _simulator(run: RunSummary) -> str:
    return run.simulator.name + (f" {run.simulator.version}" if run.simulator.version else "")


def _name(metric: MetricComparison) -> Text:
    arrow = ""
    if metric.direction is not None:
        arrow = " ↑" if metric.direction.value == "higher_is_better" else " ↓"
    name = Text(metric.name + arrow)
    if not metric.required:
        name.append(" (optional)", style="dim")
    return name


def _verdict(metric: MetricComparison) -> Text:
    emphasized = metric.blocking or metric.verdict is Verdict.PASS
    text = Text(metric.verdict.value, style=VERDICT_STYLE[metric.verdict] if emphasized else "dim")
    if not emphasized:
        text.append(" (non-blocking)", style="dim")
    return text


def _table(result: ComparisonResult) -> Table:
    table = Table(show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("metric", no_wrap=True)
    table.add_column("baseline", justify="right", no_wrap=True)
    table.add_column("candidate", justify="right", no_wrap=True)
    table.add_column("degradation", justify="right", no_wrap=True)
    table.add_column("interval", justify="right", no_wrap=True)
    table.add_column("margin", justify="right", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("verdict", no_wrap=True)
    for m in result.metrics:
        n = m.numerical
        table.add_row(
            _name(m),
            num(n.baseline) if n else "-",
            num(n.candidate) if n else "-",
            signed(n.degradation) if n else "-",
            interval(m.evidence.degradation_ci),
            num(n.margin) if n else "-",
            count(m),
            _verdict(m),
        )
    return table


def _blocks(console: Console, result: ComparisonResult) -> None:
    for m in result.metrics:
        line = _verdict(m)
        line.append("  ")
        line.append_text(_name(m))
        console.print(line)
        n = m.numerical
        if n is not None:
            u = unit_suffix(m)
            console.print(
                Padding(
                    Text(
                        f"{num(n.baseline)} → {num(n.candidate)}{u}, degradation "
                        f"{signed(n.degradation)}{u}, margin {num(n.margin)}{u}",
                        style="dim",
                    ),
                    (0, 0, 0, 2),
                    expand=False,
                )
            )
        if m.evidence.degradation_ci is not None:
            console.print(
                Padding(
                    Text(
                        f"interval {interval(m.evidence.degradation_ci)}, n {count(m)}",
                        style="dim",
                    ),
                    (0, 0, 0, 2),
                    expand=False,
                )
            )


def print_comparison(
    result: ComparisonResult, console: Console | None = None, *, verbose: bool = False
) -> None:
    console = console or make_console()
    prefix = Text("PhySim Gate  ")
    prefix.append(f" {result.verdict.value} ", style="reverse " + VERDICT_STYLE[result.verdict])
    prefix.append("  ")
    if console.width - len(prefix) >= 40:
        # Wrapped summary lines hang under the summary's first word.
        lines = _wrap(result.summary, console.width - len(prefix)) or [""]
        console.print(prefix + Text(lines[0]), soft_wrap=True)
        indent = " " * len(prefix)
    else:
        # Too narrow for a hanging indent: the summary starts on its own line.
        lines = ["", *_wrap(result.summary, console.width)]
        prefix.rstrip()
        console.print(prefix, soft_wrap=True)
        indent = ""
    for line in lines[1:]:
        console.print(Text(indent + line), soft_wrap=True)
    scenario = result.candidate.scenario
    sims = {_simulator(result.baseline), _simulator(result.candidate)}
    context = (
        f"scenario  {scenario.id} ({result.candidate.domain.value}) | "
        f"simulator {' vs '.join(sorted(sims))}"
    )
    for line in _wrap(context, console.width):
        console.print(Text(line, style="dim"), soft_wrap=True)
    console.print(_run_line("baseline", result.baseline))
    console.print(_run_line("candidate", result.candidate))
    console.print()

    table = _table(result)
    # Measure without the console's width limit: a table that would need truncating is replaced
    # by one block per metric, so no number or verdict is ever cut off.
    natural = Measurement.get(console, console.options.update_width(10_000), table).maximum
    if natural <= console.width:
        console.print(table)
        legend = (
            "degradation > 0 means worse; interval = confidence interval of the degradation; "
            "n = matched pairs or baseline/candidate observations"
        )
        if any(m.clusters is not None for m in result.metrics):
            legend += ", or the runs or clusters whose means were compared"
        for line in _wrap(legend, console.width):
            console.print(Text(line, style="dim"), soft_wrap=True)
    else:
        _blocks(console, result)

    shown = [m for m in result.metrics if verbose or m.verdict is not Verdict.PASS]
    if shown:
        console.print()
        for m in shown:
            console.print(Text(f"{m.name}: ", style="bold").append_text(_verdict(m)))
            for reason in m.reasons:
                _bullet(console, reason)
    if result.run_blocking_issues or result.run_issues:
        console.print()
        console.print(Text("run notes", style="bold"))
        for issue in result.run_blocking_issues:
            _bullet(console, f"[blocking] {issue}", style="magenta")
        for issue in result.run_issues:
            _bullet(console, issue)
    if result.untracked_metrics:
        console.print(Text(f"not gated: {', '.join(result.untracked_metrics)}", style="dim"))


def _wrap(text: str, width: int) -> list[str]:
    """Wrap at spaces only, so paths and identifiers stay whole even if a line overflows."""
    return textwrap.wrap(text, width=max(width, 20), break_long_words=False, break_on_hyphens=False)


def _bullet(console: Console, text: str, style: str = "") -> None:
    """``  - text`` with wrapped lines indented under the text, without trailing padding."""
    for i, line in enumerate(_wrap(text, console.width - 4)):
        console.print(Text(("  - " if i == 0 else "    ") + line, style=style), soft_wrap=True)
