"""Command-line interface: ``physim-gate``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import textwrap
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from physim_gate import __version__
from physim_gate.adapters.base import (
    AdapterError,
    discover,
    get_adapter,
    load_with_adapter,
    parse_options,
)
from physim_gate.combine import RUN_LABEL, cluster_names, combine_runs
from physim_gate.comparison import (
    ComparisonResult,
    ExitCode,
    InputRef,
    Inputs,
    Verdict,
    load_comparison,
)
from physim_gate.config import GateConfig, load_config
from physim_gate.engine import compare_runs
from physim_gate.planning import (
    MetricPlan,
    Plan,
    PlanningError,
    plan_binary,
    plan_continuous,
    plan_from_comparison,
)
from physim_gate.provenance import sha256_path
from physim_gate.reporting.format import count
from physim_gate.reporting.html import render_html
from physim_gate.reporting.junit import render_junit
from physim_gate.reporting.markdown import render_markdown
from physim_gate.reporting.terminal import VERDICT_STYLE, make_console, print_comparison
from physim_gate.schema import (
    FAILURE_RATE_METRIC,
    Direction,
    InputError,
    RunResult,
    SchemaError,
    dump_json,
)
from physim_gate.store import (
    DEFAULT_STORE,
    STORE_ENV,
    is_reference,
    list_baselines,
    load_baseline,
    reference_name,
    save_baseline,
)
from physim_gate.suite import (
    ResolvedScenario,
    SuiteResult,
    load_suite,
    render_suite_junit,
    render_suite_markdown,
    run_suite,
    suite_document,
)
from physim_gate.validation import lint_run

EXIT_CODES_HELP = """\
**Exit codes**

- **0** gate passed, or the command succeeded
- **1** regression detected (compare), or invalid input (validate)
- **2** usage error, unreadable input, invalid config or result document
- **3** no regression shown, but a blocking metric is inconclusive or invalid
- **4** internal error (please report it)
"""

app = typer.Typer(
    name="physim-gate",
    help="Test physical AI like software: catch behavioral regressions in simulation results.",
    epilog=EXIT_CODES_HELP,
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="markdown",
    context_settings={"help_option_names": ["-h", "--help"]},
)

out = make_console()
err = make_console(stderr=True)


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class ReportFormat(StrEnum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class SchemaKind(StrEnum):
    RUN = "run"
    COMPARISON = "comparison"
    CONFIG = "config"


def _fail(message: str, code: ExitCode = ExitCode.USAGE) -> NoReturn:
    err.print(Text("error: ", style="bold red") + Text(message))
    raise typer.Exit(int(code))


@contextmanager
def _guard() -> Iterator[None]:
    """Map known errors to exit code 2 and anything unexpected to exit code 4."""
    try:
        yield
    except (typer.Exit, typer.Abort, SystemExit):
        raise
    except (SchemaError, AdapterError) as exc:
        _fail(str(exc))
    except Exception:
        err.print(
            Text("internal error: this is a bug in physim-gate, please report it", style="bold red")
        )
        err.print(traceback.format_exc(), markup=False)
        raise typer.Exit(int(ExitCode.INTERNAL)) from None


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"physim-gate {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """Test physical AI like software: catch behavioral regressions in simulation results."""


AdapterOpt = Annotated[
    str,
    typer.Option(
        "--adapter", "-a", help="Adapter that reads the input files. See 'physim-gate adapters'."
    ),
]
AdapterOptionsOpt = Annotated[
    list[str] | None,
    typer.Option("--adapter-option", "-O", help="Adapter option as key=value. Repeatable."),
]
FormatOpt = Annotated[OutputFormat, typer.Option("--format", "-f", help="Output format on stdout.")]
ReportFormatOpt = Annotated[
    ReportFormat, typer.Option("--format", "-f", help="Output format on stdout.")
]
MarkdownOpt = Annotated[
    Path | None,
    typer.Option("--markdown", help="Write a Markdown summary here (for PR comments)."),
]


StoreOpt = Annotated[
    Path | None,
    typer.Option(
        "--store",
        help=f"Baseline store directory. Default: ${STORE_ENV} or {DEFAULT_STORE}.",
        show_default=False,
    ),
]


def _run_comparison(
    baseline: Sequence[Path],
    candidate: Sequence[Path],
    config: Path,
    adapter: str,
    options: dict[str, str],
    store: Path | None,
) -> ComparisonResult:
    """Load both runs (the baseline possibly from the store) and compare them.

    A side given as several files is combined into one run whose trials are labeled with
    :data:`RUN_LABEL`, and every metric without ``cluster_by`` is then clustered by it.
    """
    gate = load_config(config)
    cand_files = list(candidate)
    cand_run = _load_side("candidate", adapter, cand_files, options)
    if len(baseline) == 1 and is_reference(str(baseline[0])):
        if len(cand_files) > 1:
            raise InputError(
                f"--baseline {baseline[0]} uses the store, which holds one run per baseline; "
                "give the baseline runs as files to compare several runs"
            )
        base_file, base_run = load_baseline(
            cand_run.scenario.id, reference_name(str(baseline[0])), store
        )
        base_files = [base_file]
    else:
        base_files = list(baseline)
        base_run = _load_side("baseline", adapter, base_files, options)
    for path in base_files:
        if any(path.resolve() == c.resolve() for c in cand_files):
            raise InputError(
                f"baseline and candidate are the same file ({path}); a comparison of a run "
                "with itself always passes"
            )
    if len(base_files) > 1 or len(cand_files) > 1:
        gate = gate.with_default_cluster_by(RUN_LABEL)
    inputs = Inputs(
        baseline=_input_ref(base_files),
        candidate=_input_ref(cand_files),
        config=InputRef(path=str(config), sha256=sha256_path(config)),
        adapter=adapter,
        baseline_runs=_part_refs(base_files),
        candidate_runs=_part_refs(cand_files),
    )
    return compare_runs(base_run, cand_run, gate, inputs=inputs)


def _load_side(
    side: str, adapter_name: str, paths: Sequence[Path], options: dict[str, str]
) -> RunResult:
    runs = [_load(adapter_name, path, options) for path in paths]
    if len(runs) == 1:
        return runs[0]
    return combine_runs(side, cluster_names(paths, runs), runs)


def _input_ref(paths: Sequence[Path]) -> InputRef:
    if len(paths) == 1:
        return InputRef(path=str(paths[0]), sha256=sha256_path(paths[0]))
    digests = "\n".join(sha256_path(path) for path in paths)
    return InputRef(sha256=hashlib.sha256(digests.encode()).hexdigest())


def _part_refs(paths: Sequence[Path]) -> list[InputRef]:
    if len(paths) == 1:
        return []
    return [InputRef(path=str(path), sha256=sha256_path(path)) for path in paths]


def _load(adapter_name: str, path: Path, options: dict[str, str]) -> RunResult:
    if not path.exists():
        raise InputError(f"{path}: no such file")
    return load_with_adapter(adapter_name, path, options)


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _fail(f"{path}: cannot write ({exc.strerror or exc})")


@app.command(epilog=EXIT_CODES_HELP)
def compare(
    baseline: Annotated[
        list[Path],
        typer.Option("--baseline", "-b", help="Baseline run result. Repeat it for several runs."),
    ],
    candidate: Annotated[
        list[Path],
        typer.Option("--candidate", "-c", help="Candidate run result. Repeat it for several runs."),
    ],
    config: Annotated[Path, typer.Option("--config", "-g", help="Gate config (gate.yaml).")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the comparison result JSON here.")
    ] = None,
    junit: Annotated[Path | None, typer.Option("--junit", help="Write JUnit XML here.")] = None,
    html: Annotated[Path | None, typer.Option("--html", help="Write the HTML report here.")] = None,
    markdown: MarkdownOpt = None,
    adapter: AdapterOpt = "json",
    adapter_option: AdapterOptionsOpt = None,
    output_format: ReportFormatOpt = ReportFormat.TEXT,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Explain passing metrics too.")
    ] = False,
    store: StoreOpt = None,
) -> None:
    """Compare a candidate run against a baseline and gate on regressions.

    --baseline @NAME uses the stored baseline NAME of the candidate's scenario
    (see 'physim-gate baseline').

    Several --baseline or --candidate runs (one per training seed, say) are compared run by
    run: each run is a cluster, and the interval comes from the spread of the run means
    (cluster_by: physim.run, unless the gate config sets cluster_by).
    """
    with _guard():
        options = parse_options(adapter_option or [])
        result = _run_comparison(baseline, candidate, config, adapter, options, store)
        _emit(
            result,
            output=output,
            junit=junit,
            html=html,
            markdown=markdown,
            output_format=output_format,
            verbose=verbose,
        )
    raise typer.Exit(result.exit_code)


def _emit(
    result: ComparisonResult,
    *,
    output: Path | None,
    junit: Path | None,
    html: Path | None,
    markdown: Path | None,
    output_format: ReportFormat,
    verbose: bool,
) -> None:
    if output is not None:
        _write(output, dump_json(result))
    if junit is not None:
        _write(junit, render_junit(result))
    if html is not None:
        _write(html, render_html(result))
    if markdown is not None:
        _write(markdown, render_markdown(result))
    with _tolerate_closed_stdout():
        if output_format is ReportFormat.JSON:
            typer.echo(dump_json(result), nl=False)
            return
        if output_format is ReportFormat.MARKDOWN:
            typer.echo(render_markdown(result), nl=False)
            return
        print_comparison(result, out, verbose=verbose)
        written = [str(p) for p in (output, junit, html, markdown) if p is not None]
        if written:
            # Never break a path across lines: it could not be copied from the terminal.
            out.print(Text(f"wrote {', '.join(written)}", style="dim"), soft_wrap=True)


@contextmanager
def _tolerate_closed_stdout() -> Iterator[None]:
    """Keep the verdict's exit code when stdout is closed early (``... | head``).

    Without this, the BrokenPipeError would turn any verdict into exit code 1, which means
    "regression". Further writes go to /dev/null so the interpreter's final flush stays quiet.
    """
    try:
        yield
        sys.stdout.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)


@app.command(epilog=EXIT_CODES_HELP)
def report(
    comparison: Annotated[
        Path, typer.Argument(help="A comparison result written by 'compare --output'.")
    ],
    html: Annotated[Path | None, typer.Option("--html", help="Write the HTML report here.")] = None,
    junit: Annotated[Path | None, typer.Option("--junit", help="Write JUnit XML here.")] = None,
    markdown: MarkdownOpt = None,
    output_format: ReportFormatOpt = ReportFormat.TEXT,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Explain passing metrics too.")
    ] = False,
) -> None:
    """Render a saved comparison result. Exits with the comparison's original exit code."""
    with _guard():
        result = load_comparison(comparison)
        _emit(
            result,
            output=None,
            junit=junit,
            html=html,
            markdown=markdown,
            output_format=output_format,
            verbose=verbose,
        )
    raise typer.Exit(result.exit_code)


@app.command(epilog=EXIT_CODES_HELP)
def validate(
    results: Annotated[Path, typer.Argument(help="Run result file to check.")],
    adapter: AdapterOpt = "json",
    adapter_option: AdapterOptionsOpt = None,
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
    output_format: FormatOpt = OutputFormat.TEXT,
) -> None:
    """Check that a run result is valid and flag suspicious content."""
    errors: list[str] = []
    warnings: list[dict[str, str]] = []
    run: RunResult | None = None
    with _guard():
        options = parse_options(adapter_option or [])
        if not results.exists():
            _fail(f"{results}: no such file")
        try:
            run = load_with_adapter(adapter, results, options)
        except InputError:
            raise
        except SchemaError as exc:
            errors = [exc.message, *exc.problems]
        if run is not None:
            warnings = [i.model_dump() for i in lint_run(run)]

    valid = not errors and not (strict and warnings)
    if output_format is OutputFormat.JSON:
        typer.echo(
            json.dumps(
                {
                    "path": str(results),
                    "valid": valid,
                    "errors": errors,
                    "warnings": warnings,
                    "run_id": run.run_id if run else None,
                    "trials": len(run.trials) if run else None,
                    "metrics": sorted(run.metrics) if run else [],
                },
                indent=2,
            )
        )
    else:
        if errors:
            err.print(Text("invalid: ", style="bold red") + Text(errors[0]))
            for problem in errors[1:]:
                err.print(f"  - {escape(problem)}")
        elif run is not None:
            out.print(
                Text("valid: ", style="bold green")
                + Text(
                    f"{results} (run {run.run_id}, {len(run.trials)} trials, "
                    f"metrics: {', '.join(run.metrics) or 'none'})"
                )
            )
        for w in warnings:
            out.print(Text("warning: ", style="yellow") + Text(f"{w['location']}: {w['message']}"))
    raise typer.Exit(int(ExitCode.OK if valid else ExitCode.REGRESSION))


_INIT_HEADER = """\
# PhySim Gate configuration. Reference: docs/comparison-semantics.md
#
# For each gated metric, set how much degradation is acceptable:
#   max_absolute_degradation: in the metric's own unit
#   max_relative_degradation: as a fraction of the baseline value (0.05 = 5%)
# If both are set, the stricter one applies. "Degradation" already accounts for
# direction (higher_is_better / lower_is_better), so margins are always >= 0.
#
# A metric FAILs only when the whole confidence interval of the degradation lies
# beyond the margin, and PASSes only when it lies entirely within the margin.
# Anything in between is INCONCLUSIVE, which blocks by default (policy.on_inconclusive).
"""


def _init_document(run: RunResult | None) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    if run is None:
        metrics = {
            "success": {"max_absolute_degradation": 0.05},
            "episode_time": {"max_relative_degradation": 0.10},
            FAILURE_RATE_METRIC: {"max_absolute_degradation": 0.02},
        }
    else:
        for name, spec in run.metrics.items():
            values = [v for v in run.metric_values(name).values() if v is not None]
            binary = bool(values) and all(v in (0.0, 1.0) for v in values)
            mean = sum(values) / len(values) if values else 0.0
            if binary or mean == 0:
                metrics[name] = {"max_absolute_degradation": 0.05 if binary else 0.0}
            else:
                metrics[name] = {"max_relative_degradation": 0.05}
            metrics[name]["direction"] = spec.direction.value
        metrics[FAILURE_RATE_METRIC] = {
            "max_absolute_degradation": 0.02,
            "direction": Direction.LOWER_IS_BETTER.value,
        }
    document: dict[str, Any] = {
        "version": 1,
        "defaults": {"confidence": 0.95, "min_samples": 10, "pairing": "auto", "statistic": "mean"},
        "policy": {"on_inconclusive": "fail", "on_invalid": "fail"},
        "metrics": metrics,
    }
    GateConfig.model_validate(document)
    return document


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path("gate.yaml"),
    from_results: Annotated[
        Path | None,
        typer.Option("--from", help="Derive the metric list from an existing run result."),
    ] = None,
    adapter: AdapterOpt = "json",
    adapter_option: AdapterOptionsOpt = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter gate.yaml. Review every margin before relying on it."""
    with _guard():
        if path.exists() and not force:
            _fail(f"{path} already exists (use --force to overwrite)")
        options = parse_options(adapter_option or [])
        run = _load(adapter, from_results, options) if from_results is not None else None
        document = _init_document(run)
        body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
        note = (
            f"# Metrics were taken from {from_results}. The margins are placeholders.\n\n"
            if from_results
            else "# The metric names below are examples. Replace them with your own.\n\n"
        )
        _write(path, _INIT_HEADER + note + body)
    out.print(
        Text("wrote ", style="green")
        + Text(str(path))
        + Text(" - review each margin before gating on it", style="dim")
    )


def _percent(value: float) -> str:
    if 0 < value < 0.001:
        return "<0.1%"
    if 0.999 < value < 1:
        return ">99.9%"
    return f"{value * 100:.1f}%"


def _print_plan(plan: Plan, rate: float | None, direction: Direction) -> None:
    rows: list[tuple[str, str]] = []
    if plan.kind == "binary" and rate is not None:
        bad = direction is Direction.LOWER_IS_BETTER
        candidate = rate + plan.true_degradation if bad else rate - plan.true_degradation
        what = "0/1 metric, " + ("lower is better" if bad else "higher is better")
        rows += [("baseline rate", f"{rate:g}"), ("candidate rate", f"{candidate:g}")]
    else:
        what = "continuous metric"
    rows += [
        ("true degradation", f"{plan.true_degradation:g}"),
        ("margin", f"{plan.margin:g}"),
        (
            "goal",
            f"{plan.goal.value.upper()} with probability >= {plan.power:.0%}, "
            f"{plan.confidence:.0%} intervals",
        ),
        ("method", plan.method),
    ]
    out.print(Text(f"plan for a {what}", style="bold"))
    for label, value in rows:
        out.print(Text(f"  {label:<18}", style="dim") + Text(value))
    out.print()
    per = "trials per run" if plan.unit == "trials" else f"{plan.unit} per side"
    if plan.trials is None:
        out.print(
            Text(f"  more than {plan.max_trials:,} {per}", style="bold yellow")
            + Text(" - widen the margin or accept INCONCLUSIVE verdicts", style="dim")
        )
    else:
        out.print(Text(f"  {plan.trials:,} {per}", style="bold"))
    if plan.probabilities is not None:
        p = plan.probabilities
        line = Text("  verdicts at that size: ", style="dim")
        line.append(f"PASS {_percent(p.pass_)}", style=VERDICT_STYLE[Verdict.PASS])
        line.append("  ")
        line.append(
            f"INCONCLUSIVE {_percent(p.inconclusive)}", style=VERDICT_STYLE[Verdict.INCONCLUSIVE]
        )
        line.append("  ")
        line.append(f"FAIL {_percent(p.fail)}", style=VERDICT_STYLE[Verdict.FAIL])
        out.print(line)
    out.print()
    for note in plan.notes:
        out.print(Text(f"  note: {note}", style="dim"), soft_wrap=True)


def _plan_document(result: Plan) -> dict[str, Any]:
    document = dataclasses.asdict(result)
    if result.probabilities is not None:
        document["probabilities"] = {
            "pass": result.probabilities.pass_,
            "inconclusive": result.probabilities.inconclusive,
            "fail": result.probabilities.fail,
        }
    return document


def _print_pilot_plans(source: Path, comparison: ComparisonResult, plans: list[MetricPlan]) -> None:
    out.print(
        Text(f"plan from {source}", style="bold")
        + Text(
            f"  pilot: {comparison.baseline.n_trials} baseline and "
            f"{comparison.candidate.n_trials} candidate trials",
            style="dim",
        ),
        soft_wrap=True,
    )
    table = Table(show_edge=False, pad_edge=False, header_style="bold")
    clustered = any(item.plan is not None and item.plan.unit != "trials" for item in plans)
    needed = "trials or runs per side" if clustered else "trials per run"
    for column in ("metric", "pilot n", "degradation", "margin", "goal", needed):
        table.add_column(column, justify="left" if column in ("metric", "goal") else "right")
    table.add_column("P(goal)", justify="right")
    notes: list[str] = []
    by_name = {m.name: m for m in comparison.metrics}
    for item in plans:
        metric = by_name[item.metric]
        if item.plan is None:
            table.add_row(item.metric, "-", "-", "-", Text("-", style="dim"), "-", "-")
            notes.append(f"{item.metric}: {item.reason}")
            continue
        plan = item.plan
        if metric.pairing is not None and metric.pairing.mode == "paired":
            pilot = f"{metric.pairing.n_pairs} pairs"
        else:
            pilot = count(metric)
        if plan.trials is not None:
            size = f"{plan.trials:,}" + ("" if plan.unit == "trials" else f" {plan.unit}")
        else:
            size = f">{plan.max_trials:,}"
        goal_verdict = Verdict.PASS if plan.goal.value == "pass" else Verdict.FAIL
        table.add_row(
            item.metric,
            pilot,
            f"{plan.true_degradation:+.4g}",
            f"{plan.margin:.4g}",
            Text(goal_verdict.value, style=VERDICT_STYLE[goal_verdict]),
            Text(size, style="bold"),
            _percent(plan.probabilities.of(plan.goal)) if plan.probabilities else "-",
        )
        notes.append(f"{item.metric}: {'; '.join([plan.method, *plan.notes])}")
    out.print(table)
    legend = (
        "degradation = assumed true degradation (the pilot's observed value unless "
        "--true-degradation is given); goal = the verdict that degradation should get"
    )
    # Wrap here: rich would leave a trailing space at each break.
    for line in textwrap.wrap(legend, out.width):
        out.print(Text(line, style="dim"), soft_wrap=True)
    if notes:
        out.print()
        for note in notes:
            out.print(Text(f"  {note}", style="dim"), soft_wrap=True)


@app.command(epilog=EXIT_CODES_HELP)
def plan(
    margin: Annotated[
        float | None,
        typer.Option(
            "--margin", "-m", help="Allowed degradation, in the metric's unit (absolute)."
        ),
    ] = None,
    rate: Annotated[
        float | None,
        typer.Option("--rate", "-r", help="0/1 metric: expected baseline rate of 1s (0-1)."),
    ] = None,
    stdev: Annotated[
        float | None,
        typer.Option(
            "--stdev",
            "-s",
            help="Continuous metric: standard deviation of one trial's value "
            "(of the paired differences with --paired).",
        ),
    ] = None,
    pilot: Annotated[
        Path | None,
        typer.Option(
            "--from",
            help="Plan every gated metric of a pilot comparison (comparison.json) from its data.",
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option("--metric", help="With --from: only this metric. Repeatable."),
    ] = None,
    direction: Annotated[
        Direction,
        typer.Option(
            "--direction",
            "-d",
            metavar="DIRECTION",
            help="0/1 metric: higher_is_better (a success rate) or lower_is_better "
            "(a collision rate).",
        ),
    ] = Direction.HIGHER_IS_BETTER,
    true_degradation: Annotated[
        float | None,
        typer.Option(
            "--true-degradation",
            "-t",
            help="Degradation you expect the candidate to really have. Below the margin the "
            "goal is PASS, above it FAIL. Default: 0, or the pilot's observed degradation.",
        ),
    ] = None,
    paired: Annotated[
        bool, typer.Option("--paired", help="Continuous metric: runs share trial conditions.")
    ] = False,
    discordance: Annotated[
        float | None,
        typer.Option(
            "--discordance",
            help="0/1 metric, paired runs: share of trial conditions on which baseline and "
            "candidate disagree. Independent runs with rate p disagree on 2p(1-p).",
        ),
    ] = None,
    confidence: Annotated[float, typer.Option("--confidence", help="Interval level.")] = 0.95,
    power: Annotated[
        float, typer.Option("--power", help="Wanted probability of the goal verdict.")
    ] = 0.8,
    min_samples: Annotated[int, typer.Option("--min-samples", help="The gate's min_samples.")] = 10,
    output_format: FormatOpt = OutputFormat.TEXT,
) -> None:
    """Estimate how many trials per run a metric needs to reach a verdict."""
    if pilot is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--margin", margin),
                ("--rate", rate),
                ("--stdev", stdev),
                ("--discordance", discordance),
            )
            if value is not None
        ] + (["--paired"] if paired else [])
        if conflicting:
            _fail(
                f"{', '.join(conflicting)} cannot be combined with --from; the pilot "
                "comparison provides them"
            )
        with _guard():
            comparison = load_comparison(pilot)
            try:
                plans = plan_from_comparison(
                    comparison, metrics=metric, true_degradation=true_degradation, power=power
                )
            except PlanningError as exc:
                _fail(str(exc))
        if output_format is OutputFormat.JSON:
            typer.echo(
                json.dumps(
                    [
                        {
                            "metric": item.metric,
                            "plan": _plan_document(item.plan) if item.plan else None,
                            "reason": item.reason,
                        }
                        for item in plans
                    ],
                    indent=2,
                )
            )
            return
        _print_pilot_plans(pilot, comparison, plans)
        return

    if metric:
        _fail("--metric applies to --from only")
    if margin is None:
        _fail("give --margin (or plan from a pilot comparison with --from)")
    if (rate is None) == (stdev is None):
        _fail("give exactly one of --rate (0/1 metric) or --stdev (continuous metric)")
    degradation = 0.0 if true_degradation is None else true_degradation
    try:
        if rate is not None:
            if paired and discordance is None:
                _fail(
                    "paired 0/1 runs need --discordance, the share of trial conditions on "
                    "which the runs disagree"
                )
            result = plan_binary(
                rate,
                margin,
                direction=direction,
                true_degradation=degradation,
                confidence=confidence,
                power=power,
                min_samples=min_samples,
                discordance=discordance,
            )
        else:
            assert stdev is not None
            if discordance is not None:
                _fail("--discordance applies to 0/1 metrics (--rate) only")
            result = plan_continuous(
                stdev,
                margin,
                paired=paired,
                true_degradation=degradation,
                confidence=confidence,
                power=power,
                min_samples=min_samples,
            )
    except PlanningError as exc:
        _fail(str(exc))
    if output_format is OutputFormat.JSON:
        typer.echo(json.dumps(_plan_document(result), indent=2))
        return
    _print_plan(result, rate, direction)


@app.command()
def demo(
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Directory for generated runs and reports.")
    ] = Path("physim-gate-demo"),
) -> None:
    """Generate synthetic runs and gate them with the real engine (no simulator needed)."""
    from physim_gate.demo import DemoOutcome, run_demo

    def show(outcome: DemoOutcome) -> None:
        scenario = outcome.scenario
        out.rule(Text(f"{scenario.name}: {scenario.title}", style="bold"), align="left")
        print_comparison(outcome.result, out)
        out.print(
            Text(
                f"reports: {outcome.directory}/report.html, junit.xml, comparison.json", style="dim"
            )
        )
        out.print()

    with _guard():
        outcomes = run_demo(output_dir, on_result=show)

    out.rule(Text("demo summary", style="bold"), align="left")
    for outcome in outcomes:
        mark = (
            "as expected"
            if outcome.as_expected
            else f"UNEXPECTED (expected {outcome.scenario.expected.value})"
        )
        line = Text(f"  {outcome.scenario.name:<24}")
        line.append(
            f"{outcome.result.verdict.value:<13}", style=VERDICT_STYLE[outcome.result.verdict]
        )
        line.append(
            f"exit {outcome.result.exit_code}  {mark}",
            style="dim" if outcome.as_expected else "bold red",
        )
        out.print(line)
    out.print()
    out.print(f"Open {output_dir / 'regression' / 'report.html'} to see a failing report.")
    out.print(
        "Re-run any comparison: physim-gate compare "
        f"-b {output_dir}/regression/baseline.json -c {output_dir}/regression/candidate.json "
        f"-g {output_dir}/regression/gate.yaml"
    )
    if not all(o.as_expected for o in outcomes):
        _fail("a demo scenario produced an unexpected verdict", ExitCode.INTERNAL)


@app.command(epilog=EXIT_CODES_HELP)
def suite(
    suite_file: Annotated[Path, typer.Argument(help="Suite file (suite.yaml).")],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Reports: one directory per scenario, plus suite-wide files."
        ),
    ] = Path("physim-gate-suite"),
    output_format: ReportFormatOpt = ReportFormat.TEXT,
    store: StoreOpt = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every scenario's full table.")
    ] = False,
) -> None:
    """Gate several scenarios at once, with one aggregated verdict and exit code.

    Writes comparison.json, junit.xml, report.html and summary.md per scenario under
    OUTPUT_DIR/<scenario>/, and suite.json, junit.xml (one test suite per scenario) and
    summary.md for the whole suite. Exit code: 2 if any scenario could not be compared, else 1
    if any regressed, else 3 if any is blocked, else 0.
    """
    with _guard():
        scenarios = load_suite(suite_file)

        def compare_scenario(scenario: ResolvedScenario) -> ComparisonResult:
            return _run_comparison(
                [scenario.baseline],
                [scenario.candidate],
                scenario.config,
                scenario.adapter,
                scenario.adapter_options,
                store,
            )

        result = run_suite(scenarios, compare_scenario)
        for outcome in result.scenarios:
            if outcome.result is None:
                continue
            directory = output_dir / outcome.name
            _write(directory / "comparison.json", dump_json(outcome.result))
            _write(directory / "junit.xml", render_junit(outcome.result))
            _write(directory / "report.html", render_html(outcome.result))
            _write(directory / "summary.md", render_markdown(outcome.result))
        _write(output_dir / "suite.json", json.dumps(suite_document(result), indent=2) + "\n")
        _write(output_dir / "junit.xml", render_suite_junit(result))
        _write(output_dir / "summary.md", render_suite_markdown(result))

    with _tolerate_closed_stdout():
        if output_format is ReportFormat.JSON:
            typer.echo(json.dumps(suite_document(result), indent=2))
        elif output_format is ReportFormat.MARKDOWN:
            typer.echo(render_suite_markdown(result), nl=False)
        else:
            _print_suite(result, verbose=verbose)
            out.print(Text(f"wrote {output_dir}/", style="dim"), soft_wrap=True)
    raise typer.Exit(result.exit_code)


def _print_suite(result: SuiteResult, *, verbose: bool) -> None:
    for outcome in result.scenarios:
        if outcome.result is not None and (verbose or outcome.verdict is not Verdict.PASS):
            out.rule(Text(outcome.name, style="bold"), align="left")
            print_comparison(outcome.result, out, verbose=verbose)
            out.print()
        elif outcome.error is not None:
            out.rule(Text(outcome.name, style="bold"), align="left")
            out.print(Text("error: ", style="bold red") + Text(outcome.error), soft_wrap=True)
            out.print()
    out.rule(Text("suite", style="bold"), align="left")
    table = Table(show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("scenario", no_wrap=True)
    table.add_column("verdict", no_wrap=True)
    table.add_column("exit", justify="right")
    table.add_column("summary")
    for outcome in result.scenarios:
        if outcome.verdict is not None:
            verdict = Text(outcome.verdict.value, style=VERDICT_STYLE[outcome.verdict])
        else:
            verdict = Text("ERROR", style="bold red")
        table.add_row(outcome.name, verdict, str(outcome.exit_code), outcome.summary)
    out.print(table)
    out.print(Text(result.summary, style="bold"))


baseline_app = typer.Typer(
    help="Store baselines as files, keyed by scenario and name, and compare against them "
    "with --baseline @NAME.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("save")
def baseline_save(
    source: Annotated[Path, typer.Argument(help="Run result to store (any adapter's input).")],
    name: Annotated[str, typer.Option("--name", "-n", help="Baseline name, e.g. a branch.")] = (
        "main"
    ),
    adapter: AdapterOpt = "json",
    adapter_option: AdapterOptionsOpt = None,
    store: StoreOpt = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing baseline.")] = False,
) -> None:
    """Normalize a run result and store it as a baseline of its scenario."""
    with _guard():
        run = _load(adapter, source, parse_options(adapter_option or []))
        path = save_baseline(run, name, source=source, store=store, force=force)
    out.print(
        Text("saved ", style="green")
        + Text(f"{run.scenario.id} @{name}", style="bold")
        + Text(f" ({run.run_id}, {len(run.trials)} trials) to {path}", style="dim"),
        soft_wrap=True,
    )


@baseline_app.command("list")
def baseline_list(store: StoreOpt = None, output_format: FormatOpt = OutputFormat.TEXT) -> None:
    """List stored baselines."""
    with _guard():
        found = list_baselines(store)
    if output_format is OutputFormat.JSON:
        typer.echo(
            json.dumps(
                [
                    {
                        "scenario": b.scenario,
                        "name": b.name,
                        "path": str(b.path),
                        "run_id": b.run.run_id if b.run else None,
                        "trials": len(b.run.trials) if b.run else None,
                        "commit": b.run.source.commit if b.run else None,
                        "error": b.error,
                    }
                    for b in found
                ],
                indent=2,
            )
        )
        return
    if not found:
        out.print(Text(f"no baselines in {store or DEFAULT_STORE}", style="dim"))
        return
    table = Table(show_edge=False, pad_edge=False, header_style="bold")
    for column in ("scenario", "name", "run", "trials", "commit", "created"):
        table.add_column(column, justify="right" if column == "trials" else "left")
    for b in found:
        if b.run is None:
            table.add_row(b.scenario, b.name, Text(f"unreadable: {b.error}", style="red"))
            continue
        created = b.run.created_at.date().isoformat() if b.run.created_at else "-"
        table.add_row(
            b.scenario,
            Text(f"@{b.name}", style="bold"),
            b.run.run_id,
            str(len(b.run.trials)),
            (b.run.source.commit or "-")[:12],
            created,
        )
    out.print(table)


@baseline_app.command("path")
def baseline_path_command(
    scenario: Annotated[str, typer.Argument(help="Scenario id.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Baseline name.")] = "main",
    store: StoreOpt = None,
) -> None:
    """Print the file of a stored baseline (for scripts)."""
    with _guard():
        path, _ = load_baseline(scenario, name, store)
    typer.echo(str(path))


@app.command()
def adapters(output_format: FormatOpt = OutputFormat.TEXT) -> None:
    """List available adapters, including third-party plugins."""
    with _guard():
        found = discover()
    if output_format is OutputFormat.JSON:
        typer.echo(
            json.dumps(
                [
                    {"name": i.name, "target": i.target, "origin": i.origin, "error": i.error}
                    for i in found.values()
                ],
                indent=2,
            )
        )
        return
    for info in found.values():
        line = Text(f"{info.name:<12}", style="bold")
        if info.error:
            line.append(f"{info.target} ({info.origin}): {info.error}", style="red")
        else:
            description = _describe(info.name)
            line.append(f"{description}  ", style="")
            line.append(f"[{info.origin}]", style="dim")
        out.print(line)


def _describe(name: str) -> str:
    try:
        return get_adapter(name).description
    except AdapterError as exc:
        return f"unavailable: {exc}"


_SCHEMAS: dict[SchemaKind, Callable[[], dict[str, Any]]] = {
    SchemaKind.RUN: RunResult.model_json_schema,
    SchemaKind.COMPARISON: ComparisonResult.model_json_schema,
    SchemaKind.CONFIG: GateConfig.model_json_schema,
}


@app.command()
def schema(
    kind: Annotated[SchemaKind, typer.Argument(help="Which document.")] = SchemaKind.RUN,
) -> None:
    """Print the JSON Schema of a document type."""
    typer.echo(json.dumps(_SCHEMAS[kind](), indent=2))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
