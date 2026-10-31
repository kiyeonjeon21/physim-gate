"""PhySim Gate: simulator-agnostic regression gates for physical AI.

Typical library use::

    from physim_gate import compare_runs, load_config, load_run

    result = compare_runs(load_run(baseline), load_run(candidate), load_config(gate))
    print(result.verdict, result.summary)
"""

from __future__ import annotations

__version__ = "0.4.0"

from physim_gate.combine import RUN_LABEL, combine_runs
from physim_gate.comparison import (
    ComparisonResult,
    Evidence,
    ExitCode,
    MetricComparison,
    Verdict,
    load_comparison,
)
from physim_gate.config import GateConfig, MetricGate, load_config
from physim_gate.engine import compare_runs
from physim_gate.schema import (
    FAILURE_RATE_METRIC,
    Direction,
    Domain,
    MetricSpec,
    RunResult,
    SchemaError,
    Trial,
    TrialStatus,
    dump_json,
    load_run,
    write_run,
)

__all__ = [
    "FAILURE_RATE_METRIC",
    "RUN_LABEL",
    "ComparisonResult",
    "Direction",
    "Domain",
    "Evidence",
    "ExitCode",
    "GateConfig",
    "MetricComparison",
    "MetricGate",
    "MetricSpec",
    "RunResult",
    "SchemaError",
    "Trial",
    "TrialStatus",
    "Verdict",
    "__version__",
    "combine_runs",
    "compare_runs",
    "dump_json",
    "load_comparison",
    "load_config",
    "load_run",
    "write_run",
]
