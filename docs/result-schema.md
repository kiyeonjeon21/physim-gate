# Result schema

PhySim Gate reads one JSON document per run.
The schema is versioned and independent of any simulator.
The machine-readable definition is [`schemas/run-result.schema.json`](../schemas/run-result.schema.json), also printed by `physim-gate schema run`.

## Versioning

`schema_version` is `"<major>.<minor>"`, currently `"1.0"`.
A reader accepts any minor version of the major versions it supports and rejects unknown major versions with a clear error.
Minor versions only add optional fields.
Breaking changes bump the major version.

The comparison result has its own version (`kind: "physim-gate.comparison"`, `schema_version: "1.1"`), described in [`schemas/comparison.schema.json`](../schemas/comparison.schema.json).
Version 1.1 added `clusters` to each metric (the cluster means behind a `cluster_by` comparison) and `inputs.baseline_runs` / `inputs.candidate_runs` (the files of a side given as several runs).

## Run result

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | no (defaults to `"1.0"`) | |
| `run_id` | string | yes | Unique identifier of this run |
| `created_at` | ISO 8601 datetime | no | |
| `domain` | `robotics`, `autonomous_driving`, `aerial`, `generic` | no (`generic`) | |
| `simulator` | `{name, version}` | yes (`name`) | Record the version: physics changes between versions |
| `scenario` | `{id, version}` | yes (`id`) | Runs of different scenarios are not compared by default |
| `policy` | `{id, version}` | no | The system under test, for example a checkpoint or planner version |
| `seeds` | list of integers | no | Run-level seed list |
| `source` | `{repository, commit, branch, dirty}` | no | Provenance of the code under test |
| `environment` | `{python, platform, machine, extra}` | no | `extra` holds any JSON values (GPU model, container digest, ...) |
| `status` | `completed`, `partial`, `failed` | no (`completed`) | |
| `failure` | `{kind, message}` | no | Why the run failed or is partial |
| `metrics` | map of name to metric spec | yes | Declares every metric the trials may report |
| `trials` | list of trials | yes | May be empty, but then nothing can be compared |
| `artifacts` | list of artifact refs | no | Videos, logs, bags, traces |
| `labels` | map of string to string | no | Free-form tags |
| `extra` | map of string to JSON | no | Anything else; ignored by the engine |

Unknown top-level fields are rejected, so typos surface early.
Put custom data under `extra` or `labels`.

### Metric spec

| Field | Type | Required | Notes |
|---|---|---|---|
| `direction` | `higher_is_better`, `lower_is_better` | yes | |
| `unit` | string | no (`""`) | Must match between baseline and candidate |
| `description` | string | no | |

Metric names may contain letters, digits and `_ . : / -`.
The prefix `physim.` is reserved for metrics derived by PhySim Gate.

Simulators do not need to expose the same metrics.
A run declares what it measured; the gate config chooses what to compare.

### Trial

| Field | Type | Required | Notes |
|---|---|---|---|
| `trial_id` | string | yes | Unique within the run. Matching ids across runs mean matching conditions |
| `seed` | integer | no | |
| `status` | `ok`, `failed`, `timeout`, `error` | no (`ok`) | |
| `failure` | `{kind, message}` | no | Expected when `status` is not `ok` |
| `metrics` | map of name to number or `null` | no | Every key must be declared in `metrics` |
| `duration_s` | number `>= 0` | no | Wall-clock duration |
| `artifacts` | list of artifact refs | no | |
| `labels` | map of string to string | no | `cluster_by` groups trials by one of these, for example `training_seed`. `physim.run` is reserved: PhySim Gate sets it when it combines several run files |

### Artifact reference

`{name, uri, media_type, sha256}`, where `name` and `uri` are required.
`uri` can be a relative path, an absolute path or a URL.
`sha256` is a 64-character lowercase hex digest.

## Missing data versus zero

This distinction is central to sound comparisons:

| In `trial.metrics` | Meaning |
|---|---|
| `"x": 0` | A valid observation of zero |
| `"x": null` | The metric was expected but is unavailable (for example the episode crashed) |
| `x` absent | The metric was not reported for this trial |

Both `null` and absent count as missing during comparison.
So does a whole trial that the other run reports but this one leaves out, when the runs share `trial_id`s.
Missing values make a gated metric `INVALID` unless `max_missing_fraction` allows them (see [comparison semantics](comparison-semantics.md)).
Report every trial you ran, including crashed ones, with `status` set and `null` where a value is unavailable.
`physim-gate validate` reports how many values are missing per metric.

## Numbers

Metric values must be finite JSON numbers.
`NaN`, `Infinity` and `-Infinity` are rejected, including the non-standard tokens that Python's `json` module would otherwise accept.
Booleans and numeric strings are rejected; encode a success flag as `0` or `1`.

## Repeated keys

A JSON object or YAML mapping that repeats a key is rejected, in run results, comparison results, gate configs, suite files and LeRobot output alike.
Python's parsers would otherwise keep the last value silently, so a second entry for a metric or a scenario could change a verdict without anyone noticing.

## Derived metric: `physim.failure_rate`

For every trial, `physim.failure_rate` is `0` when `status` is `ok` and `1` otherwise.
Its mean is the fraction of failed, timed-out or errored trials.
It is `lower_is_better`, can be gated like any other metric, and does not need to be declared.

## Deterministic output

PhySim Gate writes JSON with a stable field order, two-space indentation and a trailing newline.
Generating the same run twice produces identical bytes, which keeps result files diff-friendly in version control.
Comparison results carry a `generated_at` timestamp; set `SOURCE_DATE_EPOCH` to fix it.

## Minimal example

```json
{
  "schema_version": "1.0",
  "run_id": "nightly-2026-10-01",
  "simulator": {"name": "mujoco", "version": "3.2.0"},
  "scenario": {"id": "peg_insertion"},
  "metrics": {
    "success": {"direction": "higher_is_better"},
    "insertion_time": {"direction": "lower_is_better", "unit": "s"}
  },
  "trials": [
    {"trial_id": "seed-0", "seed": 0, "metrics": {"success": 1, "insertion_time": 4.2}},
    {"trial_id": "seed-1", "seed": 1, "status": "timeout",
     "failure": {"kind": "timeout", "message": "exceeded 30 s"},
     "metrics": {"success": 0, "insertion_time": null}}
  ]
}
```

The simulator name above is only an example of what a producer would write.
PhySim Gate does not ship a MuJoCo adapter.
