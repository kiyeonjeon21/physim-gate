# PhySim Gate

**Test physical AI like software. Catch behavioral regressions before deployment.**

PhySim Gate is a small, simulator-agnostic regression gate for robotics, embodied AI, autonomous driving and aerial systems.
You give it the results of a baseline run and a candidate run.
It decides, metric by metric, whether the candidate behaves worse than allowed.
It explains why, writes evidence your CI can keep, and exits with a code your pipeline can act on.

It does not run your simulator, generate scenarios, or host anything.
It reads result files that your existing simulation jobs already produce.

```text
your sim job ──> baseline.json ─┐
                                ├─> physim-gate compare ──> verdict + exit code
your sim job ──> candidate.json ┘          │                 comparison.json, junit.xml
                    gate.yaml ─────────────┘                 report.html, summary.md
```

## Contents

- [The problem](#the-problem)
- [How PhySim Gate differs](#how-physim-gate-differs)
- [Quickstart (no GPU, no simulator)](#quickstart-no-gpu-no-simulator)
- [Example inputs](#example-inputs)
- [Example report](#example-report)
- [Real simulator data: LeRobot](#real-simulator-data-lerobot)
- [How many trials do you need?](#how-many-trials-do-you-need)
- [Comparing training runs, not checkpoints](#comparing-training-runs-not-checkpoints)
- [Baselines and suites](#baselines-and-suites)
- [How the verdict is reached](#how-the-verdict-is-reached)
- [Statistical limitations](#statistical-limitations)
- [CI usage](#ci-usage)
- [Writing an adapter](#writing-an-adapter)
- [Python API](#python-api)
- [Supported and not supported](#supported-and-not-supported)
- [Roadmap](#roadmap)
- [Contributing and license](#contributing-and-license)

## The problem

A change to a grasp policy, a motion planner or a flight controller can pass every unit test and still make the system behave worse.
Simulation is where teams catch that, but the comparison step is usually ad hoc.
Someone eyeballs two dashboards, or a script checks that a mean did not drop by more than 5%.

Both approaches fail in predictable ways:

- **Noise looks like a regression, or hides one.** Physics engines, sensor models and learned policies are stochastic. Thirty episodes of a 90% success policy can easily show 83% or 97%.
- **Missing data hides failures.** If crashed episodes drop out of the average, a policy that crashes more often can look better.
- **Results are not reproducible.** Nobody recorded which simulator version, seed list or commit produced the baseline.
- **CI cannot act on it.** A chart is not an exit code.

PhySim Gate treats each comparison as a small, explicit statistical decision with a reproducible record.

## How PhySim Gate differs

Several open-source projects overlap with parts of this problem.
PhySim Gate is meant to complement them, not replace them.
The short version is below; [docs/competitive-landscape.md](docs/competitive-landscape.md) has details and notes what could not be verified.

| Project | What it focuses on | Relationship to PhySim Gate |
|---|---|---|
| [RoboCI](https://github.com/gagandeepreehal/roboci) | Runs robotics scenario suites (mock, CARLA, Isaac, Gazebo, ROS 2 adapters), tracks failures, compares one aggregate per run against a baseline with a percentage threshold, includes a dashboard | Closest overlap. PhySim Gate never runs a simulator and compares per-trial distributions with an uncertainty interval instead of one number per run |
| [robot-sim-test-harness](https://github.com/Pratyush150/robot-sim-test-harness) | YAML scenarios with temporal assertions on a kinematic mock, trace diffs, per-metric tolerance checks against a committed baseline | Gates on pass/fail transitions and tolerance drift of single values. PhySim Gate compares distributions of trials and reports uncertainty |
| Python statistics and benchmarking tools: [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [statsmodels](https://www.statsmodels.org/), [rliable](https://github.com/google-research/rliable), [asv](https://github.com/airspeed-velocity/asv), [Bencher](https://bencher.dev/) | Bootstrap and score intervals, non-inferiority tests, RL evaluation statistics, statistical benchmark regression detection | PhySim Gate's statistics are standard methods, not new ones. What it adds is the run schema, the pairing and missing-data rules, the three-way verdict and the CI outputs around them |
| [LeRobot](https://github.com/huggingface/lerobot) `lerobot-eval` | Evaluates a policy and writes per-episode rewards and success to `eval_info.json` | An upstream producer. The built-in `lerobot` adapter reads its `eval_info.json` |
| [Scenic](https://github.com/Scenic-Foundation/Scenic), [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) | Scenario generation, falsification and formal analysis | Upstream and complementary. They decide what to simulate; PhySim Gate decides whether a change made the outcome worse |
| [ros2_tracing](https://github.com/ros2/ros2_tracing) | Low-overhead LTTng tracing of ROS 2 | A possible future artifact source. No comparison logic |

What PhySim Gate aims to do well:

1. **Simulator independence.** It evaluates existing run artifacts. Any simulator that can write JSON, or that has an adapter, works.
2. **Uncertainty-aware gates built on standard methods.** A confidence interval of the *difference*, paired when trials match (a bootstrap for continuous metrics, a Newcombe score interval for 0/1 metrics), with an explicit `INCONCLUSIVE` outcome. Measured error rates are published in [docs/comparison-semantics.md](docs/comparison-semantics.md#measured-error-rates).
3. **Provenance.** Simulator version, seeds, source commit, environment and input file hashes travel with every result.
4. **Machine-readable output.** Versioned JSON, JUnit XML, documented exit codes, a self-contained HTML report, and a Markdown summary for pull requests.
5. **A minimal plugin contract.** One entry point and one method, no changes to the core package.
6. **Local first.** `pip install` and run. No service, database, Docker, GPU or account.

## Quickstart (no GPU, no simulator)

PhySim Gate needs Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install physim-gate

physim-gate demo
```

The demo generates synthetic baseline and candidate runs for four scenarios, runs the real comparison engine on each, and writes reports to `./physim-gate-demo/`:

```text
demo summary ────────────────────────────────────────────────────────────
  no_regression           PASS         exit 0  as expected
  regression              FAIL         exit 1  as expected
  insufficient_evidence   INCONCLUSIVE exit 3  as expected
  invalid_comparison      INVALID      exit 3  as expected
```

Nothing in the demo is precomputed.
The runs come from YAML specs in [`src/physim_gate/fixtures/`](src/physim_gate/fixtures) and the verdicts come from the engine.
The synthetic generator is not a simulator; it draws values from simple distributions so the gate can be exercised end to end.

Then compare two files yourself.
The example files live in the repository, so clone it first:

```bash
git clone https://github.com/kiyeonjeon21/physim-gate.git && cd physim-gate
physim-gate compare \
  --baseline examples/highway-merge/baseline.json \
  --candidate examples/highway-merge/candidate.json \
  --config examples/highway-merge/gate.yaml \
  --html report.html --junit junit.xml --output comparison.json
echo $?   # 1: regression detected
```

Other commands:

```bash
physim-gate init --from my_run.json     # starter gate.yaml from an existing result
physim-gate validate my_run.json        # schema check plus warnings (missing values, no commit, ...)
physim-gate report comparison.json --html report.html   # re-render a saved comparison
physim-gate plan --rate 0.9 --margin 0.05   # trials needed to decide a 0/1 metric
physim-gate adapters                    # built-in and installed adapters
physim-gate schema run                  # JSON Schema for producers in any language
```

## Example inputs

A run result is one JSON document per run.
This is an abridged excerpt (two metrics, two of 80 trials) of [`examples/highway-merge/candidate.json`](examples/highway-merge/candidate.json) (synthetic data):

```json
{
  "schema_version": "1.0",
  "run_id": "highway-merge-planner-2.1.0-pr-3140",
  "domain": "autonomous_driving",
  "simulator": { "name": "physim-gate-synthetic", "version": "1" },
  "scenario": { "id": "highway_on_ramp_merge", "version": "v7" },
  "policy": { "id": "motion-planner", "version": "2.1.0" },
  "source": { "commit": "7e6d5c4b...", "branch": "aggressive-gap-acceptance", "dirty": false },
  "metrics": {
    "collision": { "direction": "lower_is_better", "unit": "" },
    "min_time_to_collision": { "direction": "higher_is_better", "unit": "s" }
  },
  "trials": [
    { "trial_id": "seed-5000", "seed": 5000, "status": "ok",
      "metrics": { "collision": 0.0, "min_time_to_collision": 1.996171 } },
    { "trial_id": "seed-5001", "seed": 5001, "status": "ok",
      "metrics": { "collision": 0.0, "min_time_to_collision": 2.993258 } }
  ]
}
```

The gate config says how much degradation each metric may show:

```yaml
version: 1
defaults:
  confidence: 0.95
  min_samples: 30
metrics:
  collision:
    max_absolute_degradation: 0.02      # collision rate may rise by at most 2 points
  min_time_to_collision:
    max_absolute_degradation: 0.2       # seconds
  lateral_error_rms:
    max_relative_degradation: 0.10
  jerk_p95:
    max_relative_degradation: 0.10
    required: false                     # tracked, but does not block
```

The full field reference is in [docs/result-schema.md](docs/result-schema.md).

## Example report

Terminal output for the example above:

```text
PhySim Gate   FAIL   regression detected in 2 of 4 gated metrics: collision, min_time_to_collision
scenario  highway_on_ramp_merge (autonomous_driving) | simulator physim-gate-synthetic 1
baseline  highway-merge-planner-2.0.3-main-1187  80 trials | policy motion-planner@2.0.3 | commit 1f2e3d4c5b6a
candidate highway-merge-planner-2.1.0-pr-3140  80 trials | policy motion-planner@2.1.0 | commit 7e6d5c4b3a2f

metric                  ┃ baseline ┃ candidate ┃ degradation ┃              interval ┃ margin ┃  n ┃ verdict
━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━╇━━━━━━━━
collision ↓             │   0.0375 │      0.15 │     +0.1125 │   [+0.03042, +0.2051] │   0.02 │ 80 │ FAIL
min_time_to_collision ↑ │     3.16 │     2.654 │     +0.5058 │    [+0.4328, +0.5778] │    0.2 │ 80 │ FAIL
lateral_error_rms ↓     │    0.214 │    0.2175 │   +0.003511 │ [-0.006654, +0.01395] │ 0.0214 │ 80 │ PASS
jerk_p95 ↓ (optional)   │    2.359 │     2.392 │     +0.0324 │   [-0.02786, +0.0946] │ 0.2359 │ 80 │ PASS
degradation > 0 means worse; interval = confidence interval of the degradation; n = matched pairs or
baseline/candidate observations

collision: FAIL
  - regression: the 95% interval of the degradation lies entirely above the allowed margin (0.02)
min_time_to_collision: FAIL
  - regression: the 95% interval of the degradation lies entirely above the allowed margin (0.2 s)
wrote comparison.json, junit.xml, report.html
```

The HTML report is a single file with no external assets.
It shows the verdicts, each metric's degradation interval against its margin, the observed per-trial values, and the provenance of both runs:

![Report summary](docs/images/report-summary.png)

![Metric detail](docs/images/report-metric.png)

## Real simulator data: LeRobot

[`examples/lerobot-pusht/`](examples/lerobot-pusht) holds unedited `eval_info.json` files from LeRobot's `lerobot-eval` on the PushT simulator, 100 episodes each, and a gate config.
The built-in `lerobot` adapter reads them directly, or the output directory that contains them:

```bash
physim-gate compare --adapter lerobot -O start_seed=1000 \
  -b examples/lerobot-pusht/baseline \
  -c examples/lerobot-pusht/candidate-n-action-steps-8 \
  -g examples/lerobot-pusht/gate.yaml
echo $?   # 1: max_reward regressed; the success drop is INCONCLUSIVE at 100 episodes
```

The candidate re-plans every 8 steps instead of every 32.
Its success rate fell from 11% to 1%, but with 100 episodes per run the interval of that drop (+3.4 to +17.7 points) still includes the 5-point margin, so `success` is `INCONCLUSIVE` rather than `FAIL`.
`max_reward` dropped clearly enough to fail the gate.
A second evaluation of the unchanged policy passes, because it reproduced every success flag exactly.
The example's [README](examples/lerobot-pusht/README.md) explains the runs and how to reproduce them.
[`examples/lerobot-metaworld/`](examples/lerobot-metaworld) does the same for a multi-task evaluation of a SmolVLA policy on four MetaWorld tasks, and [`examples/lerobot-suite.yaml`](examples/lerobot-suite.yaml) gates both examples with one command.

Each episode becomes a trial with `success`, `sum_reward` and `max_reward`.
`lerobot-eval` seeds episode `i` of every task with `--seed + i`, so two evaluations with the same `--seed` are compared as matched pairs.
Both layouts are read: `per_task` (LeRobot 0.4.0 and later) and `per_episode` (0.3.x and earlier).

## How many trials do you need?

The gate only says `PASS` or `FAIL` when the confidence interval clears the margin, and 0/1 metrics need far more trials than most evaluations run.
`physim-gate plan` answers the question before you spend the compute:

```text
$ physim-gate plan --rate 0.9 --margin 0.05
plan for a 0/1 metric, higher is better
  baseline rate     0.9
  candidate rate    0.9
  true degradation  0
  margin            0.05
  goal              PASS with probability >= 80%, 95% intervals
  method            exact (unpaired Newcombe score interval)

  573 trials per run
  verdicts at that size: PASS 80.1%  INCONCLUSIVE 19.9%  FAIL <0.1%
```

To show that an unchanged 90% success rate has not dropped by more than 5 points, plan for about 570 episodes per run.
With 50 episodes per run, a common size for robot evaluations, the gate says `INCONCLUSIVE` about 9 times in 10, and that is the correct answer for so little data.
For 0/1 metrics the probabilities are exact: every possible outcome is weighted by its binomial probability and decided with the engine's own score interval.
Pass `--true-degradation` to size a run that should detect a regression, and `--stdev` instead of `--rate` for continuous metrics (a normal approximation).

Runs that share trial conditions are compared as pairs, which can need far fewer trials.
How much depends on how often the two runs disagree on the same condition, so paired 0/1 plans take `--discordance` and are simulated with the engine's paired interval: if baseline and candidate disagree on 4% of conditions, the example above needs 173 trials instead of 573.

The easiest input is a pilot.
After a comparison came out `INCONCLUSIVE`, `physim-gate plan --from comparison.json` plans every gated metric from the pilot's own rates, spreads, pairing and margins, assuming the observed degradation is the true one:

```text
$ physim-gate plan --from comparison.json      # the LeRobot PushT comparison above
metric     ┃   pilot n ┃ degradation ┃ margin ┃ goal ┃ trials per run ┃ P(goal)
━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━
success    │ 100 pairs │        +0.1 │   0.05 │ FAIL │            317 │   80.4%
max_reward │ 100 pairs │     +0.1566 │   0.05 │ FAIL │            157 │   80.1%
sum_reward │ 100 pairs │      -27.47 │  8.283 │ PASS │             53 │   80.4%
```

[docs/comparison-semantics.md](docs/comparison-semantics.md#planning-the-number-of-trials) has the details.

## Comparing training runs, not checkpoints

A comparison of one run against another answers "is this checkpoint worse?".
When the change is to training (data, recipe, code), the question is usually "is this change worse?", and a second training run with another seed scores differently.
The trials of one run share that run's luck, so even thousands of episodes cannot answer it.
In a [simulation](docs/comparison-semantics.md#clusters-several-training-runs-per-side) modeled on LIBERO evaluations, a run-to-run spread of 1.4 success points made a trial-level gate wrong at the margin in up to 7% of comparisons per side, and a spread of 4.2 points in up to 25%.

Give each side several runs, and PhySim Gate compares their means instead:

```text
$ physim-gate compare --adapter lerobot -O scenario=libero_10 \
    -b clean-s0/eval_info.json -b clean-s1/eval_info.json -b clean-s2/eval_info.json \
    -b clean-s3/eval_info.json -b clean-s4/eval_info.json \
    -c desync-s0/eval_info.json -c desync-s3/eval_info.json -c desync-s4/eval_info.json \
    -g gate.yaml
metric    ┃ baseline ┃ candidate ┃ degradation ┃           interval ┃ margin ┃        n ┃ verdict
━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━
success ↑ │   0.4216 │    0.3833 │    +0.03827 │ [-0.0356, +0.1122] │      0 │ 5/3 runs │ INCONCLUSIVE
```

These are the [LeBoard calibration runs](https://huggingface.co/datasets/rogation/leboard-training-runs): SmolVLA fine-tuned on LIBERO-10, once on clean data and once on data whose video lags the actions by 3 frames, each run evaluated on the same 500 episodes.
Compared episode by episode, one clean run against the lagged run with the same training seed is a `FAIL` for two of the three seeds.
Across training runs, a 3.8-point drop is not yet distinguishable from the spread between runs, and `physim-gate plan --from` says 8 runs per side would make it so with 80% probability.

Several `-b` or `-c` files set `cluster_by: physim.run`: every trial is labeled with its run, and the interval is a t interval over run means.
To match runs by training seed across the two sides, label every trial with the seed (`labels: {training_seed: "3"}`) and set `cluster_by: training_seed` in the gate config.

## Baselines and suites

A baseline is just a run result, but keeping track of which file is "the baseline for this scenario on main" is tedious.
`physim-gate baseline` keeps them as plain files in `.physim-gate/baselines/<scenario>/<name>.json`, ready to commit:

```bash
physim-gate baseline save -a lerobot -O start_seed=1000 eval/main/   # stored as pusht @main
physim-gate baseline list
physim-gate compare -a lerobot -O start_seed=1000 -b @main -c eval/pr-42/ -g gate.yaml
```

`-b @main` picks the stored baseline named `main` for the candidate's scenario, whatever adapter reads the candidate.
`--store DIR` or `PHYSIM_GATE_STORE` moves the store.

To gate several scenarios at once, list them in a suite file:

```yaml
version: 1
defaults:
  adapter: lerobot
  adapter_options: {start_seed: "1000"}
  config: gate.yaml
scenarios:
  pusht:
    baseline: "@main"
    candidate: eval/pr-42/pusht
  metaworld:
    baseline: "@main"
    candidate: eval/pr-42/metaworld
    config: metaworld-gate.yaml
```

`physim-gate suite suite.yaml -o reports/` compares every scenario, writes the usual reports per scenario, and adds `suite.json`, a combined `junit.xml` (one test suite per scenario) and a combined `summary.md`.
One scenario that cannot be read does not stop the others.
The exit code is the most severe one: 2 if any scenario could not be compared, else 1 if any regressed, else 3 if any is blocked, else 0.

## How the verdict is reached

For every gated metric the result keeps three layers apart:

1. **Numerical.** The baseline and candidate statistic (mean by default), the difference, and the *degradation*: the difference signed so that positive always means worse.
2. **Statistical evidence.** A confidence interval of the degradation, computed directly on the difference and paired when trials match. Continuous metrics use a percentile bootstrap. Metrics whose values are all 0 or 1 (success flags, collision indicators) use a Newcombe score interval instead, because a bootstrap claims certainty when a run has no events at all. Relative to the allowed margin the interval is classified as:
   - `regression`: the whole interval lies above the margin,
   - `non_inferior`: the whole interval lies at or below the margin,
   - `inconclusive`: the interval straddles the margin,
   - `insufficient_samples`: fewer observations than `min_samples`, so no interval is computed.
3. **Policy verdict.** `PASS`, `FAIL`, `INCONCLUSIVE` or `INVALID`, and whether it blocks the gate.

The default is conservative.
A metric passes only when the evidence shows it did not degrade beyond the margin.
Inconclusive and invalid metrics block the gate unless `policy.on_inconclusive` or `policy.on_invalid` says otherwise.
Missing values make a metric `INVALID` unless `max_missing_fraction` allows them, so crashed trials cannot silently drop out.
Runs that are not comparable make the whole comparison `INVALID` whatever the policy says: a different scenario or scenario version, a failed run, a run in which every trial failed, or shared trial ids with different seeds.
The built-in metric `physim.failure_rate`, derived from each trial's status, lets you gate crashes and timeouts directly.

[docs/comparison-semantics.md](docs/comparison-semantics.md) specifies every rule, including thresholds, baselines of zero, pairing and missing data.

## Statistical limitations

PhySim Gate makes a narrow, explicit claim, and it is worth knowing where that claim is weak:

- **The intervals are approximate.** In a [Monte Carlo study](docs/comparison-semantics.md#measured-error-rates) of the real engine, a metric whose true degradation equals the margin was wrongly called PASS or FAIL in at most 3.4% of runs on each side, including with only 10 trials and against a much larger baseline; the nominal rate is 2.5%. `min_samples` (default 10) is a floor, not a guarantee.
- **Rare events need many trials.** With a 1% baseline collision rate and a 2-point margin, 80 trials per run give `INCONCLUSIVE` about 89% of the time, even when the candidate's true rate is 6%. That is the honest answer for so little data. `physim-gate plan` tells you how many trials a decision needs.
- **Each metric is tested separately.** With many gated metrics, the chance that at least one shows a spurious regression grows. `policy.correction: bonferroni` is available but off by default.
- **One run is one sample of a training process.** A trial-level comparison says whether these runs differ, not whether the training change behind them makes a difference; use [several runs per side](#comparing-training-runs-not-checkpoints) for that.
- **Seeds do not make physics deterministic.** Matching `trial_id`s are treated as matched *conditions*. Pairing reduces noise from shared conditions; it does not assume bitwise-identical rollouts.
- **Inconclusive is a real answer.** It means the data cannot tell. The fix is more trials, a less noisy metric, or a deliberate policy decision, not a looser reading of the numbers.
- **Statistics cannot judge relevance.** Whether a 2-point increase in collision rate matters is a domain decision you encode in the margin.
- **No sequential testing.** Re-running a comparison with more trials until it passes inflates the error rate.
- **Not a safety case.** PhySim Gate is a regression-testing aid. It makes no safety, certification or sim-to-real claims.

## CI usage

`physim-gate compare` exits with:

| Code | Meaning |
|---|---|
| 0 | Gate passed |
| 1 | Regression detected in a blocking metric (for `validate`: the file is invalid) |
| 2 | Usage error, unreadable input, or an invalid config or result document |
| 3 | No regression shown, but a blocking metric is inconclusive or invalid |
| 4 | Internal error (a bug; please report it) |

On GitHub Actions, use the action in this repository:

```yaml
- uses: kiyeonjeon21/physim-gate@v0.4.0
  with:
    baseline: results/baseline.json
    candidate: results/candidate.json
    config: gate.yaml
    comment: "true"   # post the summary on the pull request, updated on every push
```

It installs PhySim Gate from the same tag, writes `comparison.json`, `junit.xml`, `report.html` and `summary.md`, adds the Markdown summary to the job summary, and fails the step unless the gate passes.
Its outputs are `verdict`, `exit-code` and `report-dir`; `fail: "false"` leaves the decision to later steps.
Pull requests from forks get no comment, because their token is read-only, and a failure to comment is a warning; neither changes the verdict.
[`examples/github-actions/physim-gate.yml`](examples/github-actions/physim-gate.yml) is a complete workflow that also uploads the report and publishes the JUnit results.

Anywhere else, call the CLI:

```bash
physim-gate compare -b results/baseline.json -c results/candidate.json -g gate.yaml \
  --junit results/junit.xml --html results/report.html --markdown results/summary.md
```

`--markdown` writes a summary sized for a pull-request comment, and `--format markdown` prints it to stdout.
`--format json` prints the comparison result to stdout for other tooling.
Set `SOURCE_DATE_EPOCH` to make `generated_at`, and therefore the output files, byte-for-byte reproducible.

## Writing an adapter

An adapter turns a simulator's native output into a run result.
It is any class with a `name`, a `description` and a `load` method, registered as an entry point:

```python
# my_package/adapter.py
from pathlib import Path
from collections.abc import Mapping
from physim_gate.adapters import AdapterError
from physim_gate.schema import RunResult, run_from_data


class MySimAdapter:
    name = "mysim"
    description = "Reads MySim episode logs."

    def load(self, source: Path, options: Mapping[str, str]) -> RunResult:
        if options:
            raise AdapterError(f"mysim adapter takes no options, got {sorted(options)}")
        episodes = parse_my_format(source)  # your code
        return run_from_data(
            {
                "run_id": source.stem,
                "simulator": {"name": "mysim", "version": episodes.version},
                "scenario": {"id": episodes.scenario},
                "metrics": {"success": {"direction": "higher_is_better"}},
                "trials": [
                    {"trial_id": e.id, "seed": e.seed, "metrics": {"success": float(e.success)}}
                    for e in episodes
                ],
            },
            str(source),
        )
```

```toml
# my_package/pyproject.toml
[project.entry-points."physim_gate.adapters"]
mysim = "my_package.adapter:MySimAdapter"
```

After `pip install`, `physim-gate compare --adapter mysim ...` uses it.
Options are passed with `-O key=value`.
[`examples/csv-adapter/`](examples/csv-adapter) is a complete, installable example, and [docs/adapters.md](docs/adapters.md) describes the contract and its guarantees.
You can also skip adapters entirely and write the JSON schema directly from any language; `physim-gate schema run` prints the JSON Schema.

## Python API

```python
from pathlib import Path
from physim_gate import compare_runs, load_config, load_run
from physim_gate.reporting import render_html

result = compare_runs(
    load_run(Path("baseline.json")),
    load_run(Path("candidate.json")),
    load_config(Path("gate.yaml")),
)
print(result.verdict, result.summary)
for metric in result.metrics:
    print(metric.name, metric.verdict, metric.evidence.degradation_ci)
Path("report.html").write_text(render_html(result))
```

`combine_runs("baseline", names, runs)` merges several runs into one whose trials carry the `physim.run` label (`RUN_LABEL`); compare it under a config from `config.with_default_cluster_by(RUN_LABEL)` to compare run means, as the command line does.

With the `gymnasium` extra, `physim_gate.adapters.gymnasium.run_gymnasium(env_id, policy, seeds=..., run_id=...)` rolls out a policy and returns a run result directly.
[`examples/gymnasium/`](examples/gymnasium) compares two CartPole controllers this way.

## Supported and not supported

| Capability | Status in v0.4.0 |
|---|---|
| Run-result schema 1.0 (JSON), JSON Schema export | Supported |
| Baseline vs candidate comparison, paired and unpaired (expanded percentile bootstrap, Newcombe score intervals) | Supported |
| Baseline store (`physim-gate baseline`, `-b @name`) and multi-scenario suites (`physim-gate suite`) | Supported |
| Absolute, relative and combined margins; mean or median | Supported |
| Terminal, JSON, JUnit XML, Markdown and self-contained HTML output | Supported |
| GitHub Action with pull-request comments | Supported |
| Sample-size planning (`physim-gate plan`, also from a pilot comparison) | Supported: exact for unpaired 0/1 metrics, simulated for paired ones, approximate for continuous ones |
| Third-party adapters via entry points | Supported |
| Built-in adapters: `json`, `synthetic` (demo/tests) | Supported |
| `gymnasium` adapter (optional extra, CPU classic-control environments tested) | Supported, tested with CartPole only |
| `lerobot` adapter for `lerobot-eval` output | Supported, tested with real LeRobot 0.6.2 output on PushT and on four MetaWorld tasks; LIBERO not run (LeRobot installs it on Linux only) |
| CARLA, ROS 2 bags, PX4 SITL, Isaac Sim adapters | **Not implemented** (see roadmap) |
| Running simulators, scheduling jobs, scenario generation | Out of scope |
| Dashboards, hosted service, accounts | Out of scope |
| Time-series or trajectory-level comparison (only per-trial scalars today) | Not implemented |
| Safety or certification claims | Never |

## Roadmap

The near-term plan is in [docs/roadmap.md](docs/roadmap.md).
Next up: stratified verdicts by trial labels, equivalence gates, and adapters for ROS 2 bags and PX4 SITL logs that can be tested from recorded fixtures.
Later versions look at trajectory-level metrics.

## Contributing and license

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
The architecture is described in [docs/architecture.md](docs/architecture.md).
All example data in this repository is synthetic, except [`examples/lerobot-pusht/`](examples/lerobot-pusht) and [`examples/lerobot-metaworld/`](examples/lerobot-metaworld), which are real LeRobot output from public checkpoints.

PhySim Gate is licensed under the [Apache License 2.0](LICENSE).
