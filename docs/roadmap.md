# Roadmap

This is a plan, not a promise.
Priorities follow what makes the gate trustworthy and useful with real simulators first.

## v0.1

v0.1.1 fixed the findings of the [OSS readiness review](oss-readiness-review.md), most importantly score intervals for 0/1 metrics.

- Run-result schema 1.0, comparison schema 1.0, JSON Schemas.
- Comparison engine: absolute, relative and combined margins, mean or median, paired and unpaired percentile bootstrap, missing-data and zero-baseline rules, optional Bonferroni correction, conservative policy defaults.
- CLI: `init`, `validate`, `compare`, `report`, `demo`, `adapters`, `schema`.
- Terminal, JSON, JUnit XML and self-contained HTML reports.
- Adapters: `json`, `synthetic`, optional `gymnasium`; entry-point plugin contract with an example CSV plugin.

## v0.2

- **LeRobot adapter** for `lerobot-eval`'s `eval_info.json`, multi-task and single-task layouts, validated against real PushT output.
- **Sample-size planning** with `physim-gate plan`: exact verdict probabilities for unpaired 0/1 metrics, a normal approximation for continuous ones.
- **Markdown summary** (`--markdown`, `--format markdown`) sized for GitHub's comment limit.
- **GitHub Action** (`uses: kiyeonjeon21/physim-gate@<tag>`) with job summary, outputs and an updating pull-request comment.
- **PyPI release** under `physim-gate`, published by a release workflow with trusted publishing.

## v0.3

- **Expanded percentile bootstrap** by default. Per-side error at the margin with 10 trials fell from up to 5.5% to at most 3.5%. BCa was measured too and did not help.
- **Paired and pilot-based planning.** `plan --discordance` simulates paired 0/1 metrics; `plan --from comparison.json` plans every metric of a pilot.
- **Baseline store.** `physim-gate baseline save/list/path` and `--baseline @name`, file-based and Git-friendly.
- **Suites.** `physim-gate suite` gates several scenarios with one exit code, a JUnit test suite per scenario and one Markdown summary.
- **Cross-checks** of the statistics against scipy and statsmodels in the test suite.

## v0.5

- **Training runs as clusters.** `cluster_by` compares cluster means (training runs, seeds) with a t interval, so a gate can answer whether a training change, not one checkpoint, is worse. Several `--baseline` / `--candidate` files are combined and clustered by run. `plan --from` a clustered pilot counts runs per side.

## Next

1. **Stratified comparisons** by trial labels (task, weather, terrain), with per-stratum verdicts; LeRobot already labels each episode with its task.
2. **Equivalence gates** for "must not change in either direction" checks (calibration, determinism audits).
3. **LIBERO output.** LeRobot installs LIBERO on Linux only; a recorded LIBERO evaluation would complete the real-data examples. The public LeBoard calibration runs are a candidate.
4. **Several runs per side in suites and the GitHub Action**, as `compare` already accepts.
5. **ROS 2 bag and PX4 SITL adapters** from recorded fixtures, with user-defined metric extraction.

## Later

- A CARLA ScenarioRunner or Leaderboard adapter, only when it can be tested in CI from recorded results.
- Trajectory- and time-series-level metrics (for example maximum deviation over a rollout) computed by adapters into per-trial scalars, with helpers.
- Optional NumPy acceleration for very large trial counts, kept out of the default install.

## Explicitly out of scope

A physics engine, a scenario DSL, simulator installation, schedulers, Kubernetes, dashboards, hosted services, accounts, LLM-based diagnosis, data lakes, hardware control and safety certification.
PhySim Gate stays a small, local tool that other systems can call.
