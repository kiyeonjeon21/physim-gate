# Comparison semantics

This document specifies how `physim-gate compare` turns two run results and a gate config into verdicts.
It is the contract: if the code disagrees with this document, that is a bug.

## Terms

- **Run**: one run result document, a batch of trials of one policy on one scenario.
- **Trial**: one episode. It has a `trial_id`, a status and per-metric values.
- **Gated metric**: a metric listed under `metrics:` in `gate.yaml`.
- **Statistic**: `mean` (default) or `median` of a metric's trial values.
- **Degradation**: `candidate - baseline` for `lower_is_better` metrics, and `baseline - candidate` for `higher_is_better` metrics.
  Positive degradation always means the candidate is worse.
- **Margin**: the largest degradation that is acceptable.

## Step 1: run-level checks

Before looking at metrics, the two runs are checked for comparability.
A blocking issue makes the overall verdict `INVALID` (exit code 3) whatever `policy.on_invalid` says.
That setting tolerates metrics that cannot be compared, not runs that cannot be compared.

| Check | Effect |
|---|---|
| `scenario.id` differs | Blocking when `policy.require_same_scenario` is true (default). Otherwise a note |
| `scenario.version` differs, and both runs set it | Blocking when `policy.require_same_scenario` is true. Otherwise, or when only one run sets it, a note |
| A run's `status` is `failed` | Blocking |
| Every trial of a run has a status other than `ok` | Blocking |
| Some trials have a status other than `ok`, and `physim.failure_rate` is not gated | Note, suggesting to gate it |
| A `trial_id` present in both runs has a different recorded `seed` in each | Blocking, because shared ids claim shared conditions. A note when every metric uses `pairing: unpaired` |
| A run's `status` is `partial` | Note |
| `domain`, simulator name or simulator version differ | Note. Comparing across simulator versions is often the point |
| Same `run_id` in both runs | Note (often a copy-paste mistake) |
| Both inputs have identical content | Note. A deterministic simulator can reproduce a run byte for byte |

Giving the same file as baseline and candidate is an input error (exit code 2): such a comparison always passes.
A stored baseline (`-b @main`) is checked like any other, so a baseline saved for an older scenario version is not used silently.

Notes appear in every report under "run notes".
In v0.3.1 and earlier only a differing `scenario.id` could block, and only with `policy.on_invalid: fail`.

## Step 2: metric identity

For each gated metric:

1. Both runs must declare it (or it must be the built-in `physim.failure_rate`). Otherwise `INVALID`.
2. Both runs must declare the same `direction` and `unit`. Otherwise `INVALID`.
3. If the config sets `direction`, it must match the runs. Otherwise `INVALID`. This guards against a silently flipped direction.

## Step 3: collecting samples

Values come from **all** trials, regardless of status.
An absent key and an explicit `null` are both *missing*.
`0` is a valid value.

**Missing values.** If the fraction of missing values in either run exceeds `max_missing_fraction` (default `0.0`), the metric is `INVALID`.
The default is strict on purpose: if crashed trials simply dropped out, a policy that crashes more could look better.
To tolerate missing values, raise `max_missing_fraction` and gate `physim.failure_rate` so failures stay visible.

**Missing trials.** When the two runs share some `trial_id`s, they claim to cover the same conditions, so a trial that only one run reports counts as a missing value of the other run, exactly like an explicit `null`.
In v0.3.0 and earlier such trials were simply left out, so a candidate that omitted its crashed trials could pass where reporting them would have made the metric `INVALID`.
Runs whose `trial_id`s do not overlap at all are independent samples; nothing is considered missing then.
For `physim.failure_rate`, such a trial counts as a failure (1) of the run that left it out, so gating the failure rate keeps omitted trials visible even when `max_missing_fraction` is raised.

**Pairing.** `pairing` controls whether trials are matched by `trial_id`:

| `pairing` | Behavior |
|---|---|
| `auto` (default) | Paired if both runs have exactly the same set of `trial_id`s. Otherwise unpaired, with a note |
| `paired` | Paired on the `trial_id`s present in both runs. Unmatched trials count as missing (see above); if `max_missing_fraction` allows them, they are excluded and listed. `INVALID` if no ids overlap |
| `unpaired` | Two independent samples |

In paired mode, a pair is used only if both values are present.

A `trial_id` names the trial *condition*, for example "scenario variant 3, seed 1042".
Pairing says that both runs faced the same condition.
It does **not** assume the rollouts are identical: GPU reductions, multithreaded physics, asynchronous sensors and learned policies can all diverge from the same seed.
Pairing simply removes the part of the variation that comes from the shared condition, which usually makes the comparison much more sensitive.

**Clusters.** With `cluster_by: <label>`, trials are grouped by the value of that trial label, and the comparison is made between the groups' means instead of between trials.
`pairing` then applies to the clusters: `auto` matches them when both runs have exactly the same set of label values.
Trials are not matched to each other in this mode.
Every trial must carry the label, otherwise the metric is `INVALID`.
See [clusters](#clusters-several-training-runs-per-side) below for why and when.

## Step 4: point estimates (the numerical layer)

- `baseline`, `candidate`: the statistic of each sample. With `cluster_by`, the mean of the cluster means, so every cluster weighs the same whatever its number of trials.
- `delta = candidate - baseline`.
- `degradation`: `delta` signed by direction (positive is worse).
- `relative_degradation = degradation / |baseline|`, or `null` when the baseline statistic is 0.
- `margin`: see below.
- `exceeds_margin = degradation > margin`.

`exceeds_margin` is reported for transparency but never decides the verdict on its own.

### Margins

Each gated metric needs at least one of:

- `max_absolute_degradation`: in the metric's unit, `>= 0`.
- `max_relative_degradation`: a fraction of `|baseline statistic|`, `>= 0`. `0.05` means 5%.

If both are set, the **stricter** (smaller) margin applies.

**Baseline of zero, or of both signs.** A relative margin needs a ratio scale.
A relative margin of a zero baseline is zero, which would turn any increase into a regression.
The same happens, less visibly, when the baseline values have both signs, as for a signed tracking error: their mean can be arbitrarily close to 0, and so can the margin.
So when the baseline statistic is exactly 0, or the baseline sample contains both negative and positive values:

- if an absolute margin is also set, it is used alone, and the report says so,
- otherwise the metric is `INVALID`, with a message asking for an absolute margin.

Set `max_absolute_degradation: 0` to tolerate no degradation at all.

## Step 5: statistical evidence

If the number of usable observations is below `min_samples` (default 10), no interval is computed and the evidence is `insufficient_samples`.
For paired comparisons the count is the number of pairs; for unpaired comparisons it is the size of the smaller sample.
With `cluster_by`, the count is the number of clusters (matched clusters when paired) and the floor is `min_clusters` (default 3).

Otherwise an interval of the degradation is computed.
The method depends on the data.

### Continuous metrics: expanded percentile bootstrap

Unless every value in both samples is exactly 0 or 1, a bootstrap estimates the uncertainty of the degradation:

- **Paired:** resample pair indices with replacement, keeping each pair together, and recompute the statistic of each side.
- **Unpaired:** resample each run's values independently.
- Each replicate computes `degradation*` and `excess* = degradation* - margin*`.
  For relative margins, `margin*` is recomputed from the replicate's baseline statistic, so uncertainty in the baseline's magnitude is propagated.
- `bootstrap_resamples` (default 10,000) replicates are drawn from a pseudo-random generator seeded with `"<bootstrap_seed>:<metric name>"`. Results are therefore reproducible across machines.
- The two-sided interval at `confidence` (default 0.95) is taken from the replicate quantiles (linear interpolation).
- By default the quantiles are *expanded* for small samples (Hesterberg 2015, *The American Statistician* 69:371).
  Instead of the `alpha/2` and `1 - alpha/2` quantiles, the interval uses `Phi(-+sqrt(n/(n-1)) * t)`, where `t` is the Student t quantile for `alpha/2`.
  For pairs, `n` is the number of pairs and the t distribution has `n - 1` degrees of freedom.
  For two independent samples, `n / (n - 1)` becomes the ratio of the unbiased variance of the difference (`s_b^2 / n_b + s_c^2 / n_c`) to the plug-in variance the bootstrap sees, and the degrees of freedom are Welch-Satterthwaite's.
  For two samples of equal size and spread this is the same as `n / (n - 1)` with `2n - 2` degrees of freedom.
  For a small candidate against a large stored baseline, the degrees of freedom are close to `n_candidate - 1`, because almost all of the uncertainty comes from the candidate.
  At 10 trials this turns the 95% interval into roughly a 98% one; at 100 trials the change is negligible.

The plain percentile interval is too narrow for small samples.
In the study below it was wrong at the margin in up to 5.5% of runs per side with 10 trials, against a nominal 2.5%.
The expanded interval brings that to at most 3.4%, and costs little power: with 100 trials, an unchanged metric still PASSed in 93.7% of runs, against 93.8% with plain percentiles.
BCa intervals (bias-corrected and accelerated) were also measured, with an implementation checked against `scipy.stats.bootstrap`, and did not help: up to 7.6% per side with 10 skewed trials.
Set `defaults.bootstrap_method: percentile` to reproduce comparisons made before v0.3.0.

### 0/1 metrics: Newcombe score interval

When the statistic is `mean` and every value in both samples is exactly 0 or 1 (success flags, collision indicators, `physim.failure_rate`), the difference of the two rates uses a score interval instead:

- **Unpaired:** Newcombe's hybrid score interval, built from the Wilson intervals of both rates (Newcombe 1998, *Statistics in Medicine* 17:873, method 10).
- **Paired:** Newcombe's interval for paired proportions, which adds the continuity-corrected correlation of the pairs (Newcombe 1998, *Statistics in Medicine* 17:2635, method 10).

The reason is a failure of the bootstrap that matters in practice.
If a run has no events at all, for example 0 collisions in 30 trials, every bootstrap replicate also has 0 events, so the bootstrap reports the rate as known to be exactly 0.
Version 0.1.0 used the bootstrap here. In a 400-run simulation with 30 trials per run, it called a true increase from 1% to 6% (margin 2 points) "no regression" in 15% of runs, and at exactly the margin it said PASS in 38% of runs instead of at most about 2.5%.
The score interval correctly reflects that 0 of 30 is compatible with a true rate of about 11%.

For 0/1 metrics with a relative margin, the margin is computed once from the baseline's observed rate rather than per replicate.
The unpaired implementation reproduces the published example (56/70 vs 48/80 gives 0.0524 to 0.3339); see `tests/test_stats.py`.

### Clusters: several training runs per side

A gate over the trials of one run answers "is this checkpoint worse on these conditions?".
It does not answer "is this training recipe, dataset or code change worse?", because a second training run of the same recipe, with another seed, scores differently.
The trials of one run share that run's luck, so they are not independent evidence about the recipe.
Pooling the trials of several runs as if they were independent makes the interval too narrow, and so does a single run per side.

With `cluster_by`, each cluster (typically one training run) contributes its mean, and the interval is a Student t interval over those means:

- **Paired** (matched label values): a paired t interval of the per-cluster differences, with `k - 1` degrees of freedom for `k` matched clusters.
- **Unpaired:** Welch's t interval of the two sets of cluster means, with Welch-Satterthwaite degrees of freedom.

The trials within a cluster still matter: more of them make each cluster mean more precise, and conditions shared by every run (the same initial states, say) cancel out of the differences between cluster means.
If the cluster means do not vary at all, their spread says nothing about the uncertainty (every run 0 of 100, say), so the interval falls back to the trials, compared as independent samples, and the report says so.

`physim-gate compare` sets this up when it is given several files for a side: `-b run1.json -b run2.json -b run3.json -c ...`.
It combines each side's files into one run, labels every trial with `physim.run` (the run's id, or a shortened file name when ids repeat), and sets `cluster_by: physim.run` for every metric that does not set `cluster_by` itself.
Run names rarely match across the two sides, so runs are compared as independent samples; to match training seeds, label the trials with the seed yourself and set `cluster_by` to that label.

With few clusters, the t interval is wide.
That is the honest answer: with three training runs per side, the spread between runs can only be estimated roughly.
`physim-gate plan --from` a clustered pilot says how many runs a decision needs.
Choosing a t interval rather than a bootstrap or a random-effects model was a measured decision.
`scripts/cluster_method_study.py` (it needs NumPy and SciPy) simulated 4,000 repetitions per row, modeled on 500-episode LIBERO evaluations (42% success, initial states shared by every run), and compared four intervals with 1, 3, 5 and 10 runs per side.
At the margin, a wrong `PASS` and a wrong `FAIL` should each happen in at most about 2.5% of repetitions:

| Spread between runs (sd of a run's success rate) | Pooled trials: wrong verdict per side | t over run means: wrong verdict per side | t over run means: FAIL at margin + 4 points, 3 / 5 / 10 runs |
|---|---|---|---|
| none | 2.3% to 2.8% | 2.2% to 3.0% | 38% / 81% / 100% |
| 1.4 points | 5.8% to 7.1% | 1.9% to 2.6% | 26% / 60% / 96% |
| 4.2 points | 22% to 25% | 2.1% to 2.6% | 10% / 20% / 43% |

A random-effects meta-analysis over runs (Hartung-Knapp) gave the same results as the t interval, and its modified form (variance factor at least 1) lost most of its power at 3 runs, so the simpler t interval is used.
Pooled trials stay accurate only when runs do not differ at all, which a handful of runs cannot establish.

### Classification

The interval is always computed for the **difference** between the runs.
PhySim Gate never compares two separate per-run intervals for overlap, because overlap is not a valid significance test in general.

The evidence is classified against the margin using the `excess` interval `[lo, hi]`:

| Condition | Evidence | Verdict |
|---|---|---|
| `lo > 0` | `regression` | `FAIL` |
| `hi <= 0` | `non_inferior` | `PASS` |
| otherwise | `inconclusive` | `INCONCLUSIVE` |

Deciding on one side of a two-sided interval at confidence `c` targets a one-sided error rate of `(1 - c) / 2`, so 2.5% at the default 0.95.
The measured rates below show how close the implementation gets.

**Multiple metrics.** With `policy.correction: bonferroni`, each metric uses confidence `1 - (1 - c) / k`, where `k` is the number of gated metrics.
This widens every interval, which makes both `FAIL` and `PASS` harder to reach.
The default is `none`.

**Zero variance.** If both samples of a continuous metric are constant (a fully deterministic simulator), every replicate is identical and the interval has zero width.
The comparison then reduces to an exact check of the degradation against the margin, which is the correct behavior for deterministic results.
This does not hold for 0/1 metrics: constant 0/1 values use the score interval, which stays wide.
For example 0 events in 50 trials against 0 in 50 gives an interval of about plus or minus 0.07, so a margin of 0.01 is `INCONCLUSIVE`.
0 of 50 is compatible with a true rate of several percent, so that is the honest answer.

## Step 6: policy

Each metric verdict is `blocking` if the metric is `required` (default true) and:

| Verdict | Blocking when |
|---|---|
| `FAIL` | always |
| `INCONCLUSIVE` | `policy.on_inconclusive: fail` (default) |
| `INVALID` | `policy.on_invalid: fail` (default) |
| `PASS` | never |

Non-required metrics are evaluated and reported, but never block.

The overall verdict, in order of precedence:

1. `INVALID` if there is a blocking run-level issue,
2. `INVALID` if no gated metric could be compared at all, whatever `policy.on_invalid` and `required` say: a gate that compared nothing has shown nothing,
3. `FAIL` if any blocking metric is `FAIL`,
4. `INVALID` if any blocking metric is `INVALID`,
5. `INCONCLUSIVE` if any blocking metric is `INCONCLUSIVE`,
6. `PASS` otherwise.

When no metric is `required`, nothing can block, and the summary starts with "no metric is required" so a report-only gate is not mistaken for a passing one.

Exit codes: `PASS` is 0, `FAIL` is 1, `INCONCLUSIVE` and `INVALID` are 3.
Exit code 2 is reserved for errors that prevent a comparison (bad arguments, unreadable or invalid files), and 4 for internal errors.

## Why the default is conservative

The default answers "is the candidate shown to be no worse than the margin?", not "is the candidate shown to be worse?".
A gate built on the second question passes whenever the data is too noisy to show anything, which is exactly when a regression is most likely to slip through.
Teams that prefer to block only on demonstrated regressions can set `policy.on_inconclusive: pass` explicitly, and the reports will still flag every inconclusive metric.

## Measured error rates

`scripts/error_rate_study.py` runs the real engine on synthetic data with a known true degradation, 1,000 times per row, at confidence 0.95 with 2,000 bootstrap resamples and the default expanded percentile bootstrap.
At the margin both PASS and FAIL are errors, and each should be at most about 2.5%.
Away from the margin, the wrong verdict should be near 0 and the correct one should grow with the number of trials.

| Scenario | Trials per run | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|---|
| Normal, unpaired, at margin | 10 / 30 / 100 | 2.0% / 2.1% / 1.6% | 2.4% / 2.4% / 2.5% | rest |
| Normal, unpaired, 1,000 baseline trials, at margin | 10 / 30 / 100 candidate trials | 2.2% / 2.9% / 2.8% | 2.5% / 2.7% / 3.4% | rest |
| Normal, paired (correlation 0.8), at margin | 10 / 30 / 100 | 2.5% / 2.3% / 1.7% | 3.1% / 2.6% / 2.6% | rest |
| Lognormal, mean +10%, relative margin 10% | 10 / 30 / 100 | 3.1% / 2.4% / 3.1% | 3.0% / 2.9% / 2.4% | rest |
| Normal, unpaired, no change (correct: PASS) | 10 / 30 / 100 | 17.9% / 47.4% / 93.7% | 0.3% / 0% / 0% | rest |
| Normal, unpaired, degradation 2x margin (correct: FAIL) | 10 / 30 / 100 | 0% / 0% / 0% | 17.8% / 47.5% / 94.2% | rest |
| Success 0.90 to 0.85, margin 0.05, unpaired, at margin | 30 / 100 / 300 | 2.4% / 3.2% / 2.9% | 2.5% / 2.8% / 2.1% | rest |
| Success 0.90 to 0.85, margin 0.05, paired, at margin | 30 / 100 / 300 | 0.1% / 3.0% / 2.6% | 0.7% / 0.8% / 2.0% | rest |
| Collision 0.01 to 0.03, margin 0.02, unpaired, at margin | 30 / 80 / 300 | 0% / 0.5% / 1.8% | 0% / 0.5% / 1.6% | rest |
| Collision 0.01 to 0.06, margin 0.02, unpaired (correct: FAIL) | 30 / 80 / 300 | 0% / 0% / 0.1% | 0.5% / 11.4% / 54.4% | rest |
| Collision 0.01 to 0.06, margin 0.02, paired (correct: FAIL) | 30 / 80 / 300 | 0% / 0% / 0% | 0.2% / 7.6% / 63.9% | rest |

Several training runs per side, at the margin (`--match runs`; the second column is runs per side, 50 or 100 trials each):

| Scenario | Runs per side | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|---|
| Normal, run-to-run sd 0.5, trials pooled | 3 / 5 / 10 | 24.4% / 23.8% / 25.2% | 27.5% / 29.4% / 26.7% | rest |
| Normal, run-to-run sd 0.5, `cluster_by` | 3 / 5 / 10 | 1.1% / 2.8% / 1.9% | 1.8% / 1.9% / 3.0% | rest |
| Normal, no run-to-run spread, `cluster_by` | 3 / 5 / 10 | 1.5% / 2.4% / 2.5% | 1.9% / 2.3% / 2.1% | rest |
| Success 0.42, run-to-run sd 3 points, margin 0.02, trials pooled | 3 / 5 / 10 | 4.8% / 5.1% / 4.1% | 4.8% / 4.6% / 4.2% | rest |
| Success 0.42, run-to-run sd 3 points, margin 0.02, `cluster_by` | 3 / 5 / 10 | 2.4% / 1.8% / 1.7% | 1.7% / 2.9% / 2.6% | rest |

What this shows:

- The expanded bootstrap stays close to nominal from 10 trials on: at most 3.4% per side. With plain percentiles the same rows reached 5.5% (v0.2 and earlier; `--method percentile` reproduces them).
- Very unequal samples are covered too. Up to v0.3.1 the degrees of freedom were `n_baseline + n_candidate - 2`, and 10 candidate trials against 1,000 baseline trials were wrong at the margin in 4.3% and 4.7% of runs per side (4,000 repetitions).
- The price is power at very small samples: with 10 trials an unchanged metric PASSes in 18% of runs instead of 25%. From about 30 trials the difference is a few points, and at 100 it is gone.
- The score intervals for 0/1 metrics stay near or below nominal. For rare events they are conservative, which means many `INCONCLUSIVE` results unless there are hundreds of trials.
- Pooling the trials of several training runs is wrong at the margin far more often than nominal as soon as runs differ, up to 29% per side here. Comparing run means with `cluster_by` stays at or below 3.0%.
- The Monte Carlo uncertainty of each percentage is roughly plus or minus 1 point.

## Planning the number of trials

`physim-gate plan` estimates how many trials per run make the intended verdict likely.
You state the margin, what you expect the true degradation to be (default 0), the wanted probability of the intended verdict (`--power`, default 0.8) and the interval level (`--confidence`, default 0.95).
If the expected degradation is below the margin the intended verdict is `PASS`, and if it is above, `FAIL`.
Exactly at the margin neither verdict is expected, and the command says so.

For 0/1 metrics (`--rate`), the probabilities are exact for unpaired runs.
For every pair of event counts the planner computes the engine's unpaired Newcombe interval, decides the verdict, and weights it by the binomial probability of that pair.
The interval's bounds rise with the candidate's count and fall with the baseline's, so for each baseline count the decisive candidate counts are found with one sweep instead of a full search.
Tests compare the result with brute-force enumeration (agreement within 1e-12) and with the engine itself on simulated runs.
The Monte Carlo study above found `FAIL` in 11.4% of runs for "collision 0.01 to 0.06, unpaired, 80 trials"; the exact probability is 11.1%.

Power for 0/1 data rises unevenly with the number of trials, so a few trials more than the reported number add a little safety.

For paired 0/1 runs (`--discordance D`), the interval depends on all four cells of the 2x2 table of pairs, so the planner simulates instead: for each candidate size it draws 4,000 tables from the cell probabilities and decides each with the engine's paired Newcombe interval.
The cell probabilities follow from the baseline rate, the discordance `D` (the share of pairs whose outcomes differ) and the assumed degradation: of the discordant pairs, `(D + d) / 2` are bad only in the candidate and `(D - d) / 2` only in the baseline.
A test checks the simulation against the engine on simulated runs.
Pairing helps when the runs disagree less often than independent runs would (`2p(1-p)` for a rate `p`).
For a 90% success rate and a 5-point margin, 4% discordance needs 173 trials per run, 8% needs 284, and 20%, which is more than independent runs' 18%, needs 641 compared with 573 unpaired.

`--from comparison.json` plans every gated metric of a pilot comparison from its own data: the baseline rate or the spread (of the paired differences when paired), the discordance of paired 0/1 metrics, the margin as evaluated in the pilot (relative margins at their pilot value), the confidence after any correction, and `min_samples`.
The assumed true degradation is the pilot's observed one unless `--true-degradation` is given.
Metrics the pilot could not evaluate, and paired 0/1 metrics whose pilot pairs all agreed, are listed with the reason.
A pilot with fewer than `min_samples` observations has no interval, which is when planning is needed most; it is planned with the method and confidence the gate would use (the 0/1 method for 0/1 values, the configured confidence after correction).

A clustered pilot (`cluster_by`, or several files per side) is planned in clusters per side instead of trials, from the spread of its cluster means (of the per-cluster differences when matched).
The planner treats the estimate as normal and takes the critical value from the t distribution the engine will use (`k - 1` degrees of freedom for `k` matched clusters, `2k - 2` for two sides of `k`).
A test checks it against the engine on simulated clusters.
For the LeBoard calibration runs on LIBERO-10 (5 clean runs against 3 runs with video 3 frames late), the observed 3.8-point drop would need 8 runs per side to be shown as a regression with 80% probability, and a true 2-point drop 26.

For continuous metrics (`--stdev`), the planner uses the normal approximation

```text
n = ((z_c + z_p) * sd_diff / |margin - degradation|)^2
```

where `z_c` is the normal quantile for the interval level, `z_p` the one for the power, and `sd_diff` the standard deviation of the per-trial difference: the given `--stdev` with `--paired`, otherwise `--stdev` times the square root of 2.
The engine's bootstrap does not follow this formula exactly, and the difference grows as the number of trials shrinks, so treat small answers with care.

Some reference points at the defaults (confidence 0.95, power 0.8, unpaired):

| Metric | Expected | Margin | Trials per run |
|---|---|---|---|
| Success rate 0.90 | unchanged | 0.05 | 573 |
| Success rate 0.98 | unchanged | 0.02 | 843 |
| Success rate 0.50 | unchanged | 0.05 | 1,579 |
| Collision rate 0.02 | rises to 0.05 | 0.01 | 1,280 (to `FAIL`) |
| Continuous, stdev 2.0 | unchanged | 0.5 | 252 (126 if paired with stdev of differences 2.0) |

## Limitations

- Percentile bootstrap intervals undercover with small samples and skewed distributions (see the table above). `min_samples` is a floor, not a guarantee of accuracy. Gating a 1% event rate with confidence needs hundreds or thousands of trials.
- The median's bootstrap distribution is coarse for small samples. The median of 0/1 data still uses the bootstrap and inherits its zero-event weakness; use the mean for 0/1 metrics.
- The paired score interval is validated by simulation (the table above), not against a published worked example.
- Trials are assumed exchangeable within a run. Strong drift within a run (for example thermal effects in hardware-in-the-loop) violates this.
- Without `cluster_by`, a comparison is about the runs given, not about the process that produced them; see [clusters](#clusters-several-training-runs-per-side). With `cluster_by`, clusters are assumed exchangeable, and their means roughly normal.
- Repeating a comparison until it passes ("optional stopping") invalidates the stated error rates.
- Statistical evidence does not establish practical relevance. The margin encodes that, and it is your decision.

## Configuration reference

```yaml
version: 1
defaults:
  statistic: mean              # mean | median
  confidence: 0.95             # in (0.5, 1)
  min_samples: 10              # >= 2
  pairing: auto                # auto | paired | unpaired
  max_missing_fraction: 0.0    # 0..1
  bootstrap_resamples: 10000   # 1000..1000000
  bootstrap_seed: 0
  bootstrap_method: expanded_percentile   # expanded_percentile | percentile
  cluster_by: null             # a trial label, e.g. training_seed; physim.run for several files
  min_clusters: 3              # >= 2
policy:
  on_inconclusive: fail        # fail | pass
  on_invalid: fail             # fail | pass
  correction: none             # none | bonferroni
  require_same_scenario: true
metrics:
  <name>:
    max_absolute_degradation: <float >= 0>   # at least one of these two
    max_relative_degradation: <float >= 0>
    direction: higher_is_better | lower_is_better   # optional cross-check
    required: true
    statistic / confidence / min_samples / pairing / max_missing_fraction   # per-metric overrides
    cluster_by / min_clusters                                               # per-metric overrides
```

`physim-gate schema config` prints the JSON Schema of this file.
