# EpiNet — methods

Evaluation design, diagnostics, and the contestability theory. Companion to the [README](../README.md). All of EpiNet is a research/education demonstrator — see [Methodological boundaries](#methodological-boundaries).

## Iterative Evaluation

On small networks a single train/test split is a noisy estimate: the same model
can score 0.35 or 0.55 accuracy depending on which nodes land in the test set.
By default the toolkit re-evaluates the model on 10 different splits
(`--n-iterations 10`) with hyperparameters tuned once on the primary split, and
reports the spread:

```json
"iteration_summary": {
  "accuracy": {"mean": 0.44, "std": 0.06, "min": 0.35, "max": 0.55}
}
```

Tuning is **imbalance-aware**: the search grid includes `class_weight`
(`None`, `balanced`, `balanced_subsample`) and selects on `balanced_accuracy`
(unweighted mean recall across classes) rather than a support-weighted score.
A support-weighted score is dominated by the majority class and would never
select the minority weighting, so on the skewed outcomes typical of
epidemiology the model would otherwise collapse toward the majority. On
balanced data the search simply selects `class_weight=None`, leaving behaviour
unchanged.

The grid also tunes `min_samples_leaf` (`1`, `3`) to regularize the forest on
the small, noisy cohorts this toolkit targets. Because it is selected by the
same cross-validation, a larger leaf is chosen only when it improves held-out
balanced accuracy, so it never degrades the selected model.

If the mean is near chance, the graph features carry no signal for the outcome —
which is exactly what the bundled random synthetic data shows. Use
`--n-iterations 1` to reproduce the old single-split behavior, or raise it for
tighter estimates.

## Permutation Null Model

"Near chance" should be measured, not eyeballed. `--permutation-test N` shuffles
the outcome labels N times and re-evaluates with the same tuned configuration
and split scheme, producing an empirical null distribution and a one-sided
p-value per metric:

```bash
python -m epinet.toolkit --permutation-test 100 --no-run-paths
```

```json
"permutation_test": {
  "n_permutations": 100,
  "metrics": {
    "f1_weighted": {"observed_mean": 0.53, "null_mean": 0.50, "null_std": 0.13, "p_value": 0.44}
  }
}
```

Under a community-aware split a shuffled draw can leave an outcome class entirely
in the held-out communities (no valid score); such draws are skipped rather than
crashing the null, and the block also reports `n_permutations_used` /
`n_permutations_skipped` so the p-value denominator reflects only valid draws.

A p-value like 0.44 means shuffled labels score as well as the real ones almost
half the time — the features carry no detectable signal. On the bundled random
synthetic data this is the correct conclusion; on real data, demand a small
p-value before trusting any importance ranking.

## Community-Aware Splitting

Random train/test splits assume independent samples, but connected nodes are
not independent: a node's graph features encode information about neighbors
that may sit in the test set. `--split-strategy community` detects communities
(greedy modularity) and keeps each one entirely in train or test:

```bash
python -m epinet.toolkit --split-strategy community --n-iterations 10
```

Scores are typically lower and more variable than with random splits — that is
the honest estimate of how the model generalizes to an unseen region of the
network. Stratification is disabled in this mode (group splits and class
stratification are incompatible), and if the graph collapses into a single
community the run falls back to random splits and records a `split_note` in
the metrics.

## Visualization

Every run writes figures to `<output-dir>/plots/` (disable with `--no-make-plots`):

- `network_overview` — spring layout colored by outcome, target nodes outlined,
  nearest source→target paths highlighted
- `degree_distribution`
- `feature_importance` — permutation importance with ±1 sd error bars
- `metric_stability` — box plot of the 0–1 metrics (accuracy, balanced accuracy,
  F1, AUROC) across iterations, with the individual iterations overlaid as
  jittered points
- `confusion_matrix` — counts plus row-normalized recall, labeled colorbar
- `calibration` — reliability diagram (predicted vs observed risk) with a
  prediction histogram strip, for binary outcomes
- `learning_curve` — cross-validated F1 vs training-set size, with ±1 sd bands
- `permutation_null` — null distribution vs observed F1 (with `--permutation-test`)
- `feature_clusters` — PCA projection with explained-variance axis labels
- `contestability` — flip-distance histogram with the contested tail shaded, beside
  the value-of-information bar chart (with `--run-contest`)

All figures share one house style (consistent typography, no chartjunk spines,
colorblind-friendly Okabe-Ito palette) and render at 300 DPI by default. Use
`--plot-format pdf` (or `svg`) for vector output and `--plot-dpi N` to change the
raster resolution.

### Interactive network

Add `--interactive-network` to also write `plots/network.html`: a draggable,
zoomable, hover-labeled network rendered by vis-network (loaded from a CDN — no
extra Python dependency). It stays readable on graphs far larger than a static
spring layout can show, and colors nodes by outcome (blank = gray scaffold).


## Methods: Contestability and the Limits of "More"

Across the worked examples one result keeps recurring, and it is worth stating as
a method rather than re-discovering per dataset: **adding tests does not rescue
the cases the tests disagree on.** More features (`score_comparison`), fitted
fusion (`test_fusion`), and — by the same argument — hand-set heuristics all fail
in the same region for the same reason. Where discrimination on the disagreement
subset collapses to chance (AUC ≈ 0.5), the separating information is *not present
in the features*. No function of those features recovers it: not a fitted model,
not a centroid average, not an `if/else` rule. The grey zone is
**information-limited, not method-limited** — and a heuristic is just a hand-set
decision surface over the same inputs, inheriting the same fragility while
dropping the null model and CIs that would expose it.

That reframes what the toolkit is for in the contested region. The job is not to
decide the hard cases but to **measure and route** them: label each case with
*how contestable it is*, *whether that contest is structured or idiosyncratic*,
and *what additional measurement (if any) would resolve it*. Four existing lenses
already compose into that map.

### 1. Where is the contest, and is it structured? (implemented)

`divergence_topography.py` is the empirical contestability map: two labelings of
the same objects, a per-object `concordant`/`discordant` flag, and a permutation
null that asks whether the discordance is *locatable in feature space* or just
labeling noise.

```bash
python examples/divergence_topography.py
python -m epinet.toolkit \
  --nodes examples/lidc_divergence_nodes.csv --edges examples/lidc_edges.csv \
  --outcome-column Outcome --split-strategy community \
  --permutation-test 200 --no-run-paths \
  --output-dir examples/lidc_divergence_outputs
```

Read the p-value as the verdict: a structured contest (worth a feature-space
explanation) versus an idiosyncratic one (no amount of modeling will tidy it). On
LIDC it is structured but only shallowly (p ≈ 0.015, barely over base rate) — an
honest "mostly irreducible" answer, reported with its smallness intact.

### 2. Which cases sit on a boundary? (implemented)

The clustering lens already emits a per-case contestability signal: distance to
*every* outcome-class centroid, plus the flag for nodes whose nearest centroid
disagrees with their own label.

```bash
python -m epinet.toolkit --run-clusters --distance-metric mahalanobis \
  --cluster-labeled-only --output-dir epinet_outputs
```

In `node_clusters.csv`, a case whose two smallest class-centroid distances are
near-equal sits *on the boundary in feature space*, regardless of which side of a
score threshold it landed on. Those are the cases to route, not to trust.

### 3. Is the comparison even runnable here? (implemented)

`score_comparison.py` enforces the gap-population rule instead of papering over
it: a score is **not computed** when its dominant predictor domain is absent
(LIDC has no demographics, smoking, or growth), and the NTOG
normalize-by-available-weight rule makes that refusal explicit rather than
imputing a fake number. The honest output of a comparison can be "unmeasurable
here, and here is which missing domain caused it" — a property of the design, not
a failure of the run.

### 4. How far is the call from flipping? (`--run-contest`)

`epinet/contest.py` makes the boundary analytic. For the nearest-centroid
classifier the decision boundary between two classes is a hyperplane, so the
toolkit's `|s(x) − τ| / ‖∇s(x)‖` has a closed form — the **flip-distance**, the
smallest move in standardized feature space that reverses the call:

```
flip_distance(x) = min over competing class k of  (d_k² − d_a²) / (2·‖c_k − c_a‖)
```

where `d_a`/`d_k` are the node's distances to its nearest and a competing class
centroid. It is exact in both the Euclidean and shared-covariance Mahalanobis
metrics (in the latter, measured in the whitened space where the boundary is
again a hyperplane).

```bash
python -m epinet.toolkit \
  --nodes examples/nodule_nodes.csv --edges examples/nodule_edges.csv \
  --outcome-column Outcome --run-contest --distance-metric mahalanobis \
  --no-run-model --no-run-paths --no-run-clusters --no-make-plots \
  --output-dir examples/nodule_contest_outputs
```

`node_contestability.csv` gives, per node: `flip_distance`, the binding
`runner_up_class` (which class it would flip *to*), a `contested` flag for the
most fragile fraction (`--contest-quantile`, default lowest decile), and the
**value-of-information** columns — `most_decision_relevant_feature` and its
`single_axis_flip_distance`, the single input the call is most sensitive to. On
the nodule cohort the most-contested cases sit between adjacent risk tiers with
flip-distances near zero, and the decisive feature is consistently diameter — a
machine-checkable echo of the score-comparison finding that diameter carries the
discrimination. `contest_summary.json` adds the cohort-level flip-distance
distribution and a global feature-leverage ranking, `contestability_report.md`
is a ready-to-read table of the most-contested cases and the value-of-information
ranking, and (under `--make-plots`) `plots/contestability.png` shows the
flip-distance histogram with the contested tail shaded beside the
value-of-information bars.

Two cautions are load-bearing, and are written into the output's `caveats` field
rather than left to the reader:

- A gradient is only as meaningful as the surface under it. EpiNet's ported scores
  are explicitly unvalidated; flip-distance computed on them measures the
  **fragility of the score**, not the borderline-ness of the patient. Keep those
  two claims separate, or it is confident nonsense.
- `flip_distance` is in standardized-feature units. It is contestable in a way
  that *matters* only when it is *smaller than the real-world measurement error*
  of the inputs in the same units ("this call reverses if the nodule were measured
  0.4 SD differently"). The module reports the number; comparing it to measurement
  error is a domain step it does not take for you.

## Methodological Boundaries

The model is intentionally simple. It does not infer causality, outbreak dynamics,
clinical risk, or intervention effects. Network features can be useful descriptors,
but they can also encode sampling bias, measurement bias, and structural confounding.

Use the outputs as exploratory evidence, not as decisions.

Before using this for health, education, welfare, employment, or public-sector
decisions, add:

- domain-specific data validation
- directed/temporal assumptions
- uncertainty and sensitivity checks
- external validation
- privacy and governance review
- human review of any operational recommendations

