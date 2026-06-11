# EpiNet

EpiNet is a **general node/edge network analysis toolkit**. You load entities and
relationships from CSVs and it computes graph features, trains and *honestly*
evaluates an outcome model, finds shortest paths, and clusters nodes by
feature-space centroid — with publication-quality figures and an interactive
network view. Epidemiology is one use case; the core logic is intentionally
domain-neutral (it has been driven through lung cancer quality indicators, lung
nodule risk, and lymphoma subtyping).

EpiNet is developed and maintained as part of **Vahian**. It is released under
the permissive MIT license (see `LICENSE`).

This repository is a research and education demonstrator, **not** clinical or
public-health decision support. Any model it produces must be validated on
independent, outcome-linked data before it means anything clinically.

What distinguishes it from a thin scikit-learn wrapper is that the honest
evaluation is the default path: a label-permutation null and (where appropriate)
community-aware splitting run alongside the headline metric, so a good score
reflects real signal rather than leakage or chance.

## What Is Implemented

- CSV node and edge loading.
- **Input normalization** (on by default; `--no-normalize` for strict mode): maps
  common column aliases (`patient_id`→`ID`, `from`/`to`→`SourceID`/`TargetID`,
  `label`→`Outcome`, …) onto the canonical schema before validation, so messy
  real-world CSVs run without hand-editing. Never silent — every rename is logged
  to `ingest_report.json` and folded into provenance, which hashes both the raw
  input file and the normalized table.
- Network construction with NetworkX.
- Node-level graph features:
  - degree
  - weighted degree
  - clustering
  - component size
  - isolate flag
  - optional betweenness, closeness, and PageRank
- Optional RandomForest outcome model using graph features plus numeric node attributes.
- Semi-supervised support: nodes with a blank outcome are treated as unlabeled
  scaffold — they shape the graph features but are excluded from training, so a
  graph can mix labeled and context/infrastructure nodes.
- **Discrimination and calibration metrics**, not just accuracy: AUROC and
  average precision (AUPRC), balanced accuracy and Matthews correlation
  coefficient (robust under class imbalance), plus a Brier score and a
  calibration slope/intercept — because a risk score that discriminates but is
  miscalibrated is misleading exactly where decisions are contestable.
- Iterative model evaluation over repeated train/test splits, reporting the
  mean, standard deviation, and range of each metric — labelled honestly as
  split-to-split *variability*, which understates true uncertainty (Nadeau &
  Bengio, 2003), and complemented by a **within-split percentile-bootstrap 95%
  confidence interval** (`--n-bootstrap N`).
- **Permutation feature importance** measured on held-out data (less biased than
  impurity importance, which is retained alongside for reference).
- Community-aware splitting (`--split-strategy community`): whole graph
  communities stay on one side of the split, so scores estimate generalization
  to unseen regions of the network instead of leaking structure between
  connected train/test nodes.
- Label-permutation null model (`--permutation-test N`) with empirical,
  direction-aware p-values (computed with the *same* multi-iteration averaging as
  the observed score, and reported with a multiple-comparisons caveat), so
  observed scores are compared against chance instead of read in isolation.
- **Reproducibility provenance** stamped into every run (`provenance.json`, and
  embedded in `model_metrics.json`): git commit + dirty flag, package versions,
  random seed, and a SHA-256 of each input file.
- **TRIPOD+AI-flavored model card** (`model_card.md`): a human-readable summary
  of intended use, data, performance (discrimination *and* calibration),
  validation, limitations, and provenance.
- Small-cohort guardrails: tiny test sets and rare classes are surfaced as
  explicit `data_warnings` rather than left implicit.
- Shortest-path analysis from source nodes to explicit target nodes or to nodes with a target outcome.
- Feature-space clustering: k-means centroids plus per-node distance to every
  outcome-class centroid (Euclidean or Mahalanobis) — a nearest-centroid view
  that complements the topological shortest paths with attribute-space distance.
- Contestability / flip-distance (`--run-contest`): the closed-form smallest
  feature-space move that flips a node's nearest-centroid class, the class it
  would flip to, and a per-feature value-of-information ranking (which single
  input the call is most sensitive to) — the analytic complement to the
  divergence/centroid view.
- Per-target coverage summaries — the counterpart to the per-source nearest-target
  table — showing how many sources reach each target and at what distance.
- CSV/JSON outputs for downstream inspection.
- Publication-style PNG figures (`epinet_viz.py`): network overview, degree
  distribution, feature importance, metric stability, confusion matrix,
  calibration reliability diagram, and learning curve.

The older `epinet-analysis.py` and `epinet-analysis-v2.py` scripts remain as early prototypes.
The recommended entry point is now `epinet_toolkit.py`.

## Install

```bash
pip install -e .            # installs the package + the `epinet` command
pip install -e ".[dev]"     # also pytest + ruff (for development)
pip install -e ".[lidc]"    # pylidc, for the LIDC-IDRI / LUNA16 examples
pip install -e ".[excel]"   # xlrd + openpyxl, for the TCIA diagnosis spreadsheets
```

`requirements.txt` still lists the core runtime dependencies if you prefer not to
install the package.

## Quick Start

```bash
epinet \
  --nodes synthetic_nodes.csv \
  --edges synthetic_edges.csv \
  --outcome-column Outcome \
  --target-outcome 1 \
  --output-dir epinet_outputs
```

(`epinet ...` is the installed console command; `python epinet_toolkit.py ...`
works identically without installing.)

## Worked examples

Each example has a builder script and a walkthrough (`examples/*_usecase.md`):

| Example | What it shows |
|---------|---------------|
| Nordic lung cancer QI | capability network; semi-supervised model; why community vs random splits change the conclusion |
| Synthetic nodule cohort | Brock/Mayo/VDT risk models (validated port) + centroid risk tiers |
| LIDC-IDRI / LUNA16 | real radiologist annotations; the "indeterminate" hedge tier; reader-disagreement topography |
| Pathology validation | radiologist tier vs tissue diagnosis; the hedge bucket is 93% malignant; specification-curve sensitivity |
| Score comparison & fusion | Brock vs Mayo vs NTOG; does combining tests help? |
| **Lymphoma workflow** | turnkey digital-pathology → subtype classifier; five subtypes with real confusable pairs (CLL/MCL, DLBCL/Burkitt) + a grey-zone contestability read that names cyclin D1 as the marker to settle the hard cases (`examples/lymphoma_workflow_usecase.md`) |
| NLST harness | ingestion stub for the NLST dataset (run when CDAS data arrives) |

## Quick Start (continued)

This runs the two main lenses side by side:

1. graph feature generation and a simple outcome model
2. shortest-path summaries from non-target nodes to target outcome nodes

Generated files include:

- `graph_summary.json`
- `node_features.csv`
- `shortest_paths.csv`
- `nearest_targets.csv` — per source: nearest target and best path
- `target_coverage.csv` — per target: how many sources reach it, min/mean/max distance
- `model_metrics.json` — primary-split metrics (discrimination, classification,
  calibration), `iteration_summary`, bootstrap CI, permutation test, data
  warnings, and an embedded `provenance` block
- `model_card.md` — TRIPOD+AI-flavored human-readable model card
- `model_feature_importance.csv` — permutation importance (`importance` ±
  `importance_std`) with `impurity_importance` retained for reference
- `model_iteration_metrics.csv` — one row of metrics per evaluation iteration
- `model_permutation_metrics.csv` — one averaged row per null-model permutation (with `--permutation-test`)
- `provenance.json` — git commit, package versions, seed, and input SHA-256s
- `run_summary.json`
- `plots/*.png` — see Visualization below

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
python epinet_toolkit.py --permutation-test 100 --no-run-paths
```

```json
"permutation_test": {
  "n_permutations": 100,
  "metrics": {
    "f1_weighted": {"observed_mean": 0.53, "null_mean": 0.50, "null_std": 0.13, "p_value": 0.44}
  }
}
```

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
python epinet_toolkit.py --split-strategy community --n-iterations 10
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

## Shortest-Path Examples

Use outcome-positive nodes as targets:

```bash
python epinet_toolkit.py --outcome-column Outcome --target-outcome 1
```

Use explicit target nodes:

```bash
python epinet_toolkit.py --target-nodes Node_1,Node_5 --no-run-model
```

Limit the sources:

```bash
python epinet_toolkit.py --source-nodes Node_0,Node_9 --target-nodes Node_42 --no-run-model
```

Treat edges as directed:

```bash
python epinet_toolkit.py --directed --target-nodes Node_42 --no-run-model
```

Use an edge weight column as path distance:

```bash
python epinet_toolkit.py --weight-column Weight --path-mode distance --target-nodes Node_42 --no-run-model
```

Be careful: many datasets store edge weight as relationship strength, not distance.
If a larger weight means a stronger or more frequent connection, it should not be used
directly as a shortest-path distance without transformation.

If an edge column is a normalized 0..1 strength, you can ask for the strongest route:

```bash
python epinet_toolkit.py --weight-column Weight --path-mode strength --target-nodes Node_42 --no-run-model
```

## CiteMatch Evidence Graph Example

The `examples/` directory includes a small CiteMatch-style evidence graph:

- `examples/citematch_nodes.csv`
- `examples/citematch_edges.csv`
- `examples/citematch_usecase.md`

Run nearest contrast-evidence paths for three claims:

```bash
python epinet_toolkit.py \
  --nodes examples/citematch_nodes.csv \
  --edges examples/citematch_edges.csv \
  --outcome-column Outcome \
  --target-outcome contrast_evidence \
  --source-nodes claim_osimertinib_dfs,claim_osimertinib_os,claim_chemo_required \
  --no-run-model \
  --output-dir examples/citematch_outputs/contrast_paths
```

This is the safer non-epidemiology use case: the toolkit maps evidence structure
around claims and papers. It does not infer whether a claim is true.

For CiteMatch, avoid calling this a fastest path unless an edge column truly encodes
time or delay. The useful question is usually the best evidence route: nearest by
hops, lowest evidence distance, or strongest relationship path.

## Feature-Space Clustering

Shortest paths measure *topological* distance (hops/weights along edges).
Clustering measures *feature-space* distance instead: each node becomes a
standardized vector of graph features plus numeric attributes, k-means finds
centroids, and each node gets a distance to every outcome-class centroid — a
transparent nearest-centroid (Rocchio) view.

```bash
python epinet_toolkit.py --run-clusters --distance-metric mahalanobis --n-clusters 0
```

- `--n-clusters 0` uses the number of outcome classes when labels exist,
  otherwise selects k by silhouette score.
- `--distance-metric mahalanobis` accounts for correlated/scaled features (the
  graph centralities are highly correlated); `euclidean` is the isotropic default.

Outputs: `node_clusters.csv` (cluster id, distance to own centroid, distance to
each class centroid, nearest-centroid prediction), `cluster_centroids.csv`,
`cluster_summary.json` (silhouette, inertia, cluster×outcome composition,
nearest-centroid in-sample accuracy), and `plots/feature_clusters.png` (a PCA
projection colored by cluster). The nodes whose nearest class centroid disagrees
with their actual label are the feature-space outliers worth inspecting.
`--cluster-labeled-only` skips feature-less scaffold nodes (e.g. patient hubs).

## Pulmonary Nodule Cohort Example

A second real-domain example brings the feature-space clustering to lung-nodule
risk phenotyping, reproducing the published Brock/PanCan, Mayo/Swensen, and
volume-doubling-time models (NTOG lung-risk tools) to generate a synthetic
cohort:

- `examples/build_nodule_cohort.py` — generator (synthetic patients/nodules)
- `examples/nodule_{nodes,edges}.csv`, `examples/nodule_risk_scores.csv`
- `examples/nodule_cohort_usecase.md` — walkthrough and interpretation

Patients are scaffold hubs; nodules are labeled by risk tier and linked to their
patient and siblings, so a community split holds whole patients out. Predicting
risk tier from raw morphology survives that patient-aware split (F1 0.82,
p ≈ 0.005), and the centroid distances flag the boundary nodules whose phenotype
disagrees with their Brock threshold — the second opinion the static calculators
cannot give. The coefficient port is validated against the source formula and
the published odds ratios by `examples/validate_nodule_models.py`.

### Real LIDC-IDRI cohort

`examples/build_lidc_cohort.py` runs the same pipeline on real LIDC-IDRI
radiologist annotations (via `pip install pylidc`, no DICOMs needed): 875 scans,
2651 nodules labeled by median-reader malignancy tier. It is deliberately biased
data (subjective labels, a dominant "indeterminate" hedge tier, 29% of nodules
with ≥2-point inter-reader disagreement). Morphology predicts the tier under a
scan-aware split (F1 0.70, p ≈ 0.01); the model's errors funnel entirely through
the indeterminate middle (it never confuses benign with suspicious), and reader
disagreement turns out largely orthogonal to feature-space ambiguity. See
`examples/lidc_cohort_usecase.md`.

`examples/divergence_topography.py` goes further: it treats the up-to-four
radiologist readers as two independent labelings (split-half) and asks whether
their *disagreement* is structured in feature space. It is — but only shallowly
(accuracy 0.587 vs null 0.527, p ≈ 0.015, barely over the 0.577 base rate):
42% of nodules are internally contested, and that contest is mostly idiosyncratic
to the reader, not the nodule. A pathology drop-in (`--pathology`) runs the same
divergence analysis against a lower-variance reference when the data is supplied.
See `examples/divergence_topography_usecase.md`.

`examples/pathology_validation.py` runs the real version against the TCIA
LIDC-IDRI tissue diagnoses. The headline: on 80 histopathology-confirmed
patients, **93% of radiologist-"indeterminate" cases were malignant** — the
hedge tier hides cancer, and acting only on "suspicious" misses 40% of cancers.
The pathology reference is itself selection-biased (7 benigns; tissue is taken
when cancer is suspected), so specificity is unmeasurable from it — every
reference is a centroid with its own selection topography. See
`examples/pathology_validation_usecase.md`.

`examples/score_comparison.py` runs a parallel score comparison against tissue
pathology. Its scope note is the result: the established clinical models (Brock,
Mayo) and the NTOG research scores **cannot** be computed on LIDC, which lacks
their demographic/clinical inputs — they are not fabricated. Of the predictors
LIDC does support, a literature-weighted morphology composite ties the
established size and radiologist-gestalt benchmarks (AUC ~0.72–0.75, differences
not significant) — added morphology buys no discrimination over diameter alone
here. See `examples/score_comparison_usecase.md`.

`examples/score_comparison_synthetic.py` runs the full Brock-vs-Mayo-vs-NTOG
comparison on the synthetic cohort (which has the demographics LIDC lacks),
against an *independent* latent malignancy label so the test isn't circular. The
three scores are statistically indistinguishable (AUC 0.70–0.72, all differences
NS) and strongly rank-concordant (0.66–0.87); NTOG's growth domain demonstrably
re-ranks fast-growing nodules (r = 0.24). The honest caveat is built in: the AUC
ranking is a property of the chosen generator (whose truth has no growth term),
so it demonstrates machinery, not validity. See
`examples/score_comparison_synthetic_usecase.md`.

`examples/test_fusion.py` asks whether *combining* tests beats the best single
one, on both cohorts, with a label-free centroid fusion and a cross-validated
logistic fusion. The honest answer here is no: the centroid average ties the best
single test in both cohorts, the fitted logistic combination overfits and does
worse (significantly so on real tissue), and the cases where tests disagree are
where discrimination collapses rather than where fusion rescues. The tests are
too rank-concordant to carry complementary signal; real fusion gains need a
larger cohort with genuinely orthogonal modalities. See
`examples/test_fusion_usecase.md`.

## Nordic Lung Cancer Quality-Indicator Example

A larger, real-domain example models lung cancer pathway quality indicators
across the five Nordic countries as a measurement-capability network — country
registries and data-source infrastructure (unlabeled scaffold) plus 17 quality
indicators labeled by feasibility tier:

- `examples/nordic_lung_cancer_qi_nodes.csv` / `..._edges.csv` (generated by
  `examples/build_nordic_lung_cancer_qi.py`)
- `examples/nordic_lung_cancer_qi_usecase.md` — full walkthrough and interpretation

It exercises every lens at once: graph features and centrality, a semi-supervised
outcome model (only the indicators are labeled), community-aware evaluation with a
permutation null, and strength-weighted shortest paths read as **harmonization
routes**. The headline finding is methodological: the outcome model looks
significant under random splits (p ≈ 0.02) but is indistinguishable from chance
under community-aware splits (p ≈ 0.31) — a textbook illustration of why network
dependence inflates naive cross-validation.

## Data Model

Nodes are entities: people, places, organizations, studies, grants, hospitals,
events, risks, exposures, pathway steps, or any other objects.

Edges are relationships or flows between nodes: contact, referral, collaboration,
co-authorship, transition, shared exposure, communication, dependency, or movement.

Default node columns:

- `ID`: unique node identifier
- `Outcome`: optional target label for modeling/path targeting

Default edge columns:

- `SourceID`
- `TargetID`
- optional `Weight`

See `Data-format.md` for details.

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
python epinet_toolkit.py \
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
python epinet_toolkit.py --run-clusters --distance-metric mahalanobis \
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

`epinet_contest.py` makes the boundary analytic. For the nearest-centroid
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
python epinet_toolkit.py \
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

## Tests and linting

```bash
python -m unittest discover -s tests   # or: pytest
ruff check .
```

GitHub Actions runs both on every push and pull request across Python
3.10–3.12 (`.github/workflows/tests.yml`).

## License

MIT. See `LICENSE`.
