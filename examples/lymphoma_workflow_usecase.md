# Suggested workflow: lymphoma digital pathology → an ML model to test

A turnkey path from a lymphoma digital-pathology dataset to a trained,
honestly-evaluated subtype classifier, using EpiNet. Built for a collaborator
with whole-slide data who wants a model to test.

> Research / education only. This produces an exploratory model, not a
> diagnostic tool. Any result must be validated on independent, outcome-linked
> data before it means anything clinically.

## The idea

EpiNet works on tabular node/edge data; digital pathology is images. The bridge
is **feature extraction + a patient-similarity network**:

1. **Extract per-case features** from the slides (your step). Anything numeric:
   morphometrics (cell size, nuclear contour irregularity, mitotic count),
   proliferation (Ki-67), IHC marker positivity (CD20, CD10, BCL2, CD5, …), or
   summaries of deep patch embeddings. Save one row per case + a subtype label.
2. **Build the similarity network** — each case is connected to its *k* nearest
   neighbours in standardized feature space. This makes graph position
   informative and gives EpiNet's centroid clustering and interactive view a
   meaningful layout.
3. **Generate the model** — EpiNet trains a RandomForest subtype classifier and
   evaluates it honestly (repeated stratified splits + a label-permutation null),
   reports feature importances, clusters the subtypes by feature-space centroid,
   and writes an interactive `network.html`.

## Run it

```bash
# 1. See it run on a synthetic lymphoma-shaped cohort (DLBCL / FL / CLL):
python examples/build_lymphoma_workflow.py --demo --output examples

# 2. Your real data: a CSV with an id column, a label column, and numeric features
python examples/build_lymphoma_workflow.py \
    --features cases.csv --id-col CaseID --label-col Subtype --k 6 --output examples

# 3. Generate and evaluate the model
epinet --nodes examples/lymphoma_nodes.csv --edges examples/lymphoma_edges.csv \
    --outcome-column Subtype --no-run-paths \
    --n-iterations 30 --permutation-test 200 \
    --run-clusters --distance-metric mahalanobis --cluster-labeled-only \
    --interactive-network --output-dir examples/lymphoma_outputs
```

### Input CSV format

| CaseID | Subtype | Ki67 | CD10 | CD5 | CellSize | … |
|--------|---------|------|------|-----|----------|---|
| C001   | DLBCL   | 72   | 0.6  | 0.1 | 8.4      | … |
| C002   | FL      | 18   | 0.9  | 0.0 | 5.6      | … |

Any numeric columns are used as features; `CaseID` and `Subtype` are excluded
from the model. If several samples share a patient, add a patient id column and
tell me — that changes the split strategy (see below).

## What you get

- `model_metrics.json` — accuracy/F1 with mean ± sd across iterations, the
  permutation p-value, and the confusion matrix.
- `model_feature_importance.csv` — which features drive the subtype call.
- `node_clusters.csv` + `plots/feature_clusters.png` — subtype centroids and the
  cases whose feature-space neighbourhood disagrees with their label (review
  candidates).
- `plots/network.html` — interactive, draggable subtype-similarity network.
- the standard diagnostic figures (confusion matrix, permutation null, …).

On the synthetic demo the model reaches F1 ≈ 0.99 (the three subtypes are
separable by construction) with permutation p ≈ 0.01, and the top features are
Ki-67, mitotic count, and CD10 — the expected discriminators. On real data,
expect lower and read the permutation p-value first: it tells you whether the
features carry real signal before you trust any number.

## One important evaluation note

Use the **default stratified random split**, not `--split-strategy community`.
In a feature-similarity graph the communities *are* the subtypes, so a community
split would hold out whole subtypes (train on two, test on a third unseen one).
Community splitting is the right tool only when communities are *nuisance*
grouping — e.g. multiple biopsies per patient, where you must keep a patient's
samples together. If that is your situation, include a patient id and group on it.

## Boundaries

- Operates on extracted features, not raw pixels — the model is only as good as
  the features you give it.
- Subtype labels are themselves expert calls; treat them as a reference with
  inter-observer variability, not ground truth (see the LIDC examples for how
  EpiNet surfaces label disagreement).
- Synthetic demo data is illustrative; the feature means are plausible, not real.
