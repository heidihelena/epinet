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
   proliferation (Ki-67), IHC marker positivity (CD20, CD10, BCL2, CD5, CD23,
   cyclin D1, MYC, …), or summaries of deep patch embeddings. Save one row per
   case + a subtype label.
2. **Build the similarity network** — each case is connected to its *k* nearest
   neighbours in standardized feature space. This makes graph position
   informative and gives EpiNet's centroid clustering and interactive view a
   meaningful layout.
3. **Generate the model** — EpiNet trains a RandomForest subtype classifier and
   evaluates it honestly (repeated stratified splits + a label-permutation null),
   reports feature importances, clusters the subtypes by feature-space centroid,
   and writes an interactive `network.html`.
4. **Map the grey zone** — `--run-contest` measures, per case, how far the
   subtype call is from flipping (flip-distance), which subtype it would flip to,
   and which single marker would most cheaply settle it. The contested cases are
   the diagnostically hard ones; the marker is a value-of-information / "stain
   this next" pointer.

## Run it

```bash
# 1. See it run on a synthetic five-subtype cohort (DLBCL / FL / CLL / MCL / BL)
#    with ambiguous CLL/MCL grey-zone cases injected:
python examples/build_lymphoma_workflow.py --demo --output examples

# 2. Your real data: a CSV with an id column, a label column, and numeric features
python examples/build_lymphoma_workflow.py \
    --features cases.csv --id-col CaseID --label-col Subtype --k 6 --output examples

# 3. Generate the model AND map the contested cases
epinet --nodes examples/lymphoma_nodes.csv --edges examples/lymphoma_edges.csv \
    --outcome-column Subtype --no-run-paths \
    --n-iterations 30 --permutation-test 200 \
    --run-clusters --run-contest --distance-metric mahalanobis \
    --cluster-labeled-only --interactive-network --output-dir examples/lymphoma_outputs
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
- `node_contestability.csv` + `contest_summary.json` — per case: `flip_distance`
  (how far the call is from flipping), `runner_up_class` (the differential),
  `contested` (the most fragile fraction), and `most_decision_relevant_feature`
  with its `single_axis_flip_distance` (the marker that would settle it).
- `plots/network.html` — interactive, draggable subtype-similarity network.
- the standard diagnostic figures (confusion matrix, permutation null, …).

On the synthetic demo the model reaches F1 ≈ 0.98 across the five subtypes with
permutation p ≈ 0.01 — the bulk is separable by construction. The point of the
demo is not that headline number but the **grey zone underneath it**: see below.
On real data, expect lower and read the permutation p-value first — it tells you
whether the features carry real signal before you trust any number.

## Reading the contested cases (the grey zone)

A headline F1 hides the cases that actually matter. `node_contestability.csv`
surfaces them. On the demo cohort the most-contested cases are the injected
CLL/MCL boundary cases: their `flip_distance` is ~4× smaller than the bulk
(≈0.35 vs ≈1.3 standardized units), they are flagged `contested`, their
`runner_up_class` is the other member of the pair, and — for every one of them —
`most_decision_relevant_feature` is **cyclin D1**. That is the right answer:
CLL and MCL overlap on CD5 and CD10, and cyclin D1 (t(11;14)) is the marker that
distinguishes them. The lens has found the diagnostically hard cases *and* named
the stain that would resolve each one, without being told the immunology.

This is the value-of-information reading: a small flip-distance says "the model's
call here is fragile", and the decisive feature says "this is the one input worth
getting right (or repeating) before trusting the call." Two cautions ride along
in `contest_summary.json`'s `caveats` field and are not optional:

- `flip_distance` is in standardized-feature units, so it is decision-relevant
  only when *smaller than the real-world measurement error* of the markers in the
  same units. The module reports the number; comparing it to assay variability is
  your step.
- It measures the **centroid surface's fragility, not ground truth**. On this
  synthetic demo the surface is illustrative; on real, unvalidated features it
  tells you the classifier is unsure here, not that the case is truly borderline.

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
