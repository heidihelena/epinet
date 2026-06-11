# EpiNet — worked examples

Companion to the [README](../README.md). Every example is a research/education demonstrator, not clinical evidence; cohorts are synthetic or small and selection-biased (each example states its own limits).

A frozen, representative run is committed under [`../examples/sample-outputs/`](../examples/sample-outputs/) — model card, metrics JSON, and figures — so you can see the outputs without running anything.

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

