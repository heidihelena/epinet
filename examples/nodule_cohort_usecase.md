# Pulmonary nodule cohort — feature-space risk clustering

This example is the bridge between the **NTOG lung-nodule risk tools** (ntog.org)
and EpiNet's **feature-space centroid clustering**. The NTOG static tools turn a
nodule's feature vector into a risk *probability* (Brock/PanCan, Mayo/Swensen)
and a growth *category* (volume doubling time) but stop there — they do not place
nodules in a shared feature space or measure distance between them. This example
adds exactly that step.

> **Synthetic data.** Every patient and nodule here is sampled from plausible
> distributions — no real patient data is used. The risk-model *coefficients*
> are the published ones, reimplemented in Python in
> [`build_nodule_cohort.py`](build_nodule_cohort.py). The three-band risk tier
> is illustrative, not a clinical standard.

## Validation of the risk-model port

The Python port is validated by [`validate_nodule_models.py`](validate_nodule_models.py)
(also run in the test suite) three independent ways:

1. **Source equivalence** — the port is compared to an independent verbatim
   transcription of the NTOG tool's formula across 5000 random inputs; they
   agree to **0.0 absolute error** (the port *is* the tool).
2. **Literature anchoring** — every coefficient's `exp()` reproduces the
   published Brock/PanCan (McWilliams 2013) and Mayo/Swensen (1997) odds ratios
   exactly: female 1.82, spiculation 2.17, upper lobe 1.93, prior cancer 3.81,
   etc. (the tool's coefficients *are* the published models).
3. **Worked cases** — the 4 mm size-term zero point, a hand-traced 8 mm Brock
   probability, nodule-type ordering, diameter monotonicity, and two
   volume-doubling-time hand examples all pass.

```bash
python examples/validate_nodule_models.py
```

## The cohort

[`build_nodule_cohort.py`](build_nodule_cohort.py) generates 70 patients and
117 nodules. For each nodule it computes:

- **Brock / PanCan** malignancy probability (McWilliams et al., NEJM 2013)
- **Mayo / Swensen** pretest probability (Swensen et al., 1997)
- **Volume doubling time** from a synthetic follow-up scan

and assigns an illustrative risk tier from the Brock probability
(low < 0.10, intermediate 0.10–0.50, high ≥ 0.50). The realised mix is
imbalanced — 71 low / 37 intermediate / 9 high — which is clinically faithful:
most incidental nodules are low risk.

The cohort is written as an EpiNet network:

- **Nodes:** patients (scaffold, unlabeled) and nodules (labeled by risk tier).
- **Edges:** `has_nodule` (patient→nodule) and `same_patient` (nodule↔nodule).
  A nodule's degree therefore encodes multifocality, and the community split
  holds whole patients out of evaluation.
- **Provenance:** `nodule_risk_scores.csv` holds the Brock/Mayo/VDT values. These
  are deliberately **kept out of the modelled features** — the clustering uses
  only raw morphology and patient factors, so it cannot trivially recover a tier
  that was defined from Brock.

## How to reproduce

```bash
python examples/build_nodule_cohort.py        # regenerate the CSVs

python epinet_toolkit.py \
  --nodes examples/nodule_nodes.csv --edges examples/nodule_edges.csv \
  --outcome-column Outcome --include-centrality --no-run-paths \
  --split-strategy community --n-iterations 50 --permutation-test 200 \
  --run-clusters --distance-metric mahalanobis --n-clusters 0 --cluster-labeled-only \
  --output-dir examples/nodule_outputs
```

`--cluster-labeled-only` excludes the feature-less patient scaffold nodes from
the clustering (they would otherwise form a degenerate all-zero cluster).

## Interpretation

### 1. Risk tier is genuinely predictable — and it generalizes across patients

Predicting the Brock risk tier from raw morphology + patient factors, evaluated
over 50 **patient-aware** community splits (whole patients held out) with a
200-permutation null:

| | F1 (mean ± sd) | Null F1 | p-value |
|---|----------------|---------|---------|
| Patient-aware split | **0.82 ± 0.10** | 0.47 | **0.005** |

Unlike the small Nordic QI network (where the apparent signal collapsed under a
community split), here the signal is real and survives the strict evaluation:
nodule morphology carries the risk information, and it transfers to unseen
patients. This is the positive control to the Nordic null result — the same
machinery, an honestly significant outcome.

### 2. Centroid distances give a continuous risk phenotype

k-means finds three feature-space clusters; the silhouette is low (≈ 0.18)
because nodule risk is a **continuum**, not three crisp blobs (the PCA plot shows
a smooth gradient along PC1 from low-risk triangles to high-risk circles). The
nearest-tier-centroid classifier still agrees with the Brock tier on 87% of
nodules, and the class centroids order exactly as risk does — intermediate sits
between low and high (low↔high separation 3.21, the largest; intermediate↔low
1.28, the smallest).

### 3. The publishable output: boundary nodules

The genuinely useful, non-circular result is the **15 of 117 nodules whose Brock
tier disagrees with their nearest feature-space centroid**. These are the cases a
single probability threshold handles poorly:

- A nodule with Brock 0.49 (just under the intermediate/high cut) but a 22.6 mm
  solid morphology whose vector sits nearest the **high** centroid — a candidate
  for escalation the threshold would miss.
- A nodule with Brock 0.69 (high) whose phenotype sits nearest the
  **intermediate** centroid — less extreme than its score suggests.
- Several low-Brock nodules near the intermediate centroid, all clustered just
  below the 0.10 boundary.

This is precisely the second opinion the static NTOG calculators cannot give: a
continuous, multi-feature distance that flags nodules sitting near a tier
boundary or whose morphologic phenotype disagrees with the threshold. The
`dist_to_low / dist_to_intermediate / dist_to_high` columns in
`node_clusters.csv` make that distance explicit per nodule.

## Boundaries

- Synthetic cohort. The coefficient port is validated against the NTOG source
  formula and the published odds ratios (see Validation above); the remaining
  gap to real use is real nodule data, not the port.
- The risk tier is a deterministic function of a subset of the features, so high
  recovery is expected — the contribution is the *continuous ordering* and the
  *boundary-case flagging*, not the headline accuracy.
- Mahalanobis is used because the morphology and graph-centrality features are
  correlated; it is the appropriate metric but assumes a single shared
  covariance across tiers.
- Nothing here is clinical decision support. It is a methods demonstration of
  bringing feature-space distance to nodule phenotyping.
```
