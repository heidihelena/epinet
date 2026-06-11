# LIDC-IDRI nodule cohort — real, biased data

This is the real-data counterpart to the synthetic nodule cohort. It runs the
EpiNet feature-space pipeline on **LIDC-IDRI** (the Lung Image Database
Consortium), the standard public lung-nodule dataset, using the radiologist
annotations bundled with `pylidc` — no DICOM images required.

LIDC-IDRI was chosen *because* it is biased, to see how the pipeline behaves on
real, imperfect labels rather than a clean synthetic generator.

## The data and its biases

`build_lidc_cohort.py` clusters the 6859 radiologist annotations across 1018
scans into physical nodules and aggregates the per-reader ratings:

- **875 scans, 2651 nodules** with a computable diameter.
- Outcome = malignancy tier from the **median** reader malignancy (1–5):
  benign_low (<2.5), indeterminate (2.5–3.5), suspicious_high (>3.5).
- Realised mix: **875 benign_low / 1399 indeterminate / 377 suspicious_high**.

The biases are explicit, not hidden:

- **Subjective labels.** Malignancy is a radiologist *gestalt*, not pathology.
- **Hedging.** The indeterminate ("3") tier is 53% of nodules — readers cluster
  on the middle score.
- **Inter-reader disagreement.** Mean malignancy spread across readers is 0.93;
  757/2651 (29%) of nodules have a ≥2-point disagreement. Recorded per nodule
  as `MalignancySpread` / `NReaders` in `lidc_provenance.csv`.
- **Selection.** Only annotatable nodules ≥3 mm are in LIDC at all.

The clustering/model use the eight semantic characteristics + diameter; the
malignancy rating (the label source) is kept in the provenance file, so
predicting the tier from morphology is not circular.

## How to reproduce

```bash
pip install pylidc           # optional, for this example only
python examples/build_lidc_cohort.py        # ~3 min; writes lidc_*.csv

python epinet_toolkit.py \
  --nodes examples/lidc_nodes.csv --edges examples/lidc_edges.csv \
  --outcome-column Outcome --include-centrality --no-run-paths \
  --split-strategy community --n-iterations 30 --permutation-test 100 \
  --run-clusters --distance-metric mahalanobis --n-clusters 0 --cluster-labeled-only \
  --output-dir examples/lidc_outputs
```

Scans are scaffold hubs; nodules link to their scan and siblings, so the
community split holds whole scans/patients out.

## Interpretation — how it handles the bias

### 1. Real, significant, and stable under a patient-aware split

| | F1 (mean ± sd) | Null F1 | p-value |
|---|----------------|---------|---------|
| Scan-aware split (875 groups) | **0.696 ± 0.025** | 0.394 | **0.010** |

Morphology predicts the radiologist malignancy tier, it generalizes to unseen
scans, and — because n = 2651 rather than the 17 of the Nordic network — the
iteration spread is tight (±0.025). The top features are clinically sensible:
**diameter (0.24) and calcification (0.20)**, then subtlety, spiculation,
lobulation, margin.

### 2. The hedge bucket absorbs the errors

The held-out confusion matrix:

| actual \ predicted | benign_low | indeterminate | suspicious_high |
|--------------------|-----------|---------------|-----------------|
| **benign_low** | 94 | 76 | **3** |
| **indeterminate** | 34 | 239 | 17 |
| **suspicious_high** | **0** | 31 | 45 |

The corners are 3 and 0: the model **never** confuses clearly benign with
clearly suspicious. Every error funnels through the indeterminate middle column.
The model respects the ordinal structure — the morphologic signal is strong at
the extremes and genuinely ambiguous in the middle, exactly where the readers
themselves hedge.

### 3. Reader disagreement is largely orthogonal to feature-space ambiguity

The tempting hypothesis — "the model gets wrong the nodules the radiologists
disagreed on" — is **not** what the data shows:

- Nodules whose nearest feature-space centroid disagrees with their tier have
  *slightly lower* reader spread (0.87) than the ones it gets right (0.96).
- The indeterminate tier's reader spread (1.03) is only marginally above the
  extremes' (0.82); the "3" pile-up is substantially *consensus* hedging, not
  disagreement.
- Correlation between distance-to-centroid and reader spread is weak (−0.14).

So the two sources of uncertainty are largely **independent**: subjective
reader disagreement is not explained by the recorded morphologic features. A
morphology model and the radiologist gestalt are capturing partly different
things — a concrete, honest observation about LIDC's label bias that the
pipeline surfaces rather than masks.

## Boundaries

- Still not pathology. The label is a subjective rating; "accuracy" here means
  agreement with the median radiologist, not with ground-truth malignancy. A
  pathology-linked subset (a minority of LIDC) would be the next step.
- The semantic characteristics are themselves reader-rated, so feature and label
  share a reader, which can inflate apparent signal — read alongside the
  orthogonality result above, which argues the inflation is limited.
- `pylidc`'s bundled diameter uses the annotation contours; a few annotations
  fail to yield a diameter and are dropped.
- This is a methods demonstration on a biased benchmark, not clinical decision
  support.
```
