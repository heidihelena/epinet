# Divergence topography — where two labelings disagree

This example operationalises a simple epistemic stance: we never measure truth
directly, only the distance between representations. So instead of asking "is the
model right?", it asks **"where do two labelings of the same nodules diverge, and
is that divergence structured in feature space or idiosyncratic?"**

## Honest data note

The original intent was radiologist-tier **vs pathology**. The LIDC-IDRI pathology
diagnosis file (TCIA `tcia-diagnosis-data-2012-04-20`) is **not reachable from
this environment and is not bundled with `pylidc`**, and it has not been
fabricated. Two paths are therefore provided:

- **reader-split (run here):** LIDC gives up to four independent radiologist
  readings per nodule. Splitting the panel into two halves yields two independent
  labelings of the same objects — 100% real, in-sandbox-verifiable data, and a
  faithful methodological stand-in for "a second labeling."
- **pathology (drop-in, ready):** supply `lidc_diagnosis.csv` (`pid,diagnosis`
  with TCIA codes 1=benign, 2/3=malignant) and re-run with
  `--pathology lidc_diagnosis.csv`. The identical pipeline then measures
  radiologist-vs-pathology divergence. This module loads it but never invents it.

## Method

```bash
python examples/build_lidc_cohort.py          # writes lidc_*.csv (needs pylidc)
python examples/divergence_topography.py       # writes lidc_divergence_nodes.csv
python -m epinet.toolkit \
  --nodes examples/lidc_divergence_nodes.csv --edges examples/lidc_edges.csv \
  --outcome-column Outcome --include-centrality --no-run-paths \
  --split-strategy community --n-iterations 30 --permutation-test 200 \
  --run-clusters --distance-metric mahalanobis --cluster-labeled-only \
  --output-dir examples/lidc_divergence_outputs
```

Each nodule with ≥2 readers is labeled `concordant` / `discordant` — do the two
half-panels assign the same malignancy tier? Nodules with one reader stay as
unlabeled scaffold. The pipeline then tests whether **discordance is predictable
from morphology**: if it beats a permutation null under a patient-aware split,
the disagreement occupies locatable regions of feature space (a topography of
contestation); if not, it is unstructured.

## Result

Of 1880 nodules with ≥2 readers, **42.3% are discordant** between the two
half-panels — disagreement is the norm, not the exception, in LIDC.

Predicting discordance from morphology, patient-aware split, 200-permutation null:

| | accuracy | majority base rate | null | p |
|---|----------|--------------------|------|---|
| reader-split discordance | 0.587 ± 0.025 | 0.577 | 0.527 | **0.015** |

The finding is deliberately reported with its smallness intact. Discordance is
**structured but only shallowly**: statistically above the permutation null
(p = 0.015), yet barely above the majority-class base rate (0.587 vs 0.577). The
top predictors are diameter, sphericity, and **subtlety** — subtle, hard-to-see
nodules are read inconsistently, which is clinically sensible.

## Reading

The contested zone is *partially* locatable in feature space and *mostly* not.
Some disagreement lives in the morphology — subtle, borderline-size nodules are
reliably contentious — but the larger share is idiosyncratic to the reader, not
to the object. In the language of the topography view: the distance between two
representations of a nodule is only weakly a function of where the nodule sits in
feature space; the rest is the observer. That is the honest shape of "truth" for
a subjective, vague, processual label — a centroid with real but irreducible
spread, not a point waiting to be uncovered.

This is also a methodological caution for the malignancy model in
[`lidc_cohort_usecase.md`](lidc_cohort_usecase.md): when 42% of labels are
internally contested and that contest is mostly unstructured, "accuracy against
the median reader" has a soft ceiling. The pathology drop-in is the way to
replace the contested reference with a lower-variance one and re-measure the
divergence — honest research, pending the data.
