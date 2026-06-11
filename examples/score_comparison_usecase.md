# Score comparison — literature morphology vs established benchmarks

A parallel run of a literature-weighted score against established ones on the
LIDC-IDRI pathology cohort. Read the scope note first; it is the result.

## Why the full scores can't run here (the honest blocker)

The established clinical models (**Brock**, **Mayo**) and both **NTOG research
scores** were defined for clinical cohorts. Their dominant inputs — age, sex,
smoking/tobacco architecture, family history, emphysema/ILD, lobe location,
longitudinal growth, radon/occupational exposure, immunosuppression — are **not
present in LIDC-IDRI**, which is an imaging-annotation dataset (8 semantic
ratings + diameter + radiologist malignancy gestalt per nodule).

Computing Brock/Mayo/NTOG here would require fabricating their main predictors,
which would invalidate the comparison. The NTOG post-CT score's own rule —
*"missing validated domains are not imputed; score is normalized by available
weight"* — applied to LIDC normalizes away to essentially nothing. So those
scores are **not run** on this cohort. The full head-to-head needs a cohort with
demographics; `build_nodule_cohort.py` provides a synthetic one for that.

## What is computable on LIDC, run against tissue pathology

`score_comparison.py` compares the predictors LIDC genuinely supports, scored on
each patient's index nodule (highest median-reader malignancy) against the TCIA
diagnosis, by ROC-AUC with 2000-sample bootstrap CIs:

- **gestalt** — median radiologist malignancy rating (the established LIDC
  reference standard).
- **size** — nodule diameter (the established imaging benchmark; Fleischner/BTS).
- **lit_morph** — a transparent literature-weighted morphology composite from
  LIDC features only: `+diameter +spiculation +lobulation −margin
  +calcification-absent` (each standardised). This is the imaging-feature
  analogue of the NTOG feature philosophy — **not** the NTOG score itself.

## Result

### all confirmed (n=118: 87 malignant, 31 benign) — discrimination estimable

| score | AUC | 95% CI |
|-------|-----|--------|
| gestalt (established) | 0.729 | 0.62–0.83 |
| size (established) | 0.754 | 0.65–0.85 |
| lit_morph (literature) | 0.717 | 0.61–0.82 |

- `size − gestalt`: ΔAUC **+0.027**, CI [−0.058, +0.103], p = 0.48
- `lit_morph − gestalt`: ΔAUC **−0.011**, CI [−0.088, +0.060], p = 0.77

All three are statistically indistinguishable. A single number — diameter — is
as discriminating as the radiologist's holistic gestalt and as the multi-feature
literature composite. The morphology features (spiculation, lobulation, margin,
calcification) add no detectable discrimination over size alone here, and the
study is underpowered to find a small difference if one exists.

### histopathology only (n=80: 73 malignant, 7 benign) — not interpretable

All AUCs ≈ 0.5 with CIs spanning ~0.28–0.79. With only 7 benigns, discrimination
cannot be estimated — the same tissue-selection bias documented in
`pathology_validation_usecase.md`. The honest reading is "unmeasurable," not
"chance."

## Reading

Two honest takeaways:

1. **On real tissue truth, the literature morphology composite does not beat the
   established benchmarks** — it ties size and gestalt (AUC ~0.72–0.75). That is
   a useful negative result: added morphology did not buy discrimination over
   diameter on this cohort.
2. **The comparison you actually want — NTOG vs Brock/Mayo — is not answerable on
   LIDC at all**, because the cohort lacks the clinical inputs those scores are
   made of. Answering it requires a demographically complete cohort; the
   synthetic generator is the methods-demonstration substitute, and a real
   screening cohort (e.g. NLST, which has smoking/age and better benign
   representation) is the path to a publishable head-to-head.

## Boundaries

- AUCs rest on 31 (or 7) benigns; CIs are wide and the study is underpowered.
- `lit_morph` weights are equal-standardised by literature *direction*, not
  fitted; it is a transparent prior, not an optimised model (fitting on n=118
  would overfit and inflate its AUC).
- Patient-level, index-nodule aggregation, as in the pathology validation.
