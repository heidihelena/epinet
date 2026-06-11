# Pathology validation — replacing the contested reference with tissue

This is the run the earlier examples were building toward: the radiologist
malignancy tier we had been treating as the outcome is now compared against a
genuinely lower-variance reference — the TCIA **LIDC-IDRI diagnosis data**
(`tcia-diagnosis-data-2012-04-20`), patient-level, with histopathology
confirmation where possible.

`pathology_validation.py` matches by TCIA patient id (the nodule-level diagnosis
columns cannot be reliably linked to pylidc clusters), takes each patient's
**index nodule** (highest median-reader malignancy) as the radiologist
representation, and compares it to the tissue diagnosis.

## Honest data note

The TCIA diagnosis spreadsheet is supplied by the user (not bundled, not
fabricated). Tidy form: `pid, pt_dx, pt_method` (codes: dx 1=benign,
2=malignant primary, 3=malignant metastatic; method 2=biopsy, 3=resection,
1=2-year radiological stability). It is gitignored as derived TCIA data; the
script regenerates it from the `.xls`.

## Results

### Histopathology-confirmed (biopsy/resection), 80 patients — 73 malignant, 7 benign

| radiologist index-nodule tier | pathology benign | pathology malignant |
|------------------------------|------------------|---------------------|
| benign_low | 1 | 4 |
| **indeterminate** | 2 | **25** |
| suspicious_high | 4 | 44 |

- **The hedge bucket hides cancer.** 25/27 (93%, CI 77–98%) of radiologist-
  *indeterminate* patients were malignant on tissue. The "3" tier is not a
  neutral middle — against pathology it is overwhelmingly cancer.
- Acting only on *suspicious_high*: sensitivity **0.60** (CI 0.49–0.71) — 40% of
  cancers were missed because they were hedged to indeterminate.
- Counting indeterminate as positive: sensitivity **0.95** (CI 0.87–0.98).
- **Specificity is unmeasurable here**: only 7 benigns, because tissue is taken
  mainly when cancer is suspected. The "ground truth" is selection-biased.

### All confirmed (adds method-1 radiological-stability benigns), 118 patients — 87 malignant, 31 benign

- With 31 benigns, specificity becomes estimable: *suspicious_high* specificity
  **0.77** (CI 0.60–0.89) — but this benign set is partly defined radiologically,
  so it is a softer reference than tissue.
- Hedge bucket: 33/49 (67%, CI 53–79%) of indeterminate patients still malignant.

## Reading — the philosophy, in numbers

Three things this makes concrete, and they are the publishable core:

1. **The lower-variance reference relocates meaning.** Treating radiologist
   malignancy as the outcome (2651 nodules, `lidc_cohort_usecase.md`) made the
   indeterminate tier — 53% of nodules — look like a neutral middle. Against
   tissue it is 67–93% malignant. Changing the reference did not just change
   accuracy; it changed what the middle category *means*.

2. **Every reference is a centroid with its own selection topography.**
   Histopathology, the "gold standard," is here so biased toward malignancy
   (7 benigns) that it cannot estimate specificity at all. Recovering
   specificity requires admitting radiological-stability benigns — i.e.,
   contaminating the reference with the very modality being validated. There is
   no view from nowhere; there is only a choice of which bias to carry.

3. **The clinical finding is real and not an artifact of the toolkit.** The
   earlier observation that the model's errors funnel into the indeterminate
   bucket is, against pathology, a statement about radiology itself: the bucket
   where readers and model both hedge is where the cancers are. That is a
   genuine, citable signal — and it survived precisely because the pipeline
   never claimed the radiologist label was truth.

## Boundaries

- n is small (80 histopath; 118 confirmed) and patient-level. These are
  descriptive estimates with Wilson intervals, not a powered model.
- Index-nodule aggregation assigns the patient diagnosis to the single most
  suspicious nodule; multi-nodule patients are simplified.
- Specificity from the histopath subset is not interpretable (7 benigns).
- Method-1 benigns are radiological, not tissue — reported separately, never
  pooled silently.
