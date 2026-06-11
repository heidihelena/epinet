# Multi-test fusion — what each test sees, and whether combining helps

Treats each risk test as a separate view of a case and asks: where do the tests
diverge, and does combining them beat the best single test against truth?
`test_fusion.py` runs it on both cohorts with two fusion methods.

- **centroid fusion** — standardise each test (higher = more malignant) and
  average. The equal-weight axis through the class centroids in test-space;
  label-free, so no fitting and no leakage.
- **logistic fusion** — a cross-validated logistic combination
  (out-of-fold predictions only, so the AUC is honest).

## Result

### Real LIDC tissue truth (n=118, 87 malignant; imaging predictors only)

| predictor | AUC | 95% CI |
|-----------|-----|--------|
| gestalt | 0.729 | 0.62–0.83 |
| **size** | **0.754** | 0.65–0.85 (best single) |
| lit_morph | 0.717 | 0.61–0.82 |
| centroid fusion | 0.753 | 0.65–0.85 |
| logistic fusion | 0.719 | 0.61–0.83 |

- centroid − best single: ΔAUC −0.002 (p = 0.90) — a tie.
- logistic − best single: ΔAUC **−0.035 (p = 0.046)** — significantly *worse*.

### Synthetic latent truth (n=117, 39 malignant; Brock/Mayo/NTOG)

| predictor | AUC | 95% CI |
|-----------|-----|--------|
| Brock | 0.703 | 0.60–0.80 |
| **Mayo** | **0.716** | 0.62–0.81 (best single) |
| NTOG | 0.711 | 0.61–0.81 |
| centroid fusion | 0.723 | 0.62–0.82 |
| logistic fusion | 0.685 | 0.58–0.79 |

- centroid − best single: ΔAUC +0.007 (p = 0.71) — a tie.
- logistic − best single: ΔAUC −0.031 (p = 0.16) — worse, not significant.

## Reading

1. **Combining tests does not beat the best single test on these data.** The
   transparent centroid average *ties* the best single test in both cohorts; the
   fitted logistic combination is consistently *worse* (significantly so on real
   tissue). At n≈120 with few cases in one class, fitting fusion weights overfits
   — the simple standardized average is the robust choice, the fitted model is not.
   This is a methodological caution as much as a result: prefer the label-free
   centroid fusion at this scale.

2. **The disagreement cases are the hard core, not a fusion opportunity.** Tests
   disagree on 26% (real) and 41% (synthetic) of cases. On the real cohort,
   discrimination on those cases collapses to chance for every method
   (best-single 0.45, centroid 0.49, logistic 0.32) — these are genuinely
   ambiguous nodules where no combination of the available tests recovers signal.
   On synthetic, centroid fusion does lift the disagreement subset (0.61 vs 0.52),
   hinting that equal-weight averaging can help where tests conflict — but the
   real cohort shows no such rescue.

3. **Why fusion can't shine here, concretely.** The tests are strongly
   rank-concordant (0.66–0.87 elsewhere), so they carry largely the same
   information — there is little complementary signal to fuse. Fusion gains
   require tests that look at *different* things and are *right at different
   times*; on this data they mostly look at the same thing (size and its
   correlates), so averaging neither helps nor hurts and fitting overfits.

## What would make fusion pay off

Fusion needs (a) tests with genuinely complementary information and (b) enough
cases — especially benign cases — to estimate a stable combination. Both point to
a larger, less malignancy-skewed cohort with orthogonal modalities (e.g. NLST
with smoking/PLCO person-risk + nodule morphology + growth + a PET/Herder axis),
where the person-level and nodule-level tests genuinely diverge. On the cohorts
in hand, the honest finding is: **the best single test is hard to beat, the
simple centroid average matches it without overfitting, and the disagreement set
is where the real uncertainty lives.**

## Boundaries

- Small n with skewed positives; CIs are wide and the disagreement subsets
  smaller still (descriptive only).
- Real-cohort tests are imaging-only (LIDC has no demographics); synthetic
  truth is generator-dependent.
- Logistic fusion uses repeated stratified CV out-of-fold predictions; centroid
  fusion is unsupervised. Neither is tuned beyond defaults.
