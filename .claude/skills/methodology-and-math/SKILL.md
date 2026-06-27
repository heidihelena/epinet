---
name: methodology-and-math
description: >-
  Apply rigorous statistical methodology and mathematical verification to
  analyses, models, estimators, and derivations. Use when building or reviewing a
  predictive or epidemiological model, an estimator, or a numerical/mathematical
  result — to validate properly (discrimination + calibration + clinical
  utility), guard against data leakage and multiplicity, justify sample size,
  verify numerical stability, and report to standards (TRIPOD+AI, STROBE). Also
  use when asked "is this rigorous?", or for calibration, p-values / multiple
  comparisons, numerical precision, or deriving/checking a formula.
---

# Methodology & mathematical rigor

"Earn the claim." A result is only as good as the check that survives trying to
break it. Use this when correctness and honest evaluation matter more than a
headline number.

## Evaluating a predictive / risk model

Report and check **all three** dimensions — discrimination alone is not enough
(TRIPOD+AI, BMJ 2024, supersedes TRIPOD 2015):

1. **Discrimination** — AUROC / c-statistic (and AUPRC under class imbalance).
2. **Calibration** — do predicted risks match observed frequencies? Use a
   smoothed calibration plot + calibration slope/intercept. Calibration has a
   hierarchy (Van Calster 2016): mean → weak (slope=1, intercept=0) → moderate →
   strong; aim for at least moderate. A discriminating-but-miscalibrated model is
   misleading exactly where decisions are made.
3. **Clinical utility** — net benefit / decision-curve analysis, not just
   statistical metrics.

Report **confidence intervals** and **key subgroup** performance (fairness).
Prefer **proper scoring rules** (Brier, log loss): their expectation is optimized
by the true probabilities. Accuracy/F1 at a fixed threshold are not proper.

## Avoid data leakage (the most common fatal error)

Evaluation data must be **fully distinct** from anything used to train, tune
hyperparameters, select features, or impute (TRIPOD+AI). Concretely:
- Split **before** any data-dependent preprocessing; fit scalers/imputers/feature
  selectors on train only, apply to test.
- Keep all records for one subject/cluster in the **same** fold (group-aware CV)
  when rows are correlated (repeated measures, networks, sites).
- Report the internal-validation scheme; prefer external validation when claiming
  generalization.

## Sample size, power, multiplicity

- **Justify** sample size for the question; don't lean on rules of thumb like
  10 events-per-variable. For prediction models use Riley et al. (2019):
  small expected shrinkage, small optimism in apparent fit, and precise
  intercept/outcome-risk estimation, computed for the specific setting.
- With many comparisons, **control** family-wise error (Bonferroni/Holm) or FDR
  (Benjamini–Hochberg), and pre-specify/report the analysis to avoid p-hacking.
- Quantify uncertainty (bootstrap CIs); correct apparent performance for
  **optimism** (bootstrap/.632) or use a held-out/external set.

## Honest-evaluation toolkit

- A **permutation/label-shuffle null** measures whether signal beats chance.
- A **no-information baseline** and a simple reference model bound "good".
- Adopt an improvement only when it beats baseline AND **transports** to held-out
  data — and say so plainly when a gain is within noise.

## Mathematical & numerical rigor

- **Catastrophic cancellation**: never subtract two large near-equal numbers.
  Variance as `E[x^2] - mean^2` loses precision when the mean dominates; use the
  centered/two-pass or Welford/Chan form `sum((x-mean)^2)`. (This bit EpiNet's
  federated scaler — centered moments fixed it.)
- **Conditioning**: check the condition number before inverting; regularize /
  shrink (ridge, Ledoit–Wolf/OAS) ill-conditioned matrices rather than trusting
  a raw inverse.
- **Reproducibility**: seed every stochastic step; record versions/provenance;
  make runs deterministic.
- **Verify derivations**: cross-check a closed form against a brute-force or
  symbolic computation (e.g. `sympy`); test invariants/properties (monotonicity,
  bounds, symmetry, units) rather than a single example; assert known limiting
  cases.

## Reporting

Align to the relevant EQUATOR guideline — **TRIPOD+AI** (prediction models),
**STROBE** (observational), **CONSORT-AI** (trials of AI interventions). State
assumptions and caveats explicitly; do not imply clinical/causal validity that
the evidence does not support.

See `reference/rigor-checklist.md` for a pre-flight checklist and the key
citations.
