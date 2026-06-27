# Rigor pre-flight checklist + citations

Run through this before claiming a result. Load when applying `methodology-and-math`.

## Predictive model

- [ ] Discrimination reported (AUROC; AUPRC if imbalanced) with CI
- [ ] Calibration plot + slope/intercept (aim ≥ moderate calibration)
- [ ] Clinical utility / net benefit (decision curve) where decisions are implied
- [ ] Proper scoring rule used (Brier / log loss), not just thresholded accuracy
- [ ] Subgroup / fairness performance reported
- [ ] Test data distinct from train/tune/select; preprocessing fit on train only
- [ ] Group-aware splitting for correlated rows (subject/cluster/site/network)
- [ ] Internal validation scheme stated; external validation if generalizing
- [ ] Optimism-corrected (bootstrap) or held-out apparent performance
- [ ] Sample size justified for the question (not just EPP=10)
- [ ] Multiplicity controlled (FWER/FDR) and analysis pre-specified
- [ ] Null/baseline comparison (permutation null, no-information floor)
- [ ] Improvement beats baseline AND transports to held-out data

## Numerical / mathematical

- [ ] No subtraction of large near-equal quantities (use centered/stable forms)
- [ ] Condition number checked before matrix inversion; regularize if ill-conditioned
- [ ] All randomness seeded; runs reproducible; versions/provenance recorded
- [ ] Derivation cross-checked (brute force or symbolic, e.g. sympy)
- [ ] Property/invariant tests (bounds, monotonicity, symmetry, units, limits)

## Reporting

- [ ] Mapped to the right EQUATOR guideline (TRIPOD+AI / STROBE / CONSORT-AI)
- [ ] Assumptions, caveats, and scope of validity stated
- [ ] No clinical/causal claim beyond the evidence

## Key citations (verified)

- **TRIPOD+AI** — Collins et al., BMJ 2024;385:e078378 (16 Apr 2024). Current
  standard for prediction-model reporting; supersedes TRIPOD 2015. 27 items /
  52 subitems + 13-item abstract checklist; regression or ML; adds open-science,
  fairness, and patient/public-involvement items. https://www.equator-network.org/reporting-guidelines/tripod-statement/
- **Calibration hierarchy** — Van Calster et al., 2016 (mean/weak/moderate/strong).
- **Sample size for prediction models** — Riley et al., 2019 (three criteria:
  shrinkage, apparent-fit optimism, precise intercept/risk).
- **Proper measures / net benefit** — decision-curve analysis (Vickers);
  Lancet Digital Health 2025 on proper performance measures.
- **STROBE** (observational), **CONSORT-AI** (AI trials) — EQUATOR network.

Note: reporting guidelines mandate *justifying and reporting* choices, not a
single fixed method — pick measures appropriate to the question and defend them.
