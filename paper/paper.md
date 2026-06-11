---
title: "EpiNet: transparent network and feature-space analysis with honest evaluation and federated contestability"
tags:
  - Python
  - network analysis
  - clinical prediction models
  - calibration
  - federated analysis
  - reproducibility
authors:
  - name: "Heidi Andersén"
    affiliation: 1
    # orcid: 0000-0000-0000-0000   # add before submission
affiliations:
  - name: Vahtian
    index: 1
date: 11 June 2026
bibliography: paper.bib
---

# Summary

`EpiNet` is a Python toolkit for analysing graph-shaped tabular datasets. From
node and edge CSVs it computes graph features, evaluates an outcome model, finds
shortest paths, clusters nodes in feature space, and scores how *contestable*
each nearest-centroid call is. It is deliberately small and interpretable, and
its distinguishing feature is that **honest evaluation is the default path**:
discrimination and calibration metrics, a label-permutation null, bootstrap
confidence intervals, community-aware splitting, small-cohort warnings, a
reproducibility provenance block, and a TRIPOD+AI-style model card
[@tripodai] are produced automatically, not as opt-in extras. The toolkit is a
research and education demonstrator, not clinical decision support.

# Statement of need

Small biomedical and registry cohorts are easy to over-fit and easier to
over-claim on. Standard libraries make it trivial to report a single
cross-validated accuracy that reflects leakage or chance rather than signal, and
to ship a discriminating-but-miscalibrated risk score. `EpiNet` is built around
the failure modes themselves. It sits above `NetworkX` [@networkx] for graph
construction and uses `scikit-learn` [@scikit-learn] for modelling, but adds the
guardrails that turn a prototype into something a reviewer can interrogate:

- **Leakage-aware evaluation.** Community-aware splitting keeps whole graph
  communities on one side of the train/test split, and a label-permutation null
  with empirical, direction-aware p-values measures the headline score against
  chance.
- **Calibration, not just discrimination.** Alongside AUROC and average
  precision, every run reports a Brier score and a calibration slope/intercept,
  because a discriminating-but-miscalibrated score misleads exactly where
  decisions are contestable [@vancalster2019].
- **Honest uncertainty.** Repeated-split spread is labelled as variability
  (which understates true uncertainty [@nadeau2003]) and complemented by a
  within-split bootstrap interval.
- **Contestability.** For the nearest-centroid (Rocchio) view [@tibshirani2002],
  `EpiNet` computes the closed-form smallest feature-space move that flips a
  node's class — its decision fragility — and a per-feature value-of-information
  ranking naming the cheapest measurement that would settle the call.

# Federated contestability

`EpiNet`'s standardization and contestability spine is built from additive
sufficient statistics, so it can be reconstructed from per-site aggregates
without pooling records — extending the "take the analysis to the data"
principle of DataSHIELD [@datashield] from generalized linear models to a
decision-contestability analytic. The global scaler, class centroids, and
Mahalanobis precision (via second-moment matrices, with shrinkage
[@ledoit2004] available centrally) combine exactly from per-site counts and
sums; each site then computes flip-distance locally and returns only a
de-identified summary. A fail-closed governance gate makes egress mandatory and
disclosed: cross-boundary contributions are sealed objects whose only output is
a `disclose()` call enforcing small-cell suppression, a tier ceiling, consent,
and a tamper-evident audit ledger. The tree-based outcome model is, by
construction, not part of the federated spine. This positions `EpiNet` alongside
the broader move to federated analysis of medical data [@rieke2020] while keeping
the analytic that federates exactly — contestability — at the centre.

# Functionality and reproducibility

`EpiNet` ships as a small set of single-file modules with a console entry point
(`epinet`), publication-style figures, a continuous-integration test suite
across Python 3.10–3.12, and runnable demonstrations of every claim, including
the federated and governance pipelines. Each run stamps a provenance record
(git commit, package versions, seed, input hashes) into its outputs so results
are traceable and reproducible.

# Acknowledgements

EpiNet is developed as part of Vahtian.

# References
