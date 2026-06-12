---
title: "EpiNet — the Epistemic Network toolkit: honest evaluation and federated contestability for graph-shaped tabular data"
tags:
  - Python
  - network analysis
  - clinical prediction models
  - calibration
  - federated analysis
  - reproducibility
authors:
  - name: "Heidi Andersén"
    orcid: 0000-0001-5923-5865
    affiliation: 1
affiliations:
  - name: Vahtian
    index: 1
date: 12 June 2026
bibliography: paper.bib
---

# Summary

EpiNet analyses datasets where one table describes entities and a second table
describes relationships between them. It turns node and edge CSV files into graph
descriptors, evaluates outcome models, finds paths through the graph, groups
cases in feature space, and reports when a nearest-centroid classification is
fragile. The name reads *Epistemic Network*: the organising question is not only
what a model predicts but how well-founded each call is — how contestable, how
calibrated, and how well it transports to new data.

Its distinguishing feature is that conservative evaluation is the default rather
than an opt-in. Every outcome-model run reports discrimination (AUROC, average
precision), calibration (Brier score, calibration slope and intercept), a
label-permutation null, community-aware splitting, bootstrap intervals,
small-cohort warnings, a reproducibility provenance record, and a TRIPOD+AI-style
model card [@tripodai; @vancalster2019]. EpiNet is a research and education
demonstrator, not clinical decision support.

# Statement of need

Small biomedical and registry cohorts make over-fitting easy and over-claiming
easier. General-purpose libraries make it trivial to report a single
cross-validated accuracy that in fact reflects leakage or chance, and to ship a
risk score that discriminates well but is badly calibrated. EpiNet is built
around those failure modes: leakage-aware (community) splitting, a permutation
null measured against chance, calibration reported alongside discrimination,
honest within- and across-split uncertainty [@nadeau2003], and a closed-form measure of how
contestable a boundary call is — the smallest move in standardized feature space
that flips a nearest-centroid (Rocchio) classification [@tibshirani2002], with a
per-feature value-of-information ranking that names the cheapest measurement that
would settle the call. Researchers, students, and registry-methods groups need an
executable reference workflow that surfaces these problems before claims are
drawn; EpiNet packages that workflow with runnable demonstrations and CI-tested
examples.

# State of the field

Existing tools solve parts of this workflow but not the assembly. Graph
construction and graph measures are provided by general libraries such as
NetworkX [@networkx], igraph [@igraph], and graph-tool [@graphtool], and learned
graph representations by PyTorch Geometric [@pyg]; model fitting, calibration, and
validation by scikit-learn [@scikit-learn] and statsmodels [@statsmodels]; and
distributed analysis by federated frameworks
such as DataSHIELD [@datashield] and federated-learning libraries [@rieke2020].
EpiNet is *not* better at graph algorithms than NetworkX nor at modelling than
scikit-learn. Its contribution is the opinionated integration of graph-shaped
tabular analysis, conservative leakage-aware evaluation, nearest-centroid
contestability, and an exactly federatable analytic spine into one small,
auditable tool aimed at cohorts where the dominant failure modes are leakage,
calibration error, unstable estimates, and overconfident boundary calls. The
build-versus-contribute judgement is that no single existing package combines
these defaults for this setting; the pieces it reuses (NetworkX, scikit-learn)
are cited and depended upon rather than reimplemented.

# Software design

EpiNet uses a flat, single-file module layout and CSV inputs to keep the analysis
inspectable, and favours conservative defaults over configurability: evaluation
outputs include discrimination, calibration, a null-model comparison, bootstrap
intervals, warnings, and provenance with no extra user opt-in.

Two design decisions follow from the federated goal. First, the outcome model (a
random forest) sits deliberately *outside* the federated spine, because tree
ensembles do not combine exactly from additive site-level summaries. Second, the
nearest-centroid contestability layer was chosen because its sufficient
statistics are additive: the empirical scaler, the class centroids, and the
shared-covariance Mahalanobis precision can be reconstructed *exactly* from
per-site counts, sums, sums of squares, and second-moment matrices, so the
contestability analytic is exactly federatable while record-level data stay
local. Covariance shrinkage (Ledoit–Wolf [@ledoit2004]) is available in
centralized use; exact federated reproduction of that shrinkage would require
additional (fourth-moment) aggregates and is documented as a current limitation,
not a feature.

Disclosure of any aggregate is mediated by a fail-closed governance gate. The
gate requires a consent metadata object and refuses disclosure when required
fields or expiry dates are missing, applies small-cell suppression and a tier
ceiling, emits a disclosure manifest, and appends to a tamper-evident (not
signed) audit ledger. The aggregates it permits are de-identified, not anonymous,
and the project makes no claim of GDPR, MDR, EU AI Act, or national-framework
compliance: lawful basis, data-protection assessment, controllership, and consent
validity remain legal and policy responsibilities, documented separately.

# Research impact statement

EpiNet's current impact is as a reproducible research and education demonstrator.
The repository includes runnable demonstrations for synthetic and small
biomedical-style cohorts — nodule risk, lymphoma subtyping, a registry adapter,
federated contestability, and governance-mediated egress — together with
representation baselines (including a learned node-embedding comparison) and an
external-validation harness. The v0.3.0 release freezes these materials as a
citation snapshot with CI-tested examples (Python 3.10–3.12) and documented
methodological limits. Its near-term significance is to give reviewers, students,
and registry-methods researchers an executable reference workflow for identifying
leakage, chance-level performance, calibration failure, and contestable boundary
calls before claims are drawn from small cohorts — and a worked, auditable
pattern for keeping a contestability analytic federated rather than pooling
records.

# AI usage disclosure

Generative AI assistance (Anthropic Claude) was used for code drafting and
refactoring, documentation, and manuscript preparation. All AI-assisted changes
were reviewed by the author, exercised through the automated test suite, checked
against documented statistical identities where applicable (for example, the
closed-form flip-distance and the additive reconstruction of the scaler and
centroids), and revised before release.

# Acknowledgements

EpiNet is developed as part of Vahtian, alongside the companion project
*citevahti* (reproducible citation and provenance tooling).

# References
