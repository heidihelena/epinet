---
title: "EpiNet — the Epistemic Network toolkit: a reproducible, leakage-aware evaluation workflow for graph-shaped tabular data"
tags:
  - Python
  - reproducible research
  - model evaluation
  - network analysis
  - calibration
  - federated analysis
authors:
  - name: "Heidi Andersén"
    orcid: 0000-0001-5923-5865
    affiliation: 1
affiliations:
  - name: Vahtian
    index: 1
date: 10 July 2026
bibliography: paper.bib
---

# Summary

EpiNet analyses datasets where one table describes entities and a second
describes relationships between them. It turns node and edge CSV files into graph
descriptors, evaluates outcome models, clusters cases in feature space, and
reports when a nearest-centroid classification is fragile. The name reads
*Epistemic Network*: the organising question is not only what a model predicts
but how well-founded each call is — how contestable, how calibrated, and how well
it transports to new data.

Its distinguishing feature is that conservative evaluation is the organising
workflow rather than a post-hoc checklist. Outcome-model runs report
discrimination, calibration, repeated-split uncertainty, bootstrap intervals,
small-cohort warnings, provenance, and a TRIPOD+AI-style model card
[@tripodai; @vancalster2019]. Label-permutation nulls and community-aware splits
are first-class evaluation modes rather than separate scripts, and the Workbench
enables the permutation null in its reproducible default plan.

EpiNet is intended to improve the reproducibility of evaluation workflows rather
than to introduce a new predictive algorithm. The default predictor is a
standard random forest [@scikit-learn], with scaled regularized logistic
regression, optional XGBoost [@xgboost], and a deliberately small optional
PyTorch [@paszke2019] multilayer perceptron available as alternatives; the contribution is the
conservative, auditable workflow wrapped around the estimator, so that the same
checks travel with every analysis. The claims check also records graph semantics
and edge timing, distinguishing feature-derived similarity graphs from observed
relations before network-shaped claims are made. EpiNet is a research and
education demonstrator, not clinical decision support.

# Statement of need

Small biomedical and registry cohorts make over-fitting easy and over-claiming
easier. General-purpose libraries make it trivial to report a single
cross-validated accuracy that in fact reflects leakage or chance, and to ship a
risk score that discriminates well but is badly calibrated. EpiNet is built
around those failure modes: leakage-aware (community) splitting, optional but
integrated label-permutation nulls measured against chance, calibration reported
alongside discrimination, honest within- and across-split uncertainty
[@nadeau2003], and a closed-form contestability measure — the smallest
feature-space move that flips a nearest-centroid call [@tibshirani2002], with a
value-of-information ranking of which measurement would settle it. The need it
addresses is methodological reproducibility, not predictive performance: it
standardizes *how an evaluation is conducted and reported* so the same
conservative checks are available through one auditable workflow, rather than
re-implemented ad hoc per study. It ships runnable demonstrations on synthetic
and small biomedical-style cohorts, representation baselines, and an
external-validation harness, and is distributed on PyPI (`vahtian-epinet`) with
an R interface (`vahtian.epinet`) that wraps the same tested core through
reticulate, so results cannot diverge across languages.

# State of the field

Existing tools solve parts of this workflow but not the assembly. Graph
construction and graph measures are provided by general libraries such as
NetworkX [@networkx], igraph [@igraph], and graph-tool [@graphtool], and learned
graph representations by PyTorch Geometric [@pyg]; model fitting, calibration, and
validation by scikit-learn [@scikit-learn] and statsmodels [@statsmodels]; and
distributed analysis by federated frameworks
such as DataSHIELD [@datashield] and federated-learning libraries [@rieke2020].
EpiNet is *not* better at graph algorithms than NetworkX nor at modelling than
scikit-learn, and it introduces no new learning algorithm. Its contribution is
the opinionated integration of graph-shaped tabular analysis, leakage-aware
evaluation, nearest-centroid contestability, and an exactly federatable analytic
spine into one small, auditable tool for cohorts where the dominant failure
modes are leakage, calibration error, unstable estimates, and overconfident
boundary calls. No single existing package combines these checks for this
setting while also writing the human-readable model card, claims check,
provenance, figures, and machine-readable outputs from the same run; the pieces
EpiNet reuses are dependencies, not reimplementations.

# Software design

EpiNet uses a small namespaced module layout (`vahtian/epinet/`) and CSV inputs
to keep the analysis inspectable, and favours conservative run records over
large configuration surfaces: the CLI writes discrimination, calibration,
repeated-split summaries, bootstrap intervals, warnings, and provenance, while
the example and Workbench workflows add the label-permutation null that tests
the headline score against chance. A graphical workbench offers the same
analysis without the command line — each session emits a complete
`analysis.yaml` and executes through the identical engine, so the interface is
never the source of truth and any result it produces can be reproduced without
it. Safety gates block ill-posed runs (no or single-class outcome, an identifier
used as a feature) and downgrade under-powered cohorts to a descriptive report
rather than fabricating metrics.

EpiNet treats graph construction as a measurement step, not a cosmetic input
format. Workbench plans record whether edges are feature-derived similarities,
observed relations, or unspecified, and whether observed edges were knowable
before the outcome. The generated claims check therefore prevents a similarity
graph from being reported as evidence of contact, referral, exposure, or
message-passing effects. This gate makes explicit the boundary between ordinary
tabular modelling on graph-derived descriptors and a defensible network
epidemiology claim.

Two design decisions follow from the optional federated mode. The supervised
estimator sits deliberately *outside* the federated spine, since fitted models
(tree ensembles, neural networks, regularized regressions) do not combine exactly
from additive site summaries. The nearest-centroid contestability layer, by
contrast, has additive sufficient statistics — the scaler, class centroids, and
shared covariance reconstruct from per-site counts, sums, and (co-)moment
matrices — so the unshrunk empirical analytic is exactly federatable while
record-level data stay local; optional Mahalanobis shrinkage [@ledoit2004;
@chen2010] uses the pooled covariance rather than records.

The contestability layer is offered as a software diagnostic for triage and
review — the flip-distance and value-of-information ranking are exact properties
of the stated nearest-centroid model, not empirical claims about a cohort.
Whether contestability scores are clinically or epidemiologically useful is a
methodological question for separate validation work and is out of scope for this
software paper.

When aggregates are disclosed across sites, a fail-closed governance gate
mediates release — requiring consent metadata, applying small-cell suppression
and a tier ceiling, emitting a disclosure manifest, and appending to a
tamper-evident audit ledger. The permitted aggregates are de-identified, not
anonymous; the project makes no claim of GDPR, MDR, EU AI Act, or
national-framework compliance, which remains a legal responsibility documented
separately.

# Example

A complete analysis runs from two CSV files — one of nodes (entities, with the
outcome column) and one of edges (relationships):

```bash
epinet \
  --nodes nodes.csv --edges edges.csv \
  --outcome-column Outcome --target-outcome 1 \
  --split-strategy community --permutation-test 100 \
  --output-dir results/
```

The run writes a self-contained bundle to `results/`: a model card
(discrimination, calibration), the permutation-null comparison
(\autoref{fig:permnull}) and bootstrap intervals, diagnostic figures, a
machine-readable claims check, and a provenance record of inputs, configuration,
and seeds. The same analysis is available as a Python API, through the workbench,
and from R via the `vahtian.epinet` interface — all driving the identical engine,
so a result is reproducible however it was launched.

![Label-permutation null from the example run: the histogram is weighted F1 under
100 random label permutations (the no-signal reference) and the line is the
observed score. The separation (here $p = 0.01$) is what distinguishes real
signal from chance in the example workflow.\label{fig:permnull}](permutation_null.png){ width=70% }

# Acknowledgements

EpiNet is developed as part of Vahtian, alongside the companion project
*citevahti* (reproducible citation and provenance tooling). Generative AI
assistance was used for code drafting, documentation, and manuscript preparation;
all AI-assisted changes were reviewed by the author, exercised through the
automated test suite, and checked against documented statistical identities (for
example, the closed-form flip-distance and the additive reconstruction of the
scaler and centroids) before release.

# References
