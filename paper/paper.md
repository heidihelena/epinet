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
date: 27 June 2026
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

Its distinguishing feature is that conservative evaluation is the default rather
than an opt-in. Every outcome-model run reports discrimination, calibration, a
label-permutation null, community-aware splitting, bootstrap intervals,
small-cohort warnings, a provenance record, and a TRIPOD+AI-style model card
[@tripodai; @vancalster2019].

EpiNet is intended to improve the reproducibility of evaluation workflows rather
than to introduce a new predictive algorithm. The predictor itself is a standard
random forest [@scikit-learn]; the contribution is the conservative, auditable
workflow wrapped around it, so that the same checks travel with every analysis.
EpiNet is a research and education demonstrator, not clinical decision support.

# Statement of need

Small biomedical and registry cohorts make over-fitting easy and over-claiming
easier. General-purpose libraries make it trivial to report a single
cross-validated accuracy that in fact reflects leakage or chance, and to ship a
risk score that discriminates well but is badly calibrated. EpiNet is built
around those failure modes: leakage-aware (community) splitting, a permutation
null measured against chance, calibration reported alongside discrimination,
honest within- and across-split uncertainty [@nadeau2003], and a closed-form
contestability measure — the smallest feature-space move that flips a
nearest-centroid call [@tibshirani2002], with a value-of-information ranking of
which measurement would settle it. The need it addresses is methodological
reproducibility, not predictive performance: it standardizes *how an evaluation
is conducted and reported* so the same conservative checks are applied uniformly
and emitted as an auditable record, rather than re-implemented ad hoc per study.

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
the opinionated integration of graph-shaped tabular analysis, conservative
leakage-aware evaluation, nearest-centroid contestability, and an exactly
federatable analytic spine into one small, auditable tool for cohorts where the
dominant failure modes are leakage, calibration error, unstable estimates, and
overconfident boundary calls. No single existing package combines these defaults
for this setting; the pieces EpiNet reuses are dependencies, not reimplementations.

# Software design

EpiNet uses a small namespaced module layout (`vahtian/epinet/`) and CSV inputs to
keep the analysis inspectable, and favours conservative defaults over
configurability: every run emits discrimination, calibration, a null-model
comparison, bootstrap intervals, warnings, and provenance with no extra opt-in. A
graphical workbench offers the same analysis without the command line — each
session emits a complete `analysis.yaml` and executes through the identical
engine, so the interface is never the source of truth and any result it produces
can be reproduced without it. Safety gates block ill-posed runs (no or
single-class outcome, an identifier used as a feature) and downgrade
under-powered cohorts to a descriptive report rather than fabricating metrics.

Two design decisions follow from the optional federated mode. The outcome model
(a random forest) sits deliberately *outside* the federated spine, since tree
ensembles do not combine exactly from additive site summaries. The
nearest-centroid contestability layer, by contrast, has additive sufficient
statistics — scaler, class centroids, and shared-covariance precision reconstruct
*exactly* from per-site counts, sums, sums of squares, and second-moment matrices
— so the unshrunk-empirical analytic is exactly federatable while record-level
data stay local. Covariance shrinkage (Ledoit–Wolf [@ledoit2004]) is available
centrally; reproducing it federated would need fourth-moment aggregates and is
documented as a limitation, not a feature.

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
(discrimination, calibration), the permutation-null comparison and bootstrap
intervals, diagnostic figures, a machine-readable claims check, and a provenance
record of inputs, configuration, and seeds. The same analysis is available as a
Python API, through the workbench, and from R via the `vahtian.epinet` interface —
all driving the identical engine, so a result is reproducible however it was
launched.

![Label-permutation null from the example run: the histogram is weighted F1 under
100 random label permutations (the no-signal reference) and the line is the
observed score. The separation (here $p = 0.01$) is what distinguishes real
signal from chance, and it is emitted by
default.\label{fig:permnull}](permutation_null.png){ width=70% }

# Research impact statement

EpiNet's current impact is as a reproducible research and education demonstrator.
The repository ships runnable demonstrations on synthetic and small
biomedical-style cohorts (nodule risk, lymphoma subtyping, a registry adapter,
federated contestability, governance-mediated egress), representation baselines
including a learned node-embedding comparison, and an external-validation harness.
The v0.4.1 release — installable from PyPI as `vahtian-epinet`, with an R
interface wrapping the same tested core through reticulate — freezes these as a
citation snapshot with CI-tested examples (Python 3.10–3.12). Its near-term value
is an executable reference workflow for catching leakage, chance-level
performance, calibration failure, and overconfident boundary calls before claims
are drawn from small cohorts.

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
