# EpiNet — the Epistemic Network toolkit

[![Tests](https://github.com/heidihelena/epinet/actions/workflows/tests.yml/badge.svg)](https://github.com/heidihelena/epinet/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
<!-- After cutting a Zenodo release, add: [![DOI](https://zenodo.org/badge/DOI/<doi>.svg)](https://doi.org/<doi>) -->

EpiNet is a **transparent node/edge network and feature-space analysis toolkit**
for graph-shaped datasets. The name reads *Epistemic Network*: the core question
is not just "what does the model predict" but how well-founded each call is — how
contestable, how calibrated, how well it transports. You load entities and
relationships from CSVs and it computes graph features, *honestly* evaluates an
outcome model, finds shortest paths, clusters nodes by feature-space centroid,
and scores how **contestable** each call is — with publication-quality figures
and a model card. Epidemiology is one use case; the core is domain-neutral
(driven through lung-cancer quality indicators, lung-nodule risk, and lymphoma
subtyping).

> **Scope.** This is a **research and education demonstrator, not clinical or
> public-health decision support.** Any model it produces must be validated on
> independent, outcome-linked data before it means anything clinically. See
> [Scope and caveats](#scope-and-caveats).

What distinguishes EpiNet from a thin scikit-learn wrapper is that **honest
evaluation is the default path**: a label-permutation null, calibration, and
(where appropriate) community-aware splitting run alongside the headline metric,
so a good score reflects real signal rather than leakage or chance. Developed as
part of **Vahtian**; MIT licensed.

## What it looks like

![Contestability panel on a lymphoma cohort](examples/sample-outputs/lymphoma/contestability.png)

*The contestability lens on the lymphoma grey-zone example. Left: how far each
call is from flipping, with the most-contested tail shaded. Right: value of
information — the features that most drive boundary flips (CD10, cyclin D1,
Ki67, …). The same lens runs on any cohort. More figures in
[examples/sample-outputs/](examples/sample-outputs/).*

## What it does

- **Graph features** — degree, weighted degree, clustering, component size,
  isolate flag, optional betweenness/closeness/PageRank.
- **Honest outcome model** — RandomForest over graph features + node attributes,
  with discrimination (AUROC, AUPRC), classification (balanced accuracy, MCC, F1),
  **calibration** (Brier, slope/intercept), bootstrap CIs, permutation importance,
  a label-permutation null, community-aware splitting, small-cohort warnings, a
  reproducibility `provenance` block, and a TRIPOD+AI-flavoured `model_card.md`.
- **Shortest paths** — from sources to target nodes or target-outcome nodes, with
  per-target coverage.
- **Feature-space clustering** — k-means centroids + per-node distance to each
  outcome-class centroid (Euclidean or Ledoit–Wolf-shrunk Mahalanobis).
- **Contestability** (`--run-contest`) — the closed-form smallest feature-space
  move that flips a node's nearest-centroid class, plus a per-feature
  value-of-information ranking. See [docs/methods.md](docs/methods.md).
- **Input normalization** — maps common column aliases onto the schema before
  validation; never silently (every rename logged, raw + normalized hashed).
- **Federated pipeline** — reconstruct the scaler, centroids, and contestability
  from per-site aggregates only, behind a fail-closed governance gate. See
  [docs/federated.md](docs/federated.md) and
  [docs/governance-and-consent.md](docs/governance-and-consent.md).
- **Baselines & external validation** — compare graph features against a
  node-embedding baseline and a no-information floor under the same harness, and
  validate a model on an independent cohort. See [docs/validation.md](docs/validation.md).

## Install

```bash
pip install -e .            # installs the package + the `epinet` command
pip install -e ".[dev]"     # also pytest + ruff + hypothesis (for development)
pip install -e ".[lidc]"    # pylidc, for the LIDC-IDRI / LUNA16 examples
pip install -e ".[excel]"   # xlrd + openpyxl, for the TCIA diagnosis spreadsheets
```

`requirements.txt` lists the core runtime dependencies if you prefer not to
install the package.

## Quick start

```bash
epinet \
  --nodes synthetic_nodes.csv \
  --edges synthetic_edges.csv \
  --outcome-column Outcome \
  --target-outcome 1 \
  --output-dir epinet_outputs
```

(`epinet ...` is the installed console command; `python epinet_toolkit.py ...`
works identically without installing.) This runs graph-feature generation, an
honestly-evaluated outcome model, and shortest-path summaries side by side.

Key outputs in `epinet_outputs/`:

- `model_metrics.json` — discrimination, classification, calibration,
  `iteration_summary`, bootstrap CI, permutation test, data warnings, provenance
- `model_card.md` — TRIPOD+AI-flavoured human-readable model card
- `model_feature_importance.csv` — permutation importance (± `importance_std`)
- `node_features.csv`, `shortest_paths.csv`, `nearest_targets.csv`,
  `target_coverage.csv`, `provenance.json`, `run_summary.json`
- `plots/*.png` — network, calibration, learning curve, metric stability,
  confusion matrix, and more (see [docs/methods.md](docs/methods.md))

The data format is documented in [Data-format.md](Data-format.md).

## Documentation

- **[docs/methods.md](docs/methods.md)** — evaluation design (iterative
  evaluation, permutation null, community-aware splitting), the diagnostic
  figures, the contestability theory, and methodological boundaries.
- **[docs/examples.md](docs/examples.md)** — worked examples: shortest paths,
  the CiteMatch evidence graph, feature-space clustering, the pulmonary-nodule
  cohort, real LIDC-IDRI, and the Nordic lung-cancer quality-indicator network.
- **[docs/federated.md](docs/federated.md)** — the federated fit, federated
  contestability, the registry adapter, and the sealed-egress model.
- **[docs/governance-and-consent.md](docs/governance-and-consent.md)** — what the
  governance gate enforces vs what remains a policy/legal responsibility
  (explicitly non-legal).
- **[docs/validation.md](docs/validation.md)** — representation baselines (incl. a
  node-embedding comparison) and external validation: does the model transport?

Each worked example also has a builder script and a walkthrough under
`examples/*_usecase.md`; the federated and governance pipelines have runnable
demos under `examples/federated_*` and `examples/governance_*`.

## Scope and caveats

The model is intentionally simple. It does **not** infer causality, outbreak
dynamics, clinical risk, or intervention effects. Network features can be useful
descriptors, but they can also encode sampling bias, measurement bias, and
structural confounding. **Use the outputs as exploratory evidence, not as
decisions.**

Before using EpiNet for health, education, welfare, employment, or public-sector
decisions, add: domain-specific data validation; directed/temporal assumptions;
uncertainty and sensitivity checks; external validation on independent
outcome-linked data; privacy and governance review; and human review of any
operational recommendation. The bundled cohorts are synthetic or small and
selection-biased — see the per-example limits in [docs/examples.md](docs/examples.md).
For clinical prediction, align reporting with
[TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378); for AI interventions, the
bar moves toward prospective evaluation (e.g. CONSORT-AI).

## Tests and linting

```bash
python -m unittest discover -s tests   # or: pytest  (adds the hypothesis property tests)
ruff check .
```

GitHub Actions runs both on every push and pull request across Python
3.10–3.12 (`.github/workflows/tests.yml`).

## Citation

If you use EpiNet, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button generates APA/BibTeX from it).

## License

MIT. See `LICENSE`.
