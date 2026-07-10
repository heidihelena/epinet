# vahtian.epinet

An R interface to [EpiNet](https://github.com/heidihelena/epinet), a transparent
toolkit for **honestly-evaluated** outcome models. `vahtian.epinet` is a thin
[`reticulate`](https://rstudio.github.io/reticulate/) wrapper over the tested
Python `vahtian-epinet` package, so the algorithms are **single-sourced** — the R layer
does no modelling of its own and cannot drift from the Python core.

> Research and education demonstrator — **not clinical decision support.**

> **Not to be confused with the CRAN package [`epinet`](https://cran.r-project.org/package=epinet)**
> (Groendyke & Welch — Bayesian inference of contact networks and transmission
> trees from epidemic data). That package *infers* who-infected-whom networks
> from outbreak data; `vahtian.epinet` is the R interface to the separate **EpiNet**
> analysis toolkit (honest evaluation, calibration, contestability, federated
> aggregates). The `vahtian.` namespace marks the distinction.

## Install

```r
# install.packages("remotes")
remotes::install_github("heidihelena/epinet", subdir = "r/vahtian.epinet")
```

You also need the Python `vahtian-epinet` package on an interpreter reticulate can see:

```bash
pip install vahtian-epinet  # or: pip install -e .  from the repo root
```

If reticulate doesn't find it, point it at the right Python:

```r
reticulate::use_python("/path/to/python", required = TRUE)
# or set the RETICULATE_PYTHON environment variable
```

## Usage

```r
library(vahtian.epinet)

result <- epinet(
  data,
  outcome    = "copd",
  predictors = c("age", "sex", "smoking"),
  model      = "random_forest" # or "logistic_regression"; "xgboost" if installed
)

summary(result)   # discrimination, calibration, bootstrap CI, data warnings
plot(result)      # permutation feature importance
print(result)     # one-line headline
```

`epinet()` builds a design matrix from the predictors (one-hot encoding
non-numeric ones) and fits the selected outcome model (RandomForest by default,
scaled regularized logistic regression, or optional XGBoost) with EpiNet's
honest-evaluation defaults: imbalance-aware tuning, calibration (Brier +
slope/intercept), a percentile bootstrap interval, an optional label-permutation null
(`n_permutations`), and permutation feature importance.

The returned object is a plain list (class `"epinet"`) with `metrics`,
`importance`, `features_used`, and more — easy to pull into your own reports.

## More surfaces

```r
# Contestability lens: per-row flip-distance + value-of-information
cst <- epinet_contestability(data, outcome = "copd",
                             predictors = c("age", "sex", "smoking"))
plot(cst)   # flip-distance distribution (contested tail shaded) + VOI bars

# Graph pipeline: build the network, derive graph features, fit the model
g <- epinet_graph(nodes, edges, outcome = "Outcome")
plot(g)     # network coloured by outcome, sized by degree (needs 'igraph')

# Federated fit: reconstruct from per-site aggregates only (rows never leave)
fed <- epinet_federated(data, outcome = "copd",
                        predictors = c("age", "sex", "smoking"), n_sites = 3)
plot(fed)   # per-site sizes + reconstruction error vs centralized (~0 = exact)
```

Each surface has `print()`, `summary()`, and a native-R `plot()`.
