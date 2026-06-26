# epinetR

An R interface to [EpiNet](https://github.com/heidihelena/epinet), a transparent
toolkit for **honestly-evaluated** outcome models. `epinetR` is a thin
[`reticulate`](https://rstudio.github.io/reticulate/) wrapper over the tested
Python `epinet` package, so the algorithms are **single-sourced** — the R layer
does no modelling of its own and cannot drift from the Python core.

> Research and education demonstrator — **not clinical decision support.**

## Install

```r
# install.packages("remotes")
remotes::install_github("heidihelena/epinet", subdir = "r/epinetR")
```

You also need the Python `epinet` package on an interpreter reticulate can see:

```bash
pip install epinet         # or: pip install -e .  from the repo root
```

If reticulate doesn't find it, point it at the right Python:

```r
reticulate::use_python("/path/to/python", required = TRUE)
# or set the RETICULATE_PYTHON environment variable
```

## Usage

```r
library(epinetR)

result <- epinet(
  data,
  outcome    = "copd",
  predictors = c("age", "sex", "smoking")
)

summary(result)   # discrimination, calibration, bootstrap CI, data warnings
plot(result)      # permutation feature importance
print(result)     # one-line headline
```

`epinet()` builds a design matrix from the predictors (one-hot encoding
non-numeric ones) and fits the outcome model with EpiNet's honest-evaluation
defaults: imbalance-aware tuning, calibration (Brier + slope/intercept), a
percentile bootstrap interval, an optional label-permutation null
(`n_permutations`), and permutation feature importance.

The returned object is a plain list (class `"epinet"`) with `metrics`,
`importance`, `features_used`, and more — easy to pull into your own reports.
