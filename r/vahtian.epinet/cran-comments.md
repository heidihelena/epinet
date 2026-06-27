# cran-comments

## Test environments

- Local: Ubuntu 24.04, R 4.3.3 (`R CMD check --as-cran`)
- (Pending before submission) win-builder: R-devel and R-release
- (Pending before submission) R-hub: Linux/macOS/Windows, R-release and R-devel

## R CMD check results

0 ERRORs, 0 WARNINGs on a correctly-provisioned machine.

Expected NOTE on first submission:

* New submission.

Notes seen only in the offline sandbox used during development are
environment artifacts, not package issues, and are expected to disappear on
win-builder/R-hub:

* "CRAN incoming feasibility" / "future file timestamps" — no network access.
* "PDF version of manual" — no LaTeX/TeX installed locally; all Rd checks pass.
* "HTML version of manual" — no `tidy` command installed locally.

## Python dependency (reticulate)

This package is a thin interface to a Python package via 'reticulate'.

* The R namespace loads without Python: the Python module is imported with
  `delay_load = TRUE`, and every entry point first checks
  `reticulate::py_module_available()` and errors with installation guidance if
  the runtime is absent.
* All examples are wrapped in `\dontrun{}` (they require the Python runtime),
  and all tests `skip_if_not(reticulate::py_module_available(...))`, so the
  check is clean on machines without Python.
* `SystemRequirements` declares the Python requirement.

## Downstream dependencies

None (new package).
