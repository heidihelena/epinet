---
name: cran-compliance
description: >-
  Get an R package ready for CRAN: pass R CMD check --as-cran cleanly, satisfy
  the CRAN Repository Policy (DESCRIPTION, license, documented runnable examples,
  no writing outside tempdir, no top-level side effects, conditionally-used
  Suggests), handle packages that wrap Python via reticulate, and run the
  submission checks (win-builder, R-hub, cran-comments, reverse dependencies).
  Use when preparing or submitting an R package to CRAN, fixing R CMD check
  NOTEs/WARNINGs/ERRORs, or reviewing a package for CRAN-readiness — e.g. vahtian.epinet.
---

# CRAN compliance

The bar: **`R CMD check --as-cran` with 0 ERRORs, 0 WARNINGs, and every NOTE
explained** in `cran-comments.md`. CRAN is strict and largely automated; fix the
package, don't argue with the check. Policy: https://cran.r-project.org/web/packages/policies.html

## DESCRIPTION

- **Title**: title case, ≤65 chars, no package name, no "in R", not ending in a period.
- **Description**: one+ full sentences, doesn't start with "This package" or the
  package name; spell out and single-quote software names; cite methods with
  `Authors (year) <doi:...>`.
- **Authors@R**: use the `person()` form; exactly one `cre` (maintainer) with a
  valid email; add `aut`/`ctb` and ORCID where possible.
- **License**: a standard CRAN license; `MIT + file LICENSE` needs a 2-line
  LICENSE file (`YEAR:` / `COPYRIGHT HOLDER:`).
- **SystemRequirements**: declare external runtimes (e.g. `Python (>= 3.9)`).
- **Version** bump on resubmission.

## Documentation & examples

- Every exported object documented; every function has a `\value` section (a
  common WARNING is a missing `\value`).
- Examples should be **runnable**. Use `\donttest{}` for examples that work but
  are slow (>5s) — CRAN still runs them. Use `\dontrun{}` only for genuinely
  un-runnable examples (need credentials, interactivity, external services).
  Don't hide a broken example in `\dontrun`.
- No example may need network access, secrets, or write outside `tempdir()`.

## Behavior rules (frequent rejections)

- **Never write** to the user's home, the package library, or the working
  directory — only `tempdir()`. Don't create files at install/load.
- **No side effects on load/attach**: no `library()`/`require()` of other
  packages, no `options()`/`par()` changes that aren't restored (`on.exit()`),
  no printing/`cat()` except via `message()`/`packageStartupMessage()`.
- Don't use `T`/`F` for `TRUE`/`FALSE`; don't use `<<-` to global env; keep
  examples/tests/vignettes runnable within a few minutes total.
- All imports declared in DESCRIPTION + NAMESPACE; no `:::` into other packages.

## Suggests are conditional

Anything in `Suggests` (incl. test/vignette/example helpers) must be **guarded**:
`if (requireNamespace("pkg", quietly = TRUE))`, and tests
`testthat::skip_if_not_installed("pkg")`. The package must work without them.

## Packages that wrap Python (reticulate)

(Current reticulate ≥ 1.41 guidance.)
- Declare Python deps with `reticulate::py_require(...)` in `.onLoad`, defaulting
  to an **isolated, per-package** environment — not the user's global Python.
- Import modules with `delay_load = TRUE` so the package **loads even when Python
  is absent** (CRAN's check machines may not have it).
- **Skip** tests/examples/vignette chunks when the module is missing:
  `skip_if_not(reticulate::py_module_available("mod"))`.
- `SystemRequirements: Python (>= X)`. Treat `Config/reticulate` auto-install
  with caution (it can trigger installs at load).

## Environment artifacts vs real findings

Many `--as-cran` findings in a sandbox are **environment**, not package faults —
don't chase them locally; confirm on win-builder/R-hub instead:
- `checking CRAN incoming feasibility ... NOTE` (needs CRAN network)
- `checking for future file timestamps ... NOTE` (clock skew / no network)
- `checking PDF version of manual ... ERROR/WARNING` (no LaTeX/texlive installed)
- `cannot set locale ... en_US.UTF-8` warning (container locale)

**Real** findings to fix: missing `\value`, undocumented args, non-ASCII in code,
writing outside tempdir, examples >5s without `\donttest`, unconditional Suggests,
`T`/`F`, undeclared imports, "New submission" NOTE (expected — note it).

## Submission flow

1. Local: `R CMD check --as-cran` clean (NOTEs explained).
2. `devtools::check_win_devel()` + `rhub::rhub_check()` across R-release/devel and
   OSes.
3. If others depend on you, `revdepcheck::revdep_check()`.
4. Write **`cran-comments.md`**: test environments, check results (0/0/N NOTEs +
   why), downstream-dependency status.
5. `devtools::submit_cran()`; respond promptly to the CRAN maintainer; bump
   Version on each resubmission.

See `reference/cran-checklist.md` for a copy-paste pre-submission checklist.
