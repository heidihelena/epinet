# CRAN pre-submission checklist

Load when finalizing a CRAN submission with `cran-compliance`.

## Local check

- [ ] `R CMD build pkg` then `R CMD check --as-cran pkg_*.tar.gz`
- [ ] 0 ERRORs, 0 WARNINGs
- [ ] Every NOTE understood and explained in `cran-comments.md`
- [ ] Confirmed which NOTEs are environment-only (incoming feasibility, future
      timestamps, PDF manual/LaTeX, locale) vs real

## DESCRIPTION

- [ ] Title: title case, no package name / "in R", no trailing period, ≤65 chars
- [ ] Description: full sentences; not starting "This package"/package name;
      method refs as `Author (year) <doi:...>`; software names single-quoted
- [ ] Authors@R with one `cre` + valid email (ORCID if available)
- [ ] Standard License (+ LICENSE file if `... + file LICENSE`)
- [ ] SystemRequirements for external runtimes (e.g. Python)
- [ ] Version bumped

## Docs & examples

- [ ] All exported objects documented; `\value` present on every function
- [ ] Examples runnable; `\donttest{}` for slow (>5s), `\dontrun{}` only for
      truly un-runnable; no network/secrets/interactivity
- [ ] Vignettes build and are not huge; package tarball reasonable size

## Behavior

- [ ] Writes only to `tempdir()` — never home/library/cwd; nothing at load/install
- [ ] No load-time side effects; `options()`/`par()` restored via `on.exit()`
- [ ] No `T`/`F`; no `<<-` to globalenv; no `:::`
- [ ] All imports in DESCRIPTION + NAMESPACE
- [ ] Suggests guarded (`requireNamespace`), tests `skip_if_not_installed`

## reticulate / Python wrappers

- [ ] `reticulate::py_require()` in `.onLoad`; isolated per-package env
- [ ] Imports use `delay_load = TRUE` (package loads without Python)
- [ ] Tests/examples/vignette skip when the module is absent
- [ ] `SystemRequirements: Python (>= X)`

## Cross-platform & downstream

- [ ] `devtools::check_win_devel()` clean
- [ ] `rhub::rhub_check()` across OSes / R-release + R-devel
- [ ] `revdepcheck::revdep_check()` if you have reverse dependencies

## Submission

- [ ] `cran-comments.md`: test environments, results, NOTE explanations, revdeps
- [ ] First submission? Expect a "New submission" NOTE — state it
- [ ] `devtools::submit_cran()`; reply to CRAN maintainer; bump Version on
      resubmission

## Key sources

- CRAN Repository Policy — https://cran.r-project.org/web/packages/policies.html
- Writing R Extensions — https://cran.r-project.org/doc/manuals/r-release/R-exts.html
- reticulate (≥1.41) Python-dependencies vignette — `py_require`, `delay_load`
