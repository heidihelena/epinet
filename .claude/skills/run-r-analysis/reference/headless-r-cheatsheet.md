# Headless R cheatsheet

Detailed reference for `run-r-analysis`. Load when you hit a specific need
(devices, structured output, reporting, errors).

## Graphics devices (no display)

| Need | Call |
| --- | --- |
| Base/lattice plot to PNG | `png("f.png", width=1000, height=600, res=110); ...; dev.off()` |
| Crisp, modern raster (preferred if available) | `ragg::agg_png("f.png", width=1000, height=600, res=110); ...; dev.off()` |
| Vector output | `pdf("f.pdf", width=8, height=5); ...; dev.off()` or `svg("f.svg")` |
| ggplot2 | `ggplot2::ggsave("f.png", p, width=8, height=5, dpi=110)` |

- `ragg` (apt: `r-cran-ragg`) is the modern r-lib raster device — faster than
  Cairo and deterministic. Base `png()` is always present; use it if `ragg`
  isn't installed.
- After writing, **Read the image file** to verify the figure visually.
- Never rely on the default device under `Rscript`: it is `pdf()` →
  `Rplots.pdf`, which causes confusing "cannot open file 'Rplots.pdf'" errors.

## Capturing results (not the console)

```r
write.csv(df, "out.csv", row.names = FALSE)
saveRDS(model, "model.rds")
jsonlite::write_json(as.list(coef(model)), "coef.json", auto_unbox = TRUE)
cat(capture.output(summary(model)), sep = "\n", file = "summary.txt")
```

## Reproducible dependencies

- **renv**: `renv::init()` to snapshot, `renv::snapshot()` to update
  `renv.lock`, `renv::restore()` to rebuild an environment from it.
- Point repos at **Posit P3M** before restoring so older versions install as
  binaries (CRAN keeps binaries only for the latest version):
  `options(repos = c(P3M = "https://packagemanager.posit.co/cran/latest"))`.
- **pak** (`pak::pak("pkg")`) resolves system requirements and installs in
  parallel; it is what `r-lib/actions/setup-r-dependencies` uses in CI.
- When CRAN/P3M are unreachable (proxy), prefer apt `r-cran-*` binaries.

## reticulate (R ↔ Python)

```r
Sys.setenv(RETICULATE_PYTHON = "/usr/local/bin/python3")  # or set in the shell
reticulate::py_config()                 # confirm interpreter + numpy/pandas
mod <- reticulate::import("yourpkg")     # R data.frame <-> pandas auto-convert
```

- Set `RETICULATE_PYTHON` to an interpreter where the Python package is
  importable (`pip install -e .` from the repo, or `pip install <pkg>`).
- reticulate converts R `data.frame` ⇄ pandas and named lists ⇄ dicts
  automatically; pass R integers with `as.integer()` for Python `int` kwargs.

## Reporting (R Markdown / Quarto), headless

```bash
Rscript -e 'rmarkdown::render("report.Rmd", output_file="report.html")'
quarto render report.qmd --to html          # if quarto CLI is installed
```

Chunks that plot still need a file device; set
`knitr::opts_chunk$set(dev = "ragg_png")` (or `"png"`) for consistent output.

## Validating an R package

```bash
export RETICULATE_PYTHON=$(which python3)
R CMD build pkgdir                          # build the source tarball
R CMD check --no-manual pkg_*.tar.gz        # runs tests/ + examples
Rscript -e 'rcmdcheck::rcmdcheck(args="--no-manual")'   # if rcmdcheck installed
```

- `--no-manual` skips the LaTeX PDF manual (no TeX needed).
- testthat tests that need optional deps should `skip_if_not(...)` so the suite
  degrades gracefully when (e.g.) the Python side is absent.
- CI: `r-lib/actions/check-r-package` (it runs `rcmdcheck`, failing at the
  WARNING level by default).

## Common errors → fixes

| Symptom | Fix |
| --- | --- |
| `cannot open file 'Rplots.pdf'` / no plot | Open an explicit file device before plotting (§ Graphics devices) |
| `cannot open URL '.../PACKAGES'` | CRAN blocked — use apt `r-cran-*` or P3M |
| `there is no package called 'X'` | Install it (apt `r-cran-x`, or `pak::pak("X")`) |
| reticulate picks the wrong Python | Set `RETICULATE_PYTHON`; check with `py_config()` |
| `cannot set locale ... en_US.UTF-8` in `R CMD check` | Container locale artifact; ignore (won't occur in CI) |
| `package 'X' is not available for this version of R` | Repo unreachable or version-mismatched — use apt or P3M snapshot |
