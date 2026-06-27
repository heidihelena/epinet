---
name: run-r-analysis
description: >-
  Run R and R-package analyses inside Claude Code instead of RStudio — install R
  headlessly, run scripts non-interactively with Rscript, render plots to image
  files you can view, manage packages when CRAN is blocked, drive reticulate
  R<->Python interop, and validate R packages with R CMD check. Use when the
  user wants to run R code, an R analysis, or an R package in this environment,
  or mentions R, Rscript, reticulate, ggplot, an .R/.Rmd/.qmd file, the EpiNet
  vahtian.epinet R package, or "RStudio".
---

# Running R analyses in Claude Code

Researchers can run a full R analysis here — no RStudio, no local setup. The one
thing that bites everyone is **plotting**: a non-interactive `Rscript` has no
screen, so plot code that works in RStudio fails or silently writes nothing.
Always render to an image file and then view it. The rest is install + run.

## 1. Make sure R is available

```bash
Rscript --version 2>/dev/null || sudo apt-get update -qq && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq r-base-core
```

`r-base-core` is enough for `Rscript`, building, and `R CMD check`.

## 2. Install R packages (CRAN is often blocked here)

Try CRAN first, but in sandboxed/proxied environments `install.packages()` from
CRAN frequently fails (`cannot open URL .../PACKAGES`). Fallbacks, in order:

1. **apt `r-cran-*` binaries** (fast, no compile, no CRAN needed):
   ```bash
   sudo apt-get install -y -qq r-cran-reticulate r-cran-testthat r-cran-igraph
   ```
   Most common packages exist as `r-cran-<lowercasename>`.
2. **Posit P3M** binary repo (for `install.packages`/`renv`, when reachable) —
   it serves precompiled binaries incl. older versions, unlike CRAN:
   ```r
   options(repos = c(P3M = "https://packagemanager.posit.co/cran/latest"))
   install.packages("pak"); pak::pak("somepkg")
   ```
3. **renv** for reproducibility when a project has `renv.lock`:
   `Rscript -e 'renv::restore()'` (point repos at P3M first for binary installs).

Verify: `Rscript -e 'cat(requireNamespace("reticulate", quietly=TRUE))'`.

## 3. Run analyses non-interactively

Short snippets with `-e`, real work from a script file:

```bash
Rscript -e 'summary(lm(mpg ~ wt, mtcars))'
Rscript analysis.R              # writes outputs/plots to files
```

Save data outputs explicitly (`write.csv`, `saveRDS`, `jsonlite::write_json`) —
nothing is "in the console" to scroll back to.

## 4. Plots: render to a FILE, then view it (the #1 gotcha)

Under `Rscript` the default graphics device is PDF (`Rplots.pdf`), not a screen,
so `plot(...)`/`ggsave()` can error or go nowhere. **Open an explicit file
device, draw, close it, then Read the image:**

```r
png("plot.png", width = 1000, height = 600, res = 110)  # or ragg::agg_png()
plot(model)                                              # any base/lattice plot
dev.off()
# ggplot2: ggsave("plot.png", p, width = 8, height = 5, dpi = 110)
```

Then use the Read tool on `plot.png` to actually see the figure and judge it —
don't assume it looks right because the script exited 0.

## 5. reticulate (R calling Python)

Point reticulate at the interpreter that has your Python packages, and make the
package importable:

```bash
pip install -e .                          # or: pip install <pkg>
export RETICULATE_PYTHON=$(which python3) # reticulate then uses this interpreter
Rscript -e 'reticulate::py_config()'      # confirm the right Python
```

## 6. Validate an R package

```bash
export RETICULATE_PYTHON=$(which python3)
R CMD build path/to/pkg                    # -> pkg_x.y.z.tar.gz
R CMD check --no-manual pkg_*.tar.gz       # runs the testthat suite + examples
```

A container-locale warning (`cannot set locale ... en_US.UTF-8`) is an
environment artifact, not a package fault — CI won't show it. In CI, use
`r-lib/actions` (`check-r-package` runs `rcmdcheck`, failing at WARNING level).

## 7. Worked example — the EpiNet `vahtian.epinet` package

`vahtian.epinet` (in `r/vahtian.epinet/`) wraps the Python `vahtian.epinet` core via reticulate.

```bash
sudo apt-get install -y -qq r-base-core r-cran-reticulate r-cran-testthat
cd <repo> && pip install -e . && export RETICULATE_PYTHON=$(which python3)
R CMD INSTALL r/vahtian.epinet
Rscript -e '
  suppressMessages(library(vahtian.epinet))
  set.seed(1); n <- 150
  df <- data.frame(age=rnorm(n,60,10),
                   sex=sample(c("M","F"),n,TRUE),
                   smoking=sample(c("never","former","current"),n,TRUE))
  df$copd <- rbinom(n,1,plogis(0.05*(df$age-60)+0.9*(df$smoking=="current")-0.5))
  fit <- epinet(df, outcome="copd", predictors=c("age","sex","smoking"))
  summary(fit)
  cst <- epinet_contestability(df, outcome="copd",
                               predictors=c("age","sex","smoking"))
  png("contestability.png", width=1000, height=450); plot(cst); dev.off()
'
# then Read contestability.png to view the lens
```

## More detail

See `reference/headless-r-cheatsheet.md` for a fuller command cheatsheet
(device choices, capturing structured output, Quarto/R Markdown rendering,
common errors and fixes).
