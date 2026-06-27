# vahtian.epinet — a thin reticulate wrapper over the tested Python `vahtian.epinet` package.
# The R layer does no modelling of its own: it marshals a data frame to the
# `vahtian.epinet.r_api.fit` adapter and wraps the returned summary in an S3 object.

.epinet_api <- function() {
  if (!reticulate::py_module_available("vahtian.epinet")) {
    stop(
      "The Python package 'epinet' is not available to reticulate.\n",
      "Install it (e.g. `pip install epinet`) and/or point reticulate at the ",
      "right interpreter with reticulate::use_python() or the RETICULATE_PYTHON ",
      "environment variable.",
      call. = FALSE
    )
  }
  # delay_load = TRUE so the package namespace loads even when Python (or the
  # 'epinet' module) is absent — required for CRAN check machines. The guard
  # above gives a clear error if a function is actually called without it.
  reticulate::import("vahtian.epinet.r_api", delay_load = TRUE)
}

#' Fit EpiNet's honestly-evaluated outcome model on a data frame
#'
#' Builds a design matrix from the named predictors (non-numeric predictors are
#' one-hot encoded) and fits EpiNet's outcome model with its honest-evaluation
#' defaults: imbalance-aware tuning, calibration, a percentile bootstrap
#' interval, an optional label-permutation null, and permutation feature
#' importance. The computation runs in the tested Python core via reticulate.
#'
#' @param data A data frame: one row per subject.
#' @param outcome Name of the outcome (label) column.
#' @param predictors Character vector of predictor columns. Defaults to every
#'   column except `outcome`.
#' @param n_iterations Number of repeated train/test splits (default 1).
#' @param n_permutations Label-permutation null draws; 0 disables (default 0).
#' @param n_bootstrap Bootstrap resamples for the primary-split interval; 0
#'   disables (default 1000).
#' @param test_size Held-out fraction per split (default 0.2).
#' @param random_state Integer seed (default 42).
#' @param tune_threshold Tune the decision threshold on out-of-bag training
#'   scores instead of 0.5 (binary outcomes; default FALSE).
#' @return An object of class `"epinet"`: a list with `outcome`, `predictors`,
#'   `features_used`, `n`, `metrics`, and `importance`. Use [summary.epinet()],
#'   [plot.epinet()], and [print.epinet()].
#' @examples
#' \dontrun{
#' fit <- epinet(data, outcome = "copd", predictors = c("age", "sex", "smoking"))
#' summary(fit)
#' plot(fit)
#' }
#' @export
epinet <- function(data, outcome, predictors = NULL,
                   n_iterations = 1L, n_permutations = 0L, n_bootstrap = 1000L,
                   test_size = 0.2, random_state = 42L, tune_threshold = FALSE) {
  if (!is.data.frame(data)) stop("`data` must be a data frame", call. = FALSE)
  if (!is.character(outcome) || length(outcome) != 1L) {
    stop("`outcome` must be a single column name", call. = FALSE)
  }
  if (is.null(predictors)) predictors <- setdiff(names(data), outcome)
  api <- .epinet_api()
  res <- api$fit(
    data = data,
    outcome = outcome,
    predictors = as.list(as.character(predictors)),
    n_iterations = as.integer(n_iterations),
    n_permutations = as.integer(n_permutations),
    n_bootstrap = as.integer(n_bootstrap),
    test_size = as.numeric(test_size),
    random_state = as.integer(random_state),
    tune_threshold = isTRUE(tune_threshold)
  )
  structure(res, class = "epinet")
}

#' @rdname epinet
#' @param x,object An `"epinet"` object.
#' @param ... Unused.
#' @export
print.epinet <- function(x, ...) {
  m <- x$metrics
  cat(sprintf("EpiNet outcome model  (n = %d, outcome = '%s')\n", x$n, x$outcome))
  cat(sprintf("  predictors: %s\n", paste(x$predictors, collapse = ", ")))
  bits <- c()
  if (!is.null(m[["roc_auc"]])) bits <- c(bits, sprintf("AUROC %.3f", m[["roc_auc"]]))
  if (!is.null(m[["balanced_accuracy"]])) {
    bits <- c(bits, sprintf("balanced acc %.3f", m[["balanced_accuracy"]]))
  }
  if (length(bits)) cat(sprintf("  %s\n", paste(bits, collapse = "   ")))
  cat("  Research/education demonstrator - not clinical decision support.\n")
  cat("  summary() for calibration/CIs/warnings; plot() for importance.\n")
  invisible(x)
}

#' @rdname epinet
#' @export
summary.epinet <- function(object, ...) {
  m <- object$metrics
  fmt <- function(v) if (is.null(v)) "NA" else sprintf("%.3f", v)
  cat(sprintf("EpiNet outcome model - summary\n"))
  cat(sprintf("  rows: %d   classes: %s\n", object$n,
              paste(unlist(m[["classes"]]), collapse = ", ")))
  cat("  Discrimination:\n")
  cat(sprintf("    AUROC: %s   average precision: %s\n",
              fmt(m[["roc_auc"]]), fmt(m[["average_precision"]])))
  cat("  Classification:\n")
  cat(sprintf("    balanced accuracy: %s   MCC: %s   F1 (weighted): %s\n",
              fmt(m[["balanced_accuracy"]]), fmt(m[["mcc"]]), fmt(m[["f1_weighted"]])))
  cal <- m[["calibration"]]
  if (!is.null(cal)) {
    cat("  Calibration:\n")
    cat(sprintf("    Brier: %s   slope: %s   intercept: %s\n",
                fmt(cal[["brier"]]), fmt(cal[["slope"]]), fmt(cal[["intercept"]])))
  }
  ci <- m[["primary_split_bootstrap_ci"]]
  if (!is.null(ci) && !is.null(ci[["metrics"]][["balanced_accuracy"]])) {
    b <- ci[["metrics"]][["balanced_accuracy"]]
    cat(sprintf("  Bootstrap 95%% CI (balanced accuracy): [%.3f, %.3f]\n",
                b[["lower"]], b[["upper"]]))
  }
  warns <- m[["data_warnings"]]
  if (!is.null(warns) && length(warns)) {
    cat("  Data warnings:\n")
    for (w in unlist(warns)) cat(sprintf("    - %s\n", w))
  }
  invisible(object)
}

#' @rdname epinet
#' @param top Number of top features to show in `plot()` (default 10).
#' @export
plot.epinet <- function(x, top = 10L, ...) {
  imp <- x$importance
  if (is.null(imp) || !length(imp)) {
    stop("no feature importance available to plot", call. = FALSE)
  }
  feats <- vapply(imp, function(r) as.character(r[["feature"]]), character(1))
  vals <- vapply(imp, function(r) as.numeric(r[["importance"]]), numeric(1))
  ord <- order(vals, decreasing = TRUE)
  keep <- utils::head(ord, as.integer(top))
  feats <- feats[keep]; vals <- vals[keep]
  op <- graphics::par(mar = c(4, 9, 3, 1)); on.exit(graphics::par(op))
  graphics::barplot(
    rev(vals), names.arg = rev(feats), horiz = TRUE, las = 1,
    xlab = "permutation importance",
    main = sprintf("EpiNet feature importance (%s)", x$outcome)
  )
  invisible(x)
}
