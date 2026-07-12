# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Heidi Andersén

# Contestability surface: how far each row is from flipping class, and which
# features drive the flip. Computation runs in the Python core; the plot is
# native R (base graphics).

#' Score contestability against the outcome-class centroids
#'
#' For every row, computes the closed-form flip-distance (how far it must move
#' in standardized feature space to flip its nearest-centroid class), the
#' runner-up class, and a per-feature value-of-information ranking. The lowest
#' `contest_quantile` of flip-distances are flagged as the most contestable.
#'
#' @param data A data frame: one row per subject.
#' @param outcome Name of the outcome (label) column.
#' @param predictors Character vector of predictor columns (default: all but
#'   `outcome`). Non-numeric predictors are one-hot encoded.
#' @param metric Distance metric: `"euclidean"` (default) or `"mahalanobis"`.
#' @param contest_quantile Fraction flagged as most contestable (default 0.1).
#' @return An object of class `"epinet_contestability"`.
#' @examples
#' \dontrun{
#' cst <- epinet_contestability(data, outcome = "copd",
#'                              predictors = c("age", "sex", "smoking"))
#' summary(cst)
#' plot(cst)
#' }
#' @export
epinet_contestability <- function(data, outcome, predictors = NULL,
                                  metric = "euclidean", contest_quantile = 0.1) {
  if (!is.data.frame(data)) stop("`data` must be a data frame", call. = FALSE)
  if (!is.character(outcome) || length(outcome) != 1L) {
    stop("`outcome` must be a single column name", call. = FALSE)
  }
  if (is.null(predictors)) predictors <- setdiff(names(data), outcome)
  api <- .epinet_api()
  res <- api$contestability(
    data = data,
    outcome = outcome,
    predictors = as.list(as.character(predictors)),
    metric = metric,
    contest_quantile = as.numeric(contest_quantile)
  )
  structure(res, class = "epinet_contestability")
}

#' @rdname epinet_contestability
#' @param x,object An `"epinet_contestability"` object.
#' @param ... Unused.
#' @export
print.epinet_contestability <- function(x, ...) {
  cat(sprintf("EpiNet contestability  (n = %d, outcome = '%s', metric = %s)\n",
              x$n, x$outcome, x$metric))
  cat(sprintf("  most-contestable %.0f%%: %d rows (flip-distance <= %.3f SD)\n",
              100 * x$contest_quantile,
              x$flip_summary$n_contested, x$contest_threshold))
  top <- utils::head(names(x$feature_voi), 3L)
  cat(sprintf("  top value-of-information: %s\n", paste(top, collapse = ", ")))
  cat("  plot() for the contestability lens; summary() for details.\n")
  invisible(x)
}

#' @rdname epinet_contestability
#' @export
summary.epinet_contestability <- function(object, ...) {
  s <- object$flip_summary
  fmt <- function(v) if (is.null(v)) "NA" else sprintf("%.3f", v)
  cat(sprintf("EpiNet contestability - summary (n = %d, metric = %s)\n",
              object$n, object$metric))
  cat(sprintf("  flip-distance (SD units): mean %s  median %s  sd %s  [%s, %s]\n",
              fmt(s$mean), fmt(s$median), fmt(s$std), fmt(s$min), fmt(s$max)))
  cat(sprintf("  contested (lowest %.0f%%): %d rows at threshold %s\n",
              100 * object$contest_quantile, s$n_contested, fmt(object$contest_threshold)))
  voi <- utils::head(unlist(object$feature_voi), 8L)
  cat("  value of information (top features):\n")
  for (nm in names(voi)) cat(sprintf("    %-22s %s\n", nm, fmt(voi[[nm]])))
  invisible(object)
}

#' @rdname epinet_contestability
#' @param top Number of value-of-information features to show (default 10).
#' @export
plot.epinet_contestability <- function(x, top = 10L, ...) {
  flip <- as.numeric(x$flip_distance)
  flip <- flip[is.finite(flip)]
  thr <- x$contest_threshold
  voi <- sort(unlist(x$feature_voi), decreasing = TRUE)
  voi <- utils::head(voi, as.integer(top))

  op <- graphics::par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  on.exit(graphics::par(op))

  # Left: flip-distance distribution, contested (fragile) tail shaded.
  h <- graphics::hist(flip, breaks = 30, plot = FALSE)
  cols <- ifelse(h$mids <= thr, "#C0392B", "#BDC3C7")
  graphics::plot(h, col = cols, border = "white",
                 main = sprintf("Contestability (%s)", x$outcome),
                 xlab = "flip-distance (SD units)")
  graphics::abline(v = thr, lty = 2, col = "#C0392B")
  graphics::legend("topright", bty = "n", fill = c("#C0392B", "#BDC3C7"),
                   legend = c(sprintf("contested (<=%.2f)", thr), "other"))

  # Right: value of information — features that most drive boundary flips.
  graphics::par(mar = c(4, 9, 3, 1))
  graphics::barplot(rev(voi), names.arg = rev(names(voi)), horiz = TRUE, las = 1,
                    col = "#5E4F99", border = NA,
                    xlab = "value of information",
                    main = "What drives the flips")
  invisible(x)
}
