# Federated surface: partition rows across sites, reconstruct the fit from
# per-site aggregates only, and show it matches the centralized run.

#' Federate the fit across sites and check it reconstructs the centralized run
#'
#' Partitions the rows across sites (by the `site` column if given, otherwise
#' into `n_sites` balanced random groups) and runs EpiNet's federated
#' reconstruction: only per-site aggregates cross, never rows. The result reports
#' the per-site sizes and the maximum absolute differences from the centralized
#' fit (and contestability), which should be at floating-point level —
#' demonstrating the federation is exact.
#'
#' @param data A data frame: one row per subject.
#' @param outcome Name of the outcome (label) column.
#' @param predictors Character vector of predictor columns (default: all but
#'   `outcome` and, if used, `site`). Non-numeric predictors are one-hot encoded.
#' @param site Optional column name giving each row's site. If `NULL`, rows are
#'   split into `n_sites` random groups.
#' @param n_sites Number of synthetic sites when `site` is `NULL` (default 2).
#' @param metric Distance metric for the contestability round-trip.
#' @param contest_quantile Contested fraction for the contestability round-trip.
#' @param random_state Integer seed for the random site split.
#' @return An object of class `"epinet_federated"`.
#' @examples
#' \dontrun{
#' fed <- epinet_federated(data, outcome = "copd",
#'                         predictors = c("age", "sex", "smoking"), n_sites = 3)
#' summary(fed)
#' plot(fed)   # per-site sizes + reconstruction error vs centralized
#' }
#' @export
epinet_federated <- function(data, outcome, predictors = NULL, site = NULL,
                             n_sites = 2L, metric = "euclidean",
                             contest_quantile = 0.1, random_state = 42L) {
  if (!is.data.frame(data)) stop("`data` must be a data frame", call. = FALSE)
  if (is.null(predictors)) {
    predictors <- setdiff(names(data), c(outcome, site))
  }
  api <- .epinet_api()
  res <- api$federated(
    data = data, outcome = outcome,
    predictors = as.list(as.character(predictors)),
    site = site, n_sites = as.integer(n_sites), metric = metric,
    contest_quantile = as.numeric(contest_quantile),
    random_state = as.integer(random_state)
  )
  structure(res, class = "epinet_federated")
}

#' @rdname epinet_federated
#' @param x,object An `"epinet_federated"` object.
#' @param ... Unused.
#' @export
print.epinet_federated <- function(x, ...) {
  cat(sprintf("EpiNet federated fit  (n = %d across %d sites, outcome = '%s')\n",
              x$n, x$n_sites, x$outcome))
  d <- x$fit_diffs
  cat(sprintf("  max reconstruction diff vs centralized: mean %.1e  sd %.1e  centroid %.1e\n",
              d$mean, d$sd, d$centroid))
  worst <- max(d$mean, d$sd, d$centroid)
  cat(sprintf("  -> federation is %s (worst diff %.1e)\n",
              if (worst < 1e-8) "EXACT to floating point" else "approximate", worst))
  cat("  plot() for per-site sizes + reconstruction error.\n")
  invisible(x)
}

#' @rdname epinet_federated
#' @export
summary.epinet_federated <- function(object, ...) {
  cat(sprintf("EpiNet federated fit - summary (n = %d, sites = %d)\n",
              object$n, object$n_sites))
  cat("  rows per site:\n")
  sizes <- unlist(object$sites)
  for (s in names(sizes)) cat(sprintf("    %-12s %d\n", s, sizes[[s]]))
  d <- object$fit_diffs
  cat("  fit reconstruction vs centralized (max abs diff):\n")
  cat(sprintf("    mean %.2e   sd %.2e   centroid %.2e\n", d$mean, d$sd, d$centroid))
  cd <- object$contestability_diffs
  if (!is.null(cd)) {
    cat("  contestability reconstruction (max abs diff):\n")
    cat(sprintf("    flip-distance mean %.2e   std %.2e\n", cd$mean, cd$std))
    cat(sprintf("    runner-up counts match: %s   top VOI feature match: %s\n",
                isTRUE(object$runner_up_match), isTRUE(object$top_voi_match)))
  }
  cat("  Only per-site aggregates cross; rows never leave a site.\n")
  invisible(object)
}

#' @rdname epinet_federated
#' @export
plot.epinet_federated <- function(x, ...) {
  sizes <- unlist(x$sites)
  diffs <- c("fit: mean" = x$fit_diffs$mean,
             "fit: sd" = x$fit_diffs$sd,
             "fit: centroid" = x$fit_diffs$centroid)
  cd <- x$contestability_diffs
  if (!is.null(cd)) {
    if (!is.null(cd$mean)) diffs["contest: mean"] <- cd$mean
    if (!is.null(cd$std)) diffs["contest: std"] <- cd$std
  }
  err <- pmax(as.numeric(diffs), 1e-18)  # floor zeros for the log scale

  op <- graphics::par(mfrow = c(1, 2), mar = c(4, 8, 3, 1))
  on.exit(graphics::par(op))

  # Left: how the rows were partitioned across sites.
  graphics::barplot(rev(sizes), names.arg = rev(names(sizes)), horiz = TRUE,
                    las = 1, col = "#5E4F99", border = NA, xlab = "rows",
                    main = "Rows per site")

  # Right: reconstruction error vs centralized, log10. Bars left of the
  # tolerance line mean the federated result equals the centralized one.
  lg <- log10(err)
  graphics::barplot(rev(lg), names.arg = rev(names(diffs)), horiz = TRUE, las = 1,
                    col = "#27AE60", border = NA, xlim = c(min(lg, -16) - 1, 0),
                    xlab = "log10(max abs diff vs centralized)",
                    main = "Federated == centralized")
  graphics::abline(v = log10(1e-10), lty = 2, col = "#C0392B")
  graphics::mtext("exact tol.", side = 3, at = log10(1e-10), col = "#C0392B", cex = 0.7)
  invisible(x)
}
