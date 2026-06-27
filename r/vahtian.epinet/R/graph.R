# Graph surface: build the network, derive graph features, fit the honest
# outcome model, and draw the network natively in R (igraph if available).

#' Build a graph, derive graph features, and fit the outcome model
#'
#' Constructs the node/edge graph, computes graph features (degree, clustering,
#' component size, optional centrality), joins them with node attributes, and
#' fits EpiNet's honestly-evaluated outcome model. The result carries the model
#' metrics plus the node/edge structure for plotting.
#'
#' @param nodes A data frame of nodes (must include `id_column` and `outcome`).
#' @param edges A data frame of edges (must include `source_column`,
#'   `target_column`).
#' @param outcome Name of the outcome column in `nodes`.
#' @param id_column Node id column (default "ID").
#' @param source_column,target_column Edge endpoint columns (defaults
#'   "SourceID"/"TargetID").
#' @param directed Treat edges as directed (default FALSE).
#' @param include_centrality Also compute betweenness/closeness/PageRank
#'   (default FALSE; slower on large graphs).
#' @param n_iterations,n_bootstrap,random_state Passed to the outcome model.
#' @return An object of class `"epinet_graph"`.
#' @examples
#' \dontrun{
#' g <- epinet_graph(nodes, edges, outcome = "Outcome")
#' summary(g)
#' plot(g)   # network coloured by outcome, sized by degree (needs igraph)
#' }
#' @export
epinet_graph <- function(nodes, edges, outcome,
                         id_column = "ID", source_column = "SourceID",
                         target_column = "TargetID", directed = FALSE,
                         include_centrality = FALSE, n_iterations = 1L,
                         n_bootstrap = 1000L, random_state = 42L) {
  if (!is.data.frame(nodes) || !is.data.frame(edges)) {
    stop("`nodes` and `edges` must be data frames", call. = FALSE)
  }
  api <- .epinet_api()
  res <- api$graph(
    nodes = nodes, edges = edges, outcome = outcome,
    id_column = id_column, source_column = source_column,
    target_column = target_column, directed = isTRUE(directed),
    include_centrality = isTRUE(include_centrality),
    n_iterations = as.integer(n_iterations),
    n_bootstrap = as.integer(n_bootstrap),
    random_state = as.integer(random_state)
  )
  structure(res, class = "epinet_graph")
}

# Rebuild node/edge data frames from the adapter's list-of-records.
.epinet_graph_frames <- function(x) {
  nd <- do.call(rbind, lapply(x$nodes, function(r) data.frame(
    id = as.character(r$id), degree = as.numeric(r$degree),
    community = as.integer(r$community),
    outcome = if (is.null(r$outcome)) NA_character_ else as.character(r$outcome),
    stringsAsFactors = FALSE
  )))
  ed <- if (length(x$edges)) do.call(rbind, lapply(x$edges, function(r) data.frame(
    source = as.character(r$source), target = as.character(r$target),
    stringsAsFactors = FALSE
  ))) else data.frame(source = character(0), target = character(0))
  list(nodes = nd, edges = ed)
}

#' @rdname epinet_graph
#' @param x,object An `"epinet_graph"` object.
#' @param ... Unused.
#' @export
print.epinet_graph <- function(x, ...) {
  m <- x$metrics
  cat(sprintf("EpiNet graph model  (%d nodes, %d edges, outcome = '%s')\n",
              x$n_nodes, x$n_edges, x$outcome))
  cat(sprintf("  graph features: %s\n", paste(unlist(x$feature_columns), collapse = ", ")))
  if (!is.null(m[["balanced_accuracy"]])) {
    cat(sprintf("  balanced accuracy: %.3f", m[["balanced_accuracy"]]))
    if (!is.null(m[["roc_auc"]])) cat(sprintf("   AUROC: %.3f", m[["roc_auc"]]))
    cat("\n")
  }
  cat("  plot() for the network; summary() for the model.\n")
  invisible(x)
}

#' @rdname epinet_graph
#' @export
summary.epinet_graph <- function(object, ...) {
  m <- object$metrics
  fmt <- function(v) if (is.null(v)) "NA" else sprintf("%.3f", v)
  cat(sprintf("EpiNet graph model - summary\n"))
  cat(sprintf("  network: %d nodes, %d edges%s\n", object$n_nodes, object$n_edges,
              if (isTRUE(object$directed)) " (directed)" else ""))
  cat(sprintf("  graph features: %s\n", paste(unlist(object$feature_columns), collapse = ", ")))
  cat(sprintf("  discrimination: AUROC %s   average precision %s\n",
              fmt(m[["roc_auc"]]), fmt(m[["average_precision"]])))
  cat(sprintf("  classification: balanced accuracy %s   MCC %s\n",
              fmt(m[["balanced_accuracy"]]), fmt(m[["mcc"]])))
  warns <- m[["data_warnings"]]
  if (!is.null(warns) && length(warns)) {
    cat("  data warnings:\n")
    for (w in unlist(warns)) cat(sprintf("    - %s\n", w))
  }
  invisible(object)
}

#' @rdname epinet_graph
#' @export
plot.epinet_graph <- function(x, ...) {
  fr <- .epinet_graph_frames(x)
  nd <- fr$nodes
  pal <- c("#5E4F99", "#C0392B", "#27AE60", "#E67E22", "#2980B9")
  lvls <- sort(unique(nd$outcome[!is.na(nd$outcome)]))
  vcol <- ifelse(is.na(nd$outcome), "#CCCCCC", pal[match(nd$outcome, lvls)])

  if (!requireNamespace("igraph", quietly = TRUE)) {
    # Graceful fallback: no igraph -> degree distribution instead of the network.
    message("Install 'igraph' for the network plot; showing the degree distribution.")
    op <- graphics::par(mar = c(4, 4, 3, 1)); on.exit(graphics::par(op))
    graphics::hist(nd$degree, breaks = 20, col = "#5E4F99", border = "white",
                   main = sprintf("Node degree (%s)", x$outcome), xlab = "degree")
    return(invisible(x))
  }

  g <- igraph::graph_from_data_frame(d = fr$edges, directed = isTRUE(x$directed),
                                     vertices = nd)
  deg <- nd$degree
  vsize <- 3 + 9 * (deg / max(deg, 1))
  op <- graphics::par(mar = c(1, 1, 3, 1)); on.exit(graphics::par(op))
  set.seed(1)
  plot(g, vertex.color = vcol, vertex.size = vsize, vertex.frame.color = "white",
       vertex.label = NA, edge.color = "#DDDDDD",
       layout = igraph::layout_with_fr,
       main = sprintf("EpiNet network (%s), coloured by outcome, sized by degree",
                      x$outcome))
  if (length(lvls)) {
    graphics::legend("bottomleft", bty = "n", pch = 21, pt.cex = 1.6,
                     pt.bg = c(pal[seq_along(lvls)], "#CCCCCC"),
                     legend = c(paste0(x$outcome, " = ", lvls), "unlabeled"))
  }
  invisible(x)
}
