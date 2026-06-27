test_that("epinet_graph builds the network, fits the model, and plots", {
  skip_if_not(
    reticulate::py_module_available("vahtian.epinet"),
    "Python 'vahtian.epinet' package not available to reticulate"
  )

  set.seed(3)
  n <- 60
  nodes <- data.frame(
    ID = paste0("N", seq_len(n)),
    Outcome = rbinom(n, 1, 0.5),
    Feature1 = rnorm(n)
  )
  # A handful of random edges.
  e <- 90
  edges <- data.frame(
    SourceID = paste0("N", sample.int(n, e, replace = TRUE)),
    TargetID = paste0("N", sample.int(n, e, replace = TRUE))
  )
  edges <- edges[edges$SourceID != edges$TargetID, ]

  g <- epinet_graph(nodes, edges, outcome = "Outcome", n_bootstrap = 0)

  expect_s3_class(g, "epinet_graph")
  expect_equal(g$n_nodes, n)
  expect_gt(g$n_edges, 0)
  expect_true("degree" %in% unlist(g$feature_columns))
  expect_length(g$nodes, n)
  expect_false(is.null(g$metrics$balanced_accuracy))

  expect_output(print(g), "EpiNet graph model")
  expect_output(summary(g), "network:")

  # plot() must render without error whether or not igraph is present.
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(g))
})
