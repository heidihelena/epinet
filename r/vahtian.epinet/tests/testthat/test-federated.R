test_that("epinet_federated reconstructs the centralized fit exactly", {
  skip_if_not(
    reticulate::py_module_available("vahtian.epinet"),
    "Python 'vahtian.epinet' package not available to reticulate"
  )

  set.seed(4)
  n <- 150
  df <- data.frame(
    age = rnorm(n, 60, 10),
    sex = sample(c("M", "F"), n, replace = TRUE),
    smoking = sample(c("never", "former", "current"), n, replace = TRUE)
  )
  df$copd <- rbinom(n, 1, plogis(0.05 * (df$age - 60) + 0.9 * (df$smoking == "current") - 0.5))

  fed <- epinet_federated(df, outcome = "copd",
                          predictors = c("age", "sex", "smoking"), n_sites = 3)

  expect_s3_class(fed, "epinet_federated")
  expect_equal(fed$n, n)
  expect_equal(fed$n_sites, 3)
  # Only aggregates cross, yet the fit matches the centralized run to ~fp.
  expect_lt(fed$fit_diffs$mean, 1e-8)
  expect_lt(fed$fit_diffs$sd, 1e-8)
  expect_lt(fed$fit_diffs$centroid, 1e-8)

  expect_output(print(fed), "federated fit")
  expect_output(summary(fed), "rows per site")

  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_no_error(plot(fed))
})

test_that("epinet_federated accepts an explicit site column", {
  skip_if_not(
    reticulate::py_module_available("vahtian.epinet"),
    "Python 'vahtian.epinet' package not available to reticulate"
  )
  set.seed(5)
  n <- 120
  df <- data.frame(
    x = rnorm(n), y = rbinom(n, 1, 0.5),
    hospital = sample(c("A", "B"), n, replace = TRUE)
  )
  fed <- epinet_federated(df, outcome = "y", predictors = "x", site = "hospital")
  expect_equal(sort(names(unlist(fed$sites))), c("A", "B"))
})
