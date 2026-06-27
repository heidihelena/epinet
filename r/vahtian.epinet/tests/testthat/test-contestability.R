test_that("epinet_contestability scores rows and exposes the lens", {
  skip_if_not(
    reticulate::py_module_available("epinet"),
    "Python 'epinet' package not available to reticulate"
  )

  set.seed(2)
  n <- 150
  df <- data.frame(
    age = rnorm(n, 60, 10),
    sex = sample(c("M", "F"), n, replace = TRUE),
    smoking = sample(c("never", "former", "current"), n, replace = TRUE)
  )
  lp <- 0.05 * (df$age - 60) + 0.9 * (df$smoking == "current") - 0.5
  df$copd <- rbinom(n, 1, plogis(lp))

  cst <- epinet_contestability(df, outcome = "copd",
                               predictors = c("age", "sex", "smoking"),
                               contest_quantile = 0.1)

  expect_s3_class(cst, "epinet_contestability")
  expect_length(cst$flip_distance, n)
  expect_length(cst$contested, n)
  # ~10% flagged contestable at the 0.1 quantile.
  expect_gt(sum(unlist(cst$contested)), 0)
  expect_lt(sum(unlist(cst$contested)), n)
  expect_true(!is.null(cst$contest_threshold))
  expect_true(length(cst$feature_voi) > 0)

  expect_output(print(cst), "contestability")
  expect_output(summary(cst), "value of information")
  # plot() must render without error (PDF null device).
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_silent(plot(cst))
})
