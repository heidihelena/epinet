test_that("epinet fits a flat table and exposes summary/plot/print", {
  skip_if_not(
    reticulate::py_module_available("vahtian.epinet"),
    "Python 'vahtian.epinet' package not available to reticulate"
  )

  set.seed(1)
  n <- 140
  df <- data.frame(
    age = rnorm(n, 60, 10),
    sex = sample(c("M", "F"), n, replace = TRUE),
    smoking = sample(c("never", "former", "current"), n, replace = TRUE)
  )
  lp <- 0.04 * (df$age - 60) + 0.9 * (df$smoking == "current") - 0.5
  df$copd <- rbinom(n, 1, plogis(lp))

  fit <- epinet(
    df, outcome = "copd", predictors = c("age", "sex", "smoking"),
    n_bootstrap = 0
  )

  expect_s3_class(fit, "epinet")
  expect_equal(fit$outcome, "copd")
  expect_equal(fit$n, n)
  expect_true("smoking_current" %in% unlist(fit$features_used))
  expect_false(is.null(fit$metrics$balanced_accuracy))

  expect_output(print(fit), "EpiNet outcome model")
  expect_output(summary(fit), "Calibration")
  expect_true(length(fit$importance) > 0)
})

test_that("epinet validates its arguments", {
  df <- data.frame(a = 1:5, y = c(0, 1, 0, 1, 0))
  expect_error(epinet(df, outcome = c("a", "y")), "single column name")
  expect_error(epinet(list(a = 1), outcome = "a"), "data frame")
})
