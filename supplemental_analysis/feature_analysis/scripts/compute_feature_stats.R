# Per-feature-dimension model fits + deviation-from-grand-mean summaries.
#
# Fits eight Bayesian hierarchical regressions, one per analysis of
# interest, each reparameterised so that every feature dimension
# (animacy, environment, size, texture) receives its own fixed-effect
# coefficient with maximal by-subject random effects. For every
# analysis we harvest the per-dimension coefficient of interest and
# report the posterior deviation of each dimension from the grand mean
# of the four per-dimension coefficients, with 95% HDIs.
#
# Inputs  (data/)       choices.csv, eye_fixation_long.csv,
#                       eye_fixation_relevant.csv
# Outputs (output/)     feature_deviation_from_mean.csv,
#                       model_summaries/<fit>_summary.{txt,csv}

suppressPackageStartupMessages({
  library(brms)
  library(readr)
  library(dplyr)
  library(broom.mixed)
  library(posterior)
})

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(file_arg) > 0) {
  base_dir <- normalizePath(file.path(dirname(file_arg), ".."))
} else {
  base_dir <- getwd()
}

data_dir <- file.path(base_dir, "data")
out_dir <- file.path(base_dir, "output")
summaries_dir <- file.path(out_dir, "model_summaries")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(summaries_dir, showWarnings = FALSE, recursive = TRUE)

FEATURE_LEVELS <- c("animacy", "environment", "size", "texture")

brms_chains <- 4
brms_iter <- 2000
brms_cores <- min(4, parallel::detectCores())
brms_seed <- 2026

hdi_95 <- function(x) {
  q <- stats::quantile(x, c(0.025, 0.975), na.rm = TRUE)
  c(lo = unname(q[1]), hi = unname(q[2]))
}

save_fit_summary <- function(name, fit) {
  writeLines(capture.output(summary(fit)),
             file.path(summaries_dir, paste0(name, "_summary.txt")))
  tidy_df <- tryCatch(broom.mixed::tidy(fit, effects = "fixed"),
                      error = function(e) NULL)
  if (!is.null(tidy_df)) {
    readr::write_csv(tidy_df,
                     file.path(summaries_dir, paste0(name, "_summary.csv")))
  }
}

summarise_dev_from_mean <- function(draws_mat, label) {
  stopifnot(ncol(draws_mat) == 4L)
  colnames(draws_mat) <- FEATURE_LEVELS
  grand <- rowMeans(draws_mat)
  dev_mat <- draws_mat - grand
  do.call(rbind, lapply(FEATURE_LEVELS, function(dim) {
    d <- dev_mat[, dim]
    hdi <- hdi_95(d)
    data.frame(
      analysis = label,
      feature_dim = dim,
      dev_mean = mean(d),
      dev_median = stats::median(d),
      lo95 = hdi["lo"],
      hi95 = hdi["hi"],
      hdi_spans_zero = (hdi["lo"] < 0) & (hdi["hi"] > 0)
    )
  }))
}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
d_choices <- readr::read_csv(file.path(data_dir, "choices.csv"),
                             show_col_types = FALSE) %>%
  dplyr::filter(rt > 0) %>%
  tidyr::drop_na(rt, correct, true_offer_value, recalled_offer_value,
                 recalled_total_count, feature_dim) %>%
  mutate(
    subject = as.factor(subject),
    feature_dim = factor(feature_dim, levels = FEATURE_LEVELS),
    choice_bin = as.integer(choice),
    correct_bin = as.integer(correct),
    log_rt = log(rt),
    true_offer_value_z = as.numeric(scale(true_offer_value)),
    recalled_offer_value_z = as.numeric(scale(recalled_offer_value)),
    true_offer_value_z2 = true_offer_value_z ^ 2,
    recalled_offer_value_z2 = recalled_offer_value_z ^ 2
  )

d_eye_rel <- readr::read_csv(file.path(data_dir, "eye_fixation_relevant.csv"),
                             show_col_types = FALSE) %>%
  tidyr::drop_na(delta_relevant, feature_dim) %>%
  mutate(
    subject = as.factor(subject),
    feature_dim = factor(feature_dim, levels = FEATURE_LEVELS)
  )

d_eye_long <- readr::read_csv(file.path(data_dir, "eye_fixation_long.csv"),
                              show_col_types = FALSE) %>%
  tidyr::drop_na(delta_from_chance, decision_label, valence_label, feature_dim) %>%
  mutate(
    subject = as.factor(subject),
    feature_dim = factor(feature_dim, levels = FEATURE_LEVELS),
    decision_c = ifelse(decision_label == "take",    0.5, -0.5),
    valence_c  = ifelse(valence_label  == "positive", 0.5, -0.5)
  )

dev_rows <- list()

# ---------------------------------------------------------------------------
# 1. Accuracy ~ 1  (per-dim log-odds, no covariate)
# ---------------------------------------------------------------------------
m1 <- brm(
  correct_bin ~ 0 + feature_dim + (0 + feature_dim | subject),
  data = d_choices, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("accuracy_by_dim", m1)
mat1 <- as_draws_df(m1)
mat1 <- cbind(mat1[["b_feature_dimanimacy"]], mat1[["b_feature_dimenvironment"]],
              mat1[["b_feature_dimsize"]], mat1[["b_feature_dimtexture"]])
dev_rows[["1"]] <- summarise_dev_from_mean(
  mat1, "Accuracy (logit, no covariate)")

# ---------------------------------------------------------------------------
# 2. log(RT) ~ number of recalled memories  (per-dim slope)
# ---------------------------------------------------------------------------
m2 <- brm(
  log_rt ~ 0 + feature_dim + feature_dim:recalled_total_count +
    (0 + feature_dim + feature_dim:recalled_total_count | subject),
  data = d_choices, family = gaussian(),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("rt_by_dim_memories", m2)
dr <- as_draws_df(m2)
mat2 <- cbind(
  dr[["b_feature_dimanimacy:recalled_total_count"]],
  dr[["b_feature_dimenvironment:recalled_total_count"]],
  dr[["b_feature_dimsize:recalled_total_count"]],
  dr[["b_feature_dimtexture:recalled_total_count"]]
)
dev_rows[["2"]] <- summarise_dev_from_mean(
  mat2, "RT slope (log-RT per recalled item)")

# ---------------------------------------------------------------------------
# 3. Choice ~ true offer value  (per-dim logit slope)
# ---------------------------------------------------------------------------
m3 <- brm(
  choice_bin ~ 0 + feature_dim + feature_dim:true_offer_value_z +
    (0 + feature_dim + feature_dim:true_offer_value_z | subject),
  data = d_choices, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("choice_by_dim_true", m3)
dr <- as_draws_df(m3)
mat3 <- cbind(
  dr[["b_feature_dimanimacy:true_offer_value_z"]],
  dr[["b_feature_dimenvironment:true_offer_value_z"]],
  dr[["b_feature_dimsize:true_offer_value_z"]],
  dr[["b_feature_dimtexture:true_offer_value_z"]]
)
dev_rows[["3"]] <- summarise_dev_from_mean(
  mat3, "Choice ~ true offer value (per-dim logit slope)")

# ---------------------------------------------------------------------------
# 4. Choice ~ recalled offer value  (per-dim logit slope)
# ---------------------------------------------------------------------------
m4 <- brm(
  choice_bin ~ 0 + feature_dim + feature_dim:recalled_offer_value_z +
    (0 + feature_dim + feature_dim:recalled_offer_value_z | subject),
  data = d_choices, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("choice_by_dim_recalled", m4)
dr <- as_draws_df(m4)
mat4 <- cbind(
  dr[["b_feature_dimanimacy:recalled_offer_value_z"]],
  dr[["b_feature_dimenvironment:recalled_offer_value_z"]],
  dr[["b_feature_dimsize:recalled_offer_value_z"]],
  dr[["b_feature_dimtexture:recalled_offer_value_z"]]
)
dev_rows[["4"]] <- summarise_dev_from_mean(
  mat4, "Choice ~ recalled offer value (per-dim logit slope)")

# ---------------------------------------------------------------------------
# 5. log(RT) ~ (true offer value)^2  (per-dim quadratic coefficient)
# ---------------------------------------------------------------------------
m5 <- brm(
  log_rt ~ 0 + feature_dim +
    feature_dim:true_offer_value_z + feature_dim:true_offer_value_z2 +
    (0 + feature_dim +
       feature_dim:true_offer_value_z + feature_dim:true_offer_value_z2 |
       subject),
  data = d_choices, family = gaussian(),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("rt_by_dim_true_quadratic", m5)
dr <- as_draws_df(m5)
mat5 <- cbind(
  dr[["b_feature_dimanimacy:true_offer_value_z2"]],
  dr[["b_feature_dimenvironment:true_offer_value_z2"]],
  dr[["b_feature_dimsize:true_offer_value_z2"]],
  dr[["b_feature_dimtexture:true_offer_value_z2"]]
)
dev_rows[["5"]] <- summarise_dev_from_mean(
  mat5, "RT ~ (true offer value)^2 (per-dim quadratic)")

# ---------------------------------------------------------------------------
# 6. log(RT) ~ (recalled offer value)^2  (per-dim quadratic coefficient)
# ---------------------------------------------------------------------------
m6 <- brm(
  log_rt ~ 0 + feature_dim +
    feature_dim:recalled_offer_value_z + feature_dim:recalled_offer_value_z2 +
    (0 + feature_dim +
       feature_dim:recalled_offer_value_z + feature_dim:recalled_offer_value_z2 |
       subject),
  data = d_choices, family = gaussian(),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("rt_by_dim_recalled_quadratic", m6)
dr <- as_draws_df(m6)
mat6 <- cbind(
  dr[["b_feature_dimanimacy:recalled_offer_value_z2"]],
  dr[["b_feature_dimenvironment:recalled_offer_value_z2"]],
  dr[["b_feature_dimsize:recalled_offer_value_z2"]],
  dr[["b_feature_dimtexture:recalled_offer_value_z2"]]
)
dev_rows[["6"]] <- summarise_dev_from_mean(
  mat6, "RT ~ (recalled offer value)^2 (per-dim quadratic)")

# ---------------------------------------------------------------------------
# 7. Relevant fixation prop. ~ 1  (per-dim intercept on delta_relevant)
# ---------------------------------------------------------------------------
m7 <- brm(
  delta_relevant ~ 0 + feature_dim + (0 + feature_dim | subject),
  data = d_eye_rel, family = gaussian(),
  chains = brms_chains, iter = brms_iter, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("relevant_fixation_by_dim", m7)
dr <- as_draws_df(m7)
mat7 <- cbind(
  dr[["b_feature_dimanimacy"]],
  dr[["b_feature_dimenvironment"]],
  dr[["b_feature_dimsize"]],
  dr[["b_feature_dimtexture"]]
)
dev_rows[["7"]] <- summarise_dev_from_mean(
  mat7, "Relevant-fixation deviation from chance")

# ---------------------------------------------------------------------------
# 8. Fixation prop. ~ choice × valence
#    (full feature_dim × decision × valence interaction; per-dim three-way term)
# ---------------------------------------------------------------------------
m8 <- brm(
  delta_from_chance ~
    0 + feature_dim +
    feature_dim:decision_c +
    feature_dim:valence_c +
    feature_dim:decision_c:valence_c +
    (0 + feature_dim +
       feature_dim:decision_c +
       feature_dim:valence_c +
       feature_dim:decision_c:valence_c | subject),
  data = d_eye_long, family = gaussian(),
  chains = brms_chains, iter = 3000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.99, max_treedepth = 14)
)
save_fit_summary("fixation_by_dim_choice_valence", m8)
dr <- as_draws_df(m8)
mat8 <- cbind(
  dr[["b_feature_dimanimacy:decision_c:valence_c"]],
  dr[["b_feature_dimenvironment:decision_c:valence_c"]],
  dr[["b_feature_dimsize:decision_c:valence_c"]],
  dr[["b_feature_dimtexture:decision_c:valence_c"]]
)
dev_rows[["8"]] <- summarise_dev_from_mean(
  mat8, "Choice x Valence interaction on Delta relevant fixation")

# ---------------------------------------------------------------------------
dev_out <- dplyr::bind_rows(dev_rows)
dev_path <- file.path(out_dir, "feature_deviation_from_mean.csv")
readr::write_csv(dev_out, dev_path)
cat("Wrote", dev_path, "\n")
print(dev_out)
