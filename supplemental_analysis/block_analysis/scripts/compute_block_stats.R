# Block-number (game) interaction analyses.
#
# Nine Bayesian hierarchical regressions: six parallel to the headline
# analyses in feature_analysis (accuracy, RT by memories, choice by
# recalled offer value, quadratic RT on recalled offer value, relevant
# fixation, choice x valence fixation), plus three memory-performance
# checks (overall recall proportion, recalled-vs-true reward fidelity,
# location recall). Every fit adds `game_c` (the centered block-number
# predictor, range -3..3) to both the fixed- and random-effects
# structure, interacting with the main effect of interest to test
# whether that effect changes across the 7 rounds of the experiment.
# Maximal by-subject random effects throughout.
#
# Deliverable: one row per analysis in
#   output/block_interaction_table.csv
# with the posterior mean and 95% HDI for the interaction (or direct
# block) term of interest. Full brms summaries are saved alongside.
#
# Inputs  (data/)  choices.csv, eye_fixation_long.csv,
#                  eye_fixation_relevant.csv
# Outputs (output/) block_interaction_table.csv,
#                   model_summaries/<fit>_summary.{txt,csv}

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
# Memory CSVs are produced by the main paper pipeline; read directly.
repo_root <- normalizePath(file.path(base_dir, "..", ".."))
behav_dir <- file.path(repo_root, "output", "behavior", "stats")
summaries_dir <- file.path(out_dir, "model_summaries")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(summaries_dir, showWarnings = FALSE, recursive = TRUE)

brms_chains <- 4
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

summarise_term <- function(fit, term, analysis_label) {
  dr <- as_draws_df(fit)
  col <- paste0("b_", term)
  if (!(col %in% colnames(dr))) {
    stop(sprintf("Term '%s' not found in draws for %s",
                 term, analysis_label))
  }
  draws <- dr[[col]]
  hdi <- hdi_95(draws)
  data.frame(
    analysis = analysis_label,
    term = term,
    estimate = mean(draws),
    hdi_low = hdi["lo"],
    hdi_high = hdi["hi"],
    hdi_spans_zero = (hdi["lo"] < 0) & (hdi["hi"] > 0)
  )
}

# Data preparation -----------------------------------------------------------
d_choices <- readr::read_csv(file.path(data_dir, "choices.csv"),
                             show_col_types = FALSE) %>%
  dplyr::filter(rt > 0) %>%
  tidyr::drop_na(rt, correct, true_offer_value, recalled_offer_value,
                 recalled_total_count, game) %>%
  mutate(
    subject = as.factor(subject),
    choice_bin = as.integer(choice),
    correct_bin = as.integer(correct),
    log_rt = log(rt),
    game_c = as.numeric(game) - 4,
    true_offer_value_z = as.numeric(scale(true_offer_value)),
    recalled_offer_value_z = as.numeric(scale(recalled_offer_value)),
    true_offer_value_z2 = true_offer_value_z ^ 2,
    recalled_offer_value_z2 = recalled_offer_value_z ^ 2
  )

d_eye_rel <- readr::read_csv(file.path(data_dir, "eye_fixation_relevant.csv"),
                             show_col_types = FALSE) %>%
  tidyr::drop_na(delta_relevant, game) %>%
  mutate(
    subject = as.factor(subject),
    game_c = as.numeric(game) - 4
  )

d_eye_long <- readr::read_csv(file.path(data_dir, "eye_fixation_long.csv"),
                              show_col_types = FALSE) %>%
  tidyr::drop_na(delta_from_chance, decision_label, valence_label, game) %>%
  mutate(
    subject = as.factor(subject),
    game_c = as.numeric(game) - 4,
    decision_c = ifelse(decision_label == "take",    0.5, -0.5),
    valence_c  = ifelse(valence_label  == "positive", 0.5, -0.5)
  )

rows <- list()

# 1. Accuracy ~ game_c --------------------------------------------------------
m1 <- brm(
  correct_bin ~ 1 + game_c + (1 + game_c | subject),
  data = d_choices, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("accuracy_x_block", m1)
rows[["1"]] <- summarise_term(m1, "game_c", "Accuracy x block")

# 2. RT ~ recalled_total_count * game_c --------------------------------------
m2 <- brm(
  log_rt ~ 1 + recalled_total_count * game_c +
    (1 + recalled_total_count * game_c | subject),
  data = d_choices, family = gaussian(),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("rt_memories_x_block", m2)
rows[["2"]] <- summarise_term(m2, "recalled_total_count:game_c",
                              "RT ~ number of memories x block")

# 3. Choice ~ recalled offer value * game_c ----------------------------------
m3 <- brm(
  choice_bin ~ 1 + recalled_offer_value_z * game_c +
    (1 + recalled_offer_value_z * game_c | subject),
  data = d_choices, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("choice_recalled_x_block", m3)
rows[["3"]] <- summarise_term(m3, "recalled_offer_value_z:game_c",
                              "Choice ~ recalled offer value x block")

# 4. RT ~ (recalled offer value)^2 * game_c ----------------------------------
m4 <- brm(
  log_rt ~ 1 + (recalled_offer_value_z + recalled_offer_value_z2) * game_c +
    (1 + (recalled_offer_value_z + recalled_offer_value_z2) * game_c |
       subject),
  data = d_choices, family = gaussian(),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("rt_recalled_quadratic_x_block", m4)
rows[["4"]] <- summarise_term(m4, "recalled_offer_value_z2:game_c",
                              "RT ~ (recalled offer value)^2 x block")

# 5. Relevant fixation prop. ~ game_c ----------------------------------------
m5 <- brm(
  delta_relevant ~ 1 + game_c + (1 + game_c | subject),
  data = d_eye_rel, family = gaussian(),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("relevant_fixation_x_block", m5)
rows[["5"]] <- summarise_term(m5, "game_c",
                              "Relevant fixation prop. x block")

# 6. Fixation prop. ~ choice * valence * game_c ------------------------------
m6 <- brm(
  delta_from_chance ~ decision_c * valence_c * game_c +
    (decision_c * valence_c * game_c | subject),
  data = d_eye_long, family = gaussian(),
  chains = brms_chains, iter = 3000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.99, max_treedepth = 14)
)
save_fit_summary("fixation_choice_valence_x_block", m6)
rows[["6"]] <- summarise_term(m6, "decision_c:valence_c:game_c",
                              "Fixation prop. ~ choice x valence x block")

# 7. Item recall proportion ~ game_c ----------------------------------------
d_recall <- readr::read_csv(file.path(behav_dir, "recall_prop_df.csv"),
                            show_col_types = FALSE) %>%
  tidyr::drop_na(recall_prop_centered, game) %>%
  mutate(
    subject = as.factor(subject),
    game_c = as.numeric(game) - 4
  )

m7 <- brm(
  recall_prop_centered ~ 1 + game_c + (1 + game_c | subject),
  data = d_recall, family = gaussian(),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("recall_prop_x_block", m7)
rows[["7"]] <- summarise_term(m7, "game_c",
                              "Item recall proportion x block")

# 8. Recalled-vs-true reward x game_c ---------------------------------------
d_item <- readr::read_csv(file.path(behav_dir, "item_values_df.csv"),
                          show_col_types = FALSE) %>%
  tidyr::drop_na(true_value, recalled_value, game) %>%
  mutate(
    subject = as.factor(subject),
    game_c = as.numeric(game) - 4
  )

m8 <- brm(
  true_value ~ 1 + recalled_value * game_c +
    (1 + recalled_value * game_c | subject),
  data = d_item, family = gaussian(),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("value_recall_x_block", m8)
rows[["8"]] <- summarise_term(m8, "recalled_value:game_c",
                              "True ~ recalled reward x block")

# 9. Spatial (location) recall ~ game_c -------------------------------------
d_spatial <- readr::read_csv(file.path(behav_dir, "spatial_accuracy_df.csv"),
                             show_col_types = FALSE) %>%
  tidyr::drop_na(spatial_correct_bin, game) %>%
  mutate(
    subject = as.factor(subject),
    game_c = as.numeric(game) - 4
  )

m9 <- brm(
  spatial_correct_bin ~ 1 + game_c + (1 + game_c | subject),
  data = d_spatial, family = bernoulli(link = "logit"),
  chains = brms_chains, iter = 2000, cores = brms_cores,
  seed = brms_seed, control = list(adapt_delta = 0.95)
)
save_fit_summary("spatial_recall_x_block", m9)
rows[["9"]] <- summarise_term(m9, "game_c", "Location recall x block")

# ---------------------------------------------------------------------------
out <- dplyr::bind_rows(rows)
out_path <- file.path(out_dir, "block_interaction_table.csv")
readr::write_csv(out, out_path)
cat("Wrote", out_path, "\n")
print(out)
