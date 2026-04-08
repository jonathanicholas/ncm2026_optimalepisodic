# Bayesian tests comparing human vs NN on overview-figure measures.
#
# Measures tested: choice accuracy (A), proportion relevant fixation (E),
# decision x valence interaction (F), fixation count interaction,
# NN fixation duration ~ relevance * position.
#
# Usage:
#   Rscript metarnn/run_mixed_effects_human_vs_nn.R \
#     --nn-root metarnn/simulations/human_like_04_04_input5 \
#     --tag 04_04_input5 \
#     --out-dir metarnn/simulations/human_like_04_04_input5/output/human_vs_nn_brms

suppressPackageStartupMessages({
  library(brms)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(broom.mixed)
})

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

parse_kv_args <- function(args) {
  out <- list()
  i <- 1

  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    key <- gsub("-", "_", key)
    if (i + 1 <= length(args) && !startsWith(args[i + 1], "--")) {
      out[[key]] <- args[i + 1]
      i <- i + 2
    } else {
      out[[key]] <- TRUE
      i <- i + 1
    }
  }
  out
}

cli_args <- parse_kv_args(commandArgs(trailingOnly = TRUE))

nn_root <- cli_args$nn_root
if (is.null(nn_root)) stop("--nn-root is required")
nn_root <- normalizePath(nn_root, mustWork = TRUE)

tag <- if (!is.null(cli_args$tag)) cli_args$tag else ""

project_root <- getwd()
human_data_dir <- file.path(project_root, "output")

out_dir <- if (!is.null(cli_args$out_dir)) {
  normalizePath(cli_args$out_dir, mustWork = FALSE)
} else {
  file.path(nn_root, "output", "human_vs_nn_brms")
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}
ensure_dir(out_dir)

# ---------------------------------------------------------------------------
# brms sampling configuration (matches existing scripts)
# ---------------------------------------------------------------------------

brms_chains <- 4
brms_iter   <- 2000
brms_cores  <- min(4, parallel::detectCores())
brms_seed   <- 2026

# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

save_model_outputs <- function(prefix, fit, nn_benchmark) {
  # Text summary
  summ_txt <- capture.output(summary(fit))
  writeLines(summ_txt, file.path(out_dir, paste0(prefix, "_summary.txt")))
  # Tidy fixed effects
  tidy_df <- tryCatch(
    broom.mixed::tidy(fit, effects = "fixed"),
    error = function(e) NULL
  )
  if (!is.null(tidy_df)) {
    readr::write_csv(tidy_df, file.path(out_dir, paste0(prefix, "_fixed.csv")))
  }
  # Hypothesis test: intercept = 0 (i.e. human - nn_benchmark = 0)
  tryCatch({
    h <- hypothesis(fit, "Intercept = 0")
    h_df <- as.data.frame(h$hypothesis)
    readr::write_csv(h_df, file.path(out_dir, paste0(prefix, "_hyp_intercept.csv")))
  }, error = function(e) {
    message("  Could not run hypothesis test: ", conditionMessage(e))
  })
  # Save NN benchmark value
  bench_df <- data.frame(nn_benchmark = nn_benchmark)
  readr::write_csv(bench_df, file.path(out_dir, paste0(prefix, "_nn_benchmark.csv")))
}

# ---------------------------------------------------------------------------
# Helper to find NN cache files
# ---------------------------------------------------------------------------

find_nn_cache <- function(pattern) {
  hits <- list.files(
    file.path(nn_root, "output"),
    pattern = pattern,
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(hits) == 0) stop("No file matching '", pattern, "' under ", nn_root, "/output/")
  hits[which.max(file.mtime(hits))]
}

# ===========================================================================
# Load and compute subject-level summaries
# ===========================================================================

# --- Human choice accuracy: per-subject mean ---
h_acc_path <- file.path(human_data_dir, "behavior", "stats", "subject_behavior_summary.csv")
h_acc <- readr::read_csv(h_acc_path, show_col_types = FALSE) %>%
  transmute(
    subject = paste0("H", subject),
    accuracy = choice_accuracy
  )

# --- NN choice accuracy: per-seed mean ---
nn_beh_path <- find_nn_cache("nn_trial_level_behavior_cached\\.csv$")
nn_beh <- readr::read_csv(nn_beh_path, show_col_types = FALSE)
nn_acc <- nn_beh %>%
  group_by(subject) %>%
  summarise(accuracy = mean(correct, na.rm = TRUE), .groups = "drop")

# --- Human prop relevant fixation: per-subject mean ---
h_rel_path <- file.path(human_data_dir, "eyegaze", "stats", "choice_fixation_relevance_subject_means_relevant_only_duration.csv")
h_rel <- readr::read_csv(h_rel_path, show_col_types = FALSE) %>%
  transmute(
    subject = paste0("H", subject),
    prop_relevant = mean_prop
  )

# --- NN prop relevant fixation: per-seed mean ---
nn_eye_path <- find_nn_cache("Figure3_NN_trial_level_duration\\.csv$")
nn_eye <- readr::read_csv(nn_eye_path, show_col_types = FALSE)
nn_rel <- nn_eye %>%
  group_by(subject) %>%
  summarise(prop_relevant = mean(prop_relevant, na.rm = TRUE), .groups = "drop")

# --- Human interaction score: per-subject ---
h_tl_path <- file.path(human_data_dir, "eyegaze", "stats", "choice_fixation_relsign4_relevant_subject_means_duration.csv")
h_tl <- readr::read_csv(h_tl_path, show_col_types = FALSE)

h_interaction <- h_tl %>%
  select(subject, decision_label, valence_label, mean_prop) %>%
  pivot_wider(
    names_from = c(decision_label, valence_label),
    values_from = mean_prop,
    names_sep = "_"
  ) %>%
  mutate(
    interaction = (take_positive - take_negative) - (leave_positive - leave_negative)
  ) %>%
  transmute(
    subject = paste0("H", subject),
    interaction
  ) %>%
  drop_na(interaction)

# --- NN interaction score: per-seed ---
nn_interaction <- nn_eye %>%
  group_by(subject, decision_label) %>%
  summarise(
    mean_rel_pos = mean(rel_pos, na.rm = TRUE),
    mean_rel_neg = mean(rel_neg, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = decision_label,
    values_from = c(mean_rel_pos, mean_rel_neg),
    names_sep = "_"
  ) %>%
  mutate(
    interaction = (mean_rel_pos_take - mean_rel_neg_take) - (mean_rel_pos_leave - mean_rel_neg_leave)
  ) %>%
  transmute(
    subject = paste0("N", subject),
    interaction
  ) %>%
  drop_na(interaction)

# ===========================================================================
# 1) Choice accuracy: delta from NN benchmark
# ===========================================================================

message("\n[1/4] Choice accuracy (subject-level means)")

nn_acc_benchmark <- mean(nn_acc$accuracy)
message("  NN benchmark (mean across seeds): ", round(nn_acc_benchmark, 4))
message("  Human N: ", nrow(h_acc), "  |  Human mean: ", round(mean(h_acc$accuracy), 4))

d_acc <- h_acc %>%
  mutate(delta = accuracy - nn_acc_benchmark)

message("  Delta mean (human - NN): ", round(mean(d_acc$delta), 4))

m_acc <- brm(
  delta ~ 1,
  data = d_acc,
  family = gaussian(),
  chains = brms_chains,
  iter = brms_iter,
  cores = brms_cores,
  seed = brms_seed
)
save_model_outputs("accuracy_human_vs_nn", m_acc, nn_acc_benchmark)

# ===========================================================================
# 2) Proportion relevant fixation time: delta from NN benchmark
# ===========================================================================

message("\n[2/4] Proportion relevant fixation time (subject-level means)")

nn_rel_benchmark <- mean(nn_rel$prop_relevant)
message("  NN benchmark (mean across seeds): ", round(nn_rel_benchmark, 4))
message("  Human N: ", nrow(h_rel), "  |  Human mean: ", round(mean(h_rel$prop_relevant), 4))

d_rel <- h_rel %>%
  mutate(delta = prop_relevant - nn_rel_benchmark)

message("  Delta mean (human - NN): ", round(mean(d_rel$delta), 4))

m_rel <- brm(
  delta ~ 1,
  data = d_rel,
  family = gaussian(),
  chains = brms_chains,
  iter = brms_iter,
  cores = brms_cores,
  seed = brms_seed
)
save_model_outputs("prop_relevant_human_vs_nn", m_rel, nn_rel_benchmark)

# ===========================================================================
# 3) Decision x Valence interaction score: delta from NN benchmark
# ===========================================================================

message("\n[3/4] Decision x Valence interaction (subject-level scores)")

nn_int_benchmark <- mean(nn_interaction$interaction)
message("  NN benchmark (mean across seeds): ", round(nn_int_benchmark, 4))
message("  Human N: ", nrow(h_interaction), "  |  Human mean: ", round(mean(h_interaction$interaction), 4))

d_int <- h_interaction %>%
  mutate(delta = interaction - nn_int_benchmark)

message("  Delta mean (human - NN): ", round(mean(d_int$delta), 4))

m_int <- brm(
  delta ~ 1,
  data = d_int,
  family = gaussian(),
  chains = brms_chains,
  iter = brms_iter,
  cores = brms_cores,
  seed = brms_seed
)
save_model_outputs("interaction_human_vs_nn", m_int, nn_int_benchmark)

# ===========================================================================
# 4) Fixation count: delta ~ 1 + relevance + (1 | subject)
# ===========================================================================

message("\n[4/4] Fixation count interaction (relevant vs irrelevant)")

# --- Human fixation counts: per-subject ---
h_fix_path <- file.path(human_data_dir, "eyegaze", "stats",
                        "revisits_count_and_duration_by_subject_human.csv")
h_fix <- readr::read_csv(h_fix_path, show_col_types = FALSE) %>%
  transmute(
    subject  = paste0("H", subject),
    fix_rel  = allfix_count_relevant_per_trial,
    fix_irrel = allfix_count_irrelevant_per_trial
  )

# --- NN fixation counts: per-seed ---
nn_fix_file <- list.files(file.path(nn_root, "output", "eyegaze", "stats"),
                          pattern = "^revisits_count_and_duration_by_subject_nn",
                          full.names = TRUE)
if (length(nn_fix_file) == 0) stop("No NN revisits by-subject file found under ", nn_root)
nn_fix <- readr::read_csv(nn_fix_file[1], show_col_types = FALSE) %>%
  transmute(
    subject  = paste0("NN", subject),
    fix_rel  = allfix_count_relevant_per_trial,
    fix_irrel = allfix_count_irrelevant_per_trial
  )

nn_bench_rel   <- mean(nn_fix$fix_rel, na.rm = TRUE)
nn_bench_irrel <- mean(nn_fix$fix_irrel, na.rm = TRUE)

message("  NN benchmark (rel): ", round(nn_bench_rel, 4),
        "  |  NN benchmark (irrel): ", round(nn_bench_irrel, 4))
message("  Human N: ", nrow(h_fix),
        "  |  Human mean (rel): ", round(mean(h_fix$fix_rel, na.rm = TRUE), 4),
        "  |  Human mean (irrel): ", round(mean(h_fix$fix_irrel, na.rm = TRUE), 4))

d_fix_long <- rbind(
  data.frame(subject = h_fix$subject, relevance = 0.5,
             delta = h_fix$fix_rel - nn_bench_rel),
  data.frame(subject = h_fix$subject, relevance = -0.5,
             delta = h_fix$fix_irrel - nn_bench_irrel)
)

message("  Total per-trial delta (rel + irrel): ",
        round(mean(h_fix$fix_rel - nn_bench_rel, na.rm = TRUE) +
              mean(h_fix$fix_irrel - nn_bench_irrel, na.rm = TRUE), 4),
        "  (note: model intercept = per-condition avg = total / 2)")

m_fixcount <- brm(
  delta ~ 1 + relevance + (1 | subject),
  data = d_fix_long,
  family = gaussian(),
  chains = brms_chains,
  iter = brms_iter,
  cores = brms_cores,
  seed = brms_seed
)

# Summary
summ_txt <- capture.output(summary(m_fixcount))
writeLines(summ_txt, file.path(out_dir, "fixcount_interaction_human_vs_nn_summary.txt"))

# Fixed effects
tidy_df <- tryCatch(broom.mixed::tidy(m_fixcount, effects = "fixed"), error = function(e) NULL)
if (!is.null(tidy_df))
  readr::write_csv(tidy_df, file.path(out_dir, "fixcount_interaction_human_vs_nn_fixed.csv"))

# Hypothesis tests
tryCatch({
  h_int <- hypothesis(m_fixcount, "Intercept = 0")
  readr::write_csv(as.data.frame(h_int$hypothesis),
                   file.path(out_dir, "fixcount_interaction_human_vs_nn_hyp_intercept.csv"))
}, error = function(e) message("  Could not run intercept hypothesis test: ", conditionMessage(e)))

tryCatch({
  h_rel <- hypothesis(m_fixcount, "relevance = 0")
  readr::write_csv(as.data.frame(h_rel$hypothesis),
                   file.path(out_dir, "fixcount_interaction_human_vs_nn_hyp_relevance.csv"))
}, error = function(e) message("  Could not run relevance hypothesis test: ", conditionMessage(e)))

# NN benchmarks
readr::write_csv(
  data.frame(nn_benchmark_relevant = nn_bench_rel, nn_benchmark_irrelevant = nn_bench_irrel),
  file.path(out_dir, "fixcount_interaction_human_vs_nn_nn_benchmark.csv"))

# ===========================================================================
# 5) NN fixation duration ~ relevance * position (standalone, no comparison)
# ===========================================================================
# Tests whether log fixation duration varies by relevance and position
# in the NN simulation. No random effects, no human comparison.
# Position binned: 1-6 individually, 7+ collapsed to 7, treated as continuous.

message("\n[5/5] NN fixation duration ~ relevance * position")

nn_fix_dur_path <- file.path(nn_root, "output", "choice_fixations_clean_buffer_50.csv")

if (file.exists(nn_fix_dur_path)) {
  nn_fix_dur <- readr::read_csv(nn_fix_dur_path, show_col_types = FALSE) %>%
    filter(!is.na(fixation_duration), fixation_duration > 0) %>%
    mutate(
      log_duration = log(fixation_duration),
      position_bin = pmin(fixation_count, 7)
    )

  message("  N fixations = ", nrow(nn_fix_dur))
  message("  N seeds = ", n_distinct(nn_fix_dur$subject_id))

  m_nn_fixdur <- brm(
    log_duration ~ relevance * position_bin,
    data = nn_fix_dur,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )

  nn_fixdur_prefix <- paste0("fixdur_by_position_nn_", tag)
  nn_fixdur_dir <- file.path(nn_root, "output", "eyegaze", "stats")
  ensure_dir(nn_fixdur_dir)

  summ_txt <- capture.output(summary(m_nn_fixdur))
  writeLines(summ_txt, file.path(nn_fixdur_dir, paste0(nn_fixdur_prefix, "_summary.txt")))

  tidy_df <- tryCatch(broom.mixed::tidy(m_nn_fixdur, effects = "fixed"), error = function(e) NULL)
  if (!is.null(tidy_df))
    readr::write_csv(tidy_df, file.path(nn_fixdur_dir, paste0(nn_fixdur_prefix, "_fixed.csv")))

  message("  NN fixdur model saved to ", nn_fixdur_dir)
} else {
  message("  Skipping: NN fixation file not found at ", nn_fix_dur_path)
}

# ===========================================================================
# 6) NN proportion relevant fixated ~ position (standalone, no comparison)
# ===========================================================================
# Tests whether the proportion of fixations landing on relevant locations
# is above chance (0.5) and how it changes across fixation positions.
# Position binned: 1-6 individually, 7+ collapsed to 7, treated as continuous.

message("\n[6/7] NN proportion relevant ~ position")

if (file.exists(nn_fix_dur_path)) {
  nn_fix_data <- readr::read_csv(nn_fix_dur_path, show_col_types = FALSE) %>%
    filter(!is.na(fixation_duration), fixation_duration > 0) %>%
    mutate(
      position_bin = pmin(fixation_count, 7),
      subject_id = as.character(subject_id)
    )

  fix_prop_nn <- nn_fix_data %>%
    group_by(subject_id, position_bin) %>%
    summarise(
      prop_relevant = mean(relevance, na.rm = TRUE),
      n_fixations = n(),
      .groups = "drop"
    ) %>%
    mutate(delta_chance = prop_relevant - 0.5)

  message("  N seed x position obs = ", nrow(fix_prop_nn))

  m_nn_fixprop <- brm(
    delta_chance ~ 1 + position_bin,
    data = fix_prop_nn,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )

  nn_fixprop_prefix <- paste0("fixprop_by_position_nn_", tag)
  nn_fixprop_dir <- file.path(nn_root, "output", "eyegaze", "stats")
  ensure_dir(nn_fixprop_dir)

  writeLines(capture.output(summary(m_nn_fixprop)),
             file.path(nn_fixprop_dir, paste0(nn_fixprop_prefix, "_summary.txt")))
  tidy_df <- tryCatch(broom.mixed::tidy(m_nn_fixprop, effects = "fixed"), error = function(e) NULL)
  if (!is.null(tidy_df))
    readr::write_csv(tidy_df, file.path(nn_fixprop_dir, paste0(nn_fixprop_prefix, "_fixed.csv")))

  message("  NN fixprop model saved to ", nn_fixprop_dir)
} else {
  message("  Skipping: NN fixation file not found at ", nn_fix_dur_path)
}

# ===========================================================================
# 7) NN valence x decision x position (standalone, chance-corrected)
# ===========================================================================
# Tests whether reward valence differentiates gaze allocation across fixation
# positions differently for take vs leave decisions. Chance-corrected:
# delta = observed proportion - (n_items_in_category / 6).
# No random effects due to small number of seeds.

message("\n[7/7] NN valence x decision x position (chance-corrected)")

if (file.exists(nn_fix_dur_path)) {
  nn_fix_data <- readr::read_csv(nn_fix_dur_path, show_col_types = FALSE) %>%
    filter(!is.na(fixation_duration), fixation_duration > 0) %>%
    mutate(
      position_bin = pmin(fixation_count, 7),
      subject_id = as.character(subject_id)
    )

  # NN uses true rewards
  fix_rel <- nn_fix_data %>%
    filter(relevance == 1, !is.na(reward), reward != 0) %>%
    mutate(valence_label = ifelse(reward > 0, "positive", "negative"))

  trial_items <- nn_fix_data %>%
    filter(relevance == 1, !is.na(reward), reward != 0) %>%
    distinct(subject_id, game, trial_number, option, image, reward) %>%
    mutate(valence_label = ifelse(reward > 0, "positive", "negative")) %>%
    group_by(subject_id, game, trial_number, option, valence_label) %>%
    summarise(n_items = n(), .groups = "drop") %>%
    mutate(chance = n_items / 6)

  trial_pos_counts <- fix_rel %>%
    group_by(subject_id, game, trial_number, option, position_bin, valence_label) %>%
    summarise(n_fix = n(), .groups = "drop")

  trial_pos_total <- nn_fix_data %>%
    group_by(subject_id, game, trial_number, option, position_bin) %>%
    summarise(n_total = n(), .groups = "drop")

  grid <- tidyr::expand_grid(
    trial_pos_total %>% select(subject_id, game, trial_number, option, position_bin),
    valence_label = c("positive", "negative")
  ) %>% distinct()

  trial_pos_props <- grid %>%
    left_join(trial_pos_total,
              by = c("subject_id", "game", "trial_number", "option", "position_bin")) %>%
    left_join(trial_pos_counts,
              by = c("subject_id", "game", "trial_number", "option", "position_bin", "valence_label")) %>%
    left_join(trial_items,
              by = c("subject_id", "game", "trial_number", "option", "valence_label")) %>%
    mutate(
      n_fix = tidyr::replace_na(n_fix, 0),
      chance = tidyr::replace_na(chance, 0),
      prop = n_fix / n_total,
      delta = prop - chance
    )

  decision_info <- nn_fix_data %>%
    distinct(subject_id, game, trial_number, option, choice)

  trial_pos_props <- trial_pos_props %>%
    left_join(decision_info,
              by = c("subject_id", "game", "trial_number", "option")) %>%
    filter(!is.na(choice))

  seed_pos <- trial_pos_props %>%
    mutate(decision = ifelse(choice == 1, "take", "leave")) %>%
    group_by(subject_id, position_bin, decision, valence_label) %>%
    summarise(delta = mean(delta, na.rm = TRUE), .groups = "drop")

  d_long <- seed_pos %>%
    mutate(
      dec_c = ifelse(decision == "take", 0.5, -0.5),
      valence_c = ifelse(valence_label == "positive", 0.5, -0.5),
      position = position_bin,
      subject_id = factor(subject_id)
    )

  message("  Observations: ", nrow(d_long), " | Seeds: ", n_distinct(d_long$subject_id))

  m_nn_valence <- brm(
    delta ~ dec_c * valence_c * position,
    data = d_long,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )

  nn_valence_prefix <- paste0("valence_by_position_nn_", tag)
  nn_valence_dir <- file.path(nn_root, "output", "eyegaze", "stats")
  ensure_dir(nn_valence_dir)

  writeLines(capture.output(summary(m_nn_valence)),
             file.path(nn_valence_dir, paste0(nn_valence_prefix, "_summary.txt")))
  tidy_df <- tryCatch(broom.mixed::tidy(m_nn_valence, effects = "fixed"), error = function(e) NULL)
  if (!is.null(tidy_df))
    readr::write_csv(tidy_df, file.path(nn_valence_dir, paste0(nn_valence_prefix, "_fixed.csv")))

  message("  NN valence model saved to ", nn_valence_dir)
} else {
  message("  Skipping: NN fixation file not found at ", nn_fix_dur_path)
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

message("\nAll brms model outputs saved to ", out_dir)
message("Files: accuracy_human_vs_nn_*, prop_relevant_human_vs_nn_*, interaction_human_vs_nn_*, fixcount_interaction_human_vs_nn_*, fixdur_by_position_nn_*, fixprop_by_position_nn_*, valence_by_position_nn_*")
