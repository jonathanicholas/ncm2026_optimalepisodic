# Bayesian tests comparing human vs NN on panel-level measures
#
# Treats the NN as a fixed benchmark (mean across seeds) and tests whether
# human subject-level values differ from that benchmark. For each measure:
#   1) Compute NN benchmark = mean across seeds (per position where applicable)
#   2) For each human: delta = human_value - nn_benchmark
#   3) Fit intercept-only (simple) or position-level (complex) models on deltas
#
# Panels:
#   Panel A: initial / revisit proportion relevant duration (2 models)
#   Panel B: cumulative fixation-time AUC, 80% crossing (2 models)
#   Panel EF: proportion relevant by fixation position (1 model)
#   Panel GH: valence difference by position, take & leave (2 models)
#
# Expects CSVs exported by metarnn/lib/export_nn_nn_comparison_data.py.
#
# Usage:
#   Rscript metarnn/run_mixed_effects_nn_nn_comparison.R \
#     --data-dir metarnn/simulations/human_like_04_04_input5/output/human_vs_nn_brms/data \
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
# CLI argument parsing (same pattern as run_mixed_effects_human_vs_nn.R)
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

data_dir <- cli_args$data_dir
if (is.null(data_dir)) stop("--data-dir is required")
data_dir <- normalizePath(data_dir, mustWork = TRUE)

tag <- if (!is.null(cli_args$tag)) cli_args$tag else ""

out_dir <- if (!is.null(cli_args$out_dir)) {
  normalizePath(cli_args$out_dir, mustWork = FALSE)
} else {
  file.path(dirname(data_dir), "results")
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
  # Hypothesis test: intercept = 0 (human - nn_benchmark = 0)
  tryCatch({
    h <- hypothesis(fit, "Intercept = 0")
    h_df <- as.data.frame(h$hypothesis)
    readr::write_csv(h_df, file.path(out_dir, paste0(prefix, "_hyp_intercept.csv")))
  }, error = function(e) {
    message("  Could not run hypothesis test: ", conditionMessage(e))
  })
  # Save NN benchmark value(s)
  if (is.data.frame(nn_benchmark)) {
    readr::write_csv(nn_benchmark, file.path(out_dir, paste0(prefix, "_nn_benchmark.csv")))
  } else {
    bench_df <- data.frame(nn_benchmark = nn_benchmark)
    readr::write_csv(bench_df, file.path(out_dir, paste0(prefix, "_nn_benchmark.csv")))
  }
}

# ===========================================================================
# Panel A: initial / revisit proportion relevant duration
# ===========================================================================

message("\n========== Panel A: Initial & Revisit proportion relevant ==========")

panelA_path <- file.path(data_dir, paste0("panelA_subject_data_", tag, ".csv"))
if (file.exists(panelA_path)) {
  d_A <- readr::read_csv(panelA_path, show_col_types = FALSE)

  d_A_human <- d_A %>% filter(group == "human")
  d_A_nn    <- d_A %>% filter(group == "nn")

  nn_bench_firstfix <- mean(d_A_nn$firstfix_prop_time_relevant)
  nn_bench_revisit  <- mean(d_A_nn$revisit_prop_time_relevant)

  message("  Human N: ", nrow(d_A_human), "  |  NN seeds: ", nrow(d_A_nn))
  message("  NN benchmark (initial): ", round(nn_bench_firstfix, 4),
          "  |  NN benchmark (revisit): ", round(nn_bench_revisit, 4))

  # A1: Initial fixation delta
  message("\n[A1] delta_firstfix ~ 1")
  d_A1 <- d_A_human %>%
    mutate(delta = firstfix_prop_time_relevant - nn_bench_firstfix)
  message("  Delta mean: ", round(mean(d_A1$delta), 4))

  m_A1 <- brm(
    delta ~ 1,
    data = d_A1,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )
  save_model_outputs(paste0("panelA_initial_human_vs_nn_", tag), m_A1, nn_bench_firstfix)

  # A2: Revisit fixation delta
  message("\n[A2] delta_revisit ~ 1")
  d_A2 <- d_A_human %>%
    mutate(delta = revisit_prop_time_relevant - nn_bench_revisit)
  message("  Delta mean: ", round(mean(d_A2$delta), 4))

  m_A2 <- brm(
    delta ~ 1,
    data = d_A2,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )
  save_model_outputs(paste0("panelA_revisit_human_vs_nn_", tag), m_A2, nn_bench_revisit)
} else {
  message("  SKIPPED: ", panelA_path, " not found")
}

# ===========================================================================
# Panel B: cumulative fixation-time AUC
# ===========================================================================

message("\n========== Panel B: Cumulative fixation-time AUC ==========")

panelB_path <- file.path(data_dir, paste0("panelB_cumtime_subject_data_", tag, ".csv"))
if (file.exists(panelB_path)) {
  d_B <- readr::read_csv(panelB_path, show_col_types = FALSE)

  d_B_human <- d_B %>% filter(group == "human")
  d_B_nn    <- d_B %>% filter(group == "nn")

  nn_bench_auc <- mean(d_B_nn$auc)
  message("  Human N: ", nrow(d_B_human), "  |  NN seeds: ", nrow(d_B_nn))
  message("  NN benchmark (AUC): ", round(nn_bench_auc, 4))

  # B1: AUC delta
  message("\n[B1] delta_auc ~ 1")
  d_B1 <- d_B_human %>%
    mutate(delta = auc - nn_bench_auc)
  message("  Delta mean: ", round(mean(d_B1$delta), 4))

  m_B1 <- brm(
    delta ~ 1,
    data = d_B1,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )
  save_model_outputs(paste0("panelB_auc_human_vs_nn_", tag), m_B1, nn_bench_auc)

  # B2: 80% crossing delta (optional, may have NAs)
  d_B_human_cross <- d_B_human %>% drop_na(crossing_80)
  d_B_nn_cross    <- d_B_nn %>% drop_na(crossing_80)
  if (nrow(d_B_human_cross) >= 4 && nrow(d_B_nn_cross) >= 1) {
    nn_bench_cross <- mean(d_B_nn_cross$crossing_80)
    message("\n[B2] delta_crossing_80 ~ 1")
    message("  NN benchmark (crossing_80): ", round(nn_bench_cross, 4))

    d_B2 <- d_B_human_cross %>%
      mutate(delta = crossing_80 - nn_bench_cross)
    message("  Delta mean: ", round(mean(d_B2$delta), 4))

    m_B2 <- brm(
      delta ~ 1,
      data = d_B2,
      family = gaussian(),
      chains = brms_chains,
      iter = brms_iter,
      cores = brms_cores,
      seed = brms_seed
    )
    save_model_outputs(paste0("panelB_crossing80_human_vs_nn_", tag), m_B2, nn_bench_cross)
  } else {
    message("  SKIPPED crossing_80: too few observations")
  }
} else {
  message("  SKIPPED: ", panelB_path, " not found")
}

# ===========================================================================
# Panel E/F: proportion relevant — linear model with numeric position
# ===========================================================================
#
# Position is numeric (1–7) so that:
#   Intercept = extrapolated human–NN gap at position 0
#   position  = change in gap per additional fixation step
#
# Uses binned data: positions 1–6 individual, 7+ collapsed as position 7.

message("\n========== Panel E/F: Proportion relevant (numeric position) ==========")

panelEF_path <- file.path(data_dir, paste0("panelEF_prop_relevant_subject_data_", tag, ".csv"))
if (file.exists(panelEF_path)) {
  d_EF <- readr::read_csv(panelEF_path, show_col_types = FALSE)

  d_EF_human <- d_EF %>% filter(group == "human")
  d_EF_nn    <- d_EF %>% filter(group == "nn")

  nn_bench_EF <- d_EF_nn %>%
    group_by(fixation_position) %>%
    summarise(nn_benchmark = mean(prop_relevant), .groups = "drop")

  message("  Human subjects: ", n_distinct(d_EF_human$subject_id),
          "  |  NN seeds: ", n_distinct(d_EF_nn$subject_id))

  d_EF_delta <- d_EF_human %>%
    left_join(nn_bench_EF, by = "fixation_position") %>%
    mutate(
      delta = prop_relevant - nn_benchmark,
      position = fixation_position,
      subject_id = factor(subject_id)
    )

  message("  Positions: ", paste(sort(unique(d_EF_delta$position)), collapse = ", "))
  message("  N obs: ", nrow(d_EF_delta))

  m_EF <- brm(
    delta ~ 1 + position + (1 + position | subject_id),
    data = d_EF_delta,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed,
    control = list(adapt_delta = 0.95)
  )
  save_model_outputs(paste0("panelEF_prop_relevant_continuous_human_vs_nn_", tag), m_EF, nn_bench_EF)
} else {
  message("  SKIPPED: ", panelEF_path, " not found")
}

# ===========================================================================
# Panel G/H: 3-way valence model — linear with numeric position
# ===========================================================================
#
# Combined model across take & leave decisions:
#   delta ~ 1 + dec_c * valence * position
#         + (1 + dec_c + valence + position | subject_id)
#
# Contrast coding:
#   dec_c:   take = +0.5, leave = -0.5
#   valence: positive = +0.5, negative = -0.5
#   position: numeric (1–7)
#
# Uses binned data: positions 1–6 individual, 7+ collapsed as position 7.

message("\n========== Panel G/H: 3-way valence model (numeric position) ==========")

panelGH_path <- file.path(data_dir, paste0("panelGH_valence_diff_subject_data_", tag, ".csv"))
if (file.exists(panelGH_path)) {
  d_GH <- readr::read_csv(panelGH_path, show_col_types = FALSE)

  has_prop_cols <- all(c("prop_positive", "prop_negative") %in% colnames(d_GH))
  if (!has_prop_cols) {
    stop("prop_positive/prop_negative columns not found in GH data")
  }

  d_human <- d_GH %>% filter(group == "human")
  d_nn    <- d_GH %>% filter(group == "nn")

  # NN benchmarks per decision x position x valence
  nn_bench_3way <- d_nn %>%
    group_by(decision, fixation_position) %>%
    summarise(
      nn_benchmark_positive = mean(prop_positive, na.rm = TRUE),
      nn_benchmark_negative = mean(prop_negative, na.rm = TRUE),
      .groups = "drop"
    )

  # Stack human data long with valence factor
  d_long_3way <- bind_rows(
    d_human %>%
      transmute(subject_id, fixation_position, decision,
                value = prop_positive, valence_label = "positive"),
    d_human %>%
      transmute(subject_id, fixation_position, decision,
                value = prop_negative, valence_label = "negative")
  )

  nn_long_3way <- bind_rows(
    nn_bench_3way %>%
      transmute(decision, fixation_position, valence_label = "positive",
                nn_benchmark = nn_benchmark_positive),
    nn_bench_3way %>%
      transmute(decision, fixation_position, valence_label = "negative",
                nn_benchmark = nn_benchmark_negative)
  )

  d_long_3way <- d_long_3way %>%
    left_join(nn_long_3way, by = c("decision", "fixation_position", "valence_label")) %>%
    mutate(
      delta = value - nn_benchmark,
      dec_c = ifelse(decision == "take", 0.5, -0.5),
      valence = ifelse(valence_label == "positive", 0.5, -0.5),
      position = fixation_position,
      subject_id = factor(subject_id)
    )

  message("  Positions: ", paste(sort(unique(d_long_3way$position)), collapse = ", "))
  message("  Observations: ", nrow(d_long_3way),
          "  |  Subjects: ", n_distinct(d_long_3way$subject_id))

  m_3way <- brm(
    delta ~ 1 + dec_c * valence * position +
      (1 + dec_c + valence + position | subject_id),
    data = d_long_3way,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed,
    control = list(adapt_delta = 0.95)
  )

  prefix_3way <- paste0("panelGH_valence_prop_continuous_3way_human_vs_nn_", tag)
  save_model_outputs(prefix_3way, m_3way, nn_bench_3way)
} else {
  message("  SKIPPED: ", panelGH_path, " not found")
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

message("\nAll brms model outputs saved to ", out_dir)
