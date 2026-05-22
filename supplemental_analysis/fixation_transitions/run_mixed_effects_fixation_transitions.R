# Bayesian statistical tests for fixation-transition structure (Figure S6).
#
# Tests two families of effects (for humans and NN separately):
#   1) Delta similarity: each template direction vs 0 + pairwise comparisons
#      (bidirectional, forward, backward)
#   2) Sequence length: delta proportion at each bin vs 0 (bins 1-5, 6+)
#
# Usage:
#   Rscript supplemental_analysis/fixation_transitions/run_mixed_effects_fixation_transitions.R \
#     --nn-root metarnn/simulations/human_like_04_04_input5 \
#     --tag 04_04_input5

suppressPackageStartupMessages({
  library(brms)
  library(readr)
  library(dplyr)
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
tag_suffix <- if (nzchar(tag)) paste0("_", tag) else ""

stats_dir <- if (!is.null(cli_args$out_dir)) {
  normalizePath(cli_args$out_dir, mustWork = FALSE)
} else {
  file.path(nn_root, "output", "next_fixation_gen", "stats")
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}
ensure_dir(stats_dir)

# ---------------------------------------------------------------------------
# brms sampling configuration
# ---------------------------------------------------------------------------

brms_chains <- 4
brms_iter   <- 2000
brms_cores  <- min(4, parallel::detectCores())
brms_seed   <- 2026

# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

save_model_outputs <- function(prefix, fit) {
  # Text summary
  summ_txt <- capture.output(summary(fit))
  writeLines(summ_txt, file.path(stats_dir, paste0(prefix, "_summary.txt")))
  # Tidy fixed effects
  tidy_df <- tryCatch(
    broom.mixed::tidy(fit, effects = "fixed"),
    error = function(e) NULL
  )
  if (!is.null(tidy_df)) {
    readr::write_csv(tidy_df, file.path(stats_dir, paste0(prefix, "_summary.csv")))
  }
  tidy_df
}

save_hypothesis <- function(prefix, fit, hypotheses) {
  # Run hypothesis tests and save results
  results <- list()
  for (h in hypotheses) {
    hyp <- hypothesis(fit, h)
    results <- c(results, list(as.data.frame(hyp$hypothesis)))
  }
  hyp_df <- do.call(rbind, results)
  readr::write_csv(hyp_df, file.path(stats_dir, paste0(prefix, "_hypotheses.csv")))
  hyp_df
}

# ===========================================================================
# Load data
# ===========================================================================

message("\n=== Loading CSVs ===")

dsim_path <- file.path(stats_dir, paste0("delta_similarity_subject", tag_suffix, ".csv"))
seqlen_path <- file.path(stats_dir, paste0("seq_length_delta_subject", tag_suffix, ".csv"))

dsim_df <- readr::read_csv(dsim_path, show_col_types = FALSE)
seqlen_df <- readr::read_csv(seqlen_path, show_col_types = FALSE)

message("  Delta similarity:    ", nrow(dsim_df), " rows")
message("  Sequence length:     ", nrow(seqlen_df), " rows")

# Ensure factor types
dsim_df$subject <- as.factor(dsim_df$subject)
dsim_df$group <- as.factor(dsim_df$group)
dsim_df$template <- as.factor(dsim_df$template)
seqlen_df$subject <- as.factor(seqlen_df$subject)
seqlen_df$group <- as.factor(seqlen_df$group)
seqlen_df$seq_length <- factor(seqlen_df$seq_length, levels = c("1", "2", "3", "4", "5", "6+"))

# ===========================================================================
# Test 1: Delta Similarity — vs 0 and pairwise
# ===========================================================================

groups <- c("human", "nn")

message("\n=== Test 1: Delta Similarity ===")

for (grp in groups) {
  model_name <- paste0("dsim_", grp)
  message(sprintf("\n[%s] delta_similarity ~ 0 + template + (1 | subject)", model_name))

  d <- dsim_df %>% filter(group == grp)
  message(sprintf("  N = %d observations from %d subjects", nrow(d), length(unique(d$subject))))

  fit <- brm(
    delta_similarity ~ 0 + template + (1 | subject),
    data = d,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )

  save_model_outputs(model_name, fit)

  # Each direction vs 0 + pairwise differences
  save_hypothesis(model_name, fit, c(
    "templatebidirectional = 0",
    "templateforward = 0",
    "templatebackward = 0",
    "templatebidirectional - templateforward = 0",
    "templatebidirectional - templatebackward = 0",
    "templateforward - templatebackward = 0"
  ))
}

# ===========================================================================
# Test 2: Sequence Length Delta — per-bin vs 0
# ===========================================================================

message("\n=== Test 2: Sequence Length Delta ===")

for (grp in groups) {
  model_name <- paste0("seqlen_", grp)
  message(sprintf("\n[%s] delta_proportion ~ 0 + seq_length + (1 | subject)", model_name))

  d <- seqlen_df %>% filter(group == grp)
  message(sprintf("  N = %d observations from %d subjects", nrow(d), length(unique(d$subject))))

  fit <- brm(
    delta_proportion ~ 0 + seq_length + (1 | subject),
    data = d,
    family = gaussian(),
    chains = brms_chains,
    iter = brms_iter,
    cores = brms_cores,
    seed = brms_seed
  )

  save_model_outputs(model_name, fit)

  # Each bin vs 0
  save_hypothesis(model_name, fit, c(
    "seq_length1 = 0",
    "seq_length2 = 0",
    "seq_length3 = 0",
    "seq_length4 = 0",
    "seq_length5 = 0",
    "seq_length6P = 0"
  ))
}

message("\n=== Done. Results saved to: ", stats_dir, " ===\n")
