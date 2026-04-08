# Bayesian comparison of aDDM vs DDM held-out log-likelihoods using brms
#
# Reads the per-fold wide CSV produced by compare_cv_fits.py,
# computes the fold-level delta (aDDM - DDM), and fits a simple
# intercept-only Gaussian model:
#
#   delta_k ~ 1
#
# A credibly non-zero intercept indicates that the aDDM provides
# reliably better (or worse) held-out predictions than the DDM.
#
# Usage (from the repository root):
#   Rscript addm/run_mixed_effects_addm_comparison.R \
#     --wide-csv output/addm/kfold_compare/fix_recalled_final/cv_compare_by_game_wide.csv \
#     --out-dir  output/addm/kfold_compare/fix_recalled_final

suppressPackageStartupMessages({
  library(brms)
  library(readr)
  library(dplyr)
  library(broom.mixed)
})

# ── Parse command-line arguments ─────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 1 && idx < length(args)) return(args[idx + 1])
  if (!is.null(default)) return(default)
  stop(paste("Required argument missing:", flag))
}

wide_csv_path <- parse_arg("--wide-csv")
out_dir       <- parse_arg("--out-dir")

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ── brms sampling configuration ──────
brms_chains <- 4
brms_iter   <- 2000
brms_cores  <- min(4, parallel::detectCores())
brms_seed   <- 2026

# ── Helper: save brms outputs ──────────
save_model_outputs <- function(name, fit, save_dir) {
  # Text summary
  summ_txt <- capture.output(summary(fit))
  writeLines(summ_txt, file.path(save_dir, paste0(name, "_summary.txt")))

  # Tidy fixed-effects table
  tidy_df <- tryCatch(
    broom.mixed::tidy(fit, effects = "fixed"),
    error = function(e) NULL
  )
  if (!is.null(tidy_df)) {
    readr::write_csv(tidy_df, file.path(save_dir, paste0(name, "_fixed.csv")))
  }
}

# ── Load wide CSV and compute deltas ─────────────────────────────────────────
wide <- readr::read_csv(wide_csv_path, show_col_types = FALSE)

# Identify model columns (everything except heldout_game)
model_cols <- setdiff(names(wide), "heldout_game")
if (length(model_cols) != 2) {
  stop(paste0(
    "Expected exactly 2 model columns in wide CSV, found ",
    length(model_cols), ": ", paste(model_cols, collapse = ", ")
  ))
}

# Identify aDDM (free3 / free) vs DDM (theta1)
addm_col <- model_cols[grepl("free", model_cols, ignore.case = TRUE)]
ddm_col  <- model_cols[grepl("theta1", model_cols, ignore.case = TRUE)]

if (length(addm_col) != 1 || length(ddm_col) != 1) {
  stop(paste0(
    "Could not identify aDDM and DDM columns. Found model columns: ",
    paste(model_cols, collapse = ", ")
  ))
}

message("aDDM column: ", addm_col)
message("DDM column:  ", ddm_col)

d <- wide %>%
  mutate(delta = .data[[addm_col]] - .data[[ddm_col]]) %>%
  select(heldout_game, delta)

message("Number of folds: ", nrow(d))
message("Mean delta (aDDM - DDM): ", round(mean(d$delta), 3))

# ── Fit brms model: delta ~ 1 ───────────────────────────────────────────────
m <- brm(
  delta ~ 1,
  data   = d,
  family = gaussian(),
  chains = brms_chains,
  iter   = brms_iter,
  cores  = brms_cores,
  seed   = brms_seed
)

save_model_outputs("addm_vs_ddm_delta_brms", m, out_dir)

# ── Also save the per-fold delta data for reference ──────────────────────────
readr::write_csv(d, file.path(out_dir, "addm_vs_ddm_delta_per_fold.csv"))

message("Outputs saved to ", out_dir)
