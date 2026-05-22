# Conditional logit of next-fixation choice (fixed effects only).
#
# Used for the prior-memory network and the null oracles, which have no
# subject structure to pool over. Ten z-scored candidate-level predictors
# (see fit_conditional_logit_re.R for the predictor definitions).
#
# Usage (from repo root):
#   Rscript metarnn/next_fixation/fit_conditional_logit.R <dataset>
# where <dataset> is one of: rnn_input5_500k, walk_ring_noisy_10x, random_10x

suppressPackageStartupMessages({
  library(rstan)
  library(readr)
  library(dplyr)
  library(posterior)
})

rstan_options(auto_write = TRUE)
options(mc.cores = 4)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript metarnn/next_fixation/fit_conditional_logit.R <dataset>")
}
dataset <- args[1]

out_dir   <- file.path(getwd(), "output", "next_fixation")
stan_file <- file.path(getwd(), "metarnn", "next_fixation", "conditional_logit.stan")
chains <- 4
iter   <- 2000
warmup <- 1000

zscore <- function(x) {
  v <- as.numeric(x)
  s <- sd(v, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(v)))
  (v - mean(v, na.rm = TRUE)) / s
}

predictor_cols <- c("share_k_z",
                    "dist_cw_z", "dist_ccw_z",
                    "enc_lag_fwd_z", "enc_lag_bwd_z",
                    "is_primacy_k_z", "is_recency_k_z",
                    "abs_reward_k_z", "signed_reward_k_z",
                    "is_prev_fixation_k_z")

csv_path <- file.path(out_dir, paste0("next_fixation_long_", dataset, ".csv"))
if (!file.exists(csv_path)) stop(csv_path, " not found")
d <- readr::read_csv(csv_path, show_col_types = FALSE)

# share_k_z: decay-weighted relevant-fixated weight, z-scored.
mean_ct <- mean(d$cum_time_k, na.rm = TRUE)
if (!is.finite(mean_ct) || mean_ct == 0) mean_ct <- 1
decay <- 1 / (1 + d$cum_time_k / mean_ct)
d$share_k_z <- zscore(decay * d$is_relevant_k * d$is_fixated_k)

# Spatial distance split into clockwise / counter-clockwise; opposite item
# (d_mod = 3) is equidistant either way and counted in both.
d_mod <- (d$candidate_slot - d$current_slot) %% 6
d$dist_cw_z  <- zscore(ifelse(d_mod <= 3, d_mod, 0))
d$dist_ccw_z <- zscore(ifelse(d_mod >= 3, 6 - d_mod, 0))

# Encoding-order distance split into forward / backward.
d$enc_lag_fwd_z <- zscore(pmax(0,  d$signed_enc_lag_ik))
d$enc_lag_bwd_z <- zscore(pmax(0, -d$signed_enc_lag_ik))

d$abs_reward_k_z       <- zscore(d$abs_reward_k)
d$signed_reward_k_z    <- zscore(d$signed_reward_k)
d$is_primacy_k_z       <- zscore(d$is_primacy_k)
d$is_recency_k_z       <- zscore(d$is_recency_k)
d$is_prev_fixation_k_z <- zscore(d$is_prev_fixation_k)

d <- d %>% filter(is_fixated_k == 1) %>% arrange(event_id, candidate_slot)
ev_groups <- d %>% group_by(event_id) %>% group_split()
ev_groups <- ev_groups[vapply(ev_groups, function(rows) {
  length(which(rows$chose_k == 1)) == 1 && nrow(rows) >= 2
}, logical(1))]
N_events <- length(ev_groups)
message("  Kept ", N_events, " events")

start_idx   <- integer(N_events)
end_idx     <- integer(N_events)
chose_local <- integer(N_events)
X_rows      <- vector("list", N_events)
cursor <- 1L
for (e in seq_along(ev_groups)) {
  rows <- ev_groups[[e]]
  K_e <- nrow(rows)
  start_idx[e]   <- cursor
  end_idx[e]     <- cursor + K_e - 1L
  chose_local[e] <- which(rows$chose_k == 1)
  X_rows[[e]]    <- as.matrix(rows[, predictor_cols])
  cursor <- cursor + K_e
}
X_mat <- do.call(rbind, X_rows)

stan_data <- list(N_events = N_events,
                  N_total = nrow(X_mat),
                  P = length(predictor_cols),
                  X = X_mat,
                  start_idx = start_idx,
                  end_idx = end_idx,
                  chose_local = chose_local)

fit <- sampling(stan_model(stan_file), data = stan_data,
                chains = chains, iter = iter, warmup = warmup,
                control = list(adapt_delta = 0.92, max_treedepth = 12),
                seed = 2026)

beta_summary <- posterior::summarise_draws(fit,
  "mean", "sd", "median",
  ~quantile(.x, c(0.025, 0.5, 0.975)),
  "rhat", "ess_bulk")
beta_summary <- beta_summary[grepl("^beta\\[", beta_summary$variable), ]
beta_summary$predictor <- predictor_cols
beta_summary$dataset <- dataset
write_csv(beta_summary,
          file.path(out_dir, paste0("conditional_logit_", dataset, "_beta.csv")))

beta_draws <- as.matrix(rstan::extract(fit, pars = "beta")$beta)
colnames(beta_draws) <- predictor_cols
write_csv(as.data.frame(beta_draws),
          file.path(out_dir, paste0("conditional_logit_", dataset, "_beta_draws.csv")))

saveRDS(fit, file.path(out_dir, paste0("conditional_logit_", dataset, "_fit.rds")))
message("Saved conditional-logit outputs for ", dataset)
