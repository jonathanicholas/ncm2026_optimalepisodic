# Hierarchical conditional logit of next-fixation choice (human participants).
#
# Each next-fixation event is a categorical choice over the working-set
# candidates. Ten z-scored candidate-level predictors, with subject-level
# random slopes on all ten:
#   share_k_z          inverse cumulative relevant-fixation time
#   dist_cw_z          clockwise spatial distance
#   dist_ccw_z         counter-clockwise spatial distance
#   enc_lag_fwd_z      forward encoding-order distance
#   enc_lag_bwd_z      backward encoding-order distance
#   is_primacy_k_z     first-encoded item
#   is_recency_k_z     last-encoded item
#   abs_reward_k_z     reward magnitude
#   signed_reward_k_z  signed reward
#   is_prev_fixation_k_z   just-prior item
#
# Usage (from repo root):
#   Rscript metarnn/next_fixation/fit_conditional_logit_re.R

suppressPackageStartupMessages({
  library(rstan)
  library(readr)
  library(dplyr)
  library(posterior)
})

rstan_options(auto_write = TRUE)
options(mc.cores = 4)

out_dir   <- file.path(getwd(), "output", "next_fixation")
stan_file <- file.path(getwd(), "metarnn", "next_fixation", "conditional_logit_re.stan")
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

d <- readr::read_csv(file.path(out_dir, "next_fixation_long_human.csv"),
                     show_col_types = FALSE)

# share_k_z: decay-weighted relevant-fixated weight, z-scored. decay shrinks
# with cumulative fixation time, so larger share_k = more remaining uncertainty.
mean_ct <- mean(d$cum_time_k, na.rm = TRUE)
if (!is.finite(mean_ct) || mean_ct == 0) mean_ct <- 1
decay <- 1 / (1 + d$cum_time_k / mean_ct)
d$share_k_z <- zscore(decay * d$is_relevant_k * d$is_fixated_k)

# Spatial distance split into clockwise / counter-clockwise components. The
# opposite item (d_mod = 3) is equidistant either way and is counted in both.
d_mod <- (d$candidate_slot - d$current_slot) %% 6
d$dist_cw_z  <- zscore(ifelse(d_mod <= 3, d_mod, 0))
d$dist_ccw_z <- zscore(ifelse(d_mod >= 3, 6 - d_mod, 0))

# Encoding-order distance split into forward / backward components.
d$enc_lag_fwd_z <- zscore(pmax(0,  d$signed_enc_lag_ik))
d$enc_lag_bwd_z <- zscore(pmax(0, -d$signed_enc_lag_ik))

d$abs_reward_k_z       <- zscore(d$abs_reward_k)
d$signed_reward_k_z    <- zscore(d$signed_reward_k)
d$is_primacy_k_z       <- zscore(d$is_primacy_k)
d$is_recency_k_z       <- zscore(d$is_recency_k)
d$is_prev_fixation_k_z <- zscore(d$is_prev_fixation_k)

# Restrict to the working set; keep events with exactly one chosen candidate
# and at least two candidates.
d <- d %>% filter(is_fixated_k == 1) %>% arrange(event_id, candidate_slot)
ev_groups <- d %>% group_by(event_id) %>% group_split()
ev_groups <- ev_groups[vapply(ev_groups, function(rows) {
  length(which(rows$chose_k == 1)) == 1 && nrow(rows) >= 2
}, logical(1))]
N_events <- length(ev_groups)
message("  Kept ", N_events, " events")

subjects_all <- sort(unique(d$subject))
subj_to_idx  <- setNames(seq_along(subjects_all), as.character(subjects_all))

start_idx   <- integer(N_events)
end_idx     <- integer(N_events)
chose_local <- integer(N_events)
subj        <- integer(N_events)
X_rows      <- vector("list", N_events)
cursor <- 1L
for (e in seq_along(ev_groups)) {
  rows <- ev_groups[[e]]
  K_e <- nrow(rows)
  start_idx[e]   <- cursor
  end_idx[e]     <- cursor + K_e - 1L
  chose_local[e] <- which(rows$chose_k == 1)
  subj[e]        <- subj_to_idx[as.character(rows$subject[1])]
  X_rows[[e]]    <- as.matrix(rows[, predictor_cols])
  cursor <- cursor + K_e
}
X_mat <- do.call(rbind, X_rows)

stan_data <- list(N_events = N_events,
                  N_total = nrow(X_mat),
                  P = length(predictor_cols),
                  N_subjects = length(subjects_all),
                  X = X_mat,
                  start_idx = start_idx,
                  end_idx = end_idx,
                  chose_local = chose_local,
                  subj = subj)

fit <- sampling(stan_model(stan_file), data = stan_data,
                chains = chains, iter = iter, warmup = warmup,
                control = list(adapt_delta = 0.95, max_treedepth = 12),
                seed = 2026, init_r = 0.5)

beta_summary <- posterior::summarise_draws(fit,
  "mean", "sd", "median",
  ~quantile(.x, c(0.025, 0.5, 0.975)),
  "rhat", "ess_bulk")

beta_pop <- beta_summary[grepl("^beta\\[", beta_summary$variable), ]
beta_pop$predictor <- predictor_cols
write_csv(beta_pop,
          file.path(out_dir, "conditional_logit_human_population_beta.csv"))

beta_pop_draws <- as.matrix(rstan::extract(fit, pars = "beta")$beta)
colnames(beta_pop_draws) <- predictor_cols
write_csv(as.data.frame(beta_pop_draws),
          file.path(out_dir, "conditional_logit_human_population_beta_draws.csv"))

# Per-subject coefficients (beta + u_s).
arr <- rstan::extract(fit, pars = "beta_subject", permute = TRUE)$beta_subject
rows <- list()
for (s in seq_along(subjects_all)) {
  for (p in seq_along(predictor_cols)) {
    x <- arr[, p, s]
    rows[[length(rows) + 1]] <- data.frame(
      subject = as.character(subjects_all[s]),
      predictor = predictor_cols[p],
      mean = mean(x), sd = sd(x),
      lo = quantile(x, 0.025), hi = quantile(x, 0.975))
  }
}
write_csv(do.call(rbind, rows),
          file.path(out_dir, "conditional_logit_human_per_subject_beta.csv"))

saveRDS(fit, file.path(out_dir, "conditional_logit_human_fit.rds"))
message("Saved hierarchical conditional-logit outputs for humans")
