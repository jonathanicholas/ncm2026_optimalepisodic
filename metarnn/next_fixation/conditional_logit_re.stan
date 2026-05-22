// Variable-K flat conditional logit with subject-level random slopes for
// all P predictors. Per-event utility uses β_subj = β + u_subj where
// u_subj ~ MVN(0, diag(τ) · L · L' · diag(τ)) (non-centered, Cholesky).

data {
  int<lower=1> N_events;
  int<lower=1> N_total;
  int<lower=1> P;
  int<lower=1> N_subjects;
  array[N_total] vector[P] X;
  array[N_events] int<lower=1, upper=N_total> start_idx;
  array[N_events] int<lower=1, upper=N_total> end_idx;
  array[N_events] int<lower=1>                chose_local;
  array[N_events] int<lower=1, upper=N_subjects> subj;
}

parameters {
  vector[P] beta;                           // population-level fixed effects
  vector<lower=0>[P] tau;                   // RE SDs
  cholesky_factor_corr[P] L_Omega;          // RE correlation Cholesky
  matrix[P, N_subjects] z;                  // unit-normal RE deviates
}

transformed parameters {
  // u[, s] = diag(tau) * L_Omega * z[, s]  (non-centered)
  matrix[P, N_subjects] u = diag_pre_multiply(tau, L_Omega) * z;
}

model {
  beta ~ normal(0, 1.5);
  tau ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(2);
  to_vector(z) ~ std_normal();

  for (e in 1:N_events) {
    int s_idx = start_idx[e];
    int t_idx = end_idx[e];
    int K_e = t_idx - s_idx + 1;
    int sub = subj[e];
    vector[P] beta_s = beta + u[, sub];
    vector[K_e] utilities;
    for (k in 1:K_e) {
      utilities[k] = beta_s' * X[s_idx + k - 1];
    }
    target += utilities[chose_local[e]] - log_sum_exp(utilities);
  }
}

generated quantities {
  // posterior of subject-specific coefficients (β + u)
  matrix[P, N_subjects] beta_subject;
  for (s in 1:N_subjects) {
    beta_subject[, s] = beta + u[, s];
  }
}
