// Variable-K flat conditional logit. Each event has its own number of
// candidates `K_e`; we pack them into a long array indexed by `start_idx`
// and `end_idx` (1-indexed inclusive ranges). The chosen candidate is
// recorded as a *local* index within the event (1..K_e).

data {
  int<lower=1> N_events;
  int<lower=1> N_total;          // total candidate-rows
  int<lower=1> P;
  array[N_total] vector[P] X;
  array[N_events] int<lower=1, upper=N_total> start_idx;
  array[N_events] int<lower=1, upper=N_total> end_idx;
  array[N_events] int<lower=1>                chose_local;  // 1..K_e
}

parameters {
  vector[P] beta;
}

model {
  beta ~ normal(0, 1.5);
  for (e in 1:N_events) {
    int s = start_idx[e];
    int t = end_idx[e];
    int K_e = t - s + 1;
    vector[K_e] utilities;
    for (k in 1:K_e) {
      utilities[k] = beta' * X[s + k - 1];
    }
    target += utilities[chose_local[e]] - log_sum_exp(utilities);
  }
}
