#!/usr/bin/env python3
"""Decode belief sufficient statistics from the network's hidden states.

Trains an ordinary least squares (OLS) linear decoder for mu, lambda, rho,
and V_t at each location, for each seed in parallel. Cross-validation uses
GroupKFold with the trial as the group, so timesteps from a given trial
never appear in both train and test. Saves per-seed, per-location R^2 to a
CSV; chance is estimated from 1000 shuffled-target permutations.

Also decodes the per-location relevance-weighted reward (mu_i * rho_i) and
per-location mu_i restricted to rows where rho_i is above or below 0.5,
which together characterize how the network's representation of mu depends
on the relevance gate.

Skips computation if the output CSV already exists (use --force to override).

Example:
  conda run -n analysis python metarnn/lib/run_belief_decoding.py \\
    --data-dir metarnn/simulations/simulation_04_04_input0/with_hidden \\
    --out-dir metarnn/simulations/human_like_04_04_input0/output/evidence \\
    --prefix data_0 --seeds 5 6 7 8 9 --workers 5
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_TAKE_LOGIT_IDX = 6
_LEAVE_LOGIT_IDX = 7


def extract_aligned(data: dict):
    """Extract hidden states aligned with belief states.

    Alignment: hidden[t] has info through action[t-1],
    belief[t] has info through action[t].
    So hidden[t] corresponds to belief[t-1]. Start at t=1.

    Also returns a parallel `groups` array with the trial index for each
    row, used as the GroupKFold group label so timesteps from a given
    trial never split across train/test.
    """
    all_actions = data["actions"]
    all_hiddens = data["hiddens"]
    all_mus = data["mus"]
    all_lambdas = data["lambdas"]
    all_rhos = data["rhos"]

    hidden_rows, mu_rows, lam_rows, rho_rows, group_rows = [], [], [], [], []

    for trial_idx in range(len(all_actions)):
        actions = all_actions[trial_idx]
        hiddens = all_hiddens[trial_idx]
        mus = all_mus[trial_idx]
        lambdas = all_lambdas[trial_idx]
        rhos = all_rhos[trial_idx]

        terminal = None
        for s, a in enumerate(actions):
            if a >= 6:
                terminal = s
                break
        if terminal is None or terminal < 2:
            continue

        for t in range(1, terminal):
            if actions[t] < 0 or actions[t] > 5:
                continue
            hidden_rows.append(hiddens[t])
            mu_rows.append(mus[t - 1])
            lam_rows.append(lambdas[t - 1])
            rho_rows.append(rhos[t - 1])
            group_rows.append(trial_idx)

    return {
        "hiddens": np.array(hidden_rows),
        "mus": np.array(mu_rows),
        "lambdas": np.array(lam_rows),
        "rhos": np.array(rho_rows),
        "groups": np.array(group_rows, dtype=int),
    }


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def cv_ols(X, y, groups, n_folds, drop_below=-1.0):
    """OLS GroupKFold R^2 without shuffles.

    Folds with R^2 < `drop_below` are discarded and the remaining folds are
    averaged. This protects against pathological folds where the test set
    happens to have near-constant labels and ss_tot is tiny, which would
    make R^2 blow up to large negative values.
    """
    gkf = GroupKFold(n_splits=n_folds)
    r2s = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        P = np.linalg.pinv(X[train_idx])
        beta = P @ y[train_idx]
        r2 = _r2(y[test_idx], X[test_idx] @ beta)
        if r2 > drop_below:
            r2s.append(r2)
    if not r2s:
        return float("nan")
    return float(np.mean(r2s))


def cv_ols_with_shuffles(X, y, groups, n_folds, n_shuffles, rng):
    """OLS cross-validation with shuffled-target chance baselines.

    Computes per-fold pseudoinverse pinv(X_train) once and reuses it for
    the real fit and all n_shuffles shuffled fits. Each shuffle becomes
    two cheap matrix-vector products, so 1000 shuffles is fast.

    Returns (real_r2, [shuf_r2_iter_0, ..., shuf_r2_iter_{n_shuffles-1}]).
    Each value is the mean R^2 across folds.
    """
    gkf = GroupKFold(n_splits=n_folds)
    real_r2s = []
    shuf_r2s = [[] for _ in range(n_shuffles)]

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        # Pseudoinverse of X_train: 100 x n_train. One SVD per fold.
        P = np.linalg.pinv(X_tr)

        # Real
        beta = P @ y[train_idx]
        pred = X_te @ beta
        real_r2s.append(_r2(y[test_idx], pred))

        # Shuffled (full-array permute, then re-index)
        for s in range(n_shuffles):
            y_shuf = rng.permutation(y)
            beta = P @ y_shuf[train_idx]
            pred = X_te @ beta
            shuf_r2s[s].append(_r2(y_shuf[test_idx], pred))

    return float(np.mean(real_r2s)), [float(np.mean(s)) for s in shuf_r2s]


def decode_one_seed(args_tuple):
    """Run decoding for a single seed. Designed for ProcessPoolExecutor."""
    data_dir, prefix, seed, n_folds, n_shuffles = args_tuple

    path = os.path.join(data_dir, f"{prefix}_{seed}.json")
    print(f"  [Seed {seed}] Loading {path} ...")
    with open(path, "r") as f:
        data = json.load(f)

    print(f"  [Seed {seed}] Extracting aligned data ...")
    ext = extract_aligned(data)
    H = ext["hiddens"]
    groups = ext["groups"]
    print(f"  [Seed {seed}] {len(H)} observations")

    mus = ext["mus"]
    lambdas = ext["lambdas"]
    rhos = ext["rhos"]
    targets = {
        "mu": mus,
        "lambda": lambdas,
        "rho": rhos,
    }
    V_t = np.sum(mus * rhos, axis=1)

    rng = np.random.default_rng(seed)

    rows = []

    def _record(statistic, location, real_r2, shuf_r2_list):
        rows.append({"seed": seed, "statistic": statistic, "location": location,
                     "decoder": "OLS", "r2": real_r2, "shuffle": False,
                     "shuffle_iter": -1})
        for shuf_i, r2 in enumerate(shuf_r2_list):
            rows.append({"seed": seed, "statistic": statistic,
                         "location": location, "decoder": "OLS", "r2": r2,
                         "shuffle": True, "shuffle_iter": shuf_i})

    def _record_real(statistic, location, r2):
        rows.append({"seed": seed, "statistic": statistic, "location": location,
                     "decoder": "OLS", "r2": r2, "shuffle": False,
                     "shuffle_iter": -1})

    # V_t aggregate
    real_r2, shuf_r2s = cv_ols_with_shuffles(H, V_t, groups, n_folds, n_shuffles, rng)
    _record("V_t", "agg", real_r2, shuf_r2s)

    # Per-location sufficient statistics
    for stat_name, Y in targets.items():
        for loc in range(6):
            y = Y[:, loc]
            if np.std(y) < 1e-8:
                _record(stat_name, loc, 0.0, [0.0] * n_shuffles)
                continue
            real_r2, shuf_r2s = cv_ols_with_shuffles(H, y, groups, n_folds,
                                                      n_shuffles, rng)
            _record(stat_name, loc, real_r2, shuf_r2s)

    # Per-location relevance-weighted reward, and per-location mu split by
    # whether the relevance posterior at that slot is above or below 0.5.
    for loc in range(6):
        mu_i = mus[:, loc]
        rho_i = rhos[:, loc]
        prod_i = mu_i * rho_i

        if np.std(prod_i) > 1e-8:
            _record_real("mu*rho", loc,
                          cv_ols(H, prod_i, groups, n_folds))

        for label, mask in [("mu_when_rho_above_half", rho_i >= 0.5),
                             ("mu_when_rho_below_half", rho_i < 0.5)]:
            n_mask = int(mask.sum())
            if n_mask < 200 or len(np.unique(groups[mask])) < n_folds:
                continue
            mu_sub = mu_i[mask]
            if np.std(mu_sub) < 1e-8:
                continue
            _record_real(label, loc,
                          cv_ols(H[mask], mu_sub, groups[mask], n_folds))

    print(f"  [Seed {seed}] Done.")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Decode belief sufficient statistics from hidden states")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="data_0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output CSV already exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "belief_decoding_results.csv")
    if os.path.isfile(csv_path) and not args.force:
        print(f"Output already exists: {csv_path}")
        print("Use --force to re-run. Skipping.")
        return

    task_args = [(args.data_dir, args.prefix, seed, args.n_folds, args.n_shuffles)
                 for seed in args.seeds]

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(decode_one_seed, ta): ta[2] for ta in task_args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [Seed {seed}] FAILED: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
