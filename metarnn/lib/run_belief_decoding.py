#!/usr/bin/env python3
"""Decode belief sufficient statistics from the network's hidden states.

Runs Ridge (linear) and MLP (nonlinear) decoders for mu, lambda, rho,
and V_t at each location, for each seed in parallel. Saves per-seed,
per-location R^2 to a CSV.

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_TAKE_LOGIT_IDX = 6
_LEAVE_LOGIT_IDX = 7


def extract_aligned(data: dict):
    """Extract hidden states aligned with belief states.

    Alignment: hidden[t] has info through action[t-1],
    belief[t] has info through action[t].
    So hidden[t] corresponds to belief[t-1]. Start at t=1.
    """
    all_actions = data["actions"]
    all_hiddens = data["hiddens"]
    all_mus = data["mus"]
    all_lambdas = data["lambdas"]
    all_rhos = data["rhos"]

    hidden_rows, mu_rows, lam_rows, rho_rows = [], [], [], []

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

    return {
        "hiddens": np.array(hidden_rows),
        "mus": np.array(mu_rows),
        "lambdas": np.array(lam_rows),
        "rhos": np.array(rho_rows),
    }


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
    print(f"  [Seed {seed}] {len(H)} observations")

    targets = {
        "mu": ext["mus"],
        "lambda": ext["lambdas"],
        "rho": ext["rhos"],
    }
    V_t = np.sum(ext["mus"] * ext["rhos"], axis=1)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rng = np.random.default_rng(seed)

    rows = []

    def _eval(model_fn, X, y):
        test_r2s = []
        for train_idx, test_idx in kf.split(X):
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            ss_res = np.sum((y[test_idx] - pred) ** 2)
            ss_tot = np.sum((y[test_idx] - y[test_idx].mean()) ** 2)
            test_r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return float(np.mean(test_r2s))

    make_ridge = lambda: Ridge(0.1)
    make_mlp = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64,), max_iter=500,
            random_state=42, early_stopping=True,
            validation_fraction=0.1)),
    ])

    # V_t aggregate
    for label, mf in [("Ridge", make_ridge), ("MLP", make_mlp)]:
        r2 = _eval(mf, H, V_t)
        rows.append({"seed": seed, "statistic": "V_t", "location": "agg",
                      "decoder": label, "r2": r2, "shuffle": False,
                      "shuffle_iter": -1})
    for shuf_i in range(n_shuffles):
        y_shuf = rng.permutation(V_t)
        for shuf_label, shuf_mf in [("Ridge", make_ridge), ("MLP", make_mlp)]:
            r2_shuf = _eval(shuf_mf, H, y_shuf)
            rows.append({"seed": seed, "statistic": "V_t", "location": "agg",
                          "decoder": shuf_label, "r2": r2_shuf,
                          "shuffle": True, "shuffle_iter": shuf_i})

    # Per-location sufficient statistics
    for stat_name, Y in targets.items():
        for loc in range(6):
            y = Y[:, loc]
            if np.std(y) < 1e-8:
                for label in ["Ridge", "MLP"]:
                    rows.append({"seed": seed, "statistic": stat_name,
                                  "location": loc, "decoder": label,
                                  "r2": 0.0, "shuffle": False,
                                  "shuffle_iter": -1})
                continue

            for label, mf in [("Ridge", make_ridge), ("MLP", make_mlp)]:
                r2 = _eval(mf, H, y)
                rows.append({"seed": seed, "statistic": stat_name,
                              "location": loc, "decoder": label,
                              "r2": r2, "shuffle": False,
                              "shuffle_iter": -1})

            for shuf_i in range(n_shuffles):
                y_shuf = rng.permutation(y)
                for shuf_label, shuf_mf in [("Ridge", make_ridge), ("MLP", make_mlp)]:
                    r2_shuf = _eval(shuf_mf, H, y_shuf)
                    rows.append({"seed": seed, "statistic": stat_name,
                                  "location": loc, "decoder": shuf_label,
                                  "r2": r2_shuf, "shuffle": True,
                                  "shuffle_iter": shuf_i})

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
    parser.add_argument("--n-shuffles", type=int, default=10)
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
