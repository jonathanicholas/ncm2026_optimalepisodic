#!/usr/bin/env python3
"""Evidence accumulation and belief decoding figure (Figure 3, panels B-C).

Produces a three-panel figure:
  Top row: Network DV and Metalevel MDP V_t over fixation steps
  Bottom row: MLP decoding R² for belief sufficient statistics

Supports two modes:
  - From raw JSON files: preprocess, plot, and optionally save cache
  - From cached data: load preprocessed CSV and plot (no JSON needed)

Example (from JSON):
  conda run -n analysis python metarnn/lib/plot_evidence_figure.py \\
    --data-dir metarnn/simulations/simulation_04_04_input0/with_hidden \\
    --out-dir metarnn/simulations/human_like_04_04_input0/output/evidence \\
    --decoding-csv metarnn/simulations/human_like_04_04_input0/output/evidence/belief_decoding_results.csv \\
    --save-cache

Example (from cache):
  conda run -n analysis python metarnn/lib/plot_evidence_figure.py \\
    --cache-dir metarnn/simulations/human_like_04_04_input0/output/evidence \\
    --out-dir metarnn/simulations/human_like_04_04_input0/output/evidence \\
    --decoding-csv metarnn/simulations/human_like_04_04_input0/output/evidence/belief_decoding_results.csv
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

_TAKE_COLOR = "#7f7e95"
_LEAVE_COLOR = "#272743"
_SEM_ALPHA = 0.35
_BAR_LINEWIDTH = 2.5
_ERRORBAR_LINEWIDTH = 2.5
_PANEL_LABEL_SIZE = 26
_AXIS_LABEL_SIZE = 28
_TICK_LABEL_SIZE = 24

_TAKE_LOGIT_IDX = 6
_LEAVE_LOGIT_IDX = 7

_CACHE_FILENAME = "evidence_figure_cache.csv"


def load_seed(data_dir: str, prefix: str, seed: int) -> dict:
    path = os.path.join(data_dir, f"{prefix}_{seed}.json")
    print(f"  Loading {path} ...")
    with open(path, "r") as f:
        return json.load(f)


def preprocess_trials(data: dict) -> pd.DataFrame:
    """Extract DV, V_t, and delta_DV per fixation step, with 7+ binning."""
    all_actions = data["actions"]
    all_logits = data["logits"]
    all_mus = data["mus"]
    all_rhos = data["rhos"]
    values = np.array(data["values"])
    relevances = np.array(data["relevances"])

    rows = []
    for trial_idx in range(len(all_actions)):
        trial_actions = all_actions[trial_idx]
        trial_logits = all_logits[trial_idx]
        trial_mus = all_mus[trial_idx]
        trial_rhos = all_rhos[trial_idx]
        n_steps = len(trial_actions)

        terminal_step = None
        for s in range(n_steps):
            if trial_actions[s] >= 6:
                terminal_step = s
                break
        if terminal_step is None or terminal_step == 0:
            continue
        decision = int(trial_actions[terminal_step])

        dv_baseline = float(trial_logits[0][_TAKE_LOGIT_IDX] -
                            trial_logits[0][_LEAVE_LOGIT_IDX])

        for t in range(terminal_step):
            action_t = int(trial_actions[t])
            if action_t < 0 or action_t > 5:
                continue

            dv_t = float(trial_logits[t][_TAKE_LOGIT_IDX] -
                         trial_logits[t][_LEAVE_LOGIT_IDX]) - dv_baseline

            if t == 0:
                v_t = 0.0
            else:
                mu_aligned = np.array(trial_mus[t - 1])
                rho_aligned = np.array(trial_rhos[t - 1])
                v_t = float(np.sum(mu_aligned * rho_aligned))

            dv_raw = dv_t + dv_baseline
            if t + 1 < terminal_step:
                dv_next = float(trial_logits[t + 1][_TAKE_LOGIT_IDX] -
                                trial_logits[t + 1][_LEAVE_LOGIT_IDX])
            else:
                dv_next = float(trial_logits[terminal_step][_TAKE_LOGIT_IDX] -
                                trial_logits[terminal_step][_LEAVE_LOGIT_IDX])
            delta_dv = dv_next - dv_raw

            is_relevant = bool(relevances[trial_idx, action_t])
            reward_val = float(values[trial_idx, action_t])
            if is_relevant and reward_val > 0:
                sample_type = "Relevant\n(+)"
            elif is_relevant and reward_val < 0:
                sample_type = "Relevant\n(–)"
            else:
                sample_type = "Irrelevant"

            step_bin = t if t <= 6 else "7+"

            rows.append({
                "trial": trial_idx,
                "fixation_step": t,
                "step_bin": step_bin,
                "decision": "Take" if decision == 6 else "Leave",
                "dv": dv_t,
                "v_belief": v_t,
                "delta_dv": delta_dv,
                "sample_type": sample_type,
            })

    return pd.DataFrame(rows)


def _plot_accumulation(ax, all_dfs, col, ylabel, yticks=None):
    """Plot mean +/- SEM of col at each step bin, by decision."""
    bin_order = [0, 1, 2, 3, 4, 5, 6, "7+"]
    x_pos = np.arange(len(bin_order))

    for decision, color in [("Take", _TAKE_COLOR), ("Leave", _LEAVE_COLOR)]:
        seed_means_per_bin = {b: [] for b in bin_order}
        for df in all_dfs:
            sub = df[df["decision"] == decision]
            for b in bin_order:
                vals = sub[sub["step_bin"] == b][col]
                seed_means_per_bin[b].append(vals.mean() if len(vals) > 0 else np.nan)

        grand_mean, grand_sem = [], []
        for b in bin_order:
            vals = [v for v in seed_means_per_bin[b] if not np.isnan(v)]
            if vals:
                grand_mean.append(np.mean(vals))
                grand_sem.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                                 if len(vals) > 1 else 0)
            else:
                grand_mean.append(np.nan)
                grand_sem.append(np.nan)

        gm = np.array(grand_mean)
        gs = np.array(grand_sem)
        valid = ~np.isnan(gm)

        ax.plot(x_pos[valid], gm[valid], color=color, linewidth=3.5,
                label=decision)
        ax.fill_between(x_pos[valid], gm[valid] - gs[valid],
                        gm[valid] + gs[valid], color=color, alpha=_SEM_ALPHA,
                        linewidth=0)

    ax.axhline(0, color="black", linewidth=2, linestyle="-", alpha=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["0", "", "", "", "", "", "", "7+"])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[-1])
    ax.legend(loc="lower left", fontsize=14, frameon=True,
              edgecolor="black", fancybox=False, framealpha=1.0,
              handlelength=1.0, handletextpad=0.4, labelspacing=0.2,
              borderpad=0.3, bbox_to_anchor=(0.0, 0.0))
    sns.despine(ax=ax)


def plot_decoding_bars(ax, csv_path):
    """Plot OLS decoding R^2 from the saved CSV."""
    df = pd.read_csv(csv_path)

    real = df[(df["decoder"] == "OLS") & (~df["shuffle"])]
    shuf = df[(df["decoder"] == "OLS") & (df["shuffle"])]

    stats_order = ["V_t", "mu", "lambda", "rho"]
    labels = ["$V_t$", "$\\mu_t$", "$\\lambda_t$", "$\\rho_t$"]
    x_pos = np.arange(len(stats_order))
    rng = np.random.default_rng(99)

    chance_vals = []
    for i, (stat, label) in enumerate(zip(stats_order, labels)):
        sub = real[real["statistic"] == stat]
        seed_vals = sub.groupby("seed")["r2"].mean().values

        m = np.mean(seed_vals)
        se = (np.std(seed_vals, ddof=1) / np.sqrt(len(seed_vals))
              if len(seed_vals) > 1 else 0)

        bottom = -0.05
        bw = 0.5
        ax.bar(x_pos[i], m - bottom, bw, bottom=bottom, color=".7",
               edgecolor="none", linewidth=0, zorder=2)
        ax.bar(x_pos[i], m - bottom, bw, bottom=bottom, color="none",
               edgecolor="black", linewidth=_BAR_LINEWIDTH, zorder=4)
        ax.errorbar(x_pos[i], m, yerr=se, fmt="none", ecolor="black",
                     capsize=0, linewidth=_ERRORBAR_LINEWIDTH, zorder=5)

        jitter = rng.uniform(-0.15, 0.15, size=len(seed_vals))
        ax.scatter(x_pos[i] + jitter, seed_vals, s=6**2,
                   facecolor=(1, 1, 1, 0.5), edgecolor=".7",
                   linewidth=1, zorder=3)

        shuf_sub = shuf[shuf["statistic"] == stat]
        if len(shuf_sub) > 0:
            shuf_vals = shuf_sub.groupby("seed")["r2"].mean().values
            chance_vals.append(np.mean(shuf_vals))

    bw_chance = 0.5
    for i, chance_m in enumerate(chance_vals):
        ax.plot([x_pos[i] - bw_chance / 2, x_pos[i] + bw_chance / 2],
                [chance_m, chance_m],
                color="black", linewidth=2, linestyle=":", zorder=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_yticks([0, 1])
    sns.despine(ax=ax)


def _supplement_per_seed_means(df, target):
    sub = df[df["statistic"] == target]
    if len(sub) == 0:
        return np.array([])
    return sub.groupby("seed")["r2"].mean().values


def _supplement_bar(ax, x, seed_vals, bw=0.5, bottom=-0.05):
    if len(seed_vals) == 0:
        return
    m = float(np.nanmean(seed_vals))
    se = (float(np.nanstd(seed_vals, ddof=1) /
                np.sqrt(np.sum(~np.isnan(seed_vals))))
          if np.sum(~np.isnan(seed_vals)) > 1 else 0.0)
    ax.bar(x, m - bottom, bw, bottom=bottom, color=".7",
           edgecolor="none", linewidth=0, zorder=2)
    ax.bar(x, m - bottom, bw, bottom=bottom, color="none",
           edgecolor="black", linewidth=_BAR_LINEWIDTH, zorder=4)
    ax.errorbar(x, m, yerr=se, fmt="none", ecolor="black",
                capsize=0, linewidth=_ERRORBAR_LINEWIDTH, zorder=5)
    rng = np.random.default_rng(99)
    jitter = rng.uniform(-0.12, 0.12, size=len(seed_vals))
    ax.scatter(np.full(len(seed_vals), x) + jitter, seed_vals, s=36,
               facecolor=(1, 1, 1, 0.5), edgecolor=".7",
               linewidth=1, zorder=3)


def plot_belief_decoding_supplement(csv_path, out_dir):
    """Two-panel supplementary figure characterizing how mu decoding
    depends on the relevance posterior.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["decoder"] == "OLS") & (~df["shuffle"])]

    sns.set_context("talk")
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.5),
        gridspec_kw={"wspace": 0.55, "width_ratios": [3, 2]})

    left_targets = [("rho", r"$\rho_t$"),
                    ("mu", r"$\mu_t$"),
                    ("mu*rho", r"$\mu_t\,\rho_t$")]
    x_left = np.arange(len(left_targets))
    for i, (tgt, _) in enumerate(left_targets):
        _supplement_bar(axes[0], x_left[i],
                         _supplement_per_seed_means(df, tgt))
    axes[0].set_xticks(x_left)
    axes[0].set_xticklabels([lbl for _, lbl in left_targets])
    axes[0].set_ylabel("$R^2$")
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].set_yticks([0, 1])
    sns.despine(ax=axes[0])

    right_targets = [("mu_when_rho_above_half", r"$\rho \geq 0.5$"),
                     ("mu_when_rho_below_half", r"$\rho < 0.5$")]
    x_right = np.arange(len(right_targets))
    for i, (tgt, _) in enumerate(right_targets):
        _supplement_bar(axes[1], x_right[i],
                         _supplement_per_seed_means(df, tgt))
    axes[1].set_xticks(x_right)
    axes[1].set_xticklabels([lbl for _, lbl in right_targets])
    axes[1].set_ylabel(r"$R^2$ ($\mu_t$ decoding)")
    axes[1].set_ylim(-0.05, 1.0)
    axes[1].set_yticks([0, 1])
    sns.despine(ax=axes[1])

    out_path = os.path.join(out_dir, "belief_decoding_supplement.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evidence accumulation figure (Figure 3 panels B-C)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing JSON files with hidden states")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to load cached preprocessed data from")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="data_0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--decoding-csv", required=True,
                        help="Path to belief_decoding_results.csv")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save preprocessed data as CSV for future runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_path = None
    if args.cache_dir:
        cache_path = os.path.join(args.cache_dir, _CACHE_FILENAME)
    elif args.save_cache:
        cache_path = os.path.join(args.out_dir, _CACHE_FILENAME)

    # Load data: from JSON or from cache
    if args.data_dir and os.path.isdir(args.data_dir):
        print("Loading data from JSON files...")
        all_dfs = []
        for seed in args.seeds:
            data = load_seed(args.data_dir, args.prefix, seed)
            df = preprocess_trials(data)
            df["seed"] = seed
            all_dfs.append(df)
            print(f"    Seed {seed}: {len(df)} observations")

        if args.save_cache and cache_path:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(cache_path, index=False)
            print(f"Saved cache: {cache_path}")

    elif cache_path and os.path.isfile(cache_path):
        print(f"Loading from cache: {cache_path}")
        combined = pd.read_csv(cache_path)
        # Restore step_bin types (CSV reads "7+" as string, ints as int)
        combined["step_bin"] = combined["step_bin"].apply(
            lambda x: int(x) if str(x).isdigit() else x)
        all_dfs = [g for _, g in combined.groupby("seed")]
        print(f"    Loaded {len(combined)} observations across {len(all_dfs)} seeds")
    else:
        raise FileNotFoundError(
            "No data available. Provide --data-dir (JSON files) or --cache-dir.")

    # Plot
    print("Plotting...")
    sns.set_context("poster")
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": _AXIS_LABEL_SIZE,
        "xtick.labelsize": _TICK_LABEL_SIZE,
        "ytick.labelsize": _TICK_LABEL_SIZE,
    })

    fig = plt.figure(figsize=(7, 7.75))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.7, wspace=0.35,
                           height_ratios=[1.8, 1])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_accumulation(ax_a, all_dfs, "dv", "", yticks=[-6, 0, 6])
    _plot_accumulation(ax_b, all_dfs, "v_belief", "", yticks=[-3, 0, 3])

    ax_a.set_title("RNN", fontsize=24)
    ax_b.set_title("Metalevel MDP", fontsize=24)

    fig.text(0.5, 0.41, "Fixation number", ha="center", fontsize=_AXIS_LABEL_SIZE)
    fig.text(0.02, 0.70, "Decision variable", va="center", ha="center",
             rotation="vertical", fontsize=_AXIS_LABEL_SIZE)

    plot_decoding_bars(ax_c, args.decoding_csv)

    out_path = os.path.join(args.out_dir, "evidence_figure.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)

    plot_belief_decoding_supplement(args.decoding_csv, args.out_dir)


if __name__ == "__main__":
    main()
