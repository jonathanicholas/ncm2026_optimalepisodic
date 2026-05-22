"""Figure 5: forest plot of the next-fixation conditional logit.

Ten z-scored candidate-level predictors, grouped into three categories shown
as coloured background bands: resource-rational signals predicted by the
optimal sampling policy, encoding-order biases, and heuristic biases. Compares
human participants (hierarchical fixed effects, with per-subject random-effect
dots) against the prior-memory network (fixed-effect estimate).

Usage (from repo root):
  python metarnn/next_fixation/plot_next_fixation_forest.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUT = Path(__file__).resolve().parents[2] / "output" / "next_fixation"

PREDICTOR_ORDER = [
    "share_k_z",
    "dist_cw_z",
    "dist_ccw_z",
    "enc_lag_fwd_z",
    "enc_lag_bwd_z",
    "is_primacy_k_z",
    "is_recency_k_z",
    "abs_reward_k_z",
    "signed_reward_k_z",
    "is_prev_fixation_k_z",
]

LABELS = {
    "share_k_z":            "Inverse Relevant Fix. Time",
    "dist_cw_z":            "Spatial Distance (clockwise)",
    "dist_ccw_z":           "Spatial Distance (counter-cw)",
    "enc_lag_fwd_z":        "Temporal Distance (forward)",
    "enc_lag_bwd_z":        "Temporal Distance (backward)",
    "is_primacy_k_z":       "Primacy",
    "is_recency_k_z":       "Recency",
    "abs_reward_k_z":       "Reward (magnitude)",
    "signed_reward_k_z":    "Reward (signed)",
    "is_prev_fixation_k_z": "Previous Item",
}

# Predictor categories, each drawn as a coloured background band.
GROUPS = [
    ("Resource Rational",
     ["share_k_z", "dist_cw_z", "dist_ccw_z"],
     {"human": "#9575CD", "rnn": "#D1C4E9", "band": "#7E57C2"}),
    ("Encoding Order",
     ["enc_lag_fwd_z", "enc_lag_bwd_z", "is_primacy_k_z", "is_recency_k_z"],
     {"human": "#64B5F6", "rnn": "#BBDEFB", "band": "#42A5F5"}),
    ("Heuristic Biases",
     ["abs_reward_k_z", "signed_reward_k_z", "is_prev_fixation_k_z"],
     {"human": "#FF8A65", "rnn": "#FFCCBC", "band": "#FF7043"}),
]

PRIMARY = {}
for _name, _preds, _colors in GROUPS:
    for _p in _preds:
        if _p not in PRIMARY:
            PRIMARY[_p] = (_name, _colors)


def load_betas(beta_csv):
    df = pd.read_csv(OUT / beta_csv)
    df = df[df["variable"].str.startswith("beta[")].copy()
    return df.set_index("predictor")[["mean", "sd", "2.5%", "97.5%"]]


def main():
    human = load_betas("conditional_logit_human_population_beta.csv")
    rnn_path = OUT / "conditional_logit_rnn_input5_500k_beta.csv"
    rnn = load_betas(rnn_path.name) if rnn_path.exists() else None
    per_subj = pd.read_csv(OUT / "conditional_logit_human_per_subject_beta.csv")

    n = len(PREDICTOR_ORDER)
    y_base = np.arange(n)[::-1]
    y_idx = {p: y_base[i] for i, p in enumerate(PREDICTOR_ORDER)}
    offset = 0.18

    fig, ax = plt.subplots(1, 1, figsize=(6, 6.5))

    series = [("human", human, +offset if rnn is not None else 0.0)]
    if rnn is not None:
        series.append(("rnn", rnn, -offset))

    # Per-subject jitter dots (behind population markers).
    rng = np.random.default_rng(2026)
    for predictor in PREDICTOR_ORDER:
        sub = per_subj[per_subj["predictor"] == predictor]
        y_center = y_idx[predictor] + offset
        jitter = rng.uniform(-0.06, 0.06, size=len(sub))
        ax.scatter(sub["mean"], y_center + jitter,
                   s=24, color=PRIMARY[predictor][1]["human"],
                   alpha=0.35, edgecolor="none", zorder=2)

    # Population fixed-effect markers.
    for predictor in PREDICTOR_ORDER:
        c = PRIMARY[predictor][1]
        for which, df, shift in series:
            mean = df.loc[predictor, "mean"]
            lo = df.loc[predictor, "2.5%"]
            hi = df.loc[predictor, "97.5%"]
            y = y_idx[predictor] + shift
            ax.errorbar(mean, y,
                        xerr=[[mean - lo], [hi - mean]],
                        fmt="o", color=c[which], markersize=9,
                        markeredgecolor="black", markeredgewidth=1.5,
                        ecolor="black", elinewidth=1.5,
                        capsize=0, zorder=3)

    ax.axvline(0, ls=":", color="black", alpha=0.8, zorder=1)

    ax.set_yticks(y_base)
    ax.set_yticklabels([LABELS[p] for p in PREDICTOR_ORDER])
    for tick in ax.get_yticklabels():
        tick.set_color("black")

    ax.set_xlabel("Effect on next fixation probability (log odds)")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor="#444", markersize=9,
                   markeredgecolor="black", markeredgewidth=1.0,
                   label="Humans"),
    ]
    if rnn is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="#bbb", markersize=9,
                       markeredgecolor="black", markeredgewidth=1.0,
                       label="Network"))
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
              bbox_to_anchor=(0.99, 0.98), frameon=True, fancybox=False,
              facecolor="white", edgecolor="black", handletextpad=0.3)
    ax.set_title("Predictors of next fixation", fontsize=11)

    # Coloured category background bands, with rotated labels in the left margin.
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_left = -0.60
    x_right = 1.0
    section_label_x = x_left + 0.035

    def draw_section_box(y_lo, y_hi, label, color):
        ax.add_patch(Rectangle(
            (x_left, y_lo), x_right - x_left, y_hi - y_lo,
            facecolor=color, edgecolor="none", alpha=0.18,
            transform=blend, clip_on=False, zorder=0))
        ax.text(section_label_x, (y_lo + y_hi) / 2, label,
                color="black", va="center", ha="center",
                fontsize=10, fontweight="bold",
                transform=blend, clip_on=False, rotation=90)

    pad_inner = 0.45
    pad_outer = 0.55
    box_top, box_bot = -np.inf, np.inf
    for i, (name, preds, colors) in enumerate(GROUPS):
        ys = [y_idx[p] for p in preds]
        top_pad = pad_outer if i == 0 else pad_inner
        bot_pad = pad_outer if i == len(GROUPS) - 1 else pad_inner
        y_lo, y_hi = min(ys) - bot_pad, max(ys) + top_pad
        draw_section_box(y_lo, y_hi, name, colors["band"])
        box_top = max(box_top, y_hi)
        box_bot = min(box_bot, y_lo)

    section_gap = 1.0 - 2 * pad_inner
    ax.set_ylim(box_bot - section_gap, box_top + section_gap)
    fig.subplots_adjust(left=0.34, right=0.96)

    out_pdf = OUT / "next_fixation_forest.pdf"
    fig.savefig(OUT / "next_fixation_forest.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
