"""Forest plot of per-feature-dimension deviation from the grand mean.

Reads the deviation summary produced by compute_feature_stats.R and
produces an eight-panel forest figure (one per analysis), each showing
the four feature dimensions' posterior deviations from the grand mean
with 95% HDIs.

Input:  ../output/feature_deviation_from_mean.csv
Output: ../output/FigureS7.pdf
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUT_DIR = os.path.join(BASE_DIR, "output")
SUPPLEMENT_DIR = os.path.join(REPO_ROOT, "output", "figures", "supplementary")

FEATURE_ORDER = ["animacy", "environment", "size", "texture"]
FEATURE_DISPLAY = {
    "animacy": "Animacy",
    "environment": "Environment",
    "size": "Size",
    "texture": "Texture",
}
_SET2 = sns.color_palette("Set2", n_colors=len(FEATURE_ORDER))
FEATURE_COLORS = dict(zip(FEATURE_ORDER, [tuple(c) for c in _SET2]))

# (analysis label in CSV, panel title, x-axis label). Convention:
# `~` for regression slopes, `×` for interactions, `~ 1` for
# intercept-only (per-dim) models; sentence case.
PANEL_SPECS = [
    (
        "Accuracy (logit, no covariate)",
        "Accuracy ~ 1",
        r"Δ β from grand mean (log-odds)",
    ),
    (
        "RT slope (log-RT per recalled item)",
        "RT ~ number of memories",
        r"Δ β from grand mean (log-RT)",
    ),
    (
        "Choice ~ true offer value (per-dim logit slope)",
        "Choice ~ true offer value",
        r"Δ β from grand mean (log-odds)",
    ),
    (
        "Choice ~ recalled offer value (per-dim logit slope)",
        "Choice ~ recalled offer value",
        r"Δ β from grand mean (log-odds)",
    ),
    (
        "RT ~ (true offer value)^2 (per-dim quadratic)",
        "RT ~ (true offer value)²",
        r"Δ β from grand mean (log-RT)",
    ),
    (
        "RT ~ (recalled offer value)^2 (per-dim quadratic)",
        "RT ~ (recalled offer value)²",
        r"Δ β from grand mean (log-RT)",
    ),
    (
        "Relevant-fixation deviation from chance",
        "Relevant fixation prop. ~ 1",
        r"Δ β from grand mean (Δ proportion)",
    ),
    (
        "Choice x Valence interaction on Delta relevant fixation",
        "Fixation prop. ~ choice × valence",
        r"Δ β from grand mean (Δ proportion)",
    ),
]


def draw_panel(ax, df_panel: pd.DataFrame, title: str, xlabel: str):
    df_panel = (df_panel.set_index("feature_dim")
                .loc[FEATURE_ORDER]
                .reset_index())
    y = np.arange(len(FEATURE_ORDER))[::-1]
    for i, row in df_panel.iterrows():
        yi = y[i]
        color = FEATURE_COLORS[row["feature_dim"]]
        ax.plot([row["lo95"], row["hi95"]], [yi, yi],
                color=color, linewidth=3, zorder=2)
        ax.scatter([row["dev_mean"]], [yi],
                   s=12 ** 2, facecolor=color, edgecolor="black",
                   linewidth=2.0, zorder=3)
    ax.axvline(0, linestyle=":", color="black", linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_DISPLAY[d] for d in FEATURE_ORDER])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    lo = df_panel["lo95"].min()
    hi = df_panel["hi95"].max()
    pad = 0.15 * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "feature_deviation_from_mean.csv"))

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 16,
        "lines.solid_capstyle": "butt",
    }):
        fig, axes = plt.subplots(4, 2, figsize=(14, 17))
        for ax, (analysis_label, title, xlabel) in zip(axes.flat, PANEL_SPECS):
            df_panel = df[df["analysis"] == analysis_label]
            if df_panel.empty:
                ax.set_title(f"(no data) {title}")
                ax.axis("off")
                continue
            draw_panel(ax, df_panel, title, xlabel)

        fig.tight_layout(h_pad=2.5, w_pad=3.0)
        local_out = os.path.join(OUT_DIR, "FigureS7.pdf")
        fig.savefig(local_out, bbox_inches="tight")
        print(f"Wrote {local_out}")

        if os.path.isdir(SUPPLEMENT_DIR):
            supplement_out = os.path.join(SUPPLEMENT_DIR, "FigureS7.pdf")
            fig.savefig(supplement_out, bbox_inches="tight")
            print(f"Wrote {supplement_out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
