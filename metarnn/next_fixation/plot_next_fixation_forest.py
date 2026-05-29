"""Figure 5: forest plot of the next-fixation conditional logit + transition
panels from the spatial-structure analysis.

Left side: forest plot displaying nine of the ten z-scored candidate-level
predictors fit by the conditional logit (the alternation/previous-item
predictor is fit but excluded from this display and analyzed in the
supplementary text). Predictors are grouped into three categories shown as
coloured background bands: resource-rational signals predicted by the optimal
sampling policy, encoding-order biases, and biases. Compares human
participants (hierarchical fixed effects, with per-subject random-effect
dots) against the prior-memory network (fixed-effect estimate).

Right side: transition-matrix template + observed heatmaps and bidirectional
delta-similarity bars, rendered for both humans (top) and the network
(bottom). These reuse the helpers from the supplementary fixation-transition
script so each panel looks identical to what it did when it lived in the old
Figure S6.

Usage (from repo root):
  python metarnn/next_fixation/plot_next_fixation_forest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_choice_fixation_sweeps import (  # noqa: E402
    PANEL_ALL,
    sweep_template_matrices,
)
from supplemental_analysis.fixation_transitions.plot_fixation_transitions import (  # noqa: E402
    _load_or_compute_sweep,
    _plot_single_delta_similarity,
    _plot_single_heatmap,
)

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2
sns.set_context("poster")

OUT = Path(__file__).resolve().parents[2] / "output" / "next_fixation"
NN_ROOT = _REPO_ROOT / "metarnn" / "simulations" / "human_like_04_04_input5"

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
]

LABELS = {
    "share_k_z":            "Inverse Relevant Fix. Time",
    "dist_cw_z":            "Spatial Distance (clockwise)",
    "dist_ccw_z":           "Spatial Distance (counter)",
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
    ("Biases",
     ["abs_reward_k_z", "signed_reward_k_z"],
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


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def _render_forest_plot(fig, ax, human, rnn, per_subj):
    """Render the forest-plot panel into the given axis.

    Visuals are kept identical to the standalone Figure 5: per-subject jitter
    dots behind population markers, error bars from the 95% HDI, coloured
    category background bands with rotated labels in the left margin, and the
    x-axis pinned to the range it had when the previous-item predictor was
    still displayed.
    """
    n = len(PREDICTOR_ORDER)
    y_base = np.arange(n)[::-1]
    y_idx = {p: y_base[i] for i, p in enumerate(PREDICTOR_ORDER)}
    offset = 0.18

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
                   s=90, color=PRIMARY[predictor][1]["human"],
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
                        fmt="o", color=c[which], markersize=18,
                        markeredgecolor="black", markeredgewidth=3.0,
                        ecolor="black", elinewidth=3.0,
                        capsize=0, zorder=3)

    ax.axvline(0, ls=":", color="black", alpha=0.8, zorder=1)

    ax.set_yticks(y_base)
    ax.set_yticklabels([LABELS[p] for p in PREDICTOR_ORDER])
    for tick in ax.get_yticklabels():
        tick.set_color("black")
    ax.tick_params(axis="y", labelsize=21)

    ax.set_xlabel("Effect on next location probability (log odds)", fontsize=24)
    ax.tick_params(axis="x", labelsize=22)
    # Hold the xlim to what it would have been with all ten predictors fit
    # (when the previous-item predictor pushed the negative side wider),
    # so the figure x-axis matches the prior version of Fig. 5.
    ax.set_xlim(-0.3302, 0.5375)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor="#444", markersize=18,
                   markeredgecolor="black", markeredgewidth=1.5,
                   label="Humans"),
    ]
    if rnn is not None:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor="#bbb", markersize=18,
                       markeredgecolor="black", markeredgewidth=1.5,
                       label="Network"))
    ax.legend(handles=legend_handles, fontsize=18, loc="upper right",
              bbox_to_anchor=(0.99, 0.98), frameon=True, fancybox=False,
              facecolor="white", edgecolor="black", handletextpad=0.3)
    ax.set_title("Predictors of next fixation location", fontsize=26)

    # Coloured category background bands, with rotated labels in the left margin.
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    x_left = -0.72
    x_right = 1.0
    section_label_x = x_left + 0.05

    def draw_section_box(y_lo, y_hi, label, color):
        ax.add_patch(Rectangle(
            (x_left, y_lo), x_right - x_left, y_hi - y_lo,
            facecolor=color, edgecolor="none", alpha=0.18,
            transform=blend, clip_on=False, zorder=0))
        ax.text(section_label_x, (y_lo + y_hi) / 2, label,
                color="black", va="center", ha="center",
                fontsize=23.5, fontweight="bold",
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


# ---------------------------------------------------------------------------
# Compound transition panel (template + observed heatmaps + delta-sim)
# ---------------------------------------------------------------------------

def _obs_offdiag_range(sweep_data):
    """Per-row colour-scale range fit to off-diagonal observed cells.

    Same logic as the original Figure S6 column-1 compound panel: the
    diagonal of the observed matrix is zero by construction so it would
    pull the colour scale to one end if included. Fitting [vmin, vmax] to
    the off-diagonal cells keeps the colour gradient on the actual data.
    """
    m = sweep_data["obs_trans_by_panel"][PANEL_ALL]
    if m.size == 0:
        return 0.0, 1.0
    off_diag_mask = ~np.eye(m.shape[0], dtype=bool) & np.isfinite(m)
    vals = m[off_diag_mask]
    if vals.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(vals.min()))
    hi = max(lo + 1e-6, float(vals.max()))
    return lo, hi


def _render_compound_panel(
    fig, ax_template, ax_observed, ax_delta_sim,
    sweep, bi_template, *,
    hm_cmap="viridis",
    ds_ylim, ds_yticks,
):
    """Render one transition compound panel: template heatmap (top-left) +
    observed heatmap (bottom-left) + bidirectional delta-similarity bar
    (right). Visuals replicate the original Figure S6 column 1.
    """
    # Template heatmap (top-left)
    divider_tmpl = make_axes_locatable(ax_template)
    ax_cbar_tmpl = divider_tmpl.append_axes("right", size="5%", pad=0.08)
    _plot_single_heatmap(
        ax_template, bi_template,
        cmap="viridis",
        title="Template",
        vmin=0.0, vmax=0.5,
        show_xlabel=False, show_ylabel=False,
        mask_diagonal=True,
        cbar=True,
        cbar_ax=ax_cbar_tmpl,
    )
    ax_cbar_tmpl.set_yticks([0.0, 0.25, 0.5])
    ax_cbar_tmpl.set_yticklabels(["0.00", "0.25", "0.50"])
    ax_cbar_tmpl.tick_params(labelsize=14)

    # Observed heatmap (bottom-left)
    obs_mat = sweep["obs_trans_by_panel"][PANEL_ALL]
    hm_vmin, hm_vmax = _obs_offdiag_range(sweep)
    divider = make_axes_locatable(ax_observed)
    ax_cbar = divider.append_axes("right", size="5%", pad=0.08)
    _plot_single_heatmap(
        ax_observed, obs_mat,
        cmap=hm_cmap,
        title="Observed",
        vmin=hm_vmin, vmax=hm_vmax,
        mask_diagonal=True,
        cbar=True,
        cbar_ax=ax_cbar,
    )
    cb_mid = (hm_vmin + hm_vmax) / 2
    ax_cbar.set_yticks([hm_vmin, cb_mid, hm_vmax])
    ax_cbar.set_yticklabels([f"{hm_vmin:.2f}", f"{cb_mid:.2f}", f"{hm_vmax:.2f}"])
    ax_cbar.tick_params(labelsize=14)

    # Bidirectional delta-similarity (right, full height of compound panel)
    delta_bi = sweep["delta_similarity"][PANEL_ALL]["bidirectional"]
    _plot_single_delta_similarity(
        ax_delta_sim, delta_bi,
        ylim=ds_ylim, yticks=ds_yticks,
    )


# ---------------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------------

def main():
    # Forest-plot inputs
    human = load_betas("conditional_logit_human_population_beta.csv")
    rnn_path = OUT / "conditional_logit_rnn_input5_500k_beta.csv"
    rnn = load_betas(rnn_path.name) if rnn_path.exists() else None
    per_subj = pd.read_csv(OUT / "conditional_logit_human_per_subject_beta.csv")

    # Sweep-transition inputs (humans use repo root; network uses the
    # human-like simulation directory matching the existing Figure S6 setup).
    print("Loading sweep transition data ...", flush=True)
    human_sweep = _load_or_compute_sweep(
        _REPO_ROOT,
        buffer=50, n_sims=1000, seed=123,
        exclude_subjects=("107", "131"),
        label="human",
        collapse_null_shuffles=True,
    )
    nn_sweep = _load_or_compute_sweep(
        NN_ROOT,
        buffer=50, n_sims=1000, seed=123,
        exclude_subjects=(),
        label="NN",
        collapse_null_shuffles=True,
    )
    bi_template = sweep_template_matrices(n_items=6)["bidirectional"]

    # Combined figure: forest plot on the left, compound transition panels
    # (Humans on top, Network on bottom) on the right.
    #
    # We use two separate gridspecs so the right column's top can sit lower
    # than the forest plot's. The forest plot's title ("Predictors of next
    # fixation location") sits just above its axes; placing the right column
    # lower opens up space above the Humans compound panel for the "Humans"
    # label to be aligned vertically with that title.
    fig = plt.figure(figsize=(20, 13))
    gs_left = fig.add_gridspec(
        1, 1, left=0.13, right=0.52, top=0.94, bottom=0.06,
    )
    gs_right = fig.add_gridspec(
        2, 1, left=0.537, right=0.95, top=0.89, bottom=0.06,
        hspace=0.55,
    )

    # --- Forest plot (left) ---
    ax_forest = fig.add_subplot(gs_left[0, 0])
    _render_forest_plot(fig, ax_forest, human, rnn, per_subj)

    compound_axes = []
    row_configs = [
        {
            "label": "Humans",
            "sweep": human_sweep,
            "ds_ylim": (0, 0.6),
            "ds_yticks": [0, 0.2, 0.4, 0.6],
        },
        {
            "label": "Network",
            "sweep": nn_sweep,
            "ds_ylim": (0, 0.6),
            "ds_yticks": [0, 0.2, 0.4, 0.6],
        },
    ]

    for r, cfg in enumerate(row_configs):
        # Each row: heatmaps stack (template + observed) | delta-similarity bar
        gs_compound = gs_right[r, 0].subgridspec(
            1, 2,
            wspace=0.1,
            width_ratios=[1.0, 0.4],
        )
        gs_heatmaps = gs_compound[0, 0].subgridspec(
            2, 1,
            hspace=0.55,
        )
        ax_template = fig.add_subplot(gs_heatmaps[0, 0])
        ax_observed = fig.add_subplot(gs_heatmaps[1, 0])
        ax_delta = fig.add_subplot(gs_compound[0, 1])

        _render_compound_panel(
            fig, ax_template, ax_observed, ax_delta,
            cfg["sweep"], bi_template,
            ds_ylim=cfg["ds_ylim"], ds_yticks=cfg["ds_yticks"],
        )
        compound_axes.append({"template": ax_template, "delta": ax_delta, **cfg})

    # Place "Humans" / "Network" labels centred above each compound panel.
    # The Humans label is anchored to the same y-position as the forest
    # plot's title ("Predictors of next fixation location"), and the Network
    # label uses an analogous offset above its own panel for visual parity.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_bbox = ax_forest.title.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    title_y_center = (title_bbox.y0 + title_bbox.y1) / 2

    humans_entry = compound_axes[0]
    network_entry = compound_axes[1]

    pos_tmpl_h = humans_entry["template"].get_position()
    pos_delta_h = humans_entry["delta"].get_position()
    fig.text(
        (pos_tmpl_h.x0 + pos_delta_h.x1) / 2, title_y_center,
        humans_entry["label"],
        fontsize=26, fontweight="normal",
        ha="center", va="center",
        transform=fig.transFigure,
    )

    # Network: use the same vertical offset (top-of-panel to label-center)
    # that lands Humans on the title baseline, so the two rows feel parallel.
    humans_offset = title_y_center - pos_tmpl_h.y1
    pos_tmpl_n = network_entry["template"].get_position()
    pos_delta_n = network_entry["delta"].get_position()
    fig.text(
        (pos_tmpl_n.x0 + pos_delta_n.x1) / 2,
        pos_tmpl_n.y1 + humans_offset,
        network_entry["label"],
        fontsize=26, fontweight="normal",
        ha="center", va="center",
        transform=fig.transFigure,
    )

    out_pdf_local = OUT / "next_fixation_forest.pdf"
    out_pdf_final = Path(__file__).resolve().parents[2] / "output" / "figures" / "Figure5.pdf"
    out_pdf_final.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "next_fixation_forest.png", dpi=160, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_pdf_local, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_pdf_final, bbox_inches="tight", pad_inches=0.1)
    print(f"saved {out_pdf_local} and {out_pdf_final}")


if __name__ == "__main__":
    main()
