#!/usr/bin/env python3
"""Compare two NN simulation results on the same figure.

Same 2x4 layout as plot_NN_H_comparison.py.  Human panels are unchanged.
NN panels overlay two simulation datasets:
  - NN1: solid lines (same as original script)
  - NN2: dashed lines

Output is written to ``output/human_comparison/`` under whichever NN root
is **not** the input0 baseline (i.e. the "main" simulation being compared).
The input0 root is identified by the presence of "input0" in its path; if
neither root contains "input0", the output falls back to nn-root2.

Example
-------
conda run -n analysis python metarnn/plot_NN_NN_comparison.py \
  --nn-root1 metarnn/simulations/human_like_04_04_input0 \
  --nn-root2 metarnn/simulations/human_like_04_04_input5 \
  --label1 "Input 0" --label2 "Input 5" \
  --tag 04_04_input5_vs_input0 \
  --nn-tag1 04_04_input0 --nn-tag2 04_04_input5
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_fixation_duration_by_position import (  # noqa: E402
    COLLAPSE_KEEP_FIRST_N,
    DEFAULT_EXCLUDED as DEFAULT_EXCLUDED_EYE,
    build_duration_summaries_7plus,
    load_clean_choice_fixations as load_clean_choice_fixations_full,
)

# Import compute and helper functions from the original comparison script.
from metarnn.lib.plot_NN_H_comparison import (  # noqa: E402
    _add_panel_label,
    _add_shared_xlabel,
    _compute_prop_relevant_by_position_7plus,
    _compute_prop_positive_negative_relevant_by_decision,
    _compute_chance_positive_by_decision,
    _ensure_dir,
    _find_default_clean_choice_fixations,
    _find_revisits_by_subject_csv,
    _sem,
    _subject_cumtime_curve,
)


def _style_legend(lg):
    """Apply consistent legend frame styling."""
    lg.get_frame().set_boxstyle("square,pad=0.1")
    lg.get_frame().set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Disk-based caching utilities
# ---------------------------------------------------------------------------


def _cache_key(name: str, source_path: Path, *extra: Any) -> str:
    """Deterministic cache filename from source file mtime and parameters."""
    mtime = source_path.stat().st_mtime
    h = hashlib.md5(f"{source_path}:{mtime}:{extra}".encode()).hexdigest()[:12]
    return f"{name}_{h}"


def _disk_cache(cache_dir: Path, key: str, compute_fn: Any) -> Any:
    """Return cached result if available, otherwise compute, save, and return."""
    cache_file = cache_dir / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    result = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def _resolve_out_dir(nn_root1: Path, nn_root2: Path) -> Path:
    """Return ``output/human_comparison/`` under the non-input0 root.

    The input0 root is identified by the presence of "input0" anywhere in its
    resolved path string.  If *neither* root contains "input0" the output
    defaults to nn_root2.  If *both* contain "input0" (unusual) the output
    also defaults to nn_root2.
    """
    r1_is_input0 = "input0" in str(nn_root1)
    r2_is_input0 = "input0" in str(nn_root2)

    if r1_is_input0 and not r2_is_input0:
        main_root = nn_root2
    elif r2_is_input0 and not r1_is_input0:
        main_root = nn_root1
    else:
        # Ambiguous — fall back to nn_root2.
        main_root = nn_root2

    return main_root / "output" / "human_comparison"


# ---------------------------------------------------------------------------
# Panel A: bar chart with Humans, NN1, NN2
# ---------------------------------------------------------------------------


def _panel_A_dual(
    ax: plt.Axes,
    human_by_sub: pd.DataFrame,
    nn1_by_sub: pd.DataFrame,
    nn2_by_sub: pd.DataFrame,
    *,
    label1: str,
    label2: str,
) -> None:
    from matplotlib.patches import Patch

    hf = pd.to_numeric(human_by_sub["firstfix_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    hr = pd.to_numeric(human_by_sub["revisit_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    n1f = pd.to_numeric(nn1_by_sub["firstfix_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    n1r = pd.to_numeric(nn1_by_sub["revisit_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    n2f = pd.to_numeric(nn2_by_sub["firstfix_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)
    n2r = pd.to_numeric(nn2_by_sub["revisit_prop_time_relevant"], errors="coerce").to_numpy(dtype=float)

    # Layout: 2 groups (Initial, Revisit) x 3 bars (Human, label1, label2).
    w = 0.25
    gap = 0.0
    group_centers = np.array([0.0, 1.2])

    # Bar specs: (fill_color, outline_linestyle)
    #   Human = dark gray, solid outline
    #   label1 (Prior Memory) = lighter gray, solid outline
    #   label2 (No Prior Memory) = lighter gray, dashed outline
    fill_colors = ["0.6", "0.95", "0.95"]
    outline_styles = ["-", "-", "--"]
    bar_labels = ["Human", label1, label2]

    rng = np.random.default_rng(42)

    for gi, (group_vals, gc) in enumerate(zip(
        [(hf, n1f, n2f), (hr, n1r, n2r)],
        group_centers,
    )):
        for bi in range(3):
            vals = group_vals[bi]
            x_pos = gc + (bi - 1) * (w + gap)
            vals_clean = vals[np.isfinite(vals)]
            mean_val = float(np.nanmean(vals_clean)) if len(vals_clean) > 0 else 0.0
            sem_val = _sem(vals_clean) if len(vals_clean) > 0 else 0.0

            # Subject dots (styled like overview Panel D)
            jitter = rng.uniform(-w * 0.3, w * 0.3, size=len(vals_clean))
            ax.scatter(
                np.full(len(vals_clean), x_pos) + jitter, vals_clean,
                s=6**2, facecolor=(1, 1, 1, 0.5), edgecolor=(0, 0, 0, 0.5),
                linewidth=0.5, zorder=3,
            )

            # Fill bar
            ax.bar(x_pos, mean_val, w, color=fill_colors[bi], edgecolor="none",
                   linewidth=0, zorder=2)
            # Outline bar
            ax.bar(x_pos, mean_val, w, color="none", edgecolor="black",
                   linewidth=2.5, linestyle=outline_styles[bi], zorder=4)
            # Error bar
            ax.errorbar(x_pos, mean_val, yerr=sem_val, fmt="none",
                        ecolor="black", capsize=0, linewidth=2.5, zorder=5)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Initial", "Revisit"], fontsize=18)
    ax.set_xlabel("Fixation type")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prop. Relevant Fix. Time")
    ax.axhline(0.5, color="black", linewidth=1.5, linestyle=":")

    ax.spines[["top", "right"]].set_visible(False)

    # Legend matching overview style
    legend_handles = [
        Patch(facecolor="0.7", edgecolor="black", linewidth=2.5, label="Human"),
        Patch(facecolor="0.93", edgecolor="black", linewidth=2.5, label=label1),
        Patch(facecolor="0.93", edgecolor="black", linewidth=2.5, linestyle="--", label=label2),
    ]
    _lg = ax.legend(handles=legend_handles, frameon=True, fontsize=12, handlelength=1.5,
                     loc="best", handletextpad=0.5, labelspacing=0.3,
                     facecolor="white", edgecolor="black")
    _style_legend(_lg)


# ---------------------------------------------------------------------------
# Panel B: cumulative fixation-time share — NN subplot overlays two datasets
# ---------------------------------------------------------------------------


def _panel_B_dual(
    ax_h: plt.Axes,
    ax_n: plt.Axes,
    *,
    human_fix: pd.DataFrame,
    nn1_fix: pd.DataFrame,
    nn2_fix: pd.DataFrame,
    max_fixations: int,
    cumtime_mode: str,
    label1: str,
    label2: str,
    hline_y: float = 0.8,
) -> None:
    def _first_crossing(xs: np.ndarray, mean: np.ndarray, thresh: float) -> Optional[int]:
        ok = np.isfinite(xs) & np.isfinite(mean)
        if not np.any(ok):
            return None
        hit = np.where(mean[ok] >= float(thresh))[0]
        if len(hit) == 0:
            return None
        return int(xs[ok][int(hit[0])])

    def _curves(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        curves = []
        for _, dsub in df.groupby("subject_id", sort=True):
            try:
                v = _subject_cumtime_curve(dsub, max_fixations=max_fixations, mode=cumtime_mode)
            except Exception:
                continue
            if np.all(~np.isfinite(v)):
                continue
            curves.append(v)
        if len(curves) == 0:
            raise ValueError("No per-subject cumulative curves could be computed.")
        mat = np.vstack(curves)
        mean = np.nanmean(mat, axis=0)
        xs = np.arange(1, max_fixations + 1)
        return xs, mean, mat

    def _draw_single(ax: plt.Axes, *, df: pd.DataFrame, title: str, color: str, show_ylabel: bool) -> None:
        xs, mean, mat = _curves(df)
        for row in mat:
            ax.plot(xs, row, color=color, alpha=0.18, linewidth=1.0, zorder=1)
        ax.plot(xs, mean, color=color, linewidth=3.0, zorder=3)
        ax.axhline(float(hline_y), color="black", linewidth=1.5, zorder=2)
        hit_x = _first_crossing(xs, mean, float(hline_y))
        txt = f"80% at fixation {hit_x}" if hit_x else "80% not reached"
        ax.set_title(title)
        ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel("Cumulative Fixation Time")
        ax.set_ylim(0.0, 1.01)
        ax.set_xlim(0.0, float(max_fixations))
        ax.set_xticks([0, 20])
        ax.spines[["top", "right"]].set_visible(False)

    def _draw_dual(ax: plt.Axes, *, df1: pd.DataFrame, df2: pd.DataFrame, show_ylabel: bool) -> None:
        """NN subplot: overlay two datasets."""
        xs1, mean1, mat1 = _curves(df1)
        xs2, mean2, mat2 = _curves(df2)

        color1 = "k"
        color2 = "k"

        for row in mat1:
            ax.plot(xs1, row, color=color1, alpha=0.12, linewidth=0.8, zorder=1)
        for row in mat2:
            ax.plot(xs2, row, color=color2, alpha=0.12, linewidth=0.8, linestyle="--", zorder=1)

        ax.plot(xs1, mean1, color=color1, linewidth=3.0, zorder=3, label=label1)
        ax.plot(xs2, mean2, color=color2, linewidth=3.0, linestyle="--", zorder=3, label=label2)

        ax.axhline(float(hline_y), color="black", linewidth=1.5, zorder=2)

        hit1 = _first_crossing(xs1, mean1, float(hline_y))
        hit2 = _first_crossing(xs2, mean2, float(hline_y))
        ax.set_title("RNN")
        ax.set_xlabel("")
        if show_ylabel:
            ax.set_ylabel("Cumulative fixation time")
        ax.set_ylim(0.0, 1.01)
        ax.set_xlim(0.0, float(max_fixations))
        ax.set_xticks([0, 20])
        ax.spines[["top", "right"]].set_visible(False)
        _lg = ax.legend(frameon=True, fontsize=12, handlelength=1.5, loc="lower right",
                        handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
        _style_legend(_lg)

    _draw_single(ax_h, df=human_fix, title="Humans", color="k", show_ylabel=True)
    _draw_dual(ax_n, df1=nn1_fix, df2=nn2_fix, show_ylabel=False)


# ---------------------------------------------------------------------------
# Panel D (duration by relevance): overlay two NN datasets
# ---------------------------------------------------------------------------


def _panel_duration_dual(
    ax: plt.Axes,
    *,
    label1: str,
    label2: str,
    summary1: pd.DataFrame,
    summary2: pd.DataFrame,
    y_label: str = "Fixation Duration (steps)",
    y_scale: float = 1.0,
    y_lim: Optional[Tuple[float, float]] = None,
    keep_first_n: int,
) -> None:
    """Duration by relevance, overlaying NN1 (solid) and NN2 (dashed)."""
    color_rel = "#6fc7eb"
    color_irr = "#ba7caf"

    for summary_7p, ls, alpha_band, lbl_suffix in [
        (summary1, "-", 0.15, label1),
        (summary2, "--", 0.10, label2),
    ]:
        if summary_7p is None or summary_7p.empty:
            continue

        for subset_name, color in [("relevant", color_rel), ("irrelevant", color_irr)]:
            s = (
                summary_7p[(summary_7p["panel"] == "by_relevance") & (summary_7p["subset"] == subset_name)]
                .copy()
                .sort_values("fixation_position")
            )
            if s.empty:
                continue
            x = pd.to_numeric(s["fixation_position"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(s["mean_duration"], errors="coerce").to_numpy(dtype=float)
            se = pd.to_numeric(s.get("mean_duration_sem"), errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, se = x[ok], y[ok], se[ok] if len(se) == len(ok) else se

            y_plot = y * float(y_scale)
            se_plot = se * float(y_scale) if np.isfinite(se).any() else se

            ax.plot(x, y_plot, color=color, linewidth=2.5, linestyle=ls)
            if np.isfinite(se).any() and len(se) == len(y):
                ax.fill_between(x, y_plot - se_plot, y_plot + se_plot, color=color, alpha=alpha_band, linewidth=0)

    ax.set_title("RNN")
    ax.set_xlabel("Fixation number")
    ax.set_ylabel(str(y_label))
    if y_lim is not None:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    ax.grid(False)
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="-", label=label1),
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="--", label=label2),
    ]
    _lg = ax.legend(handles=legend_handles, frameon=True, fontsize=12, handlelength=1.5, loc="best",
                     handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
    _style_legend(_lg)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Panel F (prop relevant by position): overlay two NN datasets
# ---------------------------------------------------------------------------


def _panel_prop_relevant_dual(
    ax: plt.Axes,
    *,
    label1: str,
    label2: str,
    summary1: pd.DataFrame,
    summary2: pd.DataFrame,
    keep_first_n: int,
    y_label: str = "Prop. Relevant Fixations",
    hline_y: float = 0.5,
    y_lim: Tuple[float, float] = (0.2, 0.8),
) -> None:
    """Prop relevant by position, overlaying NN1 (solid) and NN2 (dashed)."""
    line_color = "black"

    for summary, ls, alpha_band, lbl in [
        (summary1, "-", 0.20, label1),
        (summary2, "--", 0.12, label2),
    ]:
        if summary is None or summary.empty:
            continue

        x = pd.to_numeric(summary["fixation_position"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(summary["prop_relevant_mean"], errors="coerce").to_numpy(dtype=float)
        se = pd.to_numeric(summary.get("prop_relevant_sem"), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, se = x[ok], y[ok], se[ok] if len(se) == len(ok) else se

        ax.plot(x, y, color=line_color, linewidth=2.5, linestyle=ls, label=lbl)
        if np.isfinite(se).any() and len(se) == len(y):
            lo = np.clip(y - se, 0.0, 1.0)
            hi = np.clip(y + se, 0.0, 1.0)
            ax.fill_between(x, lo, hi, color=line_color, alpha=alpha_band, linewidth=0)

    ax.axhline(float(hline_y), color="black", linewidth=1.5, linestyle=":")
    ax.set_title("RNN")
    ax.set_xlabel("Fixation number")
    ax.set_ylabel(str(y_label))
    ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ticks = list(range(1, int(keep_first_n) + 2))
    ax.set_xticks(ticks)
    tail_label = f"{int(keep_first_n) + 1}+"
    ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    _lg = ax.legend(frameon=True, fontsize=12, handlelength=1.5, loc="best",
                     handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
    _style_legend(_lg)


# ---------------------------------------------------------------------------
# Panel H (positive/negative relevant by take/leave): overlay two NN datasets
# ---------------------------------------------------------------------------


def _panel_valence_dual(
    ax_take: plt.Axes,
    ax_leave: plt.Axes,
    *,
    label1: str,
    label2: str,
    valence_summaries1: dict,
    valence_summaries2: dict,
    keep_first_n: int,
    y_lim: Tuple[float, float] = (0.2, 0.8),
) -> None:
    """Prop positive/negative relevant fixations, take/leave split, two NN datasets.

    valence_summaries1/2: dict mapping decision (1 or 2) -> summary DataFrame.
    """
    colors = {"positive": "tab:blue", "negative": "tab:orange"}

    for idx, (val_summaries, ls, alpha_band, lbl_suffix) in enumerate([
        (valence_summaries1, "-", 0.20, label1),
        (valence_summaries2, "--", 0.12, label2),
    ]):

        for ax, decision, dec_label in [(ax_take, 1, "Take"), (ax_leave, 2, "Leave")]:
            summary = val_summaries.get(decision)
            if summary is None or summary.empty:
                continue
            for val in ["positive", "negative"]:
                s = summary[summary["valence"] == val].copy().sort_values("pos_collapsed")
                if s.empty:
                    continue
                x = s["pos_collapsed"].to_numpy(dtype=float)
                y = s["prop_mean"].to_numpy(dtype=float)
                se = s["prop_sem"].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y, se = x[ok], y[ok], se[ok]

                ax.plot(x, y, color=colors[val], linewidth=2.5, linestyle=ls)
                if np.isfinite(se).any() and len(se) == len(y):
                    lo = np.clip(y - se, 0.0, 1.0)
                    hi = np.clip(y + se, 0.0, 1.0)
                    ax.fill_between(x, lo, hi, color=colors[val], alpha=alpha_band, linewidth=0)

            # Single chance baseline at 0.25 — only draw once.
            if idx == 0:
                ax.axhline(0.25, color="black", linewidth=1.5, linestyle=":")

    # Formatting.
    for ax, dec_label in [(ax_take, "Take"), (ax_leave, "Leave")]:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ticks = list(range(1, int(keep_first_n) + 2))
        ax.set_xticks(ticks)
        tail_label = f"{int(keep_first_n) + 1}+"
        ax.set_xticklabels([str(i) for i in range(1, int(keep_first_n) + 1)] + [tail_label])

    # Take subplot: subtitle, hide x tick labels via tick_params
    # (set_xticklabels([]) would clear the shared-axis formatter).
    ax_take.set_title("Take", fontsize=14)
    ax_take.tick_params(labelbottom=False)

    # Leave subplot: subtitle, show fixation number labels, x-label.
    ax_leave.set_title("Leave", fontsize=14)
    ax_leave.set_xlabel("Fixation number")

    # Shared y-axis label centered between take and leave subplots.
    ax_take.set_ylabel("Proportion of Fixations")
    ax_take.yaxis.set_label_coords(-0.22, -0.15)
    ax_leave.set_ylabel("")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="-", label=label1),
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="--", label=label2),
    ]
    _lg = ax_take.legend(handles=legend_handles, frameon=True, fontsize=12, handlelength=1.5, loc="best",
                          handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
    _style_legend(_lg)


# ---------------------------------------------------------------------------
# Reuse single-dataset panel functions for human-only panels (C, E, G).
# Import them from the original script.
# ---------------------------------------------------------------------------

from metarnn.lib.plot_NN_H_comparison import (  # noqa: E402
    _panel_duration_by_relevance_7plus,
    _panel_prop_relevant_fixated_by_position_7plus,
    _panel_prop_positive_negative_relevant_by_decision,
)


# ---------------------------------------------------------------------------
# Fixation-number supplement: 3-panel bar chart (All / Relevant / Irrelevant)
# ---------------------------------------------------------------------------


def _panel_fixation_count(
    ax: plt.Axes,
    human_by_sub: pd.DataFrame,
    nn1_by_sub: pd.DataFrame,
    nn2_by_sub: pd.DataFrame,
    *,
    label1: str,
    label2: str,
    columns: list[str],
    title: str,
    show_legend: bool = False,
    show_ylabel: bool = True,
) -> None:
    """Bar chart with 3 bars (Human, label1, label2) for fixation counts.

    Parameters
    ----------
    columns : list[str]
        Column name(s) from the revisits-by-subject CSV to sum for each
        subject.  E.g. ``["allfix_count_relevant_per_trial"]`` or
        ``["allfix_count_relevant_per_trial",
           "allfix_count_irrelevant_per_trial"]`` for the total.
    """
    from matplotlib.patches import Patch

    def _extract(df: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(df), dtype=float)
        for col in columns:
            total += pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return total

    vals_list = [_extract(human_by_sub), _extract(nn1_by_sub), _extract(nn2_by_sub)]

    w = 0.25
    fill_colors = ["0.6", "0.95", "0.95"]
    outline_styles = ["-", "-", "--"]
    bar_labels = ["Human", label1, label2]
    rng = np.random.default_rng(42)

    for bi in range(3):
        vals = vals_list[bi]
        x_pos = float(bi)
        vals_clean = vals[np.isfinite(vals)]
        mean_val = float(np.nanmean(vals_clean)) if len(vals_clean) > 0 else 0.0
        sem_val = _sem(vals_clean) if len(vals_clean) > 0 else 0.0

        # Subject dots
        jitter = rng.uniform(-w * 0.3, w * 0.3, size=len(vals_clean))
        ax.scatter(
            np.full(len(vals_clean), x_pos) + jitter, vals_clean,
            s=6**2, facecolor=(1, 1, 1, 0.5), edgecolor=(0, 0, 0, 0.5),
            linewidth=0.5, zorder=3,
        )

        # Fill bar
        ax.bar(x_pos, mean_val, w, color=fill_colors[bi], edgecolor="none",
               linewidth=0, zorder=2)
        # Outline bar
        ax.bar(x_pos, mean_val, w, color="none", edgecolor="black",
               linewidth=2.5, linestyle=outline_styles[bi], zorder=4)
        # Error bar
        ax.errorbar(x_pos, mean_val, yerr=sem_val, fmt="none",
                    ecolor="black", capsize=0, linewidth=2.5, zorder=5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(bar_labels, fontsize=18)
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("Fixations per Trial")
    ax.spines[["top", "right"]].set_visible(False)

    if show_legend:
        legend_handles = [
            Patch(facecolor="0.7", edgecolor="black", linewidth=2.5, label="Human"),
            Patch(facecolor="0.93", edgecolor="black", linewidth=2.5, label=label1),
            Patch(facecolor="0.93", edgecolor="black", linewidth=2.5, linestyle="--", label=label2),
        ]
        _lg = ax.legend(handles=legend_handles, frameon=True, fontsize=12, handlelength=1.5,
                         loc="best", handletextpad=0.5, labelspacing=0.3,
                         facecolor="white", edgecolor="black")
        _style_legend(_lg)


def _save_fixation_number_supplement(
    *,
    out_dir: Path,
    tag: str,
    human_by_sub: pd.DataFrame,
    nn1_by_sub: pd.DataFrame,
    nn2_by_sub: pd.DataFrame,
    label1: str,
    label2: str,
) -> Path:
    """Create a 3-panel fixation-count supplement figure."""
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"FixationNumberSupplement{suffix}.pdf"

    panels = [
        {
            "columns": ["allfix_count_relevant_per_trial", "allfix_count_irrelevant_per_trial"],
            "title": "All Fixations",
            "show_legend": True,
            "show_ylabel": True,
        },
        {
            "columns": ["allfix_count_relevant_per_trial"],
            "title": "Relevant Fixations",
            "show_legend": False,
            "show_ylabel": False,
        },
        {
            "columns": ["allfix_count_irrelevant_per_trial"],
            "title": "Irrelevant Fixations",
            "show_legend": False,
            "show_ylabel": False,
        },
    ]

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    sns.set_context("poster")
    with plt.rc_context({
        "font.family": "Arial",
        "axes.titlesize": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, panel_cfg in zip(axes, panels):
            _panel_fixation_count(
                ax, human_by_sub, nn1_by_sub, nn2_by_sub,
                label1=label1, label2=label2, **panel_cfg,
            )
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"Wrote {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------


def create_figure(
    *,
    nn_root1: Path,
    nn_root2: Path,
    tag: str,
    nn_tag1: str,
    nn_tag2: str,
    label1: str,
    label2: str,
    max_fixations: int,
    cumtime_mode: str,
    gh_denom: str = "all",
) -> Path:
    out_dir = _resolve_out_dir(nn_root1, nn_root2)
    _ensure_dir(out_dir)
    cache_dir = out_dir / "_cache"

    # Revisits CSVs live under output/eyegaze/stats/ in each root.
    human_stats_dir = _REPO_ROOT / "output" / "eyegaze" / "stats"
    nn1_stats_dir = nn_root1 / "output" / "eyegaze" / "stats"
    nn2_stats_dir = nn_root2 / "output" / "eyegaze" / "stats"

    # Load revisits-by-subject tables (small CSVs, no caching needed).
    human_by_sub = pd.read_csv(_find_revisits_by_subject_csv(human_stats_dir, agent_tag="human"))
    nn1_by_sub = pd.read_csv(_find_revisits_by_subject_csv(nn1_stats_dir, agent_tag="nn", tag=nn_tag1))
    nn2_by_sub = pd.read_csv(_find_revisits_by_subject_csv(nn2_stats_dir, agent_tag="nn", tag=nn_tag2))

    # Load clean choice fixation data once, cached as pickle for fast reload.
    human_clean_path = _find_default_clean_choice_fixations(_REPO_ROOT)
    nn1_clean_path = _find_default_clean_choice_fixations(nn_root1)
    nn2_clean_path = _find_default_clean_choice_fixations(nn_root2)

    excluded = set(map(str, DEFAULT_EXCLUDED_EYE))

    def _load_and_filter(path: Path) -> pd.DataFrame:
        df = load_clean_choice_fixations_full(path)
        return df[~df["subject_id"].astype(str).isin(excluded)].copy()

    human_fix = _disk_cache(cache_dir, _cache_key("fix_human", human_clean_path),
                            lambda: _load_and_filter(human_clean_path))
    nn1_fix = _disk_cache(cache_dir, _cache_key("fix_nn1", nn1_clean_path),
                          lambda: _load_and_filter(nn1_clean_path))
    nn2_fix = _disk_cache(cache_dir, _cache_key("fix_nn2", nn2_clean_path),
                          lambda: _load_and_filter(nn2_clean_path))

    # Pre-compute all summaries with disk caching.
    keep = int(COLLAPSE_KEEP_FIRST_N)

    # Duration summaries (Panels C, D).
    human_dur = _disk_cache(
        cache_dir, _cache_key("dur_human", human_clean_path, keep),
        lambda: build_duration_summaries_7plus(
            human_fix, max_fixations=8, base_dir=_REPO_ROOT,
            excluded_subjects=tuple(DEFAULT_EXCLUDED_EYE), keep_first_n=keep))
    nn1_dur = _disk_cache(
        cache_dir, _cache_key("dur_nn1", nn1_clean_path, keep),
        lambda: build_duration_summaries_7plus(
            nn1_fix, max_fixations=8, base_dir=nn_root1,
            excluded_subjects=tuple(DEFAULT_EXCLUDED_EYE), keep_first_n=keep))
    nn2_dur = _disk_cache(
        cache_dir, _cache_key("dur_nn2", nn2_clean_path, keep),
        lambda: build_duration_summaries_7plus(
            nn2_fix, max_fixations=8, base_dir=nn_root2,
            excluded_subjects=tuple(DEFAULT_EXCLUDED_EYE), keep_first_n=keep))

    # Prop-relevant summaries (Panels E, F).
    human_prop = _disk_cache(
        cache_dir, _cache_key("prop_human", human_clean_path, keep),
        lambda: _compute_prop_relevant_by_position_7plus(
            human_fix, max_fixations=8, keep_first_n=keep))
    nn1_prop = _disk_cache(
        cache_dir, _cache_key("prop_nn1", nn1_clean_path, keep),
        lambda: _compute_prop_relevant_by_position_7plus(
            nn1_fix, max_fixations=8, keep_first_n=keep))
    nn2_prop = _disk_cache(
        cache_dir, _cache_key("prop_nn2", nn2_clean_path, keep),
        lambda: _compute_prop_relevant_by_position_7plus(
            nn2_fix, max_fixations=8, keep_first_n=keep))

    # Valence summaries (Panels G, H).
    # For human panel G, we need recalled rewards merged first.
    from metarnn.lib.plot_NN_H_comparison import load_recalled_rewards  # noqa: E402
    def _human_fix_with_recalled() -> pd.DataFrame:
        fix = human_fix.copy()
        if "reward_recalled" not in fix.columns:
            recalled = load_recalled_rewards(_REPO_ROOT, excluded_subjects=tuple(DEFAULT_EXCLUDED_EYE))
            fix = fix.merge(
                recalled[["subject_id", "game", "image", "reward_recalled"]],
                on=["subject_id", "game", "image"], how="left")
        return fix

    def _compute_valence_pair(fix_df, denom, reward_col="reward"):
        return {
            dec: _compute_prop_positive_negative_relevant_by_decision(
                fix_df, max_fixations=int(max_fixations),
                keep_first_n=keep, decision=dec, denom=denom,
                reward_col=reward_col)
            for dec in [1, 2]
        }

    human_fix_recalled = _disk_cache(
        cache_dir, _cache_key("fix_human_recalled", human_clean_path),
        _human_fix_with_recalled)
    human_valence = _disk_cache(
        cache_dir, _cache_key("val_human", human_clean_path, keep, gh_denom, max_fixations),
        lambda: _compute_valence_pair(human_fix_recalled, gh_denom, reward_col="reward_recalled"))
    nn1_valence = _disk_cache(
        cache_dir, _cache_key("val_nn1", nn1_clean_path, keep, gh_denom, max_fixations),
        lambda: _compute_valence_pair(nn1_fix, gh_denom))
    nn2_valence = _disk_cache(
        cache_dir, _cache_key("val_nn2", nn2_clean_path, keep, gh_denom, max_fixations),
        lambda: _compute_valence_pair(nn2_fix, gh_denom))

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    sns.set_context("poster")
    with plt.rc_context({"font.family": "Arial", "axes.titlesize": 24, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24}):
        fig = plt.figure(figsize=(24, 12))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.55)

        gs_top = outer[0].subgridspec(1, 4, width_ratios=[0.85, 1.15, 1, 1], wspace=0.55)
        gs_bot = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.55)

        axA = fig.add_subplot(gs_top[0, 0])

        gsB = gs_top[0, 1].subgridspec(1, 2, wspace=0.10)
        axB_h = fig.add_subplot(gsB[0, 0])
        axB_n = fig.add_subplot(gsB[0, 1], sharey=axB_h)

        axC = fig.add_subplot(gs_bot[0, 0])
        axD = fig.add_subplot(gs_bot[0, 1])

        axE = fig.add_subplot(gs_top[0, 2])
        axF = fig.add_subplot(gs_top[0, 3])

        gsG = gs_bot[0, 2].subgridspec(2, 1, hspace=0.35)
        axG_take = fig.add_subplot(gsG[0, 0])
        axG_leave = fig.add_subplot(gsG[1, 0], sharex=axG_take)

        gsH = gs_bot[0, 3].subgridspec(2, 1, hspace=0.35)
        axH_take = fig.add_subplot(gsH[0, 0])
        axH_leave = fig.add_subplot(gsH[1, 0], sharex=axH_take)

        # --- Panel A: bar chart with three groups ---
        _panel_A_dual(axA, human_by_sub, nn1_by_sub, nn2_by_sub, label1=label1, label2=label2)


        # --- Panel B: cumtime, human left + dual NN right ---
        _panel_B_dual(
            axB_h, axB_n,
            human_fix=human_fix,
            nn1_fix=nn1_fix,
            nn2_fix=nn2_fix,
            max_fixations=max_fixations,
            cumtime_mode=cumtime_mode,
            label1=label1,
            label2=label2,
            hline_y=0.8,
        )
        axB_n.tick_params(left=False, labelleft=False)
        axB_h.set_yticks([0, 0.5, 1.0])
        axB_n.set_yticks([0, 0.5, 1.0])
        # Centered xlabel between B subplots, matching size/position of C/D xlabels.
        _b0 = axB_h.get_position()
        _b1 = axB_n.get_position()
        _xc = (_b0.x0 + _b1.x1) / 2.0
        # Use axC's xlabel y-position as reference.
        _yref = axE.get_position().y0 - 0.045
        fig.text(_xc, _yref, "Fixation number", ha="center", va="top", fontsize=28)

        # --- Panel C: human duration ---
        _panel_duration_by_relevance_7plus(
            axC,
            base_dir=_REPO_ROOT,
            title="Humans",
            y_label="Fixation Duration (s)",
            y_scale=1.0 / 1000.0,
            y_lim=(0.0, 1.0),
            max_fixations=8,
            keep_first_n=keep,
            show_grid=False,
            colors={"relevant": "#6fc7eb", "irrelevant": "#ba7caf"},
            precomputed_summary=human_dur,
        )
        _lg = axC.legend(frameon=True, fontsize=12, handlelength=1.5, loc="best",
                         handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
        _style_legend(_lg)
        axC.set_title("Humans", pad=31)

        # --- Panel D: dual NN duration ---
        _panel_duration_dual(
            axD,
            label1=label1,
            label2=label2,
            summary1=nn1_dur,
            summary2=nn2_dur,
            y_label="Fixation Duration (steps)",
            y_lim=(1.05, 1.3),
            keep_first_n=keep,
        )
        axD.set_title("RNN", pad=31)


        # --- Panel E: human prop relevant ---
        _panel_prop_relevant_fixated_by_position_7plus(
            axE,
            base_dir=_REPO_ROOT,
            title="Humans",
            max_fixations=8,
            keep_first_n=keep,
            y_label="Prop. Relevant Fixations",
            hline_y=0.5,
            line_color="black",
            y_lim=(0.2, 0.8),
            precomputed_summary=human_prop,
        )
        axE.set_title("Humans")

        # --- Panel F: dual NN prop relevant ---
        _panel_prop_relevant_dual(
            axF,
            label1=label1,
            label2=label2,
            summary1=nn1_prop,
            summary2=nn2_prop,
            keep_first_n=keep,
            y_lim=(0.2, 0.8),
        )
        axF.set_title("RNN")

        # --- Panel G: human valence (uses recalled reward) ---
        _gh_ylim = (0.1, 0.4)
        _panel_prop_positive_negative_relevant_by_decision(
            axG_take, axG_leave,
            base_dir=_REPO_ROOT,
            title="Humans",
            max_fixations=int(max_fixations),
            keep_first_n=keep,
            denom=gh_denom,
            y_lim=_gh_ylim,
            precomputed_summaries=human_valence,
        )


        _lg = axG_take.legend(frameon=True, fontsize=12, handlelength=1.5, loc="best",
                              handletextpad=0.5, labelspacing=0.3, facecolor="white", edgecolor="black")
        _style_legend(_lg)

        # Override G panel formatting for consistent layout.
        axG_take.set_title("Take", fontsize=14)
        axG_leave.set_title("Leave", fontsize=14)
        # Re-set xticklabels (imported function clears them via set_xticklabels([])
        # which affects the shared axis).
        _ticks_g = list(range(1, keep + 2))
        axG_leave.set_xticks(_ticks_g)
        axG_leave.set_xticklabels([str(i) for i in range(1, keep + 1)] + [f"{keep + 1}+"])
        axG_take.tick_params(labelbottom=False)
        # Shared y-axis label centered between take and leave subplots.
        axG_take.set_ylabel("Proportion of Fixations", fontsize=28)
        axG_take.yaxis.set_label_coords(-0.22, -0.15)
        axG_leave.set_ylabel("")

        # --- Panel H: dual NN valence ---
        _panel_valence_dual(
            axH_take, axH_leave,
            label1=label1,
            label2=label2,
            valence_summaries1=nn1_valence,
            valence_summaries2=nn2_valence,
            keep_first_n=keep,
            y_lim=_gh_ylim,
        )


        # Overall titles for G (Humans) and H (Network) panel groups.
        axG_take.text(0.5, 1.25, "Humans", transform=axG_take.transAxes,
                      ha='center', va='bottom', fontsize=24)
        axH_take.text(0.5, 1.25, "RNN", transform=axH_take.transAxes,
                      ha='center', va='bottom', fontsize=24)

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureNN_NN_comparison{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    print(f"Wrote {out_path}")

    # --- Fixation-number supplement figure ---
    _save_fixation_number_supplement(
        out_dir=out_dir,
        tag=tag,
        human_by_sub=human_by_sub,
        nn1_by_sub=nn1_by_sub,
        nn2_by_sub=nn2_by_sub,
        label1=label1,
        label2=label2,
    )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NN simulations (solid vs dashed) alongside human data.")
    parser.add_argument("--nn-root1", type=str, required=True, help="Path to first NN simulation root.")
    parser.add_argument("--nn-root2", type=str, required=True, help="Path to second NN simulation root.")
    parser.add_argument("--label1", type=str, default="NN1", help="Legend label for first simulation.")
    parser.add_argument("--label2", type=str, default="NN2", help="Legend label for second simulation.")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for the output filename.")
    parser.add_argument("--nn-tag1", type=str, required=True, help="Output tag for first NN simulation (used in revisits CSV naming).")
    parser.add_argument("--nn-tag2", type=str, required=True, help="Output tag for second NN simulation (used in revisits CSV naming).")
    parser.add_argument("--max-fixations", type=int, default=40, help="Max fixation number for cumtime curve.")
    parser.add_argument(
        "--cumtime-mode", type=str, default="all_trials", choices=["all_trials", "conditional"],
        help="How to average cumulative time across trials.",
    )
    parser.add_argument(
        "--gh-denom", type=str, default="all", choices=["relevant", "all"],
        help="Denominator for G/H panels.",
    )

    args = parser.parse_args()

    create_figure(
        nn_root1=Path(args.nn_root1).resolve(),
        nn_root2=Path(args.nn_root2).resolve(),
        tag=str(args.tag),
        nn_tag1=str(args.nn_tag1),
        nn_tag2=str(args.nn_tag2),
        label1=str(args.label1),
        label2=str(args.label2),
        max_fixations=int(args.max_fixations),
        cumtime_mode=str(args.cumtime_mode),
        gh_denom=str(args.gh_denom),
    )


if __name__ == "__main__":
    main()
