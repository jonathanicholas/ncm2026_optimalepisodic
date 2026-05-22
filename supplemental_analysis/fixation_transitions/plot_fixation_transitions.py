#!/usr/bin/env python3
"""Fixation-transition structure: humans vs. prior-memory network (Figure S6).

A 2 row x 2 column figure (Row 1 = Human, Row 2 = Network):
  Column 1: Bidirectional template + observed transition heatmaps (left),
            bidirectional delta-similarity scatter (right).
  Column 2: Delta fraction of transitions by run length (all fixations).

Sweep transition data is cached to disk so that subsequent runs skip the
expensive shuffle computation.

Example
-------
conda run -n analysis python \
  supplemental_analysis/fixation_transitions/plot_fixation_transitions.py \
  --nn-root metarnn/simulations/human_like_04_04_input5 \
  --tag 04_04_input5 \
  --n-sims 1000 \
  --no-show
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_choice_fixation_sweeps import (  # noqa: E402
    PANEL_ALL,
    list_subjects,
    mean_ci95,
    nanmean_safe,
    sweep_template_matrices,
)
from metarnn.lib.plot_NN_sweep_transitions import (  # noqa: E402
    _ensure_dir,
    compute_sweep_transition_data,
)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_key(
    base_dir: Path, buffer: int, n_sims: int, seed: int,
    exclude_subjects: Tuple[str, ...], collapse_null_shuffles: bool = True,
) -> str:
    """Deterministic MD5-based cache key from computation parameters."""
    raw = f"{base_dir.resolve()}|{buffer}|{n_sims}|{seed}|{sorted(exclude_subjects)}"
    if collapse_null_shuffles:
        raw += "|collapseNull=1"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _sweep_cache_path(base_dir: Path, cache_key: str) -> Path:
    return base_dir / "output" / "next_fixation_gen" / "cache" / f"sweep_data_{cache_key}.pkl"


def _derive_subject_ids(
    base_dir: Path,
    *,
    buffer: int = 50,
    exclude_subjects: Tuple[str, ...] = (),
    data: Dict[str, Any],
) -> Dict[str, list]:
    """Re-derive subject IDs for cached sweep data that lacks them."""
    data_root = base_dir / "data"
    sids = list_subjects(data_root)
    exclude_set = set(str(s) for s in exclude_subjects)
    sids = [s for s in sids if s not in exclude_set]

    valid_sids: list = []
    for sid in sids:
        log_path = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        fix_path = data_root / sid / f"{sid}_fixations_df_original_buffer_{buffer}.csv"
        if not fix_path.exists():
            fix_path = data_root / sid / f"{sid}_fixations_df_original.csv"
        if log_path.exists() and fix_path.exists():
            valid_sids.append(sid)
    sids = sorted(valid_sids)

    # Validate against array shapes for PANEL_ALL.
    n_sweep = data["transprop_obs"][PANEL_ALL].shape[0]
    if len(sids) != n_sweep:
        print(
            f"  WARNING: derived subject count ({len(sids)}) != "
            f"sweep array size ({n_sweep}) for PANEL_ALL",
            flush=True,
        )
    return {PANEL_ALL: sids}


def _load_or_compute_sweep(
    base_dir: Path,
    *,
    buffer: int = 50,
    n_sims: int = 1000,
    seed: int = 123,
    exclude_subjects: Tuple[str, ...] = ("107", "131"),
    label: str = "",
    collapse_null_shuffles: bool = True,
) -> Dict[str, Any]:
    """Load sweep data from cache, or compute and cache it."""
    key = _cache_key(
        base_dir, buffer, n_sims, seed, exclude_subjects,
        collapse_null_shuffles=collapse_null_shuffles,
    )
    cache_path = _sweep_cache_path(base_dir, key)

    if cache_path.exists():
        print(f"Loading cached {label} sweep data from {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        # Verify parameters match
        meta = data.get("_cache_meta", {})
        if (
            meta.get("n_sims") == n_sims
            and meta.get("seed") == seed
            and meta.get("buffer") == buffer
            and tuple(meta.get("exclude_subjects", ())) == tuple(exclude_subjects)
            and bool(meta.get("collapse_null_shuffles", False)) == bool(collapse_null_shuffles)
        ):
            # Backfill subject_ids if missing from older cache files.
            if "subject_ids" not in data:
                data["subject_ids"] = _derive_subject_ids(
                    base_dir, buffer=buffer,
                    exclude_subjects=exclude_subjects, data=data,
                )
            return data
        print("  Cache parameters mismatch — recomputing.", flush=True)

    print(f"Computing {label} sweep transition data ...", flush=True)
    data = compute_sweep_transition_data(
        base_dir=base_dir,
        buffer=buffer,
        n_sims=n_sims,
        seed=seed,
        exclude_subjects=exclude_subjects,
        collapse_null_shuffles=collapse_null_shuffles,
    )

    # Attach metadata and save
    data["_cache_meta"] = {
        "base_dir": str(base_dir.resolve()),
        "buffer": buffer,
        "n_sims": n_sims,
        "seed": seed,
        "exclude_subjects": list(exclude_subjects),
        "collapse_null_shuffles": bool(collapse_null_shuffles),
    }
    _ensure_dir(cache_path.parent)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {cache_path}", flush=True)

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sem(x: np.ndarray) -> float:
    """Standard error of the mean."""
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _save_stats_csvs(
    *,
    human_sweep: Dict[str, Any],
    nn_sweep: Dict[str, Any],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Save per-subject CSVs for statistical testing in R."""
    stats_dir = out_dir / "stats"
    _ensure_dir(stats_dir)
    suffix = f"_{tag}" if tag else ""

    # --- CSV 1: Delta similarity by template ---
    template_names = ["bidirectional", "forward", "backward"]
    dsim_rows: list = []
    for group_label, sweep in [("human", human_sweep), ("nn", nn_sweep)]:
        sids = sweep["subject_ids"][PANEL_ALL]
        for tname in template_names:
            delta_arr = sweep["delta_similarity"][PANEL_ALL][tname]
            for i, sid in enumerate(sids):
                dsim_rows.append({
                    "subject": sid,
                    "group": group_label,
                    "template": tname,
                    "delta_similarity": float(delta_arr[i]),
                })
    dsim_df = pd.DataFrame(dsim_rows)
    dsim_path = stats_dir / f"delta_similarity_subject{suffix}.csv"
    dsim_df.to_csv(dsim_path, index=False)
    print(f"Saved: {dsim_path} ({len(dsim_df)} rows)")

    # --- CSV 2: Sequence length delta ---
    seqlen_rows: list = []
    for group_label, sweep in [("human", human_sweep), ("nn", nn_sweep)]:
        sids = sweep["subject_ids"][PANEL_ALL]
        obs = np.asarray(sweep["transprop_obs"][PANEL_ALL], dtype=float)
        nul = np.asarray(sweep["transprop_null"][PANEL_ALL], dtype=float)
        delta = obs - nul
        bin_labels = sweep["run_bin_labels"]
        for i, sid in enumerate(sids):
            for j, bl in enumerate(bin_labels):
                seqlen_rows.append({
                    "subject": sid,
                    "group": group_label,
                    "seq_length": bl,
                    "delta_proportion": float(delta[i, j]),
                })
    seqlen_df = pd.DataFrame(seqlen_rows)
    seqlen_path = stats_dir / f"seq_length_delta_subject{suffix}.csv"
    seqlen_df.to_csv(seqlen_path, index=False)
    print(f"Saved: {seqlen_path} ({len(seqlen_df)} rows)")


# ---------------------------------------------------------------------------
# Column 1: Heatmap + bidirectional delta-similarity
# ---------------------------------------------------------------------------

def _plot_single_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    cmap="viridis",
    title: str = "",
    cbar: bool = False,
    cbar_ax: Optional[plt.Axes] = None,
    cbar_kws: Optional[Dict[str, Any]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    mask_diagonal: bool = False,
    border: bool = True,
    cell_lines: bool = True,
) -> None:
    """Plot a single 6x6 heatmap.

    If ``mask_diagonal=True``, the main diagonal is treated as missing and
    rendered as the axes background (white) rather than a colored cell.
    When ``border`` is True, a black box is drawn around the heatmap; when
    ``cell_lines`` is True, each cell gets a thin outline.
    """
    n_items = 6

    # Cells to hide: NaN inputs plus (optionally) the main diagonal.
    mask = ~np.isfinite(mat)
    if mask_diagonal:
        mask = mask | np.eye(mat.shape[0], dtype=bool)

    if vmin is None:
        finite = mat[np.isfinite(mat) & ~(mask_diagonal & np.eye(mat.shape[0], dtype=bool))]
        vmin = 0.0
        vmax = max(1e-6, float(np.nanmax(finite))) if finite.size > 0 else 1.0

    if cbar_kws is None:
        cbar_kws = {"shrink": 0.8} if (cbar and cbar_ax is None) else None

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        mask=mask,
        linewidths=1.5 if cell_lines else 0,
        linecolor="black" if cell_lines else "none",
        xticklabels=[str(j) for j in range(1, n_items + 1)],
        yticklabels=[str(j) for j in range(1, n_items + 1)],
    )
    if mask_diagonal:
        for i in range(mat.shape[0]):
            ax.add_patch(plt.Rectangle(
                (i, i), 1, 1,
                facecolor="white",
                hatch="/////",
                edgecolor="black",
                linewidth=1.5,
                zorder=3,
            ))
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    if cbar and cbar_ax is not None:
        for spine in cbar_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    if title:
        ax.set_title(title, fontsize=22)
    ax.set_xlabel("Next pos." if show_xlabel else "", fontsize=14)
    ax.set_ylabel("Current pos." if show_ylabel else "", fontsize=14)
    ax.tick_params(length=0)


def _plot_single_delta_similarity(
    ax: plt.Axes,
    delta_vals: np.ndarray,
    *,
    ylabel: str = "Similarity above chance",
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    """Bar + strip of per-subject bidirectional delta-similarity."""
    y = np.asarray(delta_vals, dtype=float)
    y = y[np.isfinite(y)]

    rng = np.random.default_rng(0)
    _gray = ".7"
    _bar_w = 0.5

    m = float(nanmean_safe(y))
    sem_val = _sem(y)

    # Gray fill bar (behind dots)
    ax.bar(0, m, width=_bar_w,
           color=_gray, edgecolor="none", linewidth=0, zorder=2)
    # Black outline bar (on top of dots)
    ax.bar(0, m, width=_bar_w,
           color="none", edgecolor="black", linewidth=1.5, zorder=4)

    # Individual subject dots
    if len(y) > 0:
        jitter = rng.uniform(-0.08, 0.08, size=len(y))
        for i in range(len(y)):
            ax.scatter(
                jitter[i], y[i],
                s=6 ** 2,
                facecolor=(1, 1, 1, 0.5),
                edgecolor=_gray,
                linewidth=1,
                zorder=3,
            )

    # SEM error bar
    ax.errorbar(
        0, m, yerr=sem_val,
        fmt="none", ecolor="black",
        linewidth=1.5, capsize=0, zorder=5,
    )

    ax.set_xticks([])
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)


# ---------------------------------------------------------------------------
# Column 2: Run-length delta bars
# ---------------------------------------------------------------------------

def _plot_single_runlength_delta(
    ax: plt.Axes,
    transprop_obs_all: np.ndarray,
    transprop_null_all: np.ndarray,
    bin_labels: Sequence[str],
    *,
    ylabel: str = "Proportion above chance",
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
) -> None:
    """Delta (observed - null) bar plot per run-length bin for PANEL_ALL."""
    k = len(bin_labels)
    xs = np.arange(k, dtype=float)
    _gray = ".7"
    _bar_w = 0.8

    obs = np.asarray(transprop_obs_all, dtype=float)
    nul = np.asarray(transprop_null_all, dtype=float)
    n = min(obs.shape[0], nul.shape[0])

    if n == 0:
        mean_val = np.full(k, np.nan)
        ci = np.full(k, np.nan)
    else:
        delta = obs[:n, :] - nul[:n, :]
        mean_val, ci = mean_ci95(delta, axis=0)

    # Gray fill bars (behind)
    ax.bar(xs, mean_val, _bar_w,
           color=_gray, edgecolor="none", linewidth=0, zorder=2)
    # Black outline bars (on top)
    ax.bar(xs, mean_val, _bar_w,
           color="none", edgecolor="black", linewidth=1.5, zorder=4)
    # Error bars
    ax.errorbar(xs, mean_val, yerr=ci,
                fmt="none", ecolor="black", capsize=0, linewidth=1.5, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(bin_labels))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def create_figure(
    *,
    nn_root: Path,
    tag: str = "",
    n_sims: int = 1000,
    seed: int = 123,
    show: bool = True,
    out_dir: Optional[Path] = None,
    collapse_null_shuffles: bool = True,
) -> Path:
    if out_dir is None:
        out_dir = nn_root / "output" / "next_fixation_gen"
    _ensure_dir(out_dir)

    # When opting out of the (now default) collapsed null, append a suffix
    # to the tag so outputs don't overwrite the default-null run.
    if (not collapse_null_shuffles) and "uncollapsedNull" not in tag:
        tag = f"{tag}_uncollapsedNull" if tag else "uncollapsedNull"

    # ------------------------------------------------------------------
    # Step 1: Compute (or load cached) sweep transition data
    # ------------------------------------------------------------------
    human_sweep = _load_or_compute_sweep(
        _REPO_ROOT,
        buffer=50, n_sims=n_sims, seed=seed,
        exclude_subjects=("107", "131"),
        label="human",
        collapse_null_shuffles=collapse_null_shuffles,
    )

    nn_sweep = _load_or_compute_sweep(
        nn_root,
        buffer=50, n_sims=n_sims, seed=seed,
        exclude_subjects=(),
        label="NN",
        collapse_null_shuffles=collapse_null_shuffles,
    )

    # ------------------------------------------------------------------
    # Step 2: Get bidirectional template
    # ------------------------------------------------------------------
    templates = sweep_template_matrices(n_items=6)
    bi_template = templates["bidirectional"]

    # ------------------------------------------------------------------
    # Step 3: Save per-subject CSVs for statistical testing
    # ------------------------------------------------------------------
    _save_stats_csvs(
        human_sweep=human_sweep,
        nn_sweep=nn_sweep,
        out_dir=out_dir,
        tag=tag,
    )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    print("Building figure ...", flush=True)

    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context({
        "font.family": "Arial",
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "hatch.linewidth": 1.5,
    }):
        fig = plt.figure(figsize=(15.5, 12))
        # Outer grid: one cell per row, each holding [heatmaps+delta-similarity
        # | run-length].
        gs_outer = fig.add_gridspec(2, 1, hspace=0.45)

        # Heatmap color scale: per-row sequential range [vmin, vmax] fitted
        # to the *off-diagonal* observed cells. The diagonal is always 0 by
        # construction (sequences are pre-collapsed), so including it in the
        # scale compresses the color gradient onto the top of the colormap
        # and washes out the off-diagonal pattern. By setting vmin to the
        # off-diagonal minimum instead of 0, the full colormap range is
        # used for the actual data variation. The diagonal cells clip to
        # the darkest color, and the template's 0.5 neighbor cells clip to
        # the brightest — both clear visually.
        def _obs_offdiag_range(sweep_data: Dict[str, Any]) -> Tuple[float, float]:
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

        hm_vmin_human, hm_vmax_human = _obs_offdiag_range(human_sweep)
        hm_vmin_nn, hm_vmax_nn = _obs_offdiag_range(nn_sweep)
        hm_cmap = "viridis"

        row_configs = [
            {
                "label": "Humans",
                "sweep": human_sweep,
                "hm_vmin": hm_vmin_human,
                "hm_vmax": hm_vmax_human,
                "hm_cmap": hm_cmap,
                "ds_ylim": (0, 0.6),
                "ds_yticks": [0, 0.2, 0.4, 0.6],
                "rl_ylim": (0, 0.06),
                "rl_yticks": [0, 0.03, 0.06],
            },
            {
                "label": "Network",
                "sweep": nn_sweep,
                "hm_vmin": hm_vmin_nn,
                "hm_vmax": hm_vmax_nn,
                "hm_cmap": hm_cmap,
                "ds_ylim": (0, 0.3),
                "ds_yticks": [0, 0.1, 0.2, 0.3],
                "rl_ylim": (0, 0.02),
                "rl_yticks": [0, 0.01, 0.02],
            },
        ]

        for r, cfg in enumerate(row_configs):
            sweep = cfg["sweep"]

            # Row layout: [heatmaps+delta-similarity | run-length].
            gs_right = gs_outer[r, 0].subgridspec(
                1, 2,
                wspace=0.45,
                width_ratios=[2.25, 1.2],
            )

            # --- Column 1: Compound panel (heatmaps + delta similarity) ---
            gs_col2 = gs_right[0, 0].subgridspec(
                2, 2,
                wspace=0.05,
                hspace=0.55,
                width_ratios=[1, 0.3],
            )
            ax_template = fig.add_subplot(gs_col2[0, 0])
            ax_obs_trans = fig.add_subplot(gs_col2[1, 0])
            ax_delta_sim = fig.add_subplot(gs_col2[:, 1])

            # Template heatmap (top-left) — viridis binary reference.
            # Diagonal cells are rendered solid black to mark that
            # self-transitions are excluded by construction.
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

            # Observed transition heatmap (bottom-left): row-normalized
            # conditional transition probabilities. The diagonal is always
            # zero by construction (sequences are pre-collapsed) and is
            # masked out (rendered white) so the color scale can fit the
            # off-diagonal variation without being pulled toward zero.
            # A colorbar is attached to the right of the observed axes.
            obs_mat = sweep["obs_trans_by_panel"][PANEL_ALL]
            divider = make_axes_locatable(ax_obs_trans)
            ax_cbar = divider.append_axes("right", size="5%", pad=0.08)
            _plot_single_heatmap(
                ax_obs_trans, obs_mat,
                cmap=cfg["hm_cmap"],
                title="Observed",
                vmin=cfg["hm_vmin"], vmax=cfg["hm_vmax"],
                mask_diagonal=True,
                cbar=True,
                cbar_ax=ax_cbar,
            )
            # Colorbar ticks at min / middle / max, two decimal places.
            cb_vmin = cfg["hm_vmin"]
            cb_vmax = cfg["hm_vmax"]
            cb_mid = (cb_vmin + cb_vmax) / 2
            ax_cbar.set_yticks([cb_vmin, cb_mid, cb_vmax])
            ax_cbar.set_yticklabels([
                f"{cb_vmin:.2f}", f"{cb_mid:.2f}", f"{cb_vmax:.2f}",
            ])
            ax_cbar.tick_params(labelsize=14)

            # Bidirectional delta-similarity (right, full height)
            delta_bi = sweep["delta_similarity"][PANEL_ALL]["bidirectional"]
            _plot_single_delta_similarity(
                ax_delta_sim, delta_bi,
                ylim=cfg["ds_ylim"], yticks=cfg["ds_yticks"],
            )

            # --- Column 2: Run-Length Delta Bars ---
            ax_col3 = fig.add_subplot(gs_right[0, 1])
            _plot_single_runlength_delta(
                ax_col3,
                sweep["transprop_obs"][PANEL_ALL],
                sweep["transprop_null"][PANEL_ALL],
                sweep["run_bin_labels"],
                ylim=cfg["rl_ylim"], yticks=cfg["rl_yticks"],
            )

            # Store axes for row title placement after layout
            cfg["_ax_left"] = ax_template
            cfg["_ax_right"] = ax_col3

        # Force layout so get_position() returns final coords
        fig.canvas.draw()

        for cfg in row_configs:
            pos_left = cfg["_ax_left"].get_position()
            pos_right = cfg["_ax_right"].get_position()
            # Center over the full row extent including labels
            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_top = max(pos_left.y1, pos_right.y1) + 0.04
            fig.text(
                x_center, y_top, cfg["label"],
                fontsize=26, fontweight="normal",
                ha="center", va="bottom",
                transform=fig.transFigure,
            )

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureFixationTransitions{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixation-transition structure: humans vs. network (Figure S6)."
    )
    parser.add_argument(
        "--nn-root", type=str, required=True,
        help="Path to NN simulation directory.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Output filename suffix.",
    )
    parser.add_argument(
        "--n-sims", type=int, default=1000,
        help="Number of shuffle simulations per subject.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display interactive plots.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--no-collapse-null-shuffles", action="store_true",
        help=(
            "Opt out of collapsing consecutive repeats in shuffled null "
            "sequences. By default, the null collapses consecutive repeats "
            "so it mirrors the observed pipeline (no self-transitions). "
            "When this flag is set, outputs get an '_uncollapsedNull' "
            "suffix so they don't overwrite the default run."
        ),
    )

    args = parser.parse_args()

    create_figure(
        nn_root=Path(args.nn_root).resolve(),
        tag=str(args.tag),
        n_sims=int(args.n_sims),
        show=(not bool(args.no_show)),
        out_dir=Path(args.out_dir).resolve() if args.out_dir else None,
        collapse_null_shuffles=(not bool(args.no_collapse_null_shuffles)),
    )


if __name__ == "__main__":
    main()
