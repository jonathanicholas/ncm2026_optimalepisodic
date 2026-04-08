#!/usr/bin/env python3
"""Plot sweep transition analysis with chance-level correction (5x3 figure).

Example:
  conda run -n analysis python metarnn/lib/plot_NN_sweep_transitions.py \\
    --base-dir . --n-sims 1000 --tag human \\
    --out-dir output/next_fixation_gen
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.lib.analyze_choice_fixation_sweeps import (
    PANEL_ALL,
    PANEL_FIRST,
    PANEL_REVISIT,
    PANELS,
    TrialSeq,
    build_trial_sequences,
    list_subjects,
    load_choice_item_fixations,
    load_main_logfile,
    matrix_similarity,
    mean_ci95,
    mean_sem,
    nanmean_safe,
    nanstd_safe,
    row_normalize_counts,
    simulate_shuffle_seq,
    sweep_template_matrices,
    transition_counts_from_sequences,
    transition_proportion_in_runs_by_length,
)

# Style constants
PANEL_TITLES = {
    PANEL_ALL: "All fixations",
    PANEL_FIRST: "First fixations",
    PANEL_REVISIT: "Revisit fixations",
}

TEMPLATE_ORDER = ["bidirectional", "forward", "backward"]
TEMPLATE_SHORT = {"bidirectional": "Bi", "forward": "Fwd", "backward": "Bwd"}

MAX_RUN_BIN = 6


def _add_panel_label(ax: plt.Axes, label: str, *, dx: float = -55, dy: float = 14) -> None:
    ax.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        annotation_clip=False,
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fast vectorised helpers (avoid Python loops in the hot shuffle path)
# ---------------------------------------------------------------------------

def _fast_transition_counts(seqs_flat: np.ndarray, boundaries: np.ndarray, n_items: int = 6) -> np.ndarray:
    """Vectorised transition counting from a flat array of concatenated sequences.

    *seqs_flat*: 1-D int array of concatenated 0-indexed positions.
    *boundaries*: 1-D int array of cumulative sequence lengths (from ``np.cumsum``).
      The first sequence occupies indices ``[0, boundaries[0])``, the second
      ``[boundaries[0], boundaries[1])``, etc.

    Returns a (n_items, n_items) float counts matrix.
    """
    # Build "from" and "to" arrays, excluding cross-sequence boundaries.
    froms = np.delete(seqs_flat[:-1], boundaries[:-1] - 1)
    tos = np.delete(seqs_flat[1:], boundaries[:-1] - 1)
    counts = np.zeros((n_items, n_items), dtype=float)
    np.add.at(counts, (froms, tos), 1.0)
    return counts


def _fast_row_normalize(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a counts matrix; zero-sum rows become NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row_sums = counts.sum(axis=1, keepdims=True)
        out = np.where(row_sums > 0, counts / row_sums, np.nan)
    return out


def _fast_similarity(mat: np.ndarray, tmpl_flat: np.ndarray, tmpl_mask: np.ndarray,
                     tmpl_mean: float, tmpl_std: float) -> float:
    """Pearson r between a matrix and a pre-processed template.

    *tmpl_flat*, *tmpl_mask*, *tmpl_mean*, *tmpl_std* are precomputed for the
    template to avoid redundant work across simulations.
    """
    o = mat.ravel()
    mask = np.isfinite(o) & tmpl_mask
    n = int(mask.sum())
    if n < 2:
        return float("nan")
    o_vals = o[mask]
    t_vals = tmpl_flat[mask]
    o_mean = o_vals.mean()
    o_std = o_vals.std()
    if o_std == 0.0:
        return float("nan")
    # Use pre-computed template stats only when the mask matches perfectly.
    # Recompute template stats on the active mask.
    t_mean = t_vals.mean()
    t_std = t_vals.std()
    if t_std == 0.0:
        return float("nan")
    return float(np.dot(o_vals - o_mean, t_vals - t_mean) / (n * o_std * t_std))


def _prepare_seqs_for_fast(seqs: List[Tuple[int, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list of position tuples to flat array + boundary vector (0-indexed)."""
    lengths = np.array([len(s) for s in seqs], dtype=int)
    flat = np.concatenate([np.array(s, dtype=int) - 1 for s in seqs])  # 0-indexed
    boundaries = np.cumsum(lengths)
    return flat, boundaries


def _collapse_flat_per_trial(
    flat: np.ndarray, boundaries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse consecutive repeats *within* each trial segment.

    Used by the "collapse null shuffles" variant so that the null sequences
    have the same no-self-transition structure as the observed sequences.
    Returns (new_flat, new_boundaries).
    """
    seg_starts = np.concatenate([[0], boundaries[:-1]])
    seg_ends = boundaries
    new_segs: List[np.ndarray] = []
    new_lengths: List[int] = []
    for s, e in zip(seg_starts, seg_ends):
        seg = flat[int(s):int(e)]
        if seg.size == 0:
            new_lengths.append(0)
            continue
        keep = np.concatenate([[True], seg[1:] != seg[:-1]])
        collapsed = seg[keep]
        new_segs.append(collapsed)
        new_lengths.append(int(collapsed.size))
    if new_segs:
        new_flat = np.concatenate(new_segs)
    else:
        new_flat = np.array([], dtype=flat.dtype)
    new_boundaries = np.cumsum(np.asarray(new_lengths, dtype=int))
    return new_flat, new_boundaries


def _fast_signed_lags(flat: np.ndarray, boundaries: np.ndarray, n: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised circular signed lags, excluding cross-sequence boundaries.

    Returns (lags, lag_boundaries) where lags are the signed lags and
    lag_boundaries are cumulative lengths per sequence (of the lag arrays).
    """
    # All adjacent pairs
    froms = np.delete(flat[:-1], boundaries[:-1] - 1)
    tos = np.delete(flat[1:], boundaries[:-1] - 1)
    cw = (tos - froms) % n
    lags = np.where(cw <= n // 2, cw, -(n - cw))

    # Boundaries for per-trial lags (each trial has len-1 transitions)
    seq_lengths = np.diff(np.concatenate([[0], boundaries]))
    lag_lengths = np.maximum(seq_lengths - 1, 0)
    lag_boundaries = np.cumsum(lag_lengths)
    return lags, lag_boundaries


def _fast_transprop_in_runs(lags: np.ndarray, lag_boundaries: np.ndarray,
                            max_bin: int = 6) -> np.ndarray:
    """Fully vectorised transition-proportion-in-runs-by-length, averaged across trials.

    Returns a (max_bin,) array.
    """
    if lags.size == 0:
        return np.full(max_bin, np.nan)

    n_trials = len(lag_boundaries)
    starts = np.concatenate([[0], lag_boundaries[:-1]])
    ends = lag_boundaries
    seq_lens = ends - starts  # number of lags per trial

    # Assign each lag to its trial index.
    trial_ids = np.repeat(np.arange(n_trials, dtype=np.int32), seq_lens)

    # Drop zero lags (self-transitions).
    nonzero = lags != 0
    lags_nz = lags[nonzero]
    trial_ids_nz = trial_ids[nonzero]

    if lags_nz.size == 0:
        return np.full(max_bin, np.nan)

    # Count non-zero transitions per trial (denominator).
    n_trans_per_trial = np.bincount(trial_ids_nz, minlength=n_trials)

    # Identify adjacent transitions (|lag| == 1).
    is_adj = np.abs(lags_nz) == 1
    signs = np.sign(lags_nz)

    # Detect run boundaries: a new run starts when:
    #   - it's the first element, OR
    #   - trial changes, OR
    #   - not adjacent, OR
    #   - previous was not adjacent, OR
    #   - sign changed
    new_run = np.ones(len(lags_nz), dtype=bool)
    same_trial = trial_ids_nz[1:] == trial_ids_nz[:-1]
    continues = same_trial & is_adj[1:] & is_adj[:-1] & (signs[1:] == signs[:-1])
    new_run[1:] = ~continues

    # Assign run IDs.
    run_ids = np.cumsum(new_run) - 1

    # Keep only adjacent lags for run-length counting.
    adj_mask = is_adj
    adj_run_ids = run_ids[adj_mask]
    adj_trial_ids = trial_ids_nz[adj_mask]

    if adj_run_ids.size == 0:
        # No adjacent transitions at all — every trial has zero adj proportion.
        valid = n_trans_per_trial > 0
        if not np.any(valid):
            return np.full(max_bin, np.nan)
        return np.zeros(max_bin, dtype=float)

    # Unique adjacent run IDs, their counts (= run lengths), and trial mapping.
    unique_adj_runs, first_idx, run_lengths = np.unique(adj_run_ids, return_index=True, return_counts=True)
    run_trial = adj_trial_ids[first_idx]

    # Clip run lengths to max_bin.
    clipped = np.minimum(run_lengths, max_bin)

    # For each trial, accumulate transitions into bins.
    # Build a (n_trials, max_bin) matrix.
    trial_bins = np.zeros((n_trials, max_bin), dtype=float)
    bin_idx = clipped - 1  # 0-indexed bin
    np.add.at(trial_bins, (run_trial, bin_idx), run_lengths.astype(float))

    # Normalise each trial by its total non-zero transitions.
    valid_trials = n_trans_per_trial > 0
    if not np.any(valid_trials):
        return np.full(max_bin, np.nan)

    trial_props = np.zeros((n_trials, max_bin), dtype=float)
    trial_props[valid_trials] = trial_bins[valid_trials] / n_trans_per_trial[valid_trials, None]

    # Average across valid trials.
    return trial_props[valid_trials].mean(axis=0)


# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------

def compute_sweep_transition_data(
    base_dir: Path,
    buffer: int = 50,
    n_sims: int = 1000,
    seed: int = 123,
    exclude_subjects: Sequence[str] = ("107", "131"),
    collapse_null_shuffles: bool = True,
) -> Dict[str, Any]:
    """Load data and compute all quantities needed for the 5-row figure.

    Returns a dict with keys:
        obs_trans_by_panel      – mean observed 6×6 transition matrix per panel
        corrected_trans_by_panel – mean chance-corrected 6×6 matrix per panel
        templates               – dict of template matrices
        obs_similarity          – {panel: {template: array of per-subject r}}
        delta_similarity        – {panel: {template: array of per-subject Δr}}
        transprop_obs           – {panel: (n_subjects, n_bins) array}
        transprop_null          – {panel: (n_subjects, n_bins) array}
        run_bin_labels          – list of bin label strings
    """
    data_root = base_dir / "data"

    sids = list_subjects(data_root)
    exclude_set = set(str(s) for s in exclude_subjects)
    sids = [s for s in sids if s not in exclude_set]

    # Check that subjects have required files.
    valid_sids: List[str] = []
    for sid in sids:
        log_path = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        fix_path = data_root / sid / f"{sid}_fixations_df_original_buffer_{buffer}.csv"
        if not fix_path.exists():
            fix_path = data_root / sid / f"{sid}_fixations_df_original.csv"
        if log_path.exists() and fix_path.exists():
            valid_sids.append(sid)
    sids = sorted(valid_sids)

    if not sids:
        raise RuntimeError(f"No valid subjects found under {data_root}")

    templates = sweep_template_matrices(n_items=6)
    rng = np.random.default_rng(seed)

    run_bin_labels = [str(i) for i in range(1, MAX_RUN_BIN)] + [f"{MAX_RUN_BIN}+"]

    # Pre-flatten templates for fast similarity computation.
    tmpl_precomp: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]] = {}
    for tname in TEMPLATE_ORDER:
        t_flat = templates[tname].ravel().astype(float)
        t_mask = np.isfinite(t_flat)
        t_mean = float(t_flat[t_mask].mean()) if t_mask.any() else 0.0
        t_std = float(t_flat[t_mask].std()) if t_mask.any() else 0.0
        tmpl_precomp[tname] = (t_flat, t_mask, t_mean, t_std)

    # Accumulators
    subj_trans: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_null_trans: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    obs_similarity: Dict[str, Dict[str, List[float]]] = {p: {t: [] for t in TEMPLATE_ORDER} for p in PANELS}
    null_similarity: Dict[str, Dict[str, List[float]]] = {p: {t: [] for t in TEMPLATE_ORDER} for p in PANELS}
    subj_transprop_obs: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_transprop_null: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_ids_by_panel: Dict[str, List[str]] = {p: [] for p in PANELS}

    for i_sid, sid in enumerate(sids):
        print(f"  [{i_sid + 1}/{len(sids)}] {sid} ...", flush=True)

        log_path = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        fix_path = data_root / sid / f"{sid}_fixations_df_original_buffer_{buffer}.csv"
        if not fix_path.exists():
            fix_path = data_root / sid / f"{sid}_fixations_df_original.csv"

        log_df = load_main_logfile(log_path)
        fix_df = load_choice_item_fixations(fix_path)
        trials = build_trial_sequences(sid, log_df, fix_df)
        if not trials:
            continue

        panel_seqs: Dict[str, List[Tuple[int, ...]]] = {
            PANEL_ALL: [t.spatial_all for t in trials if t.spatial_all and len(t.spatial_all) >= 2],
            PANEL_FIRST: [t.spatial_first for t in trials if t.spatial_first and len(t.spatial_first) >= 2],
            PANEL_REVISIT: [t.spatial_revisit for t in trials if t.spatial_revisit and len(t.spatial_revisit) >= 2],
        }

        for panel in PANELS:
            seqs = panel_seqs[panel]
            if not seqs:
                continue

            subj_ids_by_panel[panel].append(sid)

            # Pre-compute flat arrays for this panel's sequences.
            flat, boundaries = _prepare_seqs_for_fast(seqs)
            seq_lengths = np.diff(np.concatenate([[0], boundaries]))

            # --- Observed transition matrix ---
            obs_counts = _fast_transition_counts(flat, boundaries, n_items=6)
            obs_mat = _fast_row_normalize(obs_counts)
            subj_trans[panel].append(obs_mat)

            # --- Observed similarity ---
            for tname in TEMPLATE_ORDER:
                t_flat, t_mask, t_mean, t_std = tmpl_precomp[tname]
                r = _fast_similarity(obs_mat, t_flat, t_mask, t_mean, t_std)
                obs_similarity[panel][tname].append(r)

            # --- Observed transition proportion in runs ---
            obs_lags, obs_lag_bounds = _fast_signed_lags(flat, boundaries, n=6)
            obs_transprop = _fast_transprop_in_runs(obs_lags, obs_lag_bounds, max_bin=MAX_RUN_BIN)
            subj_transprop_obs[panel].append(obs_transprop)

            # --- Shuffle null (vectorised) ---
            # Accumulate null transition matrices, similarity, and transprop.
            null_trans_sum = np.zeros((6, 6), dtype=float)
            null_trans_count = 0
            sim_r_accum: Dict[str, List[float]] = {t: [] for t in TEMPLATE_ORDER}
            sim_transprop_accum: List[np.ndarray] = []

            # Pre-compute shuffle segment boundaries (only segments with len > 1).
            seg_starts = np.concatenate([[0], boundaries[:-1]])
            seg_ends = boundaries.copy()
            seg_mask = (seg_ends - seg_starts) > 1
            shuf_starts = seg_starts[seg_mask].astype(int)
            shuf_ends = seg_ends[seg_mask].astype(int)

            for _ in range(n_sims):
                # Shuffle each trial in-place within the flat array.
                shuf_flat = flat.copy()
                for s, e in zip(shuf_starts, shuf_ends):
                    rng.shuffle(shuf_flat[s:e])

                # Option: collapse consecutive repeats within each trial
                # after shuffling, so the null sequences mirror the observed
                # pipeline (which collapses consecutive identical fixations
                # upstream). Without this, shuffled multisets with repeated
                # positions produce self-transitions that the observed never
                # has, contaminating both the transition-matrix and run-length
                # nulls.
                if collapse_null_shuffles:
                    sim_flat, sim_boundaries = _collapse_flat_per_trial(
                        shuf_flat, boundaries
                    )
                else:
                    sim_flat, sim_boundaries = shuf_flat, boundaries

                sim_counts = _fast_transition_counts(sim_flat, sim_boundaries, n_items=6)
                sim_mat = _fast_row_normalize(sim_counts)

                null_trans_sum += np.where(np.isfinite(sim_mat), sim_mat, 0.0)
                null_trans_count += 1

                for tname in TEMPLATE_ORDER:
                    t_flat, t_mask, t_mean, t_std = tmpl_precomp[tname]
                    sim_r_accum[tname].append(_fast_similarity(sim_mat, t_flat, t_mask, t_mean, t_std))

                sim_lags, sim_lag_bounds = _fast_signed_lags(sim_flat, sim_boundaries, n=6)
                sim_transprop_accum.append(
                    _fast_transprop_in_runs(sim_lags, sim_lag_bounds, max_bin=MAX_RUN_BIN)
                )

            # Mean null transition matrix
            if null_trans_count > 0:
                null_mean_mat = null_trans_sum / float(null_trans_count)
            else:
                null_mean_mat = np.full((6, 6), np.nan)
            subj_null_trans[panel].append(null_mean_mat)

            # Mean null similarity
            for tname in TEMPLATE_ORDER:
                vals = [v for v in sim_r_accum[tname] if np.isfinite(v)]
                null_similarity[panel][tname].append(float(np.mean(vals)) if vals else float("nan"))

            # Mean null transprop
            if sim_transprop_accum:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    subj_transprop_null[panel].append(np.nanmean(np.stack(sim_transprop_accum), axis=0))
            else:
                subj_transprop_null[panel].append(np.full((MAX_RUN_BIN,), np.nan))

    # --- Aggregate across subjects ---
    obs_trans_by_panel: Dict[str, np.ndarray] = {}
    corrected_trans_by_panel: Dict[str, np.ndarray] = {}
    for panel in PANELS:
        if subj_trans[panel]:
            obs_stack = np.stack(subj_trans[panel])
            null_stack = np.stack(subj_null_trans[panel])
            obs_trans_by_panel[panel] = np.nanmean(obs_stack, axis=0)
            corrected_trans_by_panel[panel] = np.nanmean(obs_stack - null_stack, axis=0)
        else:
            obs_trans_by_panel[panel] = np.full((6, 6), np.nan)
            corrected_trans_by_panel[panel] = np.full((6, 6), np.nan)

    delta_similarity: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in PANELS}
    obs_sim_arrays: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in PANELS}
    for panel in PANELS:
        for tname in TEMPLATE_ORDER:
            obs_arr = np.asarray(obs_similarity[panel][tname], dtype=float)
            null_arr = np.asarray(null_similarity[panel][tname], dtype=float)
            obs_sim_arrays[panel][tname] = obs_arr
            delta_similarity[panel][tname] = obs_arr - null_arr

    transprop_obs_stacked: Dict[str, np.ndarray] = {}
    transprop_null_stacked: Dict[str, np.ndarray] = {}
    for panel in PANELS:
        if subj_transprop_obs[panel]:
            transprop_obs_stacked[panel] = np.stack(subj_transprop_obs[panel])
        else:
            transprop_obs_stacked[panel] = np.empty((0, MAX_RUN_BIN))
        if subj_transprop_null[panel]:
            transprop_null_stacked[panel] = np.stack(subj_transprop_null[panel])
        else:
            transprop_null_stacked[panel] = np.empty((0, MAX_RUN_BIN))

    return {
        "obs_trans_by_panel": obs_trans_by_panel,
        "corrected_trans_by_panel": corrected_trans_by_panel,
        "templates": templates,
        "obs_similarity": obs_sim_arrays,
        "delta_similarity": delta_similarity,
        "transprop_obs": transprop_obs_stacked,
        "transprop_null": transprop_null_stacked,
        "run_bin_labels": run_bin_labels,
        "subject_ids": subj_ids_by_panel,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_heatmap_row(
    axes: Sequence[plt.Axes],
    panels: Sequence[str],
    mat_by_panel: Dict[str, np.ndarray],
    *,
    cmap: str = "viridis",
    center: Optional[float] = None,
    row_label: str = "",
) -> None:
    """Render one 6×6 heatmap per panel into *axes*."""
    n_items = 6
    # Determine global color limits.
    all_vals = []
    for panel in panels:
        m = mat_by_panel[panel]
        finite = m[np.isfinite(m)]
        if finite.size > 0:
            all_vals.append(finite)
    if all_vals:
        flat = np.concatenate(all_vals)
        if center is not None:
            absmax = max(abs(float(np.nanmin(flat))), abs(float(np.nanmax(flat))))
            absmax = max(absmax, 1e-6)
            vmin, vmax = -absmax, absmax
        else:
            vmin = 0.0
            vmax = max(1e-6, min(1.0, float(np.nanmax(flat))))
    else:
        vmin, vmax = 0.0, 1.0

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        mat = mat_by_panel[panel]
        mask = ~np.isfinite(mat)
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            square=True,
            cbar=(i == len(panels) - 1),
            cbar_kws={"shrink": 0.8} if i == len(panels) - 1 else None,
            mask=mask,
            xticklabels=[str(j) for j in range(1, n_items + 1)],
            yticklabels=[str(j) for j in range(1, n_items + 1)],
        )
        ax.set_title(PANEL_TITLES.get(panel, panel))
        ax.set_xlabel("Next position")
        if i == 0:
            ax.set_ylabel(f"{row_label}\nCurrent position" if row_label else "Current position")
        else:
            ax.set_ylabel("")
        ax.tick_params(length=0)


def _plot_similarity_scatter(
    axes: Sequence[plt.Axes],
    panels: Sequence[str],
    sim_by_panel: Dict[str, Dict[str, np.ndarray]],
    template_names: Sequence[str],
    *,
    ylabel: str = "Similarity (Pearson r)",
) -> None:
    """Scatter + mean ± 95% CI for each template per panel."""
    xs = np.arange(len(template_names), dtype=float)

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        for j, tname in enumerate(template_names):
            y = np.asarray(sim_by_panel[panel][tname], dtype=float)
            y = y[np.isfinite(y)]

            if len(y) > 0:
                jitter = (np.random.default_rng(0).random(len(y)) - 0.5) * 0.18
                ax.scatter(
                    np.full(len(y), xs[j]) + jitter,
                    y,
                    color="0.45",
                    s=14 ** 2,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=3,
                )

            m = nanmean_safe(y)
            sd = nanstd_safe(y, ddof=1)
            n = int(np.sum(np.isfinite(y)))
            se = float(sd / np.sqrt(n)) if np.isfinite(sd) and n >= 2 else np.nan
            ci = 1.96 * se if np.isfinite(se) else np.nan
            ax.scatter([xs[j]], [m], color="black", s=14 ** 2, zorder=5, edgecolor="black", linewidth=2.5)
            if np.isfinite(ci):
                ax.errorbar([xs[j]], [m], yerr=[ci], color="black", lw=2, capsize=4, zorder=5)

        ax.axhline(0, color="gray", lw=1, alpha=0.5)
        ax.set_title(PANEL_TITLES.get(panel, panel))
        ax.set_xticks(xs)
        ax.set_xticklabels([TEMPLATE_SHORT.get(n, n) for n in template_names])
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")


def _plot_runlength_delta_bars(
    axes: Sequence[plt.Axes],
    panels: Sequence[str],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    bin_labels: Sequence[str],
) -> None:
    """Delta (observed − null) bar plot per run-length bin."""
    k = len(bin_labels)
    xs = np.arange(k, dtype=float)

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, k))), dtype=float)
        n = min(obs.shape[0], nul.shape[0])
        if n == 0:
            mean_val = np.full((k,), np.nan)
            ci = np.full((k,), np.nan)
        else:
            delta = obs[:n, :] - nul[:n, :]
            mean_val, ci = mean_ci95(delta, axis=0)

        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar(xs, mean_val, color="black", alpha=0.85, width=0.8)
        ax.errorbar(xs, mean_val, yerr=ci, fmt="none", ecolor="black", lw=2, capsize=4)
        ax.set_title(PANEL_TITLES.get(panel, panel))
        ax.set_xticks(xs)
        ax.set_xticklabels(list(bin_labels))
        ax.set_xlabel("Run length")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("\u0394 fraction of transitions")
        else:
            ax.set_ylabel("")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def create_sweep_transitions_figure(
    base_dir: Path,
    out_dir: Path,
    tag: str = "",
    buffer: int = 50,
    n_sims: int = 1000,
    seed: int = 123,
    exclude_subjects: Sequence[str] = ("107", "131"),
) -> Path:
    _ensure_dir(out_dir)

    print("Computing sweep transition data ...", flush=True)
    data = compute_sweep_transition_data(
        base_dir=base_dir,
        buffer=buffer,
        n_sims=n_sims,
        seed=seed,
        exclude_subjects=exclude_subjects,
    )
    print("Done. Building figure ...", flush=True)

    panels = PANELS
    n_panels = len(panels)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2

    with plt.rc_context(
        {
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    ):
        fig = plt.figure(figsize=(18, 28))
        gs = fig.add_gridspec(5, n_panels, hspace=0.45, wspace=0.40)

        # Row A: Observed transition heatmaps
        axes_a = [fig.add_subplot(gs[0, c]) for c in range(n_panels)]
        _plot_heatmap_row(
            axes_a,
            panels,
            data["obs_trans_by_panel"],
            cmap="viridis",
            row_label="Observed",
        )
        _add_panel_label(axes_a[0], "A", dx=-85)

        # Row B: Chance-corrected transition heatmaps
        axes_b = [fig.add_subplot(gs[1, c]) for c in range(n_panels)]
        _plot_heatmap_row(
            axes_b,
            panels,
            data["corrected_trans_by_panel"],
            cmap="RdBu_r",
            center=0.0,
            row_label="Obs \u2212 Chance",
        )
        _add_panel_label(axes_b[0], "B", dx=-85)

        # Row C: Similarity to templates
        axes_c = [fig.add_subplot(gs[2, c], sharey=None) for c in range(n_panels)]
        _plot_similarity_scatter(
            axes_c,
            panels,
            data["obs_similarity"],
            TEMPLATE_ORDER,
            ylabel="Similarity (Pearson r)",
        )
        # Share y-axis across Row C
        ylims_c = []
        for ax in axes_c:
            ylims_c.extend(ax.get_ylim())
        ymin_c, ymax_c = min(ylims_c), max(ylims_c)
        for ax in axes_c:
            ax.set_ylim(ymin_c, ymax_c)
        _add_panel_label(axes_c[0], "C", dx=-85)

        # Row D: Chance-corrected similarity
        axes_d = [fig.add_subplot(gs[3, c], sharey=None) for c in range(n_panels)]
        _plot_similarity_scatter(
            axes_d,
            panels,
            data["delta_similarity"],
            TEMPLATE_ORDER,
            ylabel="\u0394 Similarity (Obs \u2212 Chance)",
        )
        ylims_d = []
        for ax in axes_d:
            ylims_d.extend(ax.get_ylim())
        ymin_d, ymax_d = min(ylims_d), max(ylims_d)
        for ax in axes_d:
            ax.set_ylim(ymin_d, ymax_d)
        _add_panel_label(axes_d[0], "D", dx=-85)

        # Row E: Transitions in runs by length delta bars
        axes_e = [fig.add_subplot(gs[4, c]) for c in range(n_panels)]
        _plot_runlength_delta_bars(
            axes_e,
            panels,
            data["transprop_obs"],
            data["transprop_null"],
            data["run_bin_labels"],
        )
        # Share y-axis across Row E
        ylims_e = []
        for ax in axes_e:
            ylims_e.extend(ax.get_ylim())
        ymin_e, ymax_e = min(ylims_e), max(ylims_e)
        for ax in axes_e:
            ax.set_ylim(ymin_e, ymax_e)
        _add_panel_label(axes_e[0], "E", dx=-85)

        suffix = f"_{tag}" if tag else ""
        out_path = out_dir / f"FigureSweepTransitions{suffix}.pdf"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # Sidecar metadata for reproducibility.
    meta_path = out_dir / f"FigureSweepTransitions{suffix}.meta.txt"
    meta_path.write_text(
        "\n".join(
            [
                f"base_dir={base_dir}",
                f"out_dir={out_dir}",
                f"tag={tag}",
                f"buffer={buffer}",
                f"n_sims={n_sims}",
                f"seed={seed}",
                f"exclude_subjects={list(exclude_subjects)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sweep transition analysis with chance-level correction."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(_REPO_ROOT),
        help="Project base directory (default: repository root).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PDF (default: <base-dir>/output/next_fixation_gen/).",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix for output filename.")
    parser.add_argument("--buffer", type=int, default=50, help="ROI buffer size.")
    parser.add_argument("--n-sims", type=int, default=1000, help="Shuffle iterations per subject.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["107", "131"],
        help="Subject IDs to exclude (default: 107 131).",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "output" / "next_fixation_gen"

    create_sweep_transitions_figure(
        base_dir=base_dir,
        out_dir=out_dir,
        tag=args.tag,
        buffer=args.buffer,
        n_sims=args.n_sims,
        seed=args.seed,
        exclude_subjects=args.exclude_subjects,
    )


if __name__ == "__main__":
    main()
