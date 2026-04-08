"""Analyze whether choice fixations sweep around the spatial circle.

This analysis asks whether, during decision-making, participants' fixations follow
sequential paths around the 6 spatial locations (e.g., 1→2→3→4→5→6) or the reverse.

We compute several per-trial sweep metrics on spatial-position sequences derived
from item fixations during the choice phase, for two panels:
  - all: all item fixations (after collapsing consecutive same-item fixations)
  - first: the first time each item is fixated (unique sequence)

Chance baselines are computed via a within-trial shuffle null that preserves the
multiset of visited spatial positions (and sequence length) but destroys order.

Inputs (per subject):
  - Behavioral logfile: data/<SUBID>/<SUBID>_MAIN_logfile_7.csv
  - Fixations: data/<SUBID>/<SUBID>_fixations_df_original_buffer_<BUFFER>.csv

Outputs:
  - CSV summary to output/
  - Figures to figures/

Notes:
  - Eyetracking analyses exclude subjects 107 and 131 by default.
  - Raw data files are never modified.
"""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2
sns.set_style("ticks")


PANEL_ALL = "all"
PANEL_FIRST = "first"
PANEL_REVISIT = "revisit"
PANELS = [PANEL_ALL, PANEL_FIRST, PANEL_REVISIT]

DOMAIN_SPATIAL = "spatial"


def row_normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Row-normalize transition counts into probabilities.

    Rows with zero total are set to NaN.
    """
    counts = np.asarray(counts, dtype=float)
    out = np.full_like(counts, np.nan, dtype=float)
    row_sums = counts.sum(axis=1)
    for i in range(counts.shape[0]):
        if np.isfinite(row_sums[i]) and row_sums[i] > 0:
            out[i, :] = counts[i, :] / row_sums[i]
    return out


def transition_counts_from_sequences(seqs: Sequence[Sequence[int]], n_items: int = 6) -> np.ndarray:
    """Compute n_items x n_items transition counts from position sequences (1..n_items)."""
    counts = np.zeros((n_items, n_items), dtype=float)
    for seq in seqs:
        if seq is None or len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if not (1 <= int(a) <= n_items and 1 <= int(b) <= n_items):
                continue
            counts[int(a) - 1, int(b) - 1] += 1.0
    return counts


def sweep_true_transition_matrix(n_items: int = 6, p_cw: float = 0.5) -> np.ndarray:
    """Idealized sweep transition matrix on a ring.

    From each position i, next is (i+1) with prob p_cw and (i-1) with prob (1-p_cw).
    Indices are 0..n_items-1.
    """
    p_cw = float(p_cw)
    p_cw = min(1.0, max(0.0, p_cw))
    T = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        T[i, (i + 1) % n_items] = p_cw
        T[i, (i - 1) % n_items] = 1.0 - p_cw
    return T


def stay_true_transition_matrix(n_items: int = 6) -> np.ndarray:
    """No-movement transition matrix (identity)."""
    return np.eye(n_items, dtype=float)


def neighbor_confusion_matrix(n_items: int = 6, eps: float = 0.1) -> np.ndarray:
    """Measurement/label noise: mislabel to adjacent neighbors on the ring.

    P(obs=i | true=i) = 1 - 2*eps
    P(obs=i±1 | true=i) = eps
    """
    eps = float(eps)
    eps = max(0.0, min(0.49, eps))
    C = np.zeros((n_items, n_items), dtype=float)
    for i in range(n_items):
        C[i, i] = 1.0 - 2.0 * eps
        C[i, (i + 1) % n_items] = eps
        C[i, (i - 1) % n_items] = eps
    return C


def predict_observed_transition_matrix(T_true: np.ndarray, C: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray:
    """Predict observed transition matrix under latent transitions + independent label noise.

    True state: s_t -> s_{t+1} with transition T_true.
    Observed label: o_t ~ C[s_t, :], o_{t+1} ~ C[s_{t+1}, :].

    Returns P(o_{t+1}=b | o_t=a) using a stationary mixture over true states pi.
    """
    n_items = int(T_true.shape[0])
    if pi is None:
        pi = np.ones(n_items, dtype=float) / float(n_items)
    pi = np.asarray(pi, dtype=float)
    pi = pi / float(np.sum(pi))

    # joint[a,b] = sum_{i,j} pi[i] * T_true[i,j] * C[i,a] * C[j,b]
    joint = np.einsum("i,ij,ia,jb->ab", pi, T_true, C, C)
    return row_normalize_counts(joint)


def collapse_consecutive_positions(seq: Sequence[int]) -> List[int]:
    """Remove consecutive repeats (matches how we pre-collapse identical item fixations)."""
    out: List[int] = []
    prev: Optional[int] = None
    for x in seq:
        xi = int(x)
        if prev is None or xi != prev:
            out.append(xi)
        prev = xi
    return out


def simulate_observed_sequence_from_true(
    true_seq: Sequence[int],
    C: np.ndarray,
    rng: np.random.Generator,
    apply_collapse: bool = True,
) -> List[int]:
    """Sample observed labels from a true sequence via confusion matrix C."""
    n_items = int(C.shape[0])
    obs: List[int] = []
    for s in true_seq:
        si = int(s) - 1
        if si < 0 or si >= n_items:
            continue
        obs_label = int(rng.choice(np.arange(1, n_items + 1), p=C[si, :]))
        obs.append(obs_label)
    return collapse_consecutive_positions(obs) if apply_collapse else obs


def simulate_true_sequence_sweep(
    length: int,
    n_items: int,
    rng: np.random.Generator,
    p_cw: float = 0.5,
) -> List[int]:
    """Simulate an ideal sweep-like true sequence on a ring.

    Choose a direction once per trial (CW with prob p_cw), then step by ±1.
    """
    length = int(length)
    if length <= 0:
        return []
    start = int(rng.integers(1, n_items + 1))
    cw = bool(rng.random() < float(p_cw))
    step = 1 if cw else -1
    seq = [start]
    cur = start
    for _ in range(length - 1):
        cur = ((cur - 1 + step) % n_items) + 1
        seq.append(cur)
    return seq


def simulate_true_sequence_stationary(length: int, n_items: int, rng: np.random.Generator) -> List[int]:
    """Simulate a stationary true sequence (no movement)."""
    length = int(length)
    if length <= 0:
        return []
    s = int(rng.integers(1, n_items + 1))
    return [s] * length


def build_template_transition_matrix_from_lengths(
    lengths: Sequence[int],
    n_items: int,
    C: np.ndarray,
    rng: np.random.Generator,
    model: str,
    p_cw: float,
) -> np.ndarray:
    """Generate a template transition matrix by simulating sequences.

    Key idea: match the empirical per-trial sequence length distribution,
    then apply the same consecutive-collapse rule.
    """
    sim_seqs: List[List[int]] = []
    for L in lengths:
        L = int(L)
        if L < 2:
            continue
        if model == "sweep":
            true_seq = simulate_true_sequence_sweep(L, n_items=n_items, rng=rng, p_cw=p_cw)
        elif model == "stationary":
            true_seq = simulate_true_sequence_stationary(L, n_items=n_items, rng=rng)
        else:
            raise ValueError(f"Unknown template model: {model}")
        obs_seq = simulate_observed_sequence_from_true(true_seq, C=C, rng=rng, apply_collapse=True)
        if len(obs_seq) >= 2:
            sim_seqs.append(obs_seq)
    return row_normalize_counts(transition_counts_from_sequences(sim_seqs, n_items=n_items))


def plot_transition_templates(
    out_path: Path,
    panels: Sequence[str],
    obs_by_panel: Dict[str, np.ndarray],
    templates: Dict[str, np.ndarray],
    title: str,
):
    """Plot observed transitions alongside simple sweep templates.

    This figure is meant to match the templates used in the transition-similarity
    calculation: bidirectional / forward (CW) / backward (CCW).
    """
    n_items = int(next(iter(obs_by_panel.values())).shape[0])
    template_order = ["bidirectional", "forward", "backward"]
    fig, axes = plt.subplots(len(panels), 1 + len(template_order), figsize=(13, 7), constrained_layout=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}
    col_titles = ["Observed", "Bidirectional", "Forward", "Backward"]

    vmax = 0.0
    for panel in panels:
        mats = [obs_by_panel[panel]] + [templates.get(k) for k in template_order]
        for mat in mats:
            if mat is None or not np.any(np.isfinite(mat)):
                continue
            vmax = max(vmax, float(np.nanmax(mat)))
    vmax = max(1e-6, min(1.0, vmax))

    for r, panel in enumerate(panels):
        mats = [obs_by_panel[panel]] + [templates.get(k) for k in template_order]
        for c, mat in enumerate(mats):
            ax = axes[r, c]
            mask = ~np.isfinite(mat)
            sns.heatmap(
                mat,
                ax=ax,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                square=True,
                cbar=(r == 0),
                cbar_kws={"shrink": 0.8} if r == 0 else None,
                mask=mask,
                xticklabels=[str(i) for i in range(1, n_items + 1)],
                yticklabels=[str(i) for i in range(1, n_items + 1)],
            )
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(f"{panel_titles.get(panel, panel)}\nCurrent pos")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.set_xlabel("Next pos" if r == len(panels) - 1 else "")
            if r != len(panels) - 1:
                ax.set_xticklabels([])
            ax.tick_params(length=0)

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def sweep_template_matrices(n_items: int = 6) -> Dict[str, np.ndarray]:
    """Return simple sweep transition templates (true/ideal, no noise).

    Templates are conditional: from i, transition only to i+1 and/or i-1.
    """
    return {
        "bidirectional": sweep_true_transition_matrix(n_items=n_items, p_cw=0.5),
        "forward": sweep_true_transition_matrix(n_items=n_items, p_cw=1.0),
        "backward": sweep_true_transition_matrix(n_items=n_items, p_cw=0.0),
    }


def matrix_similarity(obs: np.ndarray, tmpl: np.ndarray) -> float:
    """Similarity between two matrices via Pearson correlation of flattened entries.

    Ignores NaNs in either matrix.
    Returns NaN if fewer than 2 valid entries or zero variance.
    """
    o = np.asarray(obs, dtype=float).ravel()
    t = np.asarray(tmpl, dtype=float).ravel()
    mask = np.isfinite(o) & np.isfinite(t)
    if int(np.sum(mask)) < 2:
        return float("nan")
    o = o[mask]
    t = t[mask]
    if float(np.nanstd(o)) == 0.0 or float(np.nanstd(t)) == 0.0:
        return float("nan")
    return float(np.corrcoef(o, t)[0, 1])


def plot_transition_similarity(
    out_path: Path,
    panels: Sequence[str],
    template_names: Sequence[str],
    sim_by_panel: Dict[str, Dict[str, np.ndarray]],
    title: str,
):
    """Plot per-subject similarity to each template (points + mean±95% CI)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4), sharey=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    label_map = {"bidirectional": "Bi", "forward": "Fwd", "backward": "Bwd"}
    xs = np.arange(len(template_names), dtype=float)
    for ax, panel in zip(axes, panels):
        for i, name in enumerate(template_names):
            y = np.asarray(sim_by_panel[panel][name], dtype=float)
            y = y[np.isfinite(y)]

            # jittered points
            if len(y) > 0:
                jitter = (np.random.default_rng(0).random(len(y)) - 0.5) * 0.18
                ax.scatter(np.full(len(y), xs[i]) + jitter, y, color="black", s=18, alpha=0.55)

            m = nanmean_safe(y)
            sd = nanstd_safe(y, ddof=1)
            n = int(np.sum(np.isfinite(y)))
            se = float(sd / np.sqrt(n)) if np.isfinite(sd) and n >= 2 else np.nan
            ci = 1.96 * se if np.isfinite(se) else np.nan
            ax.scatter([xs[i]], [m], color="black", s=40, zorder=5)
            if np.isfinite(ci):
                ax.errorbar([xs[i]], [m], yerr=[ci], color="black", lw=2, capsize=4, zorder=5)

        ax.axhline(0, color="gray", lw=1, alpha=0.5)
        ax.set_title(panel_titles.get(panel, panel))
        ax.set_xticks(xs)
        ax.set_xticklabels([label_map.get(n, n) for n in template_names])
        sns.despine(ax=ax)

    axes[0].set_ylabel("Transition-matrix similarity (Pearson r)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def is_image_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("_")
    return len(parts) == 4 and all(len(p) > 0 for p in parts)


def list_subjects(data_root: Path) -> List[str]:
    subs: List[str] = []
    if not data_root.exists():
        return subs
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and len(p.name) == 3:
            subs.append(p.name)
    return subs


def parse_xy(value: object) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        try:
            return float(parsed[0]), float(parsed[1])
        except Exception:
            return None
    return None


def circular_signed_lag(a: int, b: int, n: int) -> int:
    if a == b:
        return 0
    cw = (b - a) % n
    if cw == 0:
        return 0
    if cw <= n // 2:
        return int(cw)
    return -int(n - cw)


def load_choice_item_fixations(fix_path: Path) -> pd.DataFrame:
    df = pd.read_csv(fix_path)
    df = df[(df.get("phase") == "choice") & (df.get("event") == "choice")].copy()
    if df.empty:
        return df
    df = df[df.get("roi_content", pd.Series(dtype=str)).apply(is_image_name)].copy()
    if df.empty:
        return df

    needed = ["game", "trial_number", "roi_content", "fix_start", "fix_end"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Fixation file missing required columns {missing}: {fix_path}")

    df["game"] = pd.to_numeric(df["game"], errors="coerce")
    df["trial_number"] = pd.to_numeric(df["trial_number"], errors="coerce")
    df["fix_start"] = pd.to_numeric(df["fix_start"], errors="coerce")
    # Preserve ROI center coordinates when available (e.g., NN compiled datasets).
    if "roi_x" in df.columns:
        df["roi_x"] = pd.to_numeric(df["roi_x"], errors="coerce")
    else:
        df["roi_x"] = np.nan
    if "roi_y" in df.columns:
        df["roi_y"] = pd.to_numeric(df["roi_y"], errors="coerce")
    else:
        df["roi_y"] = np.nan
    df = df.dropna(subset=["game", "trial_number", "fix_start"]).copy()
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)

    df = df.sort_values(["game", "trial_number", "fix_start"])

    # Collapse consecutive identical item fixations (same logic as contiguity script)
    out_rows: List[dict] = []
    for (game, trial), sub in df.groupby(["game", "trial_number"], sort=False):
        sub = sub.sort_values("fix_start")
        prev_img: Optional[str] = None
        combined: List[dict] = []
        for _, r in sub.iterrows():
            cur = str(r["roi_content"])
            if prev_img is None or cur != prev_img:
                combined.append(
                    {
                        "game": int(game),
                        "trial_number": int(trial),
                        "image": cur,
                        "fix_start": float(r["fix_start"]),
                        "fix_end": float(r["fix_end"]) if "fix_end" in sub.columns else np.nan,
                        "roi_x": float(r.get("roi_x", np.nan)),
                        "roi_y": float(r.get("roi_y", np.nan)),
                    }
                )
            else:
                combined[-1]["fix_end"] = max(combined[-1]["fix_end"], float(r["fix_end"]))
            prev_img = cur
        for i, row in enumerate(combined, start=1):
            row["fixation_count"] = i
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(
            columns=["game", "trial_number", "image", "fix_start", "fix_end", "roi_x", "roi_y", "fixation_count"]
        )
    return pd.DataFrame(out_rows)


def load_main_logfile(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)
    for col in ["phase", "event"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ["game", "trial_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_spatial_clockwise_map(
    log_df: pd.DataFrame,
    fix_df: Optional[pd.DataFrame] = None,
    n_items: int = 6,
) -> Dict[int, Dict[str, int]]:
    mem = log_df[(log_df.get("phase") == "memory") & (log_df.get("event") == "spatial_recall")].copy()
    if mem.empty:
        # Fallback for NN-compiled datasets: infer each image's position from fixation ROI centers.
        if fix_df is None or fix_df.empty:
            return {}
        if "roi_x" not in fix_df.columns or "roi_y" not in fix_df.columns:
            return {}

        d = fix_df.dropna(subset=["game", "image", "roi_x", "roi_y"]).copy()
        if d.empty:
            return {}

        d["game"] = pd.to_numeric(d["game"], errors="coerce").astype(int)
        d["roi_x"] = pd.to_numeric(d["roi_x"], errors="coerce")
        d["roi_y"] = pd.to_numeric(d["roi_y"], errors="coerce")
        d = d.dropna(subset=["roi_x", "roi_y"]).copy()
        if d.empty:
            return {}

        # Build ONE canonical (roi_x, roi_y) -> position mapping for the whole
        # subject, using the full set of slot centers observed across all games.
        # Doing this per-game (from only the images that were actually fixated
        # in that game) produces inconsistent position labels: the same physical
        # slot can be labeled differently across games depending on which
        # subset of slots happened to be fixated, which concentrates
        # transitions in low-index cells and creates a spurious 1<->2 bias.
        d["rx"] = d["roi_x"].round(0)
        d["ry"] = d["roi_y"].round(0)
        uniq_xy = (
            d[["rx", "ry"]]
            .drop_duplicates()
            .to_numpy(dtype=float)
        )
        if len(uniq_xy) == 0:
            return {}
        # Centre the ring at the centroid of the unique slot coordinates.
        cx = float(uniq_xy[:, 0].mean())
        cy = float(uniq_xy[:, 1].mean())
        slot_angles: List[Tuple[float, float, float]] = []
        for x, y in uniq_xy:
            y_up = cy - float(y)
            ang = math.atan2(y_up, float(x) - cx)  # increases CCW
            slot_angles.append((float(x), float(y), float(ang)))
        slot_angles.sort(key=lambda t: t[2], reverse=True)  # clockwise = descending angle
        canonical: Dict[Tuple[float, float], int] = {
            (x, y): i for i, (x, y, _) in enumerate(slot_angles, start=1)
        }

        out2: Dict[int, Dict[str, int]] = {}
        for g, gdf in d.groupby("game"):
            pts = gdf.groupby("image")[["rx", "ry"]].median().dropna()
            if pts.empty:
                continue
            mapping: Dict[str, int] = {}
            for img, row in pts.iterrows():
                key = (float(row["rx"]), float(row["ry"]))
                if key in canonical:
                    mapping[str(img)] = int(canonical[key])
            if mapping:
                out2[int(g)] = mapping
        return out2
    mem = mem.dropna(subset=["game", "image", "true_position"]).copy()
    mem["game"] = mem["game"].astype(int)
    mem["image"] = mem["image"].astype(str)

    out: Dict[int, Dict[str, int]] = {}
    for g, gdf in mem.groupby("game"):
        pos_rows = []
        for img, sub in gdf.groupby("image"):
            xy = parse_xy(sub["true_position"].iloc[0])
            if xy is None:
                continue
            pos_rows.append((img, float(xy[0]), float(xy[1])))
        if not pos_rows:
            continue

        xs = np.array([r[1] for r in pos_rows], dtype=float)
        ys = np.array([r[2] for r in pos_rows], dtype=float)
        cx = float(xs.mean())
        cy = float(ys.mean())

        angles = []
        for img, x, y in pos_rows:
            y_up = cy - float(y)
            ang = math.atan2(y_up, float(x) - cx)  # increases CCW
            angles.append((img, ang))

        # Clockwise order = descending angle
        angles_sorted = sorted(angles, key=lambda t: t[1], reverse=True)
        mapping = {img: i for i, (img, _) in enumerate(angles_sorted, start=1)}
        out[int(g)] = mapping
    return out


def split_first(images: Sequence[str]) -> List[str]:
    seen: set = set()
    first: List[str] = []
    for img in images:
        if img not in seen:
            seen.add(img)
            first.append(img)
    return first


def split_revisit(images: Sequence[str]) -> List[str]:
    """Return only revisits: items that have been seen before in the sequence."""
    seen: set = set()
    revisit: List[str] = []
    for img in images:
        if img in seen:
            revisit.append(img)
        else:
            seen.add(img)
    return revisit


def to_positions(images: Sequence[str], mapping: Dict[str, int]) -> List[int]:
    return [int(mapping[img]) for img in images if img in mapping]


@dataclass(frozen=True)
class TrialSeq:
    subject_id: str
    game: int
    trial_number: int
    spatial_all: Tuple[int, ...]
    spatial_first: Tuple[int, ...]
    spatial_revisit: Tuple[int, ...]


def build_trial_sequences(subject_id: str, log_df: pd.DataFrame, fix_df: pd.DataFrame, n_items: int = 6) -> List[TrialSeq]:
    smap_by_game = build_spatial_clockwise_map(log_df, fix_df=fix_df, n_items=n_items)
    if fix_df.empty:
        return []

    trials: List[TrialSeq] = []
    for (game, trial), gdf in fix_df.groupby(["game", "trial_number"], sort=False):
        game = int(game)
        trial = int(trial)
        smap = smap_by_game.get(game)
        if not smap:
            continue

        gdf = gdf.sort_values("fixation_count")
        images = gdf["image"].astype(str).tolist()
        if not images:
            continue

        first_images = split_first(images)
        revisit_images = split_revisit(images)
        spatial_all = tuple(to_positions(images, smap))
        spatial_first = tuple(to_positions(first_images, smap))
        spatial_revisit = tuple(to_positions(revisit_images, smap))

        if len(spatial_all) < 2 and len(spatial_first) < 2 and len(spatial_revisit) < 2:
            continue

        trials.append(
            TrialSeq(
                subject_id=subject_id,
                game=game,
                trial_number=trial,
                spatial_all=spatial_all,
                spatial_first=spatial_first,
                spatial_revisit=spatial_revisit,
            )
        )
    return trials


def max_consecutive_run(lags: Sequence[int], target: int) -> int:
    best = 0
    cur = 0
    for l in lags:
        if int(l) == int(target):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def is_rotation(seq: Sequence[int], base: Sequence[int]) -> bool:
    if len(seq) != len(base):
        return False
    if len(seq) == 0:
        return False
    s = list(seq)
    b = list(base)
    # check any rotation
    for k in range(len(b)):
        if s == b[k:] + b[:k]:
            return True
    return False


def compute_sweep_metrics(seq: Sequence[int], n_items: int = 6, allow_perfect: bool = False) -> Dict[str, float]:
    seq = [int(p) for p in seq if 1 <= int(p) <= n_items]
    if len(seq) < 2:
        return {
            "n_fix": float(len(seq)),
            "n_trans": 0.0,
            "adjacent_rate": np.nan,
            "dir_consistency": np.nan,
            "adj_dir_consistency": np.nan,
            "max_adj_streak": np.nan,
            "perfect_cw": np.nan,
            "perfect_ccw": np.nan,
        }

    lags = [circular_signed_lag(a, b, n_items) for a, b in zip(seq[:-1], seq[1:])]
    lags = [int(l) for l in lags if int(l) != 0]
    if len(lags) == 0:
        return {
            "n_fix": float(len(seq)),
            "n_trans": 0.0,
            "adjacent_rate": np.nan,
            "dir_consistency": np.nan,
            "adj_dir_consistency": np.nan,
            "max_adj_streak": np.nan,
            "perfect_cw": np.nan,
            "perfect_ccw": np.nan,
        }

    abs_l = np.array([abs(l) for l in lags], dtype=float)
    adjacent_rate = float(np.mean(abs_l == 1))

    pos_rate = float(np.mean(np.array(lags, dtype=int) > 0))
    neg_rate = float(np.mean(np.array(lags, dtype=int) < 0))
    dir_consistency = float(max(pos_rate, neg_rate))

    adj_dir_consistency = float(max(np.mean(np.array(lags, dtype=int) == 1), np.mean(np.array(lags, dtype=int) == -1)))

    max_adj_streak = float(max(max_consecutive_run(lags, 1), max_consecutive_run(lags, -1)))

    perfect_cw = np.nan
    perfect_ccw = np.nan
    if allow_perfect and len(seq) == n_items and len(set(seq)) == n_items:
        perfect_cw = 1.0 if is_rotation(seq, list(range(1, n_items + 1))) else 0.0
        perfect_ccw = 1.0 if is_rotation(seq, list(range(n_items, 0, -1))) else 0.0

    return {
        "n_fix": float(len(seq)),
        "n_trans": float(len(lags)),
        "adjacent_rate": adjacent_rate,
        "dir_consistency": dir_consistency,
        "adj_dir_consistency": adj_dir_consistency,
        "max_adj_streak": max_adj_streak,
        "perfect_cw": float(perfect_cw) if np.isfinite(perfect_cw) else np.nan,
        "perfect_ccw": float(perfect_ccw) if np.isfinite(perfect_ccw) else np.nan,
    }


def adjacent_run_lengths_from_lags(lags: Sequence[int]) -> List[int]:
    """Return lengths of consecutive runs of +1 or -1 lags.

    Runs are segmented on the lag sequence; any lag with |lag|!=1 breaks a run.
    """
    out: List[int] = []
    cur_sign: Optional[int] = None
    cur_len = 0
    for l in lags:
        li = int(l)
        if abs(li) != 1:
            if cur_len > 0:
                out.append(int(cur_len))
            cur_sign = None
            cur_len = 0
            continue
        if cur_sign is None or li != cur_sign:
            if cur_len > 0:
                out.append(int(cur_len))
            cur_sign = li
            cur_len = 1
        else:
            cur_len += 1
    if cur_len > 0:
        out.append(int(cur_len))
    return out


def run_length_hist_prob(run_lengths: Sequence[int], max_bin: int = 6) -> np.ndarray:
    """Histogram over run lengths (1..max_bin-1, max_bin=="max_bin+").

    Returns probabilities that sum to 1 over runs. If no runs, returns all-NaN.
    """
    max_bin = int(max_bin)
    if max_bin < 2:
        raise ValueError("max_bin must be >=2")

    rl = [int(x) for x in run_lengths if int(x) >= 1]
    if len(rl) == 0:
        return np.full((max_bin,), np.nan, dtype=float)

    counts = np.zeros((max_bin,), dtype=float)
    for L in rl:
        idx = min(L, max_bin) - 1
        counts[idx] += 1.0
    counts /= float(np.sum(counts))
    return counts


def transition_proportion_in_runs_by_length(seq: Sequence[int], n_items: int = 6, max_bin: int = 6) -> np.ndarray:
    """Proportion of transitions belonging to adjacent runs of each length.

    For each trial, we compute the signed lag sequence (dropping 0). Adjacent
    runs are consecutive segments of lags where lag==+1 or lag==-1.

    We then allocate each adjacent transition to a bin based on its run length.
    The returned vector p[k] is the fraction of *all* transitions (non-zero lags)
    that belong to adjacent runs of length (k+1), with the last bin aggregating
    lengths >= max_bin.

    If there are no valid transitions, returns all-NaN.
    """
    max_bin = int(max_bin)
    seq = [int(p) for p in seq if 1 <= int(p) <= n_items]
    if len(seq) < 2:
        return np.full((max_bin,), np.nan, dtype=float)

    lags = [int(circular_signed_lag(a, b, n_items)) for a, b in zip(seq[:-1], seq[1:])]
    lags = [l for l in lags if l != 0]
    n_trans = int(len(lags))
    if n_trans <= 0:
        return np.full((max_bin,), np.nan, dtype=float)

    run_lengths = adjacent_run_lengths_from_lags(lags)
    out = np.zeros((max_bin,), dtype=float)
    if len(run_lengths) == 0:
        return out / float(n_trans)

    # Iterate through runs again to attribute transitions to bins
    # (we recompute run segmentation to avoid retaining indices)
    cur_sign: Optional[int] = None
    cur_len = 0
    for li in lags + [9999]:
        # sentinel to flush
        if abs(int(li)) != 1:
            if cur_len > 0:
                idx = min(cur_len, max_bin) - 1
                out[idx] += float(cur_len)
            cur_sign = None
            cur_len = 0
            continue
        if cur_sign is None or int(li) != int(cur_sign):
            if cur_len > 0:
                idx = min(cur_len, max_bin) - 1
                out[idx] += float(cur_len)
            cur_sign = int(li)
            cur_len = 1
        else:
            cur_len += 1

    return out / float(n_trans)


def abs_adjacent_direction_bias(seq: Sequence[int], n_items: int = 6) -> float:
    """Absolute CW-vs-CCW bias among adjacent transitions only.

    Let CW be lag=+1 and CCW be lag=-1 (on the n_items ring).
    This returns |(CW-CCW)/N| where N is the number of non-zero transitions.

    Interpretation:
    - 0: no net adjacent directionality (balanced or few adjacent steps)
    - 1: every transition is an adjacent step in a single direction

    If a sequence has no non-zero transitions, returns NaN.
    """
    seq = [int(p) for p in seq if 1 <= int(p) <= n_items]
    if len(seq) < 2:
        return float("nan")

    lags = [int(circular_signed_lag(a, b, n_items)) for a, b in zip(seq[:-1], seq[1:])]
    lags = [l for l in lags if l != 0]
    n_trans = int(len(lags))
    if n_trans <= 0:
        return float("nan")
    cw = int(np.sum(np.array(lags, dtype=int) == 1))
    ccw = int(np.sum(np.array(lags, dtype=int) == -1))
    return float(abs((cw - ccw) / float(n_trans)))


def nanmean_safe(x: Sequence[float]) -> float:
    arr = np.array(list(x), dtype=float)
    if arr.size == 0:
        return float("nan")
    if not np.any(np.isfinite(arr)):
        return float("nan")
    m = float(np.nanmean(arr))
    return m if np.isfinite(m) else float("nan")


def nanstd_safe(x: Sequence[float], ddof: int = 1) -> float:
    arr = np.array(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= ddof:
        return float("nan")
    return float(np.std(arr, ddof=ddof))


def mean_sem(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr)
    if arr.size == 0:
        shape = list(arr.shape)
        if axis < 0:
            axis = arr.ndim + axis
        if arr.ndim == 0:
            return np.array(np.nan), np.array(np.nan)
        if 0 <= axis < len(shape):
            shape.pop(axis)
        out_shape = tuple(shape)
        return np.full(out_shape, np.nan), np.full(out_shape, np.nan)

    finite = np.isfinite(arr)
    n = finite.sum(axis=axis)
    sum_ = np.where(finite, arr, 0.0).sum(axis=axis)
    mean = np.full(sum_.shape, np.nan, dtype=float)
    np.divide(sum_, n, out=mean, where=n > 0)

    sem = np.full(mean.shape, np.nan, dtype=float)
    if np.any(n >= 2):
        mean_b = np.expand_dims(mean, axis=axis)
        sq = np.where(finite, (arr - mean_b) ** 2, 0.0).sum(axis=axis)
        var = np.full(mean.shape, np.nan, dtype=float)
        np.divide(sq, (n - 1), out=var, where=n >= 2)
        sem[n >= 2] = np.sqrt(var[n >= 2]) / np.sqrt(n[n >= 2])
    return mean, sem


def mean_ci95(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    mean, sem = mean_sem(arr, axis=axis)
    return mean, 1.96 * sem


def simulate_shuffle_seq(seq: Sequence[int], rng: np.random.Generator) -> Tuple[int, ...]:
    if seq is None:
        return tuple()
    arr = np.array([int(p) for p in seq], dtype=int)
    if arr.size <= 1:
        return tuple(int(x) for x in arr.tolist())
    rng.shuffle(arr)
    return tuple(int(x) for x in arr.tolist())


def plot_metric_pairs(
    out_path: Path,
    metrics: Sequence[str],
    panels: Sequence[str],
    obs: Dict[str, Dict[str, np.ndarray]],
    ch: Dict[str, Dict[str, np.ndarray]],
    title: str,
):
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(10, 9), sharex=False)
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(panels) == 1:
        axes = axes[:, np.newaxis]

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for r, met in enumerate(metrics):
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            y_obs = obs[panel][met]
            y_ch = ch[panel][met]
            # paired dots
            x0 = np.zeros_like(y_obs, dtype=float)
            x1 = np.ones_like(y_ch, dtype=float)
            for i in range(len(y_obs)):
                if np.isfinite(y_obs[i]) and np.isfinite(y_ch[i]):
                    ax.plot([0, 1], [y_obs[i], y_ch[i]], color="black", alpha=0.25, lw=1)
            ax.scatter(x0, y_obs, color="black", s=18, label="Observed")
            ax.scatter(x1, y_ch, color="gray", s=18, label="Chance")

            # group mean ±95% CI
            n_obs = int(np.sum(np.isfinite(y_obs)))
            n_ch = int(np.sum(np.isfinite(y_ch)))
            m_obs = nanmean_safe(y_obs)
            m_ch = nanmean_safe(y_ch)
            sd_obs = nanstd_safe(y_obs, ddof=1)
            sd_ch = nanstd_safe(y_ch, ddof=1)
            s_obs = float(sd_obs / np.sqrt(n_obs)) if np.isfinite(sd_obs) and n_obs >= 2 else np.nan
            s_ch = float(sd_ch / np.sqrt(n_ch)) if np.isfinite(sd_ch) and n_ch >= 2 else np.nan
            if np.isfinite(m_obs) and np.isfinite(s_obs):
                ax.errorbar([0], [m_obs], yerr=[1.96 * s_obs], color="black", lw=2, capsize=4)
            if np.isfinite(m_ch) and np.isfinite(s_ch):
                ax.errorbar([1], [m_ch], yerr=[1.96 * s_ch], color="gray", lw=2, capsize=4)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Obs", "Chance"])
            if r == 0:
                ax.set_title(panel_titles.get(panel, panel))
            if c == 0:
                ax.set_ylabel(met)
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_bars(
    out_path: Path,
    metrics: Sequence[str],
    panels: Sequence[str],
    obs: Dict[str, Dict[str, np.ndarray]],
    ch: Dict[str, Dict[str, np.ndarray]],
    title: str,
):
    """Bars-only plot: group mean ±95% CI for observed vs chance."""
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(9, 9), sharex=False)
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(panels) == 1:
        axes = axes[:, np.newaxis]

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for r, met in enumerate(metrics):
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            y_obs = obs[panel][met]
            y_ch = ch[panel][met]

            n_obs = int(np.sum(np.isfinite(y_obs)))
            n_ch = int(np.sum(np.isfinite(y_ch)))
            m_obs = nanmean_safe(y_obs)
            m_ch = nanmean_safe(y_ch)
            sd_obs = nanstd_safe(y_obs, ddof=1)
            sd_ch = nanstd_safe(y_ch, ddof=1)
            se_obs = float(sd_obs / np.sqrt(n_obs)) if np.isfinite(sd_obs) and n_obs >= 2 else np.nan
            se_ch = float(sd_ch / np.sqrt(n_ch)) if np.isfinite(sd_ch) and n_ch >= 2 else np.nan

            xs = np.array([0, 1], dtype=float)
            means = np.array([m_obs, m_ch], dtype=float)
            cis = np.array([1.96 * se_obs, 1.96 * se_ch], dtype=float)

            ax.bar(xs[0], means[0], color="black", alpha=0.85, width=0.7, label="Observed")
            ax.bar(xs[1], means[1], color="gray", alpha=0.85, width=0.7, label="Chance")
            if np.isfinite(cis[0]):
                ax.errorbar([xs[0]], [means[0]], yerr=[cis[0]], color="black", lw=2, capsize=4)
            if np.isfinite(cis[1]):
                ax.errorbar([xs[1]], [means[1]], yerr=[cis[1]], color="gray", lw=2, capsize=4)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Obs", "Chance"])
            if r == 0:
                ax.set_title(panel_titles.get(panel, panel))
            if c == 0:
                ax.set_ylabel(met)
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_delta_bars(
    out_path: Path,
    metrics: Sequence[str],
    panels: Sequence[str],
    obs: Dict[str, Dict[str, np.ndarray]],
    ch: Dict[str, Dict[str, np.ndarray]],
    title: str,
):
    """Delta-only plot: group mean ±95% CI for (observed - chance)."""
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(8, 9), sharex=False)
    if len(metrics) == 1:
        axes = np.array([axes])
    if len(panels) == 1:
        axes = axes[:, np.newaxis]

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for r, met in enumerate(metrics):
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            y_obs = obs[panel][met]
            y_ch = ch[panel][met]
            delta = y_obs - y_ch
            delta = delta[np.isfinite(delta)]

            m = nanmean_safe(delta)
            sd = nanstd_safe(delta, ddof=1)
            n = int(np.sum(np.isfinite(delta)))
            se = float(sd / np.sqrt(n)) if np.isfinite(sd) and n >= 2 else np.nan
            ci = 1.96 * se if np.isfinite(se) else np.nan

            ax.axhline(0, color="gray", lw=1, alpha=0.6)
            ax.bar([0], [m], color="black", alpha=0.85, width=0.8)
            if np.isfinite(ci):
                ax.errorbar([0], [m], yerr=[ci], color="black", lw=2, capsize=4)

            ax.set_xticks([0])
            ax.set_xticklabels(["Obs−Chance"])
            if r == 0:
                ax.set_title(panel_titles.get(panel, panel))
            if c == 0:
                ax.set_ylabel(met)
            sns.despine(ax=ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_directionality_pairs(
    out_path: Path,
    panels: Sequence[str],
    obs_by_panel: Dict[str, np.ndarray],
    null_mean_by_panel: Dict[str, np.ndarray],
    title: str,
):
    """Paired subject dots: observed vs shuffle-null mean for abs adjacent direction bias."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.8), sharey=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for c, panel in enumerate(panels):
        ax = axes[c]
        y_obs = np.asarray(obs_by_panel.get(panel, np.array([])), dtype=float)
        y_null = np.asarray(null_mean_by_panel.get(panel, np.array([])), dtype=float)

        n = int(min(len(y_obs), len(y_null)))
        y_obs = y_obs[:n]
        y_null = y_null[:n]

        for i in range(n):
            if np.isfinite(y_obs[i]) and np.isfinite(y_null[i]):
                ax.plot([0, 1], [y_obs[i], y_null[i]], color="black", alpha=0.25, lw=1)

        ax.scatter(np.zeros(n), y_obs, color="black", s=18, label="Observed")
        ax.scatter(np.ones(n), y_null, color="gray", s=18, label="Shuffle null (mean)")

        # group mean ±95% CI
        def _mean_ci(y: np.ndarray) -> Tuple[float, float]:
            y = y[np.isfinite(y)]
            if y.size < 2:
                return float("nan"), float("nan")
            m = float(np.mean(y))
            se = float(np.std(y, ddof=1) / np.sqrt(y.size))
            return m, 1.96 * se

        m_obs, ci_obs = _mean_ci(y_obs)
        m_null, ci_null = _mean_ci(y_null)
        if np.isfinite(m_obs) and np.isfinite(ci_obs):
            ax.errorbar([0], [m_obs], yerr=[ci_obs], color="black", lw=2, capsize=4)
        if np.isfinite(m_null) and np.isfinite(ci_null):
            ax.errorbar([1], [m_null], yerr=[ci_null], color="gray", lw=2, capsize=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Obs", "Null"])
        ax.set_title(panel_titles.get(panel, panel))
        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Abs adjacent direction bias")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_directionality_delta(
    out_path: Path,
    panels: Sequence[str],
    obs_by_panel: Dict[str, np.ndarray],
    null_mean_by_panel: Dict[str, np.ndarray],
    title: str,
):
    """Delta-only plot: mean ±95% CI for (observed - shuffle-null mean)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.3), sharey=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for c, panel in enumerate(panels):
        ax = axes[c]
        y_obs = np.asarray(obs_by_panel.get(panel, np.array([])), dtype=float)
        y_null = np.asarray(null_mean_by_panel.get(panel, np.array([])), dtype=float)
        n = int(min(len(y_obs), len(y_null)))
        delta = (y_obs[:n] - y_null[:n]).astype(float)
        delta = delta[np.isfinite(delta)]

        m = nanmean_safe(delta)
        sd = nanstd_safe(delta, ddof=1)
        nn = int(np.sum(np.isfinite(delta)))
        se = float(sd / np.sqrt(nn)) if np.isfinite(sd) and nn >= 2 else float("nan")
        ci = 1.96 * se if np.isfinite(se) else float("nan")

        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar([0], [m], color="black", alpha=0.85, width=0.8)
        if np.isfinite(ci):
            ax.errorbar([0], [m], yerr=[ci], color="black", lw=2, capsize=4)

        ax.set_xticks([0])
        ax.set_xticklabels(["Obs−Null"])
        ax.set_title(panel_titles.get(panel, panel))
        sns.despine(ax=ax)

    axes[0].set_ylabel("Abs adjacent direction bias Δ")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_streak_distributions(
    out_path: Path,
    panels: Sequence[str],
    runprob_obs: Dict[str, np.ndarray],
    runprob_null: Dict[str, np.ndarray],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    bin_labels: Sequence[str],
    title: str,
):
    """2xN figure: run-length distribution and transition proportion in runs by length.

    For each panel (column), row 1 shows P(run length = k) over runs, and row 2
    shows fraction of transitions belonging to runs of length k. Each subplot
    compares observed (black) to shuffle-null mean (gray), with group mean ±95% CI.
    """
    n_panels = int(len(panels))
    k = int(len(bin_labels))
    fig, axes = plt.subplots(2, n_panels, figsize=(4.6 * n_panels, 6.2), sharex=False)
    if n_panels == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}
    xs = np.arange(k, dtype=float)
    bin_labels = list(bin_labels)
    bin_labels_trans = bin_labels + ["NonAdj"]
    xs_trans = np.arange(k + 1, dtype=float)
    w = 0.38

    def _append_nonadj(mat: np.ndarray) -> np.ndarray:
        """Append NonAdj column = 1 - sum(run-length bins), row-wise."""
        mat = np.asarray(mat, dtype=float)
        if mat.size == 0:
            return np.empty((0, k + 1), dtype=float)
        if mat.ndim != 2 or mat.shape[1] != k:
            raise ValueError(f"Expected matrix with shape (n,{k}); got {mat.shape}")
        out = np.full((mat.shape[0], k + 1), np.nan, dtype=float)
        out[:, :k] = mat
        for i in range(mat.shape[0]):
            row = mat[i, :]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            s = float(np.sum(row[finite]))
            v = 1.0 - s
            # Numerical guard
            v = min(1.0, max(0.0, v))
            out[i, k] = v
        return out

    def _mean_ci_by_bin(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mat = np.asarray(mat, dtype=float)
        mean, ci = mean_ci95(mat, axis=0)
        return mean, ci

    for c, panel in enumerate(panels):
        # Row 1: run length distribution (over runs)
        ax = axes[0, c]
        obs = np.asarray(runprob_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(runprob_null.get(panel, np.empty((0, k))), dtype=float)
        m_obs, ci_obs = _mean_ci_by_bin(obs)
        m_nul, ci_nul = _mean_ci_by_bin(nul)
        ax.bar(xs - w / 2, m_obs, width=w, color="black", alpha=0.85, label="Observed")
        ax.bar(xs + w / 2, m_nul, width=w, color="gray", alpha=0.75, label="Shuffle null")
        ax.errorbar(xs - w / 2, m_obs, yerr=ci_obs, fmt="none", ecolor="black", lw=2, capsize=3)
        ax.errorbar(xs + w / 2, m_nul, yerr=ci_nul, fmt="none", ecolor="gray", lw=2, capsize=3)
        ax.set_title(panel_titles.get(panel, panel))
        if c == 0:
            ax.set_ylabel("P(run length = k)")
        sns.despine(ax=ax)

        # Row 2: transition proportion in runs by length
        ax = axes[1, c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, k))), dtype=float)
        obs2 = _append_nonadj(obs)
        nul2 = _append_nonadj(nul)
        m_obs, ci_obs = _mean_ci_by_bin(obs2)
        m_nul, ci_nul = _mean_ci_by_bin(nul2)
        ax.bar(xs_trans - w / 2, m_obs, width=w, color="black", alpha=0.85)
        ax.bar(xs_trans + w / 2, m_nul, width=w, color="gray", alpha=0.75)
        ax.errorbar(xs_trans - w / 2, m_obs, yerr=ci_obs, fmt="none", ecolor="black", lw=2, capsize=3)
        ax.errorbar(xs_trans + w / 2, m_nul, yerr=ci_nul, fmt="none", ecolor="gray", lw=2, capsize=3)
        if c == 0:
            ax.set_ylabel("Fraction of transitions")
        ax.set_xticks(xs_trans)
        ax.set_xticklabels(bin_labels_trans)
        ax.set_xlabel("Run length k")
        sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_streak_distributions_delta(
    out_path: Path,
    panels: Sequence[str],
    runprob_obs: Dict[str, np.ndarray],
    runprob_null: Dict[str, np.ndarray],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    bin_labels: Sequence[str],
    title: str,
):
    """2xN figure of deltas: (Observed - shuffle-null mean) per bin.

    Uses subjects as the unit: computes per-subject delta vectors then plots the
    group mean ±95% CI for each bin.
    """
    n_panels = int(len(panels))
    k = int(len(bin_labels))
    fig, axes = plt.subplots(2, n_panels, figsize=(4.6 * n_panels, 6.0), sharex=False)
    if n_panels == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}
    xs = np.arange(k, dtype=float)
    bin_labels = list(bin_labels)
    bin_labels_trans = bin_labels + ["NonAdj"]
    xs_trans = np.arange(k + 1, dtype=float)

    def _append_nonadj(mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=float)
        if mat.size == 0:
            return np.empty((0, k + 1), dtype=float)
        if mat.ndim != 2 or mat.shape[1] != k:
            raise ValueError(f"Expected matrix with shape (n,{k}); got {mat.shape}")
        out = np.full((mat.shape[0], k + 1), np.nan, dtype=float)
        out[:, :k] = mat
        for i in range(mat.shape[0]):
            row = mat[i, :]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            s = float(np.sum(row[finite]))
            v = 1.0 - s
            v = min(1.0, max(0.0, v))
            out[i, k] = v
        return out

    def _delta_mean_ci(obs: np.ndarray, nul: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=float)
        nul = np.asarray(nul, dtype=float)
        n = int(min(obs.shape[0], nul.shape[0]))
        if n == 0:
            return np.full((k,), np.nan, dtype=float), np.full((k,), np.nan, dtype=float)
        delta = obs[:n, :] - nul[:n, :]
        mean, ci = mean_ci95(delta, axis=0)
        return mean, ci

    for c, panel in enumerate(panels):
        # Row 1: run-length distribution delta
        ax = axes[0, c]
        obs = np.asarray(runprob_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(runprob_null.get(panel, np.empty((0, k))), dtype=float)
        m, ci = _delta_mean_ci(obs, nul)
        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar(xs, m, color="black", alpha=0.85, width=0.8)
        ax.errorbar(xs, m, yerr=ci, fmt="none", ecolor="black", lw=2, capsize=3)
        ax.set_title(panel_titles.get(panel, panel))
        if c == 0:
            ax.set_ylabel("Δ P(run length = k)")
        sns.despine(ax=ax)

        # Row 2: transition proportion delta
        ax = axes[1, c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, k))), dtype=float)
        obs2 = _append_nonadj(obs)
        nul2 = _append_nonadj(nul)
        m, ci = _delta_mean_ci(obs2, nul2)
        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar(xs_trans, m, color="black", alpha=0.85, width=0.8)
        ax.errorbar(xs_trans, m, yerr=ci, fmt="none", ecolor="black", lw=2, capsize=3)
        if c == 0:
            ax.set_ylabel("Δ fraction of transitions")
        ax.set_xticks(xs_trans)
        ax.set_xticklabels(bin_labels_trans)
        ax.set_xlabel("Run length k")
        sns.despine(ax=ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_transitions_in_runs_by_length(
    out_path: Path,
    panels: Sequence[str],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    bin_labels: Sequence[str],
    title: str,
):
    """1xN figure: fraction of transitions belonging to adjacent runs of each length.

    Compares observed vs shuffle-null mean with group mean ±95% CI (subjects as unit).
    Excludes the NonAdj bin (plotted separately).
    """
    n_panels = int(len(panels))
    k = int(len(bin_labels))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 3.6), sharey=True)
    if n_panels == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}
    xs = np.arange(k, dtype=float)
    w = 0.38

    for c, panel in enumerate(panels):
        ax = axes[c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, k))), dtype=float)
        m_obs, ci_obs = mean_ci95(obs, axis=0)
        m_nul, ci_nul = mean_ci95(nul, axis=0)

        ax.bar(xs - w / 2, m_obs, width=w, color="black", alpha=0.85, label="Observed")
        ax.bar(xs + w / 2, m_nul, width=w, color="gray", alpha=0.75, label="Shuffle null")
        ax.errorbar(xs - w / 2, m_obs, yerr=ci_obs, fmt="none", ecolor="black", lw=2, capsize=3)
        ax.errorbar(xs + w / 2, m_nul, yerr=ci_nul, fmt="none", ecolor="gray", lw=2, capsize=3)

        ax.set_title(panel_titles.get(panel, panel))
        ax.set_xticks(xs)
        ax.set_xticklabels(list(bin_labels))
        ax.set_xlabel("Run length k")
        if c == 0:
            ax.set_ylabel("Fraction of transitions")
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_transitions_in_runs_by_length_delta(
    out_path: Path,
    panels: Sequence[str],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    bin_labels: Sequence[str],
    title: str,
):
    """1xN delta plot: (Observed - shuffle-null mean) per run-length bin."""
    n_panels = int(len(panels))
    k = int(len(bin_labels))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 3.4), sharey=True)
    if n_panels == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}
    xs = np.arange(k, dtype=float)

    for c, panel in enumerate(panels):
        ax = axes[c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, k))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, k))), dtype=float)
        n = int(min(obs.shape[0], nul.shape[0]))
        if n == 0:
            mean = np.full((k,), np.nan, dtype=float)
            ci = np.full((k,), np.nan, dtype=float)
        else:
            delta = obs[:n, :] - nul[:n, :]
            mean, ci = mean_ci95(delta, axis=0)

        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar(xs, mean, color="black", alpha=0.85, width=0.8)
        ax.errorbar(xs, mean, yerr=ci, fmt="none", ecolor="black", lw=2, capsize=3)
        ax.set_title(panel_titles.get(panel, panel))
        ax.set_xticks(xs)
        ax.set_xticklabels(list(bin_labels))
        ax.set_xlabel("Run length k")
        if c == 0:
            ax.set_ylabel("Δ fraction of transitions")
        sns.despine(ax=ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_nonadj_fraction(
    out_path: Path,
    panels: Sequence[str],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    title: str,
):
    """Plot NonAdj fraction (1 - sum of run bins) as observed vs null (mean ±95% CI)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), sharey=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for c, panel in enumerate(panels):
        ax = axes[c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, 0))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, 0))), dtype=float)
        if obs.size == 0 or nul.size == 0:
            y_obs = np.array([], dtype=float)
            y_nul = np.array([], dtype=float)
        else:
            y_obs = 1.0 - np.nansum(obs, axis=1)
            y_nul = 1.0 - np.nansum(nul, axis=1)
        y_obs = y_obs[np.isfinite(y_obs)]
        y_nul = y_nul[np.isfinite(y_nul)]

        def _mean_ci(y: np.ndarray) -> Tuple[float, float]:
            if y.size < 2:
                return float("nan"), float("nan")
            m = float(np.mean(y))
            se = float(np.std(y, ddof=1) / np.sqrt(y.size))
            return m, 1.96 * se

        m_obs, ci_obs = _mean_ci(y_obs)
        m_nul, ci_nul = _mean_ci(y_nul)
        xs = np.array([0, 1], dtype=float)
        ax.bar(xs[0], m_obs, color="black", alpha=0.85, width=0.7, label="Observed")
        ax.bar(xs[1], m_nul, color="gray", alpha=0.75, width=0.7, label="Shuffle null")
        if np.isfinite(ci_obs):
            ax.errorbar([xs[0]], [m_obs], yerr=[ci_obs], color="black", lw=2, capsize=4)
        if np.isfinite(ci_nul):
            ax.errorbar([xs[1]], [m_nul], yerr=[ci_nul], color="gray", lw=2, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Obs", "Null"])
        ax.set_title(panel_titles.get(panel, panel))
        if c == 0:
            ax.set_ylabel("Non-adjacent fraction")
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_nonadj_fraction_delta(
    out_path: Path,
    panels: Sequence[str],
    transprop_obs: Dict[str, np.ndarray],
    transprop_null: Dict[str, np.ndarray],
    title: str,
):
    """Delta plot for NonAdj fraction: (Observed - Null) mean ±95% CI."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.2), sharey=True)
    if len(panels) == 1:
        axes = np.array([axes])

    panel_titles = {PANEL_ALL: "All fixations", PANEL_FIRST: "First fixations", PANEL_REVISIT: "Revisit fixations"}

    for c, panel in enumerate(panels):
        ax = axes[c]
        obs = np.asarray(transprop_obs.get(panel, np.empty((0, 0))), dtype=float)
        nul = np.asarray(transprop_null.get(panel, np.empty((0, 0))), dtype=float)
        if obs.size == 0 or nul.size == 0:
            delta = np.array([], dtype=float)
        else:
            y_obs = 1.0 - np.nansum(obs, axis=1)
            y_nul = 1.0 - np.nansum(nul, axis=1)
            n = int(min(y_obs.shape[0], y_nul.shape[0]))
            delta = (y_obs[:n] - y_nul[:n]).astype(float)
            delta = delta[np.isfinite(delta)]

        m = nanmean_safe(delta)
        sd = nanstd_safe(delta, ddof=1)
        nn = int(np.sum(np.isfinite(delta)))
        se = float(sd / np.sqrt(nn)) if np.isfinite(sd) and nn >= 2 else float("nan")
        ci = 1.96 * se if np.isfinite(se) else float("nan")

        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.bar([0], [m], color="black", alpha=0.85, width=0.8)
        if np.isfinite(ci):
            ax.errorbar([0], [m], yerr=[ci], color="black", lw=2, capsize=4)
        ax.set_xticks([0])
        ax.set_xticklabels(["Obs−Null"])
        ax.set_title(panel_titles.get(panel, panel))
        if c == 0:
            ax.set_ylabel("Non-adjacent fraction Δ")
        sns.despine(ax=ax)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quantify spatial sweep structure in choice fixations.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
        help="Project base directory (default: repository root)",
    )
    parser.add_argument("--buffer", type=int, default=50, help="ROI buffer size used in fixation CSV filename")
    parser.add_argument("--n-sims", type=int, default=2000, help="Shuffle simulations per subject")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--label-noise-eps",
        type=float,
        default=0.10,
        help="Adjacent-label noise rate eps for theoretical templates (default: 0.10)",
    )
    parser.add_argument(
        "--sweep-p-cw",
        type=float,
        default=0.50,
        help="P(clockwise step) in sweep template (default: 0.50)",
    )
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject IDs to include")
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["107", "131"],
        help="Subject IDs to exclude (default: 107 131 for eyetracking)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output path for CSV (default: output/choice_fixation_sweep_summary.csv)",
    )
    parser.add_argument("--fig-dir", type=str, default=None, help="Optional figures directory (default: figures/)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    data_root = base_dir / "data"
    out_root = base_dir / "output"
    fig_root = Path(args.fig_dir) if args.fig_dir else (base_dir / "figures")

    subject_ids = args.subjects if args.subjects else list_subjects(data_root)
    subject_ids = [s for s in subject_ids if s not in set(args.exclude_subjects)]
    subject_ids = [s for s in subject_ids if (data_root / s).exists()]
    subject_ids.sort()
    if not subject_ids:
        raise SystemExit("No subjects found to analyze (after exclusions).")

    rng = np.random.default_rng(int(args.seed))

    rows: List[dict] = []

    # Directionality (CW vs CCW adjacent steps) shuffle test
    dir_rows: List[dict] = []
    dir_obs_store: Dict[str, List[float]] = {p: [] for p in PANELS}
    dir_null_store: Dict[str, List[float]] = {p: [] for p in PANELS}

    # Streak distribution summaries (per subject vectors)
    max_run_bin = 6
    run_bins = [str(i) for i in range(1, max_run_bin)] + [f"{max_run_bin}+"]
    subj_runprob_obs: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_runprob_null: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_transprop_obs: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_transprop_null: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    streak_rows: List[dict] = []

    # Store subject-level metrics for plotting
    subj_obs: Dict[str, Dict[str, List[float]]] = {p: {} for p in PANELS}
    subj_ch: Dict[str, Dict[str, List[float]]] = {p: {} for p in PANELS}

    # Store subject-level transition matrices (row-normalized) for plotting
    subj_trans: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
    subj_ids_for_trans: Dict[str, List[str]] = {p: [] for p in PANELS}

    # Store per-trial sequence lengths (post-collapse) to match in templates
    trial_lengths: Dict[str, List[int]] = {p: [] for p in PANELS}

    metrics = ["adjacent_rate", "dir_consistency", "adj_dir_consistency", "max_adj_streak", "perfect_cw", "perfect_ccw"]

    for panel in PANELS:
        for m in metrics:
            subj_obs[panel][m] = []
            subj_ch[panel][m] = []

    for sid in subject_ids:
        log_path = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        fix_path = base_dir / "data" / sid / f"{sid}_fixations_df_original_buffer_{int(args.buffer)}.csv"
        if not log_path.exists() or not fix_path.exists():
            continue

        log_df = load_main_logfile(log_path)
        fix_df = load_choice_item_fixations(fix_path)
        trials = build_trial_sequences(sid, log_df, fix_df)
        if not trials:
            continue

        # Subject-level observed transitions (pooled across trials, within panel)
        subj_seqs_all = [t.spatial_all for t in trials if t.spatial_all and len(t.spatial_all) >= 2]
        subj_seqs_first = [t.spatial_first for t in trials if t.spatial_first and len(t.spatial_first) >= 2]
        subj_seqs_revisit = [t.spatial_revisit for t in trials if t.spatial_revisit and len(t.spatial_revisit) >= 2]
        for panel, seqs in [(PANEL_ALL, subj_seqs_all), (PANEL_FIRST, subj_seqs_first), (PANEL_REVISIT, subj_seqs_revisit)]:
            counts = transition_counts_from_sequences(seqs, n_items=6)
            subj_trans[panel].append(row_normalize_counts(counts))
            subj_ids_for_trans[panel].append(sid)

        # Collect lengths for templates (use same sequences used for transitions)
        trial_lengths[PANEL_ALL].extend([len(s) for s in subj_seqs_all])
        trial_lengths[PANEL_FIRST].extend([len(s) for s in subj_seqs_first])
        trial_lengths[PANEL_REVISIT].extend([len(s) for s in subj_seqs_revisit])

        # Per trial observed
        per_trial_obs: Dict[str, List[dict]] = {PANEL_ALL: [], PANEL_FIRST: [], PANEL_REVISIT: []}
        per_trial_dirbias_obs: Dict[str, List[float]] = {PANEL_ALL: [], PANEL_FIRST: [], PANEL_REVISIT: []}

        # Streak distributions (observed)
        obs_run_lengths: Dict[str, List[int]] = {p: [] for p in PANELS}
        obs_transprop_trials: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
        for t in trials:
            seq_all = t.spatial_all
            seq_first = t.spatial_first
            seq_revisit = t.spatial_revisit
            per_trial_obs[PANEL_ALL].append(compute_sweep_metrics(seq_all, allow_perfect=False))
            per_trial_obs[PANEL_FIRST].append(compute_sweep_metrics(seq_first, allow_perfect=True))
            per_trial_obs[PANEL_REVISIT].append(compute_sweep_metrics(seq_revisit, allow_perfect=False))

            per_trial_dirbias_obs[PANEL_ALL].append(abs_adjacent_direction_bias(seq_all, n_items=6))
            per_trial_dirbias_obs[PANEL_FIRST].append(abs_adjacent_direction_bias(seq_first, n_items=6))
            per_trial_dirbias_obs[PANEL_REVISIT].append(abs_adjacent_direction_bias(seq_revisit, n_items=6))

            # Observed streak data
            for panel, seq in [(PANEL_ALL, seq_all), (PANEL_FIRST, seq_first), (PANEL_REVISIT, seq_revisit)]:
                lags = [int(circular_signed_lag(a, b, 6)) for a, b in zip(seq[:-1], seq[1:])]
                lags = [l for l in lags if l != 0]
                obs_run_lengths[panel].extend(adjacent_run_lengths_from_lags(lags))
                obs_transprop_trials[panel].append(
                    transition_proportion_in_runs_by_length(seq, n_items=6, max_bin=max_run_bin)
                )

        # Per trial chance via shuffle
        n_sims = int(args.n_sims)
        per_trial_ch_sims: Dict[str, Dict[str, List[float]]] = {p: {m: [] for m in metrics} for p in PANELS}

        # Per-subject null distribution for directionality bias (one value per sim)
        dirbias_null_sims: Dict[str, List[float]] = {p: [] for p in PANELS}

        # Per-subject null distributions for streak summaries (one vector per sim)
        runprob_null_sims: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
        transprop_null_sims: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}

        for _ in range(n_sims):
            sim_panel_vals: Dict[str, List[dict]] = {PANEL_ALL: [], PANEL_FIRST: [], PANEL_REVISIT: []}
            sim_dirbias_vals: Dict[str, List[float]] = {PANEL_ALL: [], PANEL_FIRST: [], PANEL_REVISIT: []}

            sim_run_lengths: Dict[str, List[int]] = {p: [] for p in PANELS}
            sim_transprop_trials: Dict[str, List[np.ndarray]] = {p: [] for p in PANELS}
            for t in trials:
                sim_all = simulate_shuffle_seq(t.spatial_all, rng)
                sim_first = simulate_shuffle_seq(t.spatial_first, rng)
                sim_revisit = simulate_shuffle_seq(t.spatial_revisit, rng)
                sim_panel_vals[PANEL_ALL].append(compute_sweep_metrics(sim_all, allow_perfect=False))
                sim_panel_vals[PANEL_FIRST].append(compute_sweep_metrics(sim_first, allow_perfect=True))
                sim_panel_vals[PANEL_REVISIT].append(compute_sweep_metrics(sim_revisit, allow_perfect=False))

                sim_dirbias_vals[PANEL_ALL].append(abs_adjacent_direction_bias(sim_all, n_items=6))
                sim_dirbias_vals[PANEL_FIRST].append(abs_adjacent_direction_bias(sim_first, n_items=6))
                sim_dirbias_vals[PANEL_REVISIT].append(abs_adjacent_direction_bias(sim_revisit, n_items=6))

                # Null streak data
                for panel, seq in [(PANEL_ALL, sim_all), (PANEL_FIRST, sim_first), (PANEL_REVISIT, sim_revisit)]:
                    lags = [int(circular_signed_lag(a, b, 6)) for a, b in zip(seq[:-1], seq[1:])]
                    lags = [l for l in lags if l != 0]
                    sim_run_lengths[panel].extend(adjacent_run_lengths_from_lags(lags))
                    sim_transprop_trials[panel].append(
                        transition_proportion_in_runs_by_length(seq, n_items=6, max_bin=max_run_bin)
                    )

            for panel in PANELS:
                for m in metrics:
                    per_trial_ch_sims[panel][m].append(nanmean_safe([d[m] for d in sim_panel_vals[panel]]))

                dirbias_null_sims[panel].append(nanmean_safe(sim_dirbias_vals[panel]))

                # For each sim, compute a distribution over run lengths (over runs)
                runprob_null_sims[panel].append(run_length_hist_prob(sim_run_lengths[panel], max_bin=max_run_bin))

                # For each sim, compute the average per-trial transition proportion vector
                if len(sim_transprop_trials[panel]) == 0:
                    transprop_null_sims[panel].append(np.full((max_run_bin,), np.nan, dtype=float))
                else:
                    mean_vec, _ = mean_sem(np.stack(sim_transprop_trials[panel], axis=0), axis=0)
                    transprop_null_sims[panel].append(mean_vec)

        # Subject-level aggregation and append
        for panel in PANELS:
            for m in metrics:
                obs_val = nanmean_safe([d[m] for d in per_trial_obs[panel]])
                ch_val = nanmean_safe(per_trial_ch_sims[panel][m])

                subj_obs[panel][m].append(obs_val)
                subj_ch[panel][m].append(ch_val)

                rows.append(
                    {
                        "subject_id": sid,
                        "panel": panel,
                        "metric": m,
                        "observed": obs_val,
                        "chance": ch_val,
                        "delta": obs_val - ch_val if np.isfinite(obs_val) and np.isfinite(ch_val) else np.nan,
                        "n_trials": int(len(per_trial_obs[panel])),
                        "n_sims": n_sims,
                    }
                )

        # Directionality shuffle test (per subject, per panel)
        for panel in PANELS:
            obs_dir = nanmean_safe(per_trial_dirbias_obs[panel])
            null_vals = np.asarray(dirbias_null_sims[panel], dtype=float)
            null_mean = nanmean_safe(null_vals)
            null_sd = nanstd_safe(null_vals, ddof=1)

            # One-sided p-value for abs bias being larger than expected under shuffle
            p_val = float("nan")
            if np.isfinite(obs_dir) and np.any(np.isfinite(null_vals)):
                finite_null = null_vals[np.isfinite(null_vals)]
                if finite_null.size > 0:
                    ge = int(np.sum(finite_null >= obs_dir))
                    p_val = (ge + 1.0) / (finite_null.size + 1.0)

            z = float("nan")
            if np.isfinite(obs_dir) and np.isfinite(null_mean) and np.isfinite(null_sd) and null_sd > 0:
                z = float((obs_dir - null_mean) / null_sd)

            n_valid_trials = int(np.sum(np.isfinite(np.asarray(per_trial_dirbias_obs[panel], dtype=float))))
            dir_rows.append(
                {
                    "subject_id": sid,
                    "panel": panel,
                    "abs_adj_dir_bias_observed": obs_dir,
                    "abs_adj_dir_bias_null_mean": null_mean,
                    "abs_adj_dir_bias_null_sd": null_sd,
                    "z": z,
                    "p_value": p_val,
                    "n_trials": int(len(per_trial_obs[panel])),
                    "n_valid_trials": n_valid_trials,
                    "n_sims": n_sims,
                }
            )
            dir_obs_store[panel].append(obs_dir)
            dir_null_store[panel].append(null_mean)

        # Streak distribution summaries (per subject, per panel): observed + null mean
        for panel in PANELS:
            obs_runprob = run_length_hist_prob(obs_run_lengths[panel], max_bin=max_run_bin)
            if len(obs_transprop_trials[panel]) == 0:
                obs_transprop = np.full((max_run_bin,), np.nan, dtype=float)
            else:
                obs_transprop, _ = mean_sem(np.stack(obs_transprop_trials[panel], axis=0), axis=0)

            if len(runprob_null_sims[panel]):
                null_runprob, _ = mean_sem(np.stack(runprob_null_sims[panel], axis=0), axis=0)
            else:
                null_runprob = np.full((max_run_bin,), np.nan, dtype=float)

            if len(transprop_null_sims[panel]):
                null_transprop, _ = mean_sem(np.stack(transprop_null_sims[panel], axis=0), axis=0)
            else:
                null_transprop = np.full((max_run_bin,), np.nan, dtype=float)

            subj_runprob_obs[panel].append(obs_runprob)
            subj_runprob_null[panel].append(null_runprob)
            subj_transprop_obs[panel].append(obs_transprop)
            subj_transprop_null[panel].append(null_transprop)

            for bi, blabel in enumerate(run_bins):
                streak_rows.append(
                    {
                        "subject_id": sid,
                        "panel": panel,
                        "bin": blabel,
                        "obs_runprob": float(obs_runprob[bi]) if np.isfinite(obs_runprob[bi]) else np.nan,
                        "null_runprob": float(null_runprob[bi]) if np.isfinite(null_runprob[bi]) else np.nan,
                        "obs_transprop": float(obs_transprop[bi]) if np.isfinite(obs_transprop[bi]) else np.nan,
                        "null_transprop": float(null_transprop[bi]) if np.isfinite(null_transprop[bi]) else np.nan,
                        "n_sims": n_sims,
                    }
                )

    if not rows:
        raise SystemExit("No data processed.")

    out_root.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else (out_root / "choice_fixation_sweep_summary.csv")
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    # Plot paired subject-level observed vs chance
    obs_plot = {p: {m: np.array(subj_obs[p][m], dtype=float) for m in metrics} for p in PANELS}
    ch_plot = {p: {m: np.array(subj_ch[p][m], dtype=float) for m in metrics} for p in PANELS}

    fig_path = fig_root / "choice_fixation_spatial_sweep_metrics.png"
    plot_metric_pairs(
        fig_path,
        metrics=metrics,
        panels=PANELS,
        obs=obs_plot,
        ch=ch_plot,
        title="Spatial sweep metrics (shuffle chance baseline)",
    )

    fig_path = fig_root / "choice_fixation_spatial_sweep_metrics_bars.png"
    plot_metric_bars(
        fig_path,
        metrics=metrics,
        panels=PANELS,
        obs=obs_plot,
        ch=ch_plot,
        title="Spatial sweep metrics (group mean ±95% CI)",
    )

    fig_path = fig_root / "choice_fixation_spatial_sweep_metrics_delta.png"
    plot_metric_delta_bars(
        fig_path,
        metrics=metrics,
        panels=PANELS,
        obs=obs_plot,
        ch=ch_plot,
        title="Spatial sweep metrics Δ (Observed − Chance; mean ±95% CI)",
    )

    # Plot observed transition matrix vs the same simple sweep templates used for similarity
    templates = sweep_template_matrices(n_items=6)

    # Mean observed transition matrices across subjects (elementwise), ignoring NaNs
    obs_trans_by_panel: Dict[str, np.ndarray] = {}
    for panel in PANELS:
        mats = subj_trans.get(panel, [])
        if len(mats) == 0:
            obs_trans_by_panel[panel] = np.full((6, 6), np.nan, dtype=float)
            continue
        stack = np.stack(mats, axis=0)
        obs_trans_by_panel[panel] = np.nanmean(stack, axis=0)

    fig_path = fig_root / "choice_fixation_spatial_sweep_transition_templates.png"
    plot_transition_templates(
        fig_path,
        panels=PANELS,
        obs_by_panel=obs_trans_by_panel,
        templates=templates,
        title="Observed transitions vs sweep templates",
    )

    # Similarity of observed transitions to sweep templates
    template_order = ["bidirectional", "forward", "backward"]
    sim_by_panel: Dict[str, Dict[str, np.ndarray]] = {p: {k: np.array([], dtype=float) for k in template_order} for p in PANELS}
    sim_rows: List[dict] = []
    for panel in PANELS:
        mats = subj_trans.get(panel, [])
        sids = subj_ids_for_trans.get(panel, [])
        for name in template_order:
            vals: List[float] = []
            for sid, mat in zip(sids, mats):
                r = matrix_similarity(mat, templates[name])
                vals.append(r)
                sim_rows.append({"subject_id": sid, "panel": panel, "template": name, "similarity_r": r})
            sim_by_panel[panel][name] = np.asarray(vals, dtype=float)

    sim_csv = out_root / "choice_fixation_sweep_transition_similarity.csv"
    pd.DataFrame(sim_rows).to_csv(sim_csv, index=False)

    fig_path = fig_root / "choice_fixation_sweep_transition_similarity.png"
    plot_transition_similarity(
        fig_path,
        panels=PANELS,
        template_names=template_order,
        sim_by_panel=sim_by_panel,
        title="Observed transition matrices vs sweep templates",
    )

    # Directionality (adjacent CW vs CCW) shuffle test outputs
    dir_csv = out_root / "choice_fixation_sweep_directionality_shuffle_test.csv"
    pd.DataFrame(dir_rows).to_csv(dir_csv, index=False)

    fig_path = fig_root / "choice_fixation_sweep_directionality_shuffle_test.png"
    plot_directionality_pairs(
        fig_path,
        panels=PANELS,
        obs_by_panel={p: np.asarray(dir_obs_store[p], dtype=float) for p in PANELS},
        null_mean_by_panel={p: np.asarray(dir_null_store[p], dtype=float) for p in PANELS},
        title="Adjacent directionality (abs CW–CCW bias) vs shuffle null",
    )

    fig_path = fig_root / "choice_fixation_sweep_directionality_shuffle_test_delta.png"
    plot_directionality_delta(
        fig_path,
        panels=PANELS,
        obs_by_panel={p: np.asarray(dir_obs_store[p], dtype=float) for p in PANELS},
        null_mean_by_panel={p: np.asarray(dir_null_store[p], dtype=float) for p in PANELS},
        title="Adjacent directionality Δ (Observed − Null; mean ±95% CI)",
    )

    # Streak distribution figure + CSV
    streak_csv = out_root / "choice_fixation_sweep_streak_distributions.csv"
    pd.DataFrame(streak_rows).to_csv(streak_csv, index=False)

    trans_obs = {p: np.stack(subj_transprop_obs[p], axis=0) if len(subj_transprop_obs[p]) else np.empty((0, max_run_bin)) for p in PANELS}
    trans_nul = {p: np.stack(subj_transprop_null[p], axis=0) if len(subj_transprop_null[p]) else np.empty((0, max_run_bin)) for p in PANELS}

    # Plot only the fraction-of-transitions-in-runs measure (exclude NonAdj here)
    fig_path = fig_root / "choice_fixation_sweep_transitions_in_runs_by_length.png"
    plot_transitions_in_runs_by_length(
        fig_path,
        panels=PANELS,
        transprop_obs=trans_obs,
        transprop_null=trans_nul,
        bin_labels=run_bins,
        title="Transitions in adjacent runs by run length (Observed vs shuffle null)",
    )

    fig_path = fig_root / "choice_fixation_sweep_transitions_in_runs_by_length_delta.png"
    plot_transitions_in_runs_by_length_delta(
        fig_path,
        panels=PANELS,
        transprop_obs=trans_obs,
        transprop_null=trans_nul,
        bin_labels=run_bins,
        title="Transitions in adjacent runs by length Δ (Observed − Null; mean ±95% CI)",
    )

    # Plot NonAdj fraction separately
    fig_path = fig_root / "choice_fixation_sweep_nonadj_fraction.png"
    plot_nonadj_fraction(
        fig_path,
        panels=PANELS,
        transprop_obs=trans_obs,
        transprop_null=trans_nul,
        title="Non-adjacent transition fraction (Observed vs shuffle null)",
    )

    fig_path = fig_root / "choice_fixation_sweep_nonadj_fraction_delta.png"
    plot_nonadj_fraction_delta(
        fig_path,
        panels=PANELS,
        transprop_obs=trans_obs,
        transprop_null=trans_nul,
        title="Non-adjacent transition fraction Δ (Observed − Null; mean ±95% CI)",
    )

    print(f"Wrote sweep summary CSV: {output_csv}")
    print(f"Wrote figures to: {fig_root}")
    print(f"Wrote transition similarity CSV: {sim_csv}")
    print(f"Wrote directionality shuffle-test CSV: {dir_csv}")
    print(f"Wrote streak distribution CSV: {streak_csv}")


if __name__ == "__main__":
    main()
