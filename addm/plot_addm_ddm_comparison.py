"""Combined aDDM vs DDM comparison script.

Creates a 3-column comparison figure:
  1) p(Take) vs recalled offer value bins (Data, aDDM/free3, DDM/theta1)
  2) RT vs recalled offer value bins (Data, aDDM/free3, DDM/theta1)
  3) Mean held-out log-likelihood delta across held-out games with SEM (aDDM - DDM)

This script is self-contained:
  - Loads trial data from scratch (choice fixations + encoding outcomes + value recall)
  - Runs (or loads cached) aDDM simulations from kfold fit summaries
  - Saves intermediates (trial-level CSV, simulation cache CSVs) to output/addm/ppc/
  - Saves the final figure to output/addm/ppc/

Run from the repository root:
    python -m addm.plot_addm_ddm_comparison
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Package-relative imports for aDDM simulation / fitting
# ---------------------------------------------------------------------------

try:
    from .lib.adapted_addm_simulation import ADDMParameters
    from .lib.addm_fitting import (
        BinningConfig,
        FitConfig,
        SimulationConfig,
        build_fitting_components,
        load_group_fixations,
        load_group_trial_templates,
        simulate_trials_generative_detailed,
    )
except ImportError:
    from lib.adapted_addm_simulation import ADDMParameters
    from lib.addm_fitting import (
        BinningConfig,
        FitConfig,
        SimulationConfig,
        build_fitting_components,
        load_group_fixations,
        load_group_trial_templates,
        simulate_trials_generative_detailed,
    )


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

BIN_ORDER: List[str] = [
    "neg_high",
    "neg_med",
    "neg_low",
    "pos_low",
    "pos_med",
    "pos_high",
]


def _bin_label(sign: str, tier: int, n_bins_per_sign: int) -> str:
    sign = str(sign)
    if n_bins_per_sign == 3:
        names = {1: "low", 2: "med", 3: "high"}
        return f"{sign}_{names[int(tier)]}"
    return f"{sign}_q{int(tier)}"


def _make_bin_order(n_bins_per_sign: int) -> List[str]:
    n_bins_per_sign = int(n_bins_per_sign)
    if n_bins_per_sign < 1:
        raise ValueError("n_bins_per_sign must be >= 1")
    neg = [_bin_label("neg", t, n_bins_per_sign) for t in range(n_bins_per_sign, 0, -1)]
    pos = [_bin_label("pos", t, n_bins_per_sign) for t in range(1, n_bins_per_sign + 1)]
    return neg + pos


def _fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    xr = float(x)
    if abs(xr - round(xr)) < 1e-9:
        return str(int(round(xr)))
    return f"{xr:.1f}"


def _assign_bins_signed_quantiles(
    df: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
    n_bins_per_sign: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Assign bins (neg/pos x quantiles) using sign-specific quantiles of |value|.

    Returns:
      - dataframe with `out_col` containing bin labels
      - bounds dict with cutpoints and observed extrema per sign
    """
    d = df.copy()
    x = pd.to_numeric(d[value_col], errors="coerce")
    sign = np.where(x < 0, "neg", "pos")
    mag = x.abs()

    out = pd.Series([np.nan] * len(d), index=d.index, dtype=object)
    bounds: Dict[str, Dict[str, float]] = {}

    k = int(n_bins_per_sign)
    if k < 1:
        raise ValueError("n_bins_per_sign must be >= 1")

    for sgn in ["neg", "pos"]:
        idx = (sign == sgn) & np.isfinite(mag) & np.isfinite(x)
        mags = mag[idx].astype(float)
        xs = x[idx].astype(float)
        if mags.empty:
            continue

        if k == 1:
            edges = np.array([float(np.min(mags.values)), float(np.max(mags.values))], dtype=float)
        else:
            qs = np.linspace(0.0, 1.0, k + 1)
            edges = np.quantile(mags.values, qs)
            edges[0] = float(np.min(mags.values))
            edges[-1] = float(np.max(mags.values))
            edges = np.maximum.accumulate(edges)

        tier_idx = np.digitize(mags.values, edges[1:-1], right=True) + 1
        out.loc[idx] = [_bin_label(sgn, int(t), k) for t in tier_idx.tolist()]

        bounds[sgn] = {
            "n_bins": float(k),
            "x_min": float(np.min(xs.values)),
            "x_max": float(np.max(xs.values)),
            "mag_min": float(np.min(mags.values)),
            "mag_max": float(np.max(mags.values)),
            **{f"e{i}": float(edges[i]) for i in range(len(edges))},
        }

    d[out_col] = out
    return d, bounds


def _assign_bins_from_bounds(
    df: pd.DataFrame,
    *,
    value_col: str,
    out_col: str,
    bounds: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Assign sign x quantile bins using precomputed sign-specific edges.

    Useful when binning external predictions/simulations into the same bins
    as the observed data.
    """
    d = df.copy()
    x = pd.to_numeric(d[value_col], errors="coerce")
    sign = np.where(x < 0, "neg", "pos")
    mag = x.abs()
    out = pd.Series([np.nan] * len(d), index=d.index, dtype=object)

    for sgn in ["neg", "pos"]:
        b = bounds.get(sgn)
        if not b:
            continue
        k = int(float(b.get("n_bins", 0)))
        if k <= 0:
            continue
        edges = [b.get(f"e{i}", float("nan")) for i in range(k + 1)]
        edges = np.asarray(edges, dtype=float)
        if not np.isfinite(edges).all():
            continue

        idx = (sign == sgn) & np.isfinite(mag) & np.isfinite(x)
        mags = mag[idx].astype(float)
        if mags.empty:
            continue
        tier_idx = np.digitize(mags.values, edges[1:-1], right=True) + 1
        out.loc[idx] = [_bin_label(sgn, int(t), k) for t in tier_idx.tolist()]

    d[out_col] = out
    return d


def _bin_tick_labels(bounds: Dict[str, Dict[str, float]], *, bin_order: List[str]) -> List[str]:
    """Generate tick labels for a given bin_order including numeric boundaries."""

    def _get_edges(sign: str) -> Optional[np.ndarray]:
        b = bounds.get(sign)
        if not b:
            return None
        k = int(float(b.get("n_bins", 0)))
        if k <= 0:
            return None
        edges = [b.get(f"e{i}", float("nan")) for i in range(k + 1)]
        edges = np.asarray(edges, dtype=float)
        if not np.isfinite(edges).all():
            return None
        return edges

    edges_neg = _get_edges("neg")
    edges_pos = _get_edges("pos")

    def _label_range(label: str) -> str:
        if label.startswith("neg_"):
            edges = edges_neg
            sign = "neg"
        else:
            edges = edges_pos
            sign = "pos"
        if edges is None:
            return label

        tier: Optional[int] = None
        if label.endswith("_low"):
            tier = 1
        elif label.endswith("_med"):
            tier = 2
        elif label.endswith("_high"):
            tier = 3
        else:
            m = re.search(r"_q(\d+)$", label)
            if m:
                tier = int(m.group(1))
        if tier is None:
            return label

        k = int(len(edges) - 1)
        if not (1 <= tier <= k):
            return label

        mag_lo = float(edges[tier - 1])
        mag_hi = float(edges[tier])
        if sign == "pos":
            lo, hi = mag_lo, mag_hi
        else:
            lo, hi = -mag_hi, -mag_lo
        return f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"

    return [_label_range(b) for b in bin_order]


def _parse_bin_tier(label: str) -> Optional[int]:
    if label.endswith("_low"):
        return 1
    if label.endswith("_med"):
        return 2
    if label.endswith("_high"):
        return 3
    m = re.search(r"_q(\d+)$", label)
    if m:
        return int(m.group(1))
    return None


def _bin_midpoint_from_bounds(label: str, bounds: Dict[str, Dict[str, float]]) -> Optional[float]:
    sign = "neg" if label.startswith("neg_") else "pos"
    b = bounds.get(sign)
    if not b:
        return None
    k = int(float(b.get("n_bins", 0)))
    if k <= 0:
        return None
    edges = np.asarray([b.get(f"e{i}", float("nan")) for i in range(k + 1)], dtype=float)
    if not np.isfinite(edges).all():
        return None

    tier = _parse_bin_tier(label)
    if tier is None or not (1 <= tier <= k):
        return None

    mag_lo = float(edges[tier - 1])
    mag_hi = float(edges[tier])
    if sign == "pos":
        lo, hi = mag_lo, mag_hi
    else:
        lo, hi = -mag_hi, -mag_lo
    return 0.5 * (lo + hi)


def _bin_x_positions_from_data(
    d: pd.DataFrame,
    *,
    bin_col: str,
    x_col: str,
    bin_order: List[str],
    bounds: Dict[str, Dict[str, float]],
) -> np.ndarray:
    dd = d.dropna(subset=[bin_col, x_col]).copy()
    dd[x_col] = pd.to_numeric(dd[x_col], errors="coerce")
    dd = dd.dropna(subset=[x_col]).copy()

    med = dd.groupby(bin_col)[x_col].median() if len(dd) > 0 else pd.Series(dtype=float)
    med = med.reindex(bin_order)

    xs = np.full(len(bin_order), np.nan, dtype=float)
    for i, label in enumerate(bin_order):
        v = med.get(label, np.nan)
        if np.isfinite(v):
            xs[i] = float(v)
            continue
        midpoint = _bin_midpoint_from_bounds(label, bounds)
        if midpoint is not None and np.isfinite(midpoint):
            xs[i] = float(midpoint)

    for i in range(len(xs)):
        if not np.isfinite(xs[i]):
            xs[i] = float(i)
    return xs


# ---------------------------------------------------------------------------
# Error bar helpers
# ---------------------------------------------------------------------------


def _sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size <= 1:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def _ci95_halfwidth(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n <= 1:
        return float("nan")
    se = float(np.std(v, ddof=1) / np.sqrt(n))
    return float(1.959963984540054 * se)


def _error_halfwidth(values: np.ndarray, *, error_bars: str) -> float:
    mode = str(error_bars).strip().lower()
    if mode == "sem":
        return _sem(values)
    if mode == "ci95":
        return _ci95_halfwidth(values)
    raise ValueError(f"Unknown error_bars: {error_bars}")


def _summarize_mean_err_across_subjects(
    d: pd.DataFrame,
    *,
    bin_col: str,
    y_col: str,
    error_bars: str,
) -> pd.DataFrame:
    """Compute bin-wise mean and error across subjects (equal subject weighting)."""
    dd = d.dropna(subset=["subject_id", bin_col, y_col]).copy()
    dd[y_col] = pd.to_numeric(dd[y_col], errors="coerce")
    dd = dd.dropna(subset=[y_col]).copy()

    subj = (
        dd.groupby(["subject_id", bin_col], as_index=False)
        .agg(y=(y_col, "mean"))
        .copy()
    )
    grp = (
        subj.groupby(bin_col, as_index=False)
        .agg(
            mean=("y", "mean"),
            err=("y", lambda s: _error_halfwidth(s.to_numpy(dtype=float), error_bars=error_bars)),
            n_subjects=("y", "size"),
        )
        .copy()
    )

    grp[bin_col] = grp[bin_col].astype(str)
    grp = grp.set_index(bin_col).reindex(BIN_ORDER).reset_index()
    return grp


def _normalize_subject_id(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _is_image_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split("_")
    return len(parts) == 4 and all(len(p) > 0 for p in parts)


def _load_encoding_values(main_file: Path) -> pd.DataFrame:
    """Load encoding item/outcome pairs from a MAIN logfile."""
    df = pd.read_csv(main_file)
    enc = df[(df.get("phase") == "encoding") & (df.get("event").isin(["value", "image"]))].copy()
    if not {"game", "image", "outcome"}.issubset(enc.columns):
        raise ValueError(f"Behavioral file missing required columns: {main_file}")
    enc = enc[["game", "image", "outcome", "event"]].dropna(subset=["game", "image", "outcome"]).copy()
    enc["game"] = pd.to_numeric(enc["game"], errors="coerce")
    enc["outcome"] = pd.to_numeric(enc["outcome"], errors="coerce")
    enc = enc.dropna(subset=["game", "outcome"]).copy()
    enc["game"] = enc["game"].astype(int)
    enc["image"] = enc["image"].astype(str).str.strip()

    # Prefer event='value' if both are present.
    enc["event_priority"] = (enc["event"].astype(str) != "value").astype(int)
    enc = enc.sort_values(["game", "image", "event_priority"], kind="mergesort")
    enc = enc.drop_duplicates(subset=["game", "image"], keep="first").copy()
    return enc[["game", "image", "outcome"]].reset_index(drop=True)


def _extract_memory_value_order(main_file: Path) -> Dict[int, List[str]]:
    """Return the cue order for value recall per game."""
    df = pd.read_csv(main_file)
    mem = df[(df.get("phase") == "memory") & (df.get("event") == "value_recall")].copy()
    if mem.empty or not {"game", "image"}.issubset(mem.columns):
        return {}
    cols = ["game", "image"] + (["onset"] if "onset" in mem.columns else [])
    mem = mem[cols].dropna(subset=["game", "image"]).copy()
    mem["game"] = pd.to_numeric(mem["game"], errors="coerce")
    mem = mem.dropna(subset=["game"]).copy()
    mem["game"] = mem["game"].astype(int)
    mem["image"] = mem["image"].astype(str).str.strip()
    if "onset" in mem.columns:
        mem["onset"] = pd.to_numeric(mem["onset"], errors="coerce")
        mem = mem.sort_values(["game", "onset"], kind="mergesort")
    out: Dict[int, List[str]] = {}
    for g, gdf in mem.groupby("game"):
        out[int(g)] = gdf["image"].astype(str).tolist()
    return out


def _parse_sign_token(tok: object) -> Optional[int]:
    if tok is None:
        return None
    t = str(tok).strip().lower()
    if t in {"plus", "+", "positive", "pos"}:
        return 1
    if t in {"minus", "-", "negative", "neg"}:
        return -1
    # If this is a magnitude token, not a sign.
    if t.isdigit():
        return None
    return None


def _parse_magnitude_token(tok: object) -> Optional[int]:
    if tok is None:
        return None
    t = str(tok).strip()
    if t.isdigit():
        v = int(t)
        if 1 <= v <= 9:
            return v
    return None


def _build_recalled_values_map(
    main_file: Path,
    value_recall_file: Path,
) -> Dict[int, Dict[str, float]]:
    """Return map: game -> {image -> recalled_value}.

    Aligns value-recall rows to the cue order in the main logfile.
    Missing/malformed pairs become NaN.
    """
    if not value_recall_file.exists():
        return {}
    order = _extract_memory_value_order(main_file)
    if not order:
        return {}

    vr = pd.read_csv(value_recall_file)
    if "game" not in vr.columns:
        return {}
    token_col = (
        "original_item" if "original_item" in vr.columns
        else ("item" if "item" in vr.columns else None)
    )
    if token_col is None:
        return {}
    vr["game"] = pd.to_numeric(vr["game"], errors="coerce")
    vr = vr.dropna(subset=["game"]).copy()
    vr["game"] = vr["game"].astype(int)
    vr[token_col] = vr[token_col].astype(str).str.strip()

    out: Dict[int, Dict[str, float]] = {}
    for g, gdf in vr.groupby("game"):
        game = int(g)
        items = order.get(game, [])
        if len(items) == 0:
            continue

        rows = gdf.reset_index(drop=True)
        vals: List[float] = []
        i = 0
        while i < len(rows):
            sign_tok = rows.loc[i, token_col]
            mag_tok = rows.loc[i + 1, token_col] if (i + 1) < len(rows) else None
            sign = _parse_sign_token(sign_tok)
            mag = _parse_magnitude_token(mag_tok)
            if sign is not None and mag is not None:
                vals.append(float(sign * mag))
                i += 2
            else:
                # Try reversed ordering (mag then sign)
                sign_rev = _parse_sign_token(mag_tok)
                mag_rev = _parse_magnitude_token(sign_tok)
                if sign_rev is not None and mag_rev is not None:
                    vals.append(float(sign_rev * mag_rev))
                    i += 2
                else:
                    vals.append(float("nan"))
                    i += 1

        img_to_val: Dict[str, float] = {}
        for idx, img in enumerate(items):
            v = vals[idx] if idx < len(vals) else float("nan")
            img_to_val[str(img)] = float(v)
        out[game] = img_to_val

    return out


def _option_tokens_from_images(images: Iterable[str]) -> List[str]:
    toks: List[str] = []
    for img in images:
        toks.extend(str(img).split("_"))
    return sorted(set(toks))


def _compute_true_offer_map(enc_vals: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """Compute true offer value (sum of relevant item outcomes) per (game, option)."""
    out: Dict[Tuple[int, str], float] = {}
    option_tokens = _option_tokens_from_images(enc_vals["image"].astype(str).tolist())

    for game, gdf in enc_vals.groupby("game"):
        imgs = gdf[["image", "outcome"]].copy()
        imgs["image"] = imgs["image"].astype(str)
        imgs["outcome"] = pd.to_numeric(imgs["outcome"], errors="coerce")
        imgs = imgs.dropna(subset=["outcome"]).copy()
        if imgs.empty:
            continue
        for opt in option_tokens:
            mask = imgs["image"].apply(lambda s: opt in str(s).split("_"))
            if not bool(mask.any()):
                continue
            out[(int(game), str(opt))] = float(imgs.loc[mask, "outcome"].sum())
    return out


def _compute_relevant_images_map(enc_vals: pd.DataFrame) -> Dict[Tuple[int, str], List[str]]:
    """Return list of items relevant to each (game, option)."""
    out: Dict[Tuple[int, str], List[str]] = {}
    option_tokens = _option_tokens_from_images(enc_vals["image"].astype(str).tolist())

    for game, gdf in enc_vals.groupby("game"):
        imgs = gdf[["image"]].copy()
        imgs["image"] = imgs["image"].astype(str)
        for opt in option_tokens:
            rel = [s for s in imgs["image"].tolist() if opt in str(s).split("_")]
            if not rel:
                continue
            out[(int(game), str(opt))] = rel
    return out


def _compute_recalled_offer_map(
    enc_vals: pd.DataFrame,
    recalled_values_map: Dict[int, Dict[str, float]],
) -> Dict[Tuple[int, str], float]:
    """Compute recalled offer value (sum of recalled relevant item values) per (game, option)."""
    out: Dict[Tuple[int, str], float] = {}
    rel_imgs_map = _compute_relevant_images_map(enc_vals)
    option_tokens = _option_tokens_from_images(enc_vals["image"].astype(str).tolist())

    for game in sorted(enc_vals["game"].astype(int).unique().tolist()):
        img_to_val = recalled_values_map.get(int(game), {})
        for opt in option_tokens:
            rel_imgs = rel_imgs_map.get((int(game), str(opt)), [])
            if not rel_imgs:
                continue
            vals = np.asarray(
                [img_to_val.get(str(img), float("nan")) for img in rel_imgs],
                dtype=float,
            )
            if vals.size == 0 or not np.isfinite(vals).any():
                out[(int(game), str(opt))] = float("nan")
            else:
                out[(int(game), str(opt))] = float(np.nansum(vals))
    return out


def _load_trial_level_fix_summaries(
    clean_fixations_csv: Path,
    *,
    exclude_subjects: Iterable[str],
) -> pd.DataFrame:
    """Load per-trial fixation summaries from the clean choice fixations CSV.

    Inputs:
        clean_fixations_csv: output/choice_fixations_clean_buffer_50.csv
            (from analysis/lib/prepare_choice_fixations.py)
        exclude_subjects: subject IDs to exclude

    Returns a DataFrame with one row per (subject_id, game, trial_number, option),
    with columns: rt_s, total_fix_time_ms, total_fix_time_s, n_fixations.
    """
    df = pd.read_csv(clean_fixations_csv)
    if df.empty:
        return pd.DataFrame([])

    df["subject_id"] = df["subject_id"].astype(str)
    exclude_set = set(str(s) for s in exclude_subjects)
    if exclude_set:
        df = df[~df["subject_id"].isin(exclude_set)].copy()

    df["fixation_duration"] = pd.to_numeric(df.get("fixation_duration"), errors="coerce")
    df["rt"] = pd.to_numeric(df.get("rt"), errors="coerce")
    df["game"] = pd.to_numeric(df.get("game"), errors="coerce")
    df["trial_number"] = pd.to_numeric(df.get("trial_number"), errors="coerce")
    df = df.dropna(subset=["subject_id", "game", "trial_number", "option"]).copy()
    df["game"] = df["game"].astype(int)
    df["trial_number"] = df["trial_number"].astype(int)
    df["option"] = df["option"].astype(str).str.strip()

    trial = (
        df.groupby(["subject_id", "game", "trial_number", "option"], as_index=False)
        .agg(
            rt_s=("rt", "first"),
            total_fix_time_ms=("fixation_duration", "sum"),
            n_fixations=("fixation_duration", "size"),
        )
        .copy()
    )
    trial["rt_s"] = pd.to_numeric(trial["rt_s"], errors="coerce")
    trial["total_fix_time_ms"] = pd.to_numeric(trial["total_fix_time_ms"], errors="coerce")
    trial["total_fix_time_s"] = trial["total_fix_time_ms"] / 1000.0
    return trial


def _attach_offer_values(
    trial_df: pd.DataFrame,
    *,
    data_root: Path,
) -> pd.DataFrame:
    """Attach true and recalled offer values to trial-level fixation summaries.

    For each subject, loads encoding outcomes and value recall transcripts
    from data/<subid>/ and computes offer values per (game, option).

    Inputs:
        trial_df: output of _load_trial_level_fix_summaries
        data_root: path to the data/ directory (relative to the repository root)
    """
    if trial_df.empty:
        return trial_df

    out_frames: List[pd.DataFrame] = []
    for sid, sub in trial_df.groupby("subject_id", sort=True):
        main_file = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        vr_file = data_root / sid / "valuerecall" / f"{sid}_valuerecall.csv"
        if not main_file.exists():
            continue

        enc = _load_encoding_values(main_file)
        true_offer_map = _compute_true_offer_map(enc)
        recalled_values_map = _build_recalled_values_map(main_file, vr_file)
        recalled_offer_map = _compute_recalled_offer_map(enc, recalled_values_map) if recalled_values_map else {}

        sub = sub.copy()
        sub["true_offer_value"] = [
            true_offer_map.get((int(g), str(opt)), float("nan"))
            for g, opt in zip(sub["game"].tolist(), sub["option"].tolist())
        ]
        sub["recalled_offer_value"] = [
            recalled_offer_map.get((int(g), str(opt)), float("nan"))
            for g, opt in zip(sub["game"].tolist(), sub["option"].tolist())
        ]
        out_frames.append(sub)

    if not out_frames:
        return pd.DataFrame([])
    out = pd.concat(out_frames, ignore_index=True)
    out["true_offer_value"] = pd.to_numeric(out["true_offer_value"], errors="coerce")
    out["recalled_offer_value"] = pd.to_numeric(out["recalled_offer_value"], errors="coerce")
    return out


def _load_choice_accepts(main_file: Path) -> pd.DataFrame:
    """Load choice accept/take (0/1) for a subject from the MAIN logfile."""
    df = pd.read_csv(main_file)
    if df.empty:
        return pd.DataFrame([])

    need = {"phase", "event", "game", "trial_number", "option", "choice"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame([])

    ch = df[(df["phase"] == "choice") & (df["event"] == "choice")].copy()
    if ch.empty:
        return pd.DataFrame([])

    ch["game"] = pd.to_numeric(ch["game"], errors="coerce")
    ch["trial_number"] = pd.to_numeric(ch["trial_number"], errors="coerce")
    ch["choice"] = pd.to_numeric(ch["choice"], errors="coerce")
    ch["option"] = ch["option"].astype(str).str.strip()
    ch = ch.dropna(subset=["game", "trial_number", "choice", "option"]).copy()
    if ch.empty:
        return pd.DataFrame([])

    ch["game"] = ch["game"].astype(int)
    ch["trial_number"] = ch["trial_number"].astype(int)

    # Defensive collapse: one row per (game, trial_number, option).
    ch = (
        ch.groupby(["game", "trial_number", "option"], as_index=False)
        .agg(choice=("choice", "first"))
        .copy()
    )

    # In this dataset: 1=take/accept, 2=leave.
    ch["accept"] = (ch["choice"].astype(float) == 1.0).astype(float)
    return ch[["game", "trial_number", "option", "accept"]]


def _attach_accept_choices(
    trial_df: pd.DataFrame,
    *,
    data_root: Path,
) -> pd.DataFrame:
    """Attach accept (p(take) at trial level) to the trial-level fixation summaries."""
    if trial_df.empty:
        return trial_df

    out_frames: List[pd.DataFrame] = []
    for sid, sub in trial_df.groupby("subject_id", sort=True):
        main_file = data_root / sid / f"{sid}_MAIN_logfile_7.csv"
        if not main_file.exists():
            sub = sub.copy()
            sub["accept"] = np.nan
            out_frames.append(sub)
            continue

        ch = _load_choice_accepts(main_file)
        sub2 = sub.merge(ch, on=["game", "trial_number", "option"], how="left")
        out_frames.append(sub2)

    out = pd.concat(out_frames, ignore_index=True) if out_frames else trial_df.copy()
    out["accept"] = pd.to_numeric(out.get("accept"), errors="coerce")
    return out


# ---------------------------------------------------------------------------
# PPC helpers
# ---------------------------------------------------------------------------


def _mean_or_nan(x: pd.Series) -> float:
    """Return the mean of finite values in x, or NaN if none."""
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def _load_mean_params_from_kfold_summary(
    kfold_csv: Path,
) -> Tuple["ADDMParameters", Dict[str, float], Dict[str, object]]:
    """Load mean fitted parameters from a kfold summary CSV.

    Returns:
        params:      ADDMParameters with mean estimates across folds
        param_info:  dict of scalar parameter values
        config_info: dict of fitting configuration settings
    """
    df = pd.read_csv(kfold_csv)

    if "mode" in df.columns:
        df = df[df["mode"].astype(str) == "fit"].copy()

    if df.empty:
        raise ValueError(f"No fit rows found in: {kfold_csv}")

    def _first(col: str, default: object) -> object:
        return df[col].iloc[0] if col in df.columns and len(df[col]) else default

    noise_param = str(_first("noise_param", "sigma"))
    time_col = str(_first("time_col", "fix_ms"))
    irrelevant_mode = str(_first("irrelevant_mode", "zero"))
    include_transition = int(float(_first("include_transition", 0)))
    include_center_fixations = int(float(_first("include_center_fixations", 0)))
    center_gaze_mode = str(_first("center_gaze_mode", "separate"))
    center_mode = str(_first("center_mode", "same_as_irrelevant"))

    d_hat = _mean_or_nan(df.get("d", pd.Series(dtype=float)))
    theta_hat = _mean_or_nan(df.get("theta", pd.Series(dtype=float)))
    mu_hat = _mean_or_nan(df.get("mu", pd.Series(dtype=float)))

    # sigma in the CSV is already in simulator space.
    sigma_hat = _mean_or_nan(df.get("sigma", pd.Series(dtype=float)))
    phi_center_hat = _mean_or_nan(df.get("phi_center", pd.Series(dtype=float)))
    if not np.isfinite(phi_center_hat):
        phi_center_hat = 1.0

    if noise_param == "mu" and np.isfinite(d_hat) and np.isfinite(mu_hat):
        sigma_from_mu = float(d_hat) * float(mu_hat)
        if not np.isfinite(sigma_hat):
            sigma_hat = sigma_from_mu
    if not np.isfinite(sigma_hat):
        raise ValueError("Could not infer sigma from k-fold CSV (missing/NaN sigma and mu)")

    params = ADDMParameters(
        d=float(d_hat),
        theta=float(theta_hat),
        sigma=float(sigma_hat),
        phi_center=float(phi_center_hat),
    )

    param_info = {
        "d": float(d_hat),
        "theta": float(theta_hat),
        "sigma": float(sigma_hat),
        "mu": float(mu_hat) if np.isfinite(mu_hat) else float("nan"),
        "phi_center": float(phi_center_hat),
    }

    config_info: Dict[str, object] = {
        "noise_param": noise_param,
        "time_col": time_col,
        "irrelevant_mode": irrelevant_mode,
        "include_transition": int(include_transition),
        "include_center_fixations": int(include_center_fixations),
        "center_gaze_mode": center_gaze_mode,
        "center_mode": center_mode,
    }
    return params, param_info, config_info


def _choose_sim_time_mean_col(df_sim: pd.DataFrame, *, time_col: str) -> str:
    """Choose the simulated time mean column corresponding to an observed time_col."""
    t = str(time_col)
    if t in {"rt_ms", "RT_MS"}:
        if "rt_ms_sim_mean" in df_sim.columns:
            return "rt_ms_sim_mean"
        if "time_ms_sim_mean" in df_sim.columns:
            return "time_ms_sim_mean"
        raise KeyError("Expected 'rt_ms_sim_mean' in simulation outputs")
    if t in {"fix_ms", "FIX_MS"}:
        if "fix_ms_sim_mean" in df_sim.columns:
            return "fix_ms_sim_mean"
        if "time_ms_sim_mean" in df_sim.columns:
            return "time_ms_sim_mean"
        raise KeyError("Expected 'fix_ms_sim_mean' in simulation outputs")

    # Fallback: prefer the standardized time column if present.
    if "time_ms_sim_mean" in df_sim.columns:
        return "time_ms_sim_mean"
    if "rt_ms_sim_mean" in df_sim.columns:
        return "rt_ms_sim_mean"
    if "fix_ms_sim_mean" in df_sim.columns:
        return "fix_ms_sim_mean"
    raise KeyError("Could not find a simulated time mean column in simulation outputs")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _simulate_addm_trialsim_from_kfold(
    kfold_csv: Path,
    *,
    output_dir: Path,
    data_dir: Optional[Path],
    reward_source: str,
    n_sim_per_trial: int,
    seed: int,
    dt_ms: float,
    cache_dir: Optional[Path] = None,
    fixation_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run (or load cached) per-trial aDDM simulations from a kfold summary CSV.

    Returns a DataFrame with columns:
        - subject_id, game, trial_number, v_offer
        - time_ms_sim_mean
        - accept_sim_mean

    Notes
    -----
    - Uses *mean* parameter estimates across folds.
    - Uses the same Option-B generative gaze model used in fitting.
    - For reward_source='recalled', pulls value-recall transcriptions from data_dir.
    - data_dir should point to data/ (flat per-subject layout).
    - fixation_data_dir: if provided, used for subject discovery and fixation
      CSV loading (matching the kfold fitting pipeline's --fixation-data-dir).
    """
    if n_sim_per_trial <= 0:
        raise ValueError("n_sim_per_trial must be >= 1")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")

    kfold_csv = Path(kfold_csv).expanduser().resolve()
    if not kfold_csv.exists():
        raise FileNotFoundError(f"Missing kfold summary CSV: {kfold_csv}")

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Include the parent directory name in the cache key because multiple
        # kfold directories can contain identically named summary CSVs.
        cache_key = f"{kfold_csv.parent.name}_{kfold_csv.stem}"
        cache_path = cache_dir / (
            f"addm_trialsim_{cache_key}_nsim{int(n_sim_per_trial)}_seed{int(seed)}_dt{dt_ms:g}.csv"
        )
        if cache_path.exists():
            df_cached = pd.read_csv(cache_path)
            needed = {"subject_id", "game", "trial_number", "v_offer", "time_ms_sim_mean", "accept_sim_mean"}
            if needed.issubset(df_cached.columns):
                print(f"  [cache hit] {cache_path}")
                return df_cached

    params, _, cfg0 = _load_mean_params_from_kfold_summary(kfold_csv)
    df_fit = pd.read_csv(kfold_csv)
    n_v_bins = int(float(df_fit["n_v_bins"].iloc[0])) if "n_v_bins" in df_fit.columns else 7
    rt_bins_max = int(float(df_fit["rt_bins_max"].iloc[0])) if "rt_bins_max" in df_fit.columns else 15
    rt_bins_fixed = int(float(df_fit["rt_bins_fixed"].iloc[0])) if "rt_bins_fixed" in df_fit.columns else 0
    min_trials_per_rt_bin = (
        int(float(df_fit["min_trials_per_rt_bin"].iloc[0]))
        if "min_trials_per_rt_bin" in df_fit.columns
        else 25
    )

    include_transition = bool(int(cfg0.get("include_transition", 0)))
    include_center_fixations = bool(int(cfg0.get("include_center_fixations", 0)))

    config = FitConfig(
        binning=BinningConfig(
            n_v_offer_bins=int(n_v_bins),
            rt_bins_max=int(rt_bins_max),
            min_trials_per_rt_bin=int(min_trials_per_rt_bin),
            rt_bins_fixed=int(rt_bins_fixed),
        ),
        sim=SimulationConfig(
            dt_ms=float(dt_ms),
            n_sim_per_vbin=1,  # not used in generative trial simulation
            alpha_smoothing=1.0,
            seed=int(seed),
            include_transition_time=bool(include_transition),
            irrelevant_mode=str(cfg0.get("irrelevant_mode", "zero")),
            include_center_fixations=bool(include_center_fixations),
            center_gaze_mode=str(cfg0.get("center_gaze_mode", "separate")),
            center_mode=str(cfg0.get("center_mode", "same_as_irrelevant")),
        ),
        time_col=str(cfg0.get("time_col", "fix_ms")),
    )

    df_trials = load_group_trial_templates(
        output_dir=output_dir,
        include_center_fixations=bool(include_center_fixations),
        reward_source=str(reward_source),
        data_dir=data_dir,
        fixation_data_dir=fixation_data_dir,
    )
    df_fix = load_group_fixations(
        output_dir=output_dir,
        include_center_fixations=bool(include_center_fixations),
        fixation_data_dir=fixation_data_dir,
    )
    df_trials_b, _, components = build_fitting_components(
        df_trials_train=df_trials,
        df_fix_train=df_fix,
        config=config,
    )

    df_sim = simulate_trials_generative_detailed(
        df_trials=df_trials_b,
        params=params,
        config=config,
        components=components,
        n_sim_per_trial=int(n_sim_per_trial),
        seed=int(seed),
    )

    sim_col = _choose_sim_time_mean_col(df_sim, time_col=str(cfg0.get("time_col", "fix_ms")))
    key = ["subject", "game", "trial_number"]
    sim_cols = [sim_col]
    if "accept_sim_mean" in df_sim.columns:
        sim_cols.append("accept_sim_mean")
    df_out = df_trials_b[key + ["v_offer"]].merge(df_sim[key + sim_cols], on=key, how="inner")
    df_out = df_out.rename(columns={"subject": "subject_id", sim_col: "time_ms_sim_mean"})
    df_out["subject_id"] = df_out["subject_id"].astype(str)
    df_out["game"] = pd.to_numeric(df_out["game"], errors="coerce")
    df_out["trial_number"] = pd.to_numeric(df_out["trial_number"], errors="coerce")
    df_out["v_offer"] = pd.to_numeric(df_out["v_offer"], errors="coerce")
    df_out["time_ms_sim_mean"] = pd.to_numeric(df_out["time_ms_sim_mean"], errors="coerce")
    if "accept_sim_mean" in df_out.columns:
        df_out["accept_sim_mean"] = pd.to_numeric(df_out["accept_sim_mean"], errors="coerce")
    else:
        df_out["accept_sim_mean"] = np.nan

    if cache_path is not None:
        df_out.to_csv(cache_path, index=False)
        print(f"  [saved sim cache] {cache_path}")

    return df_out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _fit_logistic_curve(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> Optional[np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if xx.size < 3 or np.unique(xx).size < 2:
        return None

    yy = np.clip(yy, 1e-6, 1 - 1e-6)
    try:
        X = sm.add_constant(xx)
        model = sm.GLM(yy, X, family=sm.families.Binomial())
        fit = model.fit()
        Xg = sm.add_constant(x_grid)
        yhat = fit.predict(Xg)
        return np.asarray(yhat, dtype=float)
    except Exception:
        return None


def _fit_quadratic_curve(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> Optional[np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]
    yy = yy[ok]
    if xx.size < 3 or np.unique(xx).size < 3:
        return None

    try:
        p = np.polyfit(xx, yy, deg=2)
        return np.polyval(p, x_grid)
    except Exception:
        return None


def _plot_data_and_models(
    ax: plt.Axes,
    *,
    x_positions: np.ndarray,
    tick_labels: List[str],
    summ_data: pd.DataFrame,
    summ_free3: pd.DataFrame,
    summ_theta1: pd.DataFrame,
    y_label: str,
    title: str,
    x_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
) -> None:
    xs = np.asarray(x_positions, dtype=float)

    y_data = pd.to_numeric(summ_data.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_data = pd.to_numeric(summ_data.get("err"), errors="coerce").to_numpy(dtype=float)
    y_free3 = pd.to_numeric(summ_free3.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_free3 = pd.to_numeric(summ_free3.get("err"), errors="coerce").to_numpy(dtype=float)
    y_theta1 = pd.to_numeric(summ_theta1.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_theta1 = pd.to_numeric(summ_theta1.get("err"), errors="coerce").to_numpy(dtype=float)

    ax.plot(xs, y_data, color="black", linewidth=3, label="Data")
    mask_data = np.isfinite(y_data) & np.isfinite(e_data)
    if np.any(mask_data):
        ax.fill_between(xs, y_data - e_data, y_data + e_data, where=mask_data, color="0.5", alpha=0.18, linewidth=0)

    ax.plot(xs, y_free3, color="C0", linewidth=3, label="aDDM")
    mask_f3 = np.isfinite(y_free3) & np.isfinite(e_free3)
    if np.any(mask_f3):
        ax.fill_between(xs, y_free3 - e_free3, y_free3 + e_free3, where=mask_f3, color="C0", alpha=0.18, linewidth=0)

    ax.plot(xs, y_theta1, color="C1", linewidth=3, label="DDM")
    mask_t1 = np.isfinite(y_theta1) & np.isfinite(e_theta1)
    if np.any(mask_t1):
        ax.fill_between(xs, y_theta1 - e_theta1, y_theta1 + e_theta1, where=mask_t1, color="C1", alpha=0.18, linewidth=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=18)
    if np.any(np.isfinite(xs)):
        x0 = float(np.nanmin(xs))
        x1 = float(np.nanmax(xs))
        dx = max(0.5, 0.06 * (x1 - x0)) if x1 > x0 else 0.5
        ax.set_xlim(x0 - dx, x1 + dx)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=16)


def _plot_data_and_models_with_fits(
    ax: plt.Axes,
    *,
    x_positions: np.ndarray,
    tick_labels: List[str],
    summ_data: pd.DataFrame,
    summ_free3: pd.DataFrame,
    summ_theta1: pd.DataFrame,
    raw_x_data: np.ndarray,
    raw_y_data: np.ndarray,
    raw_x_free3: np.ndarray,
    raw_y_free3: np.ndarray,
    raw_x_theta1: np.ndarray,
    raw_y_theta1: np.ndarray,
    fit_kind: str,
    y_label: str,
    title: str,
    x_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
) -> None:
    xs = np.asarray(x_positions, dtype=float)
    x_all = np.concatenate([
        np.asarray(raw_x_data, dtype=float),
        np.asarray(raw_x_free3, dtype=float),
        np.asarray(raw_x_theta1, dtype=float),
    ])
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size == 0:
        x_all = xs[np.isfinite(xs)]
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = -1.0, 1.0
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_grid = np.linspace(x_min, x_max, 400)

    if str(fit_kind).lower() == "logistic":
        fit_func = _fit_logistic_curve
    elif str(fit_kind).lower() == "quadratic":
        fit_func = _fit_quadratic_curve
    else:
        raise ValueError(f"Unknown fit_kind: {fit_kind}")

    y_data_curve = fit_func(raw_x_data, raw_y_data, x_grid)
    y_free3_curve = fit_func(raw_x_free3, raw_y_free3, x_grid)
    y_theta1_curve = fit_func(raw_x_theta1, raw_y_theta1, x_grid)

    if y_data_curve is not None:
        ax.plot(x_grid, y_data_curve, color="black", linewidth=3, label="Data")
    if y_free3_curve is not None:
        ax.plot(x_grid, y_free3_curve, color="C0", linewidth=3, label="aDDM")
    if y_theta1_curve is not None:
        ax.plot(x_grid, y_theta1_curve, color="C1", linewidth=3, label="DDM")

    # Binned points with SEM (no bands)
    y_data = pd.to_numeric(summ_data.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_data = pd.to_numeric(summ_data.get("err"), errors="coerce").to_numpy(dtype=float)
    y_free3 = pd.to_numeric(summ_free3.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_free3 = pd.to_numeric(summ_free3.get("err"), errors="coerce").to_numpy(dtype=float)
    y_theta1 = pd.to_numeric(summ_theta1.get("mean"), errors="coerce").to_numpy(dtype=float)
    e_theta1 = pd.to_numeric(summ_theta1.get("err"), errors="coerce").to_numpy(dtype=float)

    for yv, ev, col in [
        (y_data, e_data, "black"),
        (y_free3, e_free3, "C0"),
        (y_theta1, e_theta1, "C1"),
    ]:
        ok = np.isfinite(xs) & np.isfinite(yv)
        if np.any(ok):
            ax.errorbar(
                xs[ok],
                yv[ok],
                yerr=ev[ok],
                fmt="o",
                color=col,
                ecolor=col,
                elinewidth=1.2,
                capsize=0,
                markersize=5,
                zorder=4,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=18)

    dx = max(0.5, 0.06 * (x_max - x_min))
    ax.set_xlim(x_min - dx, x_max + dx)

    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Three-column aDDM vs DDM comparison: choice psychometric, RT, and held-out log-likelihood delta. "
            "Run from the repository root."
        )
    )
    parser.add_argument(
        "--kfold-compare-wide-csv",
        type=str,
        default="output/addm/kfold_compare/rtTrans_recalled_final/cv_compare_by_game_wide.csv",
        help="Per-held-out-game wide comparison CSV with columns free3-rtTrans and theta1-rtTrans",
    )
    parser.add_argument(
        "--addm-rt-kfold-dir",
        type=str,
        default="output/addm/kfold/rtTrans_recalled_final",
        help="Directory containing addm_kfold_fit_summary_free3-rtTrans.csv and addm_kfold_fit_summary_theta1-rtTrans.csv",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["107", "131"],
        help="Subject IDs to exclude (default eyetracking exclusions: 107 131)",
    )
    parser.add_argument(
        "--n-bins-per-sign",
        type=int,
        default=5,
        help="Number of quantile bins per sign for recalled offer value (default: 5 => 10 bins)",
    )
    parser.add_argument(
        "--error-bars",
        type=str,
        choices=["sem", "ci95"],
        default="sem",
        help="Error bars for bin summaries across subjects",
    )
    parser.add_argument(
        "--n-sim-per-trial",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations per trial (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for simulations (default: 123)",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=1.0,
        help="Simulation timestep in ms (default: 1.0)",
    )
    parser.add_argument(
        "--reward-source",
        type=str,
        default="recalled",
        choices=["recalled", "true"],
        help="Whether to use recalled or true offer values for simulation trial templates (default: recalled)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/addm/ppc",
        help="Output directory for figures and intermediate files (default: output/addm/ppc)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to output file names",
    )
    parser.add_argument(
        "--fit-curves-over-bins",
        action="store_true",
        help=(
            "If set, fit smooth curves to raw data (logistic for choice, quadratic for RT) "
            "and overlay binned points with SEM for data/aDDM/DDM."
        ),
    )
    parser.add_argument(
        "--force-resim",
        action="store_true",
        help="Ignore cached simulation CSVs and re-run simulations.",
    )
    args = parser.parse_args()

    # All paths are resolved relative to CWD (expected to be the repository root).
    cwd = Path(".").resolve()

    kfold_dir = (cwd / args.addm_rt_kfold_dir).resolve()
    free3_kfold_csv = kfold_dir / "addm_kfold_fit_summary_free3-rtTrans.csv"
    theta1_kfold_csv = kfold_dir / "addm_kfold_fit_summary_theta1-rtTrans.csv"
    ll_csv = (cwd / args.kfold_compare_wide_csv).resolve()
    out_dir = (cwd / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data paths (relative to CWD)
    clean_fixations_csv = (cwd / "output" / "choice_fixations_clean_buffer_50.csv").resolve()
    data_root = (cwd / ".." / "data").resolve()  # repo root data/ (has valuerecall/ subdirs)
    # data/ has flat per-subject layout matching what addm_fitting.py expects
    ncm_data_dir = (cwd / "data").resolve()
    output_dir = (cwd / "output").resolve()

    exclude = {str(x).strip() for x in args.exclude}

    # ------------------------------------------------------------------
    # Step 1: Load trial-level human data
    # ------------------------------------------------------------------
    trial_csv_path = out_dir / "trial_level_recalled_offer_choice_rt.csv"

    if trial_csv_path.exists():
        print(f"Loading cached trial-level CSV: {trial_csv_path}")
        trial = pd.read_csv(trial_csv_path)
        trial["subject_id"] = _normalize_subject_id(trial["subject_id"])
    else:
        print("Loading trial-level fixation summaries...")
        if not clean_fixations_csv.exists():
            raise FileNotFoundError(
                f"Missing choice fixations CSV: {clean_fixations_csv}\n"
                "Run analysis/prepare_choice_fixations.py first."
            )
        trial = _load_trial_level_fix_summaries(clean_fixations_csv, exclude_subjects=exclude)
        print(f"  {len(trial)} trial-option rows loaded from {trial['subject_id'].nunique()} subjects")

        print("Attaching offer values...")
        trial = _attach_offer_values(trial, data_root=data_root)

        print("Attaching accept choices...")
        trial = _attach_accept_choices(trial, data_root=data_root)

        trial["subject_id"] = _normalize_subject_id(trial["subject_id"])
        trial.to_csv(trial_csv_path, index=False)
        print(f"  Saved trial-level CSV: {trial_csv_path}")

    # Apply exclusions (in case CSV was cached with more subjects).
    trial = trial[~trial["subject_id"].isin(exclude)].copy()

    needed_h = {"subject_id", "recalled_offer_value", "rt_s", "accept"}
    if not needed_h.issubset(trial.columns):
        raise ValueError(f"Trial CSV missing columns: {sorted(list(needed_h - set(trial.columns)))}")

    n_subj = int(trial["subject_id"].nunique())
    if n_subj != 41:
        raise ValueError(f"Expected N=41 eyetracking subjects after exclusions; found N={n_subj}.")
    print(f"N subjects: {n_subj}")

    # ------------------------------------------------------------------
    # Step 2: Bin human trial data
    # ------------------------------------------------------------------
    global BIN_ORDER
    BIN_ORDER = _make_bin_order(int(args.n_bins_per_sign))

    d_h, bounds = _assign_bins_signed_quantiles(
        trial.dropna(subset=["recalled_offer_value"]).copy(),
        value_col="recalled_offer_value",
        out_col="bin_recalled",
        n_bins_per_sign=int(args.n_bins_per_sign),
    )
    x_positions = np.arange(len(BIN_ORDER), dtype=float)
    tick_labels = _bin_tick_labels(bounds, bin_order=BIN_ORDER)

    error_mode = "sem" if bool(args.fit_curves_over_bins) else str(args.error_bars)

    # ------------------------------------------------------------------
    # Step 3: Simulate (or load cached) model trials
    # ------------------------------------------------------------------
    for kfold_csv_path in [free3_kfold_csv, theta1_kfold_csv]:
        if not kfold_csv_path.exists():
            raise FileNotFoundError(f"Missing kfold summary CSV: {kfold_csv_path}")

    if not data_root.exists():
        raise FileNotFoundError(f"Missing data directory: {data_root}")

    # If force-resim, clear cache files so they get regenerated.
    if args.force_resim:
        for kfold_csv_path in [free3_kfold_csv, theta1_kfold_csv]:
            cache_key = f"{kfold_csv_path.parent.name}_{kfold_csv_path.stem}"
            cache_path = out_dir / (
                f"addm_trialsim_{cache_key}_nsim{int(args.n_sim_per_trial)}_seed{int(args.seed)}_dt{args.dt_ms:g}.csv"
            )
            if cache_path.exists():
                cache_path.unlink()
                print(f"  [cleared cache] {cache_path}")

    print("Running/loading aDDM (free3) simulations...")
    _recalled = str(args.reward_source).strip().lower() == "recalled"
    sim_f3 = _simulate_addm_trialsim_from_kfold(
        free3_kfold_csv,
        output_dir=output_dir,
        data_dir=ncm_data_dir if _recalled else None,
        reward_source=str(args.reward_source),
        n_sim_per_trial=int(args.n_sim_per_trial),
        seed=int(args.seed),
        dt_ms=float(args.dt_ms),
        cache_dir=out_dir,
        fixation_data_dir=ncm_data_dir,
    )

    print("Running/loading DDM (theta1) simulations...")
    sim_t1 = _simulate_addm_trialsim_from_kfold(
        theta1_kfold_csv,
        output_dir=output_dir,
        data_dir=ncm_data_dir if _recalled else None,
        reward_source=str(args.reward_source),
        n_sim_per_trial=int(args.n_sim_per_trial),
        seed=int(args.seed),
        dt_ms=float(args.dt_ms),
        cache_dir=out_dir,
        fixation_data_dir=ncm_data_dir,
    )

    # Detect whether this is a fix-time or RT fit from the kfold CSV.
    _kfold_df_tmp = pd.read_csv(free3_kfold_csv, nrows=1)
    _time_col_kfold = str(_kfold_df_tmp["time_col"].iloc[0]) if "time_col" in _kfold_df_tmp.columns else "rt_ms"
    is_fix_mode = _time_col_kfold == "fix_ms"

    # Choose human time column accordingly.
    if is_fix_mode:
        human_time_col = "total_fix_time_s"
        time_y_label = "Total Fixation Time (s)"
    else:
        human_time_col = "rt_s"
        time_y_label = "Decision Response Time (s)"

    summ_choice = _summarize_mean_err_across_subjects(
        d_h.dropna(subset=["bin_recalled"]).copy(),
        bin_col="bin_recalled",
        y_col="accept",
        error_bars=error_mode,
    )
    summ_rt = _summarize_mean_err_across_subjects(
        d_h.dropna(subset=["bin_recalled"]).copy(),
        bin_col="bin_recalled",
        y_col=human_time_col,
        error_bars=error_mode,
    )

    # Normalize and apply exclusions.
    for sim in (sim_f3, sim_t1):
        sim["subject_id"] = _normalize_subject_id(sim["subject_id"])
        sim.drop(sim[sim["subject_id"].isin(exclude)].index, inplace=True)
        # Convert time_ms_sim_mean -> time_s_sim_mean for plotting on second axis.
        sim["time_s_sim_mean"] = pd.to_numeric(sim.get("time_ms_sim_mean"), errors="coerce") / 1000.0

    # Bin simulated trials using the same bounds as the human data.
    sim_f3 = _assign_bins_from_bounds(sim_f3, value_col="v_offer", out_col="bin_recalled", bounds=bounds)
    sim_t1 = _assign_bins_from_bounds(sim_t1, value_col="v_offer", out_col="bin_recalled", bounds=bounds)

    summ_choice_f3 = _summarize_mean_err_across_subjects(
        sim_f3,
        bin_col="bin_recalled",
        y_col="accept_sim_mean",
        error_bars=error_mode,
    )
    summ_choice_t1 = _summarize_mean_err_across_subjects(
        sim_t1,
        bin_col="bin_recalled",
        y_col="accept_sim_mean",
        error_bars=error_mode,
    )
    summ_rt_f3 = _summarize_mean_err_across_subjects(
        sim_f3,
        bin_col="bin_recalled",
        y_col="time_s_sim_mean",
        error_bars=error_mode,
    )
    summ_rt_t1 = _summarize_mean_err_across_subjects(
        sim_t1,
        bin_col="bin_recalled",
        y_col="time_s_sim_mean",
        error_bars=error_mode,
    )

    # ------------------------------------------------------------------
    # Step 4: Load held-out log-likelihood comparison
    # ------------------------------------------------------------------
    if not ll_csv.exists():
        raise FileNotFoundError(f"Missing LL comparison CSV: {ll_csv}")
    ll = pd.read_csv(ll_csv)
    needed_ll = {"free3-rtTrans", "theta1-rtTrans"}
    if not needed_ll.issubset(ll.columns):
        raise ValueError(f"LL CSV missing columns: {sorted(list(needed_ll - set(ll.columns)))}")
    delta = (
        pd.to_numeric(ll["free3-rtTrans"], errors="coerce")
        - pd.to_numeric(ll["theta1-rtTrans"], errors="coerce")
    )
    delta = delta[np.isfinite(delta)]
    if delta.empty:
        raise ValueError("No finite LL deltas found in LL CSV")
    ll_mean = float(np.mean(delta.values))
    ll_sem = _sem(delta.values)

    # ------------------------------------------------------------------
    # Step 5: Create figure
    # ------------------------------------------------------------------
    sns.set_context("poster")
    sns.set_style("ticks")
    with plt.rc_context(
        {
            "font.family": "Arial",
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "lines.solid_capstyle": "butt",
            "lines.dash_capstyle": "butt",
        }
    ):
        fig = plt.figure(figsize=(12, 5.25))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.5])

        # --- Column 1: Choice psychometric vs recalled offer value ---
        ax_choice = fig.add_subplot(gs[0, 0])
        if bool(args.fit_curves_over_bins):
            _plot_data_and_models_with_fits(
                ax_choice,
                x_positions=x_positions,
                tick_labels=tick_labels,
                summ_data=summ_choice,
                summ_free3=summ_choice_f3,
                summ_theta1=summ_choice_t1,
                raw_x_data=pd.to_numeric(d_h.get("recalled_offer_value"), errors="coerce").to_numpy(dtype=float),
                raw_y_data=pd.to_numeric(d_h.get("accept"), errors="coerce").to_numpy(dtype=float),
                raw_x_free3=pd.to_numeric(sim_f3.get("v_offer"), errors="coerce").to_numpy(dtype=float),
                raw_y_free3=pd.to_numeric(sim_f3.get("accept_sim_mean"), errors="coerce").to_numpy(dtype=float),
                raw_x_theta1=pd.to_numeric(sim_t1.get("v_offer"), errors="coerce").to_numpy(dtype=float),
                raw_y_theta1=pd.to_numeric(sim_t1.get("accept_sim_mean"), errors="coerce").to_numpy(dtype=float),
                fit_kind="logistic",
                y_label="Proportion Offers Taken",
                title="",
                x_label="Recalled Offer Value",
                ylim=(-0.02, 1.02),
                show_legend=True,
            )
        else:
            _plot_data_and_models(
                ax_choice,
                x_positions=x_positions,
                tick_labels=tick_labels,
                summ_data=summ_choice,
                summ_free3=summ_choice_f3,
                summ_theta1=summ_choice_t1,
                y_label="Proportion Offers Taken",
                title="",
                x_label="Recalled Offer Value",
                ylim=(-0.02, 1.02),
                show_legend=True,
            )
            ax_choice.set_yticks([0.0, 0.5, 1.0])
            ax_choice.set_yticklabels(["0", "0.5", "1"])

        # --- Column 2: RT vs recalled offer value ---
        ax_rt = fig.add_subplot(gs[0, 1])
        if bool(args.fit_curves_over_bins):
            _plot_data_and_models_with_fits(
                ax_rt,
                x_positions=x_positions,
                tick_labels=tick_labels,
                summ_data=summ_rt,
                summ_free3=summ_rt_f3,
                summ_theta1=summ_rt_t1,
                raw_x_data=pd.to_numeric(d_h.get("recalled_offer_value"), errors="coerce").to_numpy(dtype=float),
                raw_y_data=pd.to_numeric(d_h.get(human_time_col), errors="coerce").to_numpy(dtype=float),
                raw_x_free3=pd.to_numeric(sim_f3.get("v_offer"), errors="coerce").to_numpy(dtype=float),
                raw_y_free3=pd.to_numeric(sim_f3.get("time_s_sim_mean"), errors="coerce").to_numpy(dtype=float),
                raw_x_theta1=pd.to_numeric(sim_t1.get("v_offer"), errors="coerce").to_numpy(dtype=float),
                raw_y_theta1=pd.to_numeric(sim_t1.get("time_s_sim_mean"), errors="coerce").to_numpy(dtype=float),
                fit_kind="quadratic",
                y_label=time_y_label,
                title="",
                x_label="Recalled Offer Value",
                ylim=(0.0, 8.0) if is_fix_mode else (0.0, 15.0),
                show_legend=False,
            )
        else:
            _plot_data_and_models(
                ax_rt,
                x_positions=x_positions,
                tick_labels=tick_labels,
                summ_data=summ_rt,
                summ_free3=summ_rt_f3,
                summ_theta1=summ_rt_t1,
                y_label=time_y_label,
                title="",
                x_label="Recalled Offer Value",
                ylim=(0.0, 8.0) if is_fix_mode else (0.0, 15.0),
                show_legend=False,
            )
            if is_fix_mode:
                ax_rt.set_yticks([0.0, 2.0, 4.0, 6.0, 8.0])
                ax_rt.set_yticklabels(["0", "2", "4", "6", "8"])
            else:
                ax_rt.set_yticks([0.0, 5.0, 10.0, 15.0])
                ax_rt.set_yticklabels(["0", "5", "10", "15"])

        # --- Column 3: LL difference bar (aDDM - DDM) ---
        ax_ll = fig.add_subplot(gs[0, 2])
        ax_ll.bar(
            [0],
            [ll_mean],
            width=0.55,
            color="0.5",
            edgecolor="black",
            linewidth=2,
            zorder=1,
        )
        ax_ll.errorbar(
            [0],
            [ll_mean],
            yerr=[ll_sem],
            fmt="_",
            markersize=16,
            color="black",
            linewidth=1.3,
            capsize=0,
            zorder=3,
        )
        ax_ll.axhline(0.0, linestyle="--", color="0.25", linewidth=1)
        ax_ll.set_xlim(-0.8, 0.8)
        ax_ll.set_xticks([])
        ax_ll.set_ylabel("Δ Held-out Log Likelihood\n(aDDM - DDM)")
        if is_fix_mode:
            ax_ll.set_ylim(0, 60)
            ax_ll.set_yticks([0, 20, 40, 60])
        else:
            ax_ll.set_ylim(0, 20)
            ax_ll.set_yticks([0, 10, 20])
        ax_ll.set_title("")
        ax_ll.grid(False)
        ax_ll.spines["right"].set_visible(False)
        ax_ll.spines["top"].set_visible(False)

        fig.tight_layout()

        plt.close(fig)

    print(f"Mean DeltaLL (aDDM-DDM): {ll_mean:.6f} (SEM={ll_sem:.6f}, n_games={len(delta)})")


if __name__ == "__main__":
    main()
