"""Group-level fitting utilities for the adapted aDDM.

This module implements a simulation-based likelihood inspired by Krajbich et al.
(2010), adapted to the eye-EMDM accept/leave task.

Key ideas
---------
- We define offer value V_offer per trial as the sum of relevant item rewards.
- We bin trials by V_offer (quantile bins).
- For each V_offer bin, we estimate empirical gaze statistics from the cleaned
  fixation data:
  - P(fixation is relevant | V_offer bin)
  - fixation duration distributions conditional on relevance
  - distribution of total per-trial transition time (sum of gaps between
    fixations)

- For a candidate parameter set (d, theta, sigma), we estimate predicted
  distributions over (choice, RT) by simulating trials under a generative gaze
  model (IID fixations, no immediate repeats).

- We compare predicted and observed (choice, RT) using a binned RT likelihood.
  RT bins are chosen as quantiles *within each (V_offer bin, choice)* cell, with
  an automatic cap based on minimum trials per bin.

- To avoid log(0) from finite Monte Carlo counts, we use Dirichlet/pseudocount
  smoothing for simulated bin probabilities.

This file is designed to be imported by a CLI script (see
`fit_addm_group_pybads.py`).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings
import re

import numpy as np
import pandas as pd

EXCLUDE_EYETRACKING_DEFAULT = ("107", "131")

try:
    # When imported as a package: `import addm.lib.addm_fitting`
    from .adapted_addm_simulation import ADDMParameters
    from .generative_gaze import (
        GazeStats,
        TrialTemplate,
        sample_fixation_event_iid,
        sample_transition_total_ms,
    )
except ImportError:  # pragma: no cover
    # When run as a script from within the `modeling` directory.
    from adapted_addm_simulation import ADDMParameters
    from generative_gaze import (
        GazeStats,
        TrialTemplate,
        sample_fixation_event_iid,
        sample_transition_total_ms,
    )


@dataclass(frozen=True)
class BinningConfig:
    n_v_offer_bins: int = 7
    rt_bins_max: int = 15
    min_trials_per_rt_bin: int = 25
    # If >0, use a fixed number of RT bins per (v_bin, choice) cell (still
    # limited by available data / ties). This can make likelihood comparisons
    # across runs more consistent than the adaptive n//min_trials rule.
    rt_bins_fixed: int = 0


@dataclass(frozen=True)
class SimulationConfig:
    dt_ms: float = 1.0
    n_sim_per_vbin: int = 500
    alpha_smoothing: float = 1.0
    seed: int = 123
    include_transition_time: bool = False
    # How value integrates while fixating an offer-irrelevant item.
    # - "zero": drift = 0 (current baseline)
    # - "theta_sumrel": drift uses theta * sum(relevant rewards)
    # - "sumrel": drift uses sum(relevant rewards)
    irrelevant_mode: str = "zero"

    # Optional center-fixation channel (roi_content == 'fixation')
    # If include_center_fixations is False, existing behavior is preserved.
    include_center_fixations: bool = False
    # How the gaze generator treats center fixations:
    # - "separate": sample explicit center fixations using their empirical durations
    # - "merge_with_irrelevant": treat center as additional "irrelevant" fixations
    center_gaze_mode: str = "separate"
    # How value integrates during center fixations (item_index == -1 in the gaze stream):
    # - "same_as_irrelevant": use irrelevant_mode
    # - "zero": drift = 0
    # - "theta_sumrel": drift uses theta * sum(relevant rewards)
    # - "sumrel": drift uses sum(relevant rewards)
    # - "phi_sumrel": drift uses phi_center * sum(relevant rewards) (requires params.phi_center)
    center_mode: str = "same_as_irrelevant"


@dataclass(frozen=True)
class FitConfig:
    binning: BinningConfig = BinningConfig()
    sim: SimulationConfig = SimulationConfig()
    # Which time variable to fit/bucket (in ms). Supported:
    # - "fix_ms": sum of item fixation durations per trial
    # - "rt_ms": full reaction time
    time_col: str = "fix_ms"


def _is_item_series(roi_series: pd.Series) -> pd.Series:
    """Mask for rows where roi_content looks like an item identity."""

    mask = roi_series.notna()
    roi_lower = roi_series.astype(str).str.lower().fillna("")
    mask &= ~roi_lower.isin(["fixation", "none"])
    mask &= roi_series.astype(str).str.count("_") == 3
    return mask


def _is_item_name_series(name_series: pd.Series) -> pd.Series:
    """Mask for strings that look like item identities (four tokens, three underscores)."""

    s = name_series.astype(str)
    # Preserve NaN handling: 'nan' should not pass.
    mask = name_series.notna()
    mask &= s.str.count("_") == 3
    return mask


def _load_value_recall_order_by_game(df_raw: pd.DataFrame) -> Dict[float, List[str]]:
    """Extract per-game cue order for value recall.

    Uses rows where phase=='memory' and event=='value_recall', ordered by
    trial_number within each game, and returns the cued item identities from
    the 'image' column.
    """

    if not {"phase", "event", "game", "trial_number"}.issubset(set(df_raw.columns)):
        return {}
    if "image" not in df_raw.columns:
        return {}

    df_mem = df_raw[(df_raw["phase"] == "memory") & (df_raw["event"] == "value_recall")].copy()
    if df_mem.empty:
        return {}

    df_mem["game"] = pd.to_numeric(df_mem["game"], errors="coerce")
    df_mem["trial_number"] = pd.to_numeric(df_mem["trial_number"], errors="coerce")
    df_mem = df_mem.dropna(subset=["game", "trial_number", "image"]).copy()
    if df_mem.empty:
        return {}

    order_by_game: Dict[float, List[str]] = {}
    for game, df_g in df_mem.groupby("game", sort=False):
        # The raw fixation CSV contains many rows per cue (sample-level);
        # collapse to one image per (game, trial_number) and then sort.
        df_g = df_g[["trial_number", "image"]].drop_duplicates(subset=["trial_number"], keep="first")
        df_g = df_g.sort_values("trial_number")
        out = df_g["image"].astype(str).str.strip().tolist()
        out = [im for im in out if im]
        # By task design, there should be 6 value-recall cues per game.
        if len(out) > 6:
            out = out[:6]
        if out:
            order_by_game[float(game)] = out
    return order_by_game


_MAG_RE = re.compile(r"^\s*([0-9]+)(?:\.0+)?\s*$")


def _parse_signed_int_token(token: str) -> Optional[int]:
    """Parse '+3', '-7', '3', '3.0' into int; returns None if invalid."""

    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None
    # Combined tokens like +5 / -2
    if (t.startswith("+") or t.startswith("-")) and len(t) >= 2:
        sign = 1 if t[0] == "+" else -1
        m = _MAG_RE.match(t[1:])
        if not m:
            return None
        mag = int(m.group(1))
        return sign * mag
    # Bare magnitude
    m = _MAG_RE.match(t)
    if m:
        return int(m.group(1))
    return None


def _is_positive_sign_token(token: str) -> bool:
    t = str(token).strip().upper()
    return t in {"+", "PLUS", "POSITIVE"} or "PLUS" in t or "POSITIVE" in t


def _is_negative_sign_token(token: str) -> bool:
    t = str(token).strip().upper()
    return t in {"-", "MINUS", "NEGATIVE"} or "MINUS" in t or "NEGATIVE" in t


def _parse_valuerecall_transcript_game(df_game: pd.DataFrame, *, n_expected: int) -> List[Optional[int]]:
    """Parse a single game's value-recall transcript into a fixed-length list of ints.

    Important: The value-recall transcription stream is *positionally aligned*
    to the cue order (6 items per game). Typically each recalled value is split
    across two rows (valence/sign then magnitude). If an item is not recalled,
    the transcription often contains a single empty row.

    This function preserves alignment by emitting exactly `n_expected` values,
    inserting 0 whenever a recall is missing/unparseable.
    """

    if n_expected <= 0:
        return []
    if df_game.empty or "item" not in df_game.columns:
        return [None] * int(n_expected)

    df = df_game.copy()
    if "onset" in df.columns:
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        # Stable sort to preserve original order when onset ties/missing.
        df = df.sort_values(["onset"], kind="mergesort")

    tokens = df["item"].tolist()

    out: List[Optional[int]] = []
    i = 0
    while len(out) < int(n_expected) and i < len(tokens):
        tok1 = tokens[i]

        # A single empty row corresponds to a missing recall for that cued item.
        if pd.isna(tok1):
            out.append(None)
            i += 1
            continue
        t1 = str(tok1).strip()
        if not t1:
            out.append(None)
            i += 1
            continue

        # Already combined token like '+3'/'-7'
        direct1 = _parse_signed_int_token(t1)
        if direct1 is not None and (t1.startswith("+") or t1.startswith("-")):
            out.append(int(direct1))
            i += 1
            continue

        # Typical case: sign token then magnitude token.
        if _is_positive_sign_token(t1) or _is_negative_sign_token(t1):
            sign = 1 if _is_positive_sign_token(t1) else -1
            if i + 1 < len(tokens):
                tok2 = tokens[i + 1]
                if not pd.isna(tok2):
                    t2 = str(tok2).strip()
                    mag = _parse_signed_int_token(t2)
                    if mag is not None:
                        out.append(int(sign * abs(int(mag))))
                        i += 2
                        continue
            # Sign without usable magnitude: treat as missing for this item.
            out.append(None)
            i += 1
            continue

        # Fallback: magnitude-only token (treat as positive)
        if direct1 is not None:
            out.append(int(direct1))
        else:
            out.append(None)
        i += 1

    # If transcript ended early, pad remaining items with 0.
    if len(out) < int(n_expected):
        out.extend([None] * (int(n_expected) - len(out)))

    # If transcript had extra junk rows, ignore beyond n_expected.
    return out[: int(n_expected)]


def _load_recalled_values_by_game(
    *,
    subid: str,
    data_dir: Path,
    recall_order_by_game: Dict[float, List[str]],
) -> Dict[Tuple[float, str], Optional[int]]:
    """Load recalled values and map them onto (game, image) using cue order.

    Returns Optional[int] values, where None indicates missing/unparseable.
    Callers can decide how to handle missing values (e.g., fallback to true).
    """

    vr_path = data_dir / str(subid) / f"{subid}_valuerecall.csv"
    if not vr_path.exists():
        raise FileNotFoundError(f"Missing value-recall transcription CSV: {vr_path}")

    df_vr = pd.read_csv(vr_path)
    if df_vr.empty:
        return {}

    if "game" not in df_vr.columns:
        raise ValueError(f"Value-recall transcription CSV missing 'game' column: {vr_path}")

    df_vr["game"] = pd.to_numeric(df_vr["game"], errors="coerce")
    df_vr = df_vr.dropna(subset=["game"]).copy()

    recalled: Dict[Tuple[float, str], Optional[int]] = {}
    for game_f, cue_order in recall_order_by_game.items():
        if not cue_order:
            continue
        n_expected = int(len(cue_order))
        df_g = df_vr[df_vr["game"].astype(float) == float(game_f)].copy()
        if df_g.empty:
            values: List[Optional[int]] = [None] * n_expected
        else:
            values = _parse_valuerecall_transcript_game(df_g, n_expected=n_expected)

        for idx, image in enumerate(cue_order):
            v = values[idx] if idx < len(values) else None
            recalled[(float(game_f), str(image).strip())] = v
    return recalled


def _load_subject_items_by_game(
    df_raw: pd.DataFrame,
    *,
    subid: str,
    reward_source: str,
    data_dir: Optional[Path],
) -> Dict[float, pd.DataFrame]:
    """Extract per-game item rewards from encoding-phase rows.

    Parameters
    ----------
    df_raw : DataFrame
        Raw fixation dataframe (output/<subid>/<subid>_fixations_df_original_buffer_50.csv)
    subid : str
        Subject id.
    reward_source : {'true','recalled'}
        Which reward values to attach to items.
    data_dir : Path or None
        Required when reward_source=='recalled'. Should point to the repo-level
        data folder (containing <subid>/valuerecall/<subid>_valuerecall.csv).
    """

    reward_source = str(reward_source).strip().lower()
    if reward_source not in {"true", "recalled"}:
        raise ValueError("reward_source must be one of {'true','recalled'}")

    mask_enc = df_raw["phase"] == "encoding"
    df_enc = df_raw[mask_enc].copy()

    # During encoding, `image` is more reliable than `roi_content` for item identity.
    item_col = "image" if "image" in df_enc.columns else "roi_content"
    item_mask = _is_item_name_series(df_enc[item_col])
    df_enc_items = df_enc[item_mask].copy()

    if df_enc_items.empty:
        raise ValueError("No encoding-phase item rows")
    if "outcome" not in df_enc_items.columns:
        raise ValueError("Expected 'outcome' column for item rewards in encoding phase")

    df_items = (
        df_enc_items.loc[df_enc_items["outcome"].notna(), ["game", item_col, "outcome"]]
        .drop_duplicates(subset=["game", item_col])
        .rename(columns={item_col: "roi_content", "outcome": "reward_true"})
    )

    if reward_source == "true":
        df_items["reward"] = pd.to_numeric(df_items["reward_true"], errors="coerce")
    else:
        if data_dir is None:
            raise ValueError("data_dir is required when reward_source=='recalled'")
        recall_order_by_game = _load_value_recall_order_by_game(df_raw)
        recalled_map = _load_recalled_values_by_game(
            subid=str(subid),
            data_dir=Path(data_dir),
            recall_order_by_game=recall_order_by_game,
        )

        # Verify that the cue order corresponds to encoded items.
        # (We warn rather than error to avoid breaking long batch runs.)
        for g, cue_order in recall_order_by_game.items():
            enc_items = set(
                df_items.loc[df_items["game"].astype(float) == float(g), "roi_content"].astype(str).str.strip().tolist()
            )
            cue_set = set(str(x).strip() for x in (cue_order or []))
            if len(cue_order or []) != 6:
                warnings.warn(
                    f"Subject {subid} game {g:g}: expected 6 value-recall cues but got {len(cue_order or [])}",
                    RuntimeWarning,
                )
            missing_from_enc = sorted(list(cue_set - enc_items))
            if missing_from_enc:
                warnings.warn(
                    f"Subject {subid} game {g:g}: value-recall cue items not found among encoded items: {missing_from_enc[:3]}",
                    RuntimeWarning,
                )

        def _lookup(row: pd.Series) -> int:
            g = float(row["game"])
            im = str(row["roi_content"]).strip()
            v_rec = recalled_map.get((g, im), None)
            if v_rec is None:
                # If missing, default to the true encoded reward.
                try:
                    return int(float(row.get("reward_true")))
                except Exception:
                    return 0
            return int(v_rec)

        df_items["reward"] = df_items.apply(_lookup, axis=1).astype(float)

    items_by_game: Dict[float, pd.DataFrame] = {}
    for game, df_g in df_items.groupby("game"):
        df_g_sorted = df_g.sort_values("roi_content").reset_index(drop=True)
        df_g_sorted["item_index"] = np.arange(len(df_g_sorted), dtype=int)
        items_by_game[float(game)] = df_g_sorted

    return items_by_game


def _infer_relevance_for_game_items(df_items_game: pd.DataFrame, option: str) -> np.ndarray:
    """R_i=1 if option token appears among roi_content feature tokens."""

    option_str = str(option)

    def is_rel(roi: str) -> int:
        if not isinstance(roi, str):
            return 0
        return int(option_str in roi.split("_"))

    return df_items_game["roi_content"].apply(is_rel).to_numpy(dtype=int)


def _human_choice_to_accept(choice_value: object) -> int:
    """Map raw choice coding to accept(1)/leave(0).

    In this dataset/analysis code, choice==1 corresponds to take/accept and
    choice==2 corresponds to leave/reject.
    """

    try:
        return int(float(choice_value) == 1.0)
    except Exception:
        # Conservative fallback: treat as accept if missing.
        return 1


def load_group_trial_templates(
    output_dir: Path,
    *,
    strict_missing: bool = False,
    include_center_fixations: bool = False,
    reward_scale: float = 1.0,
    reward_source: str = "true",
    data_dir: Optional[Path] = None,
    exclude_subjects: tuple[str, ...] = EXCLUDE_EYETRACKING_DEFAULT,
    fixation_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load per-trial templates across all subjects.

    Returns a DataFrame with one row per (subject, game, trial_number) containing:
    - subject, game, trial_number
    - rt_ms
    - fix_ms
    - accept (0/1)
    - rewards (object dtype: np.ndarray)
    - relevance (object dtype: np.ndarray)
    - v_offer
    - rel_indices, irrel_indices (object dtype: np.ndarray)

    Notes
    -----
    This relies on both the raw fixation CSV (to read encoding-phase rewards)
    and the cleaned choice-phase fixation CSV (to read trial metadata/RT/choice).

    Parameters
    ----------
    fixation_data_dir : Path, optional
        If provided, look for per-subject fixation CSV files under this
        directory instead of ``output_dir``.  Subject discovery (which
        subjects to iterate over) also uses this directory.
    """

    reward_scale_f = float(reward_scale)
    if not (np.isfinite(reward_scale_f) and reward_scale_f > 0):
        raise ValueError("reward_scale must be finite and > 0")

    rows: List[Dict[str, object]] = []
    skipped_missing: List[str] = []
    skipped_bad_raw: List[str] = []
    skipped_missing_recall: List[str] = []

    exclude = set(str(s) for s in (exclude_subjects or ()))

    fix_source = fixation_data_dir if fixation_data_dir is not None else output_dir

    for sub_dir in sorted(fix_source.iterdir()):
        if not sub_dir.is_dir():
            continue
        subid = sub_dir.name
        # Avoid treating non-subject directories (e.g., model_fits/) as subjects.
        if not str(subid).isdigit():
            continue
        if str(subid) in exclude:
            continue
        raw_path = sub_dir / f"{subid}_fixations_df_original_buffer_50.csv"
        clean_path = (
            sub_dir / f"{subid}_fixations_for_modeling_withcenter.csv"
            if bool(include_center_fixations)
            else sub_dir / f"{subid}_fixations_for_modeling.csv"
        )
        if not raw_path.exists() or not clean_path.exists():
            skipped_missing.append(str(subid))
            continue

        df_raw = pd.read_csv(raw_path)
        df_clean = pd.read_csv(clean_path)

        try:
            items_by_game = _load_subject_items_by_game(
                df_raw,
                subid=str(subid),
                reward_source=str(reward_source),
                data_dir=data_dir,
            )
        except FileNotFoundError as e:
            if str(reward_source).strip().lower() == "recalled":
                skipped_missing_recall.append(f"{subid}({e})")
                continue
            skipped_bad_raw.append(f"{subid}({type(e).__name__}: {e})")
            continue
        except Exception as e:
            skipped_bad_raw.append(f"{subid}({type(e).__name__}: {e})")
            continue

        # One row per (game, trial_number) from the cleaned fixation rows.
        # IMPORTANT: trial_number repeats across games in some exports, so we
        # must include game in the grouping key.
        for (game, trial_number), df_t in df_clean.groupby(["game", "trial_number"]):
            game = float(game)
            if game not in items_by_game:
                continue

            option_vals = df_t["option"].unique()
            if len(option_vals) != 1:
                continue
            option = option_vals[0]

            rt_vals = df_t["rt"].unique()
            if len(rt_vals) != 1:
                continue
            rt_s = float(rt_vals[0])
            rt_ms = rt_s * 1000.0

            # Total fixation time in ms (sum of *item* fixation durations).
            # If we loaded a with-center file, exclude center rows from fix_ms.
            if "fix_duration_bounded" in df_t.columns:
                dtmp = df_t.copy()
                if "is_center" in dtmp.columns:
                    dtmp = dtmp[pd.to_numeric(dtmp["is_center"], errors="coerce").fillna(0).astype(int) == 0]
                dur = pd.to_numeric(dtmp["fix_duration_bounded"], errors="coerce").to_numpy(dtype=float)
                dur = dur[np.isfinite(dur) & (dur > 0)]
                fix_ms = float(np.sum(dur)) if dur.size else 0.0
            else:
                fix_ms = float("nan")

            choice_vals = df_t["choice"].unique()
            if len(choice_vals) != 1:
                continue
            accept = _human_choice_to_accept(choice_vals[0])

            df_items_game = items_by_game[game]
            rewards = df_items_game["reward"].to_numpy(dtype=float) * reward_scale_f
            relevance = _infer_relevance_for_game_items(df_items_game, option=option)

            rel_indices = np.flatnonzero(relevance == 1)
            irrel_indices = np.flatnonzero(relevance == 0)

            v_offer = float(np.sum(rewards[rel_indices]))

            rows.append(
                {
                    "subject": subid,
                    "game": game,
                    "trial_number": int(trial_number),
                    "option": str(option),
                    "rt_ms": rt_ms,
                    "fix_ms": fix_ms,
                    "accept": int(accept),
                    "rewards": rewards,
                    "relevance": relevance,
                    "v_offer": v_offer,
                    "reward_scale": reward_scale_f,
                    "reward_source": str(reward_source).strip().lower(),
                    "rel_indices": rel_indices,
                    "irrel_indices": irrel_indices,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No subject data found under {output_dir}")

    if skipped_missing:
        skipped_missing_sorted = sorted(set(skipped_missing), key=lambda x: int(x) if str(x).isdigit() else x)
        msg = (
            "Skipped subjects missing required CSVs under output_dir="
            f"{output_dir}: n={len(skipped_missing_sorted)}; "
            f"examples={skipped_missing_sorted[:10]}"
        )
        if strict_missing:
            raise FileNotFoundError(msg)
        warnings.warn(msg, RuntimeWarning)

    if skipped_bad_raw:
        msg = (
            "Skipped subjects with unusable raw fixation CSVs (missing encoding items) under output_dir="
            f"{output_dir}: n={len(skipped_bad_raw)}; examples={skipped_bad_raw[:10]}"
        )
        if strict_missing:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)

    if skipped_missing_recall:
        msg = (
            "Skipped subjects missing value-recall transcription CSVs under data_dir="
            f"{data_dir}: n={len(skipped_missing_recall)}; examples={skipped_missing_recall[:10]}"
        )
        if strict_missing:
            raise FileNotFoundError(msg)
        warnings.warn(msg, RuntimeWarning)

    df = pd.DataFrame(rows)
    return df


def load_group_fixations(
    output_dir: Path,
    *,
    strict_missing: bool = False,
    include_center_fixations: bool = False,
    exclude_subjects: tuple[str, ...] = EXCLUDE_EYETRACKING_DEFAULT,
    fixation_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load merged fixation events across all subjects (cleaned CSVs).

    Parameters
    ----------
    fixation_data_dir : Path, optional
        If provided, look for per-subject fixation CSV files under this
        directory instead of ``output_dir``.  Subject discovery also uses
        this directory.
    """

    dfs: List[pd.DataFrame] = []
    skipped_missing: List[str] = []
    exclude = set(str(s) for s in (exclude_subjects or ()))

    fix_source = fixation_data_dir if fixation_data_dir is not None else output_dir

    for sub_dir in sorted(fix_source.iterdir()):
        if not sub_dir.is_dir():
            continue
        subid = sub_dir.name
        # Avoid treating non-subject directories (e.g., model_fits/) as subjects.
        if not str(subid).isdigit():
            continue
        if str(subid) in exclude:
            continue
        clean_path = (
            sub_dir / f"{subid}_fixations_for_modeling_withcenter.csv"
            if bool(include_center_fixations)
            else sub_dir / f"{subid}_fixations_for_modeling.csv"
        )
        if not clean_path.exists():
            skipped_missing.append(str(subid))
            continue
        df = pd.read_csv(clean_path)
        df["subject"] = subid
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No cleaned fixation CSVs found under {output_dir}")

    if skipped_missing:
        skipped_missing_sorted = sorted(set(skipped_missing), key=lambda x: int(x) if str(x).isdigit() else x)
        msg = (
            "Skipped subjects missing cleaned fixation CSVs under output_dir="
            f"{output_dir}: n={len(skipped_missing_sorted)}; "
            f"examples={skipped_missing_sorted[:10]}"
        )
        if strict_missing:
            raise FileNotFoundError(msg)
        warnings.warn(msg, RuntimeWarning)

    out = pd.concat(dfs, ignore_index=True)
    return out


def compute_transition_time_per_trial_ms(df_fix: pd.DataFrame) -> pd.DataFrame:
    """Compute total transition time (sum of between-fixation gaps) per trial."""

    needed = {"subject", "game", "trial_number", "fix_start", "fix_end"}
    missing = needed - set(df_fix.columns)
    if missing:
        raise ValueError(f"Missing required fixation columns: {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for (subject, game, trial_number), df_t in df_fix.groupby(["subject", "game", "trial_number"]):
        df_t = df_t.sort_values("fix_start")
        starts = df_t["fix_start"].to_numpy(dtype=float)
        ends = df_t["fix_end"].to_numpy(dtype=float)
        if starts.size <= 1:
            total_gap = 0.0
        else:
            gaps = starts[1:] - ends[:-1]
            gaps = np.maximum(gaps, 0.0)
            total_gap = float(np.sum(gaps))
        rows.append(
            {
                "subject": subject,
                "game": float(game),
                "trial_number": int(trial_number),
                "transition_total_ms": total_gap,
            }
        )

    return pd.DataFrame(rows)


def assign_v_offer_bins(v_offer: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each v_offer value to a quantile bin.

    Returns
    -------
    bin_index : np.ndarray[int]
        Values in [0, n_bins-1]
    edges : np.ndarray[float]
        Bin edges of length n_bins+1.
    """

    if n_bins <= 1:
        edges = np.array([np.min(v_offer), np.max(v_offer)], dtype=float)
        return np.zeros_like(v_offer, dtype=int), edges

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(v_offer, qs)

    # Ensure strictly non-decreasing edges.
    edges[0] = np.min(v_offer)
    edges[-1] = np.max(v_offer)

    # np.digitize returns 1..n_bins; subtract 1.
    bin_index = np.digitize(v_offer, edges[1:-1], right=True)
    return bin_index.astype(int), edges.astype(float)


def assign_v_offer_bins_from_edges(v_offer: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign v_offer to precomputed bin edges (length B+1)."""

    if edges.size < 2:
        return np.zeros_like(v_offer, dtype=int)
    return np.digitize(v_offer, edges[1:-1], right=True).astype(int)


def _rt_bin_edges_quantile(rt_ms: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based RT bin edges."""

    if n_bins <= 1:
        return np.array([float(np.min(rt_ms)), float(np.max(rt_ms))], dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(rt_ms, qs)
    edges[0] = float(np.min(rt_ms))
    edges[-1] = float(np.max(rt_ms))
    # Ensure edges are strictly increasing. Quantiles can repeat when there are
    # many ties, which breaks histogramming.
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        return np.array([float(np.min(rt_ms)), float(np.max(rt_ms))], dtype=float)
    return edges


def build_rt_bins_per_cell(
    df_trials: pd.DataFrame,
    *,
    v_bin_col: str,
    rt_col: str = "rt_ms",
    accept_col: str = "accept",
    rt_bins_max: int,
    min_trials_per_rt_bin: int,
    rt_bins_fixed: int = 0,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Build RT bin edges per (v_bin, accept) cell.

    Returns dict[(v_bin, accept)] -> edges array.
    """

    edges_by_cell: Dict[Tuple[int, int], np.ndarray] = {}
    for (vbin, accept), df_cell in df_trials.groupby([v_bin_col, accept_col]):
        n = len(df_cell)
        if int(rt_bins_fixed) > 0:
            # Fixed bin-count mode (still limited by data volume / ties).
            n_bins = max(1, min(int(rt_bins_fixed), int(n)))
        else:
            # Adaptive bin-count mode: cap by rt_bins_max and require enough
            # trials per bin.
            n_bins = max(1, min(int(rt_bins_max), int(n) // max(1, int(min_trials_per_rt_bin))))
        edges = _rt_bin_edges_quantile(df_cell[rt_col].to_numpy(dtype=float), n_bins=n_bins)
        edges_by_cell[(int(vbin), int(accept))] = edges

    return edges_by_cell


def _bin_value(x: float, edges: np.ndarray) -> int:
    """Return the bin index for x given edges (length K+1)."""

    if edges.size < 2:
        return 0
    # bins are [edges[i], edges[i+1]] with right-inclusive for stability
    return int(np.digitize([x], edges[1:-1], right=True)[0])


def estimate_gaze_stats(
    df_fix: pd.DataFrame,
    *,
    v_bin: np.ndarray,
    center_gaze_mode: str = "separate",
) -> Dict[int, GazeStats]:
    """Estimate empirical gaze stats per V_offer bin.

    The input fixation dataframe must include columns:
    - is_relevant (0/1)
    - fix_duration_bounded (ms)

    If center fixations are present (is_center==1 or roi_content=='fixation'),
    they are either modeled as a separate gaze channel (center_gaze_mode='separate')
    or merged into the irrelevant distribution (center_gaze_mode='merge_with_irrelevant').

    Parameters
    ----------
    df_fix : DataFrame
        Cleaned fixation events, merged across subjects.
    v_bin : np.ndarray[int]
        V_offer bin assignment per fixation row.

    Returns
    -------
    Dict[int, GazeStats]
    """

    df = df_fix.copy()
    df["v_bin"] = v_bin

    out: Dict[int, GazeStats] = {}
    for vbin, df_b in df.groupby("v_bin"):
        # Identify center fixations if present.
        if "is_center" in df_b.columns:
            is_center = pd.to_numeric(df_b["is_center"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        else:
            roi_lower = df_b.get("roi_content", pd.Series([""] * len(df_b))).astype(str).str.lower().fillna("")
            is_center = roi_lower.eq("fixation").astype(int).to_numpy(dtype=int)

        item_mask = is_center == 0

        # p_relevant_fix is defined conditional on *item* fixations.
        rel_item = pd.to_numeric(df_b.loc[item_mask, "is_relevant"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        p_rel = float(np.mean(rel_item)) if rel_item.size else 0.5
        p_rel = float(np.clip(p_rel, 0.01, 0.99))

        # p_center_fix is defined over all fixations (items + center) if modeled separately.
        p_center = float(np.mean(is_center)) if is_center.size else 0.0
        p_center = float(np.clip(p_center, 0.0, 0.99))

        dur = pd.to_numeric(df_b["fix_duration_bounded"], errors="coerce").to_numpy(dtype=float)

        # Split durations. Center is either separate or merged into irrelevant.
        dur_center = dur[is_center == 1]
        dur_item = dur[item_mask]
        dur_rel = dur_item[rel_item == 1]
        dur_irrel = dur_item[rel_item == 0]
        if str(center_gaze_mode) == "merge_with_irrelevant":
            dur_irrel = np.concatenate([dur_irrel, dur_center]) if dur_center.size else dur_irrel
            p_center = 0.0
        dur_rel = dur_rel[np.isfinite(dur_rel) & (dur_rel > 0)]
        dur_irrel = dur_irrel[np.isfinite(dur_irrel) & (dur_irrel > 0)]
        dur_center = dur_center[np.isfinite(dur_center) & (dur_center > 0)]

        # Transition time is attached later (estimated on trials), initialize empty.
        out[int(vbin)] = GazeStats(
            p_relevant_fix=p_rel,
            durations_relevant_ms=dur_rel.astype(float),
            durations_irrelevant_ms=dur_irrel.astype(float),
            transition_total_ms=np.array([], dtype=float),
            p_center_fix=float(p_center),
            durations_center_ms=dur_center.astype(float),
        )

    return out


@dataclass(frozen=True)
class FittingComponents:
    """Training-derived components for computing the simulation likelihood."""

    v_edges: np.ndarray
    gaze_by_bin: Dict[int, GazeStats]
    rt_edges_by_cell: Dict[Tuple[int, int], np.ndarray]
    max_time_ms: float


def build_trial_templates_from_df(df_trials: pd.DataFrame) -> List[TrialTemplate]:
    """Convert a trial-template dataframe into a list of `TrialTemplate`.

    The input is expected to be the output of `load_group_trial_templates`
    (or a subset/copy), i.e. it has object columns for `rewards`, `relevance`,
    `rel_indices`, and `irrel_indices`.
    """

    templates: List[TrialTemplate] = []
    for _, r in df_trials.iterrows():
        templates.append(
            TrialTemplate(
                rewards=np.asarray(r["rewards"], dtype=float),
                relevance=np.asarray(r["relevance"], dtype=int),
                v_offer=float(r["v_offer"]),
                rel_indices=np.asarray(r["rel_indices"], dtype=int),
                irrel_indices=np.asarray(r["irrel_indices"], dtype=int),
            )
        )
    return templates


def simulate_trials_generative(
    *,
    df_trials: pd.DataFrame,
    params: ADDMParameters,
    config: FitConfig,
    components: FittingComponents,
    n_sim_per_trial: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate choices/RTs for each trial in `df_trials`.

    This produces a synthetic dataset that preserves the original trial
    structure (rewards/relevance/V_offer), but replaces observed outcomes with
    simulated outcomes.

    Parameters
    ----------
    df_trials : DataFrame
        Trial template dataframe (output of `load_group_trial_templates`).
    params : ADDMParameters
        Accumulation parameters to simulate from.
    config : FitConfig
        Simulation config (dt_ms, etc.). Only `config.sim` is used.
    components : FittingComponents
        Must contain `v_edges` and `gaze_by_bin` learned from some empirical
        fixation dataset.
    n_sim_per_trial : int
        Number of independent simulations per trial. If >1, this function
        returns one row per trial with Monte Carlo summaries.
    seed : int, optional
        RNG seed.

    Returns
    -------
    DataFrame
        One row per trial with columns:
        - subject, game, trial_number
        - v_offer, v_bin
        - accept_sim_mean
        - time_ms_sim_mean, time_ms_sim
        - rt_ms_sim_mean, rt_ms_sim
        - accept_sim (majority vote or rounded mean)
        - n_sim
    """

    if n_sim_per_trial <= 0:
        raise ValueError("n_sim_per_trial must be >= 1")

    rng = np.random.default_rng(seed if seed is not None else config.sim.seed)

    df = df_trials.copy()
    df["v_bin"] = assign_v_offer_bins_from_edges(
        df["v_offer"].to_numpy(dtype=float),
        components.v_edges,
    )

    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        vbin = int(r["v_bin"])
        gaze = components.gaze_by_bin.get(vbin)
        if gaze is None:
            # If a bin is missing gaze stats, skip this trial.
            continue

        tmpl = TrialTemplate(
            rewards=np.asarray(r["rewards"], dtype=float),
            relevance=np.asarray(r["relevance"], dtype=int),
            v_offer=float(r["v_offer"]),
            rel_indices=np.asarray(r["rel_indices"], dtype=int),
            irrel_indices=np.asarray(r["irrel_indices"], dtype=int),
        )

        accepts: List[int] = []
        times: List[float] = []

        for _ in range(n_sim_per_trial):
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            a, rt = simulate_one_trial_generative(
                template=tmpl,
                gaze=gaze,
                params=params,
                rng=trial_rng,
                max_time_ms=components.max_time_ms,
                dt_ms=config.sim.dt_ms,
                include_transition_time=bool(config.sim.include_transition_time),
                irrelevant_mode=str(getattr(config.sim, "irrelevant_mode", "zero")),
                center_mode=str(getattr(config.sim, "center_mode", "same_as_irrelevant")),
            )
            accepts.append(int(a))
            times.append(float(rt))

        accept_mean = float(np.mean(accepts)) if accepts else np.nan
        time_mean = float(np.mean(times)) if times else np.nan

        # Single-valued summaries.
        accept_point = int(accept_mean >= 0.5) if np.isfinite(accept_mean) else np.nan

        rows.append(
            {
                "subject": r["subject"],
                "game": float(r["game"]),
                "trial_number": int(r["trial_number"]),
                "v_offer": float(r["v_offer"]),
                "v_bin": int(vbin),
                "accept_sim_mean": accept_mean,
                "time_ms_sim_mean": time_mean,
                "rt_ms_sim_mean": time_mean,
                "accept_sim": accept_point,
                "time_ms_sim": time_mean,
                "rt_ms_sim": time_mean,
                "n_sim": int(n_sim_per_trial),
            }
        )

    return pd.DataFrame(rows)


def summarize_observed_vs_simulated(
    *,
    df_trials_obs: pd.DataFrame,
    df_trials_sim: pd.DataFrame,
    by: Sequence[str] = ("v_bin",),
    time_col_obs: str = "rt_ms",
) -> pd.DataFrame:
    """Summarize observed vs simulated outcomes grouped by columns in `by`.

    Parameters
    ----------
    time_col_obs
        Column in df_trials_obs treated as the time metric (e.g., "rt_ms" or "fix_ms").

    Notes
    -----
    This function emits generic columns (time_obs_*, time_sim_*) so callers can
    use it with either RT or fixation-time objectives. It also keeps the older
    rt_ms_* columns when they can be computed.
    """

    # Observed
    obs = df_trials_obs.copy()
    if str(time_col_obs) not in obs.columns:
        raise ValueError(f"time_col_obs={time_col_obs!r} not present in df_trials_obs")
    obs_grp = obs.groupby(list(by), dropna=False)
    obs_sum = obs_grp.agg(
        n_obs=("accept", "size"),
        accept_obs=("accept", "mean"),
        time_obs_mean=(str(time_col_obs), "mean"),
        time_obs_median=(str(time_col_obs), "median"),
    ).reset_index()

    # Optional RT columns.
    if "rt_ms" in obs.columns:
        rt_extra = obs_grp.agg(
            rt_ms_obs_mean=("rt_ms", "mean"),
            rt_ms_obs_median=("rt_ms", "median"),
        ).reset_index()
        obs_sum = obs_sum.merge(rt_extra, on=list(by), how="left")
    else:
        obs_sum["rt_ms_obs_mean"] = np.nan
        obs_sum["rt_ms_obs_median"] = np.nan

    sim = df_trials_sim.copy()
    # Prefer the newer generic time columns if present.
    sim_time_mean_col = "time_ms_sim_mean" if "time_ms_sim_mean" in sim.columns else "rt_ms_sim_mean"
    sim_time_col = "time_ms_sim" if "time_ms_sim" in sim.columns else "rt_ms_sim"
    sim_grp = sim.groupby(list(by), dropna=False)
    sim_sum = sim_grp.agg(
        n_sim_trials=("accept_sim", "size"),
        accept_sim_mean=("accept_sim_mean", "mean"),
        time_sim_mean=(sim_time_mean_col, "mean"),
        time_sim_median=(sim_time_col, "median"),
    ).reset_index()

    # Optional RT columns.
    if "rt_ms_sim_mean" in sim.columns and "rt_ms_sim" in sim.columns:
        rt_sim_extra = sim_grp.agg(
            rt_ms_sim_mean=("rt_ms_sim_mean", "mean"),
            rt_ms_sim_median=("rt_ms_sim", "median"),
        ).reset_index()
        sim_sum = sim_sum.merge(rt_sim_extra, on=list(by), how="left")
    else:
        sim_sum["rt_ms_sim_mean"] = np.nan
        sim_sum["rt_ms_sim_median"] = np.nan

    out = obs_sum.merge(sim_sum, on=list(by), how="outer")
    return out


def attach_transition_distributions(
    gaze_by_bin: Dict[int, GazeStats],
    trial_transition: pd.DataFrame,
    *,
    trial_v_bins: pd.DataFrame,
) -> Dict[int, GazeStats]:
    """Attach per-bin transition time distributions to gaze stats."""

    df = trial_transition.merge(
        trial_v_bins[["subject", "game", "trial_number", "v_bin"]],
        on=["subject", "game", "trial_number"],
        how="left",
    )

    for vbin, df_b in df.groupby("v_bin"):
        vals = df_b["transition_total_ms"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals >= 0)]
        if int(vbin) in gaze_by_bin:
            g = gaze_by_bin[int(vbin)]
            gaze_by_bin[int(vbin)] = GazeStats(
                p_relevant_fix=g.p_relevant_fix,
                durations_relevant_ms=g.durations_relevant_ms,
                durations_irrelevant_ms=g.durations_irrelevant_ms,
                transition_total_ms=vals.astype(float),
                p_center_fix=float(getattr(g, "p_center_fix", 0.0)),
                durations_center_ms=getattr(g, "durations_center_ms", np.array([], dtype=float)),
            )

    return gaze_by_bin


def simulate_one_trial_generative(
    *,
    template: TrialTemplate,
    gaze: GazeStats,
    params: ADDMParameters,
    rng: np.random.Generator,
    max_time_ms: float,
    dt_ms: float,
    include_transition_time: bool = True,
    irrelevant_mode: str = "zero",
    center_mode: str = "same_as_irrelevant",
) -> Tuple[int, float]:
    """Simulate one trial under the generative gaze model.

    Returns
    -------
    accept_sim : int
        1 if simulated choice is Accept, else 0.
    rt_ms_sim : float
        Simulated RT in ms (decision time + sampled transition time).

    Notes
    -----
    - Irrelevant fixations contribute drift=0 but noise continues.
    - Fixations are sampled IID with no immediate repeats.
    - A sampled total transition time can be added at the end.
    - If no boundary is hit by max_time_ms, the trial is treated as censored:
      choice is set to sign(V_t) and RT to max_time_ms.
    """

    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")

    upper_bound = 1.0
    lower_bound = -1.0

    relevant_mask = template.relevance == 1
    sum_relevant = float(np.sum(template.rewards[relevant_mask]))

    V_t = 0.0
    steps_elapsed = 0
    prev_item: int | None = None

    max_steps = int(np.ceil(max_time_ms / dt_ms))

    while steps_elapsed < max_steps:
        item_index, duration_ms, is_rel = sample_fixation_event_iid(
            rng=rng,
            gaze=gaze,
            rel_indices=template.rel_indices,
            irrel_indices=template.irrel_indices,
            prev_item=prev_item,
        )
        prev_item = item_index

        n_steps = int(round(duration_ms / dt_ms))
        if n_steps <= 0:
            n_steps = 1
        remaining = max_steps - steps_elapsed
        n_steps = min(n_steps, remaining)
        if n_steps <= 0:
            break

        # Drift during this fixation.
        if is_rel == 0:
            # Center fixations are encoded as item_index==-1.
            mode = str(center_mode) if int(item_index) < 0 else str(irrelevant_mode)
            if mode == "same_as_irrelevant":
                mode = str(irrelevant_mode)

            if mode == "zero":
                attended_offer = 0.0
            elif mode == "theta_sumrel":
                attended_offer = float(params.theta) * float(sum_relevant)
            elif mode == "sumrel":
                attended_offer = float(sum_relevant)
            elif mode == "phi_sumrel":
                attended_offer = float(params.phi_center) * float(sum_relevant)
            else:
                raise ValueError(f"Unknown mode for non-relevant fixation: {mode}")
        else:
            r_j = float(template.rewards[item_index])
            attended_offer = r_j + params.theta * (sum_relevant - r_j)

        drift_per_step = (params.d * attended_offer) * dt_ms
        noise = params.sigma * np.sqrt(dt_ms) * rng.normal(size=n_steps)
        increments = drift_per_step + noise
        path = V_t + np.cumsum(increments)

        hit_upper = np.flatnonzero(path >= upper_bound)
        hit_lower = np.flatnonzero(path <= lower_bound)

        if hit_upper.size or hit_lower.size:
            first_upper = int(hit_upper[0]) if hit_upper.size else None
            first_lower = int(hit_lower[0]) if hit_lower.size else None

            if first_upper is None:
                hit_idx = first_lower
                accept_sim = 0
            elif first_lower is None:
                hit_idx = first_upper
                accept_sim = 1
            else:
                if first_upper <= first_lower:
                    hit_idx = first_upper
                    accept_sim = 1
                else:
                    hit_idx = first_lower
                    accept_sim = 0

            steps_elapsed += hit_idx + 1
            dec_time_ms = steps_elapsed * dt_ms
            if include_transition_time:
                time_ms = dec_time_ms + sample_transition_total_ms(rng, gaze)
            else:
                time_ms = dec_time_ms
            return int(accept_sim), float(time_ms)

        # No hit.
        V_t = float(path[-1])
        steps_elapsed += n_steps

    # Censored: no hit within max_time.
    accept_sim = 1 if V_t >= 0 else 0
    return int(accept_sim), float(max_time_ms)


def simulate_one_trial_generative_detailed(
    *,
    template: TrialTemplate,
    gaze: GazeStats,
    params: ADDMParameters,
    rng: np.random.Generator,
    max_time_ms: float,
    dt_ms: float,
    include_transition_time: bool = True,
    irrelevant_mode: str = "zero",
    center_mode: str = "same_as_irrelevant",
) -> Dict[str, float]:
    """Like `simulate_one_trial_generative`, but returns useful diagnostics.

    Returns a dict with keys:
    - accept_sim (0/1)
    - rt_ms (float)
    - decision_time_ms (float)
    - transition_total_ms (float; NaN if censored)
    - censored (0/1)
    - n_fixations (float, integer-valued)
    - total_fixation_ms (float): fixation time *used* up to decision (== decision_time_ms)
    - relevant_fixation_ms (float): fixation time on relevant items up to decision
    - prop_relevant_fixation_ms (float): relevant_fixation_ms / total_fixation_ms
    - n_unique_items_fixated (float, integer-valued)

    Notes
    -----
        This intentionally mirrors `simulate_one_trial_generative` behavior:
        - If a bound is hit and include_transition_time=True, time includes a sampled transition total.
        - If censored (no bound hit by max_time_ms), time is max_time_ms.
    """

    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")

    upper_bound = 1.0
    lower_bound = -1.0

    relevant_mask = template.relevance == 1
    sum_relevant = float(np.sum(template.rewards[relevant_mask]))

    V_t = 0.0
    steps_elapsed = 0
    prev_item: int | None = None
    n_fix = 0
    fix_ms_total = 0.0
    fix_ms_relevant = 0.0
    seen_items: set[int] = set()

    max_steps = int(np.ceil(max_time_ms / dt_ms))

    while steps_elapsed < max_steps:
        item_index, duration_ms, is_rel = sample_fixation_event_iid(
            rng=rng,
            gaze=gaze,
            rel_indices=template.rel_indices,
            irrel_indices=template.irrel_indices,
            prev_item=prev_item,
        )
        prev_item = item_index
        n_fix += 1
        if int(item_index) >= 0:
            seen_items.add(int(item_index))

        n_steps = int(round(duration_ms / dt_ms))
        if n_steps <= 0:
            n_steps = 1
        remaining = max_steps - steps_elapsed
        n_steps = min(n_steps, remaining)
        if n_steps <= 0:
            break

        # Drift during this fixation.
        if is_rel == 0:
            mode = str(center_mode) if int(item_index) < 0 else str(irrelevant_mode)
            if mode == "same_as_irrelevant":
                mode = str(irrelevant_mode)

            if mode == "zero":
                attended_offer = 0.0
            elif mode == "theta_sumrel":
                attended_offer = float(params.theta) * float(sum_relevant)
            elif mode == "sumrel":
                attended_offer = float(sum_relevant)
            elif mode == "phi_sumrel":
                attended_offer = float(params.phi_center) * float(sum_relevant)
            else:
                raise ValueError(f"Unknown mode for non-relevant fixation: {mode}")
        else:
            r_j = float(template.rewards[item_index])
            attended_offer = r_j + params.theta * (sum_relevant - r_j)

        drift_per_step = (params.d * attended_offer) * dt_ms
        noise = params.sigma * np.sqrt(dt_ms) * rng.normal(size=n_steps)
        increments = drift_per_step + noise
        path = V_t + np.cumsum(increments)

        hit_upper = np.flatnonzero(path >= upper_bound)
        hit_lower = np.flatnonzero(path <= lower_bound)

        if hit_upper.size or hit_lower.size:
            first_upper = int(hit_upper[0]) if hit_upper.size else None
            first_lower = int(hit_lower[0]) if hit_lower.size else None

            if first_upper is None:
                hit_idx = first_lower
                accept_sim = 0
            elif first_lower is None:
                hit_idx = first_upper
                accept_sim = 1
            else:
                if first_upper <= first_lower:
                    hit_idx = first_upper
                    accept_sim = 1
                else:
                    hit_idx = first_lower
                    accept_sim = 0

            used_steps = hit_idx + 1
            used_fix_ms = float(used_steps * dt_ms)
            fix_ms_total += used_fix_ms
            if int(is_rel) == 1:
                fix_ms_relevant += used_fix_ms

            steps_elapsed += used_steps
            dec_time_ms = float(steps_elapsed * dt_ms)
            if include_transition_time:
                trans_ms = float(sample_transition_total_ms(rng, gaze))
            else:
                trans_ms = 0.0
            rt_ms = float(dec_time_ms + trans_ms)
            prop_rel = float(fix_ms_relevant / fix_ms_total) if fix_ms_total > 0 else float("nan")
            return {
                "accept_sim": float(int(accept_sim)),
                "rt_ms": float(rt_ms),
                "decision_time_ms": float(dec_time_ms),
                "transition_total_ms": float(trans_ms),
                "censored": 0.0,
                "n_fixations": float(n_fix),
                "total_fixation_ms": float(fix_ms_total),
                "relevant_fixation_ms": float(fix_ms_relevant),
                "prop_relevant_fixation_ms": float(prop_rel),
                "n_unique_items_fixated": float(len(seen_items)),
            }

        # No hit: full fixation duration was used.
        used_fix_ms = float(n_steps * dt_ms)
        fix_ms_total += used_fix_ms
        if int(is_rel) == 1:
            fix_ms_relevant += used_fix_ms

        V_t = float(path[-1])
        steps_elapsed += n_steps

    # Censored: no hit within max_time.
    accept_sim = 1 if V_t >= 0 else 0
    prop_rel = float(fix_ms_relevant / fix_ms_total) if fix_ms_total > 0 else float("nan")
    return {
        "accept_sim": float(int(accept_sim)),
        "rt_ms": float(max_time_ms),
        "decision_time_ms": float(max_time_ms),
        "transition_total_ms": float("nan") if include_transition_time else 0.0,
        "censored": 1.0,
        "n_fixations": float(n_fix),
        "total_fixation_ms": float(fix_ms_total),
        "relevant_fixation_ms": float(fix_ms_relevant),
        "prop_relevant_fixation_ms": float(prop_rel),
        "n_unique_items_fixated": float(len(seen_items)),
    }


def simulate_trials_generative_detailed(
    *,
    df_trials: pd.DataFrame,
    params: ADDMParameters,
    config: FitConfig,
    components: FittingComponents,
    n_sim_per_trial: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Like `simulate_trials_generative`, but returns extra fixation diagnostics.

    Intended for posterior predictive checks where you want summary statistics
    such as p(accept), fixation-time, number of fixations, and relevant-fixation
    proportions under the generative gaze model.

    Output columns (one row per trial):
    - subject, game, trial_number, v_offer, v_bin
    - accept_sim_mean
    - rt_ms_sim_mean
    - fix_ms_sim_mean (== decision_time_ms_sim_mean)
    - n_fixations_sim_mean
    - n_unique_items_fixated_sim_mean
    - relevant_fixation_ms_sim_mean
    - prop_relevant_fixation_ms_sim_mean
    - n_sim
    """

    if n_sim_per_trial <= 0:
        raise ValueError("n_sim_per_trial must be >= 1")

    rng = np.random.default_rng(seed if seed is not None else config.sim.seed)

    df = df_trials.copy()
    df["v_bin"] = assign_v_offer_bins_from_edges(
        df["v_offer"].to_numpy(dtype=float),
        components.v_edges,
    )

    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        vbin = int(r["v_bin"])
        gaze = components.gaze_by_bin.get(vbin)
        if gaze is None:
            continue

        tmpl = TrialTemplate(
            rewards=np.asarray(r["rewards"], dtype=float),
            relevance=np.asarray(r["relevance"], dtype=int),
            v_offer=float(r["v_offer"]),
            rel_indices=np.asarray(r["rel_indices"], dtype=int),
            irrel_indices=np.asarray(r["irrel_indices"], dtype=int),
        )

        # Accumulate Monte Carlo sums (avoid storing all draws).
        sum_accept = 0.0
        sum_rt = 0.0
        sum_fix = 0.0
        sum_nfix = 0.0
        sum_nuniq = 0.0
        sum_rel_fix = 0.0
        sum_prop_rel = 0.0

        n_eff = 0
        for _ in range(int(n_sim_per_trial)):
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            sim = simulate_one_trial_generative_detailed(
                template=tmpl,
                gaze=gaze,
                params=params,
                rng=trial_rng,
                max_time_ms=components.max_time_ms,
                dt_ms=config.sim.dt_ms,
                include_transition_time=bool(config.sim.include_transition_time),
                irrelevant_mode=str(getattr(config.sim, "irrelevant_mode", "zero")),
                center_mode=str(getattr(config.sim, "center_mode", "same_as_irrelevant")),
            )

            # All returned fields should be finite except transition_total_ms can be NaN on censor.
            sum_accept += float(sim.get("accept_sim", np.nan))
            sum_rt += float(sim.get("rt_ms", np.nan))
            sum_fix += float(sim.get("decision_time_ms", np.nan))
            sum_nfix += float(sim.get("n_fixations", np.nan))
            sum_nuniq += float(sim.get("n_unique_items_fixated", np.nan))
            sum_rel_fix += float(sim.get("relevant_fixation_ms", np.nan))
            sum_prop_rel += float(sim.get("prop_relevant_fixation_ms", np.nan))
            n_eff += 1

        denom = float(n_eff) if n_eff > 0 else float("nan")
        rows.append(
            {
                "subject": r["subject"],
                "game": float(r["game"]),
                "trial_number": int(r["trial_number"]),
                "v_offer": float(r["v_offer"]),
                "v_bin": int(vbin),
                "accept_sim_mean": float(sum_accept / denom) if np.isfinite(denom) else float("nan"),
                "rt_ms_sim_mean": float(sum_rt / denom) if np.isfinite(denom) else float("nan"),
                "fix_ms_sim_mean": float(sum_fix / denom) if np.isfinite(denom) else float("nan"),
                "n_fixations_sim_mean": float(sum_nfix / denom) if np.isfinite(denom) else float("nan"),
                "n_unique_items_fixated_sim_mean": float(sum_nuniq / denom) if np.isfinite(denom) else float("nan"),
                "relevant_fixation_ms_sim_mean": float(sum_rel_fix / denom) if np.isfinite(denom) else float("nan"),
                "prop_relevant_fixation_ms_sim_mean": float(sum_prop_rel / denom) if np.isfinite(denom) else float("nan"),
                "n_sim": int(n_sim_per_trial),
            }
        )

    return pd.DataFrame(rows)


def build_fitting_components(
    *,
    df_trials_train: pd.DataFrame,
    df_fix_train: pd.DataFrame,
    config: FitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, FittingComponents]:
    """Build training-derived components used by the simulation likelihood.

    Returns
    -------
    df_trials_train_binned : DataFrame
        Copy of df_trials_train with a 'v_bin' column.
    df_fix_train_binned : DataFrame
        Copy of df_fix_train with a 'v_bin' column (from trial merge).
    components : FittingComponents
        Fixed components (edges, gaze stats, RT bin edges, max_time).
    """

    df_trials = df_trials_train.copy()
    v_bin_trials, v_edges = assign_v_offer_bins(
        df_trials["v_offer"].to_numpy(dtype=float),
        config.binning.n_v_offer_bins,
    )
    df_trials["v_bin"] = v_bin_trials

    df_fix = df_fix_train.merge(
        df_trials[["subject", "game", "trial_number", "v_bin"]],
        on=["subject", "game", "trial_number"],
        how="left",
    )
    df_fix = df_fix.dropna(subset=["v_bin"]).copy()
    df_fix["v_bin"] = df_fix["v_bin"].astype(int)

    # Empirical gaze distributions.
    gaze_by_bin = estimate_gaze_stats(
        df_fix,
        v_bin=df_fix["v_bin"].to_numpy(dtype=int),
        center_gaze_mode=str(getattr(config.sim, "center_gaze_mode", "separate")),
    )
    trial_transition = compute_transition_time_per_trial_ms(df_fix)
    gaze_by_bin = attach_transition_distributions(gaze_by_bin, trial_transition, trial_v_bins=df_trials)

    if str(config.time_col) not in df_trials.columns:
        raise ValueError(f"config.time_col={config.time_col!r} not present in df_trials")

    rt_edges_by_cell = build_rt_bins_per_cell(
        df_trials,
        v_bin_col="v_bin",
        rt_col=str(config.time_col),
        rt_bins_max=config.binning.rt_bins_max,
        min_trials_per_rt_bin=config.binning.min_trials_per_rt_bin,
        rt_bins_fixed=getattr(config.binning, "rt_bins_fixed", 0),
    )

    max_time_ms = float(np.nanmax(pd.to_numeric(df_trials[str(config.time_col)], errors="coerce").to_numpy(dtype=float)))

    components = FittingComponents(
        v_edges=v_edges,
        gaze_by_bin=gaze_by_bin,
        rt_edges_by_cell=rt_edges_by_cell,
        max_time_ms=max_time_ms,
    )

    return df_trials, df_fix, components


def compute_binned_loglik_given_components(
    *,
    df_trials_eval: pd.DataFrame,
    params: ADDMParameters,
    config: FitConfig,
    components: FittingComponents,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute binned log-likelihood for an evaluation set using fixed components."""

    if rng is None:
        rng = np.random.default_rng(config.sim.seed)

    df_trials = df_trials_eval.copy()
    df_trials["v_bin"] = assign_v_offer_bins_from_edges(
        df_trials["v_offer"].to_numpy(dtype=float),
        components.v_edges,
    )

    rt_edges_by_cell = components.rt_edges_by_cell
    gaze_by_bin = components.gaze_by_bin

    # Data counts per cell.
    data_counts: Dict[Tuple[int, int, int], int] = {}
    for _, row in df_trials.iterrows():
        vbin = int(row["v_bin"])
        accept = int(row["accept"])
        edges = rt_edges_by_cell.get((vbin, accept))
        if edges is None:
            continue
        rt_bin = _bin_value(float(row[str(config.time_col)]), edges)
        key = (vbin, accept, rt_bin)
        data_counts[key] = data_counts.get(key, 0) + 1

    # Simulation counts per cell.
    sim_counts: Dict[Tuple[int, int, int], int] = {}

    # Pre-build per-bin trial templates from the evaluation set.
    trials_by_bin: Dict[int, List[TrialTemplate]] = {}
    for vbin, df_b in df_trials.groupby("v_bin"):
        templates: List[TrialTemplate] = []
        for _, r in df_b.iterrows():
            rewards = np.asarray(r["rewards"], dtype=float)
            relevance = np.asarray(r["relevance"], dtype=int)
            templates.append(
                TrialTemplate(
                    rewards=rewards,
                    relevance=relevance,
                    v_offer=float(r["v_offer"]),
                    rel_indices=np.asarray(r["rel_indices"], dtype=int),
                    irrel_indices=np.asarray(r["irrel_indices"], dtype=int),
                )
            )
        trials_by_bin[int(vbin)] = templates

    for vbin, templates in trials_by_bin.items():
        if vbin not in gaze_by_bin:
            continue
        gaze = gaze_by_bin[vbin]
        if not templates:
            continue

        for _ in range(config.sim.n_sim_per_vbin):
            tmpl = templates[int(rng.integers(0, len(templates)))]
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            accept_sim, rt_ms_sim = simulate_one_trial_generative(
                template=tmpl,
                gaze=gaze,
                params=params,
                rng=trial_rng,
                max_time_ms=components.max_time_ms,
                dt_ms=config.sim.dt_ms,
                include_transition_time=bool(config.sim.include_transition_time),
                irrelevant_mode=str(getattr(config.sim, "irrelevant_mode", "zero")),
            )

            edges = rt_edges_by_cell.get((vbin, accept_sim))
            if edges is None:
                continue
            rt_bin = _bin_value(float(rt_ms_sim), edges)
            key = (vbin, int(accept_sim), int(rt_bin))
            sim_counts[key] = sim_counts.get(key, 0) + 1

    ll = 0.0
    alpha = float(config.sim.alpha_smoothing)

    for (vbin, accept, rt_bin), n_data in data_counts.items():
        edges = rt_edges_by_cell.get((vbin, accept))
        if edges is None:
            continue
        k_bins = max(1, edges.size - 1)

        total_sim = 0
        for b in range(k_bins):
            total_sim += sim_counts.get((vbin, accept, b), 0)

        c_k = sim_counts.get((vbin, accept, rt_bin), 0)
        p_k = (c_k + alpha) / (total_sim + alpha * k_bins)
        ll += float(n_data) * float(np.log(p_k))

    return float(ll)


def compute_binned_loglik_per_trial_given_components(
    *,
    df_trials_eval: pd.DataFrame,
    params: ADDMParameters,
    config: FitConfig,
    components: FittingComponents,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Compute per-trial binned log-likelihood contributions.

    This matches `compute_binned_loglik_given_components`, but instead of
    returning a single scalar it returns one row per trial with the trial's
    bin assignment and its log-likelihood contribution `loglik_trial`.

    Notes
    -----
    - The likelihood is defined over (v_bin, choice, rt_bin) cells.
    - A trial contributes log p_k where p_k is the smoothed simulated
      probability of that cell under the provided params/components.
    """

    if rng is None:
        rng = np.random.default_rng(config.sim.seed)

    df_trials = df_trials_eval.copy()
    df_trials["v_bin"] = assign_v_offer_bins_from_edges(
        df_trials["v_offer"].to_numpy(dtype=float),
        components.v_edges,
    )

    rt_edges_by_cell = components.rt_edges_by_cell
    gaze_by_bin = components.gaze_by_bin

    # Simulation counts per cell.
    sim_counts: Dict[Tuple[int, int, int], int] = {}

    # Pre-build per-bin trial templates from the evaluation set.
    trials_by_bin: Dict[int, List[TrialTemplate]] = {}
    for vbin, df_b in df_trials.groupby("v_bin"):
        templates: List[TrialTemplate] = []
        for _, r in df_b.iterrows():
            rewards = np.asarray(r["rewards"], dtype=float)
            relevance = np.asarray(r["relevance"], dtype=int)
            templates.append(
                TrialTemplate(
                    rewards=rewards,
                    relevance=relevance,
                    v_offer=float(r["v_offer"]),
                    rel_indices=np.asarray(r["rel_indices"], dtype=int),
                    irrel_indices=np.asarray(r["irrel_indices"], dtype=int),
                )
            )
        trials_by_bin[int(vbin)] = templates

    for vbin, templates in trials_by_bin.items():
        if vbin not in gaze_by_bin:
            continue
        gaze = gaze_by_bin[vbin]
        if not templates:
            continue

        for _ in range(int(config.sim.n_sim_per_vbin)):
            tmpl = templates[int(rng.integers(0, len(templates)))]
            trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            accept_sim, rt_ms_sim = simulate_one_trial_generative(
                template=tmpl,
                gaze=gaze,
                params=params,
                rng=trial_rng,
                max_time_ms=components.max_time_ms,
                dt_ms=config.sim.dt_ms,
                include_transition_time=bool(config.sim.include_transition_time),
                irrelevant_mode=str(getattr(config.sim, "irrelevant_mode", "zero")),
            )

            edges = rt_edges_by_cell.get((vbin, int(accept_sim)))
            if edges is None:
                continue
            rt_bin = _bin_value(float(rt_ms_sim), edges)
            key = (int(vbin), int(accept_sim), int(rt_bin))
            sim_counts[key] = sim_counts.get(key, 0) + 1

    # Compute per-trial contributions.
    alpha = float(config.sim.alpha_smoothing)
    loglik: List[float] = []
    rt_bin_list: List[int] = []
    skip_mask: List[bool] = []

    for _, row in df_trials.iterrows():
        vbin = int(row["v_bin"])
        accept = int(row["accept"])
        edges = rt_edges_by_cell.get((vbin, accept))
        if edges is None:
            loglik.append(float("nan"))
            rt_bin_list.append(-1)
            skip_mask.append(True)
            continue

        rt_bin = _bin_value(float(row[str(config.time_col)]), edges)
        k_bins = max(1, edges.size - 1)

        total_sim = 0
        for b in range(k_bins):
            total_sim += sim_counts.get((vbin, accept, b), 0)

        c_k = sim_counts.get((vbin, accept, rt_bin), 0)
        p_k = (c_k + alpha) / (total_sim + alpha * k_bins)
        loglik.append(float(np.log(p_k)))
        rt_bin_list.append(int(rt_bin))
        skip_mask.append(False)

    df_trials["rt_bin"] = np.asarray(rt_bin_list, dtype=int)
    df_trials["loglik_trial"] = np.asarray(loglik, dtype=float)
    df_trials["ll_skipped"] = np.asarray(skip_mask, dtype=bool)
    return df_trials


def compute_binned_loglik(
    *,
    df_trials: pd.DataFrame,
    df_fix: pd.DataFrame,
    params: ADDMParameters,
    config: FitConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Convenience wrapper.

    Builds components from the provided data and evaluates likelihood on the
    same set.
    """

    df_trials_b, df_fix_b, components = build_fitting_components(
        df_trials_train=df_trials,
        df_fix_train=df_fix,
        config=config,
    )
    return compute_binned_loglik_given_components(
        df_trials_eval=df_trials_b,
        params=params,
        config=config,
        components=components,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# pyBADS fitting helpers (used by run_kfold_cv and parameter_recovery_sweep)
# ---------------------------------------------------------------------------


def sigma_from_mu(d: float, mu: float) -> float:
    """Convert mu (noise multiplier) to sigma: sigma = d * mu."""
    return float(d) * float(mu)


def fit_one_fold_pybads(
    *,
    df_trials_train: pd.DataFrame,
    df_fix_train: pd.DataFrame,
    config: "FitConfig",
    seed: int,
    noise_param: str,
    bounds: tuple,
    x0: np.ndarray,
    max_iter: int,
    max_fun_evals: int,
    fixed_sigma: float | None,
    fixed_mu: float | None,
    fixed_theta: float | None,
    fit_phi_center: bool,
    df_trials_train_b: pd.DataFrame | None = None,
    components: object | None = None,
) -> tuple:
    """Fit parameters for one fold using pyBADS; returns (best_x, best_ll_train)."""

    try:
        from pybads import BADS  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyBADS is required for mode=fit. Install via `pip install pybads`."
        ) from e

    # Build fixed components on training data only (unless provided).
    if df_trials_train_b is None or components is None:
        df_trials_train_b, _, components = build_fitting_components(
            df_trials_train=df_trials_train,
            df_fix_train=df_fix_train,
            config=config,
        )

    def fun(x: np.ndarray) -> float:
        idx = 0
        d = float(x[idx])
        idx += 1

        if fixed_theta is not None:
            theta = float(fixed_theta)
        else:
            theta = float(x[idx])
            idx += 1

        if fixed_sigma is not None:
            sigma = float(fixed_sigma)
        elif fixed_mu is not None:
            if noise_param != "mu":
                raise ValueError("fixed_mu is only valid when noise_param == 'mu'")
            sigma = sigma_from_mu(d, float(fixed_mu))
        else:
            noise = float(x[idx])
            idx += 1
            if noise_param == "mu":
                sigma = sigma_from_mu(d, noise)
            elif noise_param == "sigma":
                sigma = noise
            else:
                raise ValueError(f"Unknown noise_param: {noise_param}")

        if bool(fit_phi_center):
            phi_center = float(x[idx])
            idx += 1
        else:
            phi_center = 1.0

        params = ADDMParameters(d=d, theta=theta, sigma=sigma, phi_center=phi_center)
        ll = compute_binned_loglik_given_components(
            df_trials_eval=df_trials_train_b,
            params=params,
            config=config,
            components=components,
            rng=np.random.default_rng(seed),
        )
        return -ll

    lb, ub, plb, pub = bounds
    bads = BADS(fun, x0, lb, ub, plb, pub)
    if int(max_iter) > 0:
        bads.options["max_iter"] = int(max_iter)
    if int(max_fun_evals) > 0:
        bads.options["max_fun_evals"] = int(max_fun_evals)
    result = bads.optimize()

    best = np.asarray(result.x, dtype=float)

    idx = 0
    best_d = float(best[idx])
    idx += 1
    if fixed_theta is not None:
        best_theta = float(fixed_theta)
    else:
        best_theta = float(best[idx])
        idx += 1

    if fixed_sigma is not None:
        best_sigma = float(fixed_sigma)
        best_noise = float(best_sigma) / max(best_d, 1e-300) if noise_param == "mu" else float(best_sigma)
    elif fixed_mu is not None:
        if noise_param != "mu":
            raise ValueError("fixed_mu is only valid when noise_param == 'mu'")
        best_noise = float(fixed_mu)
        best_sigma = sigma_from_mu(best_d, best_noise)
    else:
        best_noise = float(best[idx])
        idx += 1
        if noise_param == "mu":
            best_sigma = sigma_from_mu(best_d, best_noise)
        else:
            best_sigma = float(best_noise)

    if bool(fit_phi_center):
        best_phi_center = float(best[idx])
        idx += 1
    else:
        best_phi_center = 1.0

    best_full = np.asarray(result.x, dtype=float)

    best_ll_train = compute_binned_loglik_given_components(
        df_trials_eval=df_trials_train_b,
        params=ADDMParameters(d=best_d, theta=best_theta, sigma=best_sigma, phi_center=best_phi_center),
        config=config,
        components=components,
        rng=np.random.default_rng(seed),
    )

    return best_full, float(best_ll_train)
