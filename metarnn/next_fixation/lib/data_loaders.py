"""Unified loaders for human, RNN, and oracle fixation data.

Output shape (shared by all loaders):
    trials_df: one row per (subject, trial_id), with per-slot true rewards / relevances
        and per-trial encoding order.
    fixations_df: long-form fixation events with one row per event.

Both DataFrames share keys (subject, trial_id) so they join cleanly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
SIM_DIR = REPO_ROOT / "metarnn" / "simulations"
GAME_INFO_DIR = REPO_ROOT / "task" / "emdm-eyetracking" / "game_info"

NUM_SLOTS = 6
NUM_FEATURES = 4

# The four query features in the order used by the experiment.
# `options` integer in the simulation JSONs is the index into this list.
FEATURE_TOKENS = ("Animal", "Land", "Pattern", "Large")


# ---------------------------------------------------------------------------
# Trial schema
# ---------------------------------------------------------------------------

TRIAL_COLUMNS = [
    "subject",
    "trial_id",
    "game",
    "trial_number",
    "option",                # offered feature token (e.g. "Animal")
    "true_rewards",          # length-6 list, indexed by slot
    "is_relevant_per_slot",  # length-6 list of {0, 1} — under working-set
                             # convention: 1 iff slot was fixated AND truly
                             # relevant (so 0 covers both fixated-irrelevant
                             # and never-fixated)
    "is_fixated_per_slot",   # length-6 list of {0, 1}: 1 iff slot was fixated
                             # at any point during the trial
    "images_per_slot",       # length-6 list of image identifiers (None for synthetic data)
    "encoding_order_slots",  # length-6 list: slot indices in encoding presentation order
    "offer_value",
    "choice",                # 1=take, 0=leave (best effort across sources)
]

FIXATION_COLUMNS = [
    "subject",
    "trial_id",
    "fix_idx",          # 0-based event index within the trial
    "slot",             # 0..5
    "fix_start",
    "fix_duration",     # ms for human, abstract steps for RNN/oracles
    "is_relevant",
]


# ---------------------------------------------------------------------------
# Human loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HumanLoaderConfig:
    excluded_subjects: Tuple[str, ...] = ("107", "131")
    data_dir: Path = DATA_DIR


def _list_subject_ids(data_dir: Path, excluded: Iterable[str]) -> List[str]:
    excl = {str(s) for s in excluded}
    out: List[str] = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir() or not p.name.isdigit():
            continue
        if p.name in excl:
            continue
        out.append(p.name)
    return out


def _load_human_subject(subid: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one subject's trials and fixations.

    Data layout:
      - In choice-phase rows, `trial_number` indexes the decision trial; each
        decision belongs to a `game` whose 6 items were learned during the prior
        encoding block.
      - In encoding-phase rows, `trial_number` indexes ITEM presentation within
        a game (e.g. game 1 has encoding trials 1..6, game 2 has 7..12). The
        sequence of distinct images, sorted by encoding `trial_number`, is the
        per-game encoding order.

    We therefore key encoding info by `game`, not by decision trial_number.
    """

    fix_path = data_dir / subid / f"{subid}_fixations_for_modeling.csv"
    full_path = data_dir / subid / f"{subid}_fixations_df_original_buffer_50.csv"
    games_path = GAME_INFO_DIR / f"games_{subid}.json"
    fix_df = pd.read_csv(fix_path)
    full_df = pd.read_csv(full_path, low_memory=False)
    # Per-subject games file: gives the 6 images in each game (encoding order).
    # We use it to fill in item_index for items the subject never fixated, so
    # those subjects are not dropped from the analysis.
    with open(games_path) as fh:
        games_by_idx = json.load(fh)         # list[list[image_name]] indexed by game-1

    # --- choice-phase fixations as events ---
    fix_df = fix_df[fix_df["phase"] == "choice"].copy()
    fix_df["subject"] = str(subid)
    for col in ("game", "trial_number", "item_index", "is_relevant"):
        fix_df[col] = pd.to_numeric(fix_df[col], errors="coerce").astype("Int64")
    fix_df["fix_start"] = pd.to_numeric(fix_df["fix_start"], errors="coerce")
    fix_df["fix_duration_bounded"] = pd.to_numeric(fix_df["fix_duration_bounded"], errors="coerce")
    fix_df = fix_df.dropna(subset=["game", "trial_number", "item_index", "fix_start", "fix_duration_bounded"]).copy()
    fix_df["trial_id"] = (
        fix_df["subject"] + "_g" + fix_df["game"].astype(int).astype(str)
        + "_t" + fix_df["trial_number"].astype(int).astype(str)
    )

    # collapse consecutive same-slot fixations within a trial into events
    fix_df = fix_df.sort_values(["trial_id", "fix_start"]).reset_index(drop=True)
    same_as_prev = (
        (fix_df["item_index"] == fix_df["item_index"].shift(1))
        & (fix_df["trial_id"] == fix_df["trial_id"].shift(1))
    )
    fix_df["event_id"] = (~same_as_prev).cumsum()
    events = (
        fix_df.groupby("event_id", as_index=False)
        .agg(
            subject=("subject", "first"),
            trial_id=("trial_id", "first"),
            slot=("item_index", "first"),
            is_relevant=("is_relevant", "first"),
            fix_start=("fix_start", "min"),
            fix_duration=("fix_duration_bounded", "sum"),
        )
        .drop(columns=["event_id"])
    )
    events = events.sort_values(["trial_id", "fix_start"]).reset_index(drop=True)
    events["fix_idx"] = events.groupby("trial_id").cumcount()
    fixations_out = events[FIXATION_COLUMNS].copy()

    # --- per-game encoding info ---
    # Each game has 6 unique items presented sequentially in trial_number order.
    enc = full_df[full_df["phase"] == "encoding"].copy()
    enc["game"] = pd.to_numeric(enc["game"], errors="coerce").astype("Int64")
    enc["trial_number"] = pd.to_numeric(enc["trial_number"], errors="coerce").astype("Int64")
    enc["outcome"] = pd.to_numeric(enc["outcome"], errors="coerce")
    enc = enc.dropna(subset=["game", "trial_number", "image", "outcome"]).copy()
    # one row per (game, item) — image is unique within a game
    enc_first = (
        enc.sort_values(["game", "trial_number"])
        .drop_duplicates(subset=["game", "image"], keep="first")
        [["game", "trial_number", "image", "outcome"]]
        .rename(columns={"trial_number": "encoding_trial_number"})
    )

    # image -> slot mapping per game. `item_index` in fixations_for_modeling.csv
    # is assigned via pd.factorize(roi_content) in the order of first fixation,
    # so we recover that mapping for fixated images. For images the subject
    # never fixated in a given game, we assign the unused slot indices in the
    # encoding order from `games_<sub>.json`. Those filler slots never appear
    # as the current slot in any fixation event (since the item wasn't fixated),
    # only as candidates, so the choice of which unused slot to assign is
    # immaterial -- it just lets us include trials whose 6-slot vectors would
    # otherwise be incomplete.
    fixated_map = (
        fix_df.dropna(subset=["roi_content"])
        .groupby(["game", "roi_content"])["item_index"]
        .first()
        .reset_index()
        .rename(columns={"roi_content": "image", "item_index": "slot"})
    )
    img_to_slot_records = []
    for game_1based, image_list in enumerate(games_by_idx, start=1):
        sub_fixated = fixated_map[fixated_map["game"] == game_1based]
        used_slots = set(int(s) for s in sub_fixated["slot"])
        fixated_images = set(sub_fixated["image"])
        next_unused = (s for s in range(NUM_SLOTS) if s not in used_slots)
        for image_name in image_list:
            if image_name in fixated_images:
                slot = int(sub_fixated[sub_fixated["image"] == image_name]["slot"].iloc[0])
            else:
                slot = next(next_unused, None)
                if slot is None:
                    continue
            img_to_slot_records.append({
                "game": game_1based,
                "image": image_name,
                "slot": int(slot),
            })
    img_to_slot = pd.DataFrame(img_to_slot_records)
    img_to_slot["game"] = img_to_slot["game"].astype("Int64")

    enc_with_slot = enc_first.merge(img_to_slot, on=["game", "image"], how="inner")
    # Per game: dict slot -> (encoding_position_0idx, true_reward, image)
    per_game_info: dict[int, dict[int, Tuple[int, float, str]]] = {}
    for game, sub in enc_with_slot.groupby("game"):
        sub = sub.sort_values("encoding_trial_number").reset_index(drop=True)
        if len(sub) != NUM_SLOTS:
            continue
        per_game_info[int(game)] = {
            int(row["slot"]): (i, float(row["outcome"]), str(row["image"]))
            for i, row in sub.iterrows()
        }

    # --- assemble per-trial table ---
    trials_records: List[dict] = []
    trial_meta = (
        fix_df.groupby("trial_id", as_index=False)
        .agg(
            subject=("subject", "first"),
            game=("game", "first"),
            trial_number=("trial_number", "first"),
            option=("option", "first"),
            choice=("choice", "first"),
        )
    )
    for _, r in trial_meta.iterrows():
        tid = r["trial_id"]
        game = int(r["game"]) if pd.notna(r["game"]) else None
        if game is None or game not in per_game_info:
            continue
        info = per_game_info[game]
        if len(info) != NUM_SLOTS:
            continue
        true_rewards = [info[s][1] for s in range(NUM_SLOTS)]
        images_per_slot = [info[s][2] for s in range(NUM_SLOTS)]
        encoding_position_per_slot = [info[s][0] for s in range(NUM_SLOTS)]
        # encoding_order_slots[i] = slot presented i-th
        encoding_order_slots = [None] * NUM_SLOTS
        for s, (epos, _, _) in info.items():
            encoding_order_slots[epos] = s
        # is_relevant per slot — ground-truth relevance for ALL items, computed
        # from the offer token + image-name rule used in the paper's pipeline
        # (analysis/lib/choice_fixation_proportions.py). Independent of which
        # items the subject actually fixated.
        option_token = r["option"] if isinstance(r["option"], str) else None
        if option_token is None:
            is_relevant_per_slot = [0] * NUM_SLOTS
        else:
            is_relevant_per_slot = []
            for s in range(NUM_SLOTS):
                image_name = info[s][2]
                rel = (option_token in image_name.split("_")
                       or option_token in image_name)
                is_relevant_per_slot.append(int(bool(rel)))
        # Slot is "fixated" if it appears in this trial's fixation events.
        fixated_slots = set(fix_df[fix_df["trial_id"] == tid]["item_index"]
                            .dropna().astype(int))
        is_fixated_per_slot = [int(s in fixated_slots) for s in range(NUM_SLOTS)]
        offer_value = sum(rv * rel for rv, rel in zip(true_rewards, is_relevant_per_slot))
        ch = r["choice"]
        try:
            chf = float(ch)
            choice_bin = 1 if chf == 1.0 else (0 if chf == 2.0 else np.nan)
        except (TypeError, ValueError):
            choice_bin = np.nan
        trials_records.append({
            "subject": str(subid),
            "trial_id": tid,
            "game": game,
            "trial_number": int(r["trial_number"]) if pd.notna(r["trial_number"]) else None,
            "option": r["option"],
            "true_rewards": true_rewards,
            "is_relevant_per_slot": is_relevant_per_slot,
            "is_fixated_per_slot": is_fixated_per_slot,
            "images_per_slot": images_per_slot,
            "encoding_order_slots": encoding_order_slots,
            "offer_value": offer_value,
            "choice": choice_bin,
        })

    trials_out = pd.DataFrame.from_records(trials_records, columns=TRIAL_COLUMNS)
    fixations_out = fixations_out[fixations_out["trial_id"].isin(set(trials_out["trial_id"]))].reset_index(drop=True)
    return trials_out, fixations_out


def load_human(config: Optional[HumanLoaderConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = config or HumanLoaderConfig()
    trials_chunks: List[pd.DataFrame] = []
    fixs_chunks: List[pd.DataFrame] = []
    for subid in _list_subject_ids(config.data_dir, config.excluded_subjects):
        try:
            t, f = _load_human_subject(subid, config.data_dir)
        except FileNotFoundError as e:
            print(f"[human loader] missing files for {subid}: {e}")
            continue
        if len(t) == 0:
            continue
        trials_chunks.append(t)
        fixs_chunks.append(f)
    if not trials_chunks:
        raise RuntimeError(f"No human trials loaded from {config.data_dir}")
    return (
        pd.concat(trials_chunks, ignore_index=True),
        pd.concat(fixs_chunks, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# RNN loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RNNLoaderConfig:
    sim_dir: Path = SIM_DIR
    variant: str = "input5"     # 'input5' = prior memory; 'input0' = no prior
    n_synthetic_subjects: int = 35
    trials_per_subject: int = 60
    seed: int = 0


def _load_rnn_raw(sim_subdir: Path) -> dict:
    """Concatenate all data_*.json files in a simulation subdirectory."""

    json_paths = sorted(sim_subdir.glob("data_*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No data_*.json under {sim_subdir}")

    keys = ["pairs", "values", "relevances", "options", "offer_values", "actions"]
    out = {k: [] for k in keys}
    for p in json_paths:
        with open(p) as fh:
            d = json.load(fh)
        for k in keys:
            out[k].extend(d[k])
    return out


def load_rnn(config: Optional[RNNLoaderConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and subsample RNN simulations into trials/fixations format.

    Synthetic subjects each get a contiguous random subset of the trial pool.
    """
    config = config or RNNLoaderConfig()
    sim_subdir = config.sim_dir / f"simulation_04_04_{config.variant}"
    raw = _load_rnn_raw(sim_subdir)

    n_trials_total = len(raw["actions"])
    rng = np.random.default_rng(config.seed)
    n_total_needed = config.n_synthetic_subjects * config.trials_per_subject
    if n_total_needed > n_trials_total:
        raise ValueError(f"Only {n_trials_total} RNN trials available, need {n_total_needed}")
    chosen = rng.choice(n_trials_total, size=n_total_needed, replace=False)
    chosen_chunks = chosen.reshape(config.n_synthetic_subjects, config.trials_per_subject)

    trials_records: List[dict] = []
    fix_records: List[dict] = []
    for syn_idx, idxs in enumerate(chosen_chunks):
        subject = f"NN{config.variant}_{syn_idx:02d}"
        for local_t, ridx in enumerate(idxs):
            actions = raw["actions"][int(ridx)]
            true_rewards = list(raw["values"][int(ridx)])
            ground_truth_rel = list(raw["relevances"][int(ridx)])
            offer_val = float(raw["offer_values"][int(ridx)])
            opt_idx = int(raw["options"][int(ridx)])
            option_token = FEATURE_TOKENS[opt_idx] if 0 <= opt_idx < len(FEATURE_TOKENS) else str(opt_idx)
            trial_id = f"{subject}_t{local_t:04d}"

            # extract fixation slots from action sequence (drop terminal decide actions 6/7)
            fix_slots = [int(a) for a in actions if 0 <= int(a) < NUM_SLOTS]
            # `is_relevant_per_slot` = ground-truth relevance for ALL items
            # (matches the corresponding human convention now). Whether the
            # slot is in the active working set is captured separately by
            # `is_fixated_per_slot` and applied only to share_k.
            is_relevant_per_slot = [int(x) for x in ground_truth_rel]
            fixated_set = set(fix_slots)
            is_fixated_per_slot = [int(s in fixated_set) for s in range(NUM_SLOTS)]
            decide_actions = [int(a) for a in actions if int(a) >= NUM_SLOTS]
            choice_bin = 1 if (decide_actions and decide_actions[-1] == NUM_SLOTS) else 0

            # collapse consecutive same-slot to events; duration = run-length
            events: List[Tuple[int, int]] = []
            cur = None
            run = 0
            for s in fix_slots:
                if cur is None or s != cur:
                    if cur is not None:
                        events.append((cur, run))
                    cur = s
                    run = 1
                else:
                    run += 1
            if cur is not None:
                events.append((cur, run))

            t_clock = 0
            for fi, (slot, dur) in enumerate(events):
                fix_records.append({
                    "subject": subject,
                    "trial_id": trial_id,
                    "fix_idx": fi,
                    "slot": int(slot),
                    "fix_start": float(t_clock),
                    "fix_duration": float(dur),
                    "is_relevant": int(is_relevant_per_slot[int(slot)]),
                })
                t_clock += dur

            # encoding order: the env never exposes encoding order to the RNN, so any
            # per-trial encoding order we assign is a placeholder. Use a *random*
            # permutation per trial so encoding lag is decorrelated from spatial lag —
            # if we used slot-index order, encoding lag would alias the spatial lag and
            # produce spurious encoding-CRP peaks. The randomization gives the correct
            # flat null for encoding predictors that the meta-MDP / RNN should produce.
            encoding_order_slots = list(rng.permutation(NUM_SLOTS).astype(int))

            trials_records.append({
                "subject": subject,
                "trial_id": trial_id,
                "game": None,
                "trial_number": local_t,
                "option": option_token,
                "true_rewards": true_rewards,
                "is_relevant_per_slot": is_relevant_per_slot,
                "is_fixated_per_slot": is_fixated_per_slot,
                "images_per_slot": [None] * NUM_SLOTS,
                "encoding_order_slots": encoding_order_slots,
                "offer_value": offer_val,
                "choice": choice_bin,
            })

    return (
        pd.DataFrame.from_records(trials_records, columns=TRIAL_COLUMNS),
        pd.DataFrame.from_records(fix_records, columns=FIXATION_COLUMNS),
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def summarize(trials: pd.DataFrame, fixations: pd.DataFrame, label: str) -> None:
    n_subj = trials["subject"].nunique()
    n_trials = len(trials)
    n_fix = len(fixations)
    mean_fix_per_trial = n_fix / max(n_trials, 1)
    print(f"[{label}] subjects={n_subj}  trials={n_trials}  fix_events={n_fix}  mean_per_trial={mean_fix_per_trial:.2f}")
