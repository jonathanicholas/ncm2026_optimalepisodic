"""Synthetic fixation generators for next-fixation null distributions.

Both oracles take an existing (trials_df, fixations_df) pair from `data_loaders`
and produce a new (trials_df, fixations_df). For each template trial we generate
`n_repeats` synthetic sequences, each a separate row in the returned trials_df
with a unique trial_id (and matching fixation events). Increasing `n_repeats`
shrinks the null distributions' posterior credible intervals without changing
the underlying data-generating process.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from data_loaders import FIXATION_COLUMNS, NUM_SLOTS


def _per_trial_fix_counts(fixations: pd.DataFrame) -> pd.Series:
    return fixations.groupby("trial_id").size()


def _expand_trials(trials: pd.DataFrame, n_repeats: int) -> pd.DataFrame:
    """Return a trials DataFrame with `n_repeats` copies of each original trial,
    each with a suffixed trial_id (rep0, rep1, ...). Preserves all other columns."""
    rows = []
    for _, row in trials.iterrows():
        base_tid = row["trial_id"]
        for r in range(n_repeats):
            new_row = row.copy()
            new_row["trial_id"] = f"{base_tid}_rep{r}" if n_repeats > 1 else base_tid
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _randomize_encoding_order(trials: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    trials = trials.reset_index(drop=True).copy()
    trials["encoding_order_slots"] = [
        rng.permutation(NUM_SLOTS).tolist() for _ in range(len(trials))
    ]
    return trials


def random_oracle(
    trials: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    seed: int = 0,
    n_repeats: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform random over the 6 slots at each step. Matched fixation counts.

    If `n_repeats > 1`, each template trial is replayed `n_repeats` times with
    independent random sequences.
    """

    rng = np.random.default_rng(seed)
    counts = _per_trial_fix_counts(fixations)
    records: List[dict] = []
    for _, trial in trials.iterrows():
        base_tid = trial["trial_id"]
        if base_tid not in counts.index:
            continue
        n = int(counts.loc[base_tid])
        if n <= 0:
            continue
        is_rel = trial["is_relevant_per_slot"]
        for r in range(n_repeats):
            tid = f"{base_tid}_rep{r}" if n_repeats > 1 else base_tid
            # disallow consecutive same-slot picks (we model fixation events, not stays)
            prev = -1
            for fi in range(n):
                choices = [s for s in range(NUM_SLOTS) if s != prev]
                slot = int(rng.choice(choices))
                records.append({
                    "subject": trial["subject"],
                    "trial_id": tid,
                    "fix_idx": fi,
                    "slot": slot,
                    "fix_start": float(fi),
                    "fix_duration": 1.0,
                    "is_relevant": int(is_rel[slot]),
                })
                prev = slot
    fixs = pd.DataFrame.from_records(records, columns=FIXATION_COLUMNS)
    keep_ids = set(fixs["trial_id"])
    expanded = _expand_trials(trials, n_repeats)
    return (
        expanded[expanded["trial_id"].isin(keep_ids)].reset_index(drop=True),
        fixs,
    )


def walk_ring_noisy(
    trials: pd.DataFrame,
    fixations: pd.DataFrame,
    *,
    seed: int = 0,
    n_repeats: int = 1,
    p_random: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adjacent ring walk with a small random-jump probability.

    At each step, with probability ``1 - p_random`` take an adjacent +1/-1 step
    around the ring (direction drawn fresh each step); with probability
    ``p_random`` jump to a uniformly random slot != current. The random jumps
    break the quasi-complete separation on spatial distance that a pure ring
    walk produces, keeping the distance coefficient finite. Each synthetic
    trial's encoding order is re-randomized (the walk is encoding-blind), so the
    encoding-order predictors are an exact null. Matched fixation counts.
    """

    rng = np.random.default_rng(seed)
    counts = _per_trial_fix_counts(fixations)
    records: List[dict] = []
    for _, trial in trials.iterrows():
        base_tid = trial["trial_id"]
        if base_tid not in counts.index:
            continue
        n = int(counts.loc[base_tid])
        if n <= 0:
            continue
        is_rel = trial["is_relevant_per_slot"]
        for r in range(n_repeats):
            tid = f"{base_tid}_rep{r}" if n_repeats > 1 else base_tid
            slot = int(rng.integers(0, NUM_SLOTS))
            for fi in range(n):
                records.append({
                    "subject": trial["subject"],
                    "trial_id": tid,
                    "fix_idx": fi,
                    "slot": slot,
                    "fix_start": float(fi),
                    "fix_duration": 1.0,
                    "is_relevant": int(is_rel[slot]),
                })
                if rng.random() < p_random:
                    choices = [s for s in range(NUM_SLOTS) if s != slot]
                    slot = int(rng.choice(choices))
                else:
                    step = -1 if rng.random() < 0.5 else 1
                    slot = (slot + step) % NUM_SLOTS
    fixs = pd.DataFrame.from_records(records, columns=FIXATION_COLUMNS)
    keep_ids = set(fixs["trial_id"])
    expanded = _expand_trials(trials, n_repeats)
    expanded = expanded[expanded["trial_id"].isin(keep_ids)].reset_index(drop=True)
    expanded = _randomize_encoding_order(expanded, rng)
    return (expanded, fixs)
