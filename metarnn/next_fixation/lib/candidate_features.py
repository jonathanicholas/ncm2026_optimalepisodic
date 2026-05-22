"""Build the long-form per-candidate dataframe for the next-fixation
conditional logit.

For each fixation transition i -> j in the choice phase, emit one row per
candidate item k != i, with outcome `chose_k = 1[k == j]`. Columns are the
candidate-level state needed to derive the ten model predictors (spatial,
encoding-order, reward, and fixation-history features).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from data_loaders import NUM_SLOTS


def _signed_enc_lag(i_pos: int, k_pos: int) -> int:
    """Signed encoding-position lag k_pos - i_pos (linear encoding axis, no wrap)."""
    return int(k_pos) - int(i_pos)


def build_next_fixation_long(trials: pd.DataFrame, fixations: pd.DataFrame) -> pd.DataFrame:
    """Build the long-form per-candidate dataframe for the next-fixation logit.

    One row per (event_id, candidate_slot). Returns columns:
      subject, trial_id, event_id, current_slot, candidate_slot, chose_k,
      is_prev_fixation_k, cum_time_k, abs_reward_k, signed_reward_k,
      is_relevant_k, is_fixated_k, is_primacy_k, is_recency_k, signed_enc_lag_ik
    """

    trial_index = trials.set_index("trial_id")
    fixations = fixations.sort_values(["trial_id", "fix_idx"]).reset_index(drop=True)

    records: List[dict] = []
    event_id = 0
    for tid, group in fixations.groupby("trial_id", sort=False):
        if tid not in trial_index.index:
            continue
        trial = trial_index.loc[tid]
        true_rewards = trial["true_rewards"]
        is_rel_slot = trial["is_relevant_per_slot"]
        is_fix_slot = trial.get("is_fixated_per_slot", [0] * NUM_SLOTS)
        encoding_order = trial["encoding_order_slots"]
        if encoding_order is None or any(p is None for p in encoding_order):
            continue
        # encoding_position_per_slot[s] = the index (0..5) at which slot s was encoded
        encoding_position_per_slot = [0] * NUM_SLOTS
        for pos, slot in enumerate(encoding_order):
            encoding_position_per_slot[int(slot)] = pos

        slots = group["slot"].astype(int).to_numpy()
        durations = group["fix_duration"].astype(float).to_numpy()

        cum_time_so_far = np.zeros(NUM_SLOTS, dtype=float)

        for step in range(len(slots) - 1):
            i = int(slots[step])
            j = int(slots[step + 1])
            cum_time_so_far[i] += float(durations[step])
            prev_slot = int(slots[step - 1]) if step >= 1 else -1

            for k in range(NUM_SLOTS):
                if k == i:
                    continue
                k_enc_pos = encoding_position_per_slot[k]
                i_enc_pos = encoding_position_per_slot[i]
                records.append({
                    "subject": trial["subject"],
                    "trial_id": tid,
                    "event_id": event_id,
                    "current_slot": i,
                    "candidate_slot": k,
                    "chose_k": int(k == j),
                    "is_prev_fixation_k": int(prev_slot == k),
                    "cum_time_k": float(cum_time_so_far[k]),
                    "abs_reward_k": abs(float(true_rewards[k])),
                    "signed_reward_k": float(true_rewards[k]),
                    "is_relevant_k": int(is_rel_slot[k]),
                    "is_fixated_k": int(is_fix_slot[k]),
                    "is_primacy_k": int(k_enc_pos == 0),
                    "is_recency_k": int(k_enc_pos == NUM_SLOTS - 1),
                    "signed_enc_lag_ik": _signed_enc_lag(i_enc_pos, k_enc_pos),
                })
            event_id += 1

    return pd.DataFrame.from_records(records)
