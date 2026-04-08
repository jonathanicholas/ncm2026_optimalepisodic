"""Generative gaze model utilities for the adapted aDDM (Option B fitting).

This module implements the *empirical* gaze generator

- We condition gaze statistics on binned offer value V_offer.
- Each simulated fixation is generated IID (given V_offer bin):
  1) sample whether the fixation is to a relevant vs irrelevant item,
  2) sample a fixation duration from the corresponding empirical distribution,
  3) sample a target item uniformly within the chosen category,
     with the constraint that immediate repeats are not allowed.

- Transition time between fixations is not modeled dynamically; instead, we
  sample a total per-trial transition time from the empirical distribution and
  add it to the simulated decision time, following Krajbich et al. (2010).

The gaze generator itself has *no free parameters*: all probabilities and
empirical distributions are estimated directly from the cleaned fixation data.

This code assumes fixation durations are in milliseconds (ms).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrialTemplate:
    """Minimal per-trial information needed to simulate the model.

    Attributes
    ----------
    rewards : np.ndarray
        Array of length N=6 with item rewards.
    relevance : np.ndarray
        Array of length N=6 with entries in {0,1}.
    v_offer : float
        Sum of rewards for relevant items.
    rel_indices : np.ndarray
        1D array with indices of relevant items.
    irrel_indices : np.ndarray
        1D array with indices of irrelevant items.
    """

    rewards: np.ndarray
    relevance: np.ndarray
    v_offer: float
    rel_indices: np.ndarray
    irrel_indices: np.ndarray


@dataclass(frozen=True)
class GazeStats:
    """Empirical gaze distributions for a given V_offer bin."""

    p_relevant_fix: float
    durations_relevant_ms: np.ndarray
    durations_irrelevant_ms: np.ndarray
    transition_total_ms: np.ndarray

    # Optional center-fixation channel. If p_center_fix==0 or durations_center_ms
    # is empty, center fixations are never sampled.
    p_center_fix: float = 0.0
    durations_center_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _safe_choice(rng: np.random.Generator, values: np.ndarray) -> float:
    """Sample one value from a 1D array (must be non-empty)."""

    if values.size == 0:
        raise ValueError("Cannot sample from an empty array")
    idx = int(rng.integers(0, values.size))
    return float(values[idx])


def sample_fixation_event_iid(
    *,
    rng: np.random.Generator,
    gaze: GazeStats,
    rel_indices: np.ndarray,
    irrel_indices: np.ndarray,
    prev_item: int | None,
) -> Tuple[int, float, int]:
    """Sample one fixation event (item_index, duration_ms, is_relevant).

    Immediate repeats are disallowed: the returned item_index will never equal
    prev_item.

    If the sampled category combined with the no-repeat constraint would leave
    no valid targets (should not occur), the category is flipped.
    """

    # Optional center fixation.
    p_center = float(getattr(gaze, "p_center_fix", 0.0))
    p_center = float(np.clip(p_center, 0.0, 0.99))
    if p_center > 0.0 and getattr(gaze, "durations_center_ms", np.array([], dtype=float)).size > 0:
        if rng.random() < p_center:
            # Enforce no-immediate-repeats across *all* targets.
            if prev_item is None or int(prev_item) != -1:
                duration_ms = _safe_choice(rng, gaze.durations_center_ms)
                return -1, float(duration_ms), 0

    is_relevant = int(rng.random() < gaze.p_relevant_fix)

    if is_relevant == 1:
        duration_ms = _safe_choice(rng, gaze.durations_relevant_ms)
        candidates = rel_indices
    else:
        duration_ms = _safe_choice(rng, gaze.durations_irrelevant_ms)
        candidates = irrel_indices

    if prev_item is not None:
        candidates = candidates[candidates != prev_item]

    if candidates.size == 0:
        # Fallback: flip category and try again.
        is_relevant = 1 - is_relevant
        if is_relevant == 1:
            duration_ms = _safe_choice(rng, gaze.durations_relevant_ms)
            candidates = rel_indices
        else:
            duration_ms = _safe_choice(rng, gaze.durations_irrelevant_ms)
            candidates = irrel_indices

        if prev_item is not None:
            candidates = candidates[candidates != prev_item]

    if candidates.size == 0:
        raise ValueError("No valid fixation targets after enforcing no-repeat constraint")

    item_index = int(candidates[int(rng.integers(0, candidates.size))])
    return item_index, float(duration_ms), is_relevant


def sample_transition_total_ms(rng: np.random.Generator, gaze: GazeStats) -> float:
    """Sample a total transition time (ms) to add to the decision time."""

    if gaze.transition_total_ms.size == 0:
        return 0.0
    return _safe_choice(rng, gaze.transition_total_ms)
