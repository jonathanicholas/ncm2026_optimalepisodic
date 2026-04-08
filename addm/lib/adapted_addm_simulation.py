"""Simulation of the adapted aDDM for the eye-EMDM task.

This module implements the discrete-time stochastic process. It treats
the fixation sequence as exogenous input
and simulates the relative decision value (RDV) until it hits the fixed
boundaries at +1 (Accept) or -1 (Leave).

Basic usage
-----------

- Use `simulate_trial` when you already have a per-time-step fixation sequence
  j(t) over discrete steps.

- Use `simulate_trial_from_events` when you have a sequence of fixations with
  durations (e.g., from eye-tracking), and you want to expand them into
  discrete steps under a chosen time bin size.

The core parameters are:
- d:      drift scaling factor (> 0)
- theta:  attentional bias toward the fixated relevant item in [0, 1]
- sigma:  diffusion scale (> 0). In a continuous-time interpretation, noise
          enters as sigma * sqrt(dt) per time step.

The absorbing boundaries are fixed at ±1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np


@dataclass
class ADDMParameters:
    """Parameters for the adapted aDDM.

    Attributes
    ----------
    d : float
        Drift scaling factor (> 0).
    theta : float
        Attentional bias toward fixated relevant items, in [0, 1]. When theta = 1,
        gaze has no effect and the model reduces toward a standard DDM over the
        sum of relevant item rewards.
    sigma : float
        Standard deviation of Gaussian noise per time step (> 0).
    phi_center : float
        Gain on the value signal during center fixations in center-extended
        variants (>= 0). This parameter is ignored by baseline simulations that
        do not include center fixations.
    """

    d: float
    theta: float
    sigma: float
    phi_center: float = 1.0

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be > 0")
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError("theta must be in [0, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.phi_center < 0:
            raise ValueError("phi_center must be >= 0")


@dataclass
class TrialResult:
    """Result of a single simulated trial."""

    choice: Optional[str]
    decision_time_steps: Optional[int]
    rdv_trajectory: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "decision_time_steps": self.decision_time_steps,
            "rdv_trajectory": self.rdv_trajectory,
        }


def simulate_trial(
    rewards: Sequence[float],
    relevance: Sequence[int],
    fixation_sequence: Sequence[int],
    params: ADDMParameters,
    dt: float = 1.0,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    irrelevant_mode: str = "zero",
) -> TrialResult:
    """Simulate a single trial of the adapted aDDM.

    Parameters
    ----------
    rewards : sequence of float, shape (N,)
        Rewards r_i for each of the N items.
    relevance : sequence of int {0,1}, shape (N,)
        Relevance indicators R_i for each item (1 = relevant, 0 = irrelevant).
    fixation_sequence : sequence of int, shape (T,)
        Index j(t) of the fixated item at each discrete time step.
        Indices should be 0-based and in [0, N-1].
    params : ADDMParameters
        Model parameters (d, theta, sigma).
    dt : float, default 1.0
        Time step size for the discrete-time approximation. This sets the
        units of time for the simulated dynamics. If you interpret one model
        step as 1 ms, use dt = 1.0 with fixation durations expressed in ms.
        If you use seconds, dt might be 0.001 for 1 ms.
    max_steps : int, optional
        Maximum number of time steps to simulate. If None, uses len(fixation_sequence).
        If the boundary is not reached within `max_steps`, the trial ends with
        choice=None.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, defaults to np.random.default_rng().

    Returns
    -------
    TrialResult
        Contains the choice ('Accept', 'Leave', or None), the decision time in
        steps, and the full RDV trajectory as a NumPy array.
    """

    rewards_arr = np.asarray(rewards, dtype=float)
    relevance_arr = np.asarray(relevance, dtype=int)

    if dt <= 0:
        raise ValueError("dt must be > 0")

    if rewards_arr.ndim != 1:
        raise ValueError("rewards must be a 1D sequence")
    if relevance_arr.shape != rewards_arr.shape:
        raise ValueError("relevance must have the same shape as rewards")

    n_items = rewards_arr.shape[0]

    fixation_arr = np.asarray(fixation_sequence, dtype=int)
    if fixation_arr.ndim != 1:
        raise ValueError("fixation_sequence must be a 1D sequence of indices")
    if np.any((fixation_arr < 0) | (fixation_arr >= n_items)):
        raise ValueError("fixation_sequence contains invalid item indices")

    if max_steps is None:
        max_steps = fixation_arr.shape[0]
    else:
        max_steps = min(max_steps, fixation_arr.shape[0])

    if rng is None:
        rng = np.random.default_rng()

    V_t = 0.0
    rdv_traj = np.empty(max_steps + 1, dtype=float)
    rdv_traj[0] = V_t

    choice: Optional[str] = None
    decision_time_steps: Optional[int] = None

    # Absorbing boundaries at ±1
    upper_bound = 1.0
    lower_bound = -1.0

    relevant_mask = relevance_arr == 1
    sum_relevant = float(np.sum(rewards_arr[relevant_mask]))

    for t in range(max_steps):
        j_t = fixation_arr[t]

        # If the currently fixated item is irrelevant (R_{j(t)} = 0), control
        # whether evidence accumulation continues (covert integration) or
        # stops (drift=0). Default matches the baseline model: drift=0.
        if relevance_arr[j_t] == 0:
            if str(irrelevant_mode) == "zero":
                attended_offer = 0.0
            elif str(irrelevant_mode) == "theta_sumrel":
                attended_offer = float(params.theta) * float(sum_relevant)
            elif str(irrelevant_mode) == "sumrel":
                attended_offer = float(sum_relevant)
            else:
                raise ValueError(f"Unknown irrelevant_mode: {irrelevant_mode}")
        else:
            # Compute attentional weights w_i(t) according to Eq. (weights):
            #   w_i(t) = 1      if R_i = 1 and i = j(t)
            #           = theta if R_i = 1 and i != j(t)
            #           = 0     if R_i = 0
            w = np.where(relevance_arr == 1, params.theta, 0.0)
            w[j_t] = 1.0

            # Attended offer value at time t over relevant items.
            attended_offer = float(np.sum(w * rewards_arr))

        # Drift/diffusion dynamics (Euler-Maruyama discretization):
        #   V_{t+dt} = V_t + (d * attended_offer) * dt + sigma * sqrt(dt) * N(0, 1)
        #   Note: when dt=1, this reduces to the simpler form of noise = sigma * N(0, 1) per step.
        noise = params.sigma * np.sqrt(dt) * rng.normal()
        V_t = V_t + (params.d * attended_offer) * dt + noise
        rdv_traj[t + 1] = V_t

        # Check absorbing boundaries.
        if V_t >= upper_bound:
            choice = "Accept"
            decision_time_steps = t + 1
            # Truncate trajectory to the moment of decision.
            rdv_traj = rdv_traj[: decision_time_steps + 1]
            break
        if V_t <= lower_bound:
            choice = "Leave"
            decision_time_steps = t + 1
            rdv_traj = rdv_traj[: decision_time_steps + 1]
            break

    return TrialResult(choice=choice, decision_time_steps=decision_time_steps, rdv_trajectory=rdv_traj)


def simulate_trial_from_events_fast(
    rewards: Sequence[float],
    relevance: Sequence[int],
    fixation_events: Sequence[Tuple[int, float]],
    params: ADDMParameters,
    dt: float = 1.0,
    max_time: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    irrelevant_mode: str = "zero",
) -> TrialResult:
    """Fast simulation from fixation events (vectorized within each fixation).

    Compared to `simulate_trial_from_events`, this function avoids creating a
    full per-step fixation sequence and avoids a Python loop over every time
    step. Instead, it simulates one fixation event at a time, generating all
    per-step increments for that event in NumPy, then checks for the first
    boundary crossing within the event.

    This is especially useful for fitting, where many simulations are run.

    Parameters
    ----------
    rewards, relevance, fixation_events, params, dt, max_time, rng
        Same meanings as in `simulate_trial_from_events`.

    Returns
    -------
    TrialResult
        As in `simulate_trial`.
    """

    if dt <= 0:
        raise ValueError("dt must be > 0")

    rewards_arr = np.asarray(rewards, dtype=float)
    relevance_arr = np.asarray(relevance, dtype=int)

    if rewards_arr.ndim != 1:
        raise ValueError("rewards must be a 1D sequence")
    if relevance_arr.shape != rewards_arr.shape:
        raise ValueError("relevance must have the same shape as rewards")

    n_items = rewards_arr.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    upper_bound = 1.0
    lower_bound = -1.0

    # Precompute quantities used during relevant fixations.
    relevant_mask = relevance_arr == 1
    sum_relevant = float(np.sum(rewards_arr[relevant_mask]))

    V_t = 0.0
    steps_elapsed = 0
    rdv_traj: List[float] = [V_t]

    max_steps: Optional[int]
    if max_time is None:
        max_steps = None
    else:
        max_steps = int(np.floor(max_time / dt))
        if max_steps <= 0:
            max_steps = 1

    for item_index, duration in fixation_events:
        if item_index < 0 or item_index >= n_items:
            raise ValueError("fixation_events contains invalid item index")
        if duration <= 0:
            continue

        n_steps = int(round(duration / dt))
        if n_steps <= 0:
            n_steps = 1

        if max_steps is not None:
            remaining = max_steps - steps_elapsed
            if remaining <= 0:
                break
            n_steps = min(n_steps, remaining)

        # Drift contribution for this fixation.
        if relevance_arr[item_index] == 0:
            if str(irrelevant_mode) == "zero":
                attended_offer = 0.0
            elif str(irrelevant_mode) == "theta_sumrel":
                attended_offer = float(params.theta) * float(sum_relevant)
            elif str(irrelevant_mode) == "sumrel":
                attended_offer = float(sum_relevant)
            else:
                raise ValueError(f"Unknown irrelevant_mode: {irrelevant_mode}")
        else:
            # When fixating relevant item j:
            # attended_offer = r_j + theta * sum_{i relevant, i!=j} r_i
            r_j = float(rewards_arr[item_index])
            attended_offer = r_j + params.theta * (sum_relevant - r_j)

        drift_per_step = (params.d * attended_offer) * dt
        noise = params.sigma * np.sqrt(dt) * rng.normal(size=n_steps)
        increments = drift_per_step + noise

        path = V_t + np.cumsum(increments)

        # Check for first boundary crossing within this fixation.
        hit_upper = np.flatnonzero(path >= upper_bound)
        hit_lower = np.flatnonzero(path <= lower_bound)

        if hit_upper.size or hit_lower.size:
            first_upper = int(hit_upper[0]) if hit_upper.size else None
            first_lower = int(hit_lower[0]) if hit_lower.size else None

            if first_upper is None:
                hit_idx = first_lower
                choice = "Leave"
            elif first_lower is None:
                hit_idx = first_upper
                choice = "Accept"
            else:
                if first_upper <= first_lower:
                    hit_idx = first_upper
                    choice = "Accept"
                else:
                    hit_idx = first_lower
                    choice = "Leave"

            # hit_idx is 0-based within this fixation.
            hit_steps = hit_idx + 1
            steps_elapsed += hit_steps

            # Append trajectory up to the hitting point.
            rdv_traj.extend(path[:hit_steps].tolist())
            return TrialResult(
                choice=choice,
                decision_time_steps=steps_elapsed,
                rdv_trajectory=np.asarray(rdv_traj, dtype=float),
            )

        # No hit: advance state and append full path.
        V_t = float(path[-1])
        steps_elapsed += n_steps
        rdv_traj.extend(path.tolist())

    return TrialResult(
        choice=None,
        decision_time_steps=None,
        rdv_trajectory=np.asarray(rdv_traj, dtype=float),
    )


def simulate_trial_from_events(
    rewards: Sequence[float],
    relevance: Sequence[int],
    fixation_events: Sequence[Tuple[int, float]],
    params: ADDMParameters,
    dt: float = 1.0,
    max_time: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    irrelevant_mode: str = "zero",
) -> TrialResult:
    """Simulate a single trial from fixation events with durations.

    This is a convenience wrapper for `simulate_trial` when you have fixation
    events (item index and duration in seconds or arbitrary time units) instead
    of a per-step fixation sequence.

    Parameters
    ----------
    rewards : sequence of float, shape (N,)
        Rewards r_i for each of the N items.
    relevance : sequence of int {0,1}, shape (N,)
        Relevance indicators R_i for each item (1 = relevant, 0 = irrelevant).
    fixation_events : sequence of (item_index, duration)
        Each tuple specifies which item was fixated and for how long. The
        duration is interpreted in the same time units as `dt`.
        Item indices should be 0-based and in [0, N-1].
    params : ADDMParameters
        Model parameters (d, theta, sigma).
    dt : float, default 1.0
        Time step size corresponding to one model step, in the same units as
        the fixation durations. Used to convert durations into discrete steps.
        Examples:
        - If durations are in ms, dt=1.0 means 1 ms per step.
        - If durations are in seconds, dt=0.001 means 1 ms per step.
    max_time : float, optional
        Maximum time (in the same units as the fixation durations) to simulate.
        If None, uses the sum of all event durations.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, defaults to np.random.default_rng().

    Returns
    -------
    TrialResult
        As in `simulate_trial`. The decision_time_steps can be converted to
        physical time by multiplying by `dt`.
    """

    if dt <= 0:
        raise ValueError("dt must be > 0")

    rewards_arr = np.asarray(rewards, dtype=float)
    n_items = rewards_arr.shape[0]

    fix_items: List[int] = []
    for item_index, duration in fixation_events:
        if item_index < 0 or item_index >= n_items:
            raise ValueError("fixation_events contains invalid item index")
        if duration <= 0:
            continue  # skip non-positive durations
        steps = int(round(duration / dt))
        if steps <= 0:
            # If rounding makes the duration too small, enforce at least 1 step.
            steps = 1
        fix_items.extend([item_index] * steps)

    if not fix_items:
        # No valid fixation steps; just return a trajectory at V=0.
        rdv_traj = np.array([0.0], dtype=float)
        return TrialResult(choice=None, decision_time_steps=None, rdv_trajectory=rdv_traj)

    fixation_sequence = np.array(fix_items, dtype=int)

    if max_time is not None:
        max_steps = int(max_time / dt)
        if max_steps <= 0:
            max_steps = 1
    else:
        max_steps = None

    return simulate_trial(
        rewards=rewards_arr,
        relevance=relevance,
        fixation_sequence=fixation_sequence,
        params=params,
        dt=dt,
        max_steps=max_steps,
        rng=rng,
        irrelevant_mode=str(irrelevant_mode),
    )


def simulate_block(
    rewards_list: Sequence[Sequence[float]],
    relevance_list: Sequence[Sequence[int]],
    fixation_sequences: Sequence[Sequence[int]],
    params: ADDMParameters,
    max_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[TrialResult]:
    """Simulate a block of trials given per-step fixation sequences.

    Parameters
    ----------
    rewards_list : sequence of length M
        Each element is a sequence of rewards for one trial.
    relevance_list : sequence of length M
        Each element is a sequence of relevance indicators for one trial.
    fixation_sequences : sequence of length M
        Each element is a sequence of fixated-item indices for one trial.
    params : ADDMParameters
        Model parameters (d, theta, sigma).
    max_steps : int, optional
        Maximum number of steps per trial. If None, each trial uses the full
        length of its fixation sequence.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, a new generator is created.

    Returns
    -------
    list of TrialResult
        One result per trial.
    """

    if rng is None:
        rng = np.random.default_rng()

    if not (len(rewards_list) == len(relevance_list) == len(fixation_sequences)):
        raise ValueError("rewards_list, relevance_list, and fixation_sequences must have the same length")

    results: List[TrialResult] = []
    for rewards, relevance, fix_seq in zip(rewards_list, relevance_list, fixation_sequences):
        # Use a fresh RNG for each trial to avoid dependence on internal calls.
        trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        result = simulate_trial(
            rewards=rewards,
            relevance=relevance,
            fixation_sequence=fix_seq,
            params=params,
            max_steps=max_steps,
            rng=trial_rng,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Minimal example: one trial with 6 items, 3 relevant and 3 irrelevant,
    # and a toy fixation sequence.
    params = ADDMParameters(d=0.01, theta=0.5, sigma=0.1)

    rewards = [5, -3, 2, 1, -2, 4]
    relevance = [1, 1, 1, 0, 0, 0]

    # Toy fixation sequence over 100 steps: cycle through the 6 items.
    fixation_sequence = [i % 6 for i in range(100)]

    result = simulate_trial(
        rewards=rewards,
        relevance=relevance,
        fixation_sequence=fixation_sequence,
        params=params,
    )

    print("Choice:", result.choice)
    print("Decision time (steps):", result.decision_time_steps)
    print("Final RDV:", result.rdv_trajectory[-1])
