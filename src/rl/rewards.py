"""
Reward functions — one per RL agent role (per-step + terminal).
===============================================================
All rewards are designed to:
  1. Be dense enough for learning (not only terminal signal)
  2. Use potential-based shaping where possible to preserve optimality
     (Ng, A.Y., Harada, D., & Russell, S. 1999. "Policy Invariance Under
      Reward Transformations: Theory and Application to Reward Shaping."
      ICML-99, pp. 278–287.)
  3. Penalise the correct failure modes for each role

Values are kept in a small range (±20) so gradients remain stable.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Firefighter
# ---------------------------------------------------------------------------

def step_reward_firefighter(
    cells_extinguished: int,
    water_used: int,
    water_capacity: int,
    delta_burning: int,        # change in burning cell count (+ = more fire)
    fire_line_cells_created: int,
    is_refilling: bool,
) -> float:
    """
    Per-step reward for firefighter.

    Positive: extinguishing cells, creating fire lines.
    Negative: wasting water, fire growing while agent is active.

    Ref: Rothermel (1972) — fire line effectiveness informs reward shaping.
    """
    r = 0.0
    r += 2.0  * cells_extinguished           # direct suppression
    r += 0.5  * fire_line_cells_created      # proactive containment
    r -= 0.5  * (water_used / max(water_capacity, 1))  # penalise waste
    r -= 0.02 * max(delta_burning, 0)        # penalise fire growth
    r -= 0.1  * float(is_refilling)          # small downtime cost
    return float(r)


def terminal_reward_firefighter(burned_area_pct: float) -> float:
    """
    Terminal reward for firefighter at episode end.
    burned_area_pct in [0, 100].
    """
    containment_reward = 10.0 * (1.0 - burned_area_pct / 100.0)
    burn_penalty       = -10.0 * (burned_area_pct / 100.0)
    return float(containment_reward + burn_penalty)


# ---------------------------------------------------------------------------
# Rescuer
# ---------------------------------------------------------------------------

def step_reward_rescuer(
    civilians_rescued_this_step: int,  # reached safe zone
    delta_distance_to_target: float,   # negative = getting closer
    casualty_this_step: int,           # civilians who died this step
    path_risk: float,                  # max temp along path / 100
    is_waiting: bool,
    active_civilians: int,
) -> float:
    """
    Per-step reward for rescuer.

    Positive: rescuing civilians, approaching target.
    Negative: casualties near rescuer, risky paths, waiting when needed.

    Ref: Cova & Johnson (2002) — rescue timing shapes evacuation outcome.
    """
    r = 0.0
    r += 5.0  * civilians_rescued_this_step
    r -= 0.3  * max(delta_distance_to_target, 0)   # penalise moving away
    r += 0.15 * max(-delta_distance_to_target, 0)  # reward closing distance
    r -= 2.0  * casualty_this_step
    r -= 0.3  * path_risk                           # penalise dangerous routes
    r -= 0.15 * float(is_waiting and active_civilians > 0)
    return float(r)


def terminal_reward_rescuer(
    mortality_rate: float,
    evacuation_success_rate: float,
) -> float:
    """Terminal reward for rescuer. Dominated by outcome rates."""
    return float(-15.0 * mortality_rate + 15.0 * evacuation_success_rate)


# ---------------------------------------------------------------------------
# Commander
# ---------------------------------------------------------------------------

def step_reward_commander(
    civilians_evacuated_this_step: int,
    casualties_this_step: int,
    tti: float,
    ect: float,
    phase: int,
    action: int,             # 0=maintain, 1=advance, 2–5=other
    rescuers_idle: int,
    total_rescuers: int,
    cfp_issued: bool,
) -> float:
    """
    Per-step reward for commander.

    The core logic mirrors the ECT/TTI rule:
      TTI >> ECT → stay in low phase (no premature escalation)
      TTI ≤ ECT → must have already escalated
    Phase mismatches are penalised.

    Ref: Cova & Johnson (2002) — ECT/TTI framework for phase decisions.
    """
    r = 0.0
    r += 3.0 * civilians_evacuated_this_step
    r -= 5.0 * casualties_this_step

    # Phase appropriateness (ECT/TTI logic)
    if tti > 0 and ect > 0:
        ratio = tti / ect
        if ratio <= 1.0 and phase < 2:
            r -= 1.5   # too late, should have evacuated
        elif ratio > 2.5 and phase > 1:
            r -= 0.5   # premature escalation

    # Reward activating field units when needed
    if rescuers_idle > 0 and cfp_issued:
        r += 0.5

    return float(r)


def terminal_reward_commander(
    mortality_rate: float,
    evacuation_success_rate: float,
    burned_area_pct: float,
) -> float:
    """Terminal reward for commander — primary performance metric."""
    return float(
        -20.0 * mortality_rate
        +  20.0 * evacuation_success_rate
        -   5.0 * (burned_area_pct / 100.0)
    )
