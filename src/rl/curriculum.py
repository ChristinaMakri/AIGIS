"""
Training curriculum across the 9 historical fire scenarios.
============================================================
Implements difficulty-ordered curriculum learning:
  Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
  "Curriculum Learning." ICML-09, pp. 41–48.
  [Start with easy scenarios; add harder ones as training progresses.]

Scenarios are ordered by fire intensity (ROTHERMEL_BASE_ROS × WIND_SPEED):
  Phase 1 (easy):   Bages, Var, Penteli       — low wind / low ROS
  Phase 2 (medium): Rhodes, Kineta, Varibobi  — moderate wind
  Phase 3 (hard):   Carr, Glass, Woolsey      — high wind + fast spread

Mati and Camp Fire are HELD OUT — never used in training.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Scenario definitions (mirrors TRAINING_LOCATIONS in train_models.py)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Phase 1: Easy ────────────────────────────────────────────────────
    {
        'name': 'Bages, Catalonia',
        'phase': 1,
        'lat': 41.698, 'lon': 1.802, 'radius': 3000,
        'fire_locations': [(41.710, 1.814), (41.704, 1.808)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 35.0,
            'FIRE_SPREAD_PROB_BASE': 0.32, 'ROTHERMEL_BASE_ROS': 0.60,
            'NUM_CIVILIANS': 40,
        },
    },
    {
        'name': 'Var, France',
        'phase': 1,
        'lat': 43.352, 'lon': 6.198, 'radius': 3000,
        'fire_locations': [(43.364, 6.210), (43.358, 6.204)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.38, 'ROTHERMEL_BASE_ROS': 0.72,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        'name': 'Penteli, Athens',
        'phase': 1,
        'lat': 38.056, 'lon': 23.868, 'radius': 3000,
        'fire_locations': [(38.067, 23.879), (38.062, 23.873)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0,  'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.55,
            'NUM_CIVILIANS': 50,
        },
    },
    # ── Phase 2: Medium ──────────────────────────────────────────────────
    {
        'name': 'Rhodes, Greece',
        'phase': 2,
        'lat': 36.198, 'lon': 28.002, 'radius': 3000,
        'fire_locations': [(36.210, 28.014), (36.204, 28.008)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 180.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.35, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        'name': 'Kineta, Corinth',
        'phase': 2,
        'lat': 38.008, 'lon': 23.140, 'radius': 3000,
        'fire_locations': [(38.019, 23.152), (38.013, 23.146)],
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 80,
        },
    },
    {
        'name': 'Varibobi, Athens',
        'phase': 2,
        'lat': 38.128, 'lon': 23.798, 'radius': 3000,
        'fire_locations': [(38.140, 23.810), (38.134, 23.804)],
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
    },
    # ── Phase 3: Hard ────────────────────────────────────────────────────
    {
        'name': 'Carr Fire, Redding CA',
        'phase': 3,
        'lat': 40.588, 'lon': -122.392, 'radius': 3000,
        'fire_locations': [(40.600, -122.380), (40.594, -122.386)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.00,
            'NUM_CIVILIANS': 65,
        },
    },
    {
        'name': 'Glass Fire, Napa CA',
        'phase': 3,
        'lat': 38.498, 'lon': -122.402, 'radius': 3000,
        'fire_locations': [(38.510, -122.390), (38.504, -122.396)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
    },
    {
        'name': 'Woolsey Fire, Thousand Oaks CA',
        'phase': 3,
        'lat': 34.172, 'lon': -118.872, 'radius': 3000,
        'fire_locations': [(34.184, -118.860), (34.178, -118.866)],
        'params': {
            'WIND_SPEED': 28.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 90,
        },
    },
]


class ScenarioCurriculum:
    """
    Manages scenario sampling according to curriculum phase.

    Phase 1 (episodes 0   → phase1_end):   easy only
    Phase 2 (episodes ph1 → phase2_end):   easy + medium
    Phase 3 (episodes ph2 → end):          all 9 scenarios

    Bengio et al. (2009) — start simple, add complexity gradually.
    """

    def __init__(
        self,
        phase1_end: int = 2000,
        phase2_end: int = 6000,
        rng_seed: Optional[int] = None,
    ):
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.rng = np.random.default_rng(rng_seed)
        self._episode = 0

    @property
    def current_curriculum_phase(self) -> int:
        if self._episode < self.phase1_end:
            return 1
        if self._episode < self.phase2_end:
            return 2
        return 3

    def sample(self) -> dict:
        """Return a scenario dict appropriate for the current episode count."""
        max_phase = self.current_curriculum_phase
        pool = [s for s in SCENARIOS if s['phase'] <= max_phase]
        idx  = int(self.rng.integers(0, len(pool)))
        self._episode += 1
        return pool[idx]

    def advance(self, n: int = 1) -> None:
        self._episode += n
