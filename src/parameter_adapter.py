"""
ParameterAdapter — adjusts Commander and Civilian parameters
between Monte Carlo runs based on observed outcomes.

Strategy: gradient-free hill-climbing with momentum.
After each run, if mortality improved → reinforce the last change.
If mortality worsened → revert and try the opposite direction.
"""
from .config import (
    COMMANDER_PHASE_MONITOR_MULTIPLIER,
    COMMANDER_PHASE_PREALERT_MULTIPLIER,
    COMMANDER_PHASE_EVACUATE_MULTIPLIER,
    CIVILIAN_PANIC_RATIONAL,
    CIVILIAN_PANIC_CONFUSED,
)

_DELTA = 0.05          # step size per parameter update
_TARGET_MORTALITY = 0.05  # aim for < 5% casualty rate


class ParameterAdapter:
    def __init__(self):
        # Current best parameter values (start from config defaults)
        self.params = {
            'phase_monitor_mult':  COMMANDER_PHASE_MONITOR_MULTIPLIER,
            'phase_prealert_mult': COMMANDER_PHASE_PREALERT_MULTIPLIER,
            'phase_evacuate_mult': COMMANDER_PHASE_EVACUATE_MULTIPLIER,
            'panic_rational':      CIVILIAN_PANIC_RATIONAL,
            'panic_confused':      CIVILIAN_PANIC_CONFUSED,
        }
        # Bounds (min, max)
        self._bounds = {
            'phase_monitor_mult':  (1.5, 5.0),
            'phase_prealert_mult': (1.0, 3.0),
            'phase_evacuate_mult': (0.5, 2.0),
            'panic_rational':      (0.2, 0.6),
            'panic_confused':      (0.5, 0.9),
        }
        self.run_history = []            # list of mortality_rate per run
        self._last_mortality = None
        self._last_direction = {}        # param → +1 or -1 (last change direction)

    def update(self, result: dict) -> None:
        """Called after each simulation run with the results dict."""
        mortality = result.get('mortality_rate', 0.0)
        self.run_history.append(mortality)

        if self._last_mortality is None:
            # First run: try tightening phase thresholds slightly
            self._last_mortality = mortality
            self._adjust('phase_monitor_mult',  +_DELTA)
            self._adjust('phase_prealert_mult', +_DELTA)
            return

        improved = mortality < self._last_mortality

        if improved:
            # Keep going in the same direction
            for key, direction in self._last_direction.items():
                self._adjust(key, direction * _DELTA * 0.5)  # smaller follow-up step
        else:
            # Revert and try opposite
            for key, direction in self._last_direction.items():
                self._adjust(key, -direction * _DELTA)

        # Always try to tighten phase thresholds if mortality > target
        if mortality > _TARGET_MORTALITY:
            self._adjust('phase_monitor_mult',  +_DELTA)
            self._adjust('phase_prealert_mult', +_DELTA)
        elif mortality == 0.0:
            # Possibly too conservative, relax slightly
            self._adjust('phase_monitor_mult',  -_DELTA * 0.3)

        self._last_mortality = mortality

    def _adjust(self, key: str, delta: float) -> None:
        lo, hi = self._bounds[key]
        self.params[key] = max(lo, min(hi, self.params[key] + delta))
        self._last_direction[key] = 1 if delta > 0 else -1

    def get_overrides(self) -> dict:
        """Returns dict of parameter overrides for the next simulation run."""
        return dict(self.params)

    def print_summary(self) -> None:
        if not self.run_history:
            return
        import numpy as np
        print(f"\n  Parameter Adapter — {len(self.run_history)} runs:")
        print(f"    Avg mortality: {np.mean(self.run_history):.2%}")
        print(f"    Current params: {self.params}")
