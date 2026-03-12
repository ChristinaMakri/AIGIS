"""
RiskMonitorAgent — Model-Based Architecture
Pre-ignition fire risk assessment using FWI + historical ignitions + terrain.

This agent runs BEFORE any fire starts and produces an ignition_risk_grid that
tells the Commander where fires are LIKELY to start. It enables pre-emptive
asset positioning rather than reactive response.

Risk formula (per cell):
    risk = 0.40 × fwi_factor
         + 0.30 × fuel_factor         (fuel type flammability)
         + 0.20 × historical_factor   (FIRMS ignition density)
         + 0.10 × slope_factor        (uphill = faster spread)

All components normalised to [0, 1].

Canadian Forest Fire Weather Index (FWI) system used for fwi_factor:
  Van Wagner, C.E. (1987).
  Development and Structure of the Canadian Forest Fire Weather Index System.
  Forestry Technical Report 35. Canadian Forestry Service, Ottawa.

  Van Wagner, C.E. & Pickett, T.L. (1985).
  Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System.
  Forestry Technical Report 33. Canadian Forestry Service, Ottawa.

NFFL fuel models used for fuel_factor (see _build_fuel_factor()):
  Anderson, H.E. (1982).
  Aids to Determining Fuel Models for Estimating Fire Behavior.
  USDA Forest Service General Technical Report INT-122.
  Intermountain Forest and Range Experiment Station, Ogden, UT.

NASA FIRMS VIIRS data used for historical ignition density:
  Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I.A. (2014).
  "The New VIIRS 375 m active fire detection data product: Algorithm description
  and initial assessment."
  Remote Sensing of Environment, 143, pp. 85–96.
  https://doi.org/10.1016/j.rse.2013.12.008
"""
import numpy as np
from typing import Tuple, Optional
from .base_agent import Agent
from ..message import Message
from ..config import RISK_MONITOR_UPDATE_INTERVAL, FWI_HIGH_RISK_THRESHOLD, FWI_EXTREME_RISK_THRESHOLD


class RiskMonitorAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        RiskMonitor
    ARCHITECTURE: Model-Based BDI — Pre-Ignition Risk Assessment
    ───────────────────────────────────────────────────────────────────────
    BELIEFS  (world model — updated every RISK_MONITOR_UPDATE_INTERVAL steps)
      • risk_grid       per-cell ignition probability [0,1] synthesised from
                        FWI (40%) + fuel type (30%) + FIRMS density (20%)
                        + terrain slope (10%)
      • fwi_score       current Canadian Fire Weather Index composite
      • max_risk_cell   (row,col) of highest-risk cell in the grid

    DESIRES
      • Accurately predict WHERE fire is likely to start before ignition
      • Enable Commander to pre-position assets at high-risk zones

    INTENTIONS
      • Periodically (every N steps) recompute four-component risk grid
      • Normalise all components to [0,1]; zero non-fuel cells; set
        already-burning cells to 1.0
      • Broadcast RISK_FORECAST with top-3 risk zones to Commander

    COMMUNICATION
      SENDS
        → INFORM  commander  {type:'RISK_FORECAST', fwi, max_risk,
                              mean_risk, high_risk_zones, timestamp}
              pre-fire risk forecast (sent on every recomputation)
      RECEIVES
        (none — reads FWI, fuel, terrain directly from shared environment)

    BIBLIOGRAPHY
      [1] Van Wagner, C.E. (1987). Development and Structure of the Canadian
          Forest Fire Weather Index System. Forestry Technical Report 35.
          Canadian Forestry Service, Ottawa.
          FWI component used as primary (40%) risk factor.
      [2] Van Wagner, C.E. & Pickett, T.L. (1985). Equations and FORTRAN
          Program for the Canadian Forest Fire Weather Index System.
          Forestry Technical Report 33. Canadian Forestry Service, Ottawa.
      [3] Anderson, H.E. (1982). Aids to Determining Fuel Models for
          Estimating Fire Behavior. USDA Forest Service GTR INT-122.
          13 NFFL fuel model spread_multiplier → fuel factor (30%).
      [4] Schroeder, W. et al. (2014). "The New VIIRS 375 m active fire
          detection data product." Remote Sensing of Environment, 143,
          pp. 85–96. DOI: 10.1016/j.rse.2013.12.008
          FIRMS ignition density grid → historical factor (20%).
      [5] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        super().__init__(agent_id, position)

        # Internalized world model
        self.risk_grid: Optional[np.ndarray] = None
        self.fwi_score: float = 0.0
        self.max_risk_cell: Optional[Tuple[int, int]] = None
        # Start at the interval so the first decide() call triggers immediately
        self.steps_since_update: int = RISK_MONITOR_UPDATE_INTERVAL
        self._environment = None

    # ------------------------------------------------------------------
    # Perceive → Decide → Act
    # ------------------------------------------------------------------

    def perceive(self, environment) -> None:
        """Read FWI, terrain, and fuel data from the environment."""
        self._environment = environment
        fwi_data = getattr(environment, 'fwi_data', {})
        self.fwi_score = float(fwi_data.get('fwi', 5.0))

    def decide(self) -> None:
        """Recompute the ignition risk grid periodically."""
        self.steps_since_update += 1
        if self.steps_since_update >= RISK_MONITOR_UPDATE_INTERVAL:
            self._recompute_risk_grid()
            self.steps_since_update = 0

    def act(self, environment) -> None:
        """
        Write the risk grid to the environment and notify Commander.
        Sends a RISK_FORECAST whenever the grid has just been (re)computed.
        """
        first_call = self.risk_grid is None
        if first_call:
            self._recompute_risk_grid()

        # Write ignition_risk_grid into shared environment state
        environment.ignition_risk_grid = self.risk_grid

        # Notify Commander with summary statistics on fresh computations
        if (first_call or self.steps_since_update == 0) and self.risk_grid is not None:
            max_risk = float(self.risk_grid.max())
            mean_risk = float(self.risk_grid.mean())

            # Find top-3 highest-risk cells for sentinel pre-positioning
            flat_indices = np.argpartition(self.risk_grid.ravel(), -3)[-3:]
            top_cells = [
                (int(i // self.risk_grid.shape[1]), int(i % self.risk_grid.shape[1]))
                for i in flat_indices
            ]
            # Convert to lat/lon
            top_latlon = []
            for (r, c) in top_cells:
                lat, lon = environment.grid_to_latlon(r, c)
                top_latlon.append((lat, lon))

            msg = Message(
                sender=self.agent_id,
                receiver="commander",
                performative="INFORM",
                content={
                    'type': 'RISK_FORECAST',
                    'fwi': self.fwi_score,
                    'max_risk': max_risk,
                    'mean_risk': mean_risk,
                    'high_risk_zones': top_latlon,
                    'timestamp': environment.step_count,
                }
            )
            self.send_message(msg)

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    def _recompute_risk_grid(self) -> None:
        """
        Build the per-cell ignition risk grid from all available data layers.

        Components:
        1. FWI factor  — normalised Canadian Fire Weather Index (0–100+)
        2. Fuel factor — NFFL fuel model spread multiplier
        3. Historical  — FIRMS ignition density from the last N days
        4. Slope factor — terrain gradient (fire spreads faster uphill)
        """
        env = self._environment
        if env is None:
            return

        shape = env.grid_shape

        # ---- Component 1: FWI factor (0–1) --------------------------------
        fwi_factor = np.full(shape, min(self.fwi_score / 60.0, 1.0), dtype=np.float32)

        # ---- Component 2: Fuel factor (0–1) --------------------------------
        fuel_factor = self._build_fuel_factor(env)

        # ---- Component 3: Historical FIRMS density (0–1) ------------------
        firms_density = getattr(env, 'firms_density', None)
        if firms_density is not None and firms_density.shape == shape:
            hist_factor = firms_density.astype(np.float32)
        else:
            hist_factor = np.zeros(shape, dtype=np.float32)

        # ---- Component 4: Slope factor (0–1) --------------------------------
        slope_factor = self._build_slope_factor(env)

        # ---- Weighted combination -------------------------------------------
        risk = (0.40 * fwi_factor
                + 0.30 * fuel_factor
                + 0.20 * hist_factor
                + 0.10 * slope_factor)

        # Areas with no fuel cannot ignite
        no_fuel_mask = (env.fire_grid == 0) | (env.fire_grid == 2)
        risk[no_fuel_mask] = 0.0

        # Already-burning cells have maximum risk
        risk[env.fire_grid == 1] = 1.0

        self.risk_grid = np.clip(risk, 0.0, 1.0)

        # Cache the highest-risk cell for reporting
        idx = np.unravel_index(np.argmax(self.risk_grid), shape)
        self.max_risk_cell = (int(idx[0]), int(idx[1]))

        env = self._environment
        step = getattr(env, 'step_count', '?')
        print(f"  [RiskMonitor] Step {step}: FWI={self.fwi_score:.1f}, "
              f"max_risk={self.risk_grid.max():.3f}, "
              f"mean_risk={self.risk_grid.mean():.3f}, "
              f"high-risk cells (>0.5): {int((self.risk_grid > 0.5).sum())}")

    def _build_fuel_factor(self, env) -> np.ndarray:
        """
        Map NFFL fuel type → normalised flammability (0–1).
        Uses spread_multiplier from FUEL_MODELS config.

        Fuel model classifications from:
        Anderson, H.E. (1982). Aids to Determining Fuel Models for Estimating
        Fire Behavior. USDA Forest Service General Technical Report INT-122.
        """
        try:
            from ..config import FUEL_MODELS
            fuel_grid = env.fuel_type_grid
            factor = np.zeros(env.grid_shape, dtype=np.float32)
            max_mult = max(m.get('spread_multiplier', 1.0) for m in FUEL_MODELS.values())
            if max_mult == 0:
                max_mult = 1.0
            for code, params in FUEL_MODELS.items():
                mult = params.get('spread_multiplier', 0.0)
                factor[fuel_grid == code] = mult / max_mult
            return factor
        except Exception:
            return np.ones(env.grid_shape, dtype=np.float32) * 0.5

    def _build_slope_factor(self, env) -> np.ndarray:
        """
        Normalise terrain gradient to [0, 1].
        Steep uphill cells are higher risk (fire runs uphill faster).
        """
        try:
            elev = env.elevation_grid.astype(np.float32)
            grad_y, grad_x = np.gradient(elev)
            slope_mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
            max_slope = slope_mag.max()
            if max_slope > 0:
                return (slope_mag / max_slope).astype(np.float32)
            return np.zeros(env.grid_shape, dtype=np.float32)
        except Exception:
            return np.zeros(env.grid_shape, dtype=np.float32)
