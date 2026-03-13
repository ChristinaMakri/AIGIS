"""
Analyst Agent — BDI / Information Processing Architecture
Uses Rothermel's Fire Spread Model + Fuzzy Logic for risk assessment
Implements Time To Impact (TTI) and escape route analysis

Rothermel fire-spread model (Rate of Spread):
  Rothermel, R.C. (1972).
  "A mathematical model for predicting fire spread in wildland fuels."
  USDA Forest Service Research Paper INT-RP-115. Ogden, UT.
  Formula: ROS = R_base × (1 + φ_wind) × (1 + φ_slope)
           φ_wind = C × U^B     (wind factor)
           φ_slope = 5.275 × tan²(φ)  (slope factor)

NFFL Fuel Models used for fuel-type classification:
  Anderson, H.E. (1982).
  "Aids to determining fuel models for estimating fire behavior."
  USDA Forest Service General Technical Report INT-GTR-122.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import List, Dict, Tuple
from .base_agent import Agent
from ..message import Message
from ..config import (
    ROTHERMEL_BASE_ROS,
    ROTHERMEL_WIND_C,
    ROTHERMEL_WIND_B,
    ANALYST_TTI_IMMINENT,
    ANALYST_TTI_NEAR,
    ANALYST_EXIT_BOTTLENECK_THRESHOLD,
    WIND_SPEED,
    WIND_INITIAL_DIRECTION,
    WIND_OSCILLATION_PERIOD,
    WIND_OSCILLATION_AMPLITUDE
)

# Step-ahead fire predictor (optional — gracefully degrades if unavailable)
try:
    from ..fire_predictor import StepAheadPredictor as _StepAheadPredictor
    _FIRE_PREDICTOR_AVAILABLE = True
except ImportError:
    _FIRE_PREDICTOR_AVAILABLE = False
    _StepAheadPredictor = None


class AnalystAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Analyst
    ARCHITECTURE: BDI — Information Processing with Physics Models
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • fire_reports        raw detections from Sentinels [{lat,lon,intensity}]
      • risk_assessments    (lat,lon) → fuzzy risk level 0–100
      • tti_value           Time To Impact (seconds until fire reaches assets)
      • ros_value           Rate of Spread (m/s, Rothermel model)
      • current_phase       Commander's current evacuation phase (0–3)
                            received via PHASE_UPDATE → adjusts TTI thresholds
      • suppressed_cells    set of (row,col) cells recently extinguished by
                            Firefighters (received via SUPPRESSION_UPDATE)

    DESIRES
      • Provide accurate, timely risk intelligence to Commander
      • Compute TTI conservatively when evacuation is underway (Phase ≥ 2)

    INTENTIONS
      1. Aggregate Sentinel FIRE_DETECTION reports each step
      2. Drop any fire_report for a cell in suppressed_cells (already out)
      3. Apply Rothermel model → ROS; compute TTI = distance / ROS
      4. If Phase ≥ 2: apply 20% TTI reduction (more conservative estimate)
      5. Run fuzzy inference (TTI × route availability → risk 0–100)
      6. Send RISK_REPORT to Commander

    COMMUNICATION
      SENDS
        → INFORM  commander  {type:'RISK_REPORT', max_risk, avg_risk,
                              tti, ros, num_exits, fire_locations, timestamp}
      RECEIVES
        ← INFORM  sentinel    {type:'FIRE_DETECTION', lat, lon, intensity}
        ← INFORM  commander   {type:'PHASE_UPDATE', phase}
              belief update: adjust TTI conservatism for active evacuation
        ← INFORM  firefighter {type:'SUPPRESSION_UPDATE', row, col}
              belief update: remove extinguished cell from active fire reports

    BIBLIOGRAPHY
      [1] Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire
          Spread in Wildland Fuels. USDA Forest Service Research Paper
          INT-115. Intermountain Forest and Range Experiment Station, UT.
          ROS = R_base × (1+φ_wind) × (1+φ_slope)
      [2] Anderson, H.E. (1982). Aids to Determining Fuel Models for
          Estimating Fire Behavior. USDA Forest Service GTR INT-122.
          Fuel classifications for slope/ROS context.
      [3] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          Intention revision: phase & suppression updates modify beliefs
          before each TTI computation cycle.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Initialize Analyst agent with Rothermel fire model and fuzzy logic.

        The Analyst performs scientific risk assessment by:
        1. Calculating Rate of Spread (ROS) using Rothermel's fire spread model
        2. Computing Time To Impact (TTI): distance_to_assets / ROS
        3. Assessing escape route availability (bottlenecks)
        4. Using fuzzy logic inference to determine overall risk level

        Output: Risk reports sent to Commander for strategic decision making

        Architecture: Model-based deductive reasoning
        - Uses physics model (Rothermel) for fire behavior prediction
        - Fuzzy inference handles uncertainty in risk assessment
        - Does not execute actions, only provides intelligence

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # ===== FIRE INTELLIGENCE DATA =====
        self.fire_reports: List[Dict] = []  # Reports from Sentinel agents
        self.risk_assessments: Dict[Tuple[int, int], float] = {}  # Grid position → risk
        self._environment = None  # Set by perceive(); used in decide()

        # ===== KEY METRICS =====
        self.tti_value = float('inf')  # Time To Impact (how long until fire reaches civilians)
        self.ros_value = 0.0  # Rate of Spread (m/s, from Rothermel model)

        # ===== BDI BELIEF UPDATES FROM OTHER AGENTS =====
        # PHASE_UPDATE from Commander: used to apply conservative TTI margin
        # when evacuation is already underway (Phase ≥ 2).
        self.current_phase: int = 0

        # SUPPRESSION_UPDATE from Firefighters: cells recently extinguished;
        # fire_reports for these cells are dropped before TTI computation.
        self.suppressed_cells: set = set()

        # ===== STEP-AHEAD FIRE PREDICTOR =====
        self.step_ahead_predictor = None
        if _FIRE_PREDICTOR_AVAILABLE:
            try:
                self.step_ahead_predictor = _StepAheadPredictor()
            except Exception:
                self.step_ahead_predictor = None

        # ===== FUZZY LOGIC SYSTEM =====
        # Initialize fuzzy inference system for risk assessment
        # Inputs: TTI, escape route availability
        # Output: Risk level (0-100)
        self._setup_fuzzy_system()

    def _setup_fuzzy_system(self):
        """
        Initialize fuzzy logic system for risk assessment.

        FUZZY LOGIC allows handling uncertainty in risk assessment by using
        linguistic variables instead of precise thresholds. This better models
        real-world decision-making where risks aren't binary.

        INPUT 1: Time To Impact (TTI)
        - imminent: Fire will arrive very soon (0-30 meters)
        - near_future: Fire approaching (30-70 meters)
        - distant: Fire is far away (70-200 meters)

        INPUT 2: Escape Route Availability
        - bottlenecked: Few exits available (0-2 exits)
        - sufficient: Adequate exits (2+ exits)

        OUTPUT: Risk Level (0-100)
        - low (0-30): Situation under control
        - medium (20-70): Moderate concern
        - high (60-90): Dangerous situation
        - critical (85-100): Immediate threat to life

        FUZZY RULES (IF-THEN logic):
        1. IF fire imminent AND routes bottlenecked THEN risk critical (MATI SCENARIO)
        2. IF fire imminent AND routes sufficient THEN risk high
        3. IF fire near future AND routes bottlenecked THEN risk high
        4. IF fire near future AND routes sufficient THEN risk medium
        5. IF fire distant THEN risk low
        6. IF fire imminent THEN risk high (failsafe)

        Reference: Rules based on analysis of Mati Fire disaster (2018) where
        bottlenecked exits + approaching fire = critical situation
        """
        # ===== INPUT VARIABLE 1: TIME TO IMPACT (TTI) =====
        # Distance until fire reaches civilian population (meters)
        self.tti = ctrl.Antecedent(np.arange(0, 201, 1), 'tti')
        self.tti['imminent'] = fuzz.trimf(self.tti.universe, [0, 0, ANALYST_TTI_IMMINENT])
        self.tti['near_future'] = fuzz.trimf(self.tti.universe,
                                             [ANALYST_TTI_IMMINENT, ANALYST_TTI_NEAR, 150])
        self.tti['distant'] = fuzz.trimf(self.tti.universe, [ANALYST_TTI_NEAR, 200, 200])

        # ===== INPUT VARIABLE 2: ESCAPE ROUTE AVAILABILITY =====
        # Number of available evacuation exits
        self.routes = ctrl.Antecedent(np.arange(0, 11, 1), 'routes')
        self.routes['bottlenecked'] = fuzz.trimf(self.routes.universe,
                                                 [0, 0, ANALYST_EXIT_BOTTLENECK_THRESHOLD])
        self.routes['sufficient'] = fuzz.trimf(self.routes.universe,
                                               [ANALYST_EXIT_BOTTLENECK_THRESHOLD, 10, 10])

        # ===== OUTPUT VARIABLE: RISK LEVEL =====
        # Overall risk assessment (0-100 scale)
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')
        self.risk['low'] = fuzz.trimf(self.risk.universe, [0, 0, 30])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [20, 50, 70])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [60, 80, 90])
        self.risk['critical'] = fuzz.trimf(self.risk.universe, [85, 100, 100])

        # ===== FUZZY RULES (Expert Knowledge) =====
        # These rules encode disaster management expertise
        rule1 = ctrl.Rule(self.tti['imminent'] & self.routes['bottlenecked'],
                         self.risk['critical'])  # MATI FIRE SCENARIO: worst case
        rule2 = ctrl.Rule(self.tti['imminent'] & self.routes['sufficient'],
                         self.risk['high'])  # Imminent but can evacuate
        rule3 = ctrl.Rule(self.tti['near_future'] & self.routes['bottlenecked'],
                         self.risk['high'])  # Time but bottlenecked
        rule4 = ctrl.Rule(self.tti['near_future'] & self.routes['sufficient'],
                         self.risk['medium'])  # Manageable situation
        rule5 = ctrl.Rule(self.tti['distant'], self.risk['low'])  # Fire far away
        rule6 = ctrl.Rule(self.tti['imminent'], self.risk['high'])  # Failsafe: imminent = high risk

        # ===== CONTROL SYSTEM =====
        # Combines all rules using fuzzy inference
        self.risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
        self.risk_simulation = ctrl.ControlSystemSimulation(self.risk_ctrl)

    def _calculate_ros(self, slope: float, wind_speed: float) -> float:
        """
        Calculate Rate of Spread using the simplified Rothermel (1972) equation.

        ROS = R_base × (1 + φ_wind) × (1 + φ_slope)

        Wind factor:  φ_wind  = C × U^B
        Slope factor: φ_slope = 5.275 × tan²(φ)

        Where:
          R_base — base ROS for the fuel type (m/s)
          U      — wind speed (m/s)
          φ      — slope angle (radians, converted from % grade)
          C, B   — empirical wind coefficients

        Ref: Rothermel, R.C. (1972). USDA Forest Service Research Paper INT-RP-115.

        Returns: Rate of spread in m/s
        """
        # Slope factor: phi_slope = 5.275 * (tan(slope))^2
        slope_radians = np.arctan(slope / 100.0)  # Convert percentage to radians
        phi_slope = 5.275 * (np.tan(slope_radians) ** 2)

        # Wind factor: phi_wind = C * U^B
        phi_wind = ROTHERMEL_WIND_C * (wind_speed ** ROTHERMEL_WIND_B)

        # Combined ROS
        ros = ROTHERMEL_BASE_ROS * (1 + phi_wind) * (1 + phi_slope)

        return ros

    def _calculate_tti(self, distance_to_settlement: float, ros: float) -> float:
        """
        Calculate Time To Impact.
        TTI = Distance / ROS

        Returns: Time in seconds until fire reaches settlement
        """
        if ros <= 0:
            return float('inf')

        # TTI = distance / rate_of_spread (meters / meters_per_second = seconds)
        return distance_to_settlement / ros

    def perceive(self, environment) -> None:
        """
        Collect fire detection reports from Sentinels and belief updates
        from Commander (PHASE_UPDATE) and Firefighters (SUPPRESSION_UPDATE).
        """
        self._environment = environment
        for message in self.messages_inbox:
            msg_type = message.content.get('type') if message.content else None

            # ── Raw sensor data from Sentinels ──────────────────────────────
            if message.performative == "INFORM" and msg_type == 'FIRE_DETECTION':
                self.fire_reports.append(message.content)

            # ── Belief update: Commander phase transition ────────────────────
            # When evacuation is underway (Phase ≥ 2) we apply a conservative
            # TTI multiplier so the Commander isn't falsely reassured.
            elif message.performative == "INFORM" and msg_type == 'PHASE_UPDATE':
                self.current_phase = int(message.content.get('phase', 0))

            # ── Belief update: Firefighter suppression success ───────────────
            # Remove extinguished cells from active fire reports so TTI is not
            # inflated by fires that are already out.
            elif message.performative == "INFORM" and msg_type == 'SUPPRESSION_UPDATE':
                row = message.content.get('row')
                col = message.content.get('col')
                if row is not None and col is not None:
                    self.suppressed_cells.add((int(row), int(col)))

    def decide(self) -> None:
        """
        Analyze fire reports using Rothermel ROS model and calculate TTI.
        Apply fuzzy logic for final risk assessment.
        """
        environment = self._environment
        if environment is None:
            return
        self.risk_assessments.clear()

        # Drop reports for cells already extinguished by Firefighters.
        # Belief revision: suppressed_cells is populated by SUPPRESSION_UPDATE.
        if self.suppressed_cells:
            self.fire_reports = [
                r for r in self.fire_reports
                if environment.latlon_to_grid(r['lat'], r['lon'])
                not in self.suppressed_cells
            ]
            self.suppressed_cells.clear()

        if not self.fire_reports:
            self.tti_value = float('inf')
            self.ros_value = 0.0
            return

        # Find closest fire to the centroid of active civilians.
        # Using the map centre was inaccurate when civilians cluster away from it
        # (e.g., on a road network island).  Centroid gives a TTI that reflects
        # actual proximity of fire to the people who still need to evacuate.
        agents_ref = getattr(environment, 'agents', None)
        active_civs = []
        if agents_ref and 'civilians' in agents_ref:
            active_civs = [c for c in agents_ref['civilians']
                           if c.is_active and c.position is not None]
        if active_civs:
            center_lat = float(np.mean([c.position[0] for c in active_civs]))
            center_lon = float(np.mean([c.position[1] for c in active_civs]))
        else:
            center_lat = getattr(environment, 'lat_center', self.position[0])
            center_lon = getattr(environment, 'lon_center', self.position[1])

        min_distance = float('inf')
        closest_fire = None

        for report in self.fire_reports:
            lat, lon = report['lat'], report['lon']
            dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 111320  # metres

            if dist < min_distance:
                min_distance = dist
                closest_fire = report

        if closest_fire is None:
            self.tti_value = float('inf')
            self.ros_value = 0.0
            self.fire_reports.clear()
            return

        # Calculate slope from elevation grid using 3×3 neighbourhood around fire
        fire_lat, fire_lon = closest_fire['lat'], closest_fire['lon']
        fire_row, fire_col = environment.latlon_to_grid(fire_lat, fire_lon)
        r0 = max(0, fire_row - 1)
        r1 = min(environment.elevation_grid.shape[0] - 1, fire_row + 1)
        c0 = max(0, fire_col - 1)
        c1 = min(environment.elevation_grid.shape[1] - 1, fire_col + 1)
        patch = environment.elevation_grid[r0:r1+1, c0:c1+1]
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            radius = getattr(environment, 'radius', 2000.0)
            cell_h = (2.0 * radius) / environment.elevation_grid.shape[0]
            cell_w = (2.0 * radius) / environment.elevation_grid.shape[1]
            dz_row = float(np.mean(np.abs(np.diff(patch, axis=0)))) / cell_h
            dz_col = float(np.mean(np.abs(np.diff(patch, axis=1)))) / cell_w
            # Cap at 100% slope (45°) — Perlin noise can produce unrealistic spikes
            slope_percentage = min((dz_row + dz_col) * 100.0, 100.0)
        else:
            slope_percentage = 0.0

        # Calculate ROS using Rothermel model
        self.ros_value = self._calculate_ros(slope_percentage, WIND_SPEED)

        # Calculate TTI (Time To Impact) using Rothermel as baseline
        self.tti_value = self._calculate_tti(min_distance, self.ros_value)

        # BDI intention revision: when Commander has already ordered evacuation
        # (Phase ≥ 2) apply a 20% conservative reduction to TTI so we do not
        # falsely reassure the Commander while civilians are still moving.
        if self.current_phase >= 2 and self.tti_value < float('inf'):
            self.tti_value *= 0.80

        # Override TTI with step-ahead predictor if trained
        if (self.step_ahead_predictor is not None and
                self.step_ahead_predictor.is_trained and
                hasattr(environment, 'fire_grid') and
                hasattr(environment, 'elevation_grid') and
                hasattr(environment, 'fuel_type_grid')):
            try:
                # Compute slope grid
                grad_y, grad_x = np.gradient(environment.elevation_grid)
                slope_grid = np.sqrt(grad_y**2 + grad_x**2).astype(np.float32)

                # Recompute current dynamic wind direction (mirrors FireSimulation)
                step = environment.step_count
                if step == 0:
                    _wd = WIND_INITIAL_DIRECTION
                else:
                    _wd = WIND_INITIAL_DIRECTION + (
                        np.sin(step / WIND_OSCILLATION_PERIOD) * WIND_OSCILLATION_AMPLITUDE
                    )
                _wr = np.radians(_wd)
                wind_vec = np.array([np.sin(_wr), -np.cos(_wr)], dtype=np.float32)
                _wm = np.linalg.norm(wind_vec)
                if _wm > 1e-10:
                    wind_vec = wind_vec / _wm

                prob_grid = self.step_ahead_predictor.predict(
                    fire_grid=environment.fire_grid,
                    wind_vec=wind_vec,
                    slope_grid=slope_grid,
                    fuel_grid=environment.fuel_type_grid,
                    humidity=getattr(environment, 'humidity', 30.0),
                )

                # Find analyst grid position (proxy for civilian settlement)
                analyst_row, analyst_col = environment.latlon_to_grid(
                    self.position[0], self.position[1]
                )

                # Find cells with high fire probability (>0.5) that are currently fuel
                high_risk = (prob_grid > 0.5) & (environment.fire_grid == 3)
                if np.any(high_risk):
                    risk_positions = np.argwhere(high_risk)
                    distances = np.sqrt(
                        (risk_positions[:, 0] - analyst_row)**2 +
                        (risk_positions[:, 1] - analyst_col)**2
                    )
                    min_risk_dist_cells = float(distances.min())

                    # Convert cells to metres using environment radius
                    radius = getattr(environment, 'radius', 2000.0)
                    grid_size = max(environment.grid_shape)
                    metres_per_cell = (2.0 * radius) / grid_size
                    min_risk_dist_m = min_risk_dist_cells * metres_per_cell

                    # Recompute TTI from distance to high-risk cell
                    tti_predictor = self._calculate_tti(min_risk_dist_m, self.ros_value)
                    # Use the more conservative (smaller) TTI
                    if tti_predictor < self.tti_value:
                        self.tti_value = tti_predictor
            except Exception:
                pass  # Fallback to Rothermel TTI

        # Get number of exit routes from environment
        num_exits = getattr(environment, 'num_exits', 3)  # Fallback to 3 if not available

        # Run fuzzy inference
        try:
            tti_clamped = max(0, min(200, self.tti_value))
            exits_clamped = max(0, min(10, num_exits))

            self.risk_simulation.input['tti'] = tti_clamped
            self.risk_simulation.input['routes'] = exits_clamped

            self.risk_simulation.compute()
            risk_level = self.risk_simulation.output['risk']

            # Store assessment
            lat, lon = closest_fire['lat'], closest_fire['lon']
            self.risk_assessments[(lat, lon)] = risk_level

        except Exception as e:
            # Fallback to TTI-based risk
            if self.tti_value < ANALYST_TTI_IMMINENT:
                risk_level = 90.0
            elif self.tti_value < ANALYST_TTI_NEAR:
                risk_level = 60.0
            else:
                risk_level = 30.0

            lat, lon = closest_fire['lat'], closest_fire['lon']
            self.risk_assessments[(lat, lon)] = risk_level

        # Clear old reports
        self.fire_reports.clear()

    def act(self, environment) -> None:
        """Send risk reports to Commander with TTI and ROS data"""
        if self.risk_assessments:
            # Calculate overall risk (max risk in area)
            max_risk = max(self.risk_assessments.values())
            avg_risk = np.mean(list(self.risk_assessments.values()))

            # Get actual number of exits from environment
            num_exits = environment.num_exits

            message = Message(
                sender=self.agent_id,
                receiver="commander",
                performative="INFORM",
                content={
                    'type': 'RISK_REPORT',
                    'max_risk': max_risk,
                    'avg_risk': avg_risk,
                    'tti': self.tti_value / 60.0,
                    'ros': self.ros_value,
                    'num_exits': num_exits,
                    'fire_locations': list(self.risk_assessments.keys()),
                    'timestamp': environment.step_count
                }
            )
            self.send_message(message)
