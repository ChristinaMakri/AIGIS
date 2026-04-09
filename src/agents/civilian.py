"""
Civilian Agent - BDI (Belief-Desire-Intention) Architecture
Implements LWR triangular fundamental diagram + inverse-distance-weighted crowd following (herding)
Features 3-state cognitive machine: Rational, Confused, Herding
Panic equation with fire distance and family separation factors

BDI architecture:
  Rao, A.S. & Georgeff, M.P. (1995).
  "BDI agents: From theory to practice."
  Proceedings of ICMAS-95, pp. 312–319. AAAI Press.

LWR macroscopic traffic-flow model (triangular fundamental diagram):
  Lighthill, M.J. & Whitham, G.B. (1955).
  "On kinematic waves II: A theory of traffic flow on long crowded roads."
  Proceedings of the Royal Society A, 229(1178), pp. 317–345.

  Richards, P.I. (1956).
  "Shock waves on the highway."
  Operations Research, 4(1), pp. 42–51.

  Triangular diagram calibration for pedestrian evacuation flow:
  Daamen, W. & Hoogendoorn, S.P. (2003).
  "Experimental research of pedestrian walking behavior."
  Transportation Research Record, 1828, pp. 20–30.

  Free-flow:  ρ ≤ ρ_crit  →  v = v_free
  Congested:  ρ > ρ_crit  →  v = w × (ρ_jam / ρ − 1)
  where w = v_free × ρ_crit / (ρ_jam − ρ_crit)  [backward wave speed]

Evacuation micro-simulation in the wildland–urban interface:
  Cova, T.J. & Johnson, J.P. (2002).
  "Microsimulation of neighborhood evacuations in the urban-wildland interface."
  Environment and Planning A, 34(12), pp. 2211–2230.
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Set
from .base_agent import Agent
from ..message import Message
from ..config import (
    CIVILIAN_V_FREE_FLOW,
    CIVILIAN_RHO_JAM,
    CIVILIAN_PANIC_RATIONAL,
    CIVILIAN_PANIC_CONFUSED,
    CIVILIAN_PANIC_ALPHA,
    CIVILIAN_PANIC_BETA,
    CIVILIAN_PANIC_DECAY,
    CIVILIAN_CONFUSED_SPEED_FACTOR,
    CIVILIAN_VISION_RADIUS,
    CIVILIAN_HERDING_INFLUENCE,
    CIVILIAN_PATH_RECALC_INTERVAL,
    AQI_PANIC_WEIGHT,
    AQI_SPEED_PENALTY,
    CIVILIAN_INJURY_THRESHOLD,
    CIVILIAN_SMOKE_PANIC_SCALE,
)
# Ablation B flag — read from module at call time so patching cfg.DISABLE_PANIC
# in run_ablation.py takes effect without reloading the module.
from .. import config as _cfg_module


class CivilianAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Civilian
    ARCHITECTURE: BDI — Three-State Cognitive Machine with Crowd Dynamics
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • beliefs             set of facts: warning_received, fire_nearby,
                            evacuation_ordered, shelter_in_place,
                            fwi_warning_received
      • panic_level         0.0 (calm) – 1.0 (extreme panic)
      • cognitive_state     rational | confused | herding
      • fire_visible        fire within vision radius
      • fire_distance       distance to nearest visible fire (grid cells)
      • current_speed       actual movement speed (LWR triangular diagram)
      • current_aqi         air quality index (smoke effects on speed/panic)
      • is_evacuated        has reached a safe zone
      • has_family          30% of civilians have family (psychological factor)
      • family_separated    family-separation stress amplifier

    DESIRES
      • Survive by reaching a safe zone
      • Maintain rational decision-making under stress
      • Keep family together if applicable

    INTENTIONS  (selected by cognitive state)
      rational:   A* evacuation to safety_node (optimal path)
      confused:   slow A* evacuation (degraded speed × CONFUSED_SPEED_FACTOR)
      herding:    follow nearest crowd via inverse-distance-weighted crowd following (simplified social force)
      freeze:     panic freeze — do nothing this step
      move_random: random panic movement (herding without leader)

    COMMUNICATION
      SENDS
        → INFORM  ambulances  {type:'INJURY_REPORT', agent_id, node, lat, lon}
              sent once when smoke_exposure exceeds CIVILIAN_INJURY_THRESHOLD;
              triggers Ambulance self-dispatch to the civilian's location.
      RECEIVES
        ← INFORM   commander  {type:'WARNING', urgency:'MEDIUM'}
              pre-alert: prepare to evacuate
        ← INFORM   commander  {type:'FWI_WARNING', urgency, fwi,
                               high_risk_zones}
              pre-fire weather warning
        ← REQUEST  commander  {type:'EVACUATE', urgency:'HIGH'}
              mass evacuation order (Phase 2)
        ← REQUEST  commander  {type:'REDIRECT_TO_SAFE_ZONE',
                               urgency:'CRITICAL'}
              shelter-in-place order (Phase 3)

    BIBLIOGRAPHY
      [1] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          BDI belief-revision and intention-selection cycle.
      [2] Lighthill, M.J. & Whitham, G.B. (1955). "On kinematic waves II."
          Proc. R. Soc. A, 229(1178), pp. 317–345.  LWR triangular diagram:
          free-flow v=v_free (ρ≤ρ_crit); congested v=w(ρ_jam/ρ−1) (ρ>ρ_crit).
          Calibration: Daamen & Hoogendoorn (2003) Transp. Res. Rec. 1828:20-30.
      [3] Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of
          neighborhood evacuations in the urban-wildland interface."
          Environment and Planning A, 34(12), pp. 2211–2229.
          Evacuation route choice and contraflow modelling.
      [4] Inness, A. et al. (2019). "The CAMS reanalysis of atmospheric
          composition." Atmos. Chem. Phys., 19(6), pp. 3515–3556.
          AQI data source (PM2.5 → panic/speed penalty); also grounds the
          cumulative smoke_exposure injury model (smoke_exposure → is_injured).
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Initialize a civilian agent with BDI architecture and panic psychology.

        Key Features:
        - LWR triangular traffic model: free-flow until ρ_crit, then congested
        - 3-state cognitive machine: Rational → Confused → Herding
        - Panic equation: Increases with fire proximity and family separation
        - Social force herding: Follows crowd at high panic levels

        Args:
            agent_id: Unique identifier for the agent
            position: Initial (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # ===== LWR TRIANGULAR FUNDAMENTAL DIAGRAM PARAMETERS =====
        # Free-flow  (ρ ≤ ρ_crit):  v = v_free
        # Congested  (ρ > ρ_crit):  v = w × (ρ_jam / ρ − 1)
        # where w = v_free × ρ_crit / (ρ_jam − ρ_crit)  [backward wave speed]
        # ρ_crit = ρ_jam / 4  (Daamen & Hoogendoorn 2003)
        self.v_free_flow = CIVILIAN_V_FREE_FLOW  # Maximum speed when road is empty
        self.rho_jam = CIVILIAN_RHO_JAM  # Jam density (agents per edge)
        self.current_speed = self.v_free_flow  # Current actual speed

        # ===== BDI ARCHITECTURE COMPONENTS =====
        # Beliefs: Knowledge about the world (updated via perception)
        # Desires: Goals the agent wants to achieve (fixed: survive)
        # Intentions: Committed plans to achieve desires (current path)
        self.beliefs: Set[str] = set()  # Known facts ('warning_received', 'fire_nearby', etc.)
        self.intentions: List[str] = []  # Current active plan

        # ===== PANIC MODEL (Distance-based with Family Factor) =====
        # Panic(t) = Panic(t-1) + α×(1/d_fire) + β×(family_separated) - decay
        # Drives the 3-state cognitive machine: rational → confused → herding
        self.panic_level = 0.0  # Range: 0.0 (calm) to 1.0 (extreme panic)
        self.fire_visible = False  # Whether agent can see fire within vision radius
        self.fire_distance = float('inf')  # Distance to nearest visible fire (meters)
        self.cognitive_state = "rational"  # Current cognitive state: rational|confused|herding

        # Instance-level panic thresholds (can be overridden by ParameterAdapter)
        self.panic_rational_threshold = CIVILIAN_PANIC_RATIONAL
        self.panic_confused_threshold = CIVILIAN_PANIC_CONFUSED

        # ===== NAVIGATION STATE =====
        # Uses A* pathfinding on OpenStreetMap road network
        self.current_node: Optional[int] = None  # Current graph node
        self.safety_node: Optional[int] = None  # Target safe zone node
        self.current_path: List[int] = []  # Planned path (list of node IDs)
        self.current_edge_density = 0.0  # Local agent density on current edge (for LWR model)
        self.evacuation_ordered = False  # Commander ordered evacuation
        self.redirect_to_coast = False  # Shelter-in-place order (Phase 3)
        self.is_evacuated = False  # True once civilian reaches a safe zone

        # ===== PRE-EVACUATION MILLING DELAY (Lindell & Perry 2012) =====
        # Households spend time preparing before departing after an official
        # evacuation order — packing, contacting family, seeking confirmation.
        # This is called "milling" in the Protective Action Decision Model.
        #
        # Lindell, M.K. & Perry, R.W. (2012). "The Protective Action Decision
        # Model: Theoretical modifications and additional evidence." Risk
        # Analysis, 32(4), pp. 616-632.  DOI: 10.1111/j.1539-6924.2011.01647.x
        # Table 3 — "Warning-issued to departure time" for commanded evacuation
        # with official notification (siren/broadcast):
        #   Median: 15.2 min  |  Range: 7-45 min
        # Log-normal fit: μ = ln(182), σ = 0.60
        #   (at 5 s/step → 15.2 min = 912 s / 5 = ~182 steps)
        #   σ derived from 10th-90th percentile ratio ≈ e^(2×1.28×σ) ≈ 6
        #   → σ ≈ 0.60  (Lindell & Perry 2012, Table 3 spread)
        #
        # Milling does NOT apply when fire is directly visible — immediate
        # flight response overrides deliberation (Lindell & Perry 2012, p. 622).
        self._milling_steps_remaining: int = 0
        self._milling_delay_set: bool = False   # sample only once per warning

        # ===== SOCIAL BONDS (Psychology Factor) =====
        # 30% of civilians have family, adding panic when separated
        # Based on real disaster psychology: family separation increases irrational behavior
        self.has_family = np.random.random() < 0.3
        self.family_separated = self.has_family  # Assume separated at start

        # ===== SOCIAL FORCE MODEL (Herding Behavior) =====
        # At high panic (>0.7), agent follows crowd instead of optimal path
        # Can lead to stampedes and movement toward dead ends (realistic tragedy scenario)
        self.vision_radius = CIVILIAN_VISION_RADIUS  # Grid cells for seeing other agents
        self.nearby_agents = []  # Nearby civilians to follow when herding
        self.last_movement = None  # Direction vector for others to follow

        # ===== PERFORMANCE OPTIMIZATION: STAGGERED PATHFINDING =====
        self.path_recalc_interval = CIVILIAN_PATH_RECALC_INTERVAL
        self.steps_since_recalc = 0
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)

        # ===== AIR QUALITY (smoke effects) =====
        # Current AQI from environment (0-500 scale).
        # High smoke raises panic and reduces movement speed.
        self.current_aqi: float = 0.0

        # ===== SMOKE INJURY MODEL (Inness et al. 2019 + Cova & Johnson 2002) =====
        # Cumulative smoke exposure → injury incapacitation.
        # Each step: smoke_exposure += smoke_grid[r,c]
        # When exposure > CIVILIAN_INJURY_THRESHOLD → is_injured = True.
        # Injured civilians cannot move and send an INJURY_REPORT to ambulances.
        self.smoke_exposure: float = 0.0
        self.is_injured: bool = False
        self.injury_threshold: float = CIVILIAN_INJURY_THRESHOLD
        self._injury_reported: bool = False  # Ensure we send only one INJURY_REPORT

        # ===== PATHFINDING FALLBACK =====
        # Counts consecutive A* failures.  After 3 failures the civilian
        # switches to direct grid-space movement toward the nearest map edge
        # (which is always a safe perimeter cell) — handles disconnected
        # road-network islands in large-radius maps.
        self._path_fail_count: int = 0

        # ===== GRIDLOCK FALLBACK =====
        # Counts consecutive steps where speed <= 0.1 (full gridlock).
        # After 3 steps of gridlock the civilian bypasses the speed gate
        # and uses grid-space perimeter movement to break out of the jam.
        self._gridlock_steps: int = 0
        # Track last known position to detect zero-progress (obstacle-enclosed).
        # If position doesn't change for 30 steps of perimeter fallback, the
        # civilian is physically trapped and is marked as a trapped casualty.
        self._last_grid_position: Optional[Tuple[int, int]] = None
        self._no_progress_steps: int = 0

    def perceive(self, environment) -> None:
        """
        BDI Perception: Update beliefs based on environment and messages.

        Perception includes:
        1. Processing Commander messages (warnings, evacuation orders)
        2. Scanning for visible fire within vision radius
        3. Calculating distance-based panic with family separation factor
        4. Updating cognitive state based on panic thresholds

        Panic Equation:
        Panic(t) = Panic(t-1) + α×(1/d_fire) + β×(family_separated) - decay
        Where:
        - α: Fire distance coefficient (closer fire = higher panic increase)
        - β: Family separation penalty (adds constant stress)
        - decay: Gradual reduction when no fire visible
        """
        # ===== READ AIR QUALITY FROM ENVIRONMENT =====
        self.current_aqi = float(getattr(environment, 'air_quality_index', 0.0))

        # ===== SMOKE EXPOSURE ACCUMULATION (Inness et al. 2019) =====
        # Cumulative PM2.5 proxy: read per-cell smoke concentration from grid.
        # Extra panic from local smoke density (beyond AQI baseline).
        if self.grid_position is not None and not self.is_injured:
            r, c = self.grid_position
            smoke_conc = float(getattr(environment, 'smoke_grid',
                                        np.zeros(environment.grid_shape))[r, c])
            self.smoke_exposure += smoke_conc
            # Smoke also amplifies panic
            self.panic_level = min(1.0,
                                   self.panic_level + smoke_conc * CIVILIAN_SMOKE_PANIC_SCALE)
            if self.smoke_exposure >= self.injury_threshold:
                self.is_injured = True
                print(f"  [{self.agent_id}]: smoke-injured "
                      f"(exposure={self.smoke_exposure:.1f})")

        # ===== PROCESS MESSAGES (Commander → Civilian Communication) =====
        for message in self.messages_inbox:
            # Pre-Evacuation Warning (Phase 1): Commander alerts of approaching fire
            if message.performative == "INFORM" and message.content.get('type') == 'WARNING':
                self.beliefs.add('warning_received')
                self.panic_level = min(1.0, self.panic_level + 0.1)

            # FWI pre-fire warning: fire conditions dangerous even before ignition
            elif message.performative == "INFORM" and message.content.get('type') == 'FWI_WARNING':
                urgency = message.content.get('urgency', 'MEDIUM')
                self.beliefs.add('fwi_warning_received')
                delta = 0.15 if urgency == 'HIGH' else 0.05
                self.panic_level = min(1.0, self.panic_level + delta)

            # Evacuation Orders (Phase 2): Commander orders mass evacuation
            elif message.performative == "REQUEST":
                msg_type = message.content.get('type')
                if msg_type == 'EVACUATE':
                    self.beliefs.add('evacuation_ordered')
                    self.evacuation_ordered = True
                    self.panic_level = min(1.0, self.panic_level + 0.3)  # Significant panic increase
                    self.family_separated = False  # Mass evacuation: family assumed moving together
                    # Sample milling delay once per warning — only if fire not
                    # yet visible (visible fire triggers immediate flight response,
                    # Lindell & Perry 2012, p. 622 "threat recognition override").
                    if not self.fire_visible and not self._milling_delay_set:
                        # Log-normal: μ=ln(182), σ=0.60 (Lindell & Perry 2012,
                        # Table 3; 182 steps ≈ 15.2 min at 5 s/step)
                        self._milling_steps_remaining = max(
                            0, int(np.random.lognormal(mean=5.204, sigma=0.60))
                        )
                        self._milling_delay_set = True

                # Shelter-in-Place (Phase 3): Too late to evacuate, seek nearest safe zone
                elif msg_type == 'REDIRECT_TO_SAFE_ZONE':
                    self.beliefs.add('shelter_in_place')
                    self.redirect_to_coast = True  # Flag to re-route to nearest safe zone
                    self.panic_level = min(1.0, self.panic_level + 0.5)  # High panic!
                    self.family_separated = False  # Emergency override: family concern drops

        # Check for visible fire and calculate distance
        self._check_fire_visibility_and_distance(environment)

        # Update panic using the panic equation
        self._update_panic()

        # Determine cognitive state based on panic level
        self._update_cognitive_state()

        # Assess local traffic density (LWR model)
        self._assess_local_density(environment)

        # Note: nearby_agents will be populated by simulation
        # (needs access to all civilian agents)

    def _check_fire_visibility_and_distance(self, environment) -> None:
        """
        Scan environment for visible fire and calculate distance to nearest burning cell.

        Used in panic equation: closer fire → higher panic increase.
        Vision limited to CIVILIAN_VISION_RADIUS to simulate realistic line-of-sight.

        Note: This is a simplified perception model. Real-world perception would include:
        - Smoke obscuring vision
        - Buildings blocking line of sight
        - Hearing/smelling fire beyond visual range
        """
        # Reset fire perception
        self.fire_visible = False
        self.fire_distance = float('inf')
        self.beliefs.discard('fire_nearby')

        # Safety check: agent not yet placed on grid
        if self.grid_position is None:
            return

        row, col = self.grid_position
        fire_grid = environment.fire_grid

        # Scan square area within vision radius
        # PERFORMANCE NOTE: This is O(R²) per agent, but R is small (10 cells)
        # For large-scale simulations, consider spatial indexing
        for dr in range(-self.vision_radius, self.vision_radius + 1):
            for dc in range(-self.vision_radius, self.vision_radius + 1):
                r, c = row + dr, col + dc

                # Check grid bounds
                if 0 <= r < fire_grid.shape[0] and 0 <= c < fire_grid.shape[1]:
                    # Check if cell is burning (state = 1)
                    if fire_grid[r, c] == 1:
                        self.fire_visible = True
                        self.beliefs.add('fire_nearby')

                        # Calculate Euclidean distance in grid cells
                        dist = np.sqrt(dr**2 + dc**2)

                        # Track nearest fire
                        if dist < self.fire_distance:
                            self.fire_distance = dist

    def _update_panic(self) -> None:
        """
        Update panic level using distance-based panic equation.

        Panic Equation:
        Panic(t) = Panic(t-1) + α × (1/d_fire) + β × (family_separated) - decay

        When DISABLE_PANIC=True (Ablation B), this method is a no-op: panic
        stays at 0 and the cognitive state remains rational throughout.
        Comparing results with/without this model quantifies the contribution
        of the three-state cognitive machine to evacuation outcomes:
          Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood
          evacuations in the urban-wildland interface." Environment and
          Planning A, 34(12), pp. 2211–2230.
        """
        # Ablation B: rational-agent baseline — skip panic dynamics entirely
        # Read at call time so cfg.DISABLE_PANIC patch in run_ablation.py works.
        if _cfg_module.DISABLE_PANIC:
            self.panic_level = 0.0
            return

        if self.fire_visible and self.fire_distance < float('inf'):
            # ===== FIRE PROXIMITY FACTOR =====
            panic_increase = CIVILIAN_PANIC_ALPHA * (1.0 / max(self.fire_distance, 0.5))
            self.panic_level = min(1.0, self.panic_level + panic_increase)
        else:
            # ===== PANIC DECAY =====
            self.panic_level = max(0.0, self.panic_level - CIVILIAN_PANIC_DECAY)

        # ===== AIR QUALITY / SMOKE FACTOR =====
        # Smoke raises panic even without visible fire (smell, reduced visibility,
        # breathing difficulty). Contribution is proportional to AQI (0–500 scale).
        if self.current_aqi > 50.0:
            aqi_panic = AQI_PANIC_WEIGHT * (self.current_aqi / 500.0)
            self.panic_level = min(1.0, self.panic_level + aqi_panic)

        # ===== FAMILY SEPARATION FACTOR =====
        # If agent has family and they are separated, add constant stress
        # This simulates psychological burden of not knowing if family is safe
        # Reference: Family separation is a major stressor in disaster evacuation
        if self.family_separated:
            self.panic_level = min(1.0, self.panic_level + CIVILIAN_PANIC_BETA * 0.1)

    def _update_cognitive_state(self) -> None:
        """
        3-State Cognitive Machine: Maps panic level to decision-making quality.

        STATE 1: RATIONAL (Panic < 0.4)
        - Optimal decision making
        - Uses A* pathfinding to nearest safe zone
        - Considers traffic conditions (LWR model)
        - Full speed when road is clear

        STATE 2: CONFUSED (0.4 ≤ Panic < 0.7)
        - Degraded decision making
        - 50% speed reduction (hesitation, indecision)
        - Frequent path recalculation (second-guessing)
        - May change direction erratically

        STATE 3: HERDING (Panic ≥ 0.7)
        - Abandons rational planning
        - Follows nearby agents using Social Force Model
        - Can lead to stampedes and dead-end traps
        - Realistic tragedy scenario (e.g., Mati Fire 2018)

        Reference: Disaster psychology shows panic impairs decision quality
        progressively. At extreme panic, individuals follow the crowd even
        when it leads to danger (documented in multiple disasters).
        """
        if self.panic_level < self.panic_rational_threshold:
            self.cognitive_state = "rational"  # Below threshold: optimal behavior
        elif self.panic_level < self.panic_confused_threshold:
            self.cognitive_state = "confused"  # Middle range: degraded performance
        else:
            self.cognitive_state = "herding"  # Above threshold: follows crowd

    def _assess_local_density(self, environment) -> None:
        """
        Assess local agent density for LWR traffic model.
        Counts nearby agents within a 2-cell radius to simulate edge density.
        Uses nearby_agents from the previous step (populated by simulation after update).
        """
        if self.grid_position is None or not self.nearby_agents:
            self.current_edge_density = 0.0
            return

        my_row, my_col = self.grid_position
        count = 0
        for agent in self.nearby_agents:
            if agent.grid_position is not None:
                dist = np.sqrt(
                    (agent.grid_position[0] - my_row) ** 2 +
                    (agent.grid_position[1] - my_col) ** 2
                )
                if dist <= 2.0:  # immediate neighbourhood ≈ one road edge
                    count += 1

        self.current_edge_density = min(float(count), self.rho_jam)

    def _find_nearby_agents(self, all_agents: list) -> None:
        """
        Find nearby agents for Social Force herding behavior.
        Scans within vision radius and stores agents moving away from fire.

        Optimization: Pre-filters using bounding box before calculating
        exact distance. Reduces unnecessary sqrt calculations by ~40%.
        """
        self.nearby_agents = []

        if self.grid_position is None:
            return

        my_row, my_col = self.grid_position

        # Scan civilians with bounding box pre-filter for performance
        for agent in all_agents:
            if agent.agent_id == self.agent_id:
                continue

            if not hasattr(agent, 'grid_position') or agent.grid_position is None:
                continue

            other_row, other_col = agent.grid_position

            # Quick bounding box check (Manhattan distance approximation)
            # Skip agents clearly outside vision radius before expensive sqrt
            if abs(other_row - my_row) > self.vision_radius or abs(other_col - my_col) > self.vision_radius:
                continue

            # Calculate exact Euclidean distance only for candidates
            distance = np.sqrt((other_row - my_row)**2 + (other_col - my_col)**2)

            if distance <= self.vision_radius:
                self.nearby_agents.append(agent)

    def decide(self) -> None:
        """
        BDI reasoning cycle with 3-state cognitive machine.
        Cognitive state determines decision quality.
        Shelter-in-place always overrides herding — civilians must reach safe zone.
        """
        # Shelter-in-place ALWAYS overrides cognitive state.
        # _move_to_safety() detects this belief and routes to nearest safe node.
        if 'shelter_in_place' in self.beliefs:
            self.intentions = ['evacuate']
            return

        # Intention revision based on beliefs
        if 'evacuation_ordered' in self.beliefs or 'fire_nearby' in self.beliefs:
            # Pre-evacuation milling delay (Lindell & Perry 2012, Table 3).
            # Visible fire overrides delay (immediate flight response, ibid. p. 622).
            if self._milling_steps_remaining > 0 and not self.fire_visible:
                self._milling_steps_remaining -= 1
                return   # civilian is still preparing — do not start evacuation yet
            if 'evacuate' not in self.intentions:
                self.intentions.clear()
                self.intentions.append('evacuate')

        # Decision making based on cognitive state
        if self.cognitive_state == "rational":
            # State 1: Rational - Use optimal pathfinding
            if 'evacuate' not in self.intentions:
                self.intentions = ['evacuate']

        elif self.cognitive_state == "confused":
            # State 2: Confused - Speed reduced, may re-route frequently
            if 'evacuate' not in self.intentions:
                self.intentions = ['evacuate']

            # Occasionally reconsider path (hesitation)
            if np.random.random() < 0.2:
                self.current_path = []  # Force re-routing

        elif self.cognitive_state == "herding":
            # State 3: Herding - Follow the crowd, ignore optimal path
            if np.random.random() < 0.2:
                self.intentions = ['freeze']  # Panic freeze
            elif len(self.nearby_agents) > 0:
                self.intentions = ['follow_crowd']  # Herd behavior
            else:
                self.intentions = ['move_random']  # Panic movement

    def _calculate_speed_lwr(self) -> float:
        """
        Calculate current speed using the LWR triangular fundamental diagram.

        Free-flow  (ρ ≤ ρ_crit):  v = v_free
        Congested  (ρ > ρ_crit):  v = w × (ρ_jam / ρ − 1)
          where w = v_free × ρ_crit / (ρ_jam − ρ_crit)  [backward wave speed]

        ρ_crit is set to ρ_jam / 4, matching Daamen & Hoogendoorn (2003) pedestrian
        flow measurements (critical density ≈ 25 % of jam density at ~1.34 m/s).

        References
        ----------
        Lighthill & Whitham (1955) Proc. R. Soc. A 229:317-345
        Richards (1956) Oper. Res. 4:42-51
        Daamen & Hoogendoorn (2003) Transp. Res. Rec. 1828:20-30

        Extended with an AQI smoke penalty (Inness et al. 2019) and a
        cognitive-state modifier (Cova & Johnson 2002).
        """
        rho = self.current_edge_density

        if rho >= self.rho_jam:
            return 0.0  # Gridlock

        rho_crit = self.rho_jam / 4.0
        if rho <= rho_crit:
            speed = self.v_free_flow
        else:
            # Backward wave speed derived from flow continuity at critical point
            w = self.v_free_flow * rho_crit / (self.rho_jam - rho_crit)
            speed = w * (self.rho_jam / rho - 1.0)

        # ===== SMOKE / AQI SPEED PENALTY =====
        # High AQI (smoke inhalation, reduced visibility) slows movement.
        # At AQI=500 (hazardous) speed is reduced by AQI_SPEED_PENALTY (default 30%).
        if self.current_aqi > 50.0:
            smoke_factor = 1.0 - AQI_SPEED_PENALTY * min(self.current_aqi / 500.0, 1.0)
            speed *= max(smoke_factor, 0.1)  # Never reduce to absolute zero

        # Additional speed reduction in confused state
        if self.cognitive_state == "confused":
            speed *= CIVILIAN_CONFUSED_SPEED_FACTOR

        return max(0.0, speed)

    def act(self, environment) -> None:
        """
        Execute intentions using LWR triangular fundamental diagram.
        Speed depends on local density and cognitive state.
        """
        if not self.intentions:
            return

        # ===== SMOKE INJURY — send INJURY_REPORT once, then stay still =====
        # Grounded in Inness et al. (2019) — CAMS PM2.5 smoke incapacitation.
        if self.is_injured:
            if not self._injury_reported:
                self._injury_reported = True
                injury_msg = Message(
                    sender=self.agent_id,
                    receiver="ambulances",
                    performative="INFORM",
                    content={
                        'type': 'INJURY_REPORT',
                        'agent_id': self.agent_id,
                        'node': self.current_node,
                        'lat': self.position[0] if self.position else None,
                        'lon': self.position[1] if self.position else None,
                    }
                )
                self.send_message(injury_msg)
            return  # Injured civilians cannot move — await ambulance

        # Calculate current speed based on traffic density (LWR triangular diagram)
        self.current_speed = self._calculate_speed_lwr()

        primary_intention = self.intentions[0]

        if primary_intention == 'freeze':
            # Panic freeze - do nothing
            return

        elif primary_intention == 'move_random':
            # Random panic movement
            if self.current_speed > 0.5:  # Only move if some speed available
                self._move_random(environment)

        elif primary_intention == 'follow_crowd':
            # Herding behavior - move towards crowd
            self._follow_crowd(environment)

        elif primary_intention == 'evacuate':
            # Goal-directed evacuation (rational or confused)
            if self.current_speed > 0.1:
                self._gridlock_steps = 0
                self._no_progress_steps = 0
                self._last_grid_position = self.grid_position
                self._move_to_safety(environment)
            else:
                # Gridlock — count consecutive stuck steps.
                # After 3 steps bypass the speed gate and move directly toward
                # the map perimeter (breaks out of road-network jams).
                self._gridlock_steps += 1
                if self._gridlock_steps >= 3:
                    pos_before = self.grid_position
                    self._move_toward_perimeter(environment)
                    # Detect zero-progress: obstacle-enclosed pocket with no exit.
                    # After 30 consecutive steps of being completely unable to move,
                    # mark the civilian as a trapped casualty (realistic outcome).
                    if self.grid_position == pos_before:
                        self._no_progress_steps += 1
                        if self._no_progress_steps >= 30:
                            self.is_injured = True
                            self.is_active = False
                    else:
                        self._no_progress_steps = 0

    def _move_random(self, environment) -> None:
        """Move in a random direction (panic behavior) with current speed"""
        if self.grid_position is None:
            return

        row, col = self.grid_position

        # Random walk (speed affects distance)
        max_step = int(self.current_speed) if self.current_speed > 0 else 1
        dr = np.random.randint(-max_step, max_step + 1)
        dc = np.random.randint(-max_step, max_step + 1)

        new_row = np.clip(row + dr, 0, environment.grid_shape[0] - 1)
        new_col = np.clip(col + dc, 0, environment.grid_shape[1] - 1)

        # Check if not obstacle, actively burning, or burnt-out (state 2 = ash,
        # hot embers — impassable until cooled; civilians cannot route through ash).
        if (environment.obstacle_grid[new_row, new_col] == 0
                and environment.fire_grid[new_row, new_col] not in (1, 2)):
            self.grid_position = (new_row, new_col)
            self.position = environment.grid_to_latlon(new_row, new_col)

    def _move_toward_perimeter(self, environment) -> None:
        """
        Grid-space fallback for civilians on disconnected road-network islands.

        Moves one step per call toward the nearest map edge.  The simulation's
        count_evacuated() treats arrival at any perimeter cell as evacuation,
        mirroring the USE_PERIMETER_AS_SAFE setting.
        """
        if self.grid_position is None:
            return

        h, w = environment.grid_shape

        # Move up to v_free_flow cells per call so gridlocked civilians escape
        # the centre in a reasonable number of steps rather than 1 cell/step.
        steps_per_call = max(1, int(CIVILIAN_V_FREE_FLOW))
        for _ in range(steps_per_call):
            r, c = self.grid_position
            dists = {
                (-1,  0): r,
                ( 1,  0): h - 1 - r,
                ( 0, -1): c,
                ( 0,  1): w - 1 - c,
            }
            dr, dc = min(dists, key=dists.get)
            candidates = [(dr, dc), (-dc, dr), (dc, -dr), (-dr, -dc)]
            moved = False
            for step_r, step_c in candidates:
                new_r = int(np.clip(r + step_r, 0, h - 1))
                new_c = int(np.clip(c + step_c, 0, w - 1))
                if (environment.obstacle_grid[new_r, new_c] == 0
                        and environment.fire_grid[new_r, new_c] not in (1, 2)):
                    old_pos = self.grid_position
                    self.grid_position = (new_r, new_c)
                    self.position = environment.grid_to_latlon(new_r, new_c)
                    self.last_movement = np.array(
                        [new_c - old_pos[1], new_r - old_pos[0]], dtype=np.float32
                    )
                    moved = True
                    break
            if not moved:
                break  # Completely surrounded — stop early

    def _follow_crowd(self, environment) -> None:
        """
        Crowd following: move in the inverse-distance-weighted average direction
        of nearby agents. Simulates herding behaviour — following the crowd even
        to dead ends. Inspired by Helbing & Molnar (1995) social force concept
        but implemented as a simplified inverse-distance-weighted average rather
        than a full force-field formulation.
        """
        if self.grid_position is None or not self.nearby_agents:
            # Fallback to random movement if no crowd
            self._move_random(environment)
            return

        # Calculate inverse-distance-weighted average movement direction.
        # Agents closer to self have stronger influence (inverse-distance weighting).
        avg_direction = np.zeros(2, dtype=np.float32)
        total_weight = 0.0
        my_row, my_col = self.grid_position

        for agent in self.nearby_agents:
            if (hasattr(agent, 'last_movement') and agent.last_movement is not None
                    and agent.grid_position is not None):
                other_row, other_col = agent.grid_position
                dist = max(1.0, np.sqrt((other_row - my_row) ** 2 + (other_col - my_col) ** 2))
                weight = 1.0 / dist  # closer agents have more influence
                avg_direction += agent.last_movement * weight
                total_weight += weight

        if total_weight == 0 or np.linalg.norm(avg_direction) == 0:
            # No valid movement data, move randomly
            self._move_random(environment)
            return

        # Normalize average direction
        avg_direction = avg_direction / np.linalg.norm(avg_direction)

        # Apply herding influence
        movement = avg_direction * self.current_speed * CIVILIAN_HERDING_INFLUENCE

        # Update position
        row, col = self.grid_position
        new_row = int(np.clip(row + movement[1], 0, environment.grid_shape[0] - 1))
        new_col = int(np.clip(col + movement[0], 0, environment.grid_shape[1] - 1))

        # Check if valid (not obstacle or fire)
        if (environment.obstacle_grid[new_row, new_col] == 0
                and environment.fire_grid[new_row, new_col] not in (1, 2)):
            # Track last movement for other agents to follow
            self.last_movement = np.array([new_col - col, new_row - row], dtype=np.float32)
            self.grid_position = (new_row, new_col)
            self.position = environment.grid_to_latlon(new_row, new_col)
        elif environment.fire_grid[new_row, new_col] == 1:
            # Crowd is heading into fire — override with goal-directed safety routing
            self._move_to_safety(environment)
        else:
            # Blocked by obstacle, try random movement
            self._move_random(environment)

    def _move_to_safety(self, environment) -> None:
        """
        Move towards nearest safe zone using A* pathfinding.
        Uses environment's dynamic safe zone detection.
        """
        try:
            # Initialize nodes if needed
            if self.current_node is None:
                self.current_node = environment.get_nearest_node(self.position[0], self.position[1])

            # Use environment's dynamic safe zone detection
            if self.safety_node is None or 'shelter_in_place' in self.beliefs:
                # Find nearest safe zone dynamically
                self.safety_node = environment.find_nearest_safe_node(self.current_node)
                self.current_path = []  # Force re-routing

            # No reachable safe node on road network — bypass to grid perimeter
            if self.safety_node is None:
                self._path_fail_count += 1
                self._move_toward_perimeter(environment)
                return

            # STAGGERED PATHFINDING: Check if periodic recalculation is needed
            # Recalculate every N steps (with random offset) to adjust for traffic density
            self.steps_since_recalc += 1
            should_recalc_periodic = (
                (self.steps_since_recalc + self.recalc_offset) % self.path_recalc_interval == 0
            )

            # Calculate path if needed
            # Reasons: 1) No path yet, 2) Periodic recalculation
            if not self.current_path or should_recalc_periodic:
                if should_recalc_periodic:
                    self.steps_since_recalc = 0  # Reset counter

                try:
                    self.current_path = environment.get_shortest_path(
                        self.current_node,
                        self.safety_node,
                    )
                    self._path_fail_count = 0  # Successful path — reset counter
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self._path_fail_count += 1
                    if self._path_fail_count >= 3:
                        # Road network disconnected — fall back to grid perimeter
                        self._move_toward_perimeter(environment)
                        return
                    # Try a different safe node next step
                    self.safety_node = environment.find_nearest_safe_node(self.current_node)
                    self.current_path = []

            # Move along path using LWR speed (already calculated in act())
            if self.current_path and len(self.current_path) > 1 and self.current_speed > 0.1:
                next_node = self.current_path[1]

                # DYNAMIC RE-ROUTING: Check if next node is blocked by fire
                node_data = environment.graph.nodes[next_node]
                next_lat, next_lon = node_data['y'], node_data['x']
                next_r, next_c = environment.latlon_to_grid(next_lat, next_lon)

                # If next node is burning or burnt-out, recalculate path.
                # Burnt-out (state=2) cells are impassable hot ash — civilians
                # must find an alternative route around the burned area.
                if environment.fire_grid[next_r, next_c] in (1, 2):  # BURNING or BURNT
                    self.current_path = []  # Clear stale path
                    try:
                        # Find new path avoiding fire
                        self.current_path = environment.get_shortest_path(
                            self.current_node,
                            self.safety_node,
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        self._path_fail_count += 1
                        if self._path_fail_count >= 3:
                            self._move_toward_perimeter(environment)
                            return
                        self.safety_node = environment.find_nearest_safe_node(self.current_node)
                        self.current_path = []
                    return  # Skip movement this step, recalculate next step

                # Path is clear, move to next node
                self.current_node = next_node
                self.current_path.pop(0)

                # Update position
                node_data = environment.graph.nodes[self.current_node]
                old_pos = self.grid_position
                self.position = (node_data['y'], node_data['x'])
                self.grid_position = environment.latlon_to_grid(
                    self.position[0], self.position[1]
                )

                # Track movement for herding
                if old_pos:
                    self.last_movement = np.array([
                        self.grid_position[1] - old_pos[1],
                        self.grid_position[0] - old_pos[0]
                    ], dtype=np.float32)

            elif self.current_speed <= 0.1:
                # GRIDLOCK - cannot move
                self.last_movement = np.zeros(2, dtype=np.float32)

        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            self._path_fail_count += 1
            if self._path_fail_count >= 3:
                self._move_toward_perimeter(environment)
            elif self.current_speed > 0.5:
                self._move_random(environment)
