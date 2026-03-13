"""
Commander Agent - Hybrid / Utility-Based Architecture
Makes strategic decisions using ECT (Evacuation Clearance Time) vs TTI logic
Implements 4-phase protocol: Monitoring, Pre-Evacuation, Mass Evacuation, Shelter-in-Place
Enhanced with ML predictions from real historical fire data

ECT/TTI evacuation decision framework:
  Cova, T.J. & Johnson, J.P. (2002).
  "Microsimulation of neighborhood evacuations in the urban-wildland interface."
  Environment and Planning A, 34(12), pp. 2211–2230.

Evacuation planning and engineering basis:
  Wolshon, B. (2006).
  "Evacuation planning and engineering for Hurricane Katrina."
  The Bridge, 36(1), pp. 27–34. National Academy of Engineering.

Shelter-in-Place decision (Phase 3) validated against:
  Lagouvardos, K., Kotroni, V., Giannaros, T.M., & Dafis, S. (2019).
  "Meteorological analysis of the catastrophic wildfire in Mati, eastern Attica, Greece."
  Bulletin of the American Meteorological Society, 100(11), pp. 2243–2257.
  DOI: 10.1175/BAMS-D-18-0231.1
  [Late or absent evacuation orders were a key factor in the 102 deaths at Mati.]

Contract Net Protocol (CNP) for rescuer/ambulance coordination:
  Smith, R.G. (1980).
  "The contract net protocol: High-level communication and control
   in a distributed problem solver."
  IEEE Transactions on Computers, C-29(12), pp. 1104–1113.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_agent import Agent
from ..message import Message
from ..config import (
    COMMANDER_CONGESTION_FACTOR_BASE,
    COMMANDER_EXIT_CAPACITY,
    COMMANDER_PHASE_MONITOR_MULTIPLIER,
    COMMANDER_PHASE_PREALERT_MULTIPLIER,
    COMMANDER_PHASE_EVACUATE_MULTIPLIER,
    COMMANDER_REEVALUATION_INTERVAL,
    COMMANDER_TTI_RECONSIDER_THRESHOLD,
    COMMANDER_ECT_RECONSIDER_THRESHOLD,
    LOG_PHASE_TRANSITIONS,
    MAX_STEPS,
    WIND_SPEED,
    RESCUER_SAFETY_THRESHOLD,
    # Ablation A: when True, replace CNP bidding with random assignment.
    # Isolates the contribution of Smith (1980) Contract Net Protocol to
    # suppression and rescue outcomes.  See config.py DISABLE_CNP comment.
    DISABLE_CNP,
)

# ML integration
try:
    from ..ml_predictor import RiskPredictor, ML_AVAILABLE
except ImportError:
    ML_AVAILABLE = False
    RiskPredictor = None


class CommanderAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Commander
    ARCHITECTURE: BDI — Committed Decision-Making + CNP Manager
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • current_risk        max risk level 0–100 (from Analyst RISK_REPORT)
      • tti                 Time To Impact in minutes
      • ect                 Evacuation Clearance Time in minutes
      • ros                 Rate of Spread m/s
      • num_exits           available evacuation exits
      • current_phase       active evacuation phase 0–3
      • fwi_score           Canadian Fire Weather Index (from RiskMonitor)
      • high_risk_zones     top-3 pre-fire risk locations (from RiskMonitor)
      • active_missions     rescue missions in progress {id → data}
      • ambulance_missions  medical missions in progress {id → data}
      • firefighter_missions suppression missions in progress {id → data}
      • ml_predictions      ML risk predictor output (if available)

    DESIRES
      • Minimise casualties through optimally-timed evacuation
      • Coordinate all field units (Rescuers, Ambulances, Firefighters)
      • Issue early warning when FWI exceeds danger thresholds

    INTENTIONS  (BDI phase-commitment — Rao & Georgeff 1995)
      Phase 0: Monitor  — standby; FWI pre-warning if threshold exceeded
      Phase 1: Pre-Alert — broadcast WARNING; dispatch Firefighters early
      Phase 2: Evacuate  — broadcast EVACUATE; dispatch all field units
      Phase 3: Shelter   — broadcast REDIRECT; continue ambulance dispatch
      Phase transitions trigger PHASE_UPDATE to Analyst (belief propagation)

    COMMUNICATION
      SENDS
        → INFORM   broadcast   {type:'WARNING'}             Phase 1
        → INFORM   broadcast   {type:'FWI_WARNING'}         pre-fire
        → REQUEST  broadcast   {type:'EVACUATE'}            Phase 2
        → REQUEST  broadcast   {type:'REDIRECT_TO_SAFE_ZONE'} Phase 3
        → INFORM   analyst     {type:'PHASE_UPDATE', phase} on each
                                                            phase transition
        → CFP      rescuers    {mission_type:'RESCUE', target_location,
                               civilian_id, priority}       Phase 2+
        → CFP      ambulances  {type:'AMBULANCE_CFP', scene_node,
                               hospital_node}               Phase 2+
        → CFP      firefighters {type:'FIRE_SUPPRESSION_CFP',
                               target_location, priority}   Phase 1+
        → ACCEPT_PROPOSAL  <agent>  {mission_id, ...}       CNP award
        → REJECT_PROPOSAL  <agent>  {mission_id}            CNP reject
      RECEIVES
        ← INFORM   analyst      {type:'RISK_REPORT'}
        ← INFORM   risk_monitor {type:'RISK_FORECAST'}
        ← PROPOSE  rescuers     {cost, eta, path_risk, target, civilian_id}
        ← PROPOSE  ambulances   {cost, eta, scene_node, hospital_node}
        ← PROPOSE  firefighters {cost, eta, target}
        ← CONFIRM  field agents {mission_id, status}

    BIBLIOGRAPHY
      [1] Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of
          neighborhood evacuations in the urban-wildland interface."
          Environment and Planning A, 34(12), pp. 2211–2229.
          ECT vs TTI phase-trigger framework.
      [2] Smith, R.G. (1980). "The Contract Net Protocol."
          IEEE Trans. Computers, C-29(12), pp. 1104–1113.
          CFP → PROPOSE → ACCEPT/REJECT → CONFIRM cycle.
      [3] Wolshon, B. (2006). "Evacuation planning and engineering for
          Hurricane Katrina." The Bridge, 36(1), pp. 27–34.
          Contraflow and capacity-constrained ECT formula.
      [4] Lagouvardos, K. et al. (2019). "Meteorological analysis of the
          catastrophic wildfire in Mati, eastern Attica, Greece."
          BAMS, 100(11), pp. 2243–2257. DOI: 10.1175/BAMS-D-18-0231.1
          Phase 3 Shelter-in-Place validated against Mati 2018 disaster.
      [5] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          Committed BDI: phase not reconsidered unless TTI/ECT drift
          exceeds reconsideration thresholds.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Initialize Commander agent with ECT vs TTI decision framework.

        The Commander is the central decision-maker implementing a 4-phase protocol
        based on comparing:
        - ECT (Evacuation Clearance Time): How long to evacuate everyone
        - TTI (Time To Impact): How long until fire arrives

        Decision Logic:
        - TTI >> ECT: Safe, monitoring phase
        - TTI > ECT: Time to prepare/evacuate
        - TTI ≈ ECT: Emergency evacuation
        - TTI ≤ ECT: Too late, shelter-in-place

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # ===== RISK ASSESSMENT STATE =====
        # Receives risk reports from Analyst agent with fire spread predictions
        self.current_risk = 0.0  # Current maximum risk level (0-100)
        self.tti = float('inf')  # Time To Impact (minutes until fire reaches assets)
        self.ect = 0.0  # Evacuation Clearance Time (minutes to evacuate all civilians)
        self.ros = 0.0  # Rate of Spread from Analyst (m/s)
        self.num_exits = 1  # Number of available evacuation exits

        # History tracking for trend analysis
        self.risk_history: List[float] = []

        # ===== CONTRACT NET PROTOCOL STATE =====
        # Manages rescue missions using CNP (Call For Proposal → Propose → Accept/Reject)
        self.active_missions: Dict[str, Dict] = {}  # mission_id → mission_data
        self.pending_proposals: Dict[str, List[Message]] = {}  # cfp_id → list of proposals

        # Re-evaluation throttling (avoid recalculating every step)
        self.last_evaluation_step = 0

        # ===== PHASE MULTIPLIERS (instance attrs for ParameterAdapter overrides) =====
        self.phase_monitor_mult  = COMMANDER_PHASE_MONITOR_MULTIPLIER   # default 2.5
        self.phase_prealert_mult = COMMANDER_PHASE_PREALERT_MULTIPLIER  # default 1.5
        self.phase_evacuate_mult = COMMANDER_PHASE_EVACUATE_MULTIPLIER  # default 1.0

        # ===== 4-PHASE PROTOCOL STATE =====
        # Phase 0: Monitoring (TTI > 2.5×ECT)
        # Phase 1: Pre-Evacuation Warning (1.5×ECT < TTI ≤ 2.5×ECT)
        # Phase 2: Mass Evacuation (1.0×ECT < TTI ≤ 1.5×ECT)
        # Phase 3: Shelter-in-Place (TTI ≤ ECT)
        self.current_phase = 0
        self.evacuation_ordered = False  # Flag: evacuation order sent
        self.warning_sent = False  # Flag: pre-evacuation warning sent
        self.shelter_ordered = False  # Flag: shelter-in-place order sent

        # ===== ECT CALCULATION PARAMETERS =====
        # ECT = (N_agents / C_exit) × γ_congestion
        self.congestion_factor = COMMANDER_CONGESTION_FACTOR_BASE  # Congestion multiplier
        self.exit_capacity = COMMANDER_EXIT_CAPACITY  # Agents per minute per exit

        # ===== UTILITY FUNCTION WEIGHTS =====
        # For multi-objective decision making when selecting rescuers
        self.w_safety = 0.6  # Weight for safety (low risk paths)
        self.w_cost = 0.2  # Weight for cost (ETA, distance)
        self.w_congestion = 0.2  # Weight for congestion avoidance

        # ===== COMMITMENT WITH EVALUATION STATE =====
        self.committed_at_tti = float('inf')   # TTI when last phase was committed
        self.committed_at_ect = 0.0            # ECT when last phase was committed
        self.committed_phase = 0               # Phase at last commitment
        self.commitment_step = 0               # Step at last commitment
        self.reconsideration_log: list = []    # List of {step, reason, tti, ect}

        # Reference to fire simulation (set by simulation.py after init)
        self.fire_sim_ref = None

        # ===== PRE-IGNITION RISK (from RiskMonitorAgent) =====
        self.fwi_score: float = 0.0          # Current FWI composite score
        self.fwi_max_risk: float = 0.0       # Max cell risk from ignition_risk_grid
        self.high_risk_zones: list = []      # Top-3 (lat,lon) pre-fire risk zones
        self.pre_fire_warning_sent: bool = False  # Avoid repeated warnings

        # ===== AMBULANCE DISPATCH (Contract Net for medical units) =====
        self.ambulance_missions: Dict[str, Dict] = {}   # mission_id → data
        self.ambulance_proposals: Dict[str, List[Message]] = {}  # cfp_id → bids
        self._ambulance_cfp_counter: int = 0
        self._last_ambulance_dispatch_step: int = -50  # Throttle dispatch

        # ===== FIREFIGHTER DISPATCH (Contract Net for suppression units) =====
        self.firefighter_missions: Dict[str, Dict] = {}   # mission_id → data
        self.firefighter_proposals: Dict[str, List[Message]] = {}  # cfp_id → bids
        self._firefighter_cfp_counter: int = 0
        self._last_firefighter_dispatch_step: int = -20  # Throttle dispatch

        # Track last phase sent to Analyst to avoid redundant PHASE_UPDATE messages
        self._last_sent_phase: int = -1

        # ===== ML PREDICTION INTEGRATION =====
        # Initialize ML predictor for enhanced risk assessment using real historical fire data
        self.risk_predictor: Optional[RiskPredictor] = None
        self.ml_predictions: Dict = {}
        self.environment = None  # Store environment reference for ML predictions

        if ML_AVAILABLE and RiskPredictor:
            try:
                self.risk_predictor = RiskPredictor()
                if self.risk_predictor.is_trained:
                    print(f"  ML predictor initialized for {self.agent_id}")
            except Exception as e:
                pass  # Silently degrade if ML unavailable

    def perceive(self, environment) -> None:
        """
        Perceive environment through FIPA-ACL messages.

        Message Types:
        1. INFORM (from Analyst): Risk reports with TTI, ROS, and exit availability
        2. PROPOSE (from Rescuers): Bids for rescue missions (Contract Net Protocol)
        3. CONFIRM (from Rescuers): Mission completion notifications

        The Commander is a "command center" agent that doesn't directly sense the
        environment but relies on reports from specialized sensor/analysis agents.
        """
        # Store environment early so decide() can access it for ML predictions
        self.environment = environment

        for message in self.messages_inbox:
            # ===== RISK REPORTS FROM ANALYST =====
            if message.performative == "INFORM":
                if message.content.get('type') == 'RISK_REPORT':
                    # Extract fire spread predictions from Analyst
                    self.current_risk = message.content['max_risk']  # Max risk level (0-100)
                    self.tti = message.content.get('tti', float('inf'))  # Time To Impact (min)
                    self.ros = message.content.get('ros', 0.0)  # Rate of Spread (m/s)
                    self.num_exits = message.content.get('num_exits', 1)  # Available exits

                    # Track risk over time for trend analysis
                    self.risk_history.append(self.current_risk)

            # ===== PRE-IGNITION RISK FORECAST (from RiskMonitorAgent) =====
            elif message.performative == "INFORM" and message.content.get('type') == 'RISK_FORECAST':
                self.fwi_score     = message.content.get('fwi', 0.0)
                self.fwi_max_risk  = message.content.get('max_risk', 0.0)
                self.high_risk_zones = message.content.get('high_risk_zones', [])

            # ===== CONTRACT NET PROTOCOL: PROPOSALS FROM FIELD AGENTS =====
            elif message.performative == "PROPOSE":
                cfp_id = message.conversation_id
                # Route to correct proposal bucket by conversation_id prefix
                if cfp_id and cfp_id.startswith("amb_"):
                    if cfp_id not in self.ambulance_proposals:
                        self.ambulance_proposals[cfp_id] = []
                    self.ambulance_proposals[cfp_id].append(message)
                elif cfp_id and cfp_id.startswith("fire_"):
                    if cfp_id not in self.firefighter_proposals:
                        self.firefighter_proposals[cfp_id] = []
                    self.firefighter_proposals[cfp_id].append(message)
                else:
                    if cfp_id not in self.pending_proposals:
                        self.pending_proposals[cfp_id] = []
                    self.pending_proposals[cfp_id].append(message)

            # ===== MISSION STATUS UPDATES =====
            elif message.performative == "CONFIRM":
                mission_id = message.content.get('mission_id')
                if mission_id in self.active_missions:
                    del self.active_missions[mission_id]
                if mission_id in self.ambulance_missions:
                    del self.ambulance_missions[mission_id]
                if mission_id in self.firefighter_missions:
                    del self.firefighter_missions[mission_id]

    def _calculate_ect(self, num_agents: int) -> float:
        """
        Calculate Evacuation Clearance Time (ECT).

        ECT represents the time needed to evacuate all agents through available exits,
        accounting for exit capacity and congestion.

        Formula: ECT = (N_agents / C_total) × γ_congestion

        Where:
        - N_agents: Number of civilians to evacuate
        - C_total: Total exit capacity (agents/minute) = C_exit × num_exits
        - γ_congestion: Congestion multiplier (increases with density)

        Example:
        - 20 agents, 1 exit with capacity 10 agents/min, γ=1.0
        - ECT = (20 / 10) × 1.0 = 2 minutes

        This is compared against TTI (Time To Impact) to determine if evacuation
        is safe:
        - ECT < TTI: Safe to evacuate (enough time)
        - ECT ≥ TTI: Too late to evacuate (shelter in place)

        Args:
            num_agents: Number of civilians to evacuate

        Returns:
            Evacuation clearance time in minutes
        """
        # Safety check: if no exits available, evacuation is impossible
        if self.num_exits <= 0:
            return float('inf')

        # Safety check: if exit capacity is zero, evacuation is impossible
        if self.exit_capacity <= 0:
            return float('inf')

        # Total exit capacity: sum of all exit capacities
        total_capacity = self.exit_capacity * self.num_exits

        # ECT calculation with congestion factor
        # Congestion factor increases in dense crowds (bottlenecks, panic)
        ect = (num_agents / total_capacity) * self.congestion_factor

        return ect

    def _determine_phase(self, tti: float, ect: float) -> int:
        """
        Determine evacuation phase using TTI vs ECT comparison.

        This is the CORE decision logic of the Commander agent, implementing a
        4-phase protocol based on time margins. The phases represent increasing
        urgency as fire approaches.

        PHASE 0: MONITORING (TTI > 2.5 × ECT)
        - Fire is distant, no immediate threat
        - Continue surveillance, no action needed
        - Maintain situational awareness
        - Example: TTI=50min, ECT=20min → 50 > 2.5×20 → Monitor

        PHASE 1: PRE-ALERT (1.5×ECT < TTI ≤ 2.5×ECT)
        - Fire approaching but still manageable
        - Send warning to civilians (prepare to evacuate)
        - Alert rescuers to standby
        - Example: TTI=35min, ECT=20min → 35 > 1.5×20 but 35 ≤ 2.5×20 → Pre-Alert

        PHASE 2: MASS EVACUATION (1.0×ECT < TTI ≤ 1.5×ECT)
        - Fire approaching critical distance
        - Order immediate mass evacuation
        - All civilians must evacuate NOW
        - Example: TTI=25min, ECT=20min → 25 > 20 but 25 ≤ 1.5×20 → Evacuate

        PHASE 3: SHELTER-IN-PLACE (TTI ≤ ECT)
        - TOO LATE to evacuate safely
        - Fire will arrive before evacuation completes
        - Redirect civilians to nearest safe zones (water, parks, bunkers)
        - Based on real disasters (Mati Fire 2018: civilians trapped on highways)
        - Example: TTI=15min, ECT=20min → 15 ≤ 20 → Shelter

        Reference: This decision protocol is based on evacuation management
        research and lessons learned from wildfire disasters where late
        evacuation orders led to casualties.

        Args:
            tti: Time To Impact (minutes until fire reaches population)
            ect: Evacuation Clearance Time (minutes to evacuate all civilians)

        Returns:
            Phase number (0=Monitor, 1=Pre-Alert, 2=Evacuate, 3=Shelter)
        """
        if tti > self.phase_monitor_mult * ect:
            return 0  # Phase 0: Monitoring (plenty of time)
        elif tti > self.phase_prealert_mult * ect:
            return 1  # Phase 1: Pre-Alert (prepare to evacuate)
        elif tti > self.phase_evacuate_mult * ect:
            return 2  # Phase 2: Mass Evacuation (evacuate NOW)
        else:
            return 3  # Phase 3: Shelter-in-Place (too late to evacuate safely)

    def _should_reconsider_commitment(self) -> bool:
        """
        Determine whether the current commitment should be re-evaluated.

        Returns True if:
        - First call (committed_at_tti == inf)
        - TTI has drifted beyond COMMANDER_TTI_RECONSIDER_THRESHOLD
        - ECT has drifted beyond COMMANDER_ECT_RECONSIDER_THRESHOLD
        """
        if self.committed_at_tti == float('inf'):
            return True

        tti_drift = abs(self.tti - self.committed_at_tti)
        ect_drift = abs(self.ect - self.committed_at_ect)

        if tti_drift > COMMANDER_TTI_RECONSIDER_THRESHOLD:
            self.reconsideration_log.append({
                'step': len(self.risk_history),
                'reason': 'tti_drift',
                'tti': self.tti,
                'ect': self.ect,
                'tti_drift': tti_drift,
            })
            return True

        if ect_drift > COMMANDER_ECT_RECONSIDER_THRESHOLD:
            self.reconsideration_log.append({
                'step': len(self.risk_history),
                'reason': 'ect_drift',
                'tti': self.tti,
                'ect': self.ect,
                'ect_drift': ect_drift,
            })
            return True

        return False

    def decide(self) -> None:
        """
        Make strategic decisions based on ECT vs TTI comparison.
        Re-evaluate commitments periodically and when thresholds drift.
        """
        # Check if re-evaluation is needed (throttled by interval AND commitment drift)
        current_step = len(self.risk_history)
        interval_due = (current_step - self.last_evaluation_step >= COMMANDER_REEVALUATION_INTERVAL)
        if interval_due and self._should_reconsider_commitment():
            self._reevaluate_strategy()
            self.last_evaluation_step = current_step

        # Evaluate pending rescue proposals.
        # Full model: Contract Net Protocol — best-utility bid wins
        #   (Smith 1980 — "The contract net protocol", IEEE Trans. Computers C-29(12)).
        # Ablation A (DISABLE_CNP=True): random assignment — pick first proposal,
        #   ignoring cost/ETA.  Used to quantify CNP's contribution to outcome quality.
        for cfp_id, proposals in list(self.pending_proposals.items()):
            if proposals:
                if DISABLE_CNP:
                    self._assign_random_proposal(cfp_id, proposals,
                                                  self.active_missions)
                else:
                    self._select_best_proposal(cfp_id, proposals)
            else:
                del self.pending_proposals[cfp_id]

        # Evaluate pending ambulance proposals (same CNP / random split)
        for cfp_id, proposals in list(self.ambulance_proposals.items()):
            if proposals:
                if DISABLE_CNP:
                    self._assign_random_proposal(cfp_id, proposals,
                                                  self.ambulance_missions)
                else:
                    self._select_best_ambulance_proposal(cfp_id, proposals)
            else:
                del self.ambulance_proposals[cfp_id]

    def _dispatch_firefighters(self, environment) -> None:
        """
        Dispatch firefighters to the highest-intensity burning cluster.

        Strategy: find the cell with maximum temperature among burning cells
        and issue a FIRE_SUPPRESSION_CFP to all firefighter agents.
        Throttled to one CFP every 20 steps to avoid flooding.

        Contract Net Protocol (Smith 1980): Commander is Manager;
        Firefighters are Contractors who bid with PROPOSE.
        """
        import numpy as np

        step = environment.step_count
        if step - self._last_firefighter_dispatch_step < 20:
            return

        agents = getattr(environment, 'agents', {})
        firefighters = agents.get('firefighters', [])
        if not firefighters:
            return

        fire_grid = environment.fire_grid
        burning = np.argwhere(fire_grid == 1)
        if len(burning) == 0:
            return

        # Pick hottest burning cell as suppression target
        temp_grid = environment.temperature_grid
        hottest_idx = int(np.argmax([temp_grid[r, c] for r, c in burning]))
        target_row, target_col = burning[hottest_idx]
        target_lat, target_lon = environment.grid_to_latlon(int(target_row), int(target_col))

        self._firefighter_cfp_counter += 1
        cfp_id = f"fire_{self._firefighter_cfp_counter}_{step}"

        cfp = Message(
            sender=self.agent_id,
            receiver="firefighters",
            performative="CFP",
            content={
                'type': 'FIRE_SUPPRESSION_CFP',
                'target_location': (target_lat, target_lon),
                'target_grid': (int(target_row), int(target_col)),
                'mission_id': cfp_id,
                'priority': 'HIGH',
            },
            conversation_id=cfp_id,
        )
        self.send_message(cfp)
        self.firefighter_proposals[cfp_id] = []
        self._last_firefighter_dispatch_step = step

    def _select_best_firefighter_proposal(self, cfp_id: str,
                                          proposals: List[Message]) -> None:
        """Select the lowest-cost firefighter bid and award the mission."""
        best = min(proposals, key=lambda p: p.content.get('cost', float('inf')))

        for proposal in proposals:
            if proposal == best:
                accept = Message(
                    sender=self.agent_id,
                    receiver=proposal.sender,
                    performative="ACCEPT_PROPOSAL",
                    content={
                        'mission_id': cfp_id,
                        'target_location': proposal.content.get('target'),
                        'target_grid': proposal.content.get('target_grid'),
                    },
                    conversation_id=cfp_id,
                )
                self.send_message(accept)
                self.firefighter_missions[cfp_id] = {
                    'firefighter': proposal.sender,
                    'target': proposal.content.get('target'),
                }
            else:
                reject = Message(
                    sender=self.agent_id,
                    receiver=proposal.sender,
                    performative="REJECT_PROPOSAL",
                    content={'mission_id': cfp_id},
                    conversation_id=cfp_id,
                )
                self.send_message(reject)

        del self.firefighter_proposals[cfp_id]

        # Evaluate pending firefighter proposals (fire suppression CNP)
        for cfp_id, proposals in list(self.firefighter_proposals.items()):
            if proposals:
                self._select_best_firefighter_proposal(cfp_id, proposals)
            else:
                del self.firefighter_proposals[cfp_id]

    def _assign_random_proposal(self, cfp_id: str, proposals: List[Message],
                                 mission_dict: dict) -> None:
        """
        Ablation A — random task assignment (DISABLE_CNP=True).

        Accepts the FIRST proposal received instead of evaluating bid cost/ETA.
        All other proposals are rejected.  This baseline replaces:

          Smith, R.G. (1980). "The contract net protocol: High-level
          communication and control in a distributed problem solver."
          IEEE Transactions on Computers, C-29(12), pp. 1104–1113.

        Comparing full-CNP vs random-assignment results quantifies how much
        Smith's bidding mechanism improves suppression/rescue outcomes (ablation
        methodology per Grimm et al. 2020 ODD, design-concepts section).
        """
        winner = proposals[0]
        accept = Message(
            sender=self.agent_id,
            receiver=winner.sender,
            performative="ACCEPT_PROPOSAL",
            content={
                'mission_id': cfp_id,
                'target': winner.content.get('target'),
                'target_grid': winner.content.get('target_grid'),
            },
            conversation_id=cfp_id,
        )
        self.send_message(accept)
        mission_dict[cfp_id] = {
            'agent': winner.sender,
            'target': winner.content.get('target'),
        }
        for proposal in proposals[1:]:
            reject = Message(
                sender=self.agent_id,
                receiver=proposal.sender,
                performative="REJECT_PROPOSAL",
                content={'mission_id': cfp_id},
                conversation_id=cfp_id,
            )
            self.send_message(reject)
        # Clear processed set (rescue proposals keyed in pending_proposals)
        for store in (self.pending_proposals, self.ambulance_proposals):
            if cfp_id in store:
                del store[cfp_id]
                break

    def _reevaluate_strategy(self) -> None:
        """
        Re-evaluate current strategy based on ECT vs TTI.
        Implements "Commitment with Evaluation" using simulation-derived ML features.
        """
        # Update congestion factor based on active missions
        self.congestion_factor = COMMANDER_CONGESTION_FACTOR_BASE + (len(self.active_missions) * 0.1)

        # Use ML predictions if available
        if ML_AVAILABLE and self.risk_predictor and hasattr(self, 'environment') and self.environment:
            try:
                env = self.environment
                agents = getattr(env, 'agents', {}) or {}

                # Get wind direction from fire sim if available
                wind_dir = [1.0, 0.0]
                wind_speed = getattr(env, 'wind_speed', WIND_SPEED)
                if self.fire_sim_ref is not None:
                    wind_dir = list(self.fire_sim_ref.wind_direction)
                    wind_speed = self.fire_sim_ref.wind_speed

                simulation_state = {
                    'fire_grid': env.fire_grid,
                    'fuel_type_grid': getattr(env, 'fuel_type_grid', None),
                    'elevation_grid': env.elevation_grid,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_dir,
                    'humidity': getattr(env, 'humidity', 30.0),
                    'tti_minutes': self.tti,
                    'ect_minutes': self.ect,
                    'current_phase': self.current_phase,
                    'step': env.step_count,
                    'max_steps': MAX_STEPS,
                    'agents': agents,
                }

                self.ml_predictions = self.risk_predictor.predict_casualty_risk(simulation_state)

                # Log ML predictions periodically (every 20 steps)
                if len(self.risk_history) % 20 == 0 and self.ml_predictions:
                    print(f"  ML Predictions: Risk={self.ml_predictions.get('risk_level', 'N/A')}, "
                          f"Casualties={self.ml_predictions.get('predicted_casualties', 0):.1f}, "
                          f"Evacuations={self.ml_predictions.get('predicted_evacuations', 0):.0f}")

            except Exception:
                pass

    def _select_best_proposal(self, cfp_id: str, proposals: List[Message]) -> None:
        """
        Select best proposal based on utility function.
        U(A) = w_safety * P(Safety) - w_cost * Cost - w_congestion * Congestion
        """
        best_proposal = None
        best_utility = -float('inf')

        for proposal in proposals:
            # Extract proposal data
            cost = proposal.content.get('cost', 100)
            eta = proposal.content.get('eta', 10)

            # Calculate utility (simplified)
            safety_score = 100 / (eta + 1)  # Lower ETA = higher safety
            cost_score = cost
            congestion_score = len(self.active_missions) * 10

            utility = (self.w_safety * safety_score -
                      self.w_cost * cost_score -
                      self.w_congestion * congestion_score)

            if utility > best_utility:
                best_utility = utility
                best_proposal = proposal

        # Accept best proposal, reject others
        if best_proposal:
            accept_msg = Message(
                sender=self.agent_id,
                receiver=best_proposal.sender,
                performative="ACCEPT_PROPOSAL",
                content={
                    'mission_id': cfp_id,
                    'target': best_proposal.content.get('target')
                },
                conversation_id=cfp_id
            )
            self.send_message(accept_msg)

            # Track mission (civilian_id lets _dispatch_rescuers skip already-targeted civilians)
            civilian_id = best_proposal.content.get('civilian_id')
            self.active_missions[cfp_id] = {
                'rescuer': best_proposal.sender,
                'target': best_proposal.content.get('target'),
                'civilian_id': civilian_id,
            }

        # Reject others
        for proposal in proposals:
            if proposal != best_proposal:
                reject_msg = Message(
                    sender=self.agent_id,
                    receiver=proposal.sender,
                    performative="REJECT_PROPOSAL",
                    content={'mission_id': cfp_id},
                    conversation_id=cfp_id
                )
                self.send_message(reject_msg)

        # Clear processed proposals
        del self.pending_proposals[cfp_id]

    def act(self, environment) -> None:
        """
        Execute decisions based on current phase.
        Implements 4-phase protocol based on ECT vs TTI.
        """
        # Store environment reference for ML predictions
        self.environment = environment

        # Calculate ECT - count actual active civilians from environment
        # Access simulation's agent list through environment.agents
        agents = getattr(environment, 'agents', None)
        if agents and 'civilians' in agents:
            num_civilians = sum(1 for c in agents['civilians'] if c.is_active)
        else:
            num_civilians = 20  # Fallback if environment doesn't expose agents

        self.ect = self._calculate_ect(num_civilians)

        # Determine current phase from ECT vs TTI logic
        new_phase = self._determine_phase(self.tti, self.ect)

        # Apply ML-based minimum phase floor.
        # ML predictions cannot lower the phase, only raise it.
        if self.ml_predictions:
            ml_risk = self.ml_predictions.get('risk_level', 'MEDIUM')
            ml_min_phase = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 1, 'CRITICAL': 2}.get(ml_risk, 0)
            if ml_min_phase > new_phase:
                if LOG_PHASE_TRANSITIONS:
                    print(f"  ML override: {ml_risk} risk → phase escalated "
                          f"{new_phase} → {ml_min_phase}")
                new_phase = ml_min_phase

        # Log phase transitions and record commitment
        if new_phase != self.current_phase:
            if LOG_PHASE_TRANSITIONS:
                phase_names = ["Monitoring", "Pre-Alert", "Mass Evacuation", "Shelter-in-Place"]
                print(f"  Phase Transition: {phase_names[self.current_phase]} → {phase_names[new_phase]}")
                print(f"     TTI={self.tti:.1f}min, ECT={self.ect:.1f}min")

            # Record commitment state at phase change
            self.committed_at_tti = self.tti
            self.committed_at_ect = self.ect
            self.committed_phase = new_phase
            self.commitment_step = environment.step_count
            self.reconsideration_log.append({
                'step': environment.step_count,
                'reason': 'phase_change',
                'tti': self.tti,
                'ect': self.ect,
                'new_phase': new_phase,
            })

        self.current_phase = new_phase

        # BDI belief propagation: notify Analyst of phase so it can apply
        # conservative TTI margins during active evacuation (Phase ≥ 2).
        if self.current_phase != self._last_sent_phase:
            phase_msg = Message(
                sender=self.agent_id,
                receiver="analyst",
                performative="INFORM",
                content={'type': 'PHASE_UPDATE', 'phase': self.current_phase}
            )
            self.send_message(phase_msg)
            self._last_sent_phase = self.current_phase

        # ---- Pre-ignition FWI warning (Phase 0 only, before any fire) ----
        self._check_fwi_prewarning(environment)

        # Execute phase-specific actions
        if self.current_phase == 0:
            # Phase 0: Monitoring - Standby
            pass

        elif self.current_phase == 1:
            # Phase 1: Pre-Alert - Send WARNING; deploy firefighters early
            if not self.warning_sent:
                self._send_warning()
                self.warning_sent = True
            self._dispatch_firefighters(environment)

        elif self.current_phase == 2:
            # Phase 2: Mass Evacuation - Broadcast EVACUATE
            if not self.evacuation_ordered:
                self._order_evacuation(redirect_to_safe_zone=False)
                self.evacuation_ordered = True

            # Dispatch all field units
            self._dispatch_rescuers(environment)
            self._dispatch_ambulances(environment)
            self._dispatch_firefighters(environment)

        elif self.current_phase == 3:
            # Phase 3: Shelter-in-Place - TOO LATE, go to nearest safe zone
            if not self.shelter_ordered:
                self._order_shelter_in_place()
                self.shelter_ordered = True

            # Keep dispatching ambulances during shelter phase
            self._dispatch_ambulances(environment)

    def _send_warning(self) -> None:
        """Broadcast pre-evacuation warning to all civilians"""
        message = Message(
            sender=self.agent_id,
            receiver="broadcast",
            performative="INFORM",
            content={
                'type': 'WARNING',
                'urgency': 'MEDIUM',
                'message': 'Be prepared to evacuate'
            }
        )
        self.send_message(message)

    def _order_evacuation(self, redirect_to_safe_zone: bool = False) -> None:
        """Broadcast evacuation order to all civilians"""
        message = Message(
            sender=self.agent_id,
            receiver="broadcast",
            performative="REQUEST",
            content={
                'type': 'EVACUATE',
                'urgency': 'HIGH',
                'destination': 'safe_zone' if redirect_to_safe_zone else 'standard'
            }
        )
        self.send_message(message)

    def _order_shelter_in_place(self) -> None:
        """
        CRITICAL: Too late to evacuate by road - traffic will jam.
        Redirect civilians to nearest safe zone (park, water, map edge).
        Uses environment's dynamic safe zone detection.
        """
        message = Message(
            sender=self.agent_id,
            receiver="broadcast",
            performative="REQUEST",
            content={
                'type': 'REDIRECT_TO_SAFE_ZONE',
                'urgency': 'CRITICAL',
                'message': 'TOO LATE TO EVACUATE - GO TO NEAREST SAFE ZONE (PARK/WATER/EDGE)'
            }
        )
        self.send_message(message)

        if LOG_PHASE_TRANSITIONS:
            print(f"  ⚠️  PHASE 3: Shelter-in-Place activated! TTI ≤ ECT")

    def _dispatch_rescuers(self, environment) -> None:
        """
        Event-driven rescue dispatch: send CFPs only for civilians who are
        on/near burning cells or in critically hot zones.
        """
        agents = getattr(environment, 'agents', {})
        civilians = agents.get('civilians', [])
        if not civilians:
            return

        # Civilians already assigned to an active rescue mission
        already_targeted = {
            m.get('civilian_id')
            for m in self.active_missions.values()
            if 'civilian_id' in m
        }

        for civilian in civilians:
            if not civilian.is_active or civilian.is_evacuated:
                continue
            if civilian.agent_id in already_targeted:
                continue
            if civilian.grid_position is None:
                continue

            r, c = civilian.grid_position
            temp = environment.temperature_grid[r, c]
            on_fire = environment.fire_grid[r, c] in (1, 2)
            hot = temp > RESCUER_SAFETY_THRESHOLD * 0.5  # > 35°C

            if not (on_fire or hot):
                continue

            cfp_id = f"rescue_{civilian.agent_id}_{environment.step_count}"
            priority = 'CRITICAL' if on_fire else 'HIGH'

            cfp = Message(
                sender=self.agent_id,
                receiver="rescuers",
                performative="CFP",
                content={
                    'mission_type': 'RESCUE',
                    'target_location': civilian.position,
                    'civilian_id': civilian.agent_id,
                    'priority': priority,
                },
                conversation_id=cfp_id,
            )
            self.send_message(cfp)
            self.pending_proposals[cfp_id] = []
            already_targeted.add(civilian.agent_id)

    def _check_fwi_prewarning(self, environment) -> None:
        """
        Issue a pre-fire warning to civilians when FWI is dangerously high
        even if no fire has started yet. Gives civilians advance notice to
        prepare (pack essentials, check evacuation routes).

        Only fires once per simulation and only in Phase 0 (Monitoring).
        """
        from ..config import FWI_HIGH_RISK_THRESHOLD, FWI_EXTREME_RISK_THRESHOLD, LOG_PHASE_TRANSITIONS
        if self.pre_fire_warning_sent:
            return
        if self.current_phase != 0:
            return

        # Check if any fire has started yet
        if hasattr(environment, 'fire_grid'):
            import numpy as np
            burning = int(np.sum(environment.fire_grid == 1))
            if burning > 0:
                return  # Fire already active — normal phase protocol takes over

        if self.fwi_score >= FWI_EXTREME_RISK_THRESHOLD:
            urgency = 'HIGH'
            msg_text = (f'Extreme fire weather (FWI={self.fwi_score:.0f}). '
                        f'Pre-position near exits. Fire risk is EXTREME.')
        elif self.fwi_score >= FWI_HIGH_RISK_THRESHOLD:
            urgency = 'MEDIUM'
            msg_text = (f'High fire weather index (FWI={self.fwi_score:.0f}). '
                        f'Be aware of evacuation routes. Fire risk is HIGH.')
        else:
            return  # FWI below threshold — no pre-warning needed

        warn_msg = Message(
            sender=self.agent_id,
            receiver="broadcast",
            performative="INFORM",
            content={
                'type': 'FWI_WARNING',
                'urgency': urgency,
                'message': msg_text,
                'fwi': self.fwi_score,
                'high_risk_zones': self.high_risk_zones,
            }
        )
        self.send_message(warn_msg)
        self.pre_fire_warning_sent = True

        if LOG_PHASE_TRANSITIONS:
            print(f"  [Commander] Pre-fire FWI warning sent: {urgency} — FWI={self.fwi_score:.0f}")

    def _dispatch_ambulances(self, environment) -> None:
        """
        Dispatch ambulances to zones with active fire casualties.

        Strategy: find the highest-temperature zone with civilians nearby
        (or recently active), issue an AMBULANCE_CFP to all ambulance agents.
        Throttled to one dispatch every 30 steps to avoid flooding.
        """
        import numpy as np
        from ..config import FIRE_TEMP_BURNING

        step = environment.step_count
        if step - self._last_ambulance_dispatch_step < 30:
            return

        agents = getattr(environment, 'agents', {})
        ambulances = agents.get('ambulances', [])
        if not ambulances:
            return

        hospital_nodes = getattr(environment, 'hospital_nodes', [])
        # Fall back to safe nodes if no hospitals registered
        if not hospital_nodes:
            safe = getattr(environment, 'safe_nodes', set())
            hospital_nodes = list(safe)[:3] if safe else []
        if not hospital_nodes:
            return

        # Find the highest-risk grid cell that is burning or very hot
        temp_grid = environment.temperature_grid
        fire_grid  = environment.fire_grid
        hot_mask   = (fire_grid == 1) | (temp_grid > float(FIRE_TEMP_BURNING) * 0.5)
        if not np.any(hot_mask):
            return

        # Pick the hottest cell as the dispatch scene
        idx = np.unravel_index(np.argmax(temp_grid * hot_mask), temp_grid.shape)
        scene_lat, scene_lon = environment.grid_to_latlon(int(idx[0]), int(idx[1]))
        scene_node = environment.get_nearest_node(scene_lat, scene_lon)

        # Pick nearest hospital
        hospital_node = hospital_nodes[0]

        self._ambulance_cfp_counter += 1
        cfp_id = f"amb_{self._ambulance_cfp_counter}_{step}"

        cfp = Message(
            sender=self.agent_id,
            receiver="ambulances",
            performative="CFP",
            content={
                'type':          'AMBULANCE_CFP',
                'scene_node':    scene_node,
                'hospital_node': hospital_node,
                'mission_id':    cfp_id,
                'priority':      'HIGH',
            },
            conversation_id=cfp_id,
        )
        self.send_message(cfp)
        self.ambulance_proposals[cfp_id] = []
        self._last_ambulance_dispatch_step = step

    def _select_best_ambulance_proposal(self, cfp_id: str, proposals: List[Message]) -> None:
        """Select the lowest-cost ambulance bid and award the mission."""
        best = min(proposals, key=lambda p: p.content.get('cost', float('inf')))

        for proposal in proposals:
            if proposal == best:
                accept = Message(
                    sender=self.agent_id,
                    receiver=proposal.sender,
                    performative="ACCEPT_PROPOSAL",
                    content={
                        'mission_id':    cfp_id,
                        'scene_node':    proposal.content.get('scene_node'),
                        'hospital_node': proposal.content.get('hospital_node'),
                    },
                    conversation_id=cfp_id,
                )
                self.send_message(accept)
                self.ambulance_missions[cfp_id] = {
                    'ambulance': proposal.sender,
                    'scene_node': proposal.content.get('scene_node'),
                    'hospital_node': proposal.content.get('hospital_node'),
                }
            else:
                reject = Message(
                    sender=self.agent_id,
                    receiver=proposal.sender,
                    performative="REJECT_PROPOSAL",
                    content={'mission_id': cfp_id},
                    conversation_id=cfp_id,
                )
                self.send_message(reject)

        del self.ambulance_proposals[cfp_id]
