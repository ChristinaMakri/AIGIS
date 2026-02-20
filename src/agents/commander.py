"""
Commander Agent - Hybrid / Utility-Based Architecture
Makes strategic decisions using ECT (Evacuation Clearance Time) vs TTI logic
Implements 4-phase protocol: Monitoring, Pre-Evacuation, Mass Evacuation, Shelter-in-Place
Enhanced with ML predictions from real historical fire data
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
    WIND_SPEED
)

# ML integration
try:
    from ..ml_predictor import RiskPredictor, ML_AVAILABLE
except ImportError:
    ML_AVAILABLE = False
    RiskPredictor = None


class CommanderAgent(Agent):
    """
    Hybrid agent that makes strategic decisions using ECT vs TTI comparison.

    Architecture: Utility-Based + Commitment with Evaluation
    - Calculates Evacuation Clearance Time (ECT)
    - Compares ECT with Time To Impact (TTI) from Analyst
    - Implements 4-phase decision protocol
    - Uses Contract Net Protocol to coordinate Rescuers
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

            # ===== CONTRACT NET PROTOCOL: PROPOSALS FROM RESCUERS =====
            elif message.performative == "PROPOSE":
                # Rescuers respond to CFP with their bids (cost, ETA, risk)
                # Collect all proposals to select best rescuer
                cfp_id = message.conversation_id
                if cfp_id not in self.pending_proposals:
                    self.pending_proposals[cfp_id] = []
                self.pending_proposals[cfp_id].append(message)

            # ===== MISSION STATUS UPDATES =====
            elif message.performative == "CONFIRM":
                # Rescuer confirms mission completion
                mission_id = message.content.get('mission_id')
                if mission_id in self.active_missions:
                    del self.active_missions[mission_id]  # Remove from active list

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
        if tti > COMMANDER_PHASE_MONITOR_MULTIPLIER * ect:
            return 0  # Phase 0: Monitoring (plenty of time)
        elif tti > COMMANDER_PHASE_PREALERT_MULTIPLIER * ect:
            return 1  # Phase 1: Pre-Alert (prepare to evacuate)
        elif tti > COMMANDER_PHASE_EVACUATE_MULTIPLIER * ect:
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

        # Evaluate pending proposals (Contract Net Protocol)
        for cfp_id, proposals in list(self.pending_proposals.items()):
            if proposals:
                self._select_best_proposal(cfp_id, proposals)

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

            # Track mission
            self.active_missions[cfp_id] = {
                'rescuer': best_proposal.sender,
                'target': best_proposal.content.get('target')
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

        # Determine current phase
        new_phase = self._determine_phase(self.tti, self.ect)

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

        # Execute phase-specific actions
        if self.current_phase == 0:
            # Phase 0: Monitoring - Standby
            pass

        elif self.current_phase == 1:
            # Phase 1: Pre-Alert - Send WARNING
            if not self.warning_sent:
                self._send_warning()
                self.warning_sent = True

        elif self.current_phase == 2:
            # Phase 2: Mass Evacuation - Broadcast EVACUATE
            if not self.evacuation_ordered:
                self._order_evacuation(redirect_to_safe_zone=False)
                self.evacuation_ordered = True

            # Also dispatch rescuers
            self._dispatch_rescuers(environment)

        elif self.current_phase == 3:
            # Phase 3: Shelter-in-Place - TOO LATE, go to nearest safe zone
            if not self.shelter_ordered:
                self._order_shelter_in_place()
                self.shelter_ordered = True

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
        Initiate Contract Net Protocol to dispatch rescuers.
        Send CFP (Call For Proposal) to available rescuers.
        """
        # Find civilians in danger (simplified)
        if np.random.random() < 0.1:  # Only occasionally send new missions
            cfp_id = f"mission_{environment.step_count}_{np.random.randint(1000)}"

            # Send CFP to all rescuers
            cfp_message = Message(
                sender=self.agent_id,
                receiver="rescuers",  # Broadcast to all rescuers
                performative="CFP",
                content={
                    'mission_type': 'RESCUE',
                    'target_location': (self.position[0] + np.random.uniform(-0.01, 0.01),
                                      self.position[1] + np.random.uniform(-0.01, 0.01)),
                    'priority': 'HIGH' if self.current_risk > 80 else 'MEDIUM'
                },
                conversation_id=cfp_id
            )
            self.send_message(cfp_message)
            self.pending_proposals[cfp_id] = []
