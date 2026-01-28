"""
Commander Agent - Hybrid / Utility-Based Architecture
Makes strategic decisions using ECT (Evacuation Clearance Time) vs TTI logic
Implements 4-phase protocol: Monitoring, Pre-Evacuation, Mass Evacuation, Shelter-in-Place
"""
import numpy as np
from typing import Dict, List, Tuple
from .base_agent import Agent
from ..message import Message
from ..config import (
    COMMANDER_CONGESTION_FACTOR_BASE,
    COMMANDER_EXIT_CAPACITY,
    COMMANDER_PHASE_MONITOR_MULTIPLIER,
    COMMANDER_PHASE_PREALERT_MULTIPLIER,
    COMMANDER_PHASE_EVACUATE_MULTIPLIER,
    COMMANDER_REEVALUATION_INTERVAL,
    LOG_PHASE_TRANSITIONS
)


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
        super().__init__(agent_id, position)
        self.current_risk = 0.0
        self.tti = float('inf')  # Time To Impact from Analyst
        self.ect = 0.0  # Evacuation Clearance Time
        self.ros = 0.0  # Rate of Spread from Analyst
        self.num_exits = 1

        self.risk_history: List[float] = []
        self.active_missions: Dict[str, Dict] = {}  # mission_id -> mission_data
        self.pending_proposals: Dict[str, List[Message]] = {}  # cfp_id -> proposals
        self.last_evaluation_step = 0

        # Phase tracking
        self.current_phase = 0  # 0=Monitoring, 1=Pre-Evac, 2=Mass Evac, 3=Shelter
        self.evacuation_ordered = False
        self.warning_sent = False

        # ECT parameters
        self.congestion_factor = COMMANDER_CONGESTION_FACTOR_BASE
        self.exit_capacity = COMMANDER_EXIT_CAPACITY

        # Utility weights
        self.w_safety = 0.6
        self.w_cost = 0.2
        self.w_congestion = 0.2

    def perceive(self, environment) -> None:
        """Receive risk reports with TTI/ROS data and mission status updates"""
        for message in self.messages_inbox:
            if message.performative == "INFORM":
                if message.content.get('type') == 'RISK_REPORT':
                    self.current_risk = message.content['max_risk']
                    self.tti = message.content.get('tti', float('inf'))
                    self.ros = message.content.get('ros', 0.0)
                    self.num_exits = message.content.get('num_exits', 1)
                    self.risk_history.append(self.current_risk)

            elif message.performative == "PROPOSE":
                # Collect proposals for Contract Net Protocol
                cfp_id = message.conversation_id
                if cfp_id not in self.pending_proposals:
                    self.pending_proposals[cfp_id] = []
                self.pending_proposals[cfp_id].append(message)

            elif message.performative == "CONFIRM":
                # Mission completed
                mission_id = message.content.get('mission_id')
                if mission_id in self.active_missions:
                    del self.active_missions[mission_id]

    def _calculate_ect(self, num_agents: int) -> float:
        """
        Calculate Evacuation Clearance Time.
        ECT = (N_agents / C_exit) * gamma
        """
        if self.num_exits <= 0:
            return float('inf')

        # Total exit capacity
        total_capacity = self.exit_capacity * self.num_exits

        # ECT calculation
        ect = (num_agents / total_capacity) * self.congestion_factor

        return ect

    def _determine_phase(self, tti: float, ect: float) -> int:
        """
        Determine current phase based on TTI vs ECT comparison.

        Phase 0 (Monitoring): TTI > 2.5 * ECT
        Phase 1 (Pre-Alert): 1.5 * ECT < TTI <= 2.5 * ECT
        Phase 2 (Mass Evacuation): 1.0 * ECT < TTI <= 1.5 * ECT
        Phase 3 (Shelter-in-Place): TTI <= ECT
        """
        if tti > COMMANDER_PHASE_MONITOR_MULTIPLIER * ect:
            return 0  # Monitoring
        elif tti > COMMANDER_PHASE_PREALERT_MULTIPLIER * ect:
            return 1  # Pre-Alert
        elif tti > COMMANDER_PHASE_EVACUATE_MULTIPLIER * ect:
            return 2  # Mass Evacuation
        else:
            return 3  # Shelter-in-Place (too late to evacuate)

    def decide(self) -> None:
        """
        Make strategic decisions based on ECT vs TTI comparison.
        Re-evaluate commitments periodically.
        """
        # Check if re-evaluation is needed
        current_step = len(self.risk_history)
        if current_step - self.last_evaluation_step >= COMMANDER_REEVALUATION_INTERVAL:
            self._reevaluate_strategy()
            self.last_evaluation_step = current_step

        # Evaluate pending proposals (Contract Net Protocol)
        for cfp_id, proposals in list(self.pending_proposals.items()):
            if proposals:
                self._select_best_proposal(cfp_id, proposals)

    def _reevaluate_strategy(self) -> None:
        """
        Re-evaluate current strategy based on ECT vs TTI.
        This implements "Commitment with Evaluation".
        """
        # Update congestion factor based on active missions
        self.congestion_factor = COMMANDER_CONGESTION_FACTOR_BASE + (len(self.active_missions) * 0.1)

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
        # Calculate ECT (number of civilians from environment - simplified)
        # In real implementation, count civilians in danger zone
        num_civilians = 20  # Placeholder, should count from environment

        self.ect = self._calculate_ect(num_civilians)

        # Determine current phase
        new_phase = self._determine_phase(self.tti, self.ect)

        # Log phase transitions
        if new_phase != self.current_phase and LOG_PHASE_TRANSITIONS:
            phase_names = ["Monitoring", "Pre-Alert", "Mass Evacuation", "Shelter-in-Place"]
            print(f"  📊 Phase Transition: {phase_names[self.current_phase]} → {phase_names[new_phase]}")
            print(f"     TTI={self.tti:.1f}m, ECT={self.ect:.1f}min")

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
            if not hasattr(self, 'shelter_ordered'):
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
        self.evacuation_ordered = True

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
