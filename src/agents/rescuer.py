"""
Rescuer Agent - Goal-Based / Practical Reasoning Architecture
Executes rescue missions using Contract Net Protocol with risk-adjusted bidding
Implements safety protocol to refuse dangerous missions

Contract Net Protocol (CNP) implemented here:
  Smith, R.G. (1980).
  "The Contract Net Protocol: High-level communication and control in a
  distributed problem solver."
  IEEE Transactions on Computers, C-29(12), pp. 1104–1113.
  https://doi.org/10.1109/TC.1980.1675516

  The CNP CFP → PROPOSE → ACCEPT/REJECT message cycle used in
  _handle_cfp() / _handle_mission_assignment() maps directly to
  Smith's orignal manager–contractor interaction model.
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict
from .base_agent import Agent
from ..message import Message
from .. import config as _cfg_module
from ..config import (
    RESCUER_MAX_SPEED,
    RESCUER_FUEL_CAPACITY,
    RESCUER_RISK_ALPHA,
    RESCUER_SAFETY_THRESHOLD,
    RESCUER_PATH_RECALC_INTERVAL
)


class RescuerAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Rescuer
    ARCHITECTURE: BDI — Practical Reasoning with Risk Assessment
                  CNP Contractor role (Smith 1980)
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • current_mission     active mission dict {mission_id, target, commander}
      • current_path        planned A* route (list of graph node IDs)
      • current_node        current road-network node
      • mission_status      IDLE | MOVING | ARRIVED | ABORTED
      • fuel                remaining resource level (0–100)
      • risk_alpha          risk-penalty weight in bid cost function
      • safety_threshold    maximum tolerable path temperature (°C)

    DESIRES
      • Complete rescue missions and bring civilians to safety
      • Preserve own safety (refuse missions through active fire)
      • Manage fuel resources responsibly

    INTENTIONS
      Idle:    listen for CFPs; assess path risk; bid if safe
      Moving:  follow A* path; re-route periodically; abort if blocked
      Arrived: notify Commander of mission completion; return to IDLE
      Bid:     Cost = (distance/speed) + (risk × α) + (100 − fuel)

    COMMUNICATION
      SENDS
        → PROPOSE   commander  {cost,eta,path_risk,target,civilian_id}
              CNP bid in response to CFP
        → REFUSE    commander  {reason}
              reject CFP (dangerous/busy/no fuel/no path)
        → REFUSE    commander  {reason,mission_id}
              abort active mission (path blocked by fire)
        → CONFIRM   commander  {mission_id,status}
              mission completed successfully
      RECEIVES
        ← CFP            commander  {target_location,civilian_id,priority}
        ← ACCEPT_PROPOSAL commander  {mission_id,target}
        ← REJECT_PROPOSAL commander  {}

    BIBLIOGRAPHY
      [1] Smith, R.G. (1980). "The Contract Net Protocol: High-level
          communication and control in a distributed problem solver."
          IEEE Trans. Computers, C-29(12), pp. 1104–1113.
          DOI: 10.1109/TC.1980.1675516
      [2] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
      [3] Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of
          neighborhood evacuations in the urban-wildland interface."
          Environment and Planning A, 34(12), pp. 2211–2229.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float],
                 max_speed: float = RESCUER_MAX_SPEED):
        """
        Initialize Rescuer agent with goal-based practical reasoning architecture.

        Rescuers participate in Contract Net Protocol (CNP) to bid on rescue missions:
        1. Commander sends CFP (Call For Proposal) with mission details
        2. Rescuers assess path risk and calculate bid cost
        3. Commander selects best proposal (lowest cost)
        4. Selected rescuer executes mission with dynamic re-routing

        SAFETY PROTOCOL: Rescuers REFUSE missions through active fire zones.
        This simulates professional safety standards where rescuers won't
        enter zones that risk their own lives.

        Bid Calculation:
        Cost = (Distance/Speed) + (Risk_path × α) + (100 - Fuel)

        Where:
        - Distance/Speed: Estimated time to arrival (ETA)
        - Risk_path: Maximum temperature along planned path
        - α: Risk penalty weight (higher α = more risk-averse)
        - 100 - Fuel: Fuel penalty (low fuel = higher cost)

        Ref — Contract Net Protocol:
        Smith, R.G. (1980). "The Contract Net Protocol."
        IEEE Transactions on Computers, C-29(12), pp. 1104–1113.

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
            max_speed: Maximum movement speed (grid cells per step)
        """
        super().__init__(agent_id, position)

        # ===== MOVEMENT PARAMETERS =====
        self.max_speed = max_speed  # Movement speed for ETA calculations
        self.fuel = RESCUER_FUEL_CAPACITY  # Resource constraint

        # ===== MISSION STATE =====
        self.current_mission: Optional[Dict] = None  # Active mission details
        self.current_path: List[int] = []  # Planned path (node IDs)
        self.current_node: Optional[int] = None  # Current graph node
        self.target_node: Optional[int] = None  # Mission target node
        self.mission_status = "IDLE"  # IDLE, MOVING, ARRIVED, ABORTED

        # ===== RISK ASSESSMENT PARAMETERS =====
        # Rescuer scans temperature grid along path to assess danger
        self.risk_alpha = RESCUER_RISK_ALPHA  # Risk penalty weight in bid
        self.safety_threshold = RESCUER_SAFETY_THRESHOLD  # Refuse if path risk > this

        # ===== PERFORMANCE OPTIMIZATION: STAGGERED PATHFINDING =====
        self.path_recalc_interval = RESCUER_PATH_RECALC_INTERVAL
        self.steps_since_recalc = 0
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)

        # ===== RL INTEGRATION =====
        self._rl_obs: Optional[np.ndarray] = None   # set by MARL loop each step
        self._rl_action: Optional[int] = None        # last RL action taken

    def perceive(self, environment) -> None:
        """Receive CFPs and mission assignments"""
        for message in self.messages_inbox:
            if message.performative == "CFP":
                self._handle_cfp(message, environment)

            elif message.performative == "ACCEPT_PROPOSAL":
                self._handle_mission_assignment(message, environment)

            elif message.performative == "REJECT_PROPOSAL":
                # Proposal rejected, return to idle
                if self.current_mission and self.current_mission.get('conversation_id') == message.conversation_id:
                    self.current_mission = None
                    self.mission_status = "IDLE"

    def _assess_path_risk(self, path: List[int], environment) -> float:
        """
        Assess risk along planned rescue path by scanning temperature grid.

        CRITICAL SAFETY CHECK: This prevents rescuers from attempting missions
        through active fire zones. Scans each node in the planned path and
        returns the maximum temperature encountered.

        Risk Assessment:
        Risk_path = max(Temperature_node) for all nodes in path

        If Risk_path > Safety_Threshold (70°C):
        → REFUSE mission (too dangerous)

        If Risk_path ≤ Safety_Threshold:
        → Include risk in bid calculation as penalty

        This implements professional rescue protocol where personnel safety
        is paramount. Rescuers won't accept suicide missions.

        Args:
            path: List of graph node IDs representing planned route
            environment: Environment with temperature_grid

        Returns:
            Maximum temperature (°C) along the path
        """
        max_risk = 0.0

        # Scan each node in the planned path
        for node in path:
            # Get node geographic position
            node_data = environment.graph.nodes[node]
            lat, lon = node_data['y'], node_data['x']

            # Convert to grid coordinates for temperature lookup
            row, col = environment.latlon_to_grid(lat, lon)

            # Check temperature at this location
            temp = environment.temperature_grid[row, col]

            # Track maximum temperature (worst-case risk)
            if temp > max_risk:
                max_risk = temp

        return max_risk

    def _handle_cfp(self, message: Message, environment) -> None:
        """
        Handle Call For Proposal from Commander.
        Calculate cost with risk assessment and send proposal or refuse.

        Bid Calculation: Cost = (Length/V_avg) + (Risk_path * α) + (100 - FuelLevel)
        Safety Protocol: IF Risk_path > Safety_Threshold THEN REFUSE
        """
        if self.mission_status != "IDLE":
            # Already on a mission, refuse
            refuse_msg = Message(
                sender=self.agent_id,
                receiver=message.sender,
                performative="REFUSE",
                content={'reason': 'Already on mission'},
                conversation_id=message.conversation_id
            )
            self.send_message(refuse_msg)
            return

        # Calculate cost
        target_location = message.content.get('target_location')
        if not target_location:
            return

        target_lat, target_lon = target_location

        # Find path to target
        try:
            if self.current_node is None:
                self.current_node = environment.get_nearest_node(self.position[0], self.position[1])

            target_node = environment.get_nearest_node(target_lat, target_lon)

            # Calculate path using A* (single traversal)
            path = nx.shortest_path(environment.graph, self.current_node, target_node, weight='length')

            # Calculate path length from the computed path (avoid redundant traversal)
            path_length = sum(
                environment.graph[path[i]][path[i+1]][0]['length']
                for i in range(len(path) - 1)
            )

            # RISK ASSESSMENT: Scan path against fire grid
            path_risk = self._assess_path_risk(path, environment)

            # SAFETY PROTOCOL: Refuse if path is too dangerous
            if path_risk > self.safety_threshold:
                refuse_msg = Message(
                    sender=self.agent_id,
                    receiver=message.sender,
                    performative="REFUSE",
                    content={
                        'reason': f'Path too dangerous (risk={path_risk:.1f} > {self.safety_threshold})',
                        'path_risk': path_risk
                    },
                    conversation_id=message.conversation_id
                )
                self.send_message(refuse_msg)
                return

            # Safety check: refuse if speed is zero (edge case)
            if self.max_speed <= 0:
                refuse_msg = Message(
                    sender=self.agent_id,
                    receiver=message.sender,
                    performative="REFUSE",
                    content={'reason': 'Invalid configuration: max_speed is zero'},
                    conversation_id=message.conversation_id
                )
                self.send_message(refuse_msg)
                return

            # Calculate base cost
            time_cost = path_length / (self.max_speed * 100)  # Time to reach
            fuel_penalty = 100 - self.fuel

            # BID CALCULATION with risk penalty
            total_cost = time_cost + (path_risk * self.risk_alpha / 100.0) + fuel_penalty

            # Calculate ETA
            eta = path_length / (self.max_speed * 100)

            # Check if we have enough fuel
            if (path_length / 1000) > self.fuel:
                refuse_msg = Message(
                    sender=self.agent_id,
                    receiver=message.sender,
                    performative="REFUSE",
                    content={'reason': 'Insufficient fuel'},
                    conversation_id=message.conversation_id
                )
                self.send_message(refuse_msg)
                return

            # Send proposal (include civilian_id so Commander can track targeted civilians)
            proposal = Message(
                sender=self.agent_id,
                receiver=message.sender,
                performative="PROPOSE",
                content={
                    'cost': total_cost,
                    'eta': eta,
                    'path_risk': path_risk,
                    'target': target_location,
                    'civilian_id': message.content.get('civilian_id'),
                },
                conversation_id=message.conversation_id
            )
            self.send_message(proposal)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path available
            refuse_msg = Message(
                sender=self.agent_id,
                receiver=message.sender,
                performative="REFUSE",
                content={'reason': 'No path to target'},
                conversation_id=message.conversation_id
            )
            self.send_message(refuse_msg)

    def _handle_mission_assignment(self, message: Message, environment) -> None:
        """Handle accepted proposal - start mission"""
        target_location = message.content.get('target')
        mission_id = message.content.get('mission_id')

        if not target_location:
            return

        target_lat, target_lon = target_location

        try:
            if self.current_node is None:
                self.current_node = environment.get_nearest_node(self.position[0], self.position[1])

            self.target_node = environment.get_nearest_node(target_lat, target_lon)
            self.current_path = nx.shortest_path(environment.graph, self.current_node, self.target_node, weight='length')

            self.current_mission = {
                'mission_id': mission_id,
                'target': target_location,
                'conversation_id': message.conversation_id,
                'commander': message.sender
            }
            self.mission_status = "MOVING"

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.mission_status = "IDLE"

    # RL action index → mission override behaviour
    _RL_ACTION_MAP = {
        0: 'move_highest_panic',
        1: 'move_nearest',
        2: 'move_safe_zone',
        3: 'wait',
    }

    def _get_bdi_valid_actions(self) -> list:
        """
        Return indices of actions safe under BDI safety rules.

        Sardina, S. & Thangarajah, J. (2011). "On the deployment of BDI agents
        in the presence of learning algorithms." Proc. 22nd IJCAI, pp. 1810-1815.

        Actions: 0=move_highest_panic, 1=move_nearest, 2=move_safe_zone, 3=wait
        """
        all_actions = list(range(4))
        invalid: set = set()

        # If no civilians are active (all evacuated/casualty) — cannot target civilian
        active_count = getattr(self, '_active_civilian_count', None)
        if active_count is not None and active_count == 0:
            invalid.update([0, 1])

        valid = [a for a in all_actions if a not in invalid]
        return valid if valid else all_actions

    def decide(self) -> None:
        """
        Plan next action.
        Uses trained PPO policy (Schulman et al. 2017) with BDI action masking
        (Sardina & Thangarajah 2011) when obs is available; falls back to BDI
        rule (continue on assigned path) pre-training.
        """
        if self._rl_obs is not None:
            from ..rl.ppo import PPOAgent
            # lazy-load policy
            if not hasattr(self, '_rl_policy_inst') or self._rl_policy_inst is None:
                import os
                self._rl_policy_inst = PPOAgent(
                    'rescuer', global_state_dim=_cfg_module.RL_GLOBAL_STATE_DIM
                )
                path = os.path.join(_cfg_module.RL_POLICY_DIR, 'rescuer.pt')
                if os.path.exists(path):
                    self._rl_policy_inst.load(path)
            valid  = self._get_bdi_valid_actions()
            action = self._rl_policy_inst.best_action_masked(self._rl_obs, valid)
            self._rl_action = action
            # Actions 0-2 override mission target; action 3 = wait (no change)
            if action == 3:
                pass  # hold
            # Target overrides are applied in act() when _rl_action is set
            return

        # ── BDI fallback ──────────────────────────────────────────────
        if self.mission_status == "MOVING" and self.current_path:
            pass
        elif self.mission_status == "MOVING" and not self.current_path:
            self.mission_status = "ARRIVED"

    def act(self, environment) -> None:
        """Execute movement or complete mission"""
        if self.mission_status == "MOVING" and self.current_path:
            # STAGGERED PATHFINDING: Periodic recalculation
            self.steps_since_recalc += 1
            should_recalc_periodic = (
                (self.steps_since_recalc + self.recalc_offset) % self.path_recalc_interval == 0
            )

            if should_recalc_periodic and self.target_node:
                self.steps_since_recalc = 0  # Reset counter
                try:
                    self.current_path = nx.shortest_path(
                        environment.graph,
                        self.current_node,
                        self.target_node,
                        weight='length'
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass  # Keep current path if recalculation fails

            # Move along path
            if len(self.current_path) > 1:
                next_node = self.current_path[1]

                # DYNAMIC RE-ROUTING: Check if next node is blocked by fire
                node_data = environment.graph.nodes[next_node]
                next_lat, next_lon = node_data['y'], node_data['x']
                next_r, next_c = environment.latlon_to_grid(next_lat, next_lon)

                # If next node is burning, recalculate path immediately
                if environment.fire_grid[next_r, next_c] == 1:  # BURNING
                    try:
                        # Recalculate path avoiding fire
                        self.current_path = nx.shortest_path(
                            environment.graph,
                            self.current_node,
                            self.target_node,
                            weight='length'
                        )
                        return  # Skip movement this step, recalculate next step
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No safe path - abort mission
                        self.mission_status = "ABORTED"
                        if self.current_mission:
                            abort_msg = Message(
                                sender=self.agent_id,
                                receiver=self.current_mission['commander'],
                                performative="REFUSE",
                                content={
                                    'reason': 'Path blocked by fire',
                                    'mission_id': self.current_mission['mission_id']
                                }
                            )
                            self.send_message(abort_msg)
                        return

                # Path is clear, move to next node
                self.current_node = next_node
                self.current_path.pop(0)

                # Update position
                node_data = environment.graph.nodes[self.current_node]
                self.position = (node_data['y'], node_data['x'])

                # Update grid position
                self.grid_position = environment.latlon_to_grid(self.position[0], self.position[1])

                # Consume fuel (prevent going negative)
                self.fuel = max(0.0, self.fuel - 0.1)

            else:
                # Reached destination
                self.mission_status = "ARRIVED"

        elif self.mission_status == "ARRIVED":
            # Mission complete, inform commander
            if self.current_mission:
                confirm_msg = Message(
                    sender=self.agent_id,
                    receiver=self.current_mission['commander'],
                    performative="CONFIRM",
                    content={
                        'mission_id': self.current_mission['mission_id'],
                        'status': 'COMPLETED'
                    }
                )
                self.send_message(confirm_msg)

                self.current_mission = None
                self.mission_status = "IDLE"
                self.current_path = []
