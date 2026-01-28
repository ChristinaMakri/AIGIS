"""
Rescuer Agent - Goal-Based / Practical Reasoning Architecture
Executes rescue missions using Contract Net Protocol with risk-adjusted bidding
Implements safety protocol to refuse dangerous missions
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict
from .base_agent import Agent
from ..message import Message
from ..config import (
    RESCUER_MAX_SPEED,
    RESCUER_FUEL_CAPACITY,
    RESCUER_RISK_ALPHA,
    RESCUER_SAFETY_THRESHOLD,
    RESCUER_PATH_RECALC_INTERVAL
)


class RescuerAgent(Agent):
    """
    Goal-based agent that responds to rescue missions.

    Architecture: Practical Reasoning with Risk Assessment
    - Participates in Contract Net Protocol as bidder
    - Assesses path risk by scanning fire/temperature grid
    - Uses A* pathfinding for navigation
    - Refuses missions through active fire (safety protocol)
    - Manages resources (fuel)
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
        # Recalculate A* path periodically (not every step) to reduce CPU load
        # Random offset prevents all rescuers from recalculating simultaneously
        self.path_recalc_interval = RESCUER_PATH_RECALC_INTERVAL  # Steps between recalcs
        self.steps_since_recalc = 0  # Counter
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)  # Random phase

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

            # Send proposal
            proposal = Message(
                sender=self.agent_id,
                receiver=message.sender,
                performative="PROPOSE",
                content={
                    'cost': total_cost,
                    'eta': eta,
                    'path_risk': path_risk,
                    'target': target_location
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

    def decide(self) -> None:
        """Plan next action based on current mission"""
        if self.mission_status == "MOVING" and self.current_path:
            # Continue on path
            pass
        elif self.mission_status == "MOVING" and not self.current_path:
            # Arrived at destination
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
