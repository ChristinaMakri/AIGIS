"""
Civilian Agent - BDI (Belief-Desire-Intention) Architecture
Implements Greenshields' Traffic Model + Social Force Model (Herding)
Features 3-state cognitive machine: Rational, Confused, Herding
Panic equation with fire distance and family separation factors
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
    CIVILIAN_PANIC_HERDING,
    CIVILIAN_PANIC_ALPHA,
    CIVILIAN_PANIC_BETA,
    CIVILIAN_PANIC_DECAY,
    CIVILIAN_CONFUSED_SPEED_FACTOR,
    CIVILIAN_VISION_RADIUS,
    CIVILIAN_HERDING_INFLUENCE,
    CIVILIAN_PATH_RECALC_INTERVAL
)


class CivilianAgent(Agent):
    """
    BDI agent that evacuates using traffic physics and panic psychology.

    Architecture: Belief-Desire-Intention with Crowd Dynamics
    - Movement: Greenshields' Traffic Model (speed depends on local density)
    - Cognition: 3-state machine (Rational, Confused, Herding)
    - Panic: Distance-based equation with family separation factor
    - Herding: Follows crowd at high panic, even to dead ends
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        super().__init__(agent_id, position)

        # Traffic Model Parameters
        self.v_free_flow = CIVILIAN_V_FREE_FLOW
        self.rho_jam = CIVILIAN_RHO_JAM
        self.current_speed = self.v_free_flow

        # BDI Components
        self.beliefs: Set[str] = set()  # Known facts
        self.desires: List[str] = ['survive', 'reach_safety']
        self.intentions: List[str] = []  # Current plan

        # Panic Model (Enhanced)
        self.panic_level = 0.0  # 0.0 to 1.0
        self.fire_visible = False
        self.fire_distance = float('inf')  # Distance to nearest fire
        self.cognitive_state = "rational"  # rational, confused, herding

        # Navigation
        self.current_node: Optional[int] = None
        self.safety_node: Optional[int] = None
        self.current_path: List[int] = []
        self.current_edge_density = 0.0  # Local density on current edge
        self.evacuation_ordered = False
        self.redirect_to_coast = False

        # Social bonds
        self.has_family = np.random.random() < 0.3
        self.family_separated = self.has_family  # Assume separated at start

        # Vision and herding (Social Force Model)
        self.vision_radius = CIVILIAN_VISION_RADIUS
        self.nearby_agents = []  # For herding behavior
        self.last_movement = None  # Track movement direction for others to follow

        # Performance Optimization: Staggered Pathfinding
        self.path_recalc_interval = CIVILIAN_PATH_RECALC_INTERVAL
        self.steps_since_recalc = 0
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)  # Random offset

    def perceive(self, environment) -> None:
        """
        Update beliefs based on perceptions.
        Calculate panic using distance-based equation.
        """
        # Update beliefs from messages
        for message in self.messages_inbox:
            if message.performative == "INFORM" and message.content.get('type') == 'WARNING':
                self.beliefs.add('warning_received')
                self.panic_level = min(1.0, self.panic_level + 0.1)

            elif message.performative == "REQUEST":
                msg_type = message.content.get('type')
                if msg_type == 'EVACUATE':
                    self.beliefs.add('evacuation_ordered')
                    self.evacuation_ordered = True
                    self.panic_level = min(1.0, self.panic_level + 0.3)

                elif msg_type == 'REDIRECT_TO_SAFE_ZONE':
                    self.beliefs.add('shelter_in_place')
                    self.redirect_to_coast = True  # Flag to re-route
                    self.panic_level = min(1.0, self.panic_level + 0.5)  # High panic!

        # Check for visible fire and calculate distance
        self._check_fire_visibility_and_distance(environment)

        # Update panic using the panic equation
        self._update_panic()

        # Determine cognitive state based on panic level
        self._update_cognitive_state()

        # Assess local traffic density (Greenshields model)
        self._assess_local_density(environment)

        # Note: nearby_agents will be populated by simulation
        # (needs access to all civilian agents)

    def _check_fire_visibility_and_distance(self, environment) -> None:
        """
        Check if fire is visible and calculate distance to nearest fire.
        Used in panic equation.
        """
        self.fire_visible = False
        self.fire_distance = float('inf')

        if self.grid_position is None:
            return

        row, col = self.grid_position
        fire_grid = environment.fire_grid

        # Scan within vision radius
        for dr in range(-self.vision_radius, self.vision_radius + 1):
            for dc in range(-self.vision_radius, self.vision_radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < fire_grid.shape[0] and 0 <= c < fire_grid.shape[1]:
                    if fire_grid[r, c] == 1:  # Burning
                        self.fire_visible = True
                        dist = np.sqrt(dr**2 + dc**2)
                        if dist < self.fire_distance:
                            self.fire_distance = dist

    def _update_panic(self) -> None:
        """
        Update panic level using the panic equation.
        Panic(t) = Panic(t-1) + α * (1/d_fire) + β * (Family_Separated?)
        Decays slowly when fire is not visible.
        """
        if self.fire_visible and self.fire_distance < float('inf'):
            # Increase panic based on fire proximity
            panic_increase = CIVILIAN_PANIC_ALPHA * (1.0 / max(self.fire_distance, 0.5))
            self.panic_level = min(1.0, self.panic_level + panic_increase)
        else:
            # Decay panic when no fire visible
            self.panic_level = max(0.0, self.panic_level - CIVILIAN_PANIC_DECAY)

        # Family separation factor
        if self.family_separated:
            self.panic_level = min(1.0, self.panic_level + CIVILIAN_PANIC_BETA * 0.1)

    def _update_cognitive_state(self) -> None:
        """
        Determine cognitive state based on panic level.
        State 1: Rational (Panic < 0.4)
        State 2: Confused (0.4 <= Panic < 0.7)
        State 3: Herding (Panic >= 0.7)
        """
        if self.panic_level < CIVILIAN_PANIC_RATIONAL:
            self.cognitive_state = "rational"
        elif self.panic_level < CIVILIAN_PANIC_CONFUSED:
            self.cognitive_state = "confused"
        else:
            self.cognitive_state = "herding"

    def _assess_local_density(self, environment) -> None:
        """
        Assess local agent density for Greenshields' traffic model.
        This is a simplified version - in full implementation, count agents on current edge.
        """
        # Simplified: random density for now
        # In real implementation: count agents within local radius
        self.current_edge_density = np.random.uniform(0, self.rho_jam)

    def _find_nearby_agents(self, all_agents: list) -> None:
        """
        Find nearby agents for Social Force herding behavior.
        Scans within vision radius and stores agents moving away from fire.
        """
        self.nearby_agents = []

        if self.grid_position is None:
            return

        my_row, my_col = self.grid_position

        # Scan all other civilians
        for agent in all_agents:
            if agent.agent_id == self.agent_id:
                continue

            if not hasattr(agent, 'grid_position') or agent.grid_position is None:
                continue

            other_row, other_col = agent.grid_position

            # Calculate distance
            distance = np.sqrt((other_row - my_row)**2 + (other_col - my_col)**2)

            if distance <= self.vision_radius:
                self.nearby_agents.append(agent)

    def decide(self) -> None:
        """
        BDI reasoning cycle with 3-state cognitive machine.
        Cognitive state determines decision quality.
        """
        # Intention revision based on beliefs
        if 'evacuation_ordered' in self.beliefs or 'fire_nearby' in self.beliefs or 'shelter_in_place' in self.beliefs:
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

    def _calculate_speed_greenshields(self) -> float:
        """
        Calculate current speed using Greenshields' Traffic Model.
        V_current = V_free_flow * (1 - ρ_local / ρ_jam)
        """
        if self.current_edge_density >= self.rho_jam:
            # Gridlock!
            return 0.0

        speed = self.v_free_flow * (1 - (self.current_edge_density / self.rho_jam))

        # Additional speed reduction in confused state
        if self.cognitive_state == "confused":
            speed *= CIVILIAN_CONFUSED_SPEED_FACTOR

        return max(0, speed)

    def act(self, environment) -> None:
        """
        Execute intentions using Greenshields' traffic model.
        Speed depends on local density and cognitive state.
        """
        if not self.intentions:
            return

        # Calculate current speed based on traffic density
        self.current_speed = self._calculate_speed_greenshields()

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
            if self.current_speed > 0.1:  # Need minimum speed to move
                self._move_to_safety(environment)
            # else: gridlock - cannot move

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

        # Check if not obstacle or fire
        if environment.obstacle_grid[new_row, new_col] == 0 and environment.fire_grid[new_row, new_col] != 1:
            self.grid_position = (new_row, new_col)
            self.position = environment.grid_to_latlon(new_row, new_col)

    def _follow_crowd(self, environment) -> None:
        """
        Social Force Model: Move in the average direction of nearby agents.
        This simulates herding behavior - following the crowd even to dead ends.
        """
        if self.grid_position is None or not self.nearby_agents:
            # Fallback to random movement if no crowd
            self._move_random(environment)
            return

        # Calculate average movement direction from nearby agents
        avg_direction = np.zeros(2, dtype=np.float32)
        valid_agents = 0

        for agent in self.nearby_agents:
            if hasattr(agent, 'last_movement') and agent.last_movement is not None:
                avg_direction += agent.last_movement
                valid_agents += 1

        if valid_agents == 0 or np.linalg.norm(avg_direction) == 0:
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
        if (environment.obstacle_grid[new_row, new_col] == 0 and
            environment.fire_grid[new_row, new_col] != 1):
            # Track last movement for other agents to follow
            self.last_movement = np.array([new_col - col, new_row - row], dtype=np.float32)
            self.grid_position = (new_row, new_col)
            self.position = environment.grid_to_latlon(new_row, new_col)
        else:
            # Blocked, try random movement
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

            # STAGGERED PATHFINDING: Check if periodic recalculation is needed
            # Recalculate every N steps (with random offset) to adjust for traffic density
            self.steps_since_recalc += 1
            should_recalc_periodic = (
                (self.steps_since_recalc + self.recalc_offset) % self.path_recalc_interval == 0
            )

            # Calculate path if needed
            # Reasons: 1) No path yet, 2) Periodic recalculation
            if (not self.current_path or should_recalc_periodic) and self.safety_node:
                if should_recalc_periodic:
                    self.steps_since_recalc = 0  # Reset counter

                try:
                    self.current_path = nx.shortest_path(
                        environment.graph,
                        self.current_node,
                        self.safety_node,
                        weight='length'
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Try finding alternative safe node
                    self.safety_node = environment.find_nearest_safe_node(self.current_node)
                    self.current_path = []

            # Move along path using Greenshields' speed (already calculated)
            if self.current_path and len(self.current_path) > 1 and self.current_speed > 0.1:
                next_node = self.current_path[1]

                # DYNAMIC RE-ROUTING: Check if next node is blocked by fire
                node_data = environment.graph.nodes[next_node]
                next_lat, next_lon = node_data['y'], node_data['x']
                next_r, next_c = environment.latlon_to_grid(next_lat, next_lon)

                # If next node is burning, recalculate path immediately
                if environment.fire_grid[next_r, next_c] == 1:  # BURNING
                    self.current_path = []  # Clear stale path
                    try:
                        # Find new path avoiding fire
                        self.current_path = nx.shortest_path(
                            environment.graph,
                            self.current_node,
                            self.safety_node,
                            weight='length'
                        )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # If no path exists, try new safe zone
                        self.safety_node = environment.find_nearest_safe_node(self.current_node)
                        self.current_path = []
                    return  # Skip movement this step, recalculate next step

                # Path is clear, move to next node
                old_node = self.current_node
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
            # Fallback to random movement if pathfinding fails
            if self.current_speed > 0.5:
                self._move_random(environment)
