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
        """
        Initialize a civilian agent with BDI architecture and panic psychology.

        Key Features:
        - Greenshields traffic model: Speed decreases with crowd density
        - 3-state cognitive machine: Rational → Confused → Herding
        - Panic equation: Increases with fire proximity and family separation
        - Social force herding: Follows crowd at high panic levels

        Args:
            agent_id: Unique identifier for the agent
            position: Initial (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # ===== GREENSHIELDS TRAFFIC MODEL PARAMETERS =====
        # V_current = V_free × (1 - ρ_local / ρ_jam)
        # When density reaches jam density, speed → 0 (gridlock)
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

        # ===== NAVIGATION STATE =====
        # Uses A* pathfinding on OpenStreetMap road network
        self.current_node: Optional[int] = None  # Current graph node
        self.safety_node: Optional[int] = None  # Target safe zone node
        self.current_path: List[int] = []  # Planned path (list of node IDs)
        self.current_edge_density = 0.0  # Local agent density on current edge (for Greenshields)
        self.evacuation_ordered = False  # Commander ordered evacuation
        self.redirect_to_coast = False  # Shelter-in-place order (Phase 3)
        self.is_evacuated = False  # True once civilian reaches a safe zone

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
        # Only recalculate A* path every N=20 steps + random offset
        # Prevents all agents from recalculating simultaneously (CPU spike)
        # Random offset spreads computational load across steps
        self.path_recalc_interval = CIVILIAN_PATH_RECALC_INTERVAL  # Steps between recalcs
        self.steps_since_recalc = 0  # Counter for tracking when to recalculate
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)  # Random phase offset

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
        # ===== PROCESS MESSAGES (Commander → Civilian Communication) =====
        for message in self.messages_inbox:
            # Pre-Evacuation Warning (Phase 1): Commander alerts of approaching fire
            if message.performative == "INFORM" and message.content.get('type') == 'WARNING':
                self.beliefs.add('warning_received')
                self.panic_level = min(1.0, self.panic_level + 0.1)  # Small panic increase

            # Evacuation Orders (Phase 2): Commander orders mass evacuation
            elif message.performative == "REQUEST":
                msg_type = message.content.get('type')
                if msg_type == 'EVACUATE':
                    self.beliefs.add('evacuation_ordered')
                    self.evacuation_ordered = True
                    self.panic_level = min(1.0, self.panic_level + 0.3)  # Significant panic increase
                    self.family_separated = False  # Mass evacuation: family assumed moving together

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

        # Assess local traffic density (Greenshields model)
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

        Components:
        - α × (1/d_fire): Inverse relationship with fire distance
          - Closer fire causes exponential panic increase
          - Example: d=1 → +0.05, d=2 → +0.025, d=10 → +0.005
        - β × (family_separated): Constant penalty if family is separated
          - Adds psychological stress factor
        - decay: Gradual panic reduction when no fire visible
          - Simulates calming down over time

        Panic Range: [0.0, 1.0]
        - 0.0: Completely calm
        - 0.4: Rational threshold (below = optimal decision making)
        - 0.7: Confused threshold (below = degraded decision making)
        - 0.8+: Herding threshold (follows crowd, abandons planning)

        Reference: Disaster psychology research shows proximity to threat
        and family concerns are primary panic drivers.
        """
        if self.fire_visible and self.fire_distance < float('inf'):
            # ===== FIRE PROXIMITY FACTOR =====
            # Inverse distance: closer fire = higher panic
            # max(distance, 0.5) prevents division by very small numbers
            panic_increase = CIVILIAN_PANIC_ALPHA * (1.0 / max(self.fire_distance, 0.5))
            self.panic_level = min(1.0, self.panic_level + panic_increase)
        else:
            # ===== PANIC DECAY =====
            # When no fire visible, panic gradually decreases
            # Simulates calming down when threat is not immediate
            self.panic_level = max(0.0, self.panic_level - CIVILIAN_PANIC_DECAY)

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
        - Considers traffic conditions (Greenshields model)
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
        if self.panic_level < CIVILIAN_PANIC_RATIONAL:
            self.cognitive_state = "rational"  # Below 0.4: optimal behavior
        elif self.panic_level < CIVILIAN_PANIC_CONFUSED:
            self.cognitive_state = "confused"  # 0.4-0.7: degraded performance
        else:
            self.cognitive_state = "herding"  # Above 0.7: follows crowd

    def _assess_local_density(self, environment) -> None:
        """
        Assess local agent density for Greenshields' traffic model.
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

        # Calculate inverse-distance-weighted average movement direction.
        # Agents closer to self have stronger influence (realistic social force model).
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
