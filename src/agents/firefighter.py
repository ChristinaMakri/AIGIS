"""
Firefighter Agent - Utility-Based Architecture
Actively suppresses fires through water drops and creating fire lines

Architecture: Utility-Based Agent with Resource Management
- Assesses fire suppression opportunities
- Evaluates utility of different suppression strategies
- Manages water/retardant resources
- Coordinates with other firefighters
"""
import numpy as np
from typing import Tuple, List, Optional
from .base_agent import Agent
from ..message import Message


class FirefighterAgent(Agent):
    """
    Utility-based firefighting agent that actively suppresses fires.

    Actions:
    1. Water Drop - Extinguish burning cells
    2. Create Fire Line - Remove fuel in fire's path
    3. Backburn - Controlled burn to remove fuel
    4. Monitor - Watch fire behavior for strategy adjustment

    Resource Management:
    - Water capacity: 5000 gallons
    - Refill time: 10 steps
    - Drop effectiveness: 80% chance to extinguish
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Initialize Firefighter agent.

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # === RESOURCE STATE ===
        self.water_capacity = 5000  # gallons
        self.current_water = 5000  # Start full
        self.water_per_drop = 500  # gallons per drop
        self.refill_time = 10  # steps to refill
        self.is_refilling = False
        self.refill_counter = 0

        # === SUPPRESSION EFFECTIVENESS ===
        self.drop_success_rate = 0.80  # 80% chance to extinguish
        self.fire_line_width = 2  # cells wide
        self.backburn_effectiveness = 0.90

        # === UTILITY WEIGHTS ===
        self.w_threat = 0.5  # Weight for fire threat level
        self.w_efficiency = 0.3  # Weight for water efficiency
        self.w_coordination = 0.2  # Weight for team coordination

        # === MISSION STATE ===
        self.target_fire = None  # (row, col) of target fire cell
        self.suppression_strategy = None  # 'water', 'fire_line', 'backburn'

    def perceive(self, environment) -> None:
        """
        Perceive fire state and coordinate with other firefighters.

        Perception includes:
        - Fire locations and intensity
        - Wind direction (affects suppression strategy)
        - Other firefighter locations (avoid overlap)
        - Population at risk
        """
        # Scan for active fires
        burning_cells = np.argwhere(environment.fire_grid == 1)

        if len(burning_cells) > 0:
            # Find closest burning cell
            distances = [
                np.linalg.norm(np.array(self.grid_position) - np.array(cell))
                for cell in burning_cells
            ]
            closest_idx = np.argmin(distances)
            self.target_fire = tuple(burning_cells[closest_idx])

        # Check messages for coordination
        for message in self.messages_inbox:
            if message.performative == "CFP":
                # Commander is requesting a firefighting mission.
                # Respond with PROPOSE if available, REFUSE if not.
                if not self.is_refilling and self.current_water >= self.water_per_drop:
                    propose = Message(
                        sender=self.agent_id,
                        receiver=message.sender,
                        performative="PROPOSE",
                        content={
                            'cost': 1.0,
                            'eta': 1,
                            'path_risk': 0.0,
                            'target': message.content.get('target_location')
                        },
                        conversation_id=message.conversation_id
                    )
                    self.send_message(propose)
                else:
                    reason = 'refilling' if self.is_refilling else 'no_water'
                    refuse = Message(
                        sender=self.agent_id,
                        receiver=message.sender,
                        performative="REFUSE",
                        content={'reason': reason},
                        conversation_id=message.conversation_id
                    )
                    self.send_message(refuse)

    def decide(self) -> None:
        """
        Choose suppression strategy using utility function.

        Utility = w_threat × Threat + w_efficiency × Efficiency + w_coordination × Coordination

        Strategies:
        1. Water drop: High threat, close proximity
        2. Fire line: Medium threat, predictable spread
        3. Backburn: Low threat, time to prepare
        """
        if self.is_refilling:
            self.suppression_strategy = 'refill'
            return

        if self.current_water < self.water_per_drop:
            self.suppression_strategy = 'return_to_base'
            return

        if self.target_fire is None:
            self.suppression_strategy = 'patrol'
            return

        # Calculate utilities for each strategy
        utilities = {
            'water_drop': self._calculate_water_drop_utility(),
            'fire_line': self._calculate_fire_line_utility(),
            'backburn': self._calculate_backburn_utility()
        }

        # Select best strategy
        self.suppression_strategy = max(utilities, key=utilities.get)

    def _calculate_water_drop_utility(self) -> float:
        """Calculate utility of water drop action"""
        if self.target_fire is None or self.current_water < self.water_per_drop:
            return -float('inf')

        # Threat: Distance to fire
        distance = np.linalg.norm(np.array(self.grid_position) - np.array(self.target_fire))
        threat_score = 100 / (distance + 1)

        # Efficiency: Water remaining / drops available
        efficiency_score = (self.current_water / self.water_per_drop) * 10

        # Coordination: Check if other firefighters targeting same fire
        coordination_score = 50  # Simplified: always moderate

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def _calculate_fire_line_utility(self) -> float:
        """Calculate utility of creating fire line"""
        if self.target_fire is None:
            return -float('inf')

        # Fire line is useful for predictable spread
        threat_score = 60  # Medium threat response
        efficiency_score = 70  # Doesn't use water
        coordination_score = 80  # Good for team coordination

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def _calculate_backburn_utility(self) -> float:
        """Calculate utility of controlled backburn"""
        # Backburn is risky but effective when time permits
        threat_score = 40  # Lower immediate threat
        efficiency_score = 90  # Very efficient
        coordination_score = 60  # Requires careful coordination

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def act(self, environment) -> None:
        """
        Execute suppression action based on chosen strategy.
        """
        # Handle refilling
        if self.is_refilling:
            self.refill_counter += 1
            if self.refill_counter >= self.refill_time:
                self.current_water = self.water_capacity
                self.is_refilling = False
                self.refill_counter = 0
                print(f"  💧 {self.agent_id}: Water refilled")
            return

        # Execute strategy
        if self.suppression_strategy == 'water_drop':
            self._execute_water_drop(environment)
        elif self.suppression_strategy == 'fire_line':
            self._execute_fire_line(environment)
        elif self.suppression_strategy == 'backburn':
            self._execute_backburn(environment)
        elif self.suppression_strategy == 'return_to_base':
            self.is_refilling = True
            print(f"  🚁 {self.agent_id}: Returning to base for refill")
        elif self.suppression_strategy == 'patrol':
            pass  # Monitoring mode

    def _execute_water_drop(self, environment) -> None:
        """Drop water on target fire"""
        if self.target_fire is None or self.current_water < self.water_per_drop:
            return

        row, col = self.target_fire

        # Check if fire still burning
        if environment.fire_grid[row, col] == 1:
            # Use water
            self.current_water -= self.water_per_drop

            # Attempt to extinguish (probabilistic)
            if np.random.random() < self.drop_success_rate:
                # Successfully extinguished
                environment.fire_grid[row, col] = 2  # Burnt out
                environment.temperature_grid[row, col] = 50.0  # Cooling
                print(f"  💦 {self.agent_id}: Extinguished fire at ({row}, {col}) "
                      f"[Water: {self.current_water}/{self.water_capacity}]")
            else:
                # Reduced intensity but not extinguished
                print(f"  💦 {self.agent_id}: Reduced fire intensity at ({row}, {col})")

            # Clear target
            self.target_fire = None

    def _execute_fire_line(self, environment) -> None:
        """Create fire line by removing fuel"""
        if self.target_fire is None:
            return

        fire_row, fire_col = self.target_fire

        # Create fire line perpendicular to fire direction
        # Simplified: remove fuel in a line near the fire
        for offset in range(-self.fire_line_width, self.fire_line_width + 1):
            new_row = fire_row + offset
            if 0 <= new_row < environment.grid_shape[0]:
                if environment.fire_grid[new_row, fire_col] == 3:  # Fuel
                    environment.fire_grid[new_row, fire_col] = 0  # No fuel
                    print(f"  🪓 {self.agent_id}: Created fire line at ({new_row}, {fire_col})")

        self.target_fire = None

    def _execute_backburn(self, environment) -> None:
        """Controlled burn to remove fuel in fire's path"""
        # Simplified implementation
        # In reality, this is a complex and risky operation
        if self.target_fire is None:
            return

        fire_row, fire_col = self.target_fire

        # Burn fuel cells ahead of main fire
        # This removes fuel and creates a barrier
        for dr in [-3, -2, -1]:  # Burn cells ahead
            new_row = fire_row + dr
            if 0 <= new_row < environment.grid_shape[0]:
                if environment.fire_grid[new_row, fire_col] == 3:
                    # Controlled burn
                    environment.fire_grid[new_row, fire_col] = 2  # Burnt
                    print(f"  🔥 {self.agent_id}: Backburn at ({new_row}, {fire_col})")

        self.target_fire = None
