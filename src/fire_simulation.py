"""
Fire Spreading Model with Dynamic Wind
Implements Rothermel-based spread with wind direction that changes over time
"""
import numpy as np
from typing import Tuple
from .config import (
    FIRE_SPREAD_PROB_BASE,
    FIRE_BURNOUT_PROB,
    ROTHERMEL_BASE_ROS,
    ROTHERMEL_WIND_C,
    ROTHERMEL_WIND_B,
    ROTHERMEL_SLOPE_FACTOR,
    WIND_INITIAL_DIRECTION,
    WIND_OSCILLATION_PERIOD,
    WIND_OSCILLATION_AMPLITUDE,
    WIND_SPEED,
    LOG_WIND_CHANGES
)


class FireSimulation:
    """
    Cellular Automata fire spread model with dynamic wind.

    States:
    - 0: No fuel
    - 1: Burning
    - 2: Burnt out
    - 3: Fuel (unburnt)

    Wind Model: θ(t) = θ_0 + sin(t/50) × 20°
    """

    def __init__(self, environment):
        self.environment = environment
        self.wind_speed = WIND_SPEED

        # Dynamic wind parameters
        self.wind_direction_degrees = WIND_INITIAL_DIRECTION
        self.wind_initial_direction = WIND_INITIAL_DIRECTION
        self.wind_oscillation_period = WIND_OSCILLATION_PERIOD
        self.wind_oscillation_amplitude = WIND_OSCILLATION_AMPLITUDE

        # Wind direction as vector (will be updated each step)
        self._update_wind_vector()

        self.last_wind_log_step = 0

    def _update_wind_vector(self):
        """
        Update wind direction vector from current degrees.
        Wind direction follows: θ(t) = θ_0 + sin(t/50) × 20°
        """
        # Calculate current wind direction
        if self.environment.step_count == 0:
            self.wind_direction_degrees = self.wind_initial_direction
        else:
            # Dynamic wind oscillation
            oscillation = np.sin(self.environment.step_count / self.wind_oscillation_period)
            self.wind_direction_degrees = (
                self.wind_initial_direction +
                oscillation * self.wind_oscillation_amplitude
            )

        # Convert to radians
        wind_rad = np.radians(self.wind_direction_degrees)

        # Wind vector (North = 0°, clockwise)
        # Convert to grid coordinates (x, y)
        self.wind_direction = np.array([
            np.sin(wind_rad),  # dx (East component)
            -np.cos(wind_rad)  # dy (North component, negated for grid coords)
        ], dtype=np.float32)

        # Normalize
        wind_magnitude = np.linalg.norm(self.wind_direction)
        if wind_magnitude > 0:
            self.wind_direction = self.wind_direction / wind_magnitude

        # Log significant wind changes
        if LOG_WIND_CHANGES and (self.environment.step_count - self.last_wind_log_step) >= 10:
            print(f"  💨 Step {self.environment.step_count}: "
                  f"Wind Direction = {self.wind_direction_degrees:.1f}°")
            self.last_wind_log_step = self.environment.step_count

    def get_wind_direction_degrees(self) -> float:
        """Get current wind direction in degrees"""
        return self.wind_direction_degrees

    def ignite_random_fires(self, num_fires: int = 3) -> None:
        """Start random fires in fuel areas"""
        fuel_cells = np.argwhere(self.environment.fire_grid == 3)

        if len(fuel_cells) == 0:
            print("⚠️  No fuel cells available to ignite")
            return

        num_fires = min(num_fires, len(fuel_cells))
        fire_indices = np.random.choice(len(fuel_cells), size=num_fires, replace=False)

        for idx in fire_indices:
            row, col = fuel_cells[idx]
            self.environment.fire_grid[row, col] = 1
            print(f"🔥 Fire ignited at grid position ({row}, {col})")

    def step(self) -> None:
        """
        Execute one step of fire propagation with dynamic wind.

        Logic:
        - Update wind direction (oscillates over time)
        - Burning cells spread to neighboring fuel cells
        - Probability affected by: wind, slope, burning neighbors
        - Burning cells eventually burn out
        """
        # Update wind direction (dynamic)
        self._update_wind_vector()

        fire_grid = self.environment.fire_grid
        new_fire_grid = fire_grid.copy()

        rows, cols = fire_grid.shape

        # Find all burning cells
        burning_cells = np.argwhere(fire_grid == 1)

        for row, col in burning_cells:
            # Burn out with probability
            if np.random.random() < FIRE_BURNOUT_PROB:
                new_fire_grid[row, col] = 2
                continue

            # Spread to neighbors (Moore neighborhood)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue

                    nr, nc = row + dr, col + dc

                    # Check bounds
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue

                    # Can only spread to fuel cells
                    if fire_grid[nr, nc] != 3:
                        continue

                    # Calculate spread probability
                    spread_prob = self._calculate_spread_probability(
                        row, col, nr, nc, dr, dc
                    )

                    # Spread fire
                    if np.random.random() < spread_prob:
                        new_fire_grid[nr, nc] = 1

        self.environment.fire_grid = new_fire_grid
        self.environment.step_count += 1

        # Update temperature grid
        self._update_temperature_grid()

    def _calculate_spread_probability(self, from_row: int, from_col: int,
                                     to_row: int, to_col: int,
                                     dr: int, dc: int) -> float:
        """
        Calculate fire spread probability using Rothermel model.

        Factors:
        - Base probability
        - Wind direction and speed (dynamic)
        - Slope (from Perlin terrain)
        - Number of burning neighbors
        """
        base_prob = FIRE_SPREAD_PROB_BASE

        # Wind effect: fire spreads faster in wind direction
        spread_direction = np.array([dc, dr], dtype=np.float32)
        spread_mag = np.linalg.norm(spread_direction)
        if spread_mag > 0:
            spread_direction = spread_direction / spread_mag

            # Dot product: positive if fire spreads with wind
            wind_alignment = np.dot(spread_direction, self.wind_direction)
            # phi_wind = C * U^B
            phi_wind = ROTHERMEL_WIND_C * (self.wind_speed ** ROTHERMEL_WIND_B)
            wind_factor = 1.0 + phi_wind * max(0, wind_alignment)
        else:
            wind_factor = 1.0

        # Slope effect: fire spreads faster uphill (Rothermel)
        elevation_from = self.environment.elevation_grid[from_row, from_col]
        elevation_to = self.environment.elevation_grid[to_row, to_col]

        # Calculate slope angle
        distance = spread_mag  # grid cells
        if distance > 0:
            height_diff = elevation_to - elevation_from
            slope_angle = np.arctan2(height_diff, distance)

            # phi_slope = 5.275 * tan^2(slope)
            if slope_angle > 0:  # Uphill
                phi_slope = ROTHERMEL_SLOPE_FACTOR * (np.tan(slope_angle) ** 2)
                slope_multiplier = 1.0 + phi_slope
            else:  # Downhill
                slope_multiplier = 1.0
        else:
            slope_multiplier = 1.0

        # Neighbor effect: more burning neighbors = higher spread probability
        burning_neighbors = self._count_burning_neighbors(to_row, to_col)
        neighbor_factor = 1.0 + (burning_neighbors * 0.1)

        # Combined probability (Rothermel-based)
        final_prob = base_prob * wind_factor * slope_multiplier * neighbor_factor

        return min(1.0, final_prob)

    def _count_burning_neighbors(self, row: int, col: int) -> int:
        """Count number of burning neighbors around a cell"""
        count = 0
        fire_grid = self.environment.fire_grid

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nr, nc = row + dr, col + dc

                if 0 <= nr < fire_grid.shape[0] and 0 <= nc < fire_grid.shape[1]:
                    if fire_grid[nr, nc] == 1:
                        count += 1

        return count

    def _update_temperature_grid(self) -> None:
        """Update temperature grid based on fire state"""
        fire_grid = self.environment.fire_grid
        temp_grid = self.environment.temperature_grid

        # Burning cells have high temperature
        temp_grid[fire_grid == 1] = 100.0

        # Burnt cells cool down
        temp_grid[fire_grid == 2] = np.maximum(0, temp_grid[fire_grid == 2] - 5.0)

        # Fuel cells have ambient temperature
        temp_grid[fire_grid == 3] = 20.0

        # No fuel cells have ambient temperature
        temp_grid[fire_grid == 0] = 20.0

    def get_fire_statistics(self) -> dict:
        """Get current fire statistics"""
        fire_grid = self.environment.fire_grid

        return {
            'burning_cells': int(np.sum(fire_grid == 1)),
            'burnt_cells': int(np.sum(fire_grid == 2)),
            'fuel_cells': int(np.sum(fire_grid == 3)),
            'total_affected': int(np.sum((fire_grid == 1) | (fire_grid == 2))),
            'wind_direction': self.wind_direction_degrees
        }
