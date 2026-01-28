"""
Fire Spreading Model with Dynamic Wind
Implements Rothermel-based spread with wind direction that changes over time
Uses vectorized operations with scipy.signal.convolve2d for performance
"""
import numpy as np
from scipy import signal
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

        VECTORIZED IMPLEMENTATION using scipy.signal.convolve2d
        - No for loops over cells (O(1) operations on entire grid)
        - Uses convolution kernels for neighbor counting
        - Processes all 8 spread directions with numpy array operations
        """
        # Update wind direction (dynamic)
        self._update_wind_vector()

        fire_grid = self.environment.fire_grid
        new_fire_grid = fire_grid.copy()
        rows, cols = fire_grid.shape

        # ===== VECTORIZED BURNOUT =====
        burning_mask = (fire_grid == 1)
        burnout_random = np.random.random((rows, cols))
        burnout_mask = burning_mask & (burnout_random < FIRE_BURNOUT_PROB)
        new_fire_grid[burnout_mask] = 2

        # Cells still burning after burnout
        still_burning = (fire_grid == 1) & (new_fire_grid != 2)

        # ===== VECTORIZED NEIGHBOR COUNTING =====
        # Count burning neighbors using convolution
        neighbor_kernel = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=np.float32)

        burning_neighbors_count = signal.convolve2d(
            still_burning.astype(np.float32),
            neighbor_kernel,
            mode='same',
            boundary='fill',
            fillvalue=0
        )

        # ===== VECTORIZED SPREAD FOR 8 DIRECTIONS =====
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dr, dc in directions:
            # Calculate spread probabilities for this direction (vectorized)
            spread_probs = self._calculate_directional_spread_vectorized(
                dr, dc, still_burning, burning_neighbors_count
            )

            # Valid targets: fuel cells not yet ignited
            target_mask = (fire_grid == 3) & (new_fire_grid == 3)

            # Stochastic spread
            spread_random = np.random.random((rows, cols))
            spread_mask = target_mask & (spread_random < spread_probs)

            new_fire_grid[spread_mask] = 1

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
        """
        Count number of burning neighbors around a cell
        NOTE: This method is kept for compatibility but is not used in vectorized step()
        """
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

    def _calculate_directional_spread_vectorized(self, dr: int, dc: int,
                                                  burning_mask: np.ndarray,
                                                  neighbor_counts: np.ndarray) -> np.ndarray:
        """
        VECTORIZED: Calculate spread probability for a specific direction across entire grid.

        Args:
            dr, dc: Direction offset (-1, 0, or 1)
            burning_mask: Boolean array of cells that are burning
            neighbor_counts: Array of burning neighbor counts per cell

        Returns:
            Array of spread probabilities for each cell
        """
        rows, cols = burning_mask.shape

        # Shift burning mask to align sources with targets
        shifted_burning = np.zeros_like(burning_mask, dtype=bool)

        # Calculate slice indices for shift operation
        if dr < 0:
            r_src, r_tgt = slice(-dr, None), slice(None, dr)
        elif dr > 0:
            r_src, r_tgt = slice(None, -dr), slice(dr, None)
        else:
            r_src, r_tgt = slice(None), slice(None)

        if dc < 0:
            c_src, c_tgt = slice(-dc, None), slice(None, dc)
        elif dc > 0:
            c_src, c_tgt = slice(None, -dc), slice(dc, None)
        else:
            c_src, c_tgt = slice(None), slice(None)

        # Shift burning mask (align sources with their targets)
        shifted_burning[r_tgt, c_tgt] = burning_mask[r_src, c_src]

        # Base probability
        base_prob = FIRE_SPREAD_PROB_BASE

        # Wind factor (constant for this direction)
        spread_direction = np.array([dc, dr], dtype=np.float32)
        spread_mag = np.linalg.norm(spread_direction)
        if spread_mag > 0:
            spread_direction = spread_direction / spread_mag
            wind_alignment = np.dot(spread_direction, self.wind_direction)
            phi_wind = ROTHERMEL_WIND_C * (self.wind_speed ** ROTHERMEL_WIND_B)
            wind_factor = 1.0 + phi_wind * max(0, wind_alignment)
        else:
            wind_factor = 1.0

        # Slope factor (vectorized across entire grid)
        slope_factor = self._calculate_slope_factor_vectorized(dr, dc)

        # Neighbor factor (vectorized)
        neighbor_factor = 1.0 + (neighbor_counts * 0.1)

        # Combined probability
        spread_probs = base_prob * wind_factor * slope_factor * neighbor_factor

        # Clip to [0, 1] and apply only where there's a burning source
        spread_probs = np.clip(spread_probs, 0, 1) * shifted_burning.astype(np.float32)

        return spread_probs

    def _calculate_slope_factor_vectorized(self, dr: int, dc: int) -> np.ndarray:
        """
        VECTORIZED: Calculate slope factor for entire grid in a specific direction.

        Args:
            dr, dc: Direction offset

        Returns:
            Array of slope multipliers (1.0 + phi_slope for uphill, 1.0 for downhill)
        """
        elevation_grid = self.environment.elevation_grid
        rows, cols = elevation_grid.shape

        # Shift elevation grid to get target elevations
        shifted_elevation = np.zeros_like(elevation_grid)

        if dr < 0:
            r_src, r_tgt = slice(-dr, None), slice(None, dr)
        elif dr > 0:
            r_src, r_tgt = slice(None, -dr), slice(dr, None)
        else:
            r_src, r_tgt = slice(None), slice(None)

        if dc < 0:
            c_src, c_tgt = slice(-dc, None), slice(None, dc)
        elif dc > 0:
            c_src, c_tgt = slice(None, -dc), slice(dc, None)
        else:
            c_src, c_tgt = slice(None), slice(None)

        shifted_elevation[r_tgt, c_tgt] = elevation_grid[r_src, c_src]

        # Height difference
        height_diff = shifted_elevation - elevation_grid

        # Distance (constant for this direction)
        distance = np.sqrt(dr**2 + dc**2)

        # Slope angle
        slope_angle = np.arctan2(height_diff, distance)

        # Slope factor (only apply for uphill)
        slope_factor = np.ones_like(slope_angle)
        uphill_mask = (slope_angle > 0)
        phi_slope = ROTHERMEL_SLOPE_FACTOR * (np.tan(slope_angle[uphill_mask]) ** 2)
        slope_factor[uphill_mask] = 1.0 + phi_slope

        return slope_factor

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
