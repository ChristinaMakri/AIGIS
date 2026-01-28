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
        """
        Initialize fire simulation with dynamic wind model.

        Args:
            environment: Environment instance with fire_grid, elevation_grid, etc.
        """
        self.environment = environment
        self.wind_speed = WIND_SPEED  # Base wind speed in m/s

        # Dynamic wind parameters - implements oscillating wind direction
        # This simulates natural wind pattern changes during wildfire events
        self.wind_direction_degrees = WIND_INITIAL_DIRECTION
        self.wind_initial_direction = WIND_INITIAL_DIRECTION
        self.wind_oscillation_period = WIND_OSCILLATION_PERIOD  # Steps for full sine cycle
        self.wind_oscillation_amplitude = WIND_OSCILLATION_AMPLITUDE  # Max deviation in degrees

        # Wind direction as vector (will be updated each step)
        # Stored as normalized 2D vector for efficient dot product calculations
        self._update_wind_vector()

        # Throttle wind logging to every 10 steps to reduce console spam
        self.last_wind_log_step = 0

    def _update_wind_vector(self):
        """
        Update wind direction vector from current degrees.

        Wind direction follows sinusoidal oscillation: θ(t) = θ_0 + sin(t/50) × 20°
        This simulates natural wind pattern changes that affect fire spread dynamics.

        Reference: Real wildfires experience wind shifts due to terrain, temperature,
        and the fire itself creating its own weather patterns (pyro-convection).
        """
        # Calculate current wind direction based on simulation step
        if self.environment.step_count == 0:
            # Initial step: use base wind direction
            self.wind_direction_degrees = self.wind_initial_direction
        else:
            # Dynamic wind oscillation using sine wave
            # Oscillates between (θ_0 - amplitude) and (θ_0 + amplitude)
            oscillation = np.sin(self.environment.step_count / self.wind_oscillation_period)
            self.wind_direction_degrees = (
                self.wind_initial_direction +
                oscillation * self.wind_oscillation_amplitude
            )

        # Convert degrees to radians for trigonometric calculations
        wind_rad = np.radians(self.wind_direction_degrees)

        # Convert compass bearing to grid coordinate system
        # Wind vector (North = 0°, clockwise following compass convention)
        # Grid coordinates: x increases rightward (East), y increases downward (South)
        self.wind_direction = np.array([
            np.sin(wind_rad),   # dx (East component): positive = eastward
            -np.cos(wind_rad)   # dy (North component): negated because y increases downward
        ], dtype=np.float32)

        # Normalize to unit vector for consistent dot product calculations
        # This ensures wind_alignment calculations work correctly regardless of magnitude
        wind_magnitude = np.linalg.norm(self.wind_direction)
        if wind_magnitude > 0:
            self.wind_direction = self.wind_direction / wind_magnitude

        # Log significant wind changes (throttled to every 10 steps)
        # Helps users understand fire behavior changes during simulation
        if LOG_WIND_CHANGES and (self.environment.step_count - self.last_wind_log_step) >= 10:
            print(f"  💨 Step {self.environment.step_count}: "
                  f"Wind Direction = {self.wind_direction_degrees:.1f}°")
            self.last_wind_log_step = self.environment.step_count

    def get_wind_direction_degrees(self) -> float:
        """Get current wind direction in degrees"""
        return self.wind_direction_degrees

    def ignite_random_fires(self, num_fires: int = 3) -> None:
        """
        Start random fires in fuel areas to initiate the simulation.

        Args:
            num_fires: Number of initial fire ignition points

        Note:
            - Only ignites in fuel cells (state = 3), not in water/roads
            - Uses numpy's random choice for uniform random distribution
            - Sets fire_grid cells to state 1 (burning)
        """
        # Find all available fuel cells (state = 3)
        fuel_cells = np.argwhere(self.environment.fire_grid == 3)

        # Safety check: ensure we have fuel to burn
        if len(fuel_cells) == 0:
            print("⚠️  No fuel cells available to ignite")
            return

        # Limit fires to available fuel cells
        num_fires = min(num_fires, len(fuel_cells))

        # Randomly select fuel cells to ignite (without replacement)
        fire_indices = np.random.choice(len(fuel_cells), size=num_fires, replace=False)

        # Ignite selected cells
        for idx in fire_indices:
            row, col = fuel_cells[idx]
            self.environment.fire_grid[row, col] = 1  # State 1 = Burning
            print(f"🔥 Fire ignited at grid position ({row}, {col})")

    def step(self) -> None:
        """
        Execute one step of fire propagation with dynamic wind.

        PERFORMANCE OPTIMIZATION: Vectorized implementation using scipy.signal.convolve2d
        - Eliminates nested for loops over grid cells (10-50× speedup)
        - Uses convolution kernels for neighbor counting (O(1) operation)
        - Processes all 8 spread directions with numpy array operations
        - Handles entire 200×200 grid in parallel

        Algorithm Overview:
        1. Update dynamic wind direction
        2. Process burnout for all burning cells (vectorized)
        3. Count burning neighbors using 2D convolution
        4. Calculate spread probabilities for all 8 directions (vectorized)
        5. Apply stochastic fire spread based on probabilities
        6. Update temperature grid

        Reference: Rothermel (1972) fire spread model with cellular automata
        """
        # ===== STEP 1: UPDATE WIND DIRECTION =====
        # Wind affects fire spread probability via wind_factor in Rothermel equation
        self._update_wind_vector()

        fire_grid = self.environment.fire_grid
        new_fire_grid = fire_grid.copy()  # Work on copy to prevent mid-step conflicts
        rows, cols = fire_grid.shape

        # ===== STEP 2: VECTORIZED BURNOUT =====
        # Burning cells have a probability to burn out and transition to state 2
        # Vectorized: processes all burning cells simultaneously without loops
        burning_mask = (fire_grid == 1)  # Boolean mask of all burning cells
        burnout_random = np.random.random((rows, cols))  # Random values for each cell
        burnout_mask = burning_mask & (burnout_random < FIRE_BURNOUT_PROB)
        new_fire_grid[burnout_mask] = 2  # State 2 = Burnt out

        # Track cells still burning (didn't burn out this step)
        # These are the sources for fire spread
        still_burning = (fire_grid == 1) & (new_fire_grid != 2)

        # ===== STEP 3: VECTORIZED NEIGHBOR COUNTING =====
        # Use 2D convolution to count burning neighbors for each cell
        # Convolution is O(N) for the entire grid vs O(N²) for nested loops
        # Kernel: 3×3 with center = 0 (counts 8-connected neighbors)
        neighbor_kernel = np.array([[1, 1, 1],
                                    [1, 0, 1],  # Center = 0 (don't count self)
                                    [1, 1, 1]], dtype=np.float32)

        # Convolve burning mask with kernel to get neighbor counts
        # mode='same': output has same shape as input
        # boundary='fill', fillvalue=0: edges treated as non-burning
        burning_neighbors_count = signal.convolve2d(
            still_burning.astype(np.float32),
            neighbor_kernel,
            mode='same',
            boundary='fill',
            fillvalue=0
        )

        # ===== STEP 4: VECTORIZED SPREAD FOR 8 DIRECTIONS =====
        # Moore neighborhood: 8 adjacent cells (including diagonals)
        # Process each direction separately to apply directional wind/slope effects
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # North-West, North, North-East
            (0, -1),           (0, 1),    # West, East
            (1, -1),  (1, 0),  (1, 1)     # South-West, South, South-East
        ]

        for dr, dc in directions:
            # Calculate spread probabilities for this direction (vectorized across grid)
            # Accounts for: base probability, wind, slope, neighbors
            spread_probs = self._calculate_directional_spread_vectorized(
                dr, dc, still_burning, burning_neighbors_count
            )

            # Valid targets: fuel cells (state=3) not yet ignited this step
            # We check new_fire_grid to prevent double-ignition in same step
            target_mask = (fire_grid == 3) & (new_fire_grid == 3)

            # Stochastic fire spread: compare random values to probabilities
            # Each cell independently decides whether to ignite
            spread_random = np.random.random((rows, cols))
            spread_mask = target_mask & (spread_random < spread_probs)

            # Ignite cells where spread occurred
            new_fire_grid[spread_mask] = 1  # State 1 = Burning

        # ===== STEP 5: UPDATE ENVIRONMENT STATE =====
        self.environment.fire_grid = new_fire_grid
        self.environment.step_count += 1

        # Update temperature grid for agent perception
        # Temperature affects Sentinel detection and Rescuer risk assessment
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
        VECTORIZED: Calculate fire spread probability for a specific direction across entire grid.

        This method processes all cells simultaneously using array operations instead of loops.
        It shifts the burning mask to align fire sources with their potential targets in the
        specified direction, then calculates Rothermel-based spread probabilities.

        Args:
            dr, dc: Direction offset (-1, 0, or 1) representing one of 8 Moore neighbors
                   Example: dr=-1, dc=0 means spreading North
            burning_mask: Boolean array indicating which cells are currently burning
            neighbor_counts: Array of burning neighbor counts per cell (from convolution)

        Returns:
            Array of spread probabilities [0,1] for each cell in the grid
            Non-zero only where a burning cell exists in the opposite direction

        Algorithm:
        1. Shift burning mask to align sources with targets
        2. Calculate wind factor (Rothermel phi_wind)
        3. Calculate slope factor (Rothermel phi_slope) - vectorized across grid
        4. Calculate neighbor factor (increases probability with more burning neighbors)
        5. Combine factors: P = base × wind × slope × neighbors
        """
        rows, cols = burning_mask.shape

        # ===== STEP 1: SHIFT BURNING MASK =====
        # Align fire sources with their targets in this direction
        # Example: For North spread (dr=-1), shift burning cells down so row i aligns with i-1
        shifted_burning = np.zeros_like(burning_mask, dtype=bool)

        # Calculate slice indices for array shifting
        # This implements: shifted[target_position] = original[source_position]
        if dr < 0:  # Spreading upward (North)
            r_src, r_tgt = slice(-dr, None), slice(None, dr)
        elif dr > 0:  # Spreading downward (South)
            r_src, r_tgt = slice(None, -dr), slice(dr, None)
        else:  # No vertical movement
            r_src, r_tgt = slice(None), slice(None)

        if dc < 0:  # Spreading leftward (West)
            c_src, c_tgt = slice(-dc, None), slice(None, dc)
        elif dc > 0:  # Spreading rightward (East)
            c_src, c_tgt = slice(None, -dc), slice(dc, None)
        else:  # No horizontal movement
            c_src, c_tgt = slice(None), slice(None)

        # Perform the shift: burning sources now align with their targets
        shifted_burning[r_tgt, c_tgt] = burning_mask[r_src, c_src]

        # ===== STEP 2: BASE PROBABILITY =====
        base_prob = FIRE_SPREAD_PROB_BASE

        # ===== STEP 3: WIND FACTOR (ROTHERMEL) =====
        # Fire spreads faster when aligned with wind direction
        # phi_wind = C × U^B, where U = wind speed, C and B are constants
        # Wind factor applied based on alignment: 1.0 + phi_wind × cos(angle)
        spread_direction = np.array([dc, dr], dtype=np.float32)
        spread_mag = np.linalg.norm(spread_direction)
        if spread_mag > 0:
            spread_direction = spread_direction / spread_mag  # Normalize
            wind_alignment = np.dot(spread_direction, self.wind_direction)  # Cosine of angle
            phi_wind = ROTHERMEL_WIND_C * (self.wind_speed ** ROTHERMEL_WIND_B)
            # Only increase probability for wind-aligned spread (max(0, ...))
            wind_factor = 1.0 + phi_wind * max(0, wind_alignment)
        else:
            wind_factor = 1.0  # No movement direction (shouldn't happen)

        # ===== STEP 4: SLOPE FACTOR (ROTHERMEL) =====
        # Fire spreads faster uphill due to heat rising and flame angle
        # Vectorized across entire grid for this direction
        slope_factor = self._calculate_slope_factor_vectorized(dr, dc)

        # ===== STEP 5: NEIGHBOR FACTOR =====
        # More burning neighbors = higher ignition probability (preheating effect)
        # Linear increase: 10% per burning neighbor
        neighbor_factor = 1.0 + (neighbor_counts * 0.1)

        # ===== STEP 6: COMBINED PROBABILITY =====
        # Rothermel multiplicative model: ROS = R_base × (1+φ_wind) × (1+φ_slope)
        # We add neighbor effect as additional multiplier
        spread_probs = base_prob * wind_factor * slope_factor * neighbor_factor

        # Clip to valid probability range [0, 1]
        # Multiply by shifted_burning to zero out cells without burning source
        spread_probs = np.clip(spread_probs, 0, 1) * shifted_burning.astype(np.float32)

        return spread_probs

    def _calculate_slope_factor_vectorized(self, dr: int, dc: int) -> np.ndarray:
        """
        VECTORIZED: Calculate Rothermel slope factor for entire grid in a specific direction.

        Fire spreads faster uphill because:
        1. Heat rises, preheating fuel above
        2. Flames angle toward upslope fuel
        3. Radiation more directly heats upslope vegetation

        Rothermel equation: φ_slope = 5.275 × tan²(slope_angle)
        Applied only for uphill spread (downhill uses factor = 1.0)

        Args:
            dr, dc: Direction offset indicating spread direction

        Returns:
            Array of slope multipliers: (1.0 + phi_slope) for uphill, 1.0 for downhill

        Reference: Rothermel (1972) - "A Mathematical Model for Predicting Fire Spread"
        """
        elevation_grid = self.environment.elevation_grid
        rows, cols = elevation_grid.shape

        # ===== STEP 1: SHIFT ELEVATION GRID =====
        # Align source elevations with target elevations to calculate height difference
        # Same shifting logic as in _calculate_directional_spread_vectorized
        shifted_elevation = np.zeros_like(elevation_grid)

        # Calculate slice indices for shift
        if dr < 0:  # Spreading North (upward in grid)
            r_src, r_tgt = slice(-dr, None), slice(None, dr)
        elif dr > 0:  # Spreading South (downward in grid)
            r_src, r_tgt = slice(None, -dr), slice(dr, None)
        else:  # No vertical movement
            r_src, r_tgt = slice(None), slice(None)

        if dc < 0:  # Spreading West (leftward in grid)
            c_src, c_tgt = slice(-dc, None), slice(None, dc)
        elif dc > 0:  # Spreading East (rightward in grid)
            c_src, c_tgt = slice(None, -dc), slice(dc, None)
        else:  # No horizontal movement
            c_src, c_tgt = slice(None), slice(None)

        # Perform shift to align target elevations
        shifted_elevation[r_tgt, c_tgt] = elevation_grid[r_src, c_src]

        # ===== STEP 2: CALCULATE SLOPE ANGLE =====
        # Height difference: positive = uphill, negative = downhill
        height_diff = shifted_elevation - elevation_grid

        # Distance in grid cells (constant for this direction)
        # Diagonal directions have distance = sqrt(2) ≈ 1.414
        distance = np.sqrt(dr**2 + dc**2)

        # Safety check: if distance is zero (no spread direction), return no slope effect
        if distance == 0:
            return np.ones_like(elevation_grid)

        # Slope angle in radians: arctan(rise/run)
        slope_angle = np.arctan2(height_diff, distance)

        # ===== STEP 3: APPLY ROTHERMEL SLOPE FACTOR =====
        # Start with factor = 1.0 (no effect) for all cells
        slope_factor = np.ones_like(slope_angle)

        # Apply slope effect only for uphill spread (slope_angle > 0)
        uphill_mask = (slope_angle > 0)

        # Rothermel's slope factor: phi_slope = 5.275 × tan²(θ)
        # The 5.275 coefficient is empirically derived from fire experiments
        phi_slope = ROTHERMEL_SLOPE_FACTOR * (np.tan(slope_angle[uphill_mask]) ** 2)

        # Multiplicative factor: 1.0 + phi_slope
        # Example: 30° slope → tan²(30°) ≈ 0.33 → phi ≈ 1.74 → factor ≈ 2.74
        slope_factor[uphill_mask] = 1.0 + phi_slope

        return slope_factor

    def _update_temperature_grid(self) -> None:
        """
        Update temperature grid based on current fire state.

        Temperature is used by:
        - Sentinel agents: Signal detection (attenuated by distance)
        - Rescuer agents: Path risk assessment (refuse if temp > threshold)

        Temperature values are simplified for agent perception, not physically accurate.
        """
        fire_grid = self.environment.fire_grid
        temp_grid = self.environment.temperature_grid

        # Burning cells have high temperature (100°C)
        temp_grid[fire_grid == 1] = 100.0

        # Burnt cells gradually cool down (-5°C per step until ambient)
        # Simulates cooling of burnt-out areas over time
        temp_grid[fire_grid == 2] = np.maximum(0, temp_grid[fire_grid == 2] - 5.0)

        # Fuel cells have ambient temperature (20°C)
        temp_grid[fire_grid == 3] = 20.0

        # No fuel cells (water, roads) have ambient temperature
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
