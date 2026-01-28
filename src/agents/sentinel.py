"""
Sentinel Agent - Reactive Architecture
Acts as a fire detection sensor with Signal Detection Theory
Implements environmental attenuation based on distance and wind angle
"""
import numpy as np
from typing import Tuple, List
from .base_agent import Agent
from ..message import Message
from ..config import (
    SENTINEL_DETECTION_RADIUS,
    SENTINEL_SIGNAL_EPSILON,
    SENTINEL_NOISE_SIGMA,
    SENTINEL_TRIGGER_THRESHOLD,
    SENTINEL_DEBOUNCE_STEPS,
    WIND_DIRECTION
)


class SentinelAgent(Agent):
    """
    Reactive agent that detects fire using Signal Detection Theory.

    Architecture: Condition-Action Rules with Debouncing
    - Signal attenuation based on distance and wind direction
    - Debouncing protocol: 3 consecutive detections required
    - Signal equation: I_detected = I_actual/(d^2 + ε) * (1 + cos(θ)) + N(0,σ)
    """

    def __init__(self, agent_id: str, position: Tuple[float, float],
                 detection_radius: int = SENTINEL_DETECTION_RADIUS):
        """
        Initialize Sentinel agent with Signal Detection Theory parameters.

        Sentinel agents are reactive fire sensors that detect fires using a
        physics-based signal attenuation model with environmental noise.

        Signal Equation:
        I_detected = (I_actual / (d² + ε)) × (1 + cos(θ)) + N(0, σ)

        Where:
        - I_actual: True fire intensity (from temperature grid)
        - d: Euclidean distance to fire
        - ε: Small constant to prevent division by zero
        - θ: Angle between wind direction and sensor vector
        - N(0, σ): Gaussian noise (environmental interference)

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
            detection_radius: Detection range in grid cells
        """
        super().__init__(agent_id, position)

        # ===== DETECTION PARAMETERS =====
        self.detection_radius = detection_radius  # Spatial scan range
        self.detected_fires = []  # List of confirmed fire detections this step

        # ===== SIGNAL DETECTION THEORY PARAMETERS =====
        # These parameters model realistic sensor limitations
        self.epsilon = SENTINEL_SIGNAL_EPSILON  # Prevents div-by-zero (d² + ε)
        self.sigma = SENTINEL_NOISE_SIGMA  # Gaussian noise std dev (environmental)
        self.threshold = SENTINEL_TRIGGER_THRESHOLD  # Detection threshold
        self.debounce_steps = SENTINEL_DEBOUNCE_STEPS  # Consecutive detections required

        # ===== WIND DIRECTION (for signal attenuation) =====
        # Wind carries smoke/heat toward sensor → increases detection probability
        # Wind blowing away from sensor → decreases detection probability
        self.wind_direction = np.array(WIND_DIRECTION, dtype=np.float32)
        wind_mag = np.linalg.norm(self.wind_direction)
        if wind_mag > 0:
            self.wind_direction = self.wind_direction / wind_mag  # Normalize

        # ===== DEBOUNCING: PREVENT FALSE ALARMS =====
        # Requires N consecutive detections before triggering alert
        # Prevents noise spikes from causing false positives
        # Dictionary maps (row, col) → consecutive_detection_count
        self.detection_history = {}  # (row, col) -> consecutive_count

    def perceive(self, environment) -> None:
        """
        Scan local area for fire using Signal Detection Theory.
        Signal equation: I_detected = I_actual/(d^2 + ε) * (1 + cos(θ)) + N(0,σ)
        """
        self.detected_fires.clear()

        if self.grid_position is None:
            return

        row, col = self.grid_position
        fire_grid = environment.fire_grid
        temp_grid = environment.temperature_grid

        # Track current detections for debouncing
        current_detections = set()

        # SPATIAL OPTIMIZATION: Bounding Box + Circular Check
        # Only scan within detection_radius (not the entire grid)
        # This reduces complexity from O(N^2) to O(R^2) where R << N
        for dr in range(-self.detection_radius, self.detection_radius + 1):
            for dc in range(-self.detection_radius, self.detection_radius + 1):
                if dr == 0 and dc == 0:
                    continue

                # CIRCULAR BOUNDARY: Skip cells outside circular detection range
                # Use >= to include cells exactly at the detection radius
                distance_sq = dr**2 + dc**2
                if distance_sq > self.detection_radius**2:
                    continue

                r, c = row + dr, col + dc

                # Check bounds
                if 0 <= r < fire_grid.shape[0] and 0 <= c < fire_grid.shape[1]:
                    # Only process burning cells (skip empty cells immediately)
                    if fire_grid[r, c] == 1:
                        # Calculate attenuated signal
                        I_actual = temp_grid[r, c]  # Actual fire intensity
                        d = np.sqrt(dr**2 + dc**2)  # Euclidean distance

                        # Calculate angle between wind direction and sensor vector
                        to_sensor = np.array([dc, dr], dtype=np.float32)
                        sensor_mag = np.linalg.norm(to_sensor)
                        if sensor_mag > 0:
                            to_sensor = to_sensor / sensor_mag
                            cos_theta = np.dot(self.wind_direction, to_sensor)
                        else:
                            cos_theta = 0

                        # Signal Detection Theory equation
                        signal_attenuation = I_actual / (d**2 + self.epsilon)
                        wind_factor = 1 + cos_theta
                        noise = np.random.normal(0, self.sigma)
                        I_detected = signal_attenuation * wind_factor + noise

                        # Check if signal exceeds threshold
                        if I_detected > self.threshold:
                            current_detections.add((r, c))

        # Debouncing protocol: update detection history
        # Remove old detections not in current scan
        keys_to_remove = [k for k in self.detection_history.keys() if k not in current_detections]
        for key in keys_to_remove:
            del self.detection_history[key]

        # Update counts for current detections
        for location in current_detections:
            if location in self.detection_history:
                self.detection_history[location] += 1
            else:
                self.detection_history[location] = 1

            # Trigger alert if detected for N consecutive steps
            if self.detection_history[location] >= self.debounce_steps:
                r, c = location
                lat, lon = environment.grid_to_latlon(r, c)
                intensity = temp_grid[r, c]
                self.detected_fires.append((lat, lon, intensity))

    def decide(self) -> None:
        """
        Reactive decision: If fire detected, prepare to inform Analyst.
        No complex reasoning - pure condition-action.
        """
        # Decision is implicit in act() - reactive architecture
        pass

    def act(self, environment) -> None:
        """Send fire detection reports to Analyst"""
        if self.detected_fires:
            for lat, lon, intensity in self.detected_fires:
                message = Message(
                    sender=self.agent_id,
                    receiver="analyst",
                    performative="INFORM",
                    content={
                        'type': 'FIRE_DETECTION',
                        'lat': lat,
                        'lon': lon,
                        'intensity': intensity,
                        'timestamp': environment.step_count
                    }
                )
                self.send_message(message)
