"""
Sentinel Agent - Reactive Architecture
Acts as a fire detection sensor with Signal Detection Theory
Implements environmental attenuation based on distance and wind angle

Signal Detection Theory (SDT) foundation:
  Green, D.M. & Swets, J.A. (1966).
  "Signal Detection Theory and Psychophysics."
  John Wiley & Sons, New York. 467 pp.

The sensor equation  I_detected = I_actual / (d² + ε) × (1 + cos θ) + N(0, σ)
models inverse-square-law attenuation (d²), wind-enhancement (cos θ),
and Gaussian environmental noise — all core SDT concepts.
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
    WIND_INITIAL_DIRECTION,
    WIND_OSCILLATION_PERIOD,
    WIND_OSCILLATION_AMPLITUDE
)


class SentinelAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Sentinel
    ARCHITECTURE: Reactive — Signal Detection Theory (SDT)
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • detection_history   (row,col) → consecutive detection count
      • wind_direction      normalised wind vector (updated each step from
                            FireSimulation's oscillating model)
      • detected_fires      confirmed events this step [(lat,lon,intensity)]

    DESIRES
      • Detect active fires as early as possible
      • Suppress false positives (require N consecutive detections before
        reporting — SDT debouncing criterion)

    INTENTIONS
      • Scan circular neighbourhood each step using inverse-square-law
        attenuation with wind-enhancement and Gaussian noise
      • Apply debounce protocol: only report after ≥ SENTINEL_DEBOUNCE_STEPS
        consecutive positive hits on the same cell
      • Forward confirmed detections to Analyst immediately

    COMMUNICATION
      SENDS
        → INFORM  analyst  {type,lat,lon,intensity,timestamp}
              fire detection alert (after debounce threshold met)
      RECEIVES
        (none — pure reactive sensor; no incoming messages processed)

    BIBLIOGRAPHY
      [1] Green, D.M. & Swets, J.A. (1966). Signal Detection Theory and
          Psychophysics. John Wiley & Sons, New York. 467 pp.
          SDT equation: I_det = I_actual/(d²+ε) × (1+cosθ) + N(0,σ)
          Threshold criterion β maps to self.threshold; σ to self.sigma.
      [2] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          (Perceive → Decide → Act cycle inherited from BaseAgent)
    ═══════════════════════════════════════════════════════════════════════
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
        # These parameters model realistic sensor limitations.
        # SDT distinguishes signal from noise using a likelihood-ratio threshold.
        # Ref: Green & Swets (1966), "Signal Detection Theory and Psychophysics."
        self.epsilon = SENTINEL_SIGNAL_EPSILON  # Prevents div-by-zero (d² + ε)
        self.sigma = SENTINEL_NOISE_SIGMA  # Gaussian noise std dev (environmental)
        self.threshold = SENTINEL_TRIGGER_THRESHOLD  # Detection threshold (criterion β in SDT)
        self.debounce_steps = SENTINEL_DEBOUNCE_STEPS  # Consecutive detections required

        # ===== WIND DIRECTION (for signal attenuation) =====
        # Initialised to the static starting direction; updated each step in perceive()
        # to mirror FireSimulation's dynamic oscillating wind model.
        wind_rad = np.radians(WIND_INITIAL_DIRECTION)
        self.wind_direction = np.array(
            [np.sin(wind_rad), -np.cos(wind_rad)], dtype=np.float32
        )
        wind_mag = np.linalg.norm(self.wind_direction)
        if wind_mag > 1e-10:
            self.wind_direction = self.wind_direction / wind_mag

        # ===== DEBOUNCING: PREVENT FALSE ALARMS =====
        # Requires N consecutive detections before triggering alert
        # Prevents noise spikes from causing false positives
        # Dictionary maps (row, col) → consecutive_detection_count
        self.detection_history = {}  # (row, col) -> consecutive_count

    def perceive(self, environment) -> None:
        """
        Scan local area for fire using Signal Detection Theory.
        Signal equation: I_detected = I_actual/(d^2 + ε) * (1 + cos(θ)) + N(0,σ)
        Wind direction is updated every step to match FireSimulation's dynamic model:
        θ(t) = θ_0 + sin(t / T) × A
        """
        # ===== UPDATE DYNAMIC WIND DIRECTION =====
        # Mirrors FireSimulation._update_wind_vector() so detection is wind-consistent.
        step = environment.step_count
        if step == 0:
            wind_degrees = WIND_INITIAL_DIRECTION
        else:
            oscillation = np.sin(step / WIND_OSCILLATION_PERIOD)
            wind_degrees = WIND_INITIAL_DIRECTION + oscillation * WIND_OSCILLATION_AMPLITUDE
        wind_rad = np.radians(wind_degrees)
        wind_vec = np.array([np.sin(wind_rad), -np.cos(wind_rad)], dtype=np.float32)
        wind_mag = np.linalg.norm(wind_vec)
        if wind_mag > 1e-10:
            self.wind_direction = wind_vec / wind_mag

        self.detected_fires.clear()

        if self.grid_position is None:
            return

        row, col = self.grid_position
        fire_grid = environment.fire_grid
        temp_grid = environment.temperature_grid

        # ── Vectorised scan (replaces Python per-cell loop) ─────────────────
        # Build offset arrays for the bounding box and apply circular mask.
        R   = self.detection_radius
        H, W = fire_grid.shape
        r0, r1 = max(0, row - R), min(H, row + R + 1)
        c0, c1 = max(0, col - R), min(W, col + R + 1)

        dr = np.arange(r0, r1, dtype=np.int32) - row   # shape (rows,)
        dc = np.arange(c0, c1, dtype=np.int32) - col   # shape (cols,)
        DR, DC = np.meshgrid(dr, dc, indexing='ij')     # (rows, cols)

        dist2 = DR.astype(np.float32) ** 2 + DC.astype(np.float32) ** 2
        in_circle = (dist2 <= R * R) & (dist2 > 0)

        sub_fire = fire_grid[r0:r1, c0:c1]
        burning  = (sub_fire == 1) & in_circle

        current_detections: set = set()

        if burning.any():
            sub_temp = temp_grid[r0:r1, c0:c1]
            I_actual = sub_temp.astype(np.float32)
            d        = np.sqrt(dist2)

            # Wind cosine: to_sensor = (DC, DR) / d
            cos_theta = np.where(
                d > 0,
                (self.wind_direction[0] * DC + self.wind_direction[1] * DR) / d,
                0.0,
            ).astype(np.float32)

            signal    = I_actual / (dist2 + self.epsilon) * (1.0 + cos_theta)
            noise     = np.random.normal(0.0, self.sigma, signal.shape).astype(np.float32)
            I_det     = signal + noise

            detected_mask = burning & (I_det > self.threshold)
            rows_det, cols_det = np.where(detected_mask)
            for ri, ci in zip(rows_det + r0, cols_det + c0):
                current_detections.add((int(ri), int(ci)))

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
