"""
Configuration for AIGIS Location-Agnostic Multi-Agent Simulation
All parameters for universal wildfire evacuation system
"""
import numpy as np

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Random Seed (for reproducibility in Monte Carlo experiments)
# Set to None for non-deterministic behavior
RANDOM_SEED = 42

# Simulation Duration
MAX_STEPS = 500  # Maximum simulation steps (prevents infinite loops)
STEP_DELAY = 0.1  # seconds between steps (GUI mode only, for visualization)

# Grid Configuration (Cellular Automata Resolution)
# Larger grids = more spatial detail but slower performance
# Trade-off: 100×100 is fast, 200×200 is detailed
GRID_WIDTH = 200   # Grid columns
GRID_HEIGHT = 200  # Grid rows

# =============================================================================
# MAP CONFIGURATION (Location-Agnostic)
# =============================================================================

# Default coordinates (can be overridden via CLI)
DEFAULT_MAP_CENTER_LAT = 38.04  # Athens (example)
DEFAULT_MAP_CENTER_LON = 23.80
DEFAULT_MAP_RADIUS = 2000  # meters

# Aliases for backward compatibility
MAP_CENTER_LAT = DEFAULT_MAP_CENTER_LAT
MAP_CENTER_LON = DEFAULT_MAP_CENTER_LON
MAP_RADIUS = DEFAULT_MAP_RADIUS

# Safe Zone Detection (OSM Tags)
SAFE_ZONE_TAGS = {
    'natural': ['water', 'beach', 'coastline'],
    'leisure': ['park', 'nature_reserve', 'playground'],
    'place': ['square']
}

# Perimeter nodes (map edges) are always considered safe
USE_PERIMETER_AS_SAFE = True

# =============================================================================
# TERRAIN GENERATION (Perlin Noise)
# =============================================================================

PERLIN_SCALE = 100.0  # Controls terrain "chunkiness"
PERLIN_OCTAVES = 4  # Detail levels
PERLIN_PERSISTENCE = 0.5  # Amplitude decay
PERLIN_LACUNARITY = 2.0  # Frequency increase
PERLIN_BASE_HEIGHT = 50.0  # Base elevation (meters)
PERLIN_AMPLITUDE = 100.0  # Height variation (meters)

# =============================================================================
# DYNAMIC WIND MODEL
# =============================================================================

# Initial wind direction (degrees from North, clockwise)
WIND_INITIAL_DIRECTION = 90.0  # East

# Dynamic wind: θ(t) = θ_0 + sin(t/50) × 20°
WIND_OSCILLATION_PERIOD = 50.0  # steps
WIND_OSCILLATION_AMPLITUDE = 20.0  # degrees
WIND_SPEED = 5.0  # m/s (base speed)

# Wind direction vector (used by Sentinel for signal detection)
# Calculated from WIND_INITIAL_DIRECTION
_wind_rad = np.radians(WIND_INITIAL_DIRECTION)
WIND_DIRECTION = [np.sin(_wind_rad), -np.cos(_wind_rad)]  # [dx, dy] in grid coords

# =============================================================================
# FIRE SPREAD (Rothermel-Based)
# =============================================================================

FIRE_SPREAD_PROB_BASE = 0.3  # Base probability
FIRE_BURNOUT_PROB = 0.1  # Probability per step that burning cell burns out

# Rothermel Parameters
ROTHERMEL_BASE_ROS = 0.5  # meters/second
ROTHERMEL_WIND_C = 0.4
ROTHERMEL_WIND_B = 1.5
ROTHERMEL_SLOPE_FACTOR = 5.275  # Slope coefficient

# =============================================================================
# AGENT COUNTS
# =============================================================================

NUM_SENTINELS = 4
NUM_ANALYSTS = 1  # Typically one central analyst
NUM_COMMANDERS = 1
NUM_RESCUERS = 3
NUM_CIVILIANS = 20

# =============================================================================
# SENTINEL AGENT (Signal Detection Theory)
# =============================================================================
# Sentinel agents are fire detection sensors with environmental signal attenuation
# Signal equation: I_detected = I_actual/(d² + ε) × (1 + cos(θ)) + N(0,σ)
# Where: d = distance, θ = wind angle, σ = noise

SENTINEL_DETECTION_RADIUS = 30  # grid cells (spatial optimization: O(R²) scan)
SENTINEL_SIGNAL_EPSILON = 1.0  # Prevents division by zero in signal equation
SENTINEL_NOISE_SIGMA = 5.0  # Gaussian noise std dev (environmental noise)
SENTINEL_TRIGGER_THRESHOLD = 15.0  # Intensity threshold for fire alert
SENTINEL_DEBOUNCE_STEPS = 3  # Must detect for N consecutive steps (prevents false alarms)

# =============================================================================
# ANALYST AGENT (Fuzzy Logic)
# =============================================================================
# Analyst uses fuzzy logic to assess risk based on Time To Impact (TTI) and
# escape route availability. These thresholds define linguistic variables.

# Fuzzy Input Ranges
ANALYST_ROS_RANGE = (0, 10)  # Rate of spread (m/s)
ANALYST_DISTANCE_RANGE = (0, 200)  # Distance to assets (meters)

# Fuzzy Output Range
ANALYST_RISK_RANGE = (0, 100)

# Time To Impact (TTI) Thresholds (meters)
ANALYST_TTI_IMMINENT = 30  # Fire is imminent (0-30 meters)
ANALYST_TTI_NEAR = 70  # Fire approaching (30-70 meters)
# Above 70 meters = distant threat

# Escape Route Thresholds
ANALYST_EXIT_BOTTLENECK_THRESHOLD = 2  # Number of exits (0-2 = bottlenecked, 2+ = sufficient)

# Re-evaluation trigger
ANALYST_WIND_CHANGE_THRESHOLD = 5.0  # degrees (re-evaluate if wind shifts > 5°)

# =============================================================================
# COMMANDER AGENT (ECT vs TTI Decision Protocol)
# =============================================================================
# Commander uses Evacuation Clearance Time (ECT) vs Time To Impact (TTI)
# to determine evacuation phase
#
# ECT = (N_agents / C_exit) × γ_congestion  [How long to evacuate everyone]
# TTI = Distance_to_fire / ROS               [How long until fire arrives]
#
# Decision Logic:
# - TTI > 2.5×ECT → Phase 0: Monitoring (plenty of time)
# - TTI > 1.5×ECT → Phase 1: Pre-Alert (prepare to evacuate)
# - TTI > 1.0×ECT → Phase 2: Mass Evacuation (evacuate now!)
# - TTI ≤ ECT     → Phase 3: Shelter-in-Place (too late, seek nearest safe zone)

COMMANDER_EXIT_CAPACITY = 10  # agents per minute per exit (bottleneck capacity)
COMMANDER_CONGESTION_FACTOR_BASE = 1.0  # Congestion multiplier (increases with density)
COMMANDER_REEVALUATION_INTERVAL = 10  # steps (periodic re-assessment of situation)

# Phase Multipliers (thresholds for phase transitions)
COMMANDER_PHASE_MONITOR_MULTIPLIER = 2.5   # TTI > 2.5 × ECT: Monitoring
COMMANDER_PHASE_PREALERT_MULTIPLIER = 1.5  # TTI > 1.5 × ECT: Pre-Alert
COMMANDER_PHASE_EVACUATE_MULTIPLIER = 1.0  # TTI > 1.0 × ECT: Mass Evacuation
# Phase 3 (Shelter): TTI ≤ ECT (too late to evacuate safely)

# =============================================================================
# RESCUER AGENT (Risk-Adjusted Bidding via Contract Net Protocol)
# =============================================================================
# Rescuers respond to Commander's Call For Proposals (CFP) for rescue missions
# Bid Calculation: Cost = (Distance/V) + (Risk_path × α) + (100 - Fuel)
# Safety Protocol: REFUSE missions if Risk_path > Safety_Threshold
#
# This implements goal-based architecture with practical reasoning:
# - Assess path risk by scanning temperature grid along route
# - Refuse dangerous missions (professional safety protocol)
# - Dynamic re-routing if path becomes blocked by fire

RESCUER_MAX_SPEED = 3.0  # grid cells per step (rescue vehicle speed)
RESCUER_PATH_RECALC_INTERVAL = 20  # Recalculate path every N steps (staggered pathfinding)
RESCUER_FUEL_CAPACITY = 100  # Resource constraint
RESCUER_RISK_ALPHA = 50.0  # Risk penalty weight in bid calculation (higher = more risk-averse)
RESCUER_SAFETY_THRESHOLD = 70.0  # Temperature threshold (°C) - refuse if path exceeds this

# =============================================================================
# CIVILIAN AGENT (Greenshields Traffic + Social Force Herding + BDI Architecture)
# =============================================================================
# Most complex agent with realistic crowd dynamics and panic psychology
#
# Traffic Model: V = V_free × (1 - ρ_local / ρ_jam)
#   - Speed decreases linearly with local density
#   - Gridlock occurs when density reaches jam density (V → 0)
#
# Panic Equation: Panic(t) = Panic(t-1) + α×(1/d_fire) + β×(family) - decay
#   - Increases with fire proximity (inverse distance)
#   - Increases if family is separated
#   - Decays slowly when no fire visible
#
# 3-State Cognitive Machine:
#   - Rational (< 0.4): Optimal A* pathfinding, full speed
#   - Confused (0.4-0.7): 50% speed, frequent re-routing
#   - Herding (≥ 0.7): Follows crowd (Social Force), ignores optimal path

# ===== Greenshields Traffic Model Parameters =====
CIVILIAN_V_FREE_FLOW = 2.0  # Maximum speed when road is empty (grid cells/step)
CIVILIAN_RHO_JAM = 5.0  # Jam density: agents per edge (gridlock threshold)

# ===== Panic Model Parameters =====
CIVILIAN_PANIC_ALPHA = 0.05  # Fire distance coefficient (inverse relationship)
CIVILIAN_PANIC_BETA = 0.2  # Family separation penalty (constant stress)
CIVILIAN_PANIC_DECAY = 0.01  # Decay rate when no fire visible (calming down)

# ===== Cognitive State Thresholds =====
CIVILIAN_PANIC_RATIONAL = 0.4  # Below: Rational behavior (optimal decisions)
CIVILIAN_PANIC_CONFUSED = 0.7  # Below: Confused behavior (degraded performance)
CIVILIAN_PANIC_HERDING = 0.8  # Above: Herding behavior (follows crowd)

# ===== Speed Modifiers =====
CIVILIAN_CONFUSED_SPEED_FACTOR = 0.5  # 50% speed reduction when confused (hesitation)

# ===== Social Force Model (Herding) =====
CIVILIAN_VISION_RADIUS = 10  # Grid cells for seeing other agents
CIVILIAN_HERDING_INFLUENCE = 0.7  # Weight of crowd direction (0=ignore, 1=fully follow)

# ===== Performance Optimization =====
CIVILIAN_PATH_RECALC_INTERVAL = 20  # Recalculate A* path every N steps (staggered)

# =============================================================================
# MONTE CARLO / BATCH MODE
# =============================================================================

BATCH_NUM_RUNS = 10  # Default number of Monte Carlo runs
BATCH_OUTPUT_FILE = "results.csv"
BATCH_LOG_INTERVAL = 5  # Print progress every N runs

# Metrics to track
METRICS_TRACK = [
    'steps_to_evacuate',
    'mortality_rate',
    'evacuation_success_rate',
    'avg_panic_level',
    'rescuer_refusals',
    'total_burning_cells'
]

# =============================================================================
# VISUALIZATION (GUI Mode)
# =============================================================================

FIGURE_SIZE = (18, 10)
DPI = 100

# Colors
COLOR_FUEL = '#228B22'  # Forest green
COLOR_BURNING = '#FF4500'  # Orange red
COLOR_BURNT = '#2F4F4F'  # Dark slate gray
COLOR_OBSTACLE = '#696969'  # Dim gray
COLOR_WATER = '#4169E1'  # Royal blue
COLOR_SAFE_ZONE = '#90EE90'  # Light green
COLOR_ROAD = '#D3D3D3'  # Light gray

# Agent Colors
COLOR_SENTINEL = '#FFD700'  # Gold
COLOR_ANALYST = '#9370DB'  # Medium purple
COLOR_COMMANDER = '#DC143C'  # Crimson
COLOR_RESCUER = '#1E90FF'  # Dodger blue
COLOR_CIVILIAN = '#32CD32'  # Lime green

# Dashboard Layout
DASHBOARD_UPDATE_INTERVAL = 1  # Update charts every N steps
DASHBOARD_HISTORY_LENGTH = 100  # Keep last N points on charts

# =============================================================================
# DEBUGGING / LOGGING
# =============================================================================

DEBUG_MODE = False
LOG_AGENT_MESSAGES = False  # Log all FIPA-ACL messages
LOG_PHASE_TRANSITIONS = True  # Log Commander phase changes
LOG_WIND_CHANGES = True  # Log when wind direction shifts

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
# Validate critical parameters to prevent division by zero and invalid configs

assert MAX_STEPS > 0, "MAX_STEPS must be positive"
assert GRID_WIDTH > 0 and GRID_HEIGHT > 0, "Grid dimensions must be positive"
assert COMMANDER_EXIT_CAPACITY > 0, "Exit capacity must be positive (prevents division by zero)"
assert RESCUER_MAX_SPEED > 0, "Rescuer speed must be positive (prevents division by zero)"
assert CIVILIAN_V_FREE_FLOW > 0, "Civilian free flow speed must be positive"
assert CIVILIAN_RHO_JAM > 0, "Jam density must be positive"
assert ROTHERMEL_BASE_ROS >= 0, "Base rate of spread cannot be negative"
assert WIND_SPEED >= 0, "Wind speed cannot be negative"

# Validate panic thresholds are in logical order
assert 0 <= CIVILIAN_PANIC_RATIONAL <= CIVILIAN_PANIC_CONFUSED <= 1.0, \
    "Panic thresholds must be in order: 0 ≤ RATIONAL ≤ CONFUSED ≤ 1.0"

# =============================================================================
# NUMPY RANDOM SEED INITIALIZATION
# =============================================================================
# NOTE: Do NOT set global seed here! This breaks Monte Carlo simulations.
# Each simulation run must initialize its own seed (RANDOM_SEED + run_id)
# to ensure different random sequences across runs.
#
# Seed initialization moved to simulation.__init__() for proper per-run seeding.
