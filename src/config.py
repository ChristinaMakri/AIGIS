"""
Configuration for AIGIS Location-Agnostic Multi-Agent Simulation
All parameters for universal wildfire evacuation system
"""
import numpy as np

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Random Seed (for reproducibility in Monte Carlo)
RANDOM_SEED = 42

# Simulation Duration
MAX_STEPS = 500
STEP_DELAY = 0.1  # seconds (GUI mode only)

# Grid Configuration
GRID_WIDTH = 200
GRID_HEIGHT = 200

# =============================================================================
# MAP CONFIGURATION (Location-Agnostic)
# =============================================================================

# Default coordinates (can be overridden via CLI)
DEFAULT_MAP_CENTER_LAT = 38.04  # Athens (example)
DEFAULT_MAP_CENTER_LON = 23.80
DEFAULT_MAP_RADIUS = 2000  # meters

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
import numpy as np
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

SENTINEL_DETECTION_RADIUS = 30  # grid cells
SENTINEL_SIGNAL_EPSILON = 1.0  # Prevents division by zero
SENTINEL_NOISE_SIGMA = 5.0  # Gaussian noise std dev
SENTINEL_TRIGGER_THRESHOLD = 15.0  # Intensity threshold
SENTINEL_DEBOUNCE_STEPS = 3  # Must detect for N consecutive steps

# =============================================================================
# ANALYST AGENT (Fuzzy Logic)
# =============================================================================

# Fuzzy Input Ranges
ANALYST_ROS_RANGE = (0, 10)  # Rate of spread (m/s)
ANALYST_DISTANCE_RANGE = (0, 200)  # Distance to assets (meters)

# Fuzzy Output Range
ANALYST_RISK_RANGE = (0, 100)

# Re-evaluation trigger
ANALYST_WIND_CHANGE_THRESHOLD = 5.0  # degrees (re-evaluate if wind shifts > 5°)

# =============================================================================
# COMMANDER AGENT (ECT vs TTI)
# =============================================================================

COMMANDER_EXIT_CAPACITY = 10  # agents per minute per exit
COMMANDER_CONGESTION_FACTOR_BASE = 1.0
COMMANDER_REEVALUATION_INTERVAL = 10  # steps

# Phase Multipliers
COMMANDER_PHASE_MONITOR_MULTIPLIER = 2.5  # TTI > 2.5 × ECT
COMMANDER_PHASE_PREALERT_MULTIPLIER = 1.5  # TTI > 1.5 × ECT
COMMANDER_PHASE_EVACUATE_MULTIPLIER = 1.0  # TTI > 1.0 × ECT
# Phase Shelter: TTI <= ECT (too late to evacuate)

# =============================================================================
# RESCUER AGENT (Risk-Adjusted Bidding)
# =============================================================================

RESCUER_MAX_SPEED = 3.0  # grid cells per step
RESCUER_PATH_RECALC_INTERVAL = 20  # Recalculate path every N steps (performance optimization)
RESCUER_FUEL_CAPACITY = 100
RESCUER_RISK_ALPHA = 50.0  # Risk penalty weight
RESCUER_SAFETY_THRESHOLD = 70.0  # Temperature threshold (refuse if path > this)

# =============================================================================
# CIVILIAN AGENT (Greenshields + Social Force)
# =============================================================================

# Greenshields Traffic Model
CIVILIAN_V_FREE_FLOW = 2.0  # Free flow speed
CIVILIAN_RHO_JAM = 5.0  # Jam density (agents per edge)

# Panic Model
CIVILIAN_PANIC_ALPHA = 0.05  # Fire distance coefficient
CIVILIAN_PANIC_BETA = 0.2  # Family separation penalty
CIVILIAN_PANIC_DECAY = 0.01  # Decay when no fire visible

# Cognitive Thresholds
CIVILIAN_PANIC_RATIONAL = 0.4  # Below: rational
CIVILIAN_PANIC_CONFUSED = 0.7  # Below: confused
CIVILIAN_PANIC_HERDING = 0.8  # Above: herding (social force)

# Speed Modifiers
CIVILIAN_CONFUSED_SPEED_FACTOR = 0.5  # 50% speed when confused

# Social Force Model (Herding)
CIVILIAN_VISION_RADIUS = 10  # Grid cells for seeing neighbors
CIVILIAN_HERDING_INFLUENCE = 0.7  # Weight of neighbor average direction

# Performance Optimization
CIVILIAN_PATH_RECALC_INTERVAL = 20  # Recalculate path every N steps (staggered pathfinding)

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
# NUMPY RANDOM SEED INITIALIZATION
# =============================================================================

np.random.seed(RANDOM_SEED)
