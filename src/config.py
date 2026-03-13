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
RANDOM_SEED = None  # None = stochastic (required for meaningful Monte Carlo variance)

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
# Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire Spread in
# Wildland Fuels. USDA Forest Service Research Paper INT-115.
# Intermountain Forest and Range Experiment Station, Ogden, UT.
#
# Model:  ROS = R_base × (1 + φ_wind) × (1 + φ_slope)
#   φ_wind  = C × U^B         (eq. 47)
#   φ_slope = 5.275 × tan²(θ) (eq. 51)

FIRE_SPREAD_PROB_BASE = 0.3  # Base probability
FIRE_BURNOUT_PROB = 0.1  # Probability per step that burning cell burns out

# Rothermel Parameters (see Rothermel 1972)
ROTHERMEL_BASE_ROS = 0.5  # meters/second — baseline rate of spread R_base
ROTHERMEL_WIND_C = 0.4    # wind coefficient C (eq. 47)
ROTHERMEL_WIND_B = 1.5    # wind exponent B (eq. 47)
ROTHERMEL_SLOPE_FACTOR = 5.275  # slope coefficient (eq. 51, empirically derived)

# Temperature Model Parameters
FIRE_TEMP_BURNING = 100.0  # °C - Temperature of actively burning cells
FIRE_TEMP_COOLING_RATE = 5.0  # °C/step - Cooling rate for burnt-out cells
FIRE_TEMP_AMBIENT = 20.0  # °C - Ambient temperature (baseline)

# =============================================================================
# FUEL TYPE MODELS (NFFL - Northern Forest Fire Laboratory)
# =============================================================================
# Anderson, H.E. (1982). Aids to Determining Fuel Models for Estimating Fire
# Behavior. USDA Forest Service General Technical Report INT-122.
# Intermountain Forest and Range Experiment Station, Ogden, UT.
#
# 13 standard NFFL fuel model classifications used below.

# Fuel Model Definitions
# Based on Anderson's 13 Fire Behavior Fuel Models
FUEL_MODELS = {
    # Non-fuel areas (urban, water, roads) — fire cannot spread here
    0: {
        'name': 'No Fuel',
        'spread_multiplier': 0.0,   # fire cannot spread through non-fuel cells
        'intensity_multiplier': 0.0,
        'burnout_prob': 0.0         # never burns out (never ignites)
    },

    # Grass Fuels (fast spread, low intensity)
    1: {  # Short Grass (1 foot)
        'name': 'Short Grass',
        'spread_multiplier': 1.5,  # Spreads 50% faster than base
        'intensity_multiplier': 0.6,  # Lower intensity
        'burnout_prob': 0.2  # Burns out faster
    },
    2: {  # Timber (grass and understory)
        'name': 'Timber Grass',
        'spread_multiplier': 1.3,
        'intensity_multiplier': 0.8,
        'burnout_prob': 0.15
    },
    3: {  # Tall Grass (2.5 feet)
        'name': 'Tall Grass',
        'spread_multiplier': 1.8,
        'intensity_multiplier': 0.9,
        'burnout_prob': 0.18
    },

    # Brush Fuels (medium spread, high intensity)
    4: {  # Chaparral (6 feet)
        'name': 'Chaparral',
        'spread_multiplier': 1.2,
        'intensity_multiplier': 1.5,
        'burnout_prob': 0.08
    },
    5: {  # Brush (2 feet)
        'name': 'Low Brush',
        'spread_multiplier': 1.0,
        'intensity_multiplier': 1.2,
        'burnout_prob': 0.10
    },
    6: {  # Dormant Brush
        'name': 'Dormant Brush',
        'spread_multiplier': 1.4,
        'intensity_multiplier': 1.3,
        'burnout_prob': 0.12
    },
    7: {  # Southern Rough
        'name': 'Southern Rough',
        'spread_multiplier': 1.1,
        'intensity_multiplier': 1.4,
        'burnout_prob': 0.09
    },

    # Timber Fuels (slow spread, extreme intensity)
    8: {  # Closed Timber Litter
        'name': 'Closed Timber',
        'spread_multiplier': 0.6,
        'intensity_multiplier': 1.8,
        'burnout_prob': 0.05
    },
    9: {  # Hardwood Litter
        'name': 'Hardwood Litter',
        'spread_multiplier': 0.7,
        'intensity_multiplier': 1.6,
        'burnout_prob': 0.06
    },
    10: {  # Timber (litter and understory)
        'name': 'Dense Timber',
        'spread_multiplier': 0.8,
        'intensity_multiplier': 2.0,
        'burnout_prob': 0.04
    }
}

# Default fuel model (used if not specified)
DEFAULT_FUEL_MODEL = 5  # Low Brush (moderate characteristics)

# =============================================================================
# CORINE LAND COVER (CLC) INTEGRATION
# =============================================================================

USE_CORINE = True
CORINE_CLC_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/services/Corine/CLC2018_WM/MapServer/"
    "WCSServer?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE=1"
    "&CRS=EPSG:4326&BBOX={minx},{miny},{maxx},{maxy}"
    "&WIDTH={width}&HEIGHT={height}&FORMAT=GeoTIFF"
)
CORINE_CACHE_FILE = "cache/corine_clc_{lat}_{lon}_{radius}.npz"

# Mapping from CLC class codes to NFFL fuel model numbers
CLC_TO_NFFL_MAP = {
    311: 8, 312: 9, 313: 8,   # forests → closed/hardwood/closed timber
    323: 4, 322: 6, 324: 7,   # Mediterranean scrub/heath/transitional
    321: 1, 231: 3,           # grassland/pasture
    111: 0, 112: 0,           # urban (non-fuel)
    331: 0, 332: 0, 511: 0,   # bare/water
}

# =============================================================================
# ML PREDICTOR FEATURE NAMES (14-feature simulation-derived vector)
# =============================================================================

ML_FEATURE_NAMES = [
    'burning_cells_pct',      # % of grid currently burning
    'burnt_cells_pct',        # % of grid already burnt
    'wind_speed',             # current wind speed (m/s)
    'wind_dir_x',             # wind direction x-component
    'wind_dir_y',             # wind direction y-component
    'mean_slope',             # mean terrain slope in burning area
    'dominant_fuel_type',     # mode fuel type in burning area
    'active_rescuers',        # number of active rescuer agents
    'civilians_remaining',    # number of civilians still active
    'current_phase',          # commander phase (0-3)
    'tti_normalized',         # TTI clipped to [0,1] (tti/60)
    'ect_normalized',         # ECT clipped to [0,1] (ect/30)
    'step_normalized',        # current step / MAX_STEPS
    'humidity',               # relative humidity (%)
]

# Logging Intervals
WIND_LOG_INTERVAL = 10  # Steps between wind direction change logs

# =============================================================================
# AGENT COUNTS
# =============================================================================

NUM_SENTINELS = 4
NUM_RESCUERS = 3
NUM_FIREFIGHTERS = 2  # Firefighting units for active suppression
NUM_CIVILIANS = 60

# =============================================================================
# SENTINEL AGENT (Signal Detection Theory)
# =============================================================================
# Sentinel agents are fire detection sensors with environmental signal attenuation
# Signal equation: I_detected = I_actual/(d² + ε) × (1 + cos(θ)) + N(0,σ)
# Where: d = distance, θ = wind angle, σ = noise

SENTINEL_DETECTION_RADIUS = 120  # grid cells — covers full 200×200 grid from corners
SENTINEL_SIGNAL_EPSILON = 1.0    # Prevents division by zero in signal equation
SENTINEL_NOISE_SIGMA = 0.002     # Gaussian noise std dev (scaled to weak far-field signals)
SENTINEL_TRIGGER_THRESHOLD = 0.01  # Detection threshold for I/(d²+ε)×(1+cos θ) signal
SENTINEL_DEBOUNCE_STEPS = 3      # Must detect for N consecutive steps (prevents false alarms)

# =============================================================================
# ANALYST AGENT (Fuzzy Logic)
# =============================================================================
# Analyst uses fuzzy logic to assess risk based on Time To Impact (TTI) and
# escape route availability. These thresholds define linguistic variables.

# Time To Impact (TTI) Thresholds (seconds)
# TTI = distance_to_population / ROS  → units are seconds
ANALYST_TTI_IMMINENT = 30  # Fire arrives within 30 seconds → imminent
ANALYST_TTI_NEAR = 70      # Fire arrives within 70 seconds → near future
# Above 70 seconds = distant threat

# Escape Route Thresholds
ANALYST_EXIT_BOTTLENECK_THRESHOLD = 2  # Number of exits (0-2 = bottlenecked, 2+ = sufficient)

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

# Commitment thresholds for re-evaluation (in minutes)
COMMANDER_TTI_RECONSIDER_THRESHOLD = 0.5   # minutes drift triggers reconsideration
COMMANDER_ECT_RECONSIDER_THRESHOLD = 0.25  # minutes drift triggers reconsideration

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
# 3-State Cognitive Machine:
#   [0.0, 0.4): RATIONAL - Optimal A* pathfinding, full speed
#   [0.4, 0.7): CONFUSED - 50% speed, frequent re-routing
#   [0.7, 1.0]: HERDING - Follows crowd (Social Force model)
CIVILIAN_PANIC_RATIONAL = 0.4  # Threshold: panic < 0.4 = rational behavior
CIVILIAN_PANIC_CONFUSED = 0.7  # Threshold: panic < 0.7 = confused behavior
# Note: panic >= 0.7 triggers herding behavior (no separate threshold needed)

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

# Metrics are automatically tracked in simulation.py
# See AIGISSimulation.get_results() for full list of metrics

# =============================================================================
# VISUALIZATION (GUI Mode)
# =============================================================================

FIGURE_SIZE = (18, 10)
DPI = 100

# Colors
COLOR_FUEL = '#228B22'  # Forest green
COLOR_BURNING = '#FF4500'  # Orange red
COLOR_BURNT = '#2F4F4F'  # Dark slate gray
COLOR_SAFE_ZONE = '#90EE90'  # Light green

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
# DATA CONNECTORS — API KEYS
# =============================================================================
# NASA FIRMS: free key at https://firms.modaps.eosdis.nasa.gov/api/
FIRMS_MAP_KEY = "656c1b1a74cf38b3b21bd3e8b14aa800"

# OpenAQ: free key at https://openaq.org/
OPENAQ_API_KEY = "6dbff8aefa3c76caf6a50a53be92f030f848c440827e7a80c06a1360cb6a14e1"

# =============================================================================
# RISK MONITOR AGENT (Pre-Ignition)
# =============================================================================
# FWI danger thresholds from:
# Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest
# Fire Weather Index System. Forestry Technical Report 35. Canadian Forestry
# Service, Ottawa.
NUM_RISK_MONITORS = 1
RISK_MONITOR_UPDATE_INTERVAL = 20   # Steps between full risk-grid recomputations
FWI_HIGH_RISK_THRESHOLD    = 30.0   # FWI ≥ 30 → High pre-fire warning to Commander
FWI_EXTREME_RISK_THRESHOLD = 50.0   # FWI ≥ 50 → Extreme pre-fire warning

# =============================================================================
# AMBULANCE AGENT (Medical Extraction)
# =============================================================================
NUM_AMBULANCES = 2
AMBULANCE_MAX_SPEED       = 3.0  # grid cells/step (same as Rescuer)
AMBULANCE_RISK_THRESHOLD  = 0.7  # Refuse missions through zones hotter than this

# =============================================================================
# SMOKE MODEL (Inness et al. 2019 — CAMS Reanalysis)
# =============================================================================
# Wildfire smoke plume physics grounded in:
#   Inness, A. et al. (2019). "The CAMS reanalysis of atmospheric composition."
#   Atmos. Chem. Phys., 19(6), pp. 3515–3556.
#   DOI: 10.5194/acp-19-3515-2019
#
# Simplified advection–diffusion: ∂C/∂t = -U·∇C + D·∇²C + S
#   C  = smoke concentration per cell (kg/m³ proxy, 0–1 normalised)
#   U  = wind vector (from Rothermel wind model)
#   D  = isotropic diffusion coefficient
#   S  = source term (burning cells)
SMOKE_SOURCE_STRENGTH   = 0.30   # Smoke emitted per burning cell per step [0-1]
SMOKE_DIFFUSION_RATE    = 0.08   # Isotropic lateral diffusion per step
SMOKE_DECAY_RATE        = 0.05   # Dissipation rate per step (atmosphere scavenging)
SMOKE_WIND_ADVECTION    = 0.40   # Fraction advected downwind per step

# =============================================================================
# CIVILIAN INJURY MODEL (Inness et al. 2019 + Cova & Johnson 2002)
# =============================================================================
# Cumulative PM2.5 exposure → incapacitation threshold, grounded in:
#   Inness, A. et al. (2019) — CAMS PM2.5 smoke product
#   Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood
#   evacuations in the urban-wildland interface." Env. Planning A, 34(12).
#   Smoke inhalation is identified as a primary casualty driver in WUI fires.
CIVILIAN_INJURY_THRESHOLD  = 5.0   # Cumulative smoke exposure before injury
CIVILIAN_SMOKE_PANIC_SCALE = 0.02  # Extra panic per unit smoke concentration

# =============================================================================
# AIR QUALITY EFFECTS ON CIVILIANS
# =============================================================================
# Each step, civilian panic increases by: AQI_PANIC_WEIGHT × (aqi / 500)
AQI_PANIC_WEIGHT  = 0.02   # Max +0.02/step at AQI=500 (~10 steps to add 0.2 panic)
# Max speed reduction at AQI=500: 30% slower (smoke inhalation, reduced visibility)
AQI_SPEED_PENALTY = 0.30   # multiply free-flow speed by (1 - AQI_SPEED_PENALTY × aqi/500)

# =============================================================================
# ABLATION FLAGS  (Grimm et al. 2020 ODD §"Design concepts / Interaction")
# =============================================================================
# These flags disable specific architectural components for controlled ablation
# experiments.  Setting one flag to True produces a reduced model that can be
# compared statistically against the full model to isolate the contribution of
# each component.
#
# Ablation methodology follows:
#   Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
#   Other Simulation Models: A Second Update to Improve Clarity, Replication,
#   and Structural Realism." JASSS 23(2):7.
#   DOI: 10.18564/jasss.4259
#   [ODD "Design concepts" section requires documenting which sub-models can
#   be switched off to test their individual contribution to emergent outcomes.]
#
# Ablation A — DISABLE_CNP:
#   Replace Contract Net Protocol bidding (Smith 1980) with random assignment.
#   Demonstrates how much CNP coordination improves suppression / rescue outcomes.
# Ablation B — DISABLE_PANIC:
#   Remove the three-state cognitive machine (Cova & Johnson 2002).
#   All civilians stay rational (panic = 0) throughout.  Shows how panic-driven
#   herding and gridlock affect evacuation success rate.
DISABLE_CNP   = False   # True → random task assignment instead of CNP bidding
DISABLE_PANIC = False   # True → civilians always rational, no panic model

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
