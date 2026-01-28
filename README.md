# AIGIS: AI for Guardian & Intervention Systems

A sophisticated Multi-Agent System (MAS) for disaster management simulation, focusing on wildfire response and evacuation scenarios.

## Overview

AIGIS creates a "Digital Twin" of any geographic area using live OpenStreetMap data and simulates the interaction between 5 distinct types of intelligent agents under crisis conditions. The system demonstrates how different Agent Architectures (Reactive, Model-based, BDI, Hybrid) collaborate in a stochastic, partially observable environment.

**Location-Agnostic**: AIGIS works anywhere in the world - just provide latitude/longitude coordinates.

## Key Features

### 🌍 Location-Agnostic System
- **Perlin Noise Terrain**: Realistic elevation generation without hardcoded gradients
- **OSM Safe Zone Detection**: Automatically identifies water bodies, parks, squares, and map edges
- **Universal Deployment**: Works at any location globally

### 🔥 Physics-Based Fire Simulation
- **Rothermel Fire Model**: Scientific fire spread with fuel, wind, and slope effects
- **Dynamic Wind**: Oscillating wind direction: θ(t) = θ₀ + sin(t/50) × 20°
- **Cellular Automata**: 4-state fire grid (No Fuel, Burning, Burnt, Fuel)

### 👥 Multi-Agent System
- **5 Agent Types**: Sentinel, Analyst, Commander, Rescuer, Civilian
- **FIPA-ACL Communication**: Standards-compliant agent messaging
- **Contract Net Protocol**: Task allocation between Commander and Rescuers

### 🧠 Crowd Dynamics
- **Greenshields Traffic Model**: Speed = V_free × (1 - ρ/ρ_jam) → Realistic gridlock
- **Social Force Model**: Herding behavior at high panic levels
- **3-State Cognitive Machine**: Rational → Confused → Herding

### 📊 Professional Dashboard
- **3-Panel Layout**: Main map + 2 real-time line charts
- **Safe Zone Highlighting**: Visual indicators for evacuation destinations
- **Panic Visualization**: Civilians colored by panic level (colormap)
- **Live Metrics**: Casualties and evacuations tracked over time

### 🧪 Monte Carlo Batch Mode
- **Research-Ready**: Run hundreds of experiments with single command
- **CSV Export**: Pandas DataFrame with all metrics
- **Statistical Analysis**: Mean ± std, min/max ranges automatically calculated

## Agent Types

### 1. Sentinel Agent (Reactive Architecture)
- **Role**: Fire detection sensors with environmental attenuation
- **Logic**: Signal Detection Theory with distance and wind-based attenuation
- **Features**:
  - Signal equation: `I_detected = I_actual/(d² + ε) × (1 + cos(θ)) + N(0,σ)`
  - Debouncing protocol (3 consecutive detections required)
  - Wind-aware smoke drift simulation

### 2. Analyst Agent (Model-Based / Deductive)
- **Role**: Risk assessment using Rothermel fire physics
- **Logic**: Simplified Rothermel's Surface Fire Spread Model + Fuzzy Logic
- **Features**:
  - Rate of Spread (ROS): `ROS = R_base × (1 + φ_wind) × (1 + φ_slope)`
  - Time To Impact (TTI): `TTI = Distance / ROS`
  - Escape route bottleneck detection
  - Fuzzy rules based on TTI and exit availability

### 3. Commander Agent (Hybrid / Utility-Based)
- **Role**: Strategic decision making using ECT vs TTI comparison
- **Logic**: Evacuation Clearance Time (ECT) calculation and phase-based protocol
- **Features**:
  - ECT calculation: `ECT = (N_agents / C_exit) × γ`
  - 4-Phase Decision Protocol:
    - **Phase 0**: Monitoring (TTI > 2.5×ECT)
    - **Phase 1**: Pre-Evacuation Warning (1.5×ECT < TTI ≤ 2.5×ECT)
    - **Phase 2**: Mass Evacuation (1.0×ECT < TTI ≤ 1.5×ECT)
    - **Phase 3**: **Shelter-in-Place** (TTI ≤ ECT) - Redirect to nearest safe zone
  - Dynamic safe zone routing (water, parks, map edges)

### 4. Rescuer Agent (Goal-Based / Practical Reasoning)
- **Role**: Execute rescue missions with risk assessment
- **Logic**: Contract Net Protocol with path risk evaluation
- **Features**:
  - Path risk assessment: Scans temperature grid along route
  - Risk-adjusted bidding: `Cost = time + (Risk_path × α) + fuel`
  - Safety protocol: **Refuses missions through active fire**
  - A* pathfinding for navigation

### 5. Civilian Agent (BDI Architecture)
- **Role**: Evacuate with realistic traffic physics and panic psychology
- **Architecture**: BDI with Greenshields' Traffic Model + 3-State Cognitive Machine
- **Features**:
  - **Traffic Model**: `V = V_free × (1 - ρ_local/ρ_jam)` → Gridlock at jam density
  - **Cognitive States**:
    - Rational (Panic < 0.4): Optimal A* pathfinding to nearest safe zone
    - Confused (0.4-0.7): 50% speed reduction, frequent re-routing
    - Herding (> 0.7): Follows crowd (Social Force), even to dead ends
  - **Panic Equation**: `Panic(t) = Panic(t-1) + α×(1/d_fire) + β×(family)`
  - Distance-based panic with family separation factor
  - Dynamic safe zone detection (parks, water, map edges)

## Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- numpy: Numerical operations
- pandas: Data analysis and CSV export
- osmnx: OpenStreetMap integration
- networkx: Graph-based pathfinding
- matplotlib: Visualization and dashboard
- scikit-fuzzy: Fuzzy logic for analyst
- shapely: Geometric operations
- rasterio: Geospatial raster data
- geopandas: Geospatial data frames
- noise: Perlin noise terrain generation

## Usage

### CLI Arguments

```bash
python main.py [OPTIONS]
```

**Options:**
- `--lat LAT` : Center latitude (default: 38.627)
- `--lon LON` : Center longitude (default: -90.1994)
- `--radius RADIUS` : Map radius in meters (default: 1500)
- `--batch N` : Run N Monte Carlo experiments (batch mode)
- `--output FILE` : CSV output file for batch mode (default: results.csv)
- `--mode {gui,headless}` : Visualization mode (default: gui)

### Examples

#### 1. Quick Start (Default Location)
```bash
python main.py
```
Runs single simulation with live dashboard at St. Louis, MO.

#### 2. Custom Location (Los Angeles)
```bash
python main.py --lat 34.0522 --lon -118.2437 --radius 2500
```
Runs simulation in Los Angeles with 2.5km radius.

#### 3. Monte Carlo Experiment (50 runs)
```bash
python main.py --batch 50 --output results.csv
```
Runs 50 simulations at default location, exports metrics to CSV.

#### 4. Research Experiment (California Forest)
```bash
python main.py --lat 36.7783 --lon -119.4179 --radius 3000 --batch 100 --output ca_forest.csv
```
Runs 100 simulations in California forest region for statistical analysis.

#### 5. HPC Cluster Mode (Headless)
```bash
python main.py --batch 1000 --mode headless --output hpc_results.csv
```
Runs 1000 simulations without GUI for high-performance computing environments.

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or use the run script
./docker-run.sh
```

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

## Architecture

```
AIGIS/
├── main.py                    # CLI entry point with argparse
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PHYSICS_MODELS.md         # Scientific model documentation
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── src/
    ├── config.py             # Configuration constants
    ├── message.py            # FIPA-ACL message implementation
    ├── environment.py        # LiveMapBuilder with Perlin terrain
    ├── fire_simulation.py    # Fire model with dynamic wind
    ├── simulation.py         # Main simulation engine with metrics
    ├── dashboard.py          # 3-panel professional dashboard
    └── agents/
        ├── base_agent.py     # Abstract base agent
        ├── sentinel.py       # Reactive agent (Signal Detection)
        ├── analyst.py        # Model-based agent (Rothermel + Fuzzy)
        ├── commander.py      # Hybrid agent (ECT vs TTI)
        ├── rescuer.py        # Goal-based agent (Contract Net)
        └── civilian.py       # BDI agent (Greenshields + Social Force)
```

## Configuration

Edit `src/config.py` to customize:

### Location Parameters
```python
MAP_CENTER_LAT = 38.627     # Default: St. Louis, MO
MAP_CENTER_LON = -90.1994
MAP_RADIUS = 1500           # meters
GRID_HEIGHT = 100
GRID_WIDTH = 100
```

### Agent Population
```python
NUM_SENTINELS = 4           # Fire detection sensors
NUM_RESCUERS = 3            # Rescue teams
NUM_CIVILIANS = 20          # Evacuees
```

### Fire Parameters
```python
WIND_INITIAL_DIRECTION = 90.0      # degrees
WIND_OSCILLATION_PERIOD = 50.0     # steps
WIND_OSCILLATION_AMPLITUDE = 20.0  # degrees
FIRE_SPREAD_PROBABILITY = 0.4
```

### Perlin Terrain
```python
PERLIN_SCALE = 100.0
PERLIN_OCTAVES = 4
PERLIN_BASE_HEIGHT = 100.0
PERLIN_AMPLITUDE = 50.0
```

### Safe Zone Detection (OSM Tags)
```python
SAFE_ZONE_TAGS = {
    'natural': ['water', 'beach', 'coastline'],
    'leisure': ['park', 'nature_reserve', 'playground'],
    'place': ['square']
}
```

## Scientific Models

### 1. Fire Spread (Rothermel Model)
```
ROS = R_base × (1 + φ_wind) × (1 + φ_slope)
φ_wind = C × U^B × (direction_alignment)
φ_slope = 5.275 × (tan(slope))^2
```

### 2. Dynamic Wind Model
```
θ(t) = θ₀ + sin(t / T_period) × A_amplitude
Updates every simulation step
```

### 3. Traffic Model (Greenshields)
```
V_current = V_free_flow × (1 - ρ_local / ρ_jam)
When ρ_local ≥ ρ_jam → Gridlock (V = 0)
```

### 4. Panic Equation
```
Panic(t) = Panic(t-1) + α × (1/d_fire) + β × (Family_Separated?)
Decays when fire not visible
```

### 5. Social Force Model (Herding)
```
Movement = Σ(nearby_agents.direction) / N
Applied when Panic ≥ 0.7
```

### 6. ECT vs TTI Decision Logic
```
ECT = (N_agents / C_exit) × γ_congestion
Phase = f(TTI / ECT ratio)
```

## Monte Carlo Output

When running with `--batch N`, AIGIS exports a CSV file with these columns:

| Column | Description |
|--------|-------------|
| `run_id` | Experiment run number |
| `lat`, `lon`, `radius` | Location parameters |
| `steps` | Simulation duration |
| `total_civilians` | Population size |
| `casualties` | Number of deaths |
| `evacuated` | Successful evacuations |
| `mortality_rate` | Death rate (0.0-1.0) |
| `evacuation_success_rate` | Success rate (0.0-1.0) |
| `avg_panic_level` | Mean panic level |
| `max_panic_level` | Peak panic observed |
| `rescuer_refusals` | Refused missions |
| `max_fire_cells` | Peak fire intensity |
| `final_phase` | Commander's final phase (0-3) |

Statistical summary (mean ± std, ranges) is printed to console.

## Visualization

### Dashboard Layout (GUI Mode)
- **Left Panel**: Main map with fire grid, safe zones, and agents
- **Top-Right Panel**: Casualties over time (red line chart)
- **Bottom-Right Panel**: Successful evacuations (green line chart)

### Legend
- 🟢 Green: Fuel (forest/vegetation)
- 🟠 Orange: Actively burning
- ⚫ Dark Gray: Burnt out
- 🟢 Light Green Glow: Safe zones (water, parks, edges)
- 🟡 Gold Circle: Sentinel
- 🟣 Purple Square: Analyst
- 🔴 Red Triangle: Commander
- 🔵 Blue Diamond: Rescuer
- 🌈 Color-coded Dots: Civilians (green=calm → red=panic)

## Performance Notes

- **Grid size**: Smaller grids (100×100) run faster
- **Map radius**: Affects OSM data fetching time
- **Agent count**: More agents increase computation
- **Headless mode**: Faster for batch experiments (no rendering)

## Research Applications

AIGIS is designed for:
- Testing evacuation strategies
- Comparing agent architectures
- Analyzing panic psychology effects
- Validating traffic flow models
- Studying fire-evacuation scenarios
- Monte Carlo sensitivity analysis

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture and business logic
- [PHYSICS_MODELS.md](PHYSICS_MODELS.md) - Detailed scientific model documentation
- [DOCKER.md](DOCKER.md) - Docker deployment guide

## Future Extensions

- Multiple simultaneous fires
- Dynamic obstacle creation (road closures)
- Agent learning and adaptation
- 3D terrain visualization
- Network communication delays
- Resource constraints (fuel, supplies)
- Historical scenario playback

## License

MIT License

## Citation

If you use AIGIS in your research, please cite:
```
AIGIS: AI for Guardian & Intervention Systems
Multi-Agent Wildfire Evacuation Simulation
https://github.com/ChristinaMakri/AIGIS
```

