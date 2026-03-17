# AIGIS: AI for Guardian & Intervention Systems

A Multi-Agent System (MAS) for wildfire disaster management simulation. AIGIS creates a Digital Twin of any geographic area using live OpenStreetMap data and simulates the interaction between 8 distinct types of intelligent agents under crisis conditions.

**Location-Agnostic**: AIGIS works anywhere in the world. Provide latitude/longitude coordinates and it builds the environment from real data.

---

## Key Features

### Location-Agnostic Environment
- Real SRTM elevation data (NASA Shuttle Radar Topography Mission, ~30 m resolution)
- Perlin noise terrain fallback when real elevation is unavailable
- OSM safe zone detection: water bodies, parks, squares, and map perimeter edges
- CORINE land cover integration for realistic NFFL fuel model assignment

### Physics-Based Fire Simulation
- Rothermel (1972) fire spread: `ROS = R_base x (1 + phi_wind) x (1 + phi_slope)`
- Dynamic wind oscillation: `theta(t) = theta_0 + sin(t/T) x A`
- 4-state cellular automaton: No Fuel / Burning / Burnt / Fuel
- 13 NFFL fuel models (Anderson 1982) with per-type spread rate and burnout probability
- Advection-diffusion smoke plume (Inness et al. 2019 — CAMS)

### Multi-Agent System (8 Agent Types)
- FIPA-ACL standards-compliant messaging
- Contract Net Protocol (Smith 1980) for task allocation
- BDI architecture (Rao & Georgeff 1995) as the foundation for most agents
- Layered communication topology: Sensing -> Analysis -> Command -> Field

### Crowd and Evacuation Dynamics
- Greenshields (1935) traffic model: `V = V_free x (1 - rho/rho_jam)` — realistic gridlock
- Social Force herding model at high panic
- 3-state cognitive machine: Rational -> Confused -> Herding
- Cumulative smoke exposure injury model
- Gridlock bypass: after 3 consecutive speed=0 steps, civilian switches to direct grid-space perimeter movement
- Trapped casualty detection: after 30 steps of zero movement (obstacle-enclosed), civilian is marked as casualty

### Canadian Fire Weather Index (FWI)
- Pre-ignition risk grid: FWI (40%) + fuel type (30%) + FIRMS density (20%) + slope (10%)
- Van Wagner (1987) FWI chain: FFMC -> DMC -> DC -> ISI -> BUI -> FWI
- Commander pre-positions assets to high-risk zones before fire starts

### Real Data Integration
- Open-Meteo: live temperature, humidity, wind
- NASA FIRMS VIIRS: historical ignition density (Schroeder et al. 2014)
- OpenAQ: air quality index (PM2.5 proxy for smoke)
- OSM EMS connector: hospital and ambulance station nodes

### Real-Time Web Dashboard
- Live browser dashboard at `http://localhost:5000` via Flask + Server-Sent Events
- 7 Plotly.js panels: fire grid + agents, evacuation timeline, panic histogram, fire spread metrics, AQI/smoke injuries, Commander phase strip, firefighter water gauges
- Full simulation history replayed to late-joining browser connections
- `--web` flag; `--web-port` to override port

### Post-Simulation Report
Printed automatically after each run:
- Evacuation rate, casualty rate, smoke-injured count
- Burnt area, firefighter water use, rescuer refusals
- Average and peak panic levels

### Monte Carlo Batch Mode
- Run hundreds of experiments with a single command
- CSV export with all metrics
- Mean, std, min, max printed to console

---

## Agent Types

### 1. Sentinel (Reactive — Signal Detection Theory)
Distributed fire sensors placed at map corners. Detects fire via attenuated signal:
```
I_detected = [I_actual / (d^2 + epsilon)] x (1 + cos(theta)) + N(0, sigma)
```
Three consecutive detections required before alerting Analyst (debouncing).

### 2. Analyst (BDI — Information Processing)
Reads fire reports from Sentinels. Computes Rate of Spread and Time To Impact using Rothermel (1972):
```
ROS = R_base x (1 + phi_wind) x (1 + phi_slope)
TTI = distance_to_population / ROS
```
Applies 20% TTI conservatism in Phase 2+. Filters already-suppressed cells from reports. Reports risk to Commander.

### 3. Commander (BDI + Hybrid — ECT vs TTI + PPO)
Compares Evacuation Clearance Time against Time To Impact to select one of four phases:

| Phase | Condition | Action |
|-------|-----------|--------|
| 0: Monitoring | TTI > 2.5 x ECT | Standby |
| 1: Pre-Alert | 1.5 x ECT < TTI <= 2.5 x ECT | WARNING to civilians; dispatch firefighters |
| 2: Mass Evacuation | 1.0 x ECT < TTI <= 1.5 x ECT | EVACUATE order; dispatch rescuers, ambulances, firefighters |
| 3: Shelter-in-Place | TTI <= ECT | REDIRECT to nearest safe zone |

Receives pre-fire RISK_FORECAST from RiskMonitor to pre-position assets. Coordinates all field units via CNP.

### 4. RiskMonitor (Model-Based BDI — Pre-Ignition Risk)
Runs before fire starts. Builds an ignition risk grid from four components:
```
risk = 0.40 x fwi_factor + 0.30 x fuel_factor + 0.20 x firms_factor + 0.10 x slope_factor
```
Sends top-3 high-risk zones to Commander as RISK_FORECAST messages.

### 5. Firefighter (BDI + Utility + PPO — Suppression)
CNP contractor for fire suppression. Three strategies evaluated by utility function:
- water_drop: 80% success rate, consumes 500 gal
- fire_line: removes fuel perpendicular to wind direction (Rothermel 1972)
- backburn: controlled pre-burn ahead of fire front

On successful suppression: sends SUPPRESSION_UPDATE to Analyst (removes cell from TTI calculation) and CONFIRM to Commander.

### 6. Rescuer (BDI + PPO — Goal-Based CNP Contractor)
Bids on rescue missions from Commander. Path risk evaluated against temperature grid. Refuses missions through active fire. Executes A* navigation to target, recalculates on fire blockage.

### 7. Ambulance (BDI — Two-Phase Goal Stack)
Two-leg mission: scene -> hospital. Dispatched via Commander AMBULANCE_CFP or directly by civilian INJURY_REPORT (smoke incapacitation bypass). Refuses paths through fire above safety threshold.

### 8. Civilian (BDI — Three-State Cognitive Machine)
Most complex agent. Accumulates smoke exposure from smoke_grid each step:
- When `smoke_exposure >= CIVILIAN_INJURY_THRESHOLD`: becomes injured, freezes, sends INJURY_REPORT to ambulances
- Panic equation: `Panic(t) = Panic(t-1) + alpha x (1/d_fire) + beta x (family_separated) - decay`
- Smoke contributes additional panic: `panic += smoke_conc x CIVILIAN_SMOKE_PANIC_SCALE`
- AQI (PM2.5) reduces movement speed
- 3 cognitive states drive movement strategy

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages: numpy, scipy, pandas, osmnx, networkx, matplotlib, scikit-fuzzy, shapely, rasterio, geopandas, noise, requests, pytest, flask

---

## Usage

### CLI

```bash
python main.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--lat` | 38.04 | Center latitude |
| `--lon` | 23.80 | Center longitude |
| `--radius` | 2000 | Map radius in meters |
| `--fire-lat LAT [LAT ...]` | — | Latitude(s) of fire ignition point(s) |
| `--fire-lon LON [LON ...]` | — | Longitude(s) of fire ignition point(s) |
| `--batch N` | — | Run N Monte Carlo experiments |
| `--output FILE` | results.csv | CSV output for batch mode |
| `--dashboard` | off | Save 7-panel PNG dashboard (aigis_dashboard.png) |
| `--web` | off | Start real-time browser dashboard at http://localhost:5000 |
| `--web-port PORT` | 5000 | Port for the web dashboard server |
| `--ambulances N` | from config | Override ambulance agent count |
| `--visualize` | off | Save 4-panel PNG visualization after run |

### Examples

```bash
# Default location (Athens)
python main.py

# Custom location with real fire ignition points
python main.py --lat 37.918 --lon 23.957 --radius 3000 \
  --fire-lat 37.920 37.915 --fire-lon 23.960 23.955

# Real-time web dashboard (opens browser automatically)
python main.py --web --lat 37.918 --lon 23.957 --radius 3000

# Static PNG dashboard
python main.py --dashboard --lat 37.918 --lon 23.957 --radius 3000

# 50-run Monte Carlo experiment
python main.py --batch 50 --output results.csv

# Batch with PNG summary
python main.py --batch 20 --dashboard --output results.csv
```

### Train ML Models

```bash
python train_models.py            # XGBoost risk predictor (100 MC runs)
python main.py                    # Uses trained models automatically
```

### Train + Evaluate Hybrid MARL

```bash
python train_marl.py --episodes 10000   # PPO curriculum training (~2–3 h CPU)
python evaluate_marl.py --runs 30       # 95% CI across training + held-out scenarios
```

---

## File Structure

```
AIGIS/
├── main.py                        # CLI entry point
├── train_models.py                # Train XGBoost risk predictor (MC batch)
├── train_marl.py                  # Train hybrid PPO agents (curriculum MARL)
├── evaluate_ml_models.py          # Evaluate ML predictor accuracy
├── evaluate_marl.py               # Evaluate trained MARL agents (95% CI)
├── validate_mati.py               # Validate against Mati 2018 fire
├── validate_campfire.py           # Validate against Camp Fire 2018
├── validate_all_incidents.py      # Diagnostic comparison: all 11 incidents
├── run_sensitivity.py             # One-at-a-time sensitivity analysis
├── run_ablation.py                # Ablation study (CNP, panic, RL flags)
├── run_thesis_experiments.sh      # Full thesis pipeline (10 steps)
├── requirements.txt
├── README.md
├── PHYSICS_MODELS.md              # Scientific model documentation
├── ARCHITECTURE.md                # System design and agent logic
├── ODD_PROTOCOL.md                # Grimm et al. (2020) ODD description
├── DOCKER.md
└── src/
    ├── config.py                  # All parameters in one place
    ├── message.py                 # FIPA-ACL message class
    ├── environment.py             # LiveMapBuilder, smoke_grid, SRTM
    ├── fire_simulation.py         # Rothermel spread + smoke diffusion
    ├── fire_predictor.py          # Step-ahead XGBoost fire predictor
    ├── simulation.py              # Main engine, metrics, final report
    ├── dashboard.py               # 7-panel matplotlib PNG dashboard
    ├── web_dashboard.py           # Real-time Flask/SSE web dashboard
    ├── ml_predictor.py            # XGBoost prediction module
    ├── parameter_adapter.py       # Monte Carlo parameter learning
    ├── templates/
    │   └── index.html             # Web dashboard frontend (Plotly.js)
    ├── data_connectors/
    │   ├── fwi_connector.py       # Open-Meteo FWI data
    │   ├── firms_connector.py     # NASA FIRMS ignition density
    │   ├── airquality_connector.py  # OpenAQ PM2.5 / AQI
    │   └── ems_connector.py       # OSM hospital node lookup
    ├── rl/
    │   ├── ppo.py                 # PPO actor-critic (Schulman et al. 2017)
    │   ├── observations.py        # Per-role observation builders (24/22/26-dim)
    │   ├── qmix.py                # QMIX monotonic mixing network (Rashid et al. 2018)
    │   ├── rewards.py             # Per-role step + terminal reward functions
    │   └── curriculum.py         # 9-scenario curriculum (Bengio et al. 2009)
    └── agents/
        ├── base_agent.py          # Abstract BDI base (perceive/decide/act)
        ├── sentinel.py            # Reactive — Signal Detection Theory
        ├── analyst.py             # BDI — Rothermel TTI + phase feedback
        ├── commander.py           # BDI + PPO — ECT/TTI + CNP Manager
        ├── risk_monitor.py        # Model-Based — pre-ignition FWI risk
        ├── firefighter.py         # BDI + Utility + PPO — water/fire-line/backburn
        ├── rescuer.py             # BDI + PPO — CNP Contractor, A* pathfinding
        ├── ambulance.py           # BDI — two-phase medical extraction
        └── civilian.py            # BDI — crowd dynamics, smoke injury
```

---

## Configuration

All parameters are in `src/config.py`. Key groups:

### Location
```python
DEFAULT_MAP_CENTER_LAT = 38.04
DEFAULT_MAP_CENTER_LON = 23.80
DEFAULT_MAP_RADIUS = 2000   # metres
GRID_HEIGHT = 200
GRID_WIDTH = 200
```

### Agent Counts
```python
NUM_SENTINELS = 4
NUM_RESCUERS = 3
NUM_FIREFIGHTERS = 2
NUM_CIVILIANS = 60
NUM_AMBULANCES = 2
NUM_RISK_MONITORS = 1
```

### Fire Physics
```python
WIND_INITIAL_DIRECTION = 90.0
WIND_OSCILLATION_PERIOD = 50.0
WIND_OSCILLATION_AMPLITUDE = 20.0
FIRE_SPREAD_PROB_BASE = 0.4
ROTHERMEL_BASE_ROS = 0.5
```

### Smoke Model
```python
SMOKE_SOURCE_STRENGTH = 0.30
SMOKE_DIFFUSION_RATE = 0.08
SMOKE_DECAY_RATE = 0.05
SMOKE_WIND_ADVECTION = 0.40
```

### Civilian Injury
```python
CIVILIAN_INJURY_THRESHOLD = 5.0
CIVILIAN_SMOKE_PANIC_SCALE = 0.02
```

---

## Scientific Models (Summary)

| Model | Equation | Reference |
|-------|----------|-----------|
| Fire Spread | `ROS = R_base x (1+phi_wind) x (1+phi_slope)` | Rothermel (1972) |
| Wind Factor | `phi_wind = C x U^B` | Rothermel (1972) |
| Slope Factor | `phi_slope = 5.275 x tan^2(theta)` | Rothermel (1972) |
| Smoke Diffusion | `dC/dt = -U.grad(C) + D.lap(C) + S` | Inness et al. (2019) |
| Traffic Flow | `V = V_free x (1 - rho/rho_jam)` | Greenshields et al. (1935) |
| Panic | `P(t) = P(t-1) + alpha/d + beta*family - decay` | Cova & Johnson (2002) |
| Fire Detection | `I_det = I/(d^2+e) x (1+cos(theta)) + N(0,sigma)` | Green & Swets (1966) |
| FWI | FFMC -> DMC -> DC -> ISI -> BUI -> FWI | Van Wagner (1987) |
| ECT | `ECT = (N / C_exit) x gamma` | Wolshon (2006) |

---

## Monte Carlo Output

CSV columns exported with `--batch N`:

| Column | Description |
|--------|-------------|
| `run_id` | Experiment run number |
| `lat`, `lon`, `radius` | Location parameters |
| `steps` | Simulation duration |
| `total_civilians` | Population size |
| `casualties` | Deaths from fire or smoke |
| `evacuated` | Successful evacuations |
| `injured` | Smoke-incapacitated civilians |
| `mortality_rate` | Death rate 0.0–1.0 |
| `evacuation_success_rate` | Success rate 0.0–1.0 |
| `avg_panic_level` | Mean panic across all steps |
| `max_panic_level` | Peak panic observed |
| `rescuer_refusals` | Refused missions (path blocked) |
| `max_fire_cells` | Peak simultaneous burning cells |
| `final_phase` | Commander final phase (0–3) |
| `burned_area_pct` | Percentage of grid cells burned out |
| `burned_area_ha` | Absolute burned area in hectares |

---

## Docker

```bash
docker-compose up --build
# or
./docker-run.sh
```

See [DOCKER.md](DOCKER.md) for details.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Complete agent logic, communication flows, data pipeline
- [PHYSICS_MODELS.md](PHYSICS_MODELS.md) — All scientific models with equations and references
- [DOCKER.md](DOCKER.md) — Docker deployment guide

---

## Full Bibliography

1. Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels*. USDA Forest Service Research Paper INT-115.
2. Anderson, H.E. (1982). *Aids to Determining Fuel Models for Estimating Fire Behavior*. USDA Forest Service GTR INT-122.
3. Van Wagner, C.E. (1987). *Development and Structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35.
4. Van Wagner, C.E. & Pickett, T.L. (1985). *Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System*. Forestry Technical Report 33.
5. Schroeder, W. et al. (2014). "The New VIIRS 375 m active fire detection data product." *Remote Sensing of Environment*, 143, pp. 85–96.
6. Inness, A. et al. (2019). "The CAMS reanalysis of atmospheric composition." *Atmos. Chem. Phys.*, 19(6), pp. 3515–3556.
7. Green, D.M. & Swets, J.A. (1966). *Signal Detection Theory and Psychophysics*. Wiley.
8. Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to practice." *ICMAS-95*, pp. 312–319.
9. Smith, R.G. (1980). "The Contract Net Protocol." *IEEE Trans. Computers*, C-29(12), pp. 1104–1113.
10. Greenshields, B.D. et al. (1935). "A study of traffic capacity." *Highway Research Board Proceedings*, 14, pp. 448–477.
11. Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood evacuations in the urban-wildland interface." *Environment and Planning A*, 34(12), pp. 2211–2230.
12. Wolshon, B. (2006). "Evacuation planning and engineering for Hurricane Katrina." *The Bridge*, 36(1), pp. 27–34.
13. Lagouvardos, K. et al. (2019). "Meteorological analysis of the catastrophic wildfire in Mati, eastern Attica, Greece." *BAMS*, 100(11), pp. 2243–2257.
14. Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.
15. Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). "Curriculum Learning." *ICML-09*, pp. 41–48.
16. Lowe, R. et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NIPS*, pp. 6379–6390.
17. Mas, E. et al. (2021). "An interdisciplinary agent-based multimodal wildfire evacuation model." *Transportation Research Part D*, 99, 103007.
18. Mann, H.B. & Whitney, D.R. (1947). "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics*, 18(1), pp. 50–60.

---

## License

MIT License

## Citation

If you use AIGIS in your research:
```
AIGIS: AI for Guardian & Intervention Systems
Multi-Agent Wildfire Evacuation Simulation
https://github.com/ChristinaMakri/AIGIS
```
