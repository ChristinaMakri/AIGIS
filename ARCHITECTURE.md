# AIGIS System Architecture

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Agent Architectures](#2-agent-architectures)
3. [Communication Topology](#3-communication-topology)
4. [Core Algorithms](#4-core-algorithms)
5. [Data Flow](#5-data-flow)
6. [Performance](#6-performance)

---

## 1. System Overview

AIGIS is a Multi-Agent System (MAS) for wildfire disaster management. Eight autonomous agents with different architectural patterns collaborate in a shared stochastic environment.

### Design Principles
- **Location-Agnostic**: any geographic coordinates via OpenStreetMap
- **Physics-Based**: Rothermel fire, Greenshields traffic, CAMS smoke, FWI risk
- **Bibliography-Grounded**: every model equation cites its source (see PHYSICS_MODELS.md)
- **Layered Communication**: Sensing -> Analysis -> Command -> Field (no Sentinel-to-Commander shortcuts)

### Environment State (shared across all agents)

| Grid | Type | Description |
|------|------|-------------|
| `fire_grid` | int8 | 0=no fuel, 1=burning, 2=burnt, 3=fuel |
| `temperature_grid` | float32 | 0–100 C for risk assessment |
| `smoke_grid` | float32 | 0–1 smoke concentration (CAMS model) |
| `elevation_grid` | float32 | SRTM elevation in metres |
| `fuel_type_grid` | int8 | NFFL fuel model 0–13 |
| `obstacle_grid` | int8 | 0=passable, 1=obstacle |
| `ignition_risk_grid` | float32 | 0–1 pre-ignition risk (RiskMonitor) |
| `firms_density` | float32 | 0–1 historical ignition density (FIRMS) |

Scalar state: `fwi_data`, `air_quality_index`, `hospital_nodes`, `safe_nodes`, `step_count`

---

## 2. Agent Architectures

### 2.1 Sentinel — Reactive (Signal Detection Theory)

**Architecture**: Simple Reflex Agent

**Decision rule**:
```
IF I_detected > THRESHOLD for 3 consecutive steps:
    SEND FIRE_DETECTION INFORM to analyst
```

Detection equation (Green & Swets 1966):
```
I_detected = [I_actual / (d^2 + epsilon)] x (1 + cos(theta)) + N(0, sigma)
```

No memory. Purely input-output. Four instances placed at map corners.

---

### 2.2 Analyst — BDI Information Processing

**Architecture**: BDI with internal fire model

**Beliefs**: `fire_reports` list, `current_phase`, `suppressed_cells`

**Perceive**:
- Collect `FIRE_DETECTION` INFORM from sentinels -> append to `fire_reports`
- `PHASE_UPDATE` INFORM from Commander -> update `current_phase`
- `SUPPRESSION_UPDATE` INFORM from firefighters -> add cell to `suppressed_cells`

**Decide**:
- Filter `fire_reports` by `suppressed_cells` (remove extinguished cells)
- Compute ROS and TTI using Rothermel (1972)
- If `current_phase >= 2`: apply 20% TTI conservatism (`TTI *= 0.80`)

**Act**:
- Send `FIRE_ANALYSIS` INFORM to Commander: `{tti, ros, num_exits, risk_level}`

---

### 2.3 Commander — BDI + Hybrid (ECT vs TTI, CNP Manager)

**Architecture**: BDI with utility-based phase selection; CNP Manager role

**Perceive**:
- `FIRE_ANALYSIS` INFORM from Analyst -> update `tti_value`
- `RISK_FORECAST` INFORM from RiskMonitor -> update pre-fire positioning
- `PROPOSE` from Rescuers/Firefighters/Ambulances -> accumulate proposals
- `CONFIRM` from field units -> remove completed missions from tracking dicts

**Decide** (4-phase logic):
```
ECT = (N_civilians / (C_exit x num_exits)) x gamma_congestion

phase = 0 if TTI > 2.5 x ECT
phase = 1 if TTI > 1.5 x ECT   # dispatch firefighters
phase = 2 if TTI > 1.0 x ECT   # dispatch all field units
phase = 3 otherwise              # shelter-in-place
```

**Act** (phase-specific):
- Phase 0: no action
- Phase 1: WARNING to civilians + FIRE_SUPPRESSION_CFP to firefighters
- Phase 2: EVACUATE to civilians + AMBULANCE_CFP + RESCUE_CFP + FIRE_SUPPRESSION_CFP
- Phase 3: REDIRECT_TO_SAFE_ZONE to civilians

On phase transition: send `PHASE_UPDATE` INFORM to Analyst.

Mission tracking dicts: `active_missions`, `ambulance_missions`, `firefighter_missions`
All cleaned up on matching CONFIRM messages.

---

### 2.4 RiskMonitor — Model-Based BDI (Pre-Ignition)

**Architecture**: Model-Based BDI

**Runs**: every `RISK_MONITOR_UPDATE_INTERVAL` steps (default 20)

**Compute** (Van Wagner 1987, Anderson 1982, Schroeder 2014):
```
risk = 0.40 x fwi_factor + 0.30 x fuel_factor + 0.20 x firms_factor + 0.10 x slope_factor
```

**Act**:
- Write `environment.ignition_risk_grid`
- Send `RISK_FORECAST` INFORM to Commander: top-3 risk cells (lat/lon), fwi, max_risk, mean_risk

Enables Commander to pre-position firefighters before fire ignition.

---

### 2.5 Firefighter — BDI + Utility (CNP Contractor)

**Architecture**: BDI with utility-based intention selection; CNP Contractor

**Mission states**: IDLE -> ASSIGNED -> SUPPRESSING

**CNP flow**:
```
Commander --[FIRE_SUPPRESSION_CFP]--> Firefighter
Firefighter --[PROPOSE {cost, eta}]--> Commander
Commander --[ACCEPT_PROPOSAL]--> Firefighter  (sets mission_status=ASSIGNED)
Firefighter --[CONFIRM {status:COMPLETED}]--> Commander
```

**Utility function** (decides between water_drop / fire_line / backburn):
```
U = w_threat x Threat + w_efficiency x Efficiency + w_coordination x Coordination
  w_threat=0.5, w_efficiency=0.3, w_coordination=0.2
```

**On successful water_drop**:
- Sends `SUPPRESSION_UPDATE` INFORM to Analyst (cell removed from TTI calculation)
- Sends `CONFIRM` to Commander (mission cleaned up)

**Fire-line placement** (Rothermel 1972): perpendicular to wind vector, 2 cells downwind.

---

### 2.6 Rescuer — BDI (Goal-Based CNP Contractor)

**Architecture**: BDI with goal-directed navigation; CNP Contractor

**Mission states**: IDLE -> TO_TARGET -> RETURNING

**Path risk check**: Scans `temperature_grid` along A* path. Refuses if `max_temp > RESCUER_SAFETY_THRESHOLD`.

**Bid formula**:
```
cost = path_length / speed + path_risk x RESCUER_RISK_ALPHA + (100 - fuel)
```

**Dynamic re-routing**: path recalculated every `RESCUER_PATH_RECALC_INTERVAL` steps or when next node is on fire.

---

### 2.7 Ambulance — BDI (Two-Phase Goal Stack, CNP Contractor)

**Architecture**: BDI with two-leg goal stack; CNP Contractor

**Mission states**: IDLE -> TO_SCENE -> TO_HOSPITAL -> RETURNING

**Two dispatch paths**:

1. Commander CFP (`AMBULANCE_CFP`): standard CNP bidding with `scene_node` and `hospital_node`
2. Civilian INJURY_REPORT: direct self-dispatch — bypasses Commander CFP when smoke casualty needs immediate response (Inness 2019)

**Safety**: aborts mission if next path node is in active fire. Refuses CFP if path risk exceeds `AMBULANCE_RISK_THRESHOLD`.

---

### 2.8 Civilian — BDI (Three-State Cognitive Machine)

**Architecture**: BDI with crowd dynamics and smoke injury model

**Three cognitive states** (Cova & Johnson 2002):

| State | Panic | Behavior |
|-------|-------|----------|
| Rational | < 0.4 | Optimal A* to safety_node |
| Confused | 0.4–0.7 | A* at 50% speed, stochastic re-route |
| Herding | >= 0.7 | Social Force crowd-following |

**Smoke injury loop** (Inness 2019):
```
smoke_exposure += smoke_grid[r, c]   each step
if smoke_exposure >= threshold:
    is_injured = True
    send INJURY_REPORT to ambulances
    halt movement
```

**Herding safety**: if crowd direction leads to burning cell, falls back to `_move_to_safety()`.

**Social Force**: inverse-distance-weighted average of nearby agents' `last_movement` vectors.

---

## 3. Communication Topology

Layered architecture — messages flow between defined pairs only:

```
Sensing Layer
  FIRMS/FWI data ---------> RiskMonitor
  fire_grid/temperature ---> Sentinel

Analysis Layer
  Sentinel  --[FIRE_DETECTION INFORM]---------> Analyst
  Analyst   --[FIRE_ANALYSIS INFORM]----------> Commander
  RiskMonitor --[RISK_FORECAST INFORM]--------> Commander

Command Layer
  Commander --[PHASE_UPDATE INFORM]-----------> Analyst
  Commander --[WARNING / FWI_WARNING INFORM]--> Civilians (broadcast)
  Commander --[EVACUATE REQUEST]--------------> Civilians (broadcast)
  Commander --[REDIRECT_TO_SAFE_ZONE REQUEST]-> Civilians (broadcast)

Field Coordination (Contract Net Protocol)
  Commander  --[FIRE_SUPPRESSION_CFP CFP]-----> Firefighters
  Firefighter --[PROPOSE]------------------> Commander
  Firefighter --[REFUSE]-------------------> Commander
  Commander  --[ACCEPT/REJECT_PROPOSAL]-----> Firefighter
  Firefighter --[CONFIRM]-----------------> Commander

  Commander  --[RESCUE_CFP CFP]-------------> Rescuers
  Rescuer    --[PROPOSE/REFUSE/CONFIRM]-----> Commander

  Commander  --[AMBULANCE_CFP CFP]----------> Ambulances
  Ambulance  --[PROPOSE/REFUSE/CONFIRM]-----> Commander

Feedback Loops
  Firefighter --[SUPPRESSION_UPDATE INFORM]-> Analyst
  Civilian    --[INJURY_REPORT INFORM]------> Ambulances (direct dispatch)
```

### Message Types Reference

| Performative | Used By | Purpose |
|---|---|---|
| INFORM | all | Share information (detections, risk reports, status updates) |
| REQUEST | Commander | Issue orders to civilians |
| CFP | Commander | Call For Proposal (CNP initiation) |
| PROPOSE | Rescuer, Firefighter, Ambulance | CNP bid |
| ACCEPT_PROPOSAL | Commander | Award mission |
| REJECT_PROPOSAL | Commander | Reject bid |
| REFUSE | Rescuer, Firefighter, Ambulance | Decline CFP (too dangerous / no resources) |
| CONFIRM | Rescuer, Firefighter, Ambulance | Mission completed |

### INFORM Content Types

| type field | Sender -> Receiver | Description |
|---|---|---|
| FIRE_DETECTION | Sentinel -> Analyst | Fire location, intensity |
| FIRE_ANALYSIS | Analyst -> Commander | TTI, ROS, risk level |
| RISK_FORECAST | RiskMonitor -> Commander | Pre-ignition risk zones, FWI |
| PHASE_UPDATE | Commander -> Analyst | Current phase number |
| SUPPRESSION_UPDATE | Firefighter -> Analyst | Extinguished cell coordinates |
| INJURY_REPORT | Civilian -> Ambulances | Smoke-injured civilian location |
| WARNING | Commander -> Civilians | Pre-evacuation alert |
| FWI_WARNING | Commander -> Civilians | Pre-fire weather warning |

---

## 4. Core Algorithms

### Fire Spread (Vectorized Cellular Automaton)

Each step:
1. Compute burnout: `random < burnout_prob_grid` -> state 1 -> state 2
2. Count burning neighbours via `scipy.signal.convolve2d` (O(WH) vs O(WH x 8) naive)
3. For each of 8 Moore directions: compute `spread_prob = base x wind_factor x slope_factor x neighbour_factor x fuel_factor`
4. Stochastic ignition: `random < spread_prob` -> state 3 -> state 1
5. Update temperature grid
6. Update smoke grid (advection-diffusion)

### Smoke Diffusion (Explicit Finite Difference)

Each step, operating on `smoke_grid`:
1. Source emission from burning cells
2. Wind advection: shift grid by wind unit vector x SMOKE_WIND_ADVECTION fraction
3. Isotropic diffusion: 4-neighbour average x SMOKE_DIFFUSION_RATE
4. Atmospheric decay: multiply by (1 - SMOKE_DECAY_RATE)

### A* Navigation

All mobile agents (Rescuer, Ambulance, Civilian) use `networkx.shortest_path(..., weight='length')` on the OSM road network graph. Path recalculation is staggered with a random per-agent offset to avoid all agents recalculating simultaneously.

### Contract Net Protocol (CNP) Lifecycle

```
1. Commander detects need (fire, rescue, medical)
2. Commander generates mission_id and broadcasts CFP to relevant group
3. Contractors evaluate path safety and resources
4. Qualified contractors send PROPOSE; unqualified send REFUSE
5. Commander selects best bid (minimum cost)
6. Commander sends ACCEPT to winner, REJECT to others
7. Contractor executes mission
8. Contractor sends CONFIRM on completion
9. Commander removes mission from tracking dict
```

### Pre-Ignition Risk Assessment

```
1. FWIConnector.fetch() -> fwi_data dict
2. FIRMSConnector.build_ignition_density() -> firms_density grid
3. RiskMonitorAgent._recompute_risk_grid():
   a. fwi_factor = min(fwi_score / 60, 1.0)
   b. fuel_factor = fuel_type_grid spread_multipliers normalised
   c. hist_factor = firms_density
   d. slope_factor = gradient magnitude normalised
   e. risk = 0.4*fwi + 0.3*fuel + 0.2*hist + 0.1*slope
4. environment.ignition_risk_grid = risk
5. Commander receives RISK_FORECAST; pre-positions firefighters
```

---

## 5. Data Flow

### Initialization Sequence

```
1. LiveMapBuilder.build():
   a. Download OSM graph (roads, buildings)
   b. SRTM elevation download (or Perlin fallback)
   c. CORINE land cover -> NFFL fuel_type_grid
   d. Identify safe_nodes (OSM tags + perimeter)
2. Load live data (parallel connectors):
   - Open-Meteo weather -> fwi_data, wind_speed, temperature, humidity
   - FIRMS VIIRS -> firms_density
   - OpenAQ -> air_quality_index
   - OSM EMS -> hospital_nodes
3. FireSimulation.__init__() -> wind model initialised
4. _ignite_fires() -> FIRMS/highest-elevation fuel cells
5. _initialize_agents() -> spawn 8 agent types at OSM positions
6. environment.fire_simulation = fire_sim  (wind access for firefighters)
```

### Per-Step Loop

```
Step N:
  1. fire_sim.step():
       - Wind direction update
       - Burnout (vectorized)
       - Fire spread (8-directional, vectorized)
       - Temperature grid update
       - Smoke grid update (advection-diffusion)

  2. _update_agents() — all agents in order:
       RiskMonitors -> Sentinels -> Analyst -> Commander
       -> Rescuers -> Firefighters -> Civilians -> Ambulances
       Each: perceive(env) -> decide() -> act(env)
       Clear inboxes after each group

  3. _route_messages():
       Collect all agent outboxes
       Route by receiver string: "analyst" / "commander" / "ambulances" /
         "firefighters" / "rescuers" / "broadcast" / direct agent_id
       Track REFUSE count for metrics

  4. _update_civilian_neighbors():
       Each civilian: find_nearby_agents() for Social Force herding

  5. _collect_metrics():
       casualties, evacuated, injured, panic_levels, fire stats, phase

  6. is_complete() check
```

### Post-Simulation

```
_print_final_report():
  steps, phase, evacuated %, casualties %, smoke-injured,
  burnt cells, firefighter water use, rescuer refusals,
  avg/peak panic

get_results() -> dict for CSV export (batch mode)
```

---

## 6. Performance

### Computational Complexity

| Component | Complexity | Optimization |
|-----------|------------|--------------|
| Fire spread | O(W x H) | scipy.convolve2d replaces 8 nested loops |
| Smoke diffusion | O(W x H) | Vectorized numpy array operations |
| Sentinel sensing | O(R^2) per sentinel | Bounding box + circular check |
| Agent pathfinding | O(E log V / N_steps) | Staggered recalc every 20 steps |
| CNP bid evaluation | O(N_proposals) | Evaluated once per phase step |
| Civilian herding | O(N_civ x R^2) | Limited vision radius = 10 cells |

W, H = grid dimensions; R = detection radius; E, V = OSM graph edges/vertices

### Key Optimizations

**Vectorized fire spread**: `scipy.signal.convolve2d` counts burning neighbours for all cells simultaneously. All 8 spread directions processed with array slicing — no Python loops over cells.

**Staggered pathfinding**: each agent carries a `recalc_offset` drawn at init from `[0, path_recalc_interval)`. Paths recalculated only every 20 steps, offset so agents never all recalculate in the same step.

**Claimed fire cells**: `environment.claimed_fire_cells` set prevents multiple firefighters targeting the same burning cell in the same step. Reset each step.

**Message routing by string**: `_route_messages()` dispatches in O(1) for named groups (analyst, commander, ambulances, firefighters, rescuers, broadcast) before falling back to linear scan for direct agent IDs.

---

## Summary

| Agent | Architecture | Primary Protocol | Key Model |
|-------|-------------|-----------------|-----------|
| Sentinel | Reactive | — (INFORM output) | Green & Swets 1966 SDT |
| Analyst | BDI | — (INFORM output) | Rothermel 1972 TTI |
| Commander | BDI + Hybrid | CNP Manager | Wolshon 2006 ECT |
| RiskMonitor | Model-Based BDI | — (INFORM output) | Van Wagner 1987 FWI |
| Firefighter | BDI + Utility | CNP Contractor | Rothermel 1972 fire-line |
| Rescuer | BDI | CNP Contractor | — |
| Ambulance | BDI | CNP Contractor + direct | Inness 2019 smoke injury |
| Civilian | BDI | — (receives orders) | Greenshields 1935 + Inness 2019 |
