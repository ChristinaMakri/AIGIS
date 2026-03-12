# AIGIS Physics Models Documentation

All mathematical and empirical models implemented in AIGIS. Each section gives the equation, parameter table, config keys, implementation file, and primary reference.

---

## 1. Fire Spread — Rothermel Model

### Reference
Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels*. USDA Forest Service Research Paper INT-115.

### Equations

Rate of Spread (ROS):
```
ROS = R_base x (1 + phi_wind) x (1 + phi_slope) x fuel_factor
```

Wind factor (Rothermel eq. 47):
```
phi_wind = C x U^B x max(0, cos(theta_spread - theta_wind))
```

Slope factor (Rothermel eq. 51):
```
phi_slope = 5.275 x tan^2(slope_angle)
  applied only for uphill spread; = 0 for downhill
```

Dynamic wind oscillation:
```
theta(t) = theta_0 + sin(t / T_period) x A_amplitude
```

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `ROTHERMEL_BASE_ROS` | 0.5 | Base rate of spread (m/s) |
| `ROTHERMEL_WIND_C` | 0.4 | Wind coefficient C |
| `ROTHERMEL_WIND_B` | 1.5 | Wind exponent B |
| `ROTHERMEL_SLOPE_FACTOR` | 5.275 | Slope coefficient |
| `WIND_INITIAL_DIRECTION` | 90.0 | Starting wind direction (degrees) |
| `WIND_OSCILLATION_PERIOD` | 50.0 | Steps per sine cycle |
| `WIND_OSCILLATION_AMPLITUDE` | 20.0 | Max wind deviation (degrees) |
| `WIND_SPEED` | 5.0 | Wind speed (m/s) |
| `FIRE_SPREAD_PROB_BASE` | 0.4 | Base ignition probability per step |
| `FIRE_BURNOUT_PROB` | 0.05 | Default burnout probability per step |

### Implementation

File: `src/fire_simulation.py`

- `step()`: Vectorized cellular automaton using `scipy.signal.convolve2d`
- `_calculate_directional_spread_vectorized()`: Wind and slope factors per direction
- `_calculate_slope_factor_vectorized()`: Rothermel phi_slope for all 8 directions
- `_update_wind_vector()`: Sinusoidal wind oscillation

Neighbor preheating: each burning neighbor adds 10% to spread probability.

---

## 2. NFFL Fuel Models

### Reference
Anderson, H.E. (1982). *Aids to Determining Fuel Models for Estimating Fire Behavior*. USDA Forest Service GTR INT-122.

### Description

Thirteen Northern Forest Fire Laboratory (NFFL) fuel models classify vegetation type. Each model has a `spread_multiplier` (affects ROS) and a `burnout_prob` (affects how quickly a cell transitions from Burning to Burnt).

| Fuel Code | Type | spread_multiplier | burnout_prob |
|-----------|------|------------------|--------------|
| 1 | Short grass | 1.0 | 0.10 |
| 2 | Timber grass | 0.8 | 0.08 |
| 3 | Tall grass | 1.5 | 0.12 |
| 4 | Chaparral | 1.2 | 0.06 |
| 5 | Brush | 0.9 | 0.07 |
| 6 | Dormant brush | 1.1 | 0.08 |
| 7 | Southern rough | 0.8 | 0.06 |
| 8 | Closed timber litter | 0.5 | 0.04 |
| 9 | Hardwood litter | 0.7 | 0.05 |
| 10 | Timber (litter+understory) | 0.9 | 0.05 |
| 11 | Light slash | 1.0 | 0.08 |
| 12 | Medium slash | 1.3 | 0.07 |
| 13 | Heavy slash | 1.4 | 0.06 |

Fuel type per cell is derived from CORINE land cover data (CLC_TO_NFFL_MAP in config) or defaults to model 4.

### Implementation

File: `src/config.py` (FUEL_MODELS dict), `src/fire_simulation.py` (applied per-cell in step()), `src/agents/risk_monitor.py` (fuel_factor in risk grid)

---

## 3. Smoke Advection-Diffusion Model

### Reference
Inness, A. et al. (2019). "The CAMS reanalysis of atmospheric composition." *Atmospheric Chemistry and Physics*, 19(6), pp. 3515–3556. DOI: 10.5194/acp-19-3515-2019

### Equation

Simplified explicit finite-difference advection-diffusion:
```
dC/dt = -U . grad(C) + D . lap(C) + S

C   = smoke concentration [0–1 normalised]
U   = wind vector (from Rothermel dynamic-wind model)
D   = isotropic diffusion coefficient
S   = source term: SMOKE_SOURCE_STRENGTH per burning cell per step
```

Each step:
1. Source: `C[burning] += SMOKE_SOURCE_STRENGTH`
2. Wind advection: shift grid downwind by SMOKE_WIND_ADVECTION fraction
3. Isotropic diffusion: 4-neighbour average weighted by SMOKE_DIFFUSION_RATE
4. Atmospheric decay: `C *= (1 - SMOKE_DECAY_RATE)`

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `SMOKE_SOURCE_STRENGTH` | 0.30 | Emission per burning cell per step |
| `SMOKE_DIFFUSION_RATE` | 0.08 | Isotropic diffusion coefficient D |
| `SMOKE_DECAY_RATE` | 0.05 | Atmospheric scavenging per step |
| `SMOKE_WIND_ADVECTION` | 0.40 | Fraction advected downwind per step |

### Implementation

File: `src/fire_simulation.py`, method `_update_smoke_grid()`

Output: `environment.smoke_grid` — float32 array [0, 1] updated every step.

Downstream consumers:
- `CivilianAgent.perceive()`: reads smoke_grid for injury accumulation and panic amplification
- (Future) `SentinelAgent.perceive()`: increasing noise sigma with smoke concentration

---

## 4. Civilian Injury Model (Smoke Inhalation)

### References
- Inness, A. et al. (2019). CAMS PM2.5 smoke product (above).
- Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood evacuations in the urban-wildland interface." *Environment and Planning A*, 34(12), pp. 2211–2230.

### Equation

Cumulative PM2.5 proxy exposure:
```
smoke_exposure(t) = smoke_exposure(t-1) + smoke_grid[r, c]

When smoke_exposure >= CIVILIAN_INJURY_THRESHOLD:
    is_injured = True
    send INJURY_REPORT to ambulances
    halt movement
```

Smoke also amplifies panic:
```
panic(t) += smoke_grid[r, c] x CIVILIAN_SMOKE_PANIC_SCALE
```

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `CIVILIAN_INJURY_THRESHOLD` | 5.0 | Cumulative exposure before incapacitation |
| `CIVILIAN_SMOKE_PANIC_SCALE` | 0.02 | Extra panic per unit smoke concentration |

### Implementation

File: `src/agents/civilian.py`, `perceive()` and `act()`

On injury:
- Civilian sends `INFORM` with `type='INJURY_REPORT'` to `"ambulances"` (once only, `_injury_reported` flag)
- Movement is skipped until ambulance transports to hospital (external resolution)

---

## 5. Sentinel Detection — Signal Detection Theory

### Reference
Green, D.M. & Swets, J.A. (1966). *Signal Detection Theory and Psychophysics*. Wiley, New York.

### Equation

```
I_detected = [I_actual / (d^2 + epsilon)] x (1 + cos(theta)) + N(0, sigma)
```

Where:
- `I_actual`: Fire temperature at target cell
- `d`: Euclidean distance from sentinel to fire cell
- `epsilon`: Stability constant (prevents division by zero)
- `theta`: Angle between wind direction and sensor-to-fire vector
- `N(0, sigma)`: Gaussian sensor noise

Debouncing: alert sent to Analyst only after 3 consecutive steps above threshold.

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `SENTINEL_SIGNAL_EPSILON` | 1.0 | Stability constant |
| `SENTINEL_NOISE_SIGMA` | 5.0 | Gaussian noise std dev |
| `SENTINEL_TRIGGER_THRESHOLD` | 15.0 | Detection threshold |
| `SENTINEL_DEBOUNCE_STEPS` | 3 | Consecutive detections required |
| `SENTINEL_DETECTION_RADIUS` | 30 | Grid cells (spatial search bound) |

### Implementation

File: `src/agents/sentinel.py`, `perceive()`

---

## 6. Analyst — Rothermel TTI Calculation

### Reference
Rothermel (1972) — see Section 1 above.

### Equations

```
ROS = R_base x (1 + phi_wind) x (1 + phi_slope)
TTI = distance_to_nearest_population / ROS     [minutes]
```

Phase-aware conservatism (when Commander is in Phase 2 or 3):
```
TTI_adjusted = TTI x 0.80    (20% reduction — more urgent estimate)
```

### Implementation

File: `src/agents/analyst.py`, `decide()`

- `fire_reports` filtered by `suppressed_cells` (cells confirmed extinguished by firefighters)
- `current_phase` updated by Commander PHASE_UPDATE messages
- Fuzzy logic system produces risk level (LOW / MEDIUM / HIGH / CRITICAL) from TTI and exit count

---

## 7. Commander — ECT vs TTI Decision Logic

### References
- Wolshon, B. (2006). "Evacuation planning and engineering for Hurricane Katrina." *The Bridge*, 36(1), pp. 27–34.
- Lagouvardos, K. et al. (2019). "Meteorological analysis of the catastrophic wildfire in Mati, eastern Attica, Greece." *BAMS*, 100(11), pp. 2243–2257.
- Cova, T.J. & Johnson, J.P. (2002) — Phase 3 Shelter-in-Place logic.

### Equations

Evacuation Clearance Time:
```
ECT = (N_civilians / C_total) x gamma_congestion
C_total = COMMANDER_EXIT_CAPACITY x num_exits
gamma_congestion = 1.0 + 0.1 x active_missions
```

Phase selection:
```
IF TTI > 2.5 x ECT:  Phase 0 (Monitor)
IF TTI > 1.5 x ECT:  Phase 1 (Pre-Alert) + dispatch firefighters
IF TTI > 1.0 x ECT:  Phase 2 (Evacuate) + dispatch all field units
ELSE:                 Phase 3 (Shelter-in-Place)
```

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `COMMANDER_EXIT_CAPACITY` | 10 | Agents per minute per exit |
| `COMMANDER_CONGESTION_FACTOR_BASE` | 1.0 | Base congestion multiplier |
| `COMMANDER_PHASE_0_MULTIPLIER` | 2.5 | TTI/ECT ratio for Phase 0 |
| `COMMANDER_PHASE_1_MULTIPLIER` | 1.5 | TTI/ECT ratio for Phase 1 |
| `COMMANDER_PHASE_2_MULTIPLIER` | 1.0 | TTI/ECT ratio for Phase 2 |

### Implementation

File: `src/agents/commander.py`

On phase transition: sends `PHASE_UPDATE` (INFORM) to Analyst so it adjusts TTI conservatism.

---

## 8. Pre-Ignition Risk — Canadian Fire Weather Index (FWI)

### References
- Van Wagner, C.E. (1987). *Development and Structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35. Canadian Forestry Service.
- Van Wagner, C.E. & Pickett, T.L. (1985). *Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System*. Forestry Technical Report 33.

### Equation

Per-cell ignition risk grid:
```
risk = 0.40 x fwi_factor
     + 0.30 x fuel_factor        (NFFL spread_multiplier normalised to [0,1])
     + 0.20 x firms_factor       (VIIRS historical ignition density)
     + 0.10 x slope_factor       (terrain gradient magnitude)
```

Components normalised to [0, 1]. No-fuel cells set to 0. Burning cells set to 1.

FWI factor:
```
fwi_factor = min(fwi_score / 60.0, 1.0)
```

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `RISK_MONITOR_UPDATE_INTERVAL` | 20 | Steps between recomputations |
| `FWI_HIGH_RISK_THRESHOLD` | 30.0 | FWI >= 30: high pre-fire warning |
| `FWI_EXTREME_RISK_THRESHOLD` | 50.0 | FWI >= 50: extreme warning |

### Implementation

File: `src/agents/risk_monitor.py`, `_recompute_risk_grid()`

Output: `environment.ignition_risk_grid` — shared with Commander for asset pre-positioning.

---

## 9. NASA FIRMS — Historical Ignition Density

### Reference
Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I.A. (2014). "The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment." *Remote Sensing of Environment*, 143, pp. 85–96. DOI: 10.1016/j.rse.2013.12.008

### Usage

The FIRMSConnector queries the FIRMS VIIRS API for the last 7 days of active fire detections within a radius around the simulation centre. Detections are gridded into `environment.firms_density` [0, 1] and used as the 20% historical factor in the RiskMonitor risk grid.

### Implementation

File: `src/data_connectors/firms_connector.py`

Config key: `FIRMS_MAP_KEY` — free API key from https://firms.modaps.eosdis.nasa.gov/api/

---

## 10. Civilian Traffic — Greenshields Model

### Reference
Greenshields, B.D., Bibbins, J.R., Channing, W.S. & Miller, H.H. (1935). "A study of traffic capacity." *Highway Research Board Proceedings*, 14, pp. 448–477.

### Equation

```
V_current = V_free x (1 - rho_local / rho_jam)
```

Where:
- `V_free`: Free-flow speed (empty road)
- `rho_local`: Local density — agents on current edge
- `rho_jam`: Jam density — gridlock threshold

When `rho_local >= rho_jam`, speed = 0 (gridlock).

Cognitive state modifier:
```
IF confused: V_current x= 0.5
IF herding:  V_current x= CIVILIAN_HERDING_INFLUENCE (speed from crowd vector)
```

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `CIVILIAN_V_FREE_FLOW` | 2.0 | Free-flow speed (grid cells/step) |
| `CIVILIAN_RHO_JAM` | 5.0 | Jam density (agents per edge) |
| `CIVILIAN_CONFUSED_SPEED_FACTOR` | 0.5 | Speed factor in confused state |
| `CIVILIAN_HERDING_INFLUENCE` | 0.7 | Crowd direction weight (0=ignore, 1=full) |
| `AQI_SPEED_PENALTY` | 0.3 | Max speed penalty at AQI=500 |

### Implementation

File: `src/agents/civilian.py`, `_calculate_speed_greenshields()`

---

## 11. Civilian Panic — Psychological Model

### Reference
Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood evacuations in the urban-wildland interface." *Environment and Planning A*, 34(12), pp. 2211–2230.

### Equation

```
Panic(t) = Panic(t-1)
         + alpha x (1 / d_fire)          [fire proximity]
         + beta x family_separated        [social stress]
         + AQI_PANIC_WEIGHT x (AQI/500)  [smoke/AQI effect]
         + smoke_conc x SMOKE_PANIC_SCALE [local smoke]
         - PANIC_DECAY                    [recovery]
```

3-State Cognitive Machine:

| State | Panic Range | Movement Strategy |
|-------|-------------|-------------------|
| Rational | < 0.4 | Optimal A* to safety_node |
| Confused | 0.4 – 0.7 | A* at 50% speed, stochastic re-route |
| Herding | >= 0.7 | Social Force crowd-following |

Herding fallback: if crowd direction leads into burning cell, falls back to `_move_to_safety()` (A* re-route) rather than random movement.

### Parameters

| Config Key | Value | Description |
|------------|-------|-------------|
| `CIVILIAN_PANIC_ALPHA` | 0.05 | Fire distance coefficient |
| `CIVILIAN_PANIC_BETA` | 0.20 | Family separation penalty |
| `CIVILIAN_PANIC_DECAY` | 0.01 | Decay rate when no fire visible |
| `CIVILIAN_PANIC_RATIONAL` | 0.40 | Rational state threshold |
| `CIVILIAN_PANIC_CONFUSED` | 0.70 | Confused state threshold |
| `AQI_PANIC_WEIGHT` | 0.10 | AQI contribution to panic |

### Implementation

File: `src/agents/civilian.py`, `_update_panic()` and `_update_cognitive_state()`

---

## 12. Contract Net Protocol (CNP) — Task Allocation

### Reference
Smith, R.G. (1980). "The Contract Net Protocol: High-level communication and control in a distributed problem solver." *IEEE Transactions on Computers*, C-29(12), pp. 1104–1113. DOI: 10.1109/TC.1980.1675516

### Protocol

```
Commander (Manager)   -->  CFP (Call For Proposal)
Contractor (Rescuer / Firefighter / Ambulance)  -->  PROPOSE {cost, eta, ...}
                                                 -->  REFUSE {reason}
Commander             -->  ACCEPT_PROPOSAL (best bid)
Commander             -->  REJECT_PROPOSAL (others)
Contractor            -->  CONFIRM {mission_id, status:COMPLETED}
```

Bid cost formulas:

Rescuer:
```
cost = path_length / speed + path_risk x RESCUER_RISK_ALPHA + (100 - fuel)
```

Firefighter:
```
cost = dist_to_target + (capacity - water) x 0.01
```

Ambulance:
```
cost = path_length_to_scene + path_risk x 50
```

Safety protocol: all contractors refuse missions through active fire.

### Implementation

Files: `src/agents/commander.py`, `rescuer.py`, `firefighter.py`, `ambulance.py`

Ambulance direct dispatch: when a civilian sends INJURY_REPORT, idle ambulances self-assign without waiting for a Commander CFP.

---

## 13. Firefighter Fire-Line Placement — Wind-Perpendicular Strategy

### Reference
Rothermel (1972) — Section 1 above (wind-aligned spread direction).

### Logic

An effective fire line must be placed perpendicular to the wind vector, ahead of the fire front, so the advancing fire runs into a fuel gap.

```
wind_vec = environment.fire_simulation.wind_direction   # unit vector [dx, dy]
perp_vec = [-wind_dy, wind_dx]                          # 90-degree rotation

anchor = target_cell + 2 x wind_vec                     # 2 cells downwind
line   = {anchor + perp_vec x k : k in [-width, +width]}
```

Cells in `line` with `fire_grid == 3` (Fuel) are set to 0 (No Fuel).

Fallback: if `fire_simulation` is unavailable, falls back to an axis-aligned line.

### Implementation

File: `src/agents/firefighter.py`, `_execute_fire_line()`

---

## 14. Air Quality — CAMS/OpenAQ Integration

### Reference
Inness, A. et al. (2019) — Section 3 above.

### Usage

AQI is fetched from OpenAQ at simulation startup:
- AQI > 50: adds panic `AQI_PANIC_WEIGHT x (AQI/500)` per step
- AQI > 50: reduces movement speed by `AQI_SPEED_PENALTY x (AQI/500)`

This is separate from the per-cell smoke_grid model (Section 3), which represents local smoke from the active fire rather than regional air quality.

### Implementation

File: `src/data_connectors/airquality_connector.py`; applied in `src/agents/civilian.py`, `_update_panic()` and `_calculate_speed_greenshields()`

---

## Configuration Reference

All physics parameters are in `src/config.py`. Full table:

### Fire Physics
| Key | Default | Source |
|-----|---------|--------|
| FIRE_SPREAD_PROB_BASE | 0.4 | Calibrated |
| FIRE_BURNOUT_PROB | 0.05 | Calibrated |
| ROTHERMEL_BASE_ROS | 0.5 | Rothermel 1972 |
| ROTHERMEL_WIND_C | 0.4 | Rothermel 1972 |
| ROTHERMEL_WIND_B | 1.5 | Rothermel 1972 |
| ROTHERMEL_SLOPE_FACTOR | 5.275 | Rothermel 1972 |
| WIND_SPEED | 5.0 | Calibrated |
| WIND_INITIAL_DIRECTION | 90.0 | Calibrated |
| WIND_OSCILLATION_PERIOD | 50.0 | Calibrated |
| WIND_OSCILLATION_AMPLITUDE | 20.0 | Calibrated |
| FIRE_TEMP_BURNING | 100.0 | Calibrated |
| FIRE_TEMP_COOLING_RATE | 5.0 | Calibrated |
| FIRE_TEMP_AMBIENT | 20.0 | Calibrated |

### Smoke Model
| Key | Default | Source |
|-----|---------|--------|
| SMOKE_SOURCE_STRENGTH | 0.30 | Inness 2019 |
| SMOKE_DIFFUSION_RATE | 0.08 | Inness 2019 |
| SMOKE_DECAY_RATE | 0.05 | Inness 2019 |
| SMOKE_WIND_ADVECTION | 0.40 | Inness 2019 |

### Civilian Injury
| Key | Default | Source |
|-----|---------|--------|
| CIVILIAN_INJURY_THRESHOLD | 5.0 | Inness 2019 |
| CIVILIAN_SMOKE_PANIC_SCALE | 0.02 | Calibrated |

### FWI / Risk Monitor
| Key | Default | Source |
|-----|---------|--------|
| RISK_MONITOR_UPDATE_INTERVAL | 20 | Calibrated |
| FWI_HIGH_RISK_THRESHOLD | 30.0 | Van Wagner 1987 |
| FWI_EXTREME_RISK_THRESHOLD | 50.0 | Van Wagner 1987 |

---

## Validation Scenarios

### Scenario 1: Early Warning (Successful Evacuation)
```python
ROTHERMEL_BASE_ROS = 0.3   # slow fire
COMMANDER_EXIT_CAPACITY = 15
NUM_CIVILIANS = 15
```
Expected: Phase 0 -> 1 -> 2 progression; most civilians evacuate before Phase 3.

### Scenario 2: Mati Tragedy (Shelter-in-Place)
```python
ROTHERMEL_BASE_ROS = 1.5   # fast fire
COMMANDER_EXIT_CAPACITY = 5  # bottleneck
NUM_CIVILIANS = 30
```
Expected: Phase 0 -> 3 jump (too late to evacuate); civilians redirect to nearest safe zone; some casualties.

### Scenario 3: Gridlock
```python
CIVILIAN_RHO_JAM = 2.0   # low jam density
NUM_CIVILIANS = 50
```
Expected: Greenshields model drives V -> 0; civilians unable to move; evacuation stalls.

### Scenario 4: Smoke Incapacitation
```python
SMOKE_SOURCE_STRENGTH = 0.50
CIVILIAN_INJURY_THRESHOLD = 3.0
NUM_CIVILIANS = 20
```
Expected: Several civilians injured by smoke before fire reaches them; ambulances dispatch to INJURY_REPORT messages.
