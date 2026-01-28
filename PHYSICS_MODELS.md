# AIGIS Physics-Based Models Documentation

This document details the mathematical and psychological models implemented in AIGIS for realistic disaster simulation.

## Overview

AIGIS has been enhanced with physics-based fire spread models, traffic dynamics, and cognitive psychology models to create a highly realistic wildfire evacuation simulation. All agent behaviors are now governed by scientific equations and empirical models from fire science, traffic engineering, and crowd psychology.

---

## 1. Sentinel Agent - Signal Detection Theory

### Model: Environmental Signal Attenuation

Sentinels no longer have perfect vision. Instead, they detect an attenuated signal based on:
- **Distance** (inverse square law)
- **Wind direction** (smoke drift)
- **Sensor noise** (Gaussian)

### Signal Equation

```
I_detected = [I_actual / (d² + ε)] × (1 + cos(θ)) + N(0, σ)
```

Where:
- `I_actual`: Actual fire intensity (temperature) at target cell
- `d`: Euclidean distance to fire
- `ε`: Small constant to prevent division by zero (config: `SENTINEL_SIGNAL_EPSILON`)
- `θ`: Angle between wind direction and sensor-to-fire vector
- `N(0, σ)`: Gaussian noise (config: `SENTINEL_NOISE_SIGMA`)

### Debouncing Protocol

To prevent false alerts from noise spikes:
- **Rule**: Alert is triggered **only if** signal exceeds threshold for **3 consecutive steps**
- Implements: `detection_history` dictionary tracking consecutive detections per location
- Config: `SENTINEL_DEBOUNCE_STEPS = 3`

### Implementation

File: `src/agents/sentinel.py`

Key methods:
- `perceive()`: Implements signal equation with distance and wind attenuation
- Detection history tracking for debouncing

---

## 2. Analyst Agent - Rothermel Fire Spread Model

### Model: Physics-Based Fire Propagation

The Analyst uses **Rothermel's Surface Fire Spread Model** to calculate how fast fire moves.

### Rate of Spread (ROS) Equation

```
ROS = R_base × (1 + φ_wind) × (1 + φ_slope)
```

**Slope Factor** (fire accelerates exponentially uphill):
```
φ_slope = 5.275 × tan²(φ)
```

**Wind Factor**:
```
φ_wind = C × U^B
```

Where:
- `R_base`: Base rate of spread (config: `ROTHERMEL_BASE_ROS = 0.5 m/s`)
- `C`: Wind coefficient (config: `ROTHERMEL_WIND_C = 0.4`)
- `B`: Wind exponent (config: `ROTHERMEL_WIND_B = 1.5`)
- `U`: Wind speed
- `φ`: Slope angle

### Time To Impact (TTI)

```
TTI = Distance_to_Settlement / ROS
```

Tells the commander **how long until fire reaches civilians**.

### Fuzzy Logic Risk Assessment

**New Input Variables:**
1. **TTI** (Time To Impact)
   - Imminent: < 30m
   - Near Future: 30-90m
   - Distant: > 90m

2. **Escape Route Availability**
   - Bottlenecked: < 2 exits
   - Sufficient: ≥ 2 exits

**Fuzzy Rules (Mati Fire Scenario):**
```
IF TTI is Imminent AND Routes are Bottlenecked THEN Risk is CRITICAL
IF TTI is Imminent AND Routes are Sufficient THEN Risk is HIGH
IF TTI is Near Future AND Routes are Bottlenecked THEN Risk is HIGH
IF TTI is Near Future AND Routes are Sufficient THEN Risk is MEDIUM
IF TTI is Distant THEN Risk is LOW
```

### Implementation

File: `src/agents/analyst.py`

Key methods:
- `_calculate_ros()`: Rothermel equation implementation
- `_calculate_tti()`: Distance-to-time conversion
- `_setup_fuzzy_system()`: New fuzzy rules with TTI and routes

---

## 3. Commander Agent - ECT vs TTI Decision Logic

### Model: Evacuation Clearance Time Comparison

The Commander constantly compares:
- **TTI**: Time until fire arrives (from Analyst)
- **ECT**: Time needed to evacuate everyone

### Evacuation Clearance Time (ECT)

```
ECT = (N_agents / C_exit) × γ
```

Where:
- `N_agents`: Number of civilians in danger zone
- `C_exit`: Aggregate exit capacity (agents/minute per exit)
- `γ`: Congestion factor (increases with active missions)

Config:
- `COMMANDER_EXIT_CAPACITY = 10` agents/min
- `COMMANDER_CONGESTION_FACTOR_BASE = 1.0`

### 4-Phase Decision Protocol

The Commander operates in distinct phases based on `TTI / ECT` ratio:

| Phase | Condition | Action |
|-------|-----------|--------|
| **0: Monitoring** | TTI > 2.5 × ECT | Standby, monitor situation |
| **1: Pre-Evacuation** | 1.5 × ECT < TTI ≤ 2.5 × ECT | Send WARNING, prepare civilians |
| **2: Mass Evacuation** | 1.0 × ECT < TTI ≤ 1.5 × ECT | EVACUATE order, dispatch rescuers |
| **3: Shelter-in-Place** | TTI ≤ ECT | **CRITICAL**: Redirect to coast/water |

**Phase 3 Logic (Mati Scenario):**
When `TTI ≤ ECT`, it's **too late to evacuate** by road. Roads will jam. The Commander broadcasts:
```
REDIRECT_TO_COAST - PROCEED TO NEAREST COAST/WATER
```

Civilians abandon highway exits and head for the sea (as occurred in Mati, Greece 2018).

### Implementation

File: `src/agents/commander.py`

Key methods:
- `_calculate_ect()`: ECT formula
- `_determine_phase()`: Phase logic based on TTI/ECT
- `_order_shelter_in_place()`: Emergency coast redirect
- `act()`: Phase-specific actions

---

## 4. Rescuer Agent - Risk-Adjusted Bidding

### Model: Path Risk Assessment

Rescuers no longer bid based only on distance. They **scan the temperature grid** along their planned path.

### Path Risk Calculation

```
Risk_path = max(Temperature[node]) for all nodes in path
```

### Bid Calculation

```
Cost = (Length / V_avg) + (Risk_path × α) + (100 - FuelLevel)
```

Where:
- `α`: Risk penalty weight (config: `RESCUER_RISK_ALPHA = 50.0`)
- Higher path risk → higher bid cost → less likely to win contract

### Safety Protocol

**CRITICAL RULE:**
```
IF Risk_path > Safety_Threshold THEN send(REFUSE)
```

Rescuers **refuse missions through active fire**, regardless of urgency.

Config: `RESCUER_SAFETY_THRESHOLD = 70.0°`

### Implementation

File: `src/agents/rescuer.py`

Key methods:
- `_assess_path_risk()`: Scans temperature grid along path
- `_handle_cfp()`: Enhanced with risk assessment and safety check

---

## 5. Civilian Agent - Traffic Physics + Cognitive Psychology

### Model 1: Greenshields' Traffic Model

Civilians don't move at constant speed. Speed depends on **local agent density**.

### Speed Equation

```
V_current = V_free_flow × (1 - ρ_local / ρ_jam)
```

Where:
- `V_free_flow`: Maximum speed in free-flowing traffic (config: `CIVILIAN_V_FREE_FLOW = 2.0`)
- `ρ_local`: Number of agents on current edge
- `ρ_jam`: Jam density (config: `CIVILIAN_RHO_JAM = 5.0`)

**Consequence**: When `ρ_local ≥ ρ_jam` → **GRIDLOCK** → `V = 0` (cannot move)

### Model 2: 3-State Cognitive Machine

Panic level determines cognitive state and behavior:

| State | Panic Range | Behavior |
|-------|-------------|----------|
| **1: Rational** | < 0.4 | Optimal A* pathfinding to exits |
| **2: Confused** | 0.4 - 0.7 | Speed × 50%, frequent re-routing |
| **3: Herding** | > 0.7 | Follow crowd, even to dead ends |

Config:
- `CIVILIAN_PANIC_RATIONAL = 0.4`
- `CIVILIAN_PANIC_CONFUSED = 0.7`
- `CIVILIAN_CONFUSED_SPEED_FACTOR = 0.5`

### Panic Equation

```
Panic(t) = Panic(t-1) + α × (1/d_fire) + β × (Family_Separated?) - decay
```

Where:
- `α`: Fire distance coefficient (config: `CIVILIAN_PANIC_ALPHA = 0.05`)
- `d_fire`: Distance to nearest visible fire
- `β`: Family separation penalty (config: `CIVILIAN_PANIC_BETA = 0.2`)
- `decay`: Slow decay when no fire visible (config: `CIVILIAN_PANIC_DECAY = 0.01`)

### Herding Behavior (State 3)

At high panic (`> 0.7`), civilians:
1. **Ignore optimal paths**
2. **Follow nearby agents** moving away from fire
3. May follow crowd to **dead ends** (realistic tragedy scenario)

### Implementation

File: `src/agents/civilian.py`

Key methods:
- `_calculate_speed_greenshields()`: Traffic model
- `_update_panic()`: Panic equation with fire distance
- `_update_cognitive_state()`: State machine
- `_follow_crowd()`: Herding behavior
- `act()`: Uses traffic-based speed for all movement

---

## Configuration Parameters

All physics models are configurable in `src/config.py`:

### Sentinel
```python
SENTINEL_SIGNAL_EPSILON = 1.0
SENTINEL_NOISE_SIGMA = 5.0
SENTINEL_TRIGGER_THRESHOLD = 15.0
SENTINEL_DEBOUNCE_STEPS = 3
```

### Analyst (Rothermel)
```python
ROTHERMEL_BASE_ROS = 0.5  # m/s
ROTHERMEL_WIND_C = 0.4
ROTHERMEL_WIND_B = 1.5
ANALYST_TTI_IMMINENT = 30  # meters
ANALYST_TTI_NEAR = 90
ANALYST_EXIT_BOTTLENECK_THRESHOLD = 2
```

### Commander (ECT)
```python
COMMANDER_CONGESTION_FACTOR_BASE = 1.0
COMMANDER_EXIT_CAPACITY = 10  # agents/min
COMMANDER_PHASE_0_MULTIPLIER = 2.5
COMMANDER_PHASE_1_MULTIPLIER = 1.5
COMMANDER_PHASE_2_MULTIPLIER = 1.0
```

### Rescuer
```python
RESCUER_RISK_ALPHA = 50.0
RESCUER_SAFETY_THRESHOLD = 70.0  # temperature
```

### Civilian
```python
CIVILIAN_V_FREE_FLOW = 2.0
CIVILIAN_RHO_JAM = 5.0
CIVILIAN_PANIC_RATIONAL = 0.4
CIVILIAN_PANIC_CONFUSED = 0.7
CIVILIAN_PANIC_HERDING = 0.7
CIVILIAN_PANIC_ALPHA = 0.05
CIVILIAN_PANIC_BETA = 0.2
CIVILIAN_PANIC_DECAY = 0.01
CIVILIAN_CONFUSED_SPEED_FACTOR = 0.5
CIVILIAN_VISION_RADIUS = 10
```

---

## Environment Enhancements

### Temperature Grid

Added: `environment.temperature_grid`
- Tracks temperature (0-100°) at each grid cell
- Burning cells: 100°
- Burnt cells: Decay by 5° per step
- Used by Rescuers for path risk assessment

### Exit Node Identification

Added: `environment.exit_nodes`
- Easternmost 10% of road network nodes
- Represents exits towards coast/safety
- Used by Commander for ECT calculation

---

## Scientific References

1. **Rothermel, R.C. (1972)** - "A Mathematical Model for Predicting Fire Spread in Wildland Fuels"
2. **Greenshields, B.D. (1935)** - "A Study of Traffic Capacity" (Traffic flow theory)
3. **Helbing, D. et al. (2000)** - "Simulating dynamical features of escape panic" (Herding behavior)
4. **Kontou, A. et al. (2019)** - Analysis of Mati Fire evacuation failure

---

## Testing Recommendations

### Scenario 1: Early Warning
- Set `ROTHERMEL_BASE_ROS = 0.3` (slow fire)
- Observe: Phase 0 → 1 → 2 transition
- Expected: Successful evacuation via exits

### Scenario 2: Mati Tragedy
- Set `ROTHERMEL_BASE_ROS = 1.5` (fast fire)
- Set `COMMANDER_EXIT_CAPACITY = 5` (bottleneck)
- Observe: Direct Phase 0 → 3 jump
- Expected: REDIRECT_TO_COAST, herding behavior

### Scenario 3: Gridlock
- Set `CIVILIAN_RHO_JAM = 2.0` (low jam density)
- Set `NUM_CIVILIANS = 50`
- Observe: Traffic gridlock, V = 0
- Expected: Civilians unable to evacuate

---

## Validation Metrics

The simulation now provides realistic validation:

1. **TTI vs ECT ratio** - Correlates with evacuation success
2. **Panic distribution** - Should show spatial clustering near fire
3. **Cognitive state distribution** - Rational → Confused → Herding progression
4. **Traffic flow** - Speed decreases with density (Greenshields curve)
5. **Rescuer refusals** - Should increase as fire spreads

---

## Future Enhancements

Possible extensions to the models:

1. **Dynamic wind** - Wind direction/speed changes during simulation
2. **Agent casualties** - Remove agents caught by fire
3. **Social networks** - Model family reunification behavior
4. **Multi-fire ignition** - Multiple simultaneous fire fronts
5. **Real elevation data** - Use DEM instead of generated gradients
6. **Lane-level traffic** - Track density per road lane
7. **Building structures** - Shelter-in-place in fireproof buildings

---

## Code Structure

```
src/
├── config.py              # All physics parameters
├── environment.py         # Temperature grid, exit nodes
├── fire_simulation.py     # Temperature updates
├── agents/
│   ├── sentinel.py       # Signal Detection Theory
│   ├── analyst.py        # Rothermel + Fuzzy Logic
│   ├── commander.py      # ECT vs TTI logic
│   ├── rescuer.py        # Path risk assessment
│   └── civilian.py       # Traffic model + Cognitive states
```

All models are fully implemented and tested. The simulation is now scientifically grounded and ready for research use.
