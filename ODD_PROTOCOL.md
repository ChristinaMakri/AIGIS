# ODD Protocol for AIGIS
## An Agent-Based Wildfire Evacuation Simulation

**Version:** 1.0
**Date:** 2026-03-13
**Authors:** AIGIS Development Team

---

> **Reference standard:**
> Grimm, V., Railsback, S.F., Vincenot, C.E., Berger, U., Gallagher, C., DeAngelis, D.L., Edmonds, B., Ge, J., Giske, J., Gotts, N., Guo, Q., Huth, A., Jepsen, J.U., Kawul, C., Kleinhans, M.G., Langangen, O., Latombe, G., Le Page, C., Li, F., Litchman, E., Matsinos, Y.G., Müller, B., Murray-Rust, D., Nikolskiy, P., Noe, D.A., Piou, C., Radchuk, V., Robbins, A.M., Robbins, M.M., Rossmanith, E., Ruger, N., Strand, E., Souissi, S., Stillman, R.A., Vabo, R., Visser, U., Wiegand, T., Ayllón, D., Zabala, A. (2020).
> "The ODD Protocol for Describing Agent-Based and Other Simulation Models: A Second Update to Improve Clarity, Replication, and Structural Realism."
> *JASSS*, 23(2):7. DOI: [10.18564/jasss.4259](https://doi.org/10.18564/jasss.4259)

---

## 1. Purpose and Patterns

### 1.1 Purpose

AIGIS (Artificial Intelligence Geographic Information System) is an agent-based simulation designed to study wildfire propagation and emergency evacuation in complex urban-wildland interface environments. The primary purpose is to:

1. Model the spatiotemporal dynamics of fire spread under realistic meteorological and topographic conditions.
2. Simulate the behavioural responses of civilians, emergency responders, and command-level agents to an evolving fire threat.
3. Identify bottlenecks in evacuation networks (gridlock, road capacity, building obstruction).
4. Evaluate the effectiveness of coordination protocols (Contract Net Protocol) and cognitive-behavioural models (three-state panic machine) on casualty outcomes.
5. Serve as a decision-support and training tool for emergency management planners.

### 1.2 Patterns

The following emergent patterns are used to evaluate whether the model is behaving as intended:

| Pattern | Expected observation | Source |
|---------|---------------------|--------|
| Fire spread anisotropy | Fire propagates faster in the downwind direction; elliptical burn perimeter | Rothermel (1972) |
| Panic contagion | Panic levels rise rapidly in civilians near the fire front, then spread socially | Cova & Johnson (2002) |
| Gridlock formation | High civilian density on narrow roads produces zero-velocity queues | Greenshields (1935) |
| CNP allocation efficiency | CNP-dispatched rescuers reach injured civilians faster than randomly assigned | Smith (1980) |
| Mortality-density correlation | Higher civilian density in the fire zone correlates with higher mortality | Mas et al. (2021) |
| Smoke AQI gradient | Air quality index degrades near active fire cells, recovers as wind disperses smoke | EPA AQI standard |

---

## 2. Entities, State Variables, and Scales

### 2.1 Entities

AIGIS contains the following agent types:

| Agent | Count (default) | Role |
|-------|----------------|------|
| `SentinelAgent` | 4 | Reactive fire detection via Signal Detection Theory; debounced INFORM to Analyst |
| `AnalystAgent` | 1 | BDI — computes Rothermel TTI/ROS from sentinel reports; filters suppressed cells |
| `CommanderAgent` | 1 | BDI + PPO — ECT/TTI 4-phase logic; CNP Manager for all field units |
| `RiskMonitorAgent` | 1 | Model-Based BDI — pre-ignition risk grid (FWI + FIRMS + fuel + slope) |
| `FirefighterAgent` | 2 | BDI + Utility + PPO — CNP Contractor; water drop / fire-line / backburn |
| `RescuerAgent` | 3 | BDI + PPO — CNP Contractor; A* path to highest-priority civilian |
| `AmbulanceAgent` | 2 | BDI — two-leg CNP Contractor; direct self-dispatch on civilian INJURY_REPORT |
| `CivilianAgent` | 60 | BDI — three-state cognitive machine; smoke injury accumulation; Greenshields traffic |

In addition, the **Environment** object contains a 200×200 grid representing the simulation area, overlaid with:
- `fire_grid` — cell-level fire state (no fire / burning / burned)
- `smoke_grid` — AQI value per cell
- `fuel_grid` — fuel load (kg/m²) per cell, from OSM land cover
- `elevation_grid` — SRTM DEM data (m)
- Road network graph (OpenStreetMap via OSMnx)

### 2.2 State Variables

#### CivilianAgent
| Variable | Type | Description |
|----------|------|-------------|
| `lat`, `lon` | float | Geographic position |
| `grid_position` | tuple(int,int) | Discrete grid cell |
| `current_speed` | float | Current travel speed (m/s) |
| `panic_level` | float ∈ [0,1] | Three-state panic value |
| `is_injured` | bool | True if trapped / smoke-incapacitated |
| `is_active` | bool | False when dead or safely evacuated |
| `_gridlock_steps` | int | Consecutive steps at speed ≈ 0 |
| `_no_progress_steps` | int | Consecutive zero-displacement perimeter moves |
| `evacuation_route` | list | A*-computed path to safe zone |
| `home_node` | int | OSM node ID of starting position |

#### FirefighterAgent
| Variable | Type | Description |
|----------|------|-------------|
| `water_level` | float | Remaining suppressant (0–100%) |
| `position` | tuple | Current grid cell |
| `target_cell` | tuple | Current suppression target |
| `status` | str | 'idle' / 'en_route' / 'suppressing' / 'refilling' |

#### CommanderAgent
| Variable | Type | Description |
|----------|------|-------------|
| `active_missions` | dict | cfp_id → assigned agent |
| `pending_proposals` | dict | cfp_id → list of proposals |
| `current_phase` | str | 'monitoring' / 'active_response' / 'mass_evacuation' / 'recovery' |
| `cnp_refusals` | int | Count of unaccepted CNP bids |

#### Environment
| Variable | Type | Description |
|----------|------|-------------|
| `fire_grid` | np.ndarray (200×200) | Fire state per cell |
| `smoke_grid` | np.ndarray (200×200) | AQI per cell |
| `fuel_grid` | np.ndarray (200×200) | Fuel load (kg/m²) |
| `wind_direction` | float | Current wind direction (degrees from N) |
| `wind_speed` | float | Current wind speed (m/s) |
| `temperature` | float | Ambient temperature (°C) |
| `relative_humidity` | float | % relative humidity |

### 2.3 Scales

| Scale | Value |
|-------|-------|
| Spatial extent | Configurable; default 3 km radius |
| Grid resolution | 200×200 cells; default cell size ≈ 15 m² (at 3 km radius) |
| Time step | 5 seconds simulated time |
| Maximum steps | 500 steps (≈ 42 minutes simulated time) |
| Coordinate system | WGS84 geographic (lat/lon) with local Cartesian projection for grid |

---

## 3. Process Overview and Scheduling

Each simulation step executes the following processes in order:

1. **Weather update** — wind direction, speed, temperature, and relative humidity are updated using sinusoidal oscillation plus Gaussian perturbation (`environment.update_weather()`).

2. **Fire spread** — `FireSimulation.step()` applies the Rothermel (1972) rate-of-spread model to each burning cell's Moore neighbourhood (8 neighbours). Spread probability is modulated by wind alignment, slope, and fuel load.

3. **Smoke diffusion** — AQI is computed per burning cell; Gaussian plume dispersion propagates downwind.

4. **Sentinel perception** — `SentinelAgent.perceive()` samples fire perimeter cells, computes Fire Weather Index, and broadcasts alerts.

5. **Analyst prediction** — `AnalystAgent.perceive()` collects recent fire observations; `decide()` invokes ML models (XGBoost) to generate risk predictions and broadcasts a risk map.

6. **Commander decision** — `CommanderAgent.decide()` evaluates incoming CNP proposals, awards missions, updates operational phase, and issues evacuation orders.

7. **Firefighter action** — Each `FirefighterAgent.act()` moves toward its target cell and applies suppression or refills at the depot.

8. **Rescuer action** — Each `RescuerAgent.act()` navigates to injured civilians and carries them to the medical zone.

9. **Ambulance action** — `AmbulanceAgent.act()` transports critically injured civilians to hospital.

10. **Civilian action** — Each `CivilianAgent.act()` updates panic state, selects intention (shelter / evacuate / help), and moves.

11. **Risk monitor** — `RiskMonitorAgent.act()` records AQI exposures and marks smoke-injured civilians.

12. **Dashboard broadcast** — Simulation state is serialised to the web dashboard via SSE.

### Scheduling note
All agents within a type are updated sequentially in a fixed order (list index). No concurrent agent interactions occur within a single step. The fire spread process is asynchronous relative to agents — fire updates first, then agents perceive the updated state.

---

## 4. Design Concepts

### 4.1 Basic Principles

AIGIS is grounded in three theoretical frameworks:

**Belief-Desire-Intention (BDI) architecture:**
> Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to practice." *Proceedings of ICMAS-95*, pp. 312–319.

Each agent maintains: *Beliefs* (perceived environment state), *Desires* (goal set: evacuate / suppress fire / rescue civilian), and *Intentions* (committed plan). `perceive()` updates beliefs; `decide()` selects intention; `act()` executes the action.

**Contract Net Protocol (CNP) for task allocation:**
> Smith, R.G. (1980). "The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver." *IEEE Transactions on Computers*, C-29(12), pp. 1104–1113. DOI: 10.1109/TC.1980.1675516

The `CommanderAgent` broadcasts Call-for-Proposals (CFPs) for fire suppression and rescue missions. `FirefighterAgent` and `RescuerAgent` submit cost-based bids (Euclidean distance + remaining resource level). The commander awards to the lowest-cost bidder.

**Three-state civilian cognitive model:**
> Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood-level evacuation in the urban-wildland interface." *Environment and Planning A*, 34(12), pp. 2211–2229. DOI: 10.1068/a34251

Civilians transition between three states based on panic level:
- **Rational** (`panic < 0.4`): follows A*-computed optimal route
- **Confused** (`0.4 ≤ panic < 0.7`): route compliance degrades; random detours possible
- **Herding** (`panic ≥ 0.7`): follows nearest civilian (social contagion)

### 4.2 Emergence

The following outcomes emerge from local agent interactions without explicit programming:

- **Evacuation wave formation:** As the commander issues evacuation orders, civilians near the fire front begin moving first, creating a propagating wave of traffic demand on the road network.
- **Gridlock cascade:** High demand on a road segment reduces speed for all agents using that segment (Greenshields model), which may trigger panic escalation in stationary civilians, further reducing throughput.
- **Trapped-pocket casualties:** Civilians in building-enclosed areas (present in OSM data for dense urban fabric) cannot find any passable grid cell and are marked as trapped casualties — a physically realistic outcome not explicitly scripted.

### 4.3 Adaptation

Agents adapt their behaviour based on perceived environment state:

- **Civilians** switch intention from 'shelter' to 'evacuate' when perceived fire distance falls below a threshold or when a commander evacuation order is received.
- **Firefighters** switch from suppression to refill when `water_level < 20%`, then return to the nearest unassigned fire cell.
- **Commander** escalates operational phase (Phase 0: Monitor → Phase 1: Pre-Alert → Phase 2: Mass Evacuation → Phase 3: Shelter-in-Place) based on ECT vs TTI ratio (Wolshon 2006; Cova & Johnson 2002), or via the trained PPO policy.

### 4.4 Objectives

| Agent | Objective function |
|-------|--------------------|
| Civilian | Minimise travel time to safe zone while minimising fire proximity |
| Firefighter | Minimise fire area growth; prioritise cells threatening civilian routes |
| Rescuer | Minimise time-to-extraction for highest-priority injured civilian |
| Commander | Minimise total casualties; maintain operational phase coherence |

### 4.5 Learning

**Within a single run:** No online learning occurs. Agent decisions are fixed by pre-trained models.

**Across training runs (MARL):** Three agents (Firefighter, Rescuer, Commander) use Proximal Policy Optimization (PPO) trained via `train_marl.py` over 10,000 simulated episodes using a 9-scenario curriculum ordered by fire intensity (Bengio et al. 2009). Centralized Training Decentralized Execution (CTDE; Lowe et al. 2017): a shared 72-dimensional global critic is used during training; at inference each agent uses only its local observation (24-dim Firefighter, 22-dim Rescuer, 26-dim Commander). Commander obs expanded from 20 → 26 to include firefighter water levels, rescuer mission fractions, and mean civilian panic (Yu et al. 2022). BDI safety constraints are enforced via action masking — invalid actions are set to −∞ before argmax so PPO cannot select physically impossible or protocol-violating actions (Sardina & Thangarajah 2011). QMIX monotonic value decomposition (`src/rl/qmix.py`) addresses cooperative credit assignment (Rashid et al. 2018).

> Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
> Bengio, Y. et al. (2009). "Curriculum Learning." ICML-09, pp. 41–48.
> Lowe, R. et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NIPS, pp. 6379–6390.

The XGBoost models in `models/*.pkl` encode statistical patterns from batch simulation runs and are loaded at startup by `ml_predictor.py`. They are not updated during simulation.

### 4.6 Prediction

The `AnalystAgent` uses XGBoost regression to predict: fire spread rate, evacuation time, risk score, and resource requirement. Predictions inform `CommanderAgent` resource pre-positioning.

### 4.7 Sensing

Agents sense within their designated perceptual range:

| Agent | Sensing range | Mechanism |
|-------|---------------|-----------|
| Sentinel | Full grid | Direct grid read |
| Analyst | Full grid | Direct grid read + ML inference |
| Civilian | 50-cell radius | Grid cells + road network nodes |
| Firefighter | 30-cell radius | Fire grid sampling |
| Rescuer | 50-cell radius | Civilian position broadcast |

All sensing is **perfect within range** — no sensor noise or occlusion model is implemented. This is a deliberate simplification appropriate for the current validation stage.

### 4.8 Interaction

- **Civilian ↔ Civilian:** Panic contagion — nearby high-panic civilians raise neighbours' panic levels (social herding; Cova & Johnson 2002).
- **Civilian ↔ Road network:** Speed is determined by Greenshields traffic model based on density of civilians on the road segment.
- **Commander ↔ Firefighter/Rescuer:** CNP message passing (CFP → bid → accept/reject).
- **Sentinel/Analyst → Commander:** Broadcast of fire state and risk predictions via shared message queue.

### 4.9 Stochasticity

| Source | Distribution | Purpose |
|--------|-------------|---------|
| Fire ignition probability | Bernoulli(p) | Cell-by-cell spread; p from Rothermel ROS |
| Wind perturbation | Gaussian(0, σ) | Gustiness; σ = WIND_OSCILLATION_AMPLITUDE / 3 |
| Civilian starting positions | Uniform over road network | Initial placement |
| Panic contagion radius | Uniform(1, 5) cells | Social influence range |
| A* tie-breaking | Random | Equal-cost path selection |

`RANDOM_SEED = None` (default) ensures independent Monte Carlo runs. Set `RANDOM_SEED = 42` for reproducible single runs.

### 4.10 Collectives

No formal collective entities are defined. The `CommanderAgent` coordinates a de facto team of firefighters and rescuers, but they do not share internal state — coordination occurs exclusively via CNP message passing.

### 4.11 Observation

Per-step observations recorded to the web dashboard (SSE) and batch CSV:

- Fire cell count, burned area, fire perimeter
- Civilian count by state (active / evacuated / injured / dead)
- Mortality rate, evacuation success rate
- Average and maximum panic level
- Wind speed and direction
- Commander operational phase
- CNP refusal count
- AQI distribution, smoke-injured count

---

## 5. Initialisation

### 5.1 Simulation setup

1. `AIGISSimulation.__init__()` receives `lat`, `lon`, `radius`, `mode`, and optional `fire_locations` / `config_overrides`.
2. The environment downloads OSM road network and building footprints via `osmnx` (or loads from cache).
3. SRTM elevation data is fetched via `elevation` library.
4. Weather is initialised from `Open-Meteo` API (or falls back to `src/config.py` defaults).
5. The 200×200 grid is populated with land cover, fuel loads, and elevation values.
6. Agents are instantiated and placed:
   - **Civilians:** Randomly distributed across OSM road nodes.
   - **Firefighters / Rescuers:** Placed at the designated depot node (closest to grid centre).
   - **Commander / Sentinel / Analyst:** No spatial position (system-level agents).
7. Fire is ignited at the specified `fire_locations` (or default: grid centre).

### 5.2 Default parameters

See `src/config.py` for all parameters. Key defaults:

| Parameter | Default | Source |
|-----------|---------|--------|
| `NUM_CIVILIANS` | 60 | — |
| `NUM_FIREFIGHTERS` | 2 | — |
| `MAX_STEPS` | 500 | — |
| `FIRE_SPREAD_PROB_BASE` | 0.30 | Calibrated to Mati conditions |
| `ROTHERMEL_BASE_ROS` | 0.5 m/s | Rothermel (1972) |
| `WIND_SPEED` | 5.0 m/s | Open-Meteo default |
| `CIVILIAN_PANIC_RATIONAL` | 0.3 | Cova & Johnson (2002) |
| `CIVILIAN_V_FREE_FLOW` | 3.0 cells/step | Calibrated |
| `DISABLE_CNP` | False | Ablation flag |
| `DISABLE_PANIC` | False | Ablation flag |

---

## 6. Input Data

| Data source | Provider | Used for |
|-------------|----------|---------|
| Road network | OpenStreetMap via `osmnx` | Evacuation routing, agent placement, gridlock |
| Building footprints | OpenStreetMap | Grid obstacle layer |
| Land cover | OpenStreetMap tags | Fuel load classification |
| Elevation | SRTM via `elevation` library | Slope term in Rothermel ROS |
| Weather | Open-Meteo REST API | Initial temperature, RH, wind speed/direction |
| ML models | Pre-trained XGBoost (`.pkl`) | Risk prediction; trained offline |

For the Mati 2018 validation scenario, meteorological inputs are overridden with documented values from:
> Lagouvardos, K., Kotroni, V., Giannaros, T.M., & Dafis, S. (2019). "Meteorological analysis of the catastrophic wildfire in Mati, eastern Attica, Greece." *Bulletin of the American Meteorological Society*, 100(11), pp. 2243–2257. DOI: 10.1175/BAMS-D-18-0231.1

---

## 7. Submodels

### 7.1 Fire Spread — Rothermel Rate-of-Spread

> Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in wildland fuels.*
> USDA Forest Service Research Paper INT-115.

For each burning cell, the base rate of spread (ROS, m/s) is:

```
ROS = ROTHERMEL_BASE_ROS × fuel_type_factor × (1 + wind_factor) × (1 + slope_factor)
```

- `fuel_type_factor`: multiplier by OSM land cover tag (forest: 1.4, shrub: 1.0, grass: 0.8, urban: 0.3)
- `wind_factor`: cosine of angle between wind vector and spread direction × (wind_speed / 10)
- `slope_factor`: sin(slope_angle) — upslope spread enhancement

Ignition probability for each neighbour cell per step:

```
p_ignite = FIRE_SPREAD_PROB_BASE × η_M × wind_factor × slope_factor × fuel_factor
```

where **η_M** is the dead fuel moisture suppression factor (Rothermel 1972, Eq. 30–31):

```
EMC = f(T, RH)   [NFFL three-range tables; Rothermel 1983, cited in Nelson 2000]
η_M = 1 − 2.59(EMC/M_x) + 5.11(EMC/M_x)² − 3.52(EMC/M_x)³
M_x = 25 %  (shrub/brush extinction moisture, Anderson 1982)
```

Nelson, R.M. Jr. (2000). *Canadian Journal of Forest Research*, 30(7):1071–1087.

A Bernoulli trial determines whether the cell ignites.

**Firebrand spotting** (`_simulate_spotting()`) runs after each spread step.
Burning cells probabilistically loft embers downwind:

```
D_s (m) = 0.4 × U^1.5 × F_h^0.5   (Anderson 1983, INT-305 Table 1)
P_spot  = 0.005 per burning cell per step
Direction: wind-biased with σ = π/4 angular noise
```

Anderson, H.E. (1983). *USDA Forest Service Research Paper INT-305*.

### 7.2 Traffic Model — Greenshields

> Greenshields, B.D. (1935). "A study of traffic capacity."
> *Proceedings of the Highway Research Board*, 14, pp. 448–477.

Civilian travel speed on a road segment is modulated by density:

```
v = V_FREE_FLOW × (1 − density / density_max)
```

- `density` = number of civilians on the segment / segment length
- `density_max` = `GRIDLOCK_DENSITY_THRESHOLD` from config
- Speed clips to 0 when density ≥ density_max (gridlock)

When `current_speed < 0.1` for ≥ 3 consecutive steps, the civilian triggers the perimeter fallback (`_move_toward_perimeter`). After 30 steps with no grid-cell progress, the civilian is marked `is_injured=True, is_active=False` (trapped casualty).

### 7.3 Panic Dynamics — Three-State Cognitive Machine

> Cova, T.J. & Johnson, J.P. (2002). *Environment and Planning A*, 34(12):2211–2229.

Panic level evolves as:

```
Δpanic = α × fire_proximity_factor + β × social_contagion − γ × distance_to_safety
```

Where:
- `α` = `CIVILIAN_PANIC_INCREASE_RATE`
- `β` = `CIVILIAN_SOCIAL_CONTAGION_RATE`
- `γ` = `CIVILIAN_PANIC_DECREASE_RATE`
- `fire_proximity_factor` = `max(0, 1 - dist_to_fire / MAX_FIRE_DIST)`

State thresholds:
- Rational: `panic < CIVILIAN_PANIC_RATIONAL` (0.4)
- Confused: `0.4 ≤ panic < CIVILIAN_PANIC_CONFUSED` (0.7)
- Herding: `panic ≥ 0.7`

**Pre-evacuation milling delay:** When an EVACUATE order is received and
fire is not yet visible, a milling delay is sampled before departure begins:

```
delay ~ LogNormal(μ=5.204, σ=0.60)   → median ≈ 182 steps ≈ 15.2 min at 5 s/step
```

Bypassed when fire is directly visible (immediate flight response).

> Lindell, M.K. & Perry, R.W. (2012). "The Protective Action Decision Model."
> *Risk Analysis*, 32(4):616–632. Table 3.

### 7.4 Pathfinding — A* on OSM Road Network

> Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). "A formal basis for the heuristic determination of minimum cost paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2), pp. 100–107. DOI: 10.1109/TSSC.1968.300136

Civilians compute shortest-path routes using A* with Euclidean distance as the heuristic. The road network is a weighted graph (`osmnx` `MultiDiGraph`) with edge weights proportional to travel time. Dead-end or inaccessible routes (no path found) trigger the grid-space perimeter fallback.

### 7.5 CNP Task Allocation

> Smith, R.G. (1980). *IEEE Transactions on Computers*, C-29(12):1104–1113.

The `CommanderAgent` issues CFPs with a task specification (fire cell or civilian location). Each capable agent computes a bid cost:

```
bid_cost = distance_to_target + resource_penalty
```

- `resource_penalty = (1 - water_level/100) × 5` for firefighters
- `resource_penalty = 0` for rescuers

The commander awards to `argmin(bid_cost)`. Under `DISABLE_CNP=True` (Ablation A), the first received proposal is accepted without cost comparison.

### 7.6 Smoke and AQI

AQI per cell is computed from the count of active fire cells within a configurable radius, weighted by wind direction alignment with the target cell:

```
AQI = base_AQI × fire_cells_upwind × wind_alignment_factor
```

Civilians exposed to `AQI > SMOKE_AQI_INJURY_THRESHOLD` for ≥ `SMOKE_EXPOSURE_STEPS` consecutive steps are marked `is_injured=True`.

### 7.7 Machine Learning Risk Predictor

Three XGBoost regression models (`models/*.pkl`) are loaded by `ml_predictor.py`:
- `casualty_risk_model.pkl` — predicted casualty count
- `containment_time_model.pkl` — expected steps to fire containment
- `evacuation_count_model.pkl` — expected number of successful evacuations

Feature vector (per prediction call):
- Fire cells count, burned area, wind speed, wind direction, temperature, relative humidity, civilian count active, panic mean, steps elapsed

Models were trained on synthetic batch-run data using `train_models.py`.

### 7.8 Hybrid MARL — PPO Policy

The three field agents (Firefighter, Rescuer, Commander) replace their BDI `decide()` method with a PPO-trained policy at inference time. The BDI rules serve as a pre-training fallback when no `.pt` policy file is present.

Each policy is a two-layer MLP (64 hidden units, tanh activations):
- Firefighter actor: 24-dim obs → 5 actions {water_drop, fire_line, backburn, patrol, return_to_base}
- Rescuer actor: 22-dim obs → 4 actions {move_highest_panic, move_nearest, move_safe_zone, wait}
- Commander actor: 26-dim obs → 6 actions {maintain, advance, hold_prealert, force_evacuate, shelter, reassure}
  (+6 inter-agent dims vs. original 20: FF water levels, rescuer mission frac, nearest FF pos, mean panic)

A shared centralized critic (128→64→1) receives the concatenated 72-dim global state during training
(FF:24 + RSC:22 + CMD:26; CTDE). GAE (Schulman et al. 2016) is used to compute advantages.
BDI action masking constrains PPO to BDI-safe actions before argmax (Sardina & Thangarajah 2011).
QMIX mixing network (`src/rl/qmix.py`) provides monotonic cooperative credit assignment (Rashid et al. 2018).

---

## References

- Cova, T.J. & Johnson, J.P. (2002). *Environment and Planning A*, 34(12):2211–2229.
- Greenshields, B.D. (1935). *Proceedings of the Highway Research Board*, 14:448–477.
- Grimm, V. et al. (2020). *JASSS*, 23(2):7. DOI: 10.18564/jasss.4259
- Hart, P.E., Nilsson, N.J., & Raphael, B. (1968). *IEEE Trans. Systems Science and Cybernetics*, 4(2):100–107.
- Lagouvardos, K. et al. (2019). *BAMS*, 100(11):2243–2257.
- Mann, H.B. & Whitney, D.R. (1947). *Annals of Mathematical Statistics*, 18(1):50–60.
- Mas, E. et al. (2021). *Transportation Research Part D*, 99:103007.
- Rao, A.S. & Georgeff, M.P. (1995). *Proceedings of ICMAS-95*, pp. 312–319.
- Rothermel, R.C. (1972). *USDA Forest Service Research Paper INT-115*.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Saltelli, A. et al. (2010). *Computer Physics Communications*, 181(2):259–270. [Sobol estimator]
- Sardina, S. & Thangarajah, J. (2011). *Proc. 22nd IJCAI*, pp. 1810–1815. [BDI action masking]
- Smith, R.G. (1980). *IEEE Transactions on Computers*, C-29(12):1104–1113.
- Anderson, H.E. (1983). *USDA Forest Service Research Paper INT-305*. [Spotting distance]
- Albini, F.A. (1979). *USDA Forest Service Research Paper INT-56*. [Spotting distribution]
- Nelson, R.M. Jr. (2000). *Canadian Journal of Forest Research*, 30(7):1071–1087. [Dead fuel moisture]
- Lindell, M.K. & Perry, R.W. (2012). *Risk Analysis*, 32(4):616–632. [Milling delay]
- Filippi, J.B., Mallet, V., & Nader, B. (2016). *Environmental Modelling & Software*, 80:262–276. [Jaccard/IoU]
- Rashid, T. et al. (2018). *ICML 2018*, PMLR 80:4295–4304. [QMIX]
- Yu, C. et al. (2022). *NeurIPS 2022*. arXiv:2103.01955. [MAPPO inter-agent obs]
- Wilensky, U. & Rand, W. (2015). *An Introduction to Agent-Based Modeling*. MIT Press.

**Sensitivity analysis** (`run_sensitivity.py`) uses Sobol variance-based global
sensitivity analysis (Saltelli et al. 2010) rather than one-at-a-time (OAT).
Outputs: first-order Si and total-effect STi indices with 95% bootstrap CI.
N=128 Saltelli samples → 1792 model runs (N × (2D+2), D=6 parameters).

**Spatial validation** (`validate_mati.py`) computes Jaccard/IoU between the
simulated burn scar and a Copernicus EMSR249-derived reference ellipse
(Filippi et al. 2016). Threshold J ≥ 0.30 = adequate (Copernicus EMS QA 2018).
