# AIGIS System Architecture & Business Logic

## Table of Contents
1. [System Overview](#system-overview)
2. [Agent Architectures](#agent-architectures)
3. [Decision Logic](#decision-logic)
4. [Communication Protocol](#communication-protocol)
5. [Core Algorithms](#core-algorithms)
6. [Data Flow](#data-flow)

---

## System Overview

AIGIS is a **Multi-Agent System (MAS)** for disaster management simulation featuring 5 autonomous agents with different architectural patterns, demonstrating how heterogeneous agents collaborate in crisis scenarios.

### Core Philosophy
- **Location-Agnostic**: Works anywhere globally using OpenStreetMap
- **Physics-Based**: Rothermel fire model, Greenshields traffic, Social Force herding
- **Scientifically Grounded**: Each agent uses established AI architectures and real-world models
- **Research-Ready**: Monte Carlo experiments with statistical analysis

---

## Agent Architectures

### 1. Sentinel Agent - **Reactive Architecture**

**Purpose**: Fire detection sensors distributed around the perimeter

**Architecture Pattern**: Simple Reflex Agent (Condition-Action Rules)

**Business Logic**:
```
IF fire_detected_in_vision_radius THEN
    IF consecutive_detections >= 3 THEN  // Debouncing
        SEND fire_alert TO analyst
    END IF
END IF
```

**Key Features**:
- **Signal Detection Theory**: `I_detected = I_actual/(d² + ε) × (1 + cos(θ)) + N(0,σ)`
- **Environmental Attenuation**: Distance and wind affect detection accuracy
- **Debouncing Protocol**: Requires 3 consecutive detections to avoid false positives
- **No Memory**: Purely reactive - responds only to current perceptions

**Decision Trigger**: Fire cell enters vision radius

---

### 2. Analyst Agent - **Model-Based Reflex Architecture**

**Purpose**: Risk assessment and fire spread prediction

**Architecture Pattern**: Model-Based Agent (maintains internal model of world state)

**Business Logic**:
```
PERCEIVE:
    Collect fire_reports FROM sentinels
    Build fire_map (internal world model)

DECIDE:
    FOR each fire location:
        Calculate ROS using Rothermel model
        ROS = R_base × (1 + φ_wind) × (1 + φ_slope)

        Calculate TTI (Time To Impact)
        TTI = distance_to_population / ROS

    END FOR

    Apply Fuzzy Logic:
    IF TTI < 5min AND exits_blocked THEN
        risk = CRITICAL
    ELSE IF TTI < 15min THEN
        risk = HIGH
    ELSE
        risk = MEDIUM
    END IF

ACT:
    SEND risk_report TO commander
    Report: {max_risk, TTI, ROS, num_exits}
```

**Key Features**:
- **Internal Model**: Maintains fire map from sensor reports
- **Rothermel Physics**: Rate of Spread based on wind, slope, fuel
- **Fuzzy Logic Risk Assessment**: 3 fuzzy variables (TTI, exit capacity, fire intensity)
- **Predictive**: Projects future fire behavior

**Decision Trigger**: New fire reports OR periodic re-evaluation (every 10 steps)

---

### 3. Commander Agent - **Hybrid (Utility-Based + Deliberative)**

**Purpose**: Strategic decision making and resource coordination

**Architecture Pattern**: Hybrid Architecture with ECT vs TTI logic

**Business Logic**:
```
PERCEIVE:
    Receive risk_report FROM analyst
    Extract: TTI, ROS, num_exits
    Track active_missions status

DECIDE:
    // Calculate Evacuation Clearance Time
    ECT = (N_civilians / (C_exit × num_exits)) × γ_congestion

    // Determine Phase based on TTI/ECT ratio
    IF TTI > 2.5 × ECT THEN
        phase = 0  // Monitoring
    ELSE IF TTI > 1.5 × ECT THEN
        phase = 1  // Pre-Alert
    ELSE IF TTI > 1.0 × ECT THEN
        phase = 2  // Mass Evacuation
    ELSE
        phase = 3  // Shelter-in-Place (TOO LATE!)
    END IF

    // Evaluate pending rescue proposals
    FOR each proposal IN pending_proposals:
        utility = w_safety × (100/ETA)
                - w_cost × cost
                - w_congestion × active_missions

        IF utility > best_utility THEN
            best_proposal = proposal
        END IF
    END FOR

ACT:
    // Phase-specific actions
    CASE phase:
        0: Monitor (no action)
        1: BROADCAST warning TO civilians
        2: BROADCAST evacuation_order TO civilians
           SEND CFP (rescue missions) TO rescuers
        3: BROADCAST redirect_to_safe_zone TO civilians
    END CASE

    // Accept best rescue proposal
    SEND accept_proposal TO best_rescuer
    SEND reject_proposal TO other_rescuers
```

**Key Features**:
- **ECT Calculation**: `ECT = (N_agents / C_exit) × γ` where γ accounts for congestion
- **4-Phase Protocol**:
  - Phase 0: Monitoring (TTI > 2.5×ECT)
  - Phase 1: Pre-Alert (1.5×ECT < TTI ≤ 2.5×ECT)
  - Phase 2: Mass Evacuation (1.0×ECT < TTI ≤ 1.5×ECT)
  - Phase 3: Shelter-in-Place (TTI ≤ ECT) → Too late, go to nearest safe zone
- **Contract Net Protocol**: Sends CFP, evaluates proposals, selects best bid
- **Utility Function**: Balances safety, cost, and congestion
- **Re-evaluation**: Periodically reassesses strategy (Commitment with Evaluation)

**Decision Trigger**: Risk report received OR re-evaluation interval reached

---

### 4. Rescuer Agent - **Goal-Based (Practical Reasoning)**

**Purpose**: Execute rescue missions with risk-aware pathfinding

**Architecture Pattern**: Goal-Based Agent with BDI elements

**Business Logic**:
```
PERCEIVE:
    Receive CFP FROM commander
    Extract: mission_location, priority

DECIDE:
    // Calculate path to mission location
    path = A_star(current_position, mission_location)

    // Assess risk along path
    risk_score = 0
    FOR each node IN path:
        IF fire_grid[node] == BURNING THEN
            risk_score = INFINITY  // REFUSE mission through fire!
            BREAK
        END IF
        risk_score += temperature[node] × distance
    END FOR

    // Calculate bid
    IF risk_score == INFINITY THEN
        SEND refuse TO commander
        RETURN
    END IF

    cost = path_length + (risk_score × α) + fuel_consumed
    ETA = path_length / speed

ACT:
    SEND proposal TO commander
    Content: {cost, ETA, risk_score}

    IF accepted THEN
        // Execute mission
        WHILE NOT reached_target:
            move_along_path()

            // Dynamic re-routing if path becomes dangerous
            IF fire_on_path THEN
                recalculate_path()
            END IF
        END WHILE

        SEND confirm (mission complete) TO commander
    END IF
```

**Key Features**:
- **Risk-Aware Pathfinding**: Refuses missions through active fire
- **A* Navigation**: Uses networkx shortest path with risk-weighted edges
- **Dynamic Re-routing**: Recalculates path if conditions change
- **Safety Protocol**: `IF fire_on_path THEN refuse OR reroute`
- **Bidding System**: Cost = time + risk + fuel

**Decision Trigger**: CFP received OR mission accepted

---

### 5. Civilian Agent - **BDI (Belief-Desire-Intention)**

**Purpose**: Evacuate with realistic panic psychology and crowd dynamics

**Architecture Pattern**: Full BDI with 3-state cognitive machine

**Business Logic**:
```
PERCEIVE:
    // Update beliefs from messages
    IF warning_received THEN
        ADD 'warning' TO beliefs
        panic_level += 0.1
    END IF

    IF evacuation_ordered THEN
        ADD 'evacuation_ordered' TO beliefs
        panic_level += 0.3
    END IF

    IF shelter_in_place THEN
        ADD 'shelter_in_place' TO beliefs
        panic_level += 0.5  // High panic!
        redirect_to_coast = TRUE
    END IF

    // Calculate fire distance
    fire_distance = INFINITY
    FOR each cell IN vision_radius:
        IF fire_grid[cell] == BURNING THEN
            fire_visible = TRUE
            distance = euclidean_distance(position, cell)
            fire_distance = MIN(fire_distance, distance)
        END IF
    END FOR

    // Update panic using panic equation
    IF fire_visible THEN
        panic_level += α × (1 / fire_distance)
    ELSE
        panic_level -= decay_rate  // Decay when no fire
    END IF

    IF family_separated THEN
        panic_level += β × 0.1
    END IF

    // Assess local traffic density
    density = count_agents_nearby / area

DECIDE:
    // Determine cognitive state based on panic
    IF panic_level < 0.4 THEN
        cognitive_state = "rational"
    ELSE IF panic_level < 0.7 THEN
        cognitive_state = "confused"
    ELSE
        cognitive_state = "herding"
    END IF

    // Decision making per cognitive state
    CASE cognitive_state:
        rational:
            intentions = ['evacuate']
            // Use optimal A* pathfinding to nearest safe zone

        confused:
            intentions = ['evacuate']
            // Occasionally reconsider path (hesitation)
            IF random() < 0.2 THEN
                clear_path()  // Force re-routing
            END IF

        herding:
            // High panic - follow the crowd!
            IF random() < 0.2 THEN
                intentions = ['freeze']  // Panic freeze
            ELSE IF nearby_agents > 0 THEN
                intentions = ['follow_crowd']  // Social Force
            ELSE
                intentions = ['move_random']  // Panic movement
            END IF
    END CASE

ACT:
    // Calculate speed using Greenshields Traffic Model
    V_current = V_free_flow × (1 - density / density_jam)

    IF density >= density_jam THEN
        V_current = 0  // GRIDLOCK!
    END IF

    // Apply cognitive state speed reduction
    IF cognitive_state == "confused" THEN
        V_current × 0.5  // 50% speed reduction
    END IF

    // Execute intention
    CASE primary_intention:
        'freeze':
            // Do nothing (panic freeze)

        'move_random':
            // Random panic movement
            direction = random()
            position += direction × V_current

        'follow_crowd':
            // Social Force Model (Herding)
            avg_direction = Σ(nearby_agents.movement) / count
            position += avg_direction × V_current

        'evacuate':
            // Goal-directed movement to safe zone
            IF path_empty OR redirect_to_coast THEN
                safety_node = find_nearest_safe_node()  // OSM water/parks/edges
                path = A_star(current_node, safety_node)
            END IF

            IF V_current > 0.1 THEN
                next_node = path[1]
                move_to(next_node)
            ELSE
                // GRIDLOCK - cannot move
            END IF
    END CASE
```

**Key Features**:
- **BDI Components**:
  - **Beliefs**: {warning_received, evacuation_ordered, fire_visible, ...}
  - **Desires**: {survive, reach_safety, find_family}
  - **Intentions**: {evacuate, follow_crowd, freeze, move_random}
- **3-State Cognitive Machine**:
  - **Rational** (panic < 0.4): Optimal pathfinding, full speed
  - **Confused** (0.4-0.7): 50% speed, frequent re-routing, hesitation
  - **Herding** (panic ≥ 0.7): Follows crowd via Social Force, ignores optimal path
- **Panic Equation**: `Panic(t) = Panic(t-1) + α×(1/d_fire) + β×(family_separated)`
- **Greenshields Traffic Model**: `V = V_free × (1 - ρ/ρ_jam)` → Gridlock at jam density
- **Social Force Herding**: Calculates average movement of nearby agents
- **Dynamic Safe Zone Detection**: Uses OSM tags (water, parks) + map edges

**Decision Trigger**: Every step (continuous perception-decision-action loop)

---

## Communication Protocol

### FIPA-ACL Message Structure

```python
Message:
    sender: agent_id
    receiver: agent_id | "broadcast"
    performative: FIPA_performative
    content: dict
    conversation_id: uuid
```

### Performatives Used

| Performative | Usage | Example |
|--------------|-------|---------|
| **INFORM** | Share information | Sentinel → Analyst: fire detected |
| **REQUEST** | Request action | Commander → Civilians: evacuate |
| **CFP** | Call For Proposal | Commander → Rescuers: rescue mission |
| **PROPOSE** | Bid on task | Rescuer → Commander: proposal with cost |
| **ACCEPT_PROPOSAL** | Accept bid | Commander → Rescuer: you're selected |
| **REJECT_PROPOSAL** | Reject bid | Commander → Rescuer: bid rejected |
| **REFUSE** | Refuse task | Rescuer → Commander: too dangerous |
| **CONFIRM** | Task complete | Rescuer → Commander: mission done |

### Communication Flow

```
1. Fire Detection:
   Sentinel --[INFORM: fire_detected]--> Analyst

2. Risk Assessment:
   Analyst --[INFORM: risk_report]--> Commander

3. Evacuation Order:
   Commander --[REQUEST: evacuate]--> Civilians (broadcast)

4. Rescue Coordination (Contract Net Protocol):
   Commander --[CFP: rescue_mission]--> Rescuers (broadcast)
   Rescuers --[PROPOSE: bid]--> Commander
   Commander --[ACCEPT_PROPOSAL]--> Best Rescuer
   Commander --[REJECT_PROPOSAL]--> Other Rescuers
   Rescuer --[CONFIRM: complete]--> Commander
```

---

## Core Algorithms

### 1. Fire Spread (Rothermel Model)

```
Rate of Spread (ROS):
    ROS = R_base × (1 + φ_wind) × (1 + φ_slope)

Wind Factor:
    φ_wind = C × U^B × (cos(θ_difference))
    where:
        C = 7.47 (wind coefficient)
        U = wind speed (m/s)
        B = 0.785 (exponent)
        θ_difference = angle between wind and spread direction

Slope Factor:
    φ_slope = 5.275 × (tan(slope))^2

Dynamic Wind:
    θ(t) = θ_0 + sin(t / T_period) × A_amplitude
    Updates every step
```

### 2. Perlin Noise Terrain

```
Elevation Generation:
    FOR each grid cell (x, y):
        noise_value = pnoise2(
            x / PERLIN_SCALE,
            y / PERLIN_SCALE,
            octaves = PERLIN_OCTAVES,
            persistence = 0.5,
            lacunarity = 2.0
        )

        elevation[y, x] = BASE_HEIGHT + (noise_value × AMPLITUDE)
    END FOR

Parameters:
    - PERLIN_SCALE = 100.0 (controls feature size)
    - PERLIN_OCTAVES = 4 (detail layers)
    - BASE_HEIGHT = 100.0m
    - AMPLITUDE = 50.0m
```

### 3. Safe Zone Detection (OSM)

```
Safe Zone Identification:
    safe_nodes = {}

    // 1. Fetch OSM features
    FOR each tag_category, tag_values IN SAFE_ZONE_TAGS:
        features = osm.features_from_place(
            location,
            tags={tag_category: tag_values}
        )

        FOR each feature IN features:
            nearby_nodes = graph.nearest_nodes(feature.geometry)
            safe_nodes.add(nearby_nodes)
        END FOR
    END FOR

    // 2. Add map perimeter nodes
    perimeter_nodes = get_perimeter_nodes(graph)
    safe_nodes.update(perimeter_nodes)

    RETURN safe_nodes

SAFE_ZONE_TAGS = {
    'natural': ['water', 'beach', 'coastline'],
    'leisure': ['park', 'nature_reserve', 'playground'],
    'place': ['square']
}
```

### 4. Greenshields Traffic Model

```
Current Speed Calculation:
    V_current = V_free_flow × (1 - ρ_local / ρ_jam)

Where:
    V_free_flow = 5.0 (m/s) - speed in free flow
    ρ_jam = 10.0 (agents/area) - jam density
    ρ_local = count_agents_nearby / area

Gridlock Condition:
    IF ρ_local ≥ ρ_jam THEN
        V_current = 0  // Cannot move!
    END IF

Cognitive State Modifier:
    IF state == "confused" THEN
        V_current × 0.5  // 50% reduction
    END IF
```

### 5. Social Force Model (Herding)

```
Herding Behavior (when panic ≥ 0.7):

    // Find nearby agents within vision radius
    nearby_agents = []
    FOR each other_agent IN all_civilians:
        distance = euclidean_distance(self, other_agent)
        IF distance ≤ vision_radius THEN
            nearby_agents.add(other_agent)
        END IF
    END FOR

    // Calculate average movement direction
    avg_direction = Vector2D(0, 0)
    FOR each agent IN nearby_agents:
        IF agent.last_movement EXISTS THEN
            avg_direction += agent.last_movement
        END IF
    END FOR

    avg_direction = normalize(avg_direction)

    // Apply herding force
    movement = avg_direction × V_current × HERDING_INFLUENCE

    // Update position
    new_position = position + movement

    // Store movement for others to follow
    last_movement = new_position - position
```

---

## Data Flow

### Simulation Loop

```
INITIALIZATION:
    1. Build environment from OSM (roads, buildings, forests)
    2. Generate Perlin noise terrain
    3. Identify safe zones (OSM + perimeter)
    4. Initialize 5 agent types
    5. Ignite random fires

MAIN LOOP (each step):
    1. UPDATE FIRE:
        - Calculate dynamic wind direction
        - Spread fire using Rothermel model
        - Update fire grid

    2. UPDATE AGENTS (in order):
        a. Sentinels:
            perceive() → detect fire with attenuation
            decide() → check debouncing threshold
            act() → send fire alerts

        b. Analyst:
            perceive() → collect fire reports
            decide() → calculate ROS, TTI, risk
            act() → send risk report to commander

        c. Commander:
            perceive() → receive risk reports
            decide() → calculate ECT, determine phase, evaluate proposals
            act() → broadcast orders, dispatch rescuers

        d. Rescuers:
            perceive() → receive CFPs, mission status
            decide() → assess path risk, calculate bid
            act() → send proposals, execute missions

        e. Civilians:
            perceive() → check fire visibility, messages
            decide() → update panic, determine cognitive state
            act() → move based on state (rational/confused/herding)

    3. ROUTE MESSAGES:
        - Collect all outbox messages from agents
        - Route to recipients (direct or broadcast)
        - Deliver to inboxes
        - Clear outboxes

    4. UPDATE CIVILIAN NEIGHBORS:
        - For civilians in "herding" state:
            find_nearby_agents() for Social Force calculation

    5. COLLECT METRICS:
        - Count casualties (civilians in fire)
        - Count evacuated (civilians at safe zones)
        - Track panic levels
        - Track active fires
        - Record commander phase

    6. CHECK TERMINATION:
        - Fire burnt out? (no burning cells AND no fuel)
        - All evacuated? (active civilians == evacuated)
        - Max steps reached?

        IF complete THEN BREAK

POST-SIMULATION:
    - Calculate final statistics
    - Export to CSV (if batch mode)
    - Display dashboard (if GUI mode)
```

### Agent Update Cycle (Universal Pattern)

```
FOR each agent:
    perceive(environment):
        - Read sensor data
        - Check messages inbox
        - Update internal state

    decide():
        - Process perceptions
        - Apply agent-specific logic
        - Plan actions

    act(environment):
        - Execute planned actions
        - Send messages
        - Update position
        - Modify environment

    clear_messages():
        - Empty inbox after processing
```

---

## Key Design Decisions

### 1. Why 5 Different Agent Architectures?

**Educational Purpose**: Demonstrates heterogeneous multi-agent systems where agents with different architectures collaborate. Shows that:
- Simple reactive agents (Sentinel) can coexist with complex deliberative agents (Civilian)
- No single architecture is "best" - each fits specific roles
- Coordination emerges from communication, not centralized control

### 2. Why ECT vs TTI Logic?

**Real-World Basis**: Based on the 2018 Mati fire disaster where evacuation was ordered too late, causing traffic gridlock. People died in cars or ran to the beach.

**Key Insight**: If evacuation takes longer than time until fire arrives → DON'T EVACUATE, go to nearest safe zone (water, park, open space).

### 3. Why 3 Cognitive States for Civilians?

**Psychological Realism**: Research shows panic affects decision-making quality:
- **Rational**: Normal state, optimal decisions
- **Confused**: Moderate panic, degraded performance (50% speed, hesitation)
- **Herding**: High panic, follows crowd even to dangerous areas (documented in disasters)

### 4. Why Location-Agnostic?

**Generalizability**: Original version only worked in one hardcoded location. Now works anywhere:
- Perlin noise replaces hardcoded terrain
- OSM tags identify safe zones universally
- No "east coastline" assumptions

### 5. Why Monte Carlo Mode?

**Research Validation**: Single runs don't prove anything. Need statistical confidence:
- Run 100+ simulations
- Get mean ± standard deviation
- Compare strategies with statistical significance
- Publish results in academic papers

---

## Performance Considerations

### Computational Complexity

| Component | Time Complexity | Bottleneck |
|-----------|-----------------|------------|
| Fire Spread | O(W × H) | Grid size |
| Agent Updates | O(N_agents) | Number of agents |
| Pathfinding | O(E log V) | Graph size |
| Herding | O(N_civilians²) | Vision radius |
| Message Routing | O(N_messages) | Communication density |

### Optimization Strategies

1. **Smaller Grids**: 100×100 instead of 200×200
2. **Limited Vision**: Civilians scan 15-cell radius, not entire grid
3. **Periodic Updates**: Commander re-evaluates every 10 steps, not every step
4. **Headless Mode**: Disable visualization for batch experiments
5. **Caching**: OSM data cached for repeated runs at same location

---

## Extension Points

### Easy Extensions (Configuration Changes)

1. **More Agents**: Increase `NUM_CIVILIANS`, `NUM_RESCUERS` in config
2. **Larger Maps**: Increase `MAP_RADIUS`, `GRID_SIZE`
3. **Different Locations**: Change `MAP_CENTER_LAT`, `MAP_CENTER_LON`
4. **Fire Parameters**: Adjust `FIRE_SPREAD_PROBABILITY`, wind parameters

### Medium Extensions (Code Modifications)

1. **Multiple Fires**: Modify `ignite_random_fires()` to start N fires
2. **Agent Learning**: Add Q-learning to Rescuer pathfinding
3. **Dynamic Obstacles**: Road closures from fire damage
4. **Communication Delays**: Add network latency to message routing

### Hard Extensions (Architecture Changes)

1. **3D Visualization**: Replace matplotlib with Unity/Unreal
2. **Real-Time Data**: Integrate live weather APIs
3. **Human-in-the-Loop**: Allow user to control Commander
4. **Distributed Simulation**: Run agents on separate processes/machines

---

## Summary

AIGIS demonstrates a complete Multi-Agent System with:
- ✅ **5 Different Architectures**: Reactive, Model-Based, Hybrid, Goal-Based, BDI
- ✅ **Physics-Based Models**: Rothermel fire, Greenshields traffic, Social Force herding
- ✅ **Real-World Grounding**: ECT vs TTI from actual disasters
- ✅ **Location-Agnostic**: Works globally via OSM
- ✅ **Research-Ready**: Monte Carlo with statistical analysis

**Core Philosophy**: "The right architecture for the right role, coordinated through communication."
