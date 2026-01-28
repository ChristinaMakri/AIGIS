# AIGIS Location-Agnostic System - Implementation Status

## ✅ COMPLETED (Tasks 1-3)

### 1. Configuration (src/config.py) - ✅ COMPLETE
- Added 217 lines of comprehensive configuration
- Perlin noise parameters for terrain generation
- Dynamic wind parameters (θ(t) = θ_0 + sin(t/50) × 20°)
- Safe zone OSM tags (water, parks, squares)
- Monte Carlo batch mode settings
- Dashboard visualization settings
- All physics parameters fully configurable

### 2. Environment (src/environment.py) - ✅ COMPLETE
- **Perlin Noise Terrain**: Generates realistic elevation using `noise` library
- **Dynamic Safe Zone Detection**:
  - Fetches OSM features (water, parks, squares)
  - Identifies perimeter nodes (map edges) as safe zones
  - Provides `find_nearest_safe_node()` method for agents
- **Location-Agnostic**: Works anywhere in the world
- **No Hardcoded Directions**: Removed all "east" references

### 3. Fire Simulation (src/fire_simulation.py) - ✅ COMPLETE
- **Dynamic Wind**: θ(t) = θ_0 + sin(t/50) × 20°
- **Rothermel Physics**: Slope and wind factors properly implemented
- Wind vector updates every step
- Logs wind direction changes every 10 steps
- Temperature grid tracking for path risk assessment

### 4. Requirements (requirements.txt) - ✅ UPDATED
- Added `noise>=1.2.2` for Perlin terrain generation

---

## 🔄 REMAINING TASKS (4-8)

### Task 4: Update Civilian Agent for Dynamic Safe Zones

**File**: `src/agents/civilian.py`

**Changes Needed**:
1. Remove hardcoded `_find_safety_node()` method that finds "easternmost" node
2. Replace with `environment.find_nearest_safe_node(self.current_node)`
3. Update `_move_to_safety()` to use dynamic safe zones
4. Implement proper **Social Force Model** for herding:
   - At panic > 0.8, calculate average direction vector of visible neighbors
   - Move towards that direction (even if suboptimal)
5. Update redirect logic to use nearest safe zone (not hardcoded coast)

**Key Implementation**:
```python
def _find_nearest_safe_zone(self, environment):
    """Use environment's dynamic safe zone detection"""
    if self.current_node is None:
        self.current_node = environment.get_nearest_node(self.position[0], self.position[1])

    # Use environment's method
    self.safety_node = environment.find_nearest_safe_node(self.current_node)

def _calculate_herding_direction(self, environment):
    """Social Force Model: Calculate average direction of neighbors"""
    if not self.nearby_agents:
        return None

    # Calculate average movement vector
    avg_direction = np.zeros(2)
    for agent in self.nearby_agents:
        if hasattr(agent, 'movement_direction'):
            avg_direction += agent.movement_direction

    if np.linalg.norm(avg_direction) > 0:
        return avg_direction / np.linalg.norm(avg_direction)
    return None
```

---

### Task 5: Update Commander Agent for Dynamic Safe Zones

**File**: `src/agents/commander.py`

**Changes Needed**:
1. Remove hardcoded "redirect to coast" message
2. Update `_order_shelter_in_place()` to broadcast "REDIRECT_TO_NEAREST_SAFE_ZONE"
3. Remove all references to hardcoded directions ("east", "coast")

**Key Implementation**:
```python
def _order_shelter_in_place(self) -> None:
    """
    CRITICAL: Too late to evacuate - roads will jam.
    Redirect civilians to nearest safe zone (park, water, map edge).
    """
    message = Message(
        sender=self.agent_id,
        receiver="broadcast",
        performative="REQUEST",
        content={
            'type': 'REDIRECT_TO_SAFE_ZONE',
            'urgency': 'CRITICAL',
            'message': 'TOO LATE TO EVACUATE - PROCEED TO NEAREST SAFE ZONE'
        }
    )
    self.send_message(message)
```

---

### Task 6: Create simulation.py (NEW FILE)

**File**: `src/simulation.py` (create new)

**Purpose**: Main simulation engine with data logging for Monte Carlo

**Structure**:
```python
class AIGISSimulation:
    def __init__(self, lat, lon, radius, mode='gui'):
        self.mode = mode  # 'gui' or 'batch'
        self.metrics = {
            'casualties': [],
            'evacuations': [],
            'panic_levels': [],
            'steps_to_evacuate': 0,
            # ... more metrics
        }

        # Build environment
        # Initialize agents
        # Initialize fire

    def run_step(self):
        """Execute one simulation step"""
        # Update fire
        # Update agents
        # Route messages
        # Collect metrics

    def run_until_complete(self, max_steps):
        """Run simulation until completion or max_steps"""
        while not self.is_complete() and self.step < max_steps:
            self.run_step()
        return self.get_results()

    def get_results(self):
        """Return metrics dictionary for CSV export"""
        return {
            'steps_to_evacuate': self.steps,
            'mortality_rate': self.calc_mortality(),
            'evacuation_success_rate': self.calc_success(),
            # ...
        }

    def export_to_csv(self, filename):
        """Export results to CSV for batch mode"""
        import csv
        # Write metrics to CSV
```

---

### Task 7: Rewrite main.py with CLI Arguments (NEW FILE)

**File**: `main.py` (completely rewrite)

**Purpose**: CLI entry point with argparse

**Structure**:
```python
import argparse
from src.simulation import AIGISSimulation

def parse_args():
    parser = argparse.ArgumentParser(description='AIGIS Multi-Agent Wildfire Evacuation Simulation')
    parser.add_argument('--lat', type=float, default=38.04, help='Center latitude')
    parser.add_argument('--lon', type=float, default=23.80, help='Center longitude')
    parser.add_argument('--radius', type=float, default=2000, help='Map radius (meters)')
    parser.add_argument('--batch', type=int, default=None, help='Monte Carlo: Number of runs')
    parser.add_argument('--mode', choices=['gui', 'headless'], default='gui')
    parser.add_argument('--output', default='results.csv', help='Output CSV file for batch mode')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.batch:
        # Batch/Monte Carlo mode
        print(f"🔬 Running Monte Carlo with {args.batch} simulations...")
        run_monte_carlo(args)
    else:
        # Single GUI mode
        print(f"🎮 Running single simulation...")
        run_gui_mode(args)

def run_monte_carlo(args):
    """Run N simulations headless and export results"""
    results = []
    for i in range(args.batch):
        print(f"  Run {i+1}/{args.batch}...")
        sim = AIGISSimulation(args.lat, args.lon, args.radius, mode='batch')
        result = sim.run_until_complete(MAX_STEPS)
        results.append(result)

    # Export to CSV
    # Print statistics (mean, std dev)

def run_gui_mode(args):
    """Run single simulation with dashboard"""
    sim = AIGISSimulation(args.lat, args.lon, args.radius, mode='gui')
    dashboard = Dashboard(sim)  # See Task 8
    dashboard.run()
```

---

### Task 8: Create Professional Dashboard (NEW FILE)

**File**: `src/dashboard.py` (create new)

**Purpose**: Real-time visualization with matplotlib GridSpec

**Structure**:
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Dashboard:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig = plt.figure(figsize=FIGURE_SIZE)

        # GridSpec layout
        gs = gridspec.GridSpec(2, 2, figure=self.fig)
        self.ax_map = self.fig.add_subplot(gs[:, 0])      # Left: Main map
        self.ax_casualties = self.fig.add_subplot(gs[0, 1])  # Top-right: Casualties
        self.ax_evacuations = self.fig.add_subplot(gs[1, 1])  # Bottom-right: Evacuations

        plt.ion()

        # Data history
        self.steps = []
        self.casualties_history = []
        self.evacuations_history = []

    def update(self):
        """Update all plots"""
        self._update_map()
        self._update_casualties_chart()
        self._update_evacuations_chart()
        plt.draw()
        plt.pause(STEP_DELAY)

    def _update_map(self):
        """Draw main map with fire, agents, safe zones"""
        self.ax_map.clear()
        # Draw fire grid
        # Draw road network
        # Draw safe zones (green highlights)
        # Draw agents

    def _update_casualties_chart(self):
        """Line chart of cumulative casualties"""
        self.ax_casualties.clear()
        self.ax_casualties.plot(self.steps, self.casualties_history)
        self.ax_casualties.set_title('Casualties Over Time')

    def _update_evacuations_chart(self):
        """Line chart of successful evacuations"""
        self.ax_evacuations.clear()
        self.ax_evacuations.plot(self.steps, self.evacuations_history)
        self.ax_evacuations.set_title('Evacuations Over Time')

    def run(self):
        """Main dashboard loop"""
        while self.simulation.step < MAX_STEPS:
            self.simulation.run_step()
            self.steps.append(self.simulation.step)
            self.casualties_history.append(self.simulation.count_casualties())
            self.evacuations_history.append(self.simulation.count_evacuated())
            self.update()
```

---

## 📝 CRITICAL NOTES

### Safe Zone Visualization
In the dashboard, safe zones should be highlighted in **light green** on the map to show civilians where to evacuate.

### Social Force Herding
At high panic (> 0.8), civilians should:
1. Scan for nearby agents within `CIVILIAN_VISION_RADIUS`
2. Calculate average movement direction
3. Move towards that direction (weighted by `CIVILIAN_HERDING_INFLUENCE`)
4. This can lead to **dead ends** - realistic tragedy simulation

### Monte Carlo Output Format
CSV should have columns:
```
run_id, steps_to_evacuate, mortality_rate, evacuation_success_rate, avg_panic_level, rescuer_refusals, total_burning_cells
```

---

## 🚀 NEXT STEPS

1. **Update Civilian Agent** (Task 4)
2. **Update Commander Agent** (Task 5)
3. **Create simulation.py** (Task 6)
4. **Rewrite main.py** (Task 7)
5. **Create dashboard.py** (Task 8)
6. **Test with different locations**:
   - `--lat 40.7128 --lon -74.0060` (New York)
   - `--lat 34.0522 --lon -118.2437` (Los Angeles)
   - `--lat -33.8688 --lon 151.2093` (Sydney)
7. **Run Monte Carlo**: `python main.py --batch 100`

---

## ✅ VALIDATION CHECKLIST

- [ ] All files compile without syntax errors
- [ ] `python -m py_compile src/*.py src/agents/*.py`
- [ ] Perlin terrain generates properly
- [ ] Safe zones are detected dynamically
- [ ] Wind direction oscillates (check logs)
- [ ] Civilians navigate to nearest safe zone
- [ ] Social force herding activates at panic > 0.8
- [ ] Dashboard displays real-time graphs
- [ ] Monte Carlo exports to CSV
- [ ] Works with any lat/lon coordinates

---

## 📦 DOCKER COMPATIBILITY

The existing Docker files need minor updates:
- Dockerfile: Already includes all dependencies
- docker-compose.yml: Works as-is
- New CLI args: `docker run aigis --lat 40.71 --lon -74.00`

---

## 🎯 FINAL RESULT

Once complete, the system will be a **location-agnostic, scientifically rigorous wildfire evacuation simulator** that:
- Works anywhere in the world
- Uses realistic physics (Perlin terrain, dynamic wind, Rothermel fire)
- Implements proper crowd psychology (social force herding)
- Supports research (Monte Carlo batch mode)
- Provides professional visualization (real-time dashboard)

**Ready for thesis-level research and publication.**
