"""
AIGIS Simulation Engine with Metrics Tracking
Supports both GUI and headless batch modes
"""
import numpy as np
from typing import Dict, List, Any
from .environment import LiveMapBuilder
from .fire_simulation import FireSimulation
from .message import Message
from .agents import (
    SentinelAgent, AnalystAgent, CommanderAgent,
    RescuerAgent, FirefighterAgent, CivilianAgent,
    RiskMonitorAgent, AmbulanceAgent,
)
from .config import *


class AIGISSimulation:
    """
    Main simulation engine for AIGIS.
    Handles agent updates, message routing, and metrics tracking.
    """

    def __init__(self, lat: float, lon: float, radius: float, mode: str = 'gui',
                 run_id: int = 0, fire_locations: list = None,
                 config_overrides: dict = None):
        """
        Initialize simulation.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius: Map radius in meters
            mode: 'gui' or 'batch'
            run_id: Unique ID for Monte Carlo runs (default 0 for single runs)
            fire_locations: List of (lat, lon) tuples for real fire ignition points.
                            If None, ignites at highest-elevation fuel zones.
        """
        self.mode = mode
        self.lat, self.lon, self.radius = lat, lon, radius
        self._config_overrides = config_overrides or {}

        # Initialize random seed per-run for Monte Carlo experiments
        # Each run gets a unique seed to ensure different random sequences
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED + run_id)

        # Build environment
        print(f"\n🌍 Initializing simulation at ({lat:.4f}, {lon:.4f})...")
        builder = LiveMapBuilder(lat, lon, radius, (GRID_HEIGHT, GRID_WIDTH))
        self.environment = builder.build()

        # Load real population data if available
        self._load_population_data()

        # Load real weather data if available
        self._load_weather_data()

        # Load Fire Weather Index (FWI) data — pre-ignition risk
        self._load_fwi_data()

        # Load NASA FIRMS historical hotspot density
        self._load_firms_data()

        # Load air quality index (AQI) from OpenAQ
        self._load_air_quality_data()

        # Load EMS facilities (hospitals, ambulance stations) from OSM
        self._load_ems_data()

        # Initialize fire simulation
        self.fire_sim = FireSimulation(self.environment)

        # Ignite fires from real coordinates or highest-elevation fuel zones
        self._ignite_fires(fire_locations)

        # Initialize agents
        self.agents = self._initialize_agents()

        # Expose agents to environment for ECT calculation
        # This allows Commander to count active civilians for accurate ECT
        self.environment.agents = self.agents

        # Expose fire_simulation on environment so FirefighterAgent can read
        # wind_direction for wind-aware fire-line placement (Rothermel 1972).
        self.environment.fire_simulation = self.fire_sim

        # Give commander a reference to the fire simulation for ML feature extraction
        if self.agents['commander']:
            self.agents['commander'].fire_sim_ref = self.fire_sim

        # Metrics tracking
        self.metrics = {
            'casualties': [],
            'evacuated': [],
            'injured': [],
            'panic_levels': [],
            'active_fires': [],
            'burnt_cells': [],
            'panic_snapshots': [],
            'phase_history': [],
            'rescuer_refusals': 0
        }

        self.step = 0

    def _ignite_fires(self, fire_locations: list) -> None:
        """
        Ignite fires using real coordinates.

        If fire_locations is provided (e.g., from CLI or FIRMS), ignites there.
        Otherwise, ignites at the highest-elevation fuel cells — topographically
        realistic, as wildfires typically start on ridges and hillsides.
        """
        if fire_locations:
            print(f"  🛰️  Igniting {len(fire_locations)} fires from provided coordinates...")
            # Ensure there are fuel cells first (creates synthetic fuel in urban areas)
            if not np.any(self.environment.fire_grid == 3):
                print("  ℹ️  No natural fuel in map — marking non-obstacle cells as combustible")
                non_obstacle = (self.environment.obstacle_grid == 0)
                self.environment.fire_grid[non_obstacle] = 3
                self.environment.fuel_type_grid[non_obstacle] = 1  # NFFL 1 = short grass
            self.fire_sim.ignite_at_locations(fire_locations)
            return

        # Fallback: ignite at highest-elevation fuel zones (realistic ignition)
        print("  ⛰️  No fire coordinates provided — igniting at highest-elevation fuel zones...")
        fuel_cells = np.argwhere(self.environment.fire_grid == 3)
        if len(fuel_cells) == 0:
            # No natural fuel (e.g., fully urban area): treat all non-obstacle cells as fuel
            print("  ℹ️  No natural fuel in map — marking non-obstacle cells as combustible")
            non_obstacle = (self.environment.obstacle_grid == 0)
            self.environment.fire_grid[non_obstacle] = 3
            self.environment.fuel_type_grid[non_obstacle] = 1  # NFFL 1 = short grass
            fuel_cells = np.argwhere(self.environment.fire_grid == 3)
            if len(fuel_cells) == 0:
                print("  ⚠️  Could not create any fuel cells — skipping ignition")
                return

        elevations = self.environment.elevation_grid[fuel_cells[:, 0], fuel_cells[:, 1]]
        num_ignition = min(3, len(fuel_cells))
        top_indices = np.argpartition(elevations, -num_ignition)[-num_ignition:]
        locations = []
        for idx in top_indices:
            r, c = fuel_cells[idx]
            lat, lon = self.environment.grid_to_latlon(r, c)
            locations.append((lat, lon))
        self.fire_sim.ignite_at_locations(locations)

    def _load_population_data(self):
        """Load real population data from OSM if available"""
        try:
            from pathlib import Path
            pop_file = Path("data/population_data.npz")

            if pop_file.exists():
                # Load population data
                pop_data = np.load(pop_file)
                pop_grid = pop_data["population_grid"]

                # Resize if needed to match simulation grid
                if pop_grid.shape != self.environment.grid_shape:
                    from scipy.ndimage import zoom
                    scale_y = self.environment.grid_shape[0] / pop_grid.shape[0]
                    scale_x = self.environment.grid_shape[1] / pop_grid.shape[1]
                    pop_grid = zoom(pop_grid, (scale_y, scale_x), order=1)

                self.environment.population_density = pop_grid
                total_pop = int(pop_grid.sum())
                print(f"  ✅ Loaded real population data: {total_pop:,} people")
            else:
                print(f"  ℹ️  No population data found. Run 'python get_population_data.py' to fetch real data.")

        except Exception as e:
            print(f"  ⚠️  Failed to load population data: {e}")

    def _load_fwi_data(self):
        """Fetch Fire Weather Index components from Open-Meteo (no key required)."""
        try:
            from .data_connectors import FWIConnector
            connector = FWIConnector()
            fwi = connector.fetch(self.lat, self.lon)
            self.environment.fwi_data = fwi
            print(f"  ✅ FWI loaded: index={fwi['fwi']:.1f} ({fwi['risk_level']})")
        except Exception as e:
            print(f"  ⚠️  FWI data unavailable ({e}). Pre-ignition risk uses defaults.")

    def _load_firms_data(self):
        """Fetch NASA FIRMS ignition density. Requires FIRMS_MAP_KEY in config."""
        try:
            from .data_connectors import FIRMSConnector
            from .config import FIRMS_MAP_KEY
            connector = FIRMSConnector(map_key=FIRMS_MAP_KEY)
            radius_deg = self.radius / 111320  # metres → degrees (approx)
            density = connector.build_ignition_density(
                self.lat, self.lon,
                radius_deg=max(radius_deg * 3, 0.1),
                grid_shape=self.environment.grid_shape,
                days=7,
            )
            if density is not None:
                self.environment.firms_density = density
                hotspots = int((density > 0).sum())
                if hotspots:
                    print(f"  ✅ FIRMS: {hotspots} historical ignition cells loaded.")
        except Exception as e:
            print(f"  ⚠️  FIRMS data unavailable ({e}). Historical ignition density = 0.")

    def _load_air_quality_data(self):
        """Fetch current AQI from OpenAQ. Requires OPENAQ_API_KEY in config."""
        try:
            from .data_connectors import AirQualityConnector
            from .config import OPENAQ_API_KEY
            connector = AirQualityConnector(api_key=OPENAQ_API_KEY)
            aq = connector.fetch(self.lat, self.lon)
            self.environment.air_quality_index = float(aq.get('aqi', 0.0))
            print(f"  ✅ Air quality: AQI={aq['aqi']:.0f} ({aq['advisory']})")
        except Exception as e:
            print(f"  ⚠️  Air quality data unavailable ({e}). AQI=0 (clean air assumed).")

    def _load_ems_data(self):
        """Fetch hospital and station locations from OSM via osmnx."""
        try:
            from .data_connectors import EMSConnector
            connector = EMSConnector()
            nodes = connector.hospital_nodes(
                self.lat, self.lon, self.radius, self.environment.graph
            )
            self.environment.hospital_nodes = nodes
            if nodes:
                print(f"  ✅ EMS: {len(nodes)} hospital node(s) registered for ambulance routing.")
            else:
                print("  ℹ️  EMS: no hospitals in OSM for this area — ambulances will use safe zones.")
        except Exception as e:
            print(f"  ⚠️  EMS data unavailable ({e}). Ambulances will use safe zones.")

    def _load_weather_data(self):
        """Load real weather data if available"""
        try:
            from pathlib import Path
            import json

            weather_file = Path("data/current_weather.json")

            if weather_file.exists():
                # Load weather data
                with open(weather_file, 'r') as f:
                    weather_data = json.load(f)

                # Set environment weather parameters
                self.environment.temperature = weather_data.get('temperature', 25.0)
                self.environment.humidity = weather_data.get('humidity', 30.0)

                # Update wind speed in fire simulation if available
                wind_speed = weather_data.get('wind_speed')
                if wind_speed and hasattr(self, 'fire_sim'):
                    self.fire_sim.wind_speed = wind_speed

                wind_str = f"{wind_speed:.1f} m/s" if wind_speed is not None else "N/A"
                print(f"  ✅ Loaded real weather data: {self.environment.temperature:.1f}°C, "
                      f"{self.environment.humidity:.0f}% humidity, wind {wind_str}")
            else:
                print(f"  ℹ️  No weather data found. Run 'python fetch_real_weather.py' to fetch real data.")

        except Exception as e:
            print(f"  ⚠️  Failed to load weather data: {e}")

    def _fetch_station_positions(self) -> list:
        """
        Fetch real fire/emergency station positions from OSM.
        Returns list of (lat, lon) tuples.
        Falls back to road network nodes near the map centre if none found.
        """
        import osmnx as ox
        center_lat = self.environment.lat_center
        center_lon = self.environment.lon_center
        positions = []
        try:
            gdf = ox.features_from_point(
                (center_lat, center_lon),
                tags={'amenity': ['fire_station', 'police', 'hospital']},
                dist=self.radius
            )
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    positions.append((geom.y, geom.x))
                else:
                    c = geom.centroid
                    positions.append((c.y, c.x))
            if positions:
                print(f"  ✅ Found {len(positions)} real emergency service locations from OSM")
        except Exception:
            pass  # Fall through to road-network fallback

        if not positions:
            # Use road network nodes closest to map centre as station proxies
            print("  ℹ️  No OSM emergency stations found — using road-network positions")
            nodes = [(data['y'], data['x'])
                     for _, data in self.environment.graph.nodes(data=True)
                     if 'y' in data and 'x' in data]
            if nodes:
                nodes.sort(key=lambda p: (p[0] - center_lat)**2 + (p[1] - center_lon)**2)
                positions = nodes[:max(NUM_RESCUERS + NUM_FIREFIGHTERS, 10)]
        return positions

    def _civilian_positions(self) -> list:
        """
        Build civilian spawn positions from real road network nodes,
        weighted by population density when available.
        Returns list of (lat, lon) tuples of length NUM_CIVILIANS.
        """
        nodes = [(data['y'], data['x'])
                 for _, data in self.environment.graph.nodes(data=True)
                 if 'y' in data and 'x' in data]

        if not nodes:
            # Minimal fallback graph — place on the centre node
            lat, lon = self.environment.lat_center, self.environment.lon_center
            return [(lat, lon)] * NUM_CIVILIANS

        lats = np.array([p[0] for p in nodes])
        lons = np.array([p[1] for p in nodes])

        # Compute per-node population weight from density grid
        weights = np.ones(len(nodes), dtype=np.float64)
        pop = self.environment.population_density
        if pop is not None and pop.max() > 0:
            for idx, (lat, lon) in enumerate(zip(lats, lons)):
                r, c = self.environment.latlon_to_grid(lat, lon)
                weights[idx] = max(pop[r, c], 1e-6)

        weights /= weights.sum()
        chosen = np.random.choice(len(nodes), size=NUM_CIVILIANS, replace=True, p=weights)
        return [(lats[i], lons[i]) for i in chosen]

    def _initialize_agents(self) -> Dict[str, Any]:
        """Create all 6 agent types using real geographic data for positioning."""
        agents = {
            'sentinels': [],
            'analyst': None,
            'commander': None,
            'rescuers': [],
            'firefighters': [],
            'civilians': [],
            'risk_monitors': [],
            'ambulances': [],
        }

        min_lon, min_lat, max_lon, max_lat = self.environment.bounds
        center_lat = self.environment.lat_center
        center_lon = self.environment.lon_center

        # --- Sentinels: four corners of the map for maximum coverage of edge-ignition zones ---
        print(f"  🔭 Creating {NUM_SENTINELS} Sentinel agents...")
        corner_offsets = [
            ( 0.85,  0.85),   # NE corner
            (-0.85,  0.85),   # SE corner
            (-0.85, -0.85),   # SW corner
            ( 0.85, -0.85),   # NW corner
        ]
        for i in range(NUM_SENTINELS):
            dlat, dlon = corner_offsets[i % len(corner_offsets)]
            lat = center_lat + dlat * (max_lat - min_lat) * 0.5
            lon = center_lon + dlon * (max_lon - min_lon) * 0.5
            sentinel = SentinelAgent(f"sentinel_{i}", (lat, lon))
            sentinel.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['sentinels'].append(sentinel)

        # --- Analyst + Commander: map centre ---
        print("  🧠 Creating Analyst agent...")
        agents['analyst'] = AnalystAgent("analyst", (center_lat, center_lon))
        agents['analyst'].grid_position = self.environment.latlon_to_grid(center_lat, center_lon)

        print("  ⚔️  Creating Commander agent...")
        agents['commander'] = CommanderAgent("commander", (center_lat, center_lon))
        agents['commander'].grid_position = self.environment.latlon_to_grid(center_lat, center_lon)
        # Apply learned parameter overrides
        overrides = self._config_overrides
        if 'phase_monitor_mult'  in overrides: agents['commander'].phase_monitor_mult  = overrides['phase_monitor_mult']
        if 'phase_prealert_mult' in overrides: agents['commander'].phase_prealert_mult = overrides['phase_prealert_mult']
        if 'phase_evacuate_mult' in overrides: agents['commander'].phase_evacuate_mult = overrides['phase_evacuate_mult']

        # --- Rescuers + Firefighters: real OSM emergency stations ---
        station_positions = self._fetch_station_positions()
        total_emergency = NUM_RESCUERS + NUM_FIREFIGHTERS

        # Cycle through stations if fewer stations than agents needed
        def station_at(idx):
            return station_positions[idx % len(station_positions)]

        print(f"  🚑 Creating {NUM_RESCUERS} Rescuer agents...")
        for i in range(NUM_RESCUERS):
            lat, lon = station_at(i)
            rescuer = RescuerAgent(f"rescuer_{i}", (lat, lon))
            rescuer.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['rescuers'].append(rescuer)

        print(f"  🚒 Creating {NUM_FIREFIGHTERS} Firefighter agents...")
        for i in range(NUM_FIREFIGHTERS):
            lat, lon = station_at(NUM_RESCUERS + i)
            firefighter = FirefighterAgent(f"firefighter_{i}", (lat, lon))
            firefighter.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['firefighters'].append(firefighter)

        # --- Civilians: road network nodes weighted by population density ---
        print(f"  🏃 Creating {NUM_CIVILIANS} Civilian agents...")
        civ_positions = self._civilian_positions()
        for i, (lat, lon) in enumerate(civ_positions):
            civilian = CivilianAgent(f"civilian_{i}", (lat, lon))
            civilian.grid_position = self.environment.latlon_to_grid(lat, lon)
            if 'panic_rational' in overrides: civilian.panic_rational_threshold = overrides['panic_rational']
            if 'panic_confused' in overrides: civilian.panic_confused_threshold = overrides['panic_confused']
            agents['civilians'].append(civilian)

        # --- RiskMonitor: one instance, placed at map centre ---
        print(f"  🔥 Creating {NUM_RISK_MONITORS} RiskMonitor agent(s)...")
        for i in range(NUM_RISK_MONITORS):
            rm = RiskMonitorAgent(f"risk_monitor_{i}", (center_lat, center_lon))
            rm.grid_position = self.environment.latlon_to_grid(center_lat, center_lon)
            agents['risk_monitors'].append(rm)

        # --- Ambulances: spawned at hospital/station positions (or road nodes) ---
        print(f"  🚑 Creating {NUM_AMBULANCES} Ambulance agent(s)...")
        for i in range(NUM_AMBULANCES):
            lat, lon = station_at(NUM_RESCUERS + NUM_FIREFIGHTERS + i)
            amb = AmbulanceAgent(f"ambulance_{i}", (lat, lon))
            amb.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['ambulances'].append(amb)

        print("✅ All agents initialized!\n")
        return agents

    def run_step(self):
        """Execute one simulation step"""
        # 1. Fire spread (with dynamic wind)
        self.fire_sim.step()

        # Reset claimed fire cells so each step firefighters pick fresh targets
        self.environment.claimed_fire_cells = set()

        # 2. Update all agents (perceive -> decide -> act)
        self._update_agents()

        # 3. Route messages between agents
        self._route_messages()

        # 4. Update nearby agents for civilians (Social Force herding)
        self._update_civilian_neighbors()

        # 5. Collect metrics
        self._collect_metrics()

        self.step += 1
        self.environment.step_count = self.step

    def _update_agents(self):
        """Update all agents (perceive-decide-act cycle)"""
        # Update sentinels
        for sentinel in self.agents['sentinels']:
            sentinel.update(self.environment)

        # Update analyst
        if self.agents['analyst']:
            self.agents['analyst'].update(self.environment)

        # Update commander
        if self.agents['commander']:
            self.agents['commander'].update(self.environment)

        # Update rescuers
        for rescuer in self.agents['rescuers']:
            rescuer.update(self.environment)

        # Update firefighters
        for firefighter in self.agents['firefighters']:
            firefighter.update(self.environment)

        # Update civilians
        for civilian in self.agents['civilians']:
            civilian.update(self.environment)

        # Update risk monitors (pre-ignition assessment)
        for rm in self.agents['risk_monitors']:
            rm.update(self.environment)

        # Update ambulances
        for amb in self.agents['ambulances']:
            amb.update(self.environment)

        # Clear inboxes after processing
        for agent_group in self.agents.values():
            if isinstance(agent_group, list):
                for agent in agent_group:
                    agent.clear_messages()
            elif agent_group:
                agent_group.clear_messages()

    def _route_messages(self):
        """Route messages between agents"""
        all_messages = []

        # Collect all outgoing messages
        for agent_type in self.agents.values():
            if isinstance(agent_type, list):
                for agent in agent_type:
                    all_messages.extend(agent.get_outbox_messages())
            elif agent_type is not None:
                all_messages.extend(agent_type.get_outbox_messages())

        # Route messages to recipients
        for message in all_messages:
            receiver = message.receiver

            if receiver == "analyst":
                if self.agents['analyst']:
                    self.agents['analyst'].receive_message(message)

            elif receiver == "commander":
                if self.agents['commander']:
                    self.agents['commander'].receive_message(message)

            elif receiver == "ambulances":
                for amb in self.agents['ambulances']:
                    amb.receive_message(message)

            elif receiver == "firefighters":
                for ff in self.agents['firefighters']:
                    ff.receive_message(message)

            elif receiver == "rescuers" or receiver == "broadcast":
                # Broadcast to all rescuers
                for rescuer in self.agents['rescuers']:
                    rescuer.receive_message(message)

                # If broadcast, also send to civilians
                if receiver == "broadcast":
                    for civilian in self.agents['civilians']:
                        civilian.receive_message(message)

            else:
                # Direct message to specific agent
                for agent_type in self.agents.values():
                    if isinstance(agent_type, list):
                        for agent in agent_type:
                            if agent.agent_id == receiver:
                                agent.receive_message(message)
                    elif agent_type and agent_type.agent_id == receiver:
                        agent_type.receive_message(message)

            # Track rescuer refusals
            if message.performative == "REFUSE":
                self.metrics['rescuer_refusals'] += 1

    def _update_civilian_neighbors(self):
        """
        Update nearby agents for each civilian (for Social Force herding).
        This must be done after all agents have moved.
        """
        civilians = self.agents['civilians']

        for civilian in civilians:
            if civilian.is_active:
                civilian._find_nearby_agents(civilians)

    def _collect_metrics(self):
        """Track metrics for analysis"""
        # Count casualties (civilians caught in fire)
        casualties = self.count_casualties()
        self.metrics['casualties'].append(casualties)

        # Count evacuated (civilians that reached safe zones)
        evacuated = self.count_evacuated()
        self.metrics['evacuated'].append(evacuated)

        # Average panic level
        if self.agents['civilians']:
            avg_panic = np.mean([c.panic_level for c in self.agents['civilians']])
        else:
            avg_panic = 0.0
        self.metrics['panic_levels'].append(avg_panic)

        # Active fire cells + burnt cells
        fire_stats = self.fire_sim.get_fire_statistics()
        self.metrics['active_fires'].append(fire_stats['burning_cells'])
        self.metrics['burnt_cells'].append(fire_stats['burnt_cells'])

        # Per-step panic snapshot
        if self.agents['civilians']:
            self.metrics['panic_snapshots'].append(
                [c.panic_level for c in self.agents['civilians'] if c.is_active]
            )
        else:
            self.metrics['panic_snapshots'].append([])

        # Injured civilians (smoke inhalation)
        injured = sum(1 for c in self.agents['civilians'] if c.is_injured)
        self.metrics['injured'].append(injured)

        # Commander phase
        if self.agents['commander']:
            self.metrics['phase_history'].append(self.agents['commander'].current_phase)

    def count_casualties(self) -> int:
        """Count civilians caught in fire"""
        count = 0
        for civilian in self.agents['civilians']:
            if not civilian.is_active:
                if not civilian.is_evacuated:
                    count += 1
                continue

            if civilian.grid_position:
                r, c = civilian.grid_position
                # Check if in burning or burnt area
                if self.environment.fire_grid[r, c] in [1, 2]:
                    civilian.is_active = False
                    count += 1

        return count

    def count_evacuated(self) -> int:
        """Count civilians that reached safe zones"""
        count = 0
        h, w = self.environment.grid_shape
        for civilian in self.agents['civilians']:
            if civilian.is_evacuated:
                count += 1
                continue
            if not civilian.is_active:
                continue

            # Road-network safe node check
            if civilian.current_node is not None:
                if self.environment.is_safe_node(civilian.current_node):
                    civilian.is_evacuated = True
                    civilian.is_active = False
                    count += 1
                    continue

            # Grid-perimeter check: civilians using the perimeter fallback
            # (_move_toward_perimeter) are evacuated when they reach any map edge.
            if civilian.grid_position is not None:
                r, c = civilian.grid_position
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    civilian.is_evacuated = True
                    civilian.is_active = False
                    count += 1

        return count

    def is_complete(self) -> bool:
        """Check if simulation is finished"""
        fire_stats = self.fire_sim.get_fire_statistics()

        # Complete if fire burned out — but only after at least one step has run
        # (prevents immediate exit when fuel_cells==0 in urban environments)
        if self.step > 0 and fire_stats['burning_cells'] == 0 and fire_stats['fuel_cells'] == 0:
            return True

        # Complete if all civilians are either evacuated or casualties
        self.count_evacuated()  # mark newly evacuated civilians as inactive
        active_civilians = sum(1 for c in self.agents['civilians'] if c.is_active)

        if active_civilians == 0:
            return True

        return False

    def run_until_complete(self, max_steps: int = MAX_STEPS) -> Dict[str, Any]:
        """
        Run simulation until completion.
        Returns final metrics.
        """
        while self.step < max_steps and not self.is_complete():
            self.run_step()
            self._print_step_summary()

        self._print_final_report()
        return self.get_results()

    def _print_step_summary(self):
        """Print a one-line status for the current step."""
        fire_stats = self.fire_sim.get_fire_statistics()
        evacuated  = self.count_evacuated()
        total      = len(self.agents['civilians'])
        casualties = self.count_casualties()
        active     = sum(1 for c in self.agents['civilians'] if c.is_active)
        aqi        = getattr(self.environment, 'air_quality_index', 0.0)
        phase      = (self.agents['commander'].current_phase
                      if self.agents['commander'] else 0)

        print(f"  Step {self.step:>4} | "
              f"Fire: {fire_stats['burning_cells']:>3} burning / "
              f"{fire_stats['burnt_cells']:>4} burnt | "
              f"Evac: {evacuated}/{total} | "
              f"Cas: {casualties} | "
              f"Active: {active} | "
              f"Phase: {phase} | "
              f"AQI: {aqi:.0f}")

    def _print_final_report(self) -> None:
        """
        Post-simulation summary report.

        Metrics grounded in:
          Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood
          evacuations in the urban-wildland interface." Environment and
          Planning A, 34(12), pp. 2211-2230.
            -> Evacuation clearance time, % evacuated, bottleneck analysis.
          Wolshon, B. (2006). "Evacuation planning and engineering for
          Hurricane Katrina." The Bridge, 36(1), pp. 27-34. NAE.
            -> Evacuation success rate as primary performance indicator.
        """
        total   = len(self.agents['civilians'])
        evac    = self.count_evacuated()
        cas     = self.count_casualties()
        injured = sum(1 for c in self.agents['civilians'] if c.is_injured)
        active  = sum(1 for c in self.agents['civilians'] if c.is_active)
        fire_stats = self.fire_sim.get_fire_statistics()
        phase = self.agents['commander'].current_phase if self.agents['commander'] else 0

        evac_rate = evac / total * 100 if total > 0 else 0
        mort_rate = cas  / total * 100 if total > 0 else 0

        ff_drops = sum(
            1 for ff in self.agents['firefighters']
            if ff.current_water < ff.water_capacity
        )

        print("\n" + "=" * 60)
        print("AIGIS SIMULATION REPORT")
        print("=" * 60)
        print(f"  Steps run           : {self.step}")
        print(f"  Final phase         : {phase}")
        print(f"  Civilians total     : {total}")
        print(f"  Evacuated           : {evac}  ({evac_rate:.1f}%)")
        print(f"  Casualties          : {cas}   ({mort_rate:.1f}%)")
        print(f"  Smoke-injured       : {injured}")
        print(f"  Still active        : {active}")
        print(f"  Burnt cells         : {fire_stats['burnt_cells']}")
        print(f"  Remaining burning   : {fire_stats['burning_cells']}")
        print(f"  Firefighter units   : {len(self.agents['firefighters'])} "
              f"({ff_drops} used water)")
        print(f"  Rescuer refusals    : {self.metrics['rescuer_refusals']}")
        if self.metrics['panic_levels']:
            avg_p = np.mean(self.metrics['panic_levels'])
            max_p = np.max(self.metrics['panic_levels'])
            print(f"  Avg panic           : {avg_p:.2f}")
            print(f"  Peak panic          : {max_p:.2f}")
        print("=" * 60 + "\n")

    def get_results(self) -> Dict[str, Any]:
        """Return final metrics dictionary"""
        total_civilians = len(self.agents['civilians'])
        casualties = self.count_casualties()
        evacuated = self.count_evacuated()

        reconsideration_log = []
        if self.agents['commander']:
            reconsideration_log = self.agents['commander'].reconsideration_log

        injured = sum(1 for c in self.agents['civilians'] if c.is_injured)
        # Burned area metrics
        fire_grid = self.environment.fire_grid
        total_cells = fire_grid.size
        burned_cells = int(np.sum(fire_grid == 2))   # state 2 = burnt out
        cell_side_m = (2.0 * self.radius) / fire_grid.shape[1]
        cell_area_ha = (cell_side_m ** 2) / 10_000
        burned_area_ha = burned_cells * cell_area_ha
        burned_area_pct = (burned_cells / total_cells * 100) if total_cells > 0 else 0.0

        return {
            'steps': self.step,
            'steps_to_evacuate': self.step,
            'total_civilians': total_civilians,
            'casualties': casualties,
            'evacuated': evacuated,
            'injured': injured,
            'mortality_rate': casualties / total_civilians if total_civilians > 0 else 0,
            'evacuation_success_rate': evacuated / total_civilians if total_civilians > 0 else 0,
            'burned_area_pct': burned_area_pct,
            'burned_area_ha': round(burned_area_ha, 1),
            'avg_panic_level': np.mean(self.metrics['panic_levels']) if self.metrics['panic_levels'] else 0,
            'max_panic_level': np.max(self.metrics['panic_levels']) if self.metrics['panic_levels'] else 0,
            'rescuer_refusals': self.metrics['rescuer_refusals'],
            'total_burning_cells': sum(self.metrics['active_fires']),
            'max_fire_cells': max(self.metrics['active_fires']) if self.metrics['active_fires'] else 0,
            'final_phase': self.agents['commander'].current_phase if self.agents['commander'] else 0,
            'reconsideration_log': reconsideration_log,
            'history': {
                'casualties': self.metrics['casualties'],
                'evacuated': self.metrics['evacuated'],
                'injured': self.metrics['injured'],
                'panic_levels': self.metrics['panic_levels'],
                'active_fires': self.metrics['active_fires'],
                'burnt_cells': self.metrics['burnt_cells'],
                'panic_snapshots': self.metrics['panic_snapshots'],
                'phase_history': self.metrics['phase_history'],
            }
        }
