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
    RescuerAgent, CivilianAgent
)
from .config import *


class AIGISSimulation:
    """
    Main simulation engine for AIGIS.
    Handles agent updates, message routing, and metrics tracking.
    """

    def __init__(self, lat: float, lon: float, radius: float, mode: str = 'gui', run_id: int = 0):
        """
        Initialize simulation.

        Args:
            lat: Center latitude
            lon: Center longitude
            radius: Map radius in meters
            mode: 'gui' or 'batch'
            run_id: Unique ID for Monte Carlo runs (default 0 for single runs)
        """
        self.mode = mode
        self.lat, self.lon, self.radius = lat, lon, radius

        # Initialize random seed per-run for Monte Carlo experiments
        # Each run gets a unique seed to ensure different random sequences
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED + run_id)

        # Build environment
        print(f"\n🌍 Initializing simulation at ({lat:.4f}, {lon:.4f})...")
        builder = LiveMapBuilder(lat, lon, radius, (GRID_HEIGHT, GRID_WIDTH))
        self.environment = builder.build()

        # Initialize fire simulation
        self.fire_sim = FireSimulation(self.environment)

        # Initialize agents
        self.agents = self._initialize_agents()

        # Expose agents to environment for ECT calculation
        # This allows Commander to count active civilians for accurate ECT
        self.environment.agents = self.agents

        # Metrics tracking
        self.metrics = {
            'casualties': [],
            'evacuated': [],
            'panic_levels': [],
            'active_fires': [],
            'phase_history': [],
            'rescuer_refusals': 0
        }

        self.step = 0

    def _initialize_agents(self) -> Dict[str, Any]:
        """Create all 5 agent types"""
        agents = {
            'sentinels': [],
            'analyst': None,
            'commander': None,
            'rescuers': [],
            'civilians': []
        }

        # Get bounds for positioning
        min_lon, min_lat, max_lon, max_lat = self.environment.bounds
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Sentinel agents - positioned around the perimeter
        print(f"  🔭 Creating {NUM_SENTINELS} Sentinel agents...")
        for i in range(NUM_SENTINELS):
            angle = (2 * np.pi * i) / NUM_SENTINELS
            offset = 0.4
            lat = center_lat + offset * (max_lat - min_lat) * np.sin(angle)
            lon = center_lon + offset * (max_lon - min_lon) * np.cos(angle)

            sentinel = SentinelAgent(f"sentinel_{i}", (lat, lon))
            sentinel.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['sentinels'].append(sentinel)

        # Analyst agent - central position
        print("  🧠 Creating Analyst agent...")
        agents['analyst'] = AnalystAgent("analyst", (center_lat, center_lon))
        agents['analyst'].grid_position = self.environment.latlon_to_grid(center_lat, center_lon)

        # Commander agent - central position
        print("  ⚔️  Creating Commander agent...")
        agents['commander'] = CommanderAgent("commander", (center_lat, center_lon))
        agents['commander'].grid_position = self.environment.latlon_to_grid(center_lat, center_lon)

        # Rescuer agents - positioned near center
        print(f"  🚑 Creating {NUM_RESCUERS} Rescuer agents...")
        for i in range(NUM_RESCUERS):
            lat = center_lat + np.random.uniform(-0.3, 0.3) * (max_lat - min_lat)
            lon = center_lon + np.random.uniform(-0.3, 0.3) * (max_lon - min_lon)

            rescuer = RescuerAgent(f"rescuer_{i}", (lat, lon))
            rescuer.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['rescuers'].append(rescuer)

        # Civilian agents - random positions
        print(f"  🏃 Creating {NUM_CIVILIANS} Civilian agents...")
        for i in range(NUM_CIVILIANS):
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)

            civilian = CivilianAgent(f"civilian_{i}", (lat, lon))
            civilian.grid_position = self.environment.latlon_to_grid(lat, lon)
            agents['civilians'].append(civilian)

        print("✅ All agents initialized!\n")
        return agents

    def run_step(self):
        """Execute one simulation step"""
        # 1. Fire spread (with dynamic wind)
        self.fire_sim.step()

        # 2. Update all agents (perceive -> decide -> act)
        self._update_agents()

        # 3. Route messages between agents
        self._route_messages()

        # 4. Update nearby agents for civilians (Social Force herding)
        self._update_civilian_neighbors()

        # 5. Collect metrics
        self._collect_metrics()

        self.step += 1

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

        # Update civilians
        for civilian in self.agents['civilians']:
            civilian.update(self.environment)

        # Clear inboxes after processing
        for agent_type in self.agents.values():
            if isinstance(agent_type, list):
                for agent in agent_type:
                    agent.clear_messages()
            elif agent_type:
                agent_type.clear_messages()

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
            if civilian.cognitive_state == "herding":
                # Find nearby civilians for herding
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

        # Active fire cells
        fire_stats = self.fire_sim.get_fire_statistics()
        self.metrics['active_fires'].append(fire_stats['burning_cells'])

        # Commander phase
        if self.agents['commander']:
            self.metrics['phase_history'].append(self.agents['commander'].current_phase)

    def count_casualties(self) -> int:
        """Count civilians caught in fire"""
        count = 0
        for civilian in self.agents['civilians']:
            if not civilian.is_active:
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
        for civilian in self.agents['civilians']:
            if not civilian.is_active:
                continue

            if civilian.current_node is not None:
                # Check if at a safe node
                if self.environment.is_safe_node(civilian.current_node):
                    count += 1

        return count

    def is_complete(self) -> bool:
        """Check if simulation is finished"""
        fire_stats = self.fire_sim.get_fire_statistics()

        # Complete if fire burned out
        if fire_stats['burning_cells'] == 0 and fire_stats['fuel_cells'] == 0:
            return True

        # Complete if all civilians evacuated or casualties
        active_civilians = sum(1 for c in self.agents['civilians'] if c.is_active)
        evacuated = self.count_evacuated()

        if active_civilians == evacuated or active_civilians == 0:
            return True

        return False

    def run_until_complete(self, max_steps: int = MAX_STEPS) -> Dict[str, Any]:
        """
        Run simulation until completion.
        Returns final metrics.
        """
        while self.step < max_steps and not self.is_complete():
            self.run_step()

            # Progress update for batch mode
            if self.mode == 'batch' and self.step % 50 == 0:
                print(f"    Step {self.step}/{max_steps}")

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Return final metrics dictionary"""
        total_civilians = len(self.agents['civilians'])
        casualties = self.count_casualties()
        evacuated = self.count_evacuated()

        return {
            'steps': self.step,
            'steps_to_evacuate': self.step,
            'total_civilians': total_civilians,
            'casualties': casualties,
            'evacuated': evacuated,
            'mortality_rate': casualties / total_civilians if total_civilians > 0 else 0,
            'evacuation_success_rate': evacuated / total_civilians if total_civilians > 0 else 0,
            'avg_panic_level': np.mean(self.metrics['panic_levels']) if self.metrics['panic_levels'] else 0,
            'max_panic_level': np.max(self.metrics['panic_levels']) if self.metrics['panic_levels'] else 0,
            'rescuer_refusals': self.metrics['rescuer_refusals'],
            'total_burning_cells': sum(self.metrics['active_fires']),
            'max_fire_cells': max(self.metrics['active_fires']) if self.metrics['active_fires'] else 0,
            'final_phase': self.agents['commander'].current_phase if self.agents['commander'] else 0
        }
