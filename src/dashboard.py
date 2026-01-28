"""
Professional Dashboard with Real-Time Graphs
Uses matplotlib GridSpec for 3-panel layout
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from .config import *


class Dashboard:
    """
    Real-time visualization dashboard for AIGIS simulation.
    3-panel layout: Main map + 2 line charts
    """

    def __init__(self, simulation):
        """
        Initialize dashboard.

        Args:
            simulation: AIGISSimulation instance
        """
        self.simulation = simulation

        # Create figure with GridSpec
        self.fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        gs = gridspec.GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Main map (left side, full height)
        self.ax_map = self.fig.add_subplot(gs[:, 0])

        # Casualties chart (top-right)
        self.ax_casualties = self.fig.add_subplot(gs[0, 1])

        # Evacuations chart (bottom-right)
        self.ax_evacuations = self.fig.add_subplot(gs[1, 1])

        # Enable interactive mode
        plt.ion()
        plt.show()

        # Data history for charts
        self.history = {
            'steps': [],
            'casualties': [],
            'evacuated': []
        }

        print("📊 Dashboard initialized")

    def run(self):
        """Main dashboard loop"""
        print("▶️  Starting simulation with live dashboard...\n")

        try:
            while self.simulation.step < MAX_STEPS:
                # Step simulation
                self.simulation.run_step()

                # Update history
                self.history['steps'].append(self.simulation.step)
                self.history['casualties'].append(
                    self.simulation.count_casualties()
                )
                self.history['evacuated'].append(
                    self.simulation.count_evacuated()
                )

                # Update dashboard every N steps
                if self.simulation.step % DASHBOARD_UPDATE_INTERVAL == 0:
                    self.update()

                # Check completion
                if self.simulation.is_complete():
                    print("\n✅ Simulation complete!")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")

        finally:
            # Final update
            self.update()

            # Print final statistics
            self._print_final_stats()

            # Keep window open
            plt.ioff()
            plt.show()

    def update(self):
        """Update all plots"""
        self._update_map()
        self._update_casualties_chart()
        self._update_evacuations_chart()

        plt.draw()
        plt.pause(STEP_DELAY)

    def _update_map(self):
        """Draw main map with fire, agents, and safe zones"""
        self.ax_map.clear()

        env = self.simulation.environment
        step = self.simulation.step

        # Title with current step and wind direction
        wind_dir = self.simulation.fire_sim.get_wind_direction_degrees()
        self.ax_map.set_title(
            f"AIGIS Simulation - Step {step} | Wind: {wind_dir:.1f}°",
            fontsize=12, fontweight='bold'
        )

        # Create RGB visualization grid
        vis_grid = np.zeros((*env.grid_shape, 3), dtype=np.float32)

        # Fuel (forest green)
        vis_grid[env.fire_grid == 3] = [0.13, 0.55, 0.13]

        # Burning (orange-red)
        vis_grid[env.fire_grid == 1] = [1.0, 0.27, 0.0]

        # Burnt (dark gray)
        vis_grid[env.fire_grid == 2] = [0.18, 0.31, 0.31]

        # Obstacles (gray)
        vis_grid[env.obstacle_grid > 0] = [0.41, 0.41, 0.41]

        # Highlight safe zones (light green glow)
        self._highlight_safe_zones(vis_grid, env)

        # Display grid
        self.ax_map.imshow(vis_grid, origin='upper')

        # Plot agents
        self._plot_agents()

        # Add legend
        self._add_legend()

        # Remove axis ticks
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])

    def _highlight_safe_zones(self, vis_grid, env):
        """Highlight safe zones on the map"""
        for node in env.safe_nodes:
            if len(env.graph.nodes) == 0:
                continue

            try:
                data = env.graph.nodes[node]
                lat, lon = data['y'], data['x']
                r, c = env.latlon_to_grid(lat, lon)

                # Draw small circle around safe zone
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        dist = np.sqrt(dr**2 + dc**2)
                        if dist <= 3:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < env.grid_shape[0] and 0 <= nc < env.grid_shape[1]:
                                # Light green with distance-based intensity
                                intensity = 1.0 - (dist / 3.0) * 0.5
                                vis_grid[nr, nc] = [
                                    0.56 * intensity,
                                    0.93 * intensity,
                                    0.56 * intensity
                                ]
            except:
                continue

    def _plot_agents(self):
        """Plot all agents on the map"""
        agents = self.simulation.agents

        # Sentinels (gold circles)
        for agent in agents['sentinels']:
            if agent.grid_position:
                self.ax_map.plot(
                    agent.grid_position[1], agent.grid_position[0],
                    'o', color=COLOR_SENTINEL, markersize=8,
                    markeredgecolor='black', markeredgewidth=1
                )

        # Analyst (purple square)
        if agents['analyst'] and agents['analyst'].grid_position:
            self.ax_map.plot(
                agents['analyst'].grid_position[1],
                agents['analyst'].grid_position[0],
                's', color=COLOR_ANALYST, markersize=10,
                markeredgecolor='black', markeredgewidth=1
            )

        # Commander (red triangle)
        if agents['commander'] and agents['commander'].grid_position:
            self.ax_map.plot(
                agents['commander'].grid_position[1],
                agents['commander'].grid_position[0],
                '^', color=COLOR_COMMANDER, markersize=12,
                markeredgecolor='black', markeredgewidth=1
            )

        # Rescuers (blue diamonds)
        for agent in agents['rescuers']:
            if agent.grid_position:
                self.ax_map.plot(
                    agent.grid_position[1], agent.grid_position[0],
                    'd', color=COLOR_RESCUER, markersize=8,
                    markeredgecolor='black', markeredgewidth=1
                )

        # Civilians (colored by panic level)
        for agent in agents['civilians']:
            if agent.grid_position and agent.is_active:
                # Color from green (calm) to red (panic)
                panic_color = plt.cm.RdYlGn_r(agent.panic_level)
                self.ax_map.plot(
                    agent.grid_position[1], agent.grid_position[0],
                    'o', color=panic_color, markersize=6,
                    markeredgecolor='black', markeredgewidth=0.5
                )

    def _add_legend(self):
        """Add legend to the map"""
        legend_elements = [
            mpatches.Patch(color=COLOR_FUEL, label='Fuel (Forest)'),
            mpatches.Patch(color=COLOR_BURNING, label='Burning'),
            mpatches.Patch(color=COLOR_BURNT, label='Burnt'),
            mpatches.Patch(color=COLOR_SAFE_ZONE, label='Safe Zone'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COLOR_SENTINEL, markersize=8,
                      markeredgecolor='black', label='Sentinel'),
            plt.Line2D([0], [0], marker='^', color='w',
                      markerfacecolor=COLOR_COMMANDER, markersize=8,
                      markeredgecolor='black', label='Commander'),
            plt.Line2D([0], [0], marker='d', color='w',
                      markerfacecolor=COLOR_RESCUER, markersize=8,
                      markeredgecolor='black', label='Rescuer'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COLOR_CIVILIAN, markersize=6,
                      markeredgecolor='black', label='Civilian'),
        ]

        self.ax_map.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=8,
            framealpha=0.9
        )

    def _update_casualties_chart(self):
        """Line chart: Casualties over time"""
        self.ax_casualties.clear()
        self.ax_casualties.set_title('Casualties Over Time', fontsize=10, fontweight='bold')
        self.ax_casualties.set_xlabel('Simulation Step', fontsize=9)
        self.ax_casualties.set_ylabel('Casualties', fontsize=9)

        if len(self.history['steps']) > 0:
            self.ax_casualties.plot(
                self.history['steps'],
                self.history['casualties'],
                'r-', linewidth=2, label='Casualties'
            )
            self.ax_casualties.fill_between(
                self.history['steps'],
                self.history['casualties'],
                alpha=0.3, color='red'
            )
            self.ax_casualties.grid(True, alpha=0.3, linestyle='--')
            self.ax_casualties.legend(loc='upper left', fontsize=8)

            # Current value annotation
            if self.history['casualties']:
                current = self.history['casualties'][-1]
                self.ax_casualties.text(
                    0.98, 0.98, f'Current: {current}',
                    transform=self.ax_casualties.transAxes,
                    fontsize=10, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                )

    def _update_evacuations_chart(self):
        """Line chart: Successful evacuations over time"""
        self.ax_evacuations.clear()
        self.ax_evacuations.set_title('Successful Evacuations Over Time', fontsize=10, fontweight='bold')
        self.ax_evacuations.set_xlabel('Simulation Step', fontsize=9)
        self.ax_evacuations.set_ylabel('Evacuated', fontsize=9)

        if len(self.history['steps']) > 0:
            self.ax_evacuations.plot(
                self.history['steps'],
                self.history['evacuated'],
                'g-', linewidth=2, label='Evacuated'
            )
            self.ax_evacuations.fill_between(
                self.history['steps'],
                self.history['evacuated'],
                alpha=0.3, color='green'
            )
            self.ax_evacuations.grid(True, alpha=0.3, linestyle='--')
            self.ax_evacuations.legend(loc='upper left', fontsize=8)

            # Current value annotation
            if self.history['evacuated']:
                current = self.history['evacuated'][-1]
                self.ax_evacuations.text(
                    0.98, 0.98, f'Current: {current}',
                    transform=self.ax_evacuations.transAxes,
                    fontsize=10, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
                )

    def _print_final_stats(self):
        """Print final statistics to console"""
        results = self.simulation.get_results()

        print("\n" + "="*70)
        print("📊 FINAL STATISTICS")
        print("="*70)
        print(f"  Total Steps:              {results['steps']}")
        print(f"  Total Civilians:          {results['total_civilians']}")
        print(f"  Casualties:               {results['casualties']}")
        print(f"  Successfully Evacuated:   {results['evacuated']}")
        print(f"  Mortality Rate:           {results['mortality_rate']:.2%}")
        print(f"  Evacuation Success Rate:  {results['evacuation_success_rate']:.2%}")
        print(f"  Average Panic Level:      {results['avg_panic_level']:.2f}")
        print(f"  Max Panic Level:          {results['max_panic_level']:.2f}")
        print(f"  Rescuer Refusals:         {results['rescuer_refusals']}")
        print(f"  Max Active Fire Cells:    {results['max_fire_cells']}")
        print(f"  Final Commander Phase:    {results['final_phase']}")
        print("="*70)
