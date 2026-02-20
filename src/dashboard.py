"""
Professional Dashboard with Real-Time Graphs — 3×3 GridSpec Layout

Layout:
  [  MAP (full height, col 0)  ] | [ Casualties+Evacuations dual-axis (row 0, cols 1-2) ]
  [                             ] | [ Fire spread+containment% (row 1, col 1) ] [ Panic hist (row 1, col 2) ]
  [                             ] | [ Phase timeline coloured bands (row 2, cols 1-2)   ]
"""
import collections
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from .config import (
    DPI, MAX_STEPS, STEP_DELAY,
    DASHBOARD_UPDATE_INTERVAL, DASHBOARD_HISTORY_LENGTH,
    COLOR_FUEL, COLOR_BURNING, COLOR_BURNT, COLOR_SAFE_ZONE,
    COLOR_SENTINEL, COLOR_ANALYST, COLOR_COMMANDER,
    COLOR_RESCUER, COLOR_CIVILIAN
)

# Phase colour palette
_PHASE_COLORS = {0: '#2196F3', 1: '#FF9800', 2: '#F44336', 3: '#9C27B0'}
_PHASE_NAMES = {0: 'Monitor', 1: 'Pre-Alert', 2: 'Evacuate', 3: 'Shelter'}


class Dashboard:
    """
    Real-time visualization dashboard for AIGIS.
    3×3 GridSpec layout.
    """

    def __init__(self, simulation):
        self.simulation = simulation

        # ── Figure / axes ──────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(20, 11), dpi=DPI)
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.45, wspace=0.35,
                               left=0.04, right=0.97, top=0.95, bottom=0.06)

        # Map occupies full left column
        self.ax_map = self.fig.add_subplot(gs[:, 0])

        # Right-side panels
        self.ax_cas_evac = self.fig.add_subplot(gs[0, 1:])   # row 0, cols 1-2
        self.ax_fire     = self.fig.add_subplot(gs[1, 1])    # row 1, col 1
        self.ax_panic    = self.fig.add_subplot(gs[1, 2])    # row 1, col 2
        self.ax_phase    = self.fig.add_subplot(gs[2, 1:])   # row 2, cols 1-2

        plt.ion()
        plt.show()

        # ── Agent trail deques (maxlen=15) ─────────────────────────────────
        self._trails = {}

        # ── History ────────────────────────────────────────────────────────
        self.history = {
            'steps': [],
            'casualties': [],
            'evacuated': [],
            'burning_cells': [],
            'burnt_cells': [],
            'phase': [],
            'panic_levels': [],       # mean per step
            'panic_snapshots': [],    # list of per-step panic arrays (for histogram)
        }

        print("Dashboard initialized (3×3 GridSpec)")

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        print("Starting simulation with live dashboard...\n")
        try:
            while self.simulation.step < MAX_STEPS:
                self.simulation.run_step()
                self._record_history()

                if self.simulation.step % DASHBOARD_UPDATE_INTERVAL == 0:
                    self.update()

                if self.simulation.is_complete():
                    print("\nSimulation complete!")
                    break
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            self.update()
            self._print_final_stats()
            plt.ioff()
            plt.show()

    def _record_history(self):
        step = self.simulation.step
        self.history['steps'].append(step)
        self.history['casualties'].append(self.simulation.count_casualties())
        self.history['evacuated'].append(self.simulation.count_evacuated())

        fire_stats = self.simulation.fire_sim.get_fire_statistics()
        self.history['burning_cells'].append(fire_stats['burning_cells'])
        self.history['burnt_cells'].append(fire_stats['burnt_cells'])

        commander = self.simulation.agents.get('commander')
        self.history['phase'].append(commander.current_phase if commander else 0)

        civilians = self.simulation.agents.get('civilians', [])
        panic_vals = [c.panic_level for c in civilians if c.is_active]
        self.history['panic_levels'].append(np.mean(panic_vals) if panic_vals else 0.0)
        self.history['panic_snapshots'].append(panic_vals)

        # Trim to history length
        limit = DASHBOARD_HISTORY_LENGTH
        for k in self.history:
            if isinstance(self.history[k], list) and len(self.history[k]) > limit:
                self.history[k] = self.history[k][-limit:]

        # Update agent trails
        agents = self.simulation.agents
        for agent_list_key in ('sentinels', 'rescuers'):
            for agent in agents.get(agent_list_key, []):
                aid = agent.agent_id
                if aid not in self._trails:
                    self._trails[aid] = collections.deque(maxlen=15)
                if agent.grid_position:
                    self._trails[aid].append(agent.grid_position)

        for key in ('analyst', 'commander'):
            agent = agents.get(key)
            if agent and agent.grid_position:
                aid = agent.agent_id
                if aid not in self._trails:
                    self._trails[aid] = collections.deque(maxlen=15)
                self._trails[aid].append(agent.grid_position)

    # ── Full update ────────────────────────────────────────────────────────

    def update(self):
        self._update_map()
        self._update_casualties_evacuations_chart()
        self._update_fire_spread_chart()
        self._update_panic_histogram()
        self._update_phase_timeline()
        plt.draw()
        plt.pause(STEP_DELAY)

    # ── Map panel ──────────────────────────────────────────────────────────

    def _update_map(self):
        self.ax_map.clear()
        env = self.simulation.environment
        step = self.simulation.step

        # Wind direction
        wind_dir = self.simulation.fire_sim.get_wind_direction_degrees()
        commander = self.simulation.agents.get('commander')
        phase = commander.current_phase if commander else 0

        self.ax_map.set_title(
            f"AIGIS  Step {step}  |  Wind {wind_dir:.1f}°",
            fontsize=11, fontweight='bold'
        )

        # Base RGB grid
        vis_grid = np.zeros((*env.grid_shape, 3), dtype=np.float32)
        vis_grid[env.fire_grid == 3] = [0.13, 0.55, 0.13]
        vis_grid[env.fire_grid == 1] = [1.0,  0.27, 0.0]
        vis_grid[env.fire_grid == 2] = [0.18, 0.31, 0.31]
        vis_grid[env.obstacle_grid > 0] = [0.41, 0.41, 0.41]
        self._highlight_safe_zones(vis_grid, env)
        self.ax_map.imshow(vis_grid, origin='upper', zorder=1)

        # Fire intensity (temperature) overlay
        if hasattr(env, 'temperature_grid') and env.temperature_grid is not None:
            self.ax_map.imshow(
                env.temperature_grid, cmap='hot', alpha=0.35,
                vmin=0, vmax=100, origin='upper', zorder=2
            )

        # Agent trails
        self._draw_agent_trails()

        # Agents
        self._plot_agents()

        # Phase banner at top of map
        phase_color = _PHASE_COLORS.get(phase, '#2196F3')
        self.ax_map.axhspan(-5, 8, facecolor=phase_color, alpha=0.85, zorder=5)
        self.ax_map.text(
            env.grid_shape[1] / 2, 2,
            f"Phase {phase}: {_PHASE_NAMES.get(phase, '')}",
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=6
        )

        # Wind arrow (bottom-right of map)
        wind_rad = np.radians(wind_dir)
        arrow_len = 15
        ox, oy = env.grid_shape[1] - 20, env.grid_shape[0] - 20
        dx = arrow_len * np.sin(wind_rad)
        dy = -arrow_len * np.cos(wind_rad)
        self.ax_map.annotate(
            '', xy=(ox + dx, oy + dy), xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='cyan', lw=2),
            zorder=7
        )

        self._add_legend()
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        self.ax_map.set_xlim(-2, env.grid_shape[1] + 2)
        self.ax_map.set_ylim(env.grid_shape[0] + 2, -10)

    def _highlight_safe_zones(self, vis_grid, env):
        for node in env.safe_nodes:
            if len(env.graph.nodes) == 0:
                continue
            try:
                data = env.graph.nodes[node]
                r, c = env.latlon_to_grid(data['y'], data['x'])
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        dist = np.sqrt(dr**2 + dc**2)
                        if dist <= 3:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < env.grid_shape[0] and 0 <= nc < env.grid_shape[1]:
                                intensity = 1.0 - (dist / 3.0) * 0.5
                                vis_grid[nr, nc] = [0.56 * intensity, 0.93 * intensity, 0.56 * intensity]
            except Exception:
                continue

    def _draw_agent_trails(self):
        for aid, trail in self._trails.items():
            pts = list(trail)
            n = len(pts)
            if n < 2:
                continue
            for i in range(1, n):
                alpha = (i / n) * 0.6
                self.ax_map.plot(
                    [pts[i-1][1], pts[i][1]],
                    [pts[i-1][0], pts[i][0]],
                    '-', color='white', alpha=alpha, linewidth=1, zorder=3
                )

    def _plot_agents(self):
        agents = self.simulation.agents

        for agent in agents.get('sentinels', []):
            if agent.grid_position:
                self.ax_map.plot(agent.grid_position[1], agent.grid_position[0],
                                 'o', color=COLOR_SENTINEL, markersize=8,
                                 markeredgecolor='black', markeredgewidth=1, zorder=4)

        analyst = agents.get('analyst')
        if analyst and analyst.grid_position:
            self.ax_map.plot(analyst.grid_position[1], analyst.grid_position[0],
                             's', color=COLOR_ANALYST, markersize=10,
                             markeredgecolor='black', markeredgewidth=1, zorder=4)

        commander = agents.get('commander')
        if commander and commander.grid_position:
            self.ax_map.plot(commander.grid_position[1], commander.grid_position[0],
                             '^', color=COLOR_COMMANDER, markersize=12,
                             markeredgecolor='black', markeredgewidth=1, zorder=4)

        for agent in agents.get('rescuers', []):
            if agent.grid_position:
                self.ax_map.plot(agent.grid_position[1], agent.grid_position[0],
                                 'd', color=COLOR_RESCUER, markersize=8,
                                 markeredgecolor='black', markeredgewidth=1, zorder=4)

        for agent in agents.get('civilians', []):
            if agent.grid_position and agent.is_active:
                panic_color = plt.cm.RdYlGn_r(agent.panic_level)
                self.ax_map.plot(agent.grid_position[1], agent.grid_position[0],
                                 'o', color=panic_color, markersize=6,
                                 markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    def _add_legend(self):
        legend_elements = [
            mpatches.Patch(color=COLOR_FUEL,    label='Fuel'),
            mpatches.Patch(color=COLOR_BURNING, label='Burning'),
            mpatches.Patch(color=COLOR_BURNT,   label='Burnt'),
            mpatches.Patch(color=COLOR_SAFE_ZONE, label='Safe Zone'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SENTINEL,
                       markersize=7, markeredgecolor='black', label='Sentinel'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_COMMANDER,
                       markersize=7, markeredgecolor='black', label='Commander'),
            plt.Line2D([0], [0], marker='d', color='w', markerfacecolor=COLOR_RESCUER,
                       markersize=7, markeredgecolor='black', label='Rescuer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CIVILIAN,
                       markersize=6, markeredgecolor='black', label='Civilian'),
        ]
        self.ax_map.legend(handles=legend_elements, loc='lower left',
                           fontsize=7, framealpha=0.9)

    # ── Right-side panels ──────────────────────────────────────────────────

    def _update_casualties_evacuations_chart(self):
        ax = self.ax_cas_evac
        ax.clear()
        ax.set_title('Casualties & Evacuations', fontsize=10, fontweight='bold')

        steps = self.history['steps']
        if not steps:
            return

        # Left axis: casualties
        line_cas = ax.plot(steps, self.history['casualties'],
                           'r-', linewidth=2, label='Casualties')[0]
        ax.fill_between(steps, self.history['casualties'], alpha=0.2, color='red')
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Casualties', color='red', fontsize=9)
        ax.tick_params(axis='y', labelcolor='red')

        # Right axis: evacuated
        ax2 = ax.twinx()
        line_evac = ax2.plot(steps, self.history['evacuated'],
                             'g-', linewidth=2, label='Evacuated')[0]
        ax2.fill_between(steps, self.history['evacuated'], alpha=0.15, color='green')
        ax2.set_ylabel('Evacuated', color='green', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='green')

        # Combined legend
        ax.legend(handles=[line_cas, line_evac], loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _update_fire_spread_chart(self):
        ax = self.ax_fire
        ax.clear()
        ax.set_title('Fire Spread', fontsize=10, fontweight='bold')

        steps = self.history['steps']
        if not steps:
            return

        # Burning cells
        ax.fill_between(steps, self.history['burning_cells'],
                         alpha=0.5, color='orangered', label='Burning')
        ax.plot(steps, self.history['burning_cells'], color='orangered', linewidth=1.5)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Cells', color='orangered', fontsize=9)
        ax.tick_params(axis='y', labelcolor='orangered')

        # Containment % on right axis
        env = self.simulation.environment
        total_fuel = int(np.sum(env.fire_grid >= 1)) + int(np.sum(env.fire_grid == 3))
        if total_fuel > 0:
            containment = [100.0 * (1 - b / total_fuel) for b in self.history['burning_cells']]
            ax2 = ax.twinx()
            ax2.plot(steps, containment, 'b--', linewidth=1.2, label='Containment %')
            ax2.set_ylabel('Containment %', color='blue', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylim(0, 105)

        ax.grid(True, alpha=0.3)

    def _update_panic_histogram(self):
        ax = self.ax_panic
        ax.clear()
        ax.set_title('Panic Distribution', fontsize=10, fontweight='bold')

        # Use latest panic snapshot
        snapshots = self.history['panic_snapshots']
        if snapshots:
            latest = snapshots[-1]
            if latest:
                ax.hist(latest, bins=10, range=(0, 1), color='purple',
                         alpha=0.7, edgecolor='black')
                ax.axvline(0.4, color='orange', linestyle='--', linewidth=1.5,
                            label='Confused (0.4)')
                ax.axvline(0.7, color='red', linestyle='--', linewidth=1.5,
                            label='Herding (0.7)')
                ax.legend(fontsize=7)

        ax.set_xlabel('Panic Level', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    def _update_phase_timeline(self):
        ax = self.ax_phase
        ax.clear()
        ax.set_title('Phase Timeline', fontsize=10, fontweight='bold')

        phase_hist = self.history['phase']
        if not phase_hist:
            return

        for i, ph in enumerate(phase_hist):
            ax.axvspan(i, i + 1, facecolor=_PHASE_COLORS.get(ph, '#2196F3'), alpha=0.7)

        legend_patches = [
            mpatches.Patch(color=_PHASE_COLORS[p], label=_PHASE_NAMES[p], alpha=0.8)
            for p in sorted(_PHASE_COLORS)
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)
        ax.set_xlim(0, max(len(phase_hist), 1))
        ax.set_xlabel('Step', fontsize=9)
        ax.set_yticks([])
        ax.grid(False)

    # ── Final stats ────────────────────────────────────────────────────────

    def _print_final_stats(self):
        results = self.simulation.get_results()
        print("\n" + "="*70)
        print("FINAL STATISTICS")
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
        recon = results.get('reconsideration_log', [])
        print(f"  Reconsideration Events:   {len(recon)}")
        print("="*70)
