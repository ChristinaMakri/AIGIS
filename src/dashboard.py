"""
AIGIS Live Dashboard — 7-panel real-time simulation visualisation.

Panel layout (GridSpec 3x3, fire grid spans rows 0-1 col 0):
  [1: Fire Grid + Agents + Smoke] [2: Evacuation Timeline ] [3: Panic Histogram    ]
  [          (continued)        ] [4: Fire Spread Metrics ] [5: AQI + Smoke Injured ]
  [  6: Commander Phase Strip (spans cols 0-1)            ] [7: FF Resources + CNP  ]

Post-batch figure (separate, 2x3):
  [8: Outcome violins] [9: Panic violins] [10: Mortality vs Fire scatter]
  [11: Duration hist ] [12: Phase bar   ] [13: CNP refusals histogram   ]

References:
  Wolshon 2006      — evacuation clearance time (panel 2)
  Cova & Johnson 2002 — panic thresholds (panel 3)
  Rothermel 1972    — fire spread metrics (panel 4)
  Inness et al. 2019 — AQI / smoke model (panel 5)
  Rao & Georgeff 1995 — BDI commander phases (panel 6)
  Smith 1980        — Contract Net Protocol refusals (panel 7)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import AIGISSimulation

from .config import (
    DASHBOARD_UPDATE_INTERVAL,
    DASHBOARD_HISTORY_LENGTH,
    CIVILIAN_PANIC_RATIONAL,
    CIVILIAN_PANIC_CONFUSED,
)

_PHASE_NAMES = {0: 'Monitor', 1: 'Pre-Alert', 2: 'Evacuate', 3: 'Shelter-in-Place'}
_PHASE_COLS  = {0: '#3a86ff', 1: '#ffbe0b', 2: '#ff006e', 3: '#8338ec'}

_FIRE_CMAP = mcolors.ListedColormap(['#2c7bb6', '#fdae61', '#d7191c'])
_FIRE_NORM = mcolors.BoundaryNorm([0, 1, 2, 3], _FIRE_CMAP.N)

_AQI_BANDS = [
    (0,   50,  '#00e400', 'Good'),
    (51,  100, '#ffff00', 'Moderate'),
    (101, 150, '#ff7e00', 'Unhealthy (Sensitive)'),
    (151, 200, '#ff0000', 'Unhealthy'),
    (201, 300, '#8f3f97', 'Very Unhealthy'),
    (301, 999, '#7e0023', 'Hazardous'),
]

_BG_DARK  = '#1a1a2e'
_BG_PANEL = '#16213e'
_FG_TEXT  = '#e0e0e0'
_FG_DIM   = '#888888'
_SPINE    = '#3a3a5c'


def _aqi_color_label(aqi: float):
    for lo, hi, col, label in _AQI_BANDS:
        if aqi <= hi:
            return col, label
    return '#7e0023', 'Hazardous'


def _style_ax(ax):
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_FG_TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)


class Dashboard:
    """
    Headless matplotlib dashboard for AIGIS simulation.

    Uses the Agg (PNG) backend — renders to file.  Call update() each step
    to refresh the figure in memory, then save() to write PNG.

    Usage — standard single-run mode:
        dash = Dashboard()
        while not sim.is_complete():
            sim.run_step()
            if sim.step % DASHBOARD_UPDATE_INTERVAL == 0:
                dash.update(sim)
        dash.finalize(sim)
        dash.save("aigis_dashboard.png")
        dash.close()

    Usage — batch mode (post-run summary):
        Dashboard.batch_summary(df, "aigis_batch_summary.png")
    """

    def __init__(self):
        self._fig = plt.figure(figsize=(18, 11))
        self._fig.patch.set_facecolor(_BG_DARK)

        gs = gridspec.GridSpec(
            3, 3,
            figure=self._fig,
            height_ratios=[5, 4, 2],
            width_ratios=[4, 3, 3],
            hspace=0.42,
            wspace=0.32,
        )
        self._ax_fire    = self._fig.add_subplot(gs[0:2, 0])
        self._ax_evac    = self._fig.add_subplot(gs[0,   1])
        self._ax_panic   = self._fig.add_subplot(gs[0,   2])
        self._ax_fire_ts = self._fig.add_subplot(gs[1,   1])
        self._ax_aqi     = self._fig.add_subplot(gs[1,   2])
        self._ax_phase   = self._fig.add_subplot(gs[2,   0:2])
        self._ax_res     = self._fig.add_subplot(gs[2,   2])

        for ax in self._fig.axes:
            _style_ax(ax)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, sim: 'AIGISSimulation') -> None:
        """Refresh all panels from current simulation state."""
        self._draw_fire_grid(sim)
        self._draw_evac_timeline(sim)
        self._draw_panic_histogram(sim)
        self._draw_fire_metrics(sim)
        self._draw_aqi_panel(sim)
        self._draw_phase_strip(sim)
        self._draw_resources(sim)

        phase = sim.agents['commander'].current_phase if sim.agents['commander'] else 0
        self._fig.suptitle(
            f"AIGIS Dashboard  |  Step {sim.step}  |  "
            f"({sim.lat:.4f}, {sim.lon:.4f})  |  "
            f"Phase: {_PHASE_NAMES.get(phase, '?')}",
            color=_FG_TEXT, fontsize=11, fontweight='bold', y=0.99,
        )

    def finalize(self, sim: 'AIGISSimulation') -> None:
        """Final update after simulation ends."""
        self.update(sim)

    def save(self, path: str = 'aigis_dashboard.png') -> str:
        """Save current figure to PNG. Returns saved path."""
        self._fig.savefig(path, dpi=150, bbox_inches='tight',
                          facecolor=self._fig.get_facecolor())
        print(f"  Dashboard saved to: {path}")
        return path

    def close(self) -> None:
        plt.close(self._fig)

    # ------------------------------------------------------------------
    # Panel 1 — Fire grid + agents + smoke overlay
    # ------------------------------------------------------------------

    def _draw_fire_grid(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_fire
        ax.cla()
        _style_ax(ax)
        env = sim.environment

        # Base fire state
        ax.imshow(env.fire_grid, cmap=_FIRE_CMAP, norm=_FIRE_NORM,
                  origin='upper', interpolation='nearest')

        # Smoke concentration overlay (semi-transparent)
        if env.smoke_grid is not None and env.smoke_grid.max() > 0:
            ax.imshow(env.smoke_grid / env.smoke_grid.max(), cmap='YlOrBr',
                      vmin=0, vmax=1, origin='upper', interpolation='bilinear',
                      alpha=0.35)

        # Civilians coloured by panic level (green=calm, red=panic)
        civ_xs, civ_ys, civ_p = [], [], []
        for c in sim.agents['civilians']:
            if c.is_active and c.grid_position:
                r, col_ = c.grid_position
                civ_xs.append(col_)
                civ_ys.append(r)
                civ_p.append(c.panic_level)
        if civ_xs:
            ax.scatter(civ_xs, civ_ys, c=civ_p, cmap='RdYlGn_r', vmin=0, vmax=1,
                       s=10, marker='o', alpha=0.65, edgecolors='none', zorder=4)

        def _scatter(agents, marker, color, size, label):
            xs, ys = [], []
            for ag in agents:
                if ag.grid_position:
                    r, c_ = ag.grid_position
                    xs.append(c_); ys.append(r)
            if xs:
                ax.scatter(xs, ys, marker=marker, c=color, s=size, zorder=5,
                           edgecolors='white', linewidths=0.3, label=label)

        _scatter(sim.agents['firefighters'], 's', '#1E90FF', 40, 'Firefighters')
        _scatter(sim.agents['rescuers'],     '^', '#00ff88', 40, 'Rescuers')
        _scatter(sim.agents['sentinels'],    '*', '#FFD700', 60, 'Sentinels')
        if sim.agents['ambulances']:
            _scatter(sim.agents['ambulances'], 'D', '#ff69b4', 30, 'Ambulances')

        ax.legend(handles=[
            Patch(fc='#2c7bb6', label='Unburned'),
            Patch(fc='#fdae61', label='Burning'),
            Patch(fc='#d7191c', label='Burnt'),
        ], loc='lower right', fontsize=6, framealpha=0.55,
           facecolor=_BG_DARK, labelcolor=_FG_TEXT)

        fwi_data = getattr(env, 'fwi_data', {}) or {}
        fwi_val  = fwi_data.get('fwi', 0.0)
        fwi_risk = fwi_data.get('risk_level', 'N/A')
        aqi      = getattr(env, 'air_quality_index', 0.0)
        wind_spd = getattr(sim.fire_sim, 'wind_speed', 0.0)
        ax.text(
            0.01, 0.02,
            f"FWI {fwi_val:.1f} ({fwi_risk})   AQI {aqi:.0f}   Wind {wind_spd:.1f} m/s",
            transform=ax.transAxes, fontsize=7, color=_FG_TEXT, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc=_BG_DARK, alpha=0.75),
        )

        fs = sim.fire_sim.get_fire_statistics()
        ax.set_title(
            f"Fire Grid + Agents  |  Burning: {fs['burning_cells']}  "
            f"Burnt: {fs['burnt_cells']}",
            color=_FG_TEXT, fontsize=9,
        )
        ax.axis('off')

    # ------------------------------------------------------------------
    # Panel 2 — Evacuation & casualty timeline
    # ------------------------------------------------------------------

    def _draw_evac_timeline(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_evac
        ax.cla()
        _style_ax(ax)
        m = sim.metrics
        n = len(m['evacuated'])
        if n == 0:
            ax.set_title('Evacuation & Casualties  (Wolshon 2006)', color=_FG_TEXT, fontsize=8)
            return

        s   = list(range(max(0, n - DASHBOARD_HISTORY_LENGTH), n))
        ev  = m['evacuated'][-DASHBOARD_HISTORY_LENGTH:]
        cas = m['casualties'][-DASHBOARD_HISTORY_LENGTH:]
        inj = m['injured'][-DASHBOARD_HISTORY_LENGTH:]

        ax.plot(s, ev,  color='#00ff88', lw=1.5, label='Evacuated')
        ax.plot(s, cas, color='#ff4444', lw=1.5, linestyle='--', label='Casualties')
        ax.plot(s, inj, color='#ffaa00', lw=1.2, linestyle=':',  label='Smoke Injured')

        total = len(sim.agents['civilians'])
        ax.axhline(total, color=_FG_DIM, lw=0.7, linestyle=':')
        ax.text(s[0], total + max(total * 0.01, 0.5),
                f'Total ({total})', color=_FG_DIM, fontsize=5)

        ax.set_xlabel('Step', color=_FG_TEXT, fontsize=7)
        ax.set_ylabel('Civilians', color=_FG_TEXT, fontsize=7)
        ax.set_title('Evacuation & Casualties  (Wolshon 2006)', color=_FG_TEXT, fontsize=8)
        ax.legend(fontsize=5, facecolor=_BG_DARK, labelcolor=_FG_TEXT,
                  framealpha=0.6, loc='upper left')
        ax.tick_params(colors=_FG_TEXT, labelsize=6)

    # ------------------------------------------------------------------
    # Panel 3 — Panic distribution histogram
    # ------------------------------------------------------------------

    def _draw_panic_histogram(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_panic
        ax.cla()
        _style_ax(ax)
        snaps = sim.metrics['panic_snapshots']
        title = 'Panic Distribution  (Cova & Johnson 2002)'
        if not snaps or not snaps[-1]:
            ax.set_title(title, color=_FG_TEXT, fontsize=8)
            return

        p    = np.array(snaps[-1])
        bins = np.linspace(0, 1, 21)
        r    = p[p < CIVILIAN_PANIC_RATIONAL]
        cf   = p[(p >= CIVILIAN_PANIC_RATIONAL) & (p < CIVILIAN_PANIC_CONFUSED)]
        h    = p[p >= CIVILIAN_PANIC_CONFUSED]

        for subset, col, lbl in [
            (r,  '#00cc44', f'Rational (<{CIVILIAN_PANIC_RATIONAL})'),
            (cf, '#ffbb00', f'Confused ({CIVILIAN_PANIC_RATIONAL}-{CIVILIAN_PANIC_CONFUSED})'),
            (h,  '#ff3333', f'Herding (>{CIVILIAN_PANIC_CONFUSED})'),
        ]:
            if len(subset):
                ax.hist(subset, bins=bins, color=col, alpha=0.75, label=lbl)

        ax.axvline(CIVILIAN_PANIC_RATIONAL, color='#ffbb00', lw=1, linestyle='--')
        ax.axvline(CIVILIAN_PANIC_CONFUSED, color='#ff3333', lw=1, linestyle='--')

        avg = float(np.mean(p))
        ax.axvline(avg, color='white', lw=0.8, linestyle=':')
        ylim = ax.get_ylim()
        ax.text(min(avg + 0.02, 0.95), ylim[1] * 0.85, f'avg {avg:.2f}',
                color='white', fontsize=5)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Panic Level', color=_FG_TEXT, fontsize=7)
        ax.set_ylabel('# Civilians', color=_FG_TEXT, fontsize=7)
        ax.set_title(f'{title}  n={len(p)}', color=_FG_TEXT, fontsize=8)
        ax.legend(fontsize=5, facecolor=_BG_DARK, labelcolor=_FG_TEXT, framealpha=0.6)
        ax.tick_params(colors=_FG_TEXT, labelsize=6)

    # ------------------------------------------------------------------
    # Panel 4 — Fire spread metrics (active / burnt cells over time)
    # ------------------------------------------------------------------

    def _draw_fire_metrics(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_fire_ts
        ax.cla()
        _style_ax(ax)
        m = sim.metrics
        n = len(m['active_fires'])
        if n == 0:
            ax.set_title('Fire Spread  (Rothermel 1972)', color=_FG_TEXT, fontsize=8)
            return

        s  = list(range(max(0, n - DASHBOARD_HISTORY_LENGTH), n))
        af = m['active_fires'][-DASHBOARD_HISTORY_LENGTH:]
        bc = m['burnt_cells'][-DASHBOARD_HISTORY_LENGTH:]

        ax.fill_between(s, af, color='#ff7700', alpha=0.35)
        ax.plot(s, af, color='#ff7700', lw=1.5, label='Burning')
        ax.fill_between(s, bc, color='#8b0000', alpha=0.25)
        ax.plot(s, bc, color='#cc2200', lw=1.5, label='Burnt-out')

        ax.set_xlabel('Step', color=_FG_TEXT, fontsize=7)
        ax.set_ylabel('Grid Cells', color=_FG_TEXT, fontsize=7)
        ax.set_title('Fire Spread  (Rothermel 1972)', color=_FG_TEXT, fontsize=8)
        ax.legend(fontsize=6, facecolor=_BG_DARK, labelcolor=_FG_TEXT, framealpha=0.6)
        ax.tick_params(colors=_FG_TEXT, labelsize=6)

    # ------------------------------------------------------------------
    # Panel 5 — AQI gauge + smoke-injured trend
    # ------------------------------------------------------------------

    def _draw_aqi_panel(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_aqi
        ax.cla()
        _style_ax(ax)
        aqi = getattr(sim.environment, 'air_quality_index', 0.0)
        aqi_col, aqi_label = _aqi_color_label(aqi)

        # Smoke-injured trend as main chart
        m = sim.metrics
        n = len(m['injured'])
        if n:
            s   = list(range(max(0, n - DASHBOARD_HISTORY_LENGTH), n))
            inj = m['injured'][-DASHBOARD_HISTORY_LENGTH:]
            ax.fill_between(s, inj, color='#ff9500', alpha=0.4)
            ax.plot(s, inj, color='#ff9500', lw=1.2)

        ax.set_xlabel('Step', color=_FG_TEXT, fontsize=7)
        ax.set_ylabel('Smoke Injured', color=_FG_TEXT, fontsize=7)
        ax.tick_params(colors=_FG_TEXT, labelsize=6)

        # AQI value overlaid in top-right corner
        ax.text(
            0.97, 0.97,
            f"AQI {aqi:.0f}\n{aqi_label}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontweight='bold', color=aqi_col,
            bbox=dict(boxstyle='round,pad=0.3', fc=_BG_DARK, alpha=0.80),
        )
        ax.set_title('Smoke & Air Quality  (Inness et al. 2019)',
                     color=_FG_TEXT, fontsize=8)

    # ------------------------------------------------------------------
    # Panel 6 — Commander phase colour strip
    # ------------------------------------------------------------------

    def _draw_phase_strip(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_phase
        ax.cla()
        _style_ax(ax)
        phases = sim.metrics['phase_history']
        title  = 'Commander Phase  (Rao & Georgeff 1995 BDI)'
        if not phases:
            ax.set_title(title, color=_FG_TEXT, fontsize=8)
            return

        start    = max(0, len(phases) - DASHBOARD_HISTORY_LENGTH)
        phases_t = phases[-DASHBOARD_HISTORY_LENGTH:]
        steps_t  = list(range(start, start + len(phases_t)))

        prev = None
        for step, phase in zip(steps_t, phases_t):
            ax.barh(0, 1, left=step, height=1,
                    color=_PHASE_COLS.get(phase, '#666666'), alpha=0.85, edgecolor='none')
            if phase != prev:
                ax.text(step + 0.2, 0.5, _PHASE_NAMES.get(phase, str(phase)),
                        ha='left', va='center', fontsize=6.5, color='white',
                        fontweight='bold', transform=ax.get_xaxis_transform())
                prev = phase

        ax.set_xlim(steps_t[0], steps_t[-1] + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_xlabel('Step', color=_FG_TEXT, fontsize=7)
        ax.tick_params(axis='x', colors=_FG_TEXT, labelsize=6)

        ax.legend(
            handles=[Patch(fc=_PHASE_COLS[i], label=_PHASE_NAMES[i]) for i in range(4)],
            loc='upper right', fontsize=5, facecolor=_BG_DARK, labelcolor=_FG_TEXT,
            framealpha=0.6, ncol=4,
        )
        current = phases[-1]
        ax.set_title(
            f"{title}  |  Current: {_PHASE_NAMES.get(current, '?')}",
            color=_FG_TEXT, fontsize=8,
        )

    # ------------------------------------------------------------------
    # Panel 7 — Firefighter water resources + CNP refusals
    # ------------------------------------------------------------------

    def _draw_resources(self, sim: 'AIGISSimulation') -> None:
        ax = self._ax_res
        ax.cla()
        _style_ax(ax)
        ffs = sim.agents['firefighters']

        if ffs:
            for i, ff in enumerate(ffs):
                level = (ff.current_water / ff.water_capacity
                         if ff.water_capacity > 0 else 0.0)
                col = '#1E90FF' if level > 0.3 else '#ff4444'
                ax.barh(i, level, color=col, alpha=0.8, height=0.6)
                ax.text(min(level + 0.03, 1.05), i, f'{level:.0%}',
                        va='center', ha='left', color='white', fontsize=6)
            ax.set_xlim(0, 1.25)
            ax.set_ylim(-0.5, len(ffs) - 0.5)
            ax.set_yticks(range(len(ffs)))
            ax.set_yticklabels(
                [ff.agent_id.replace('firefighter_', 'FF-') for ff in ffs],
                fontsize=6, color=_FG_TEXT,
            )
            ax.set_xlabel('Water Level', color=_FG_TEXT, fontsize=7)
            ax.tick_params(axis='x', colors=_FG_TEXT, labelsize=6)

        refusals = sim.metrics['rescuer_refusals']
        ax.text(
            0.97, 0.04,
            f"CNP Refusals: {refusals}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            color='#ff9500',
            bbox=dict(boxstyle='round,pad=0.3', fc=_BG_DARK, alpha=0.75),
        )
        ax.set_title('FF Resources  (Smith 1980 CNP)', color=_FG_TEXT, fontsize=8)

    # ------------------------------------------------------------------
    # Batch / Monte Carlo summary (standalone figure, 6 panels)
    # ------------------------------------------------------------------

    @staticmethod
    def batch_summary(df, out_path: str = 'aigis_batch_summary.png') -> str:
        """
        Render Monte Carlo summary charts from a results DataFrame.

        Panels:
          P1: Violin — mortality & evacuation rates
          P2: Violin — avg/max panic levels
          P3: Scatter — mortality vs max fire extent (colour = avg panic)
          P4: Histogram — simulation duration distribution
          P5: Bar — final commander phase distribution
          P6: Histogram — CNP rescuer refusals distribution

        Args:
            df: pandas DataFrame from batch mode (18 CSV columns)
            out_path: output PNG path
        Returns:
            Saved file path
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor(_BG_DARK)
        fig.suptitle('AIGIS Monte Carlo Summary', fontsize=13, fontweight='bold',
                     color=_FG_TEXT, y=0.98)

        for ax in axes.flat:
            ax.set_facecolor(_BG_PANEL)
            ax.tick_params(colors=_FG_TEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(_SPINE)

        # P1: Violin — mortality & evacuation rates
        ax = axes[0, 0]
        data_v = [
            df['mortality_rate'].dropna().values * 100,
            df['evacuation_success_rate'].dropna().values * 100,
        ]
        parts = ax.violinplot(data_v, positions=[1, 2], showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], ['#ff4444', '#00cc44']):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        parts['cmedians'].set_color('white')
        for k in ('cmins', 'cmaxes', 'cbars'):
            parts[k].set_color(_FG_DIM)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Mortality %', 'Evacuation %'], color=_FG_TEXT, fontsize=8)
        ax.set_ylabel('Rate (%)', color=_FG_TEXT, fontsize=8)
        ax.set_title('Outcome Rates', color=_FG_TEXT, fontsize=9)

        # P2: Violin — panic levels
        ax = axes[0, 1]
        data_p = [
            df['avg_panic_level'].dropna().values,
            df['max_panic_level'].dropna().values,
        ]
        parts2 = ax.violinplot(data_p, positions=[1, 2], showmedians=True, showextrema=True)
        for pc, col in zip(parts2['bodies'], ['#ffbb00', '#ff5500']):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        parts2['cmedians'].set_color('white')
        for k in ('cmins', 'cmaxes', 'cbars'):
            parts2[k].set_color(_FG_DIM)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Avg Panic', 'Max Panic'], color=_FG_TEXT, fontsize=8)
        ax.set_ylabel('Panic Level [0-1]', color=_FG_TEXT, fontsize=8)
        ax.set_title('Panic Levels  (Cova & Johnson 2002)', color=_FG_TEXT, fontsize=9)

        # P3: Scatter — mortality rate vs max fire cells, colour = avg panic
        ax = axes[0, 2]
        if 'max_fire_cells' in df.columns:
            sc = ax.scatter(
                df['max_fire_cells'], df['mortality_rate'] * 100,
                c=df['avg_panic_level'], cmap='YlOrRd',
                s=35, alpha=0.75, edgecolors='#444', lw=0.3,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Avg Panic', color=_FG_TEXT, fontsize=7)
            cbar.ax.yaxis.set_tick_params(color=_FG_TEXT, labelsize=6)
            ax.set_xlabel('Max Fire Cells', color=_FG_TEXT, fontsize=8)
            ax.set_ylabel('Mortality Rate (%)', color=_FG_TEXT, fontsize=8)
            ax.set_title('Mortality vs Fire Extent', color=_FG_TEXT, fontsize=9)

        # P4: Histogram — simulation duration
        ax = axes[1, 0]
        ax.hist(df['steps'].dropna(), bins=20, color='#3a86ff', alpha=0.75,
                edgecolor='#2255aa')
        mean_s = df['steps'].mean()
        ax.axvline(mean_s, color='white', lw=1, linestyle='--')
        ylim4 = ax.get_ylim()
        ax.text(mean_s + 0.5, ylim4[1] * 0.9, f'mean={mean_s:.0f}',
                color='white', fontsize=7)
        ax.set_xlabel('Steps', color=_FG_TEXT, fontsize=8)
        ax.set_ylabel('Count', color=_FG_TEXT, fontsize=8)
        ax.set_title('Simulation Duration', color=_FG_TEXT, fontsize=9)

        # P5: Bar — final commander phase distribution
        ax = axes[1, 1]
        if 'final_phase' in df.columns:
            counts = df['final_phase'].value_counts().sort_index()
            ax.bar(
                [_PHASE_NAMES.get(int(p), str(p)) for p in counts.index],
                counts.values,
                color=[_PHASE_COLS.get(int(p), '#666') for p in counts.index],
                alpha=0.8, edgecolor='#555',
            )
            ax.set_xlabel('Phase', color=_FG_TEXT, fontsize=8)
            ax.set_ylabel('Run Count', color=_FG_TEXT, fontsize=8)
            ax.set_title('Final Phase Distribution', color=_FG_TEXT, fontsize=9)
            ax.tick_params(axis='x', rotation=20, labelsize=7, colors=_FG_TEXT)

        # P6: Histogram — CNP rescuer refusals
        ax = axes[1, 2]
        if 'rescuer_refusals' in df.columns:
            ax.hist(df['rescuer_refusals'].dropna(), bins=15,
                    color='#ff9500', alpha=0.75, edgecolor='#cc6600')
            mean_r = df['rescuer_refusals'].mean()
            ax.axvline(mean_r, color='white', lw=1, linestyle='--')
            ylim6 = ax.get_ylim()
            ax.text(mean_r + 0.2, ylim6[1] * 0.9, f'mean={mean_r:.1f}',
                    color='white', fontsize=7)
            ax.set_xlabel('Rescuer Refusals', color=_FG_TEXT, fontsize=8)
            ax.set_ylabel('Count', color=_FG_TEXT, fontsize=8)
            ax.set_title('CNP Refusals  (Smith 1980)', color=_FG_TEXT, fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Batch summary saved to: {out_path}")
        return out_path
