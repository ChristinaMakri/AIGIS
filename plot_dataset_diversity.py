"""
AIGIS Dataset Diversity Chart
==============================
Generates a single publication-quality figure showing the geographic,
temporal, and parametric diversity of the 15 training and 9 held-out
(OOD) scenarios used in the AIGIS thesis experiments.

Panels
------
1. World map   — geographic coverage with curriculum phase colours
2. Parameter space — wind speed vs Rothermel ROS bubble chart
3. Key parameter distributions — box plots for wind speed, ROS, civilians
4. Dataset summary — temporal range, continental coverage

Usage
-----
  python plot_dataset_diversity.py [--output FILE]

Output
------
  dataset_diversity.png  (or --output path)
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Dataset definition — all 24 scenarios
# ---------------------------------------------------------------------------

TRAINING = [
    # ── Phase 1: Easy ────────────────────────────────────────────────────────
    dict(name='Bages, Catalonia',      year=2021, phase=1, continent='Europe',
         lat=41.698,  lon=1.802,    wind=12.0, ros=0.60, civ=40),
    dict(name='Var, France',           year=2021, phase=1, continent='Europe',
         lat=43.352,  lon=6.198,    wind=14.0, ros=0.72, civ=45),
    dict(name='Penteli, Athens',       year=2022, phase=1, continent='Europe',
         lat=38.056,  lon=23.868,   wind=12.0, ros=0.55, civ=50),
    # ── Phase 2: Medium ──────────────────────────────────────────────────────
    dict(name='Manavgat, Turkey',      year=2021, phase=2, continent='Asia',
         lat=36.786,  lon=31.437,   wind=10.0, ros=0.85, civ=55),
    dict(name='Rhodes, Greece',        year=2023, phase=2, continent='Europe',
         lat=36.198,  lon=28.002,   wind=13.0, ros=0.70, civ=45),
    dict(name='Kineta, Corinth',       year=2018, phase=2, continent='Europe',
         lat=38.008,  lon=23.140,   wind=17.0, ros=0.85, civ=80),
    dict(name='Varibobi, Athens',      year=2021, phase=2, continent='Europe',
         lat=38.128,  lon=23.798,   wind=15.0, ros=0.90, civ=70),
    dict(name='Dadia, Evros',          year=2022, phase=2, continent='Europe',
         lat=41.300,  lon=26.200,   wind=12.0, ros=0.82, civ=50),
    # ── Phase 3: Hard ────────────────────────────────────────────────────────
    dict(name='Fort McMurray, AB',     year=2016, phase=3, continent='N. America',
         lat=56.726,  lon=-111.379, wind=20.0, ros=1.10, civ=60),
    dict(name='Gospers Mtn, NSW',      year=2019, phase=3, continent='Australia',
         lat=-33.250, lon=150.400,  wind=17.0, ros=1.05, civ=55),
    dict(name='Carr Fire, CA',         year=2018, phase=3, continent='N. America',
         lat=40.588,  lon=-122.392, wind=18.0, ros=1.00, civ=65),
    dict(name='Glass Fire, CA',        year=2020, phase=3, continent='N. America',
         lat=38.498,  lon=-122.402, wind=25.0, ros=1.20, civ=75),
    dict(name='Woolsey Fire, CA',      year=2018, phase=3, continent='N. America',
         lat=34.172,  lon=-118.872, wind=28.0, ros=1.15, civ=90),
    dict(name='Thomas Fire, CA',       year=2017, phase=3, continent='N. America',
         lat=34.354,  lon=-119.065, wind=22.0, ros=1.12, civ=70),
    dict(name='Evia Fire, Greece',     year=2021, phase=3, continent='Europe',
         lat=38.953,  lon=23.150,   wind=15.0, ros=0.95, civ=65),
]

HELD_OUT = [
    dict(name='Mati 2018',             year=2018, continent='Europe',
         lat=38.090,  lon=23.920,   wind=11.0, ros=0.70, civ=60),
    dict(name='Camp Fire 2018',        year=2018, continent='N. America',
         lat=39.759,  lon=-121.622, wind=16.0, ros=0.85, civ=60),
    dict(name='Pedrogao 2017',         year=2017, continent='Europe',
         lat=39.947,  lon=-8.148,   wind=22.0, ros=0.95, civ=60),
    dict(name='Alexandroupoli 2023',   year=2023, continent='Europe',
         lat=41.049,  lon=26.357,   wind=16.0, ros=1.05, civ=60),
    dict(name='Lahaina 2023',          year=2023, continent='N. America',
         lat=20.888,  lon=-156.673, wind=27.0, ros=1.25, civ=60),
    dict(name='Black Saturday 2009',   year=2009, continent='Australia',
         lat=-37.515, lon=145.365,  wind=18.0, ros=1.20, civ=60),
    dict(name='Tubbs Fire 2017',       year=2017, continent='N. America',
         lat=38.479,  lon=-122.728, wind=25.0, ros=1.18, civ=60),
    dict(name='Peloponnese 2007',      year=2007, continent='Europe',
         lat=37.489,  lon=21.648,   wind=14.0, ros=0.98, civ=60),
    dict(name='Valparaiso 2014',       year=2014, continent='S. America',
         lat=-33.047, lon=-71.613,  wind=12.0, ros=0.90, civ=60),
]

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
BG    = '#1a1a2e'
PANEL = '#16213e'
FG    = '#e0e0e0'
GRID  = '#2a2a4e'

PHASE_COLOURS = {1: '#4cc9f0', 2: '#f77f00', 3: '#e63946'}
HELD_COLOUR   = '#f4a261'
TRAIN_MARKER  = 'o'
HELD_MARKER   = 'D'

CONTINENT_COLOURS = {
    'Europe':    '#4895ef',
    'N. America':'#f4a261',
    'Australia': '#06d6a0',
    'Asia':      '#e63946',
    'S. America':'#ffd166',
}

# ---------------------------------------------------------------------------
# Minimal world coastline (very coarse — avoids external dependencies)
# Lat/lon polylines for the major continental outlines, good enough for
# context at small scale.  Each entry is a list of (lon, lat) tuples.
# ---------------------------------------------------------------------------
def _world_outlines():
    """Return list of (lon_arr, lat_arr) for approximate continental outlines."""
    # Coarse outline data encoded as run-length compressed lat/lon pairs.
    # These were hand-digitised from standard world outline datasets.
    outlines = []

    # Europe (simplified)
    eu = [(-9,36),(-9,43),(-2,43),(-2,46),(2,46),(2,51),(8,51),(8,55),
          (12,55),(12,58),(18,58),(18,65),(28,65),(28,70),(30,70),(30,73),
          (32,73),(32,68),(38,68),(38,63),(28,63),(28,58),(24,58),(24,55),
          (20,55),(20,51),(16,48),(14,44),(16,40),(20,37),(28,36),(36,36),
          (36,33),(28,33),(18,33),(12,36),(6,36),(-2,36),(-9,36)]
    eu = np.array(eu)
    outlines.append((eu[:,0], eu[:,1]))

    # Iberian peninsula (more detail)
    ib = [(-9,36),(-9,44),(-2,44),(-2,43),(0,43),(3,42),(3,40),(0,38),
          (-1,37),(-5,36),(-7,37),(-9,38),(-9,36)]
    ib = np.array(ib)
    outlines.append((ib[:,0], ib[:,1]))

    # Africa (simplified)
    af = [(-17,14),(-17,20),(-13,20),(-13,25),(-5,25),(-5,32),(4,32),
          (12,32),(24,32),(36,22),(40,12),(44,11),(44,5),(40,-5),(34,-10),
          (30,-18),(28,-30),(20,-36),(18,-34),(14,-22),(10,-5),(6,4),(2,6),
          (-2,5),(-8,5),(-14,10),(-17,14)]
    af = np.array(af)
    outlines.append((af[:,0], af[:,1]))

    # North America (simplified)
    na = [(-168,71),(-140,71),(-140,60),(-130,60),(-125,50),(-124,38),
          (-120,30),(-110,23),(-88,16),(-83,10),(-77,8),(-72,10),(-70,19),
          (-65,18),(-60,15),(-60,9),(-70,9),(-76,8),(-80,8),(-82,10),
          (-84,9),(-90,16),(-100,20),(-109,23),(-117,28),(-120,34),
          (-122,37),(-124,48),(-123,49),(-110,49),(-100,49),(-90,49),
          (-82,45),(-76,44),(-74,45),(-70,46),(-66,44),(-64,47),(-66,60),
          (-72,65),(-80,65),(-85,68),(-90,68),(-95,70),(-100,70),
          (-110,70),(-120,70),(-130,68),(-140,60),(-152,60),(-160,65),
          (-168,65),(-168,71)]
    na = np.array(na)
    outlines.append((na[:,0], na[:,1]))

    # South America (simplified)
    sa = [(-80,8),(-77,4),(-72,-1),(-70,-4),(-74,-10),(-76,-14),(-75,-18),
          (-70,-22),(-68,-28),(-65,-35),(-62,-38),(-56,-38),(-52,-32),
          (-50,-28),(-48,-26),(-44,-20),(-40,-14),(-36,-10),(-36,-5),
          (-36,0),(-42,4),(-50,5),(-60,6),(-66,2),(-72,0),(-78,2),(-80,8)]
    sa = np.array(sa)
    outlines.append((sa[:,0], sa[:,1]))

    # Australia (simplified)
    au = [(114,-22),(114,-35),(117,-35),(121,-34),(126,-34),(130,-32),
          (131,-12),(136,-12),(136,-17),(140,-17),(140,-12),(144,-11),
          (148,-15),(150,-22),(154,-28),(154,-34),(150,-38),(144,-38),
          (138,-36),(136,-34),(130,-32),(126,-34),(120,-34),(114,-30),
          (114,-22)]
    au = np.array(au)
    outlines.append((au[:,0], au[:,1]))

    # Asia (simplified — Middle East + Turkey + South Asia)
    as1 = [(26,37),(30,37),(36,36),(40,36),(42,40),(44,42),(48,42),
           (52,40),(56,36),(60,36),(68,36),(72,22),(80,12),(80,8),
           (78,8),(72,12),(68,22),(60,22),(52,22),(44,22),(40,20),
           (36,18),(32,24),(36,28),(38,28),(36,32),(32,32),(28,36),(26,37)]
    as1 = np.array(as1)
    outlines.append((as1[:,0], as1[:,1]))

    return outlines


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_diversity(output_file: str = 'dataset_diversity.png') -> None:
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        'AIGIS Dataset Diversity  —  15 Training Scenarios + 9 Held-Out (OOD) Scenarios\n'
        'Sources: Copernicus EMS, CAL FIRE, NFPA, Royal Commission, Koutsias et al. (2012), Encinas et al. (2015)',
        color=FG, fontsize=11, fontweight='bold', y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.6, 1],
        hspace=0.42,
        wspace=0.32,
        left=0.05, right=0.97, top=0.92, bottom=0.07,
    )

    # ── Panel 1: World map ───────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.set_facecolor(PANEL)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-55, 80)
    ax_map.set_xlabel('Longitude', color=FG, fontsize=9)
    ax_map.set_ylabel('Latitude', color=FG, fontsize=9)
    ax_map.set_title('Geographic Coverage', color=FG, fontsize=10, fontweight='bold')
    ax_map.tick_params(colors=FG, labelsize=8)
    ax_map.grid(color=GRID, linestyle='--', linewidth=0.4, alpha=0.6)
    for sp in ax_map.spines.values():
        sp.set_edgecolor(GRID)

    # Draw world outlines
    for lon_arr, lat_arr in _world_outlines():
        ax_map.plot(lon_arr, lat_arr, color='#3a3a5c', linewidth=0.9, zorder=1)

    # Plot training scenarios
    for s in TRAINING:
        c = PHASE_COLOURS[s['phase']]
        ax_map.scatter(s['lon'], s['lat'], s=120, c=c, marker=TRAIN_MARKER,
                       zorder=4, edgecolors='white', linewidths=0.6, alpha=0.92)

    # Plot held-out scenarios
    for s in HELD_OUT:
        ax_map.scatter(s['lon'], s['lat'], s=160, c=HELD_COLOUR, marker=HELD_MARKER,
                       zorder=5, edgecolors='white', linewidths=0.8, alpha=0.95)

    # Annotations for selected key scenarios
    label_offsets = {
        'Fort McMurray, AB':   (-6, 2),
        'Gospers Mtn, NSW':    (2, -4),
        'Lahaina 2023':        (-12, -4),
        'Valparaiso 2014':     (-22, 2),
        'Black Saturday 2009': (2, 2),
        'Camp Fire 2018':      (-24, 2),
        'Pedrogao 2017':       (-20, -4),
    }
    for s in TRAINING + HELD_OUT:
        if s['name'] in label_offsets:
            dx, dy = label_offsets[s['name']]
            ax_map.annotate(
                s['name'].split(',')[0].split(' 20')[0].split(' 20')[0],
                xy=(s['lon'], s['lat']),
                xytext=(s['lon'] + dx, s['lat'] + dy),
                fontsize=6.5, color=FG, alpha=0.85,
                arrowprops=dict(arrowstyle='-', color=FG, lw=0.5),
            )

    # Legend
    legend_elements = [
        Line2D([0],[0], marker=TRAIN_MARKER, color='none',
               markerfacecolor=PHASE_COLOURS[1], markersize=9,
               markeredgecolor='white', markeredgewidth=0.5,
               label='Training Phase 1 — Easy (3 scenarios)'),
        Line2D([0],[0], marker=TRAIN_MARKER, color='none',
               markerfacecolor=PHASE_COLOURS[2], markersize=9,
               markeredgecolor='white', markeredgewidth=0.5,
               label='Training Phase 2 — Medium (5 scenarios)'),
        Line2D([0],[0], marker=TRAIN_MARKER, color='none',
               markerfacecolor=PHASE_COLOURS[3], markersize=9,
               markeredgecolor='white', markeredgewidth=0.5,
               label='Training Phase 3 — Hard (7 scenarios)'),
        Line2D([0],[0], marker=HELD_MARKER, color='none',
               markerfacecolor=HELD_COLOUR, markersize=9,
               markeredgecolor='white', markeredgewidth=0.7,
               label='Held-Out OOD (9 scenarios — never seen in training)'),
    ]
    leg = ax_map.legend(handles=legend_elements, loc='lower left',
                        fontsize=8, facecolor='#0d0d1e', labelcolor=FG,
                        edgecolor=GRID, framealpha=0.9)

    # ── Panel 2: Wind Speed distribution ─────────────────────────────────────
    ax_wind = fig.add_subplot(gs[1, 0])
    ax_wind.set_facecolor(PANEL)
    ax_wind.tick_params(colors=FG, labelsize=8)
    ax_wind.set_title('Wind Speed Distribution', color=FG, fontsize=9, fontweight='bold')
    ax_wind.set_ylabel('Wind Speed (m/s)', color=FG, fontsize=8)
    for sp in ax_wind.spines.values():
        sp.set_edgecolor(GRID)
    ax_wind.grid(color=GRID, linestyle='--', linewidth=0.4, alpha=0.5, axis='y')

    train_wind_by_phase = {
        1: [s['wind'] for s in TRAINING if s['phase'] == 1],
        2: [s['wind'] for s in TRAINING if s['phase'] == 2],
        3: [s['wind'] for s in TRAINING if s['phase'] == 3],
    }
    held_wind = [s['wind'] for s in HELD_OUT]

    positions = [1, 2, 3, 4.2]
    data_sets = [train_wind_by_phase[1], train_wind_by_phase[2],
                 train_wind_by_phase[3], held_wind]
    colours_bp = [PHASE_COLOURS[1], PHASE_COLOURS[2], PHASE_COLOURS[3], HELD_COLOUR]
    labels_bp  = ['Train P1', 'Train P2', 'Train P3', 'Held-Out']

    for pos, data, col, lbl in zip(positions, data_sets, colours_bp, labels_bp):
        bp = ax_wind.boxplot(data, positions=[pos], widths=0.55,
                             patch_artist=True, notch=False,
                             medianprops=dict(color='white', linewidth=2),
                             whiskerprops=dict(color=FG),
                             capprops=dict(color=FG),
                             flierprops=dict(marker='o', color=col, markersize=5))
        for patch in bp['boxes']:
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        ax_wind.scatter([pos] * len(data), data, color=col, alpha=0.7, s=25, zorder=3)

    ax_wind.set_xticks(positions)
    ax_wind.set_xticklabels(labels_bp, color=FG, fontsize=7.5)
    ax_wind.tick_params(axis='y', labelcolor=FG)

    # ── Panel 3: Fire Intensity (Wind × ROS) scatter ─────────────────────────
    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_scatter.set_facecolor(PANEL)
    ax_scatter.tick_params(colors=FG, labelsize=8)
    ax_scatter.set_title('Parameter Space Coverage\n(Wind Speed vs Rothermel ROS)',
                         color=FG, fontsize=9, fontweight='bold')
    ax_scatter.set_xlabel('Wind Speed (m/s)', color=FG, fontsize=8)
    ax_scatter.set_ylabel('Rothermel Base ROS (m/s)', color=FG, fontsize=8)
    for sp in ax_scatter.spines.values():
        sp.set_edgecolor(GRID)
    ax_scatter.grid(color=GRID, linestyle='--', linewidth=0.4, alpha=0.5)

    for s in TRAINING:
        c = PHASE_COLOURS[s['phase']]
        # Bubble size proportional to civilians
        ax_scatter.scatter(s['wind'], s['ros'], s=s['civ'] * 1.8,
                           c=c, marker=TRAIN_MARKER, alpha=0.82,
                           edgecolors='white', linewidths=0.5, zorder=3)

    for s in HELD_OUT:
        ax_scatter.scatter(s['wind'], s['ros'], s=s['civ'] * 1.8,
                           c=HELD_COLOUR, marker=HELD_MARKER, alpha=0.88,
                           edgecolors='white', linewidths=0.7, zorder=4)

    # Fire intensity lines (wind * ROS = const)
    x_line = np.linspace(9, 30, 200)
    for intensity, label in [(10, 'FI=10'), (20, 'FI=20'), (30, 'FI=30')]:
        y_line = intensity / x_line
        mask = (y_line >= 0.5) & (y_line <= 1.35)
        if mask.any():
            ax_scatter.plot(x_line[mask], y_line[mask], '--',
                            color=GRID, linewidth=0.8, alpha=0.7)
            idx = np.argwhere(mask).flatten()
            if len(idx) > 0:
                mid = idx[len(idx) // 2]
                ax_scatter.text(x_line[mid], y_line[mid] + 0.01, label,
                                color='#6a6a9e', fontsize=6.5, alpha=0.8)

    ax_scatter.set_xlim(9, 30)
    ax_scatter.set_ylim(0.48, 1.32)
    ax_scatter.tick_params(axis='both', labelcolor=FG)

    # Bubble size legend
    for n_civ, lbl in [(40, '40 civilians'), (70, '70'), (90, '90')]:
        ax_scatter.scatter([], [], s=n_civ * 1.8, c='#555577',
                           edgecolors='white', linewidths=0.4, label=lbl)
    ax_scatter.legend(title='Simulated pop.', title_fontsize=7,
                      fontsize=6.5, facecolor='#0d0d1e', labelcolor=FG,
                      edgecolor=GRID, loc='upper right')

    # ── Panel 4: Temporal + Continental diversity ─────────────────────────────
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.set_facecolor(PANEL)
    ax_info.tick_params(colors=FG, labelsize=8)
    ax_info.set_title('Temporal & Regional Coverage', color=FG, fontsize=9,
                      fontweight='bold')
    for sp in ax_info.spines.values():
        sp.set_edgecolor(GRID)

    # Timeline scatter: year vs intensity
    ax_info.set_xlabel('Year', color=FG, fontsize=8)
    ax_info.set_ylabel('Fire Intensity  (wind × ROS)', color=FG, fontsize=8)
    ax_info.grid(color=GRID, linestyle='--', linewidth=0.4, alpha=0.5)

    for s in TRAINING:
        c = PHASE_COLOURS[s['phase']]
        intensity = s['wind'] * s['ros']
        ax_info.scatter(s['year'], intensity, s=90, c=c, marker=TRAIN_MARKER,
                        alpha=0.85, edgecolors='white', linewidths=0.5, zorder=3)

    for s in HELD_OUT:
        intensity = s['wind'] * s['ros']
        ax_info.scatter(s['year'], intensity, s=120, c=HELD_COLOUR,
                        marker=HELD_MARKER, alpha=0.88, edgecolors='white',
                        linewidths=0.7, zorder=4)

    # Annotate a few notable events
    notable = {
        'Black Saturday 2009': (-0.3, 1.5),
        'Fort McMurray, AB': (0.3, -1.2),
        'Lahaina 2023': (-3.5, 1.0),
        'Peloponnese 2007': (0.3, -1.5),
    }
    all_scenarios = TRAINING + HELD_OUT
    for s in all_scenarios:
        if s['name'] in notable:
            dx, dy = notable[s['name']]
            intensity = s['wind'] * s['ros']
            short = s['name'].split(',')[0].replace(' 20', "'").replace('Fort McMurray', 'Ft McMurray')
            ax_info.annotate(short, xy=(s['year'], intensity),
                             xytext=(s['year'] + dx, intensity + dy),
                             fontsize=6.5, color=FG, alpha=0.85,
                             arrowprops=dict(arrowstyle='-', color=FG, lw=0.5))

    ax_info.tick_params(axis='both', labelcolor=FG)
    ax_info.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

    # ── Continent summary inset ───────────────────────────────────────────────
    from collections import Counter
    all_continents = [s['continent'] for s in TRAINING] + [s['continent'] for s in HELD_OUT]
    counts = Counter(all_continents)
    cont_labels = list(counts.keys())
    cont_vals   = [counts[k] for k in cont_labels]
    cont_cols   = [CONTINENT_COLOURS.get(k, '#888') for k in cont_labels]

    ax_inset = ax_info.inset_axes([0.02, 0.02, 0.38, 0.42])
    ax_inset.set_facecolor('#0d0d1e')
    wedges, _ = ax_inset.pie(cont_vals, colors=cont_cols, startangle=90,
                              wedgeprops=dict(width=0.6, edgecolor='#1a1a2e', linewidth=0.8))
    ax_inset.set_title('Regions', color=FG, fontsize=6.5, pad=2)

    # Continent legend below pie
    for k, col in zip(cont_labels, cont_cols):
        ax_inset.plot([], [], 's', color=col, markersize=5,
                      label=f'{k} ({counts[k]})')
    ax_inset.legend(fontsize=5.5, facecolor='#0d0d1e', labelcolor=FG,
                    edgecolor='none', loc='lower center',
                    bbox_to_anchor=(0.5, -0.65), ncol=1)

    # ── Dataset summary text box ──────────────────────────────────────────────
    n_train = len(TRAINING)
    n_held  = len(HELD_OUT)
    all_yrs = sorted({s['year'] for s in TRAINING + HELD_OUT})
    n_cont  = len(set(s['continent'] for s in TRAINING + HELD_OUT))
    wind_all = [s['wind'] for s in TRAINING + HELD_OUT]
    ros_all  = [s['ros']  for s in TRAINING + HELD_OUT]

    summary = (
        f"Dataset: {n_train + n_held} real incidents\n"
        f"Training: {n_train}  |  Held-Out OOD: {n_held}\n"
        f"Period: {min(all_yrs)}–{max(all_yrs)}\n"
        f"Continents: {n_cont}\n"
        f"Wind: {min(wind_all):.0f}–{max(wind_all):.0f} m/s\n"
        f"ROS:  {min(ros_all):.2f}–{max(ros_all):.2f} m/s\n"
        f"Curriculum: 3 phases (Bengio et al. 2009)"
    )
    ax_info.text(0.98, 0.97, summary,
                 transform=ax_info.transAxes,
                 fontsize=7, color=FG, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d1e',
                           edgecolor=GRID, alpha=0.9),
                 family='monospace')

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Dataset diversity chart saved to: {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate AIGIS dataset diversity chart'
    )
    parser.add_argument('--output', type=str, default='dataset_diversity.png',
                        help='Output PNG file (default: dataset_diversity.png)')
    args = parser.parse_args()
    plot_diversity(output_file=args.output)


if __name__ == '__main__':
    main()
