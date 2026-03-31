"""
Scenario World Map
==================
Plots all 24 AIGIS scenarios on a world map:
  - 15 training scenarios (blue circles, phases 1-3)
  - 9 held-out validation scenarios (red triangles)

Output: scenario_map.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

TRAINING = [
    # Phase 1 — easy
    {'name': 'Bages, Spain',         'lat': 41.698,  'lon':   1.802, 'phase': 1},
    {'name': 'Var, France',          'lat': 43.352,  'lon':   6.198, 'phase': 1},
    {'name': 'Penteli, Greece',      'lat': 38.056,  'lon':  23.868, 'phase': 1},
    # Phase 2 — medium
    {'name': 'Manavgat, Turkey',     'lat': 36.786,  'lon':  31.437, 'phase': 2},
    {'name': 'Rhodes, Greece',       'lat': 36.198,  'lon':  28.002, 'phase': 2},
    {'name': 'Kineta, Greece',       'lat': 38.008,  'lon':  23.140, 'phase': 2},
    {'name': 'Varibobi, Greece',     'lat': 38.128,  'lon':  23.798, 'phase': 2},
    {'name': 'Dadia, Greece',        'lat': 41.300,  'lon':  26.200, 'phase': 2},
    # Phase 3 — hard
    {'name': 'Fort McMurray, Canada','lat': 56.726,  'lon':-111.379, 'phase': 3},
    {'name': 'Gospers Mtn, Australia','lat':-33.250, 'lon': 150.400, 'phase': 3},
    {'name': 'Carr Fire, USA',       'lat': 40.588,  'lon':-122.392, 'phase': 3},
    {'name': 'Glass Fire, USA',      'lat': 38.498,  'lon':-122.402, 'phase': 3},
    {'name': 'Woolsey Fire, USA',    'lat': 34.172,  'lon':-118.872, 'phase': 3},
    {'name': 'Thomas Fire, USA',     'lat': 34.354,  'lon':-119.065, 'phase': 3},
    {'name': 'Evia, Greece',         'lat': 38.953,  'lon':  23.150, 'phase': 3},
]

HELD_OUT = [
    {'name': 'Mati 2018, Greece',           'lat': 38.090, 'lon':  23.920},
    {'name': 'Camp Fire 2018, USA',         'lat': 39.810, 'lon':-121.437},
    {'name': 'Pedrogao Grande 2017, Portugal','lat': 39.930, 'lon':  -8.130},
    {'name': 'Alexandroupoli 2023, Greece', 'lat': 40.850, 'lon':  25.874},
    {'name': 'Lahaina 2023, USA',           'lat': 20.880, 'lon':-156.680},
    {'name': 'Black Saturday 2009, Australia','lat':-37.390,'lon': 145.360},
    {'name': 'Tubbs Fire 2017, USA',        'lat': 38.580, 'lon':-122.720},
    {'name': 'Peloponnese 2007, Greece',    'lat': 37.650, 'lon':  21.630},
    {'name': 'Valparaiso 2014, Chile',      'lat':-33.046, 'lon': -71.617},
]

# ---------------------------------------------------------------------------
# Colours per phase
# ---------------------------------------------------------------------------
PHASE_COLORS = {1: '#2196F3', 2: '#FF9800', 3: '#F44336'}  # blue / orange / red
HELD_COLOR   = '#4CAF50'   # green

# ---------------------------------------------------------------------------
# Label offsets to reduce overlap  (lon_offset, lat_offset)
# ---------------------------------------------------------------------------
TRAIN_OFFSETS = {
    'Bages, Spain':          (-18,  2),
    'Var, France':           (  2,  3),
    'Penteli, Greece':       (  2, -4),
    'Manavgat, Turkey':      (  2,  2),
    'Rhodes, Greece':        (  3, -4),
    'Kineta, Greece':        ( -16, -4),
    'Varibobi, Greece':      (  2,  2),
    'Dadia, Greece':         (  2,  2),
    'Fort McMurray, Canada': (  2,  2),
    'Gospers Mtn, Australia':(  2, -4),
    'Carr Fire, USA':        (-22,  2),
    'Glass Fire, USA':       (  2,  2),
    'Woolsey Fire, USA':     (  2, -4),
    'Thomas Fire, USA':      (-22, -4),
    'Evia, Greece':          (  2,  2),
}

HELD_OFFSETS = {
    'Mati 2018, Greece':              ( 2,  2),
    'Camp Fire 2018, USA':            ( 2,  2),
    'Pedrogao Grande 2017, Portugal': (-26, -4),
    'Alexandroupoli 2023, Greece':    ( 2,  2),
    'Lahaina 2023, USA':              ( 2, -4),
    'Black Saturday 2009, Australia': ( 2,  2),
    'Tubbs Fire 2017, USA':           (-22,  2),
    'Peloponnese 2007, Greece':       ( 2, -4),
    'Valparaiso 2014, Chile':         ( 2,  2),
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_with_cartopy():
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    ax.add_feature(cfeature.LAND,  facecolor='#F5F5F0', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#D6EAF8')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#777')
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor='#AAA', linestyle=':')

    transform = ccrs.PlateCarree()

    # Training scenarios
    for s in TRAINING:
        c = PHASE_COLORS[s['phase']]
        ax.plot(s['lon'], s['lat'], 'o', color=c, markersize=9,
                markeredgecolor='white', markeredgewidth=0.8,
                transform=transform, zorder=5)
        dx, dy = TRAIN_OFFSETS.get(s['name'], (2, 2))
        ax.annotate(s['name'], xy=(s['lon'], s['lat']),
                    xytext=(s['lon'] + dx, s['lat'] + dy),
                    fontsize=6.5, color='#222',
                    transform=transform, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.4))

    # Held-out scenarios
    for s in HELD_OUT:
        ax.plot(s['lon'], s['lat'], '^', color=HELD_COLOR, markersize=10,
                markeredgecolor='white', markeredgewidth=0.8,
                transform=transform, zorder=5)
        dx, dy = HELD_OFFSETS.get(s['name'], (2, 2))
        ax.annotate(s['name'], xy=(s['lon'], s['lat']),
                    xytext=(s['lon'] + dx, s['lat'] + dy),
                    fontsize=6.5, color='#222',
                    transform=transform, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.4))

    _add_legend(ax)
    _finalise(fig)


def plot_without_cartopy():
    """Fallback using plain matplotlib with a simple Robinson-like projection."""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_facecolor('#D6EAF8')

    # Simple world background from matplotlib
    import matplotlib.image as mpimg
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude',  fontsize=10)

    # Draw a basic land polygon outline using built-in data
    try:
        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='robin', lon_0=0, ax=ax)
        m.drawcoastlines(linewidth=0.5, color='#777')
        m.drawcountries(linewidth=0.3, color='#AAA')
        m.fillcontinents(color='#F5F5F0', lake_color='#D6EAF8')
        m.drawmapboundary(fill_color='#D6EAF8')

        for s in TRAINING:
            x, y = m(s['lon'], s['lat'])
            c = PHASE_COLORS[s['phase']]
            ax.plot(x, y, 'o', color=c, markersize=9,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)

        for s in HELD_OUT:
            x, y = m(s['lon'], s['lat'])
            ax.plot(x, y, '^', color=HELD_COLOR, markersize=10,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)

    except ImportError:
        # Absolute fallback: plain scatter on lat/lon axes
        ax.set_facecolor('#D6EAF8')
        ax.axhline(0, color='#AAA', linewidth=0.5)
        ax.axvline(0, color='#AAA', linewidth=0.5)

        for s in TRAINING:
            c = PHASE_COLORS[s['phase']]
            ax.plot(s['lon'], s['lat'], 'o', color=c, markersize=9,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            dx, dy = TRAIN_OFFSETS.get(s['name'], (2, 2))
            ax.annotate(s['name'], xy=(s['lon'], s['lat']),
                        xytext=(s['lon'] + dx, s['lat'] + dy),
                        fontsize=6.5, color='#222', zorder=6,
                        arrowprops=dict(arrowstyle='-', color='#999', lw=0.4))

        for s in HELD_OUT:
            ax.plot(s['lon'], s['lat'], '^', color=HELD_COLOR, markersize=10,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
            dx, dy = HELD_OFFSETS.get(s['name'], (2, 2))
            ax.annotate(s['name'], xy=(s['lon'], s['lat']),
                        xytext=(s['lon'] + dx, s['lat'] + dy),
                        fontsize=6.5, color='#222', zorder=6,
                        arrowprops=dict(arrowstyle='-', color='#999', lw=0.4))

    _add_legend(ax)
    _finalise(fig)


def _add_legend(ax):
    handles = [
        mlines.Line2D([], [], color=PHASE_COLORS[1], marker='o', linestyle='None',
                      markersize=9, markeredgecolor='white', label='Training — Phase 1 (easy)'),
        mlines.Line2D([], [], color=PHASE_COLORS[2], marker='o', linestyle='None',
                      markersize=9, markeredgecolor='white', label='Training — Phase 2 (medium)'),
        mlines.Line2D([], [], color=PHASE_COLORS[3], marker='o', linestyle='None',
                      markersize=9, markeredgecolor='white', label='Training — Phase 3 (hard)'),
        mlines.Line2D([], [], color=HELD_COLOR, marker='^', linestyle='None',
                      markersize=10, markeredgecolor='white', label='Held-out validation'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=9,
              framealpha=0.9, edgecolor='#CCC')


def _finalise(fig):
    fig.suptitle(
        'AIGIS Scenario Locations: 15 Training + 9 Held-Out Validation Scenarios',
        fontsize=13, fontweight='bold', y=0.97
    )
    plt.tight_layout()
    plt.savefig('scenario_map.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Saved: scenario_map.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if HAS_CARTOPY:
        print("Using cartopy for high-quality map...")
        plot_with_cartopy()
    else:
        print("cartopy not found — using fallback renderer...")
        plot_without_cartopy()
