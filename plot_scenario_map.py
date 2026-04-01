"""
Scenario World Map
==================
Plots all 32 AIGIS scenarios on a world map:
  - 23 training scenarios (circles, colour-coded by curriculum phase)
  - 9 held-out validation scenarios (triangles)

Also produces a Mediterranean inset: scenario_map_mediterranean.png

Output: scenario_map.png, scenario_map_mediterranean.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

TRAINING = [
    {'name': 'Bages, Spain',          'lat': 41.698,  'lon':   1.802, 'phase': 1},
    {'name': 'Var, France',           'lat': 43.352,  'lon':   6.198, 'phase': 1},
    {'name': 'Penteli, Greece',       'lat': 38.056,  'lon':  23.868, 'phase': 1},
    {'name': 'Manavgat, Turkey',      'lat': 36.786,  'lon':  31.437, 'phase': 2},
    {'name': 'Rhodes, Greece',        'lat': 36.198,  'lon':  28.002, 'phase': 2},
    {'name': 'Kineta, Greece',        'lat': 38.008,  'lon':  23.140, 'phase': 2},
    {'name': 'Varibobi, Greece',      'lat': 38.128,  'lon':  23.798, 'phase': 2},
    {'name': 'Dadia, Greece',         'lat': 41.300,  'lon':  26.200, 'phase': 2},
    {'name': 'Fort McMurray, Canada', 'lat': 56.726,  'lon':-111.379, 'phase': 3},
    {'name': 'Gospers Mtn, Australia','lat':-33.250,  'lon': 150.400, 'phase': 3},
    {'name': 'Carr Fire, USA',        'lat': 40.588,  'lon':-122.392, 'phase': 3},
    {'name': 'Glass Fire, USA',       'lat': 38.498,  'lon':-122.402, 'phase': 3},
    {'name': 'Woolsey Fire, USA',     'lat': 34.172,  'lon':-118.872, 'phase': 3},
    {'name': 'Thomas Fire, USA',      'lat': 34.354,  'lon':-119.065, 'phase': 3},
    {'name': 'Evia, Greece',          'lat': 38.953,  'lon':  23.150, 'phase': 3},
    {'name': 'Corsica, France',       'lat': 42.302,  'lon':   9.148, 'phase': 1},
    {'name': 'Tuscany, Italy',        'lat': 43.720,  'lon':  10.458, 'phase': 1},
    {'name': 'Carmel, Israel',        'lat': 32.698,  'lon':  35.018, 'phase': 2},
    {'name': 'Dwellingup, Australia', 'lat':-32.714,  'lon': 116.063, 'phase': 2},
    {'name': 'Monchique, Portugal',   'lat': 37.322,  'lon':  -8.553, 'phase': 2},
    {'name': 'Oristano, Italy',       'lat': 40.081,  'lon':   8.595, 'phase': 3},
    {'name': 'Lytton, Canada',        'lat': 50.232,  'lon':-121.583, 'phase': 3},
    {'name': 'Knysna, S. Africa',     'lat':-34.036,  'lon':  23.047, 'phase': 3},
]

HELD_OUT = [
    {'name': 'Mati 2018\nGreece',              'lat': 38.090, 'lon':  23.920},
    {'name': 'Camp Fire 2018\nUSA',            'lat': 39.810, 'lon':-121.437},
    {'name': 'Pedrogao Grande 2017\nPortugal', 'lat': 39.930, 'lon':  -8.130},
    {'name': 'Alexandroupoli 2023\nGreece',    'lat': 40.850, 'lon':  25.874},
    {'name': 'Lahaina 2023\nUSA',              'lat': 20.880, 'lon':-156.680},
    {'name': 'Black Saturday 2009\nAustralia', 'lat':-37.390, 'lon': 145.360},
    {'name': 'Tubbs Fire 2017\nUSA',           'lat': 38.580, 'lon':-122.720},
    {'name': 'Peloponnese 2007\nGreece',       'lat': 37.650, 'lon':  21.630},
    {'name': 'Valparaiso 2014\nChile',         'lat':-33.046, 'lon': -71.617},
]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PHASE_COLORS  = {1: '#1976D2', 2: '#F57C00', 3: '#C62828'}
PHASE_LABELS  = {1: 'Phase 1 — easy (5)', 2: 'Phase 2 — medium (8)', 3: 'Phase 3 — hard (10)'}
HELD_COLOR    = '#2E7D32'
MARKER_SIZE_W = 10   # world map
MARKER_SIZE_M = 13   # Mediterranean map

FONT = {'family': 'DejaVu Sans'}
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    10,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
})

# ---------------------------------------------------------------------------
# World-map label offsets  (lon, lat)
# ---------------------------------------------------------------------------

TRAIN_OFFSETS_W = {
    'Bages, Spain':          (-20,  3),
    'Var, France':           (  2,  3),
    'Penteli, Greece':       (  2, -5),
    'Manavgat, Turkey':      (  2,  3),
    'Rhodes, Greece':        (  3, -5),
    'Kineta, Greece':        (-18, -5),
    'Varibobi, Greece':      (  2,  3),
    'Dadia, Greece':         (  2,  3),
    'Fort McMurray, Canada': (  2,  3),
    'Gospers Mtn, Australia':(  2, -5),
    'Carr Fire, USA':        (-24,  3),
    'Glass Fire, USA':       (  2,  3),
    'Woolsey Fire, USA':     (  2, -5),
    'Thomas Fire, USA':      (-24, -5),
    'Evia, Greece':          (  2,  3),
}

HELD_OFFSETS_W = {
    'Mati 2018\nGreece':              (  2,  3),
    'Camp Fire 2018\nUSA':            (  2,  3),
    'Pedrogao Grande 2017\nPortugal': (-28, -6),
    'Alexandroupoli 2023\nGreece':    (  2,  3),
    'Lahaina 2023\nUSA':              (  2, -5),
    'Black Saturday 2009\nAustralia': (  2,  3),
    'Tubbs Fire 2017\nUSA':           (-24,  3),
    'Peloponnese 2007\nGreece':       (  2, -5),
    'Valparaiso 2014\nChile':         (  2,  3),
}

# Mediterranean label offsets  (lon, lat)
MED_TRAIN_OFFSETS = {
    'Bages, Spain':      (-2.0,  0.4),
    'Var, France':       ( 0.4,  0.4),
    'Penteli, Greece':   ( 0.3, -0.6),
    'Manavgat, Turkey':  ( 0.4,  0.4),
    'Rhodes, Greece':    ( 0.4, -0.6),
    'Kineta, Greece':    (-2.4, -0.6),
    'Varibobi, Greece':  ( 0.3,  0.5),
    'Dadia, Greece':     ( 0.4,  0.4),
    'Evia, Greece':      ( 0.4,  0.4),
}

MED_HELD_OFFSETS = {
    'Mati 2018\nGreece':              ( 0.4,  0.4),
    'Pedrogao Grande 2017\nPortugal': (-4.0, -0.6),
    'Alexandroupoli 2023\nGreece':    ( 0.4,  0.4),
    'Peloponnese 2007\nGreece':       ( 0.4, -0.6),
}

MED_EXTENT = [-12, 38, 33, 48]   # [lon_min, lon_max, lat_min, lat_max]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _shadow():
    return [pe.withStroke(linewidth=2.5, foreground='white')]


def _add_legend(ax, marker_size=10, loc='lower left'):
    handles = [
        mlines.Line2D([], [], color=PHASE_COLORS[1], marker='o', linestyle='None',
                      markersize=marker_size, markeredgecolor='white', markeredgewidth=0.8,
                      label=f'Training {PHASE_LABELS[1]}'),
        mlines.Line2D([], [], color=PHASE_COLORS[2], marker='o', linestyle='None',
                      markersize=marker_size, markeredgecolor='white', markeredgewidth=0.8,
                      label=f'Training {PHASE_LABELS[2]}'),
        mlines.Line2D([], [], color=PHASE_COLORS[3], marker='o', linestyle='None',
                      markersize=marker_size, markeredgecolor='white', markeredgewidth=0.8,
                      label=f'Training {PHASE_LABELS[3]}'),
        mlines.Line2D([], [], color=HELD_COLOR, marker='^', linestyle='None',
                      markersize=marker_size + 1, markeredgecolor='white', markeredgewidth=0.8,
                      label='Held-out validation (9)'),
    ]
    legend = ax.legend(handles=handles, loc=loc, fontsize=9,
                       framealpha=0.95, edgecolor='#BBBBBB',
                       fancybox=True, shadow=False,
                       title='Scenario Sets', title_fontsize=9)
    legend.get_frame().set_linewidth(0.8)


def _annotate(ax, lon, lat, label, dx, dy, fontsize, transform):
    ax.annotate(
        label,
        xy=(lon, lat), xytext=(lon + dx, lat + dy),
        fontsize=fontsize, color='#111111',
        transform=transform, zorder=7,
        path_effects=_shadow(),
        arrowprops=dict(arrowstyle='-', color='#888888',
                        lw=0.5, shrinkA=0, shrinkB=2),
    )


def _draw_markers(ax, scenarios, offsets, color, marker, size, transform, fontsize):
    for s in scenarios:
        c = color if isinstance(color, str) else color[s['phase']]
        ax.plot(s['lon'], s['lat'], marker, color=c,
                markersize=size, markeredgecolor='white', markeredgewidth=1.0,
                transform=transform, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='#333')])
        dx, dy = offsets.get(s['name'], (2, 2))
        _annotate(ax, s['lon'], s['lat'], s['name'], dx, dy, fontsize, transform)


# ---------------------------------------------------------------------------
# World map
# ---------------------------------------------------------------------------

def plot_world():
    fig = plt.figure(figsize=(20, 11), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    # Background
    ax.add_feature(cfeature.LAND,       facecolor='#EEF0E8', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN,      facecolor='#C8DCF0', zorder=1)
    ax.add_feature(cfeature.LAKES,      facecolor='#C8DCF0', edgecolor='none', zorder=2)
    ax.add_feature(cfeature.RIVERS,     edgecolor='#A8C8E8', linewidth=0.3, zorder=2)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6, edgecolor='#666666', zorder=3)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, edgecolor='#AAAAAA',
                   linestyle=':', zorder=3)

    # Subtle grid
    gl = ax.gridlines(linewidth=0.3, color='#CCCCCC', alpha=0.6,
                      linestyle='--', zorder=2)
    gl.top_labels = gl.right_labels = False

    transform = ccrs.PlateCarree()

    # Training
    for s in TRAINING:
        c = PHASE_COLORS[s['phase']]
        ax.plot(s['lon'], s['lat'], 'o', color=c,
                markersize=MARKER_SIZE_W, markeredgecolor='white', markeredgewidth=1.0,
                transform=transform, zorder=6)
        dx, dy = TRAIN_OFFSETS_W.get(s['name'], (2, 2))
        _annotate(ax, s['lon'], s['lat'], s['name'], dx, dy, 6.5, transform)

    # Held-out
    for s in HELD_OUT:
        ax.plot(s['lon'], s['lat'], '^', color=HELD_COLOR,
                markersize=MARKER_SIZE_W + 1, markeredgecolor='white', markeredgewidth=1.0,
                transform=transform, zorder=6)
        dx, dy = HELD_OFFSETS_W.get(s['name'], (2, 2))
        _annotate(ax, s['lon'], s['lat'], s['name'], dx, dy, 6.5, transform)

    # Med region box
    lon0, lon1, lat0, lat1 = MED_EXTENT
    ax.plot([lon0, lon1, lon1, lon0, lon0],
            [lat0, lat0, lat1, lat1, lat0],
            transform=transform, color='#555555',
            linewidth=1.2, linestyle='--', zorder=8)
    ax.text(lon0 + 0.5, lat1 + 1.5, 'see inset',
            transform=transform, fontsize=7, color='#555555',
            style='italic', zorder=8)

    _add_legend(ax, marker_size=MARKER_SIZE_W)

    fig.suptitle(
        'AIGIS Wildfire Simulation — Scenario Locations\n'
        '15 Training Scenarios (Curriculum Phases 1–3) + 9 Held-Out Validation Scenarios',
        fontsize=13, fontweight='bold', y=0.98, color='#1A1A1A'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('scenario_map.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: scenario_map.png")


# ---------------------------------------------------------------------------
# Mediterranean inset
# ---------------------------------------------------------------------------

def plot_mediterranean():
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(MED_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      facecolor='#EEF0E8', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor='#C8DCF0', zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor='#C8DCF0', edgecolor='none', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555', zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor='#999999',
                   linestyle=':', zorder=3)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#CCCCCC',
                      alpha=0.7, linestyle='--', zorder=2,
                      x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 8, 'color': '#444444'}
    gl.ylabel_style = {'size': 8, 'color': '#444444'}

    transform = ccrs.PlateCarree()

    # Mediterranean training scenarios
    med_train = [s for s in TRAINING
                 if MED_EXTENT[0] <= s['lon'] <= MED_EXTENT[1]
                 and MED_EXTENT[2] <= s['lat'] <= MED_EXTENT[3]]
    for s in med_train:
        c = PHASE_COLORS[s['phase']]
        ax.plot(s['lon'], s['lat'], 'o', color=c,
                markersize=MARKER_SIZE_M, markeredgecolor='white', markeredgewidth=1.2,
                transform=transform, zorder=6)
        dx, dy = MED_TRAIN_OFFSETS.get(s['name'], (0.4, 0.4))
        _annotate(ax, s['lon'], s['lat'], s['name'], dx, dy, 8.5, transform)

    # Mediterranean held-out scenarios
    med_held = [s for s in HELD_OUT
                if MED_EXTENT[0] <= s['lon'] <= MED_EXTENT[1]
                and MED_EXTENT[2] <= s['lat'] <= MED_EXTENT[3]]
    for s in med_held:
        ax.plot(s['lon'], s['lat'], '^', color=HELD_COLOR,
                markersize=MARKER_SIZE_M + 1, markeredgecolor='white', markeredgewidth=1.2,
                transform=transform, zorder=6)
        dx, dy = MED_HELD_OFFSETS.get(s['name'], (0.4, 0.4))
        _annotate(ax, s['lon'], s['lat'], s['name'], dx, dy, 8.5, transform)

    # Greece micro-inset (Attica cluster)
    _add_greece_inset(fig, transform)

    _add_legend(ax, marker_size=MARKER_SIZE_M, loc='lower left')

    fig.suptitle(
        'AIGIS Wildfire Simulation — Mediterranean Region\n'
        'Scenario Locations (Training + Held-Out Validation)',
        fontsize=13, fontweight='bold', y=0.99, color='#1A1A1A'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('scenario_map_mediterranean.png', dpi=220, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Saved: scenario_map_mediterranean.png")


def _add_greece_inset(fig, transform):
    """Micro-inset for the dense Attica/Greece cluster."""
    GREECE_EXTENT = [21.5, 25.5, 36.8, 40.2]

    # Inset axes position [left, bottom, width, height] in figure coords
    ax_ins = fig.add_axes([0.63, 0.52, 0.26, 0.34],
                          projection=ccrs.Mercator())
    ax_ins.set_extent(GREECE_EXTENT, crs=ccrs.PlateCarree())

    ax_ins.add_feature(cfeature.LAND,      facecolor='#EEF0E8', edgecolor='none')
    ax_ins.add_feature(cfeature.OCEAN,     facecolor='#C8DCF0')
    ax_ins.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#555555')
    ax_ins.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='#AAAAAA',
                       linestyle=':')

    # Label offsets for micro-inset (lon, lat)
    micro_train_offsets = {
        'Penteli, Greece':  ( 0.15, -0.25),
        'Kineta, Greece':   (-0.9,  -0.25),
        'Varibobi, Greece': ( 0.15,  0.18),
        'Dadia, Greece':    ( 0.15,  0.18),
        'Rhodes, Greece':   ( 0.15, -0.25),
        'Evia, Greece':     ( 0.15,  0.18),
        'Manavgat, Turkey': ( 0.15,  0.18),
    }
    micro_held_offsets = {
        'Mati 2018\nGreece':           ( 0.15,  0.18),
        'Alexandroupoli 2023\nGreece': ( 0.15,  0.18),
        'Peloponnese 2007\nGreece':    ( 0.15, -0.28),
    }

    greece_train = [s for s in TRAINING
                    if GREECE_EXTENT[0] <= s['lon'] <= GREECE_EXTENT[1]
                    and GREECE_EXTENT[2] <= s['lat'] <= GREECE_EXTENT[3]]
    for s in greece_train:
        c = PHASE_COLORS[s['phase']]
        ax_ins.plot(s['lon'], s['lat'], 'o', color=c,
                    markersize=7, markeredgecolor='white', markeredgewidth=0.8,
                    transform=transform, zorder=6)
        dx, dy = micro_train_offsets.get(s['name'], (0.15, 0.15))
        ax_ins.annotate(
            s['name'], xy=(s['lon'], s['lat']),
            xytext=(s['lon'] + dx, s['lat'] + dy),
            fontsize=6, color='#111111', transform=transform, zorder=7,
            path_effects=_shadow(),
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.4,
                            shrinkA=0, shrinkB=1),
        )

    greece_held = [s for s in HELD_OUT
                   if GREECE_EXTENT[0] <= s['lon'] <= GREECE_EXTENT[1]
                   and GREECE_EXTENT[2] <= s['lat'] <= GREECE_EXTENT[3]]
    for s in greece_held:
        ax_ins.plot(s['lon'], s['lat'], '^', color=HELD_COLOR,
                    markersize=8, markeredgecolor='white', markeredgewidth=0.8,
                    transform=transform, zorder=6)
        dx, dy = micro_held_offsets.get(s['name'], (0.15, 0.15))
        ax_ins.annotate(
            s['name'], xy=(s['lon'], s['lat']),
            xytext=(s['lon'] + dx, s['lat'] + dy),
            fontsize=6, color='#111111', transform=transform, zorder=7,
            path_effects=_shadow(),
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.4,
                            shrinkA=0, shrinkB=1),
        )

    # Box border + label
    for spine in ax_ins.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.2)
    ax_ins.set_title('Greece / Attica detail', fontsize=7.5,
                     color='#333333', pad=3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if HAS_CARTOPY:
        print("Using cartopy for high-quality maps...")
        plot_world()
        plot_mediterranean()
    else:
        print("cartopy not found — please install: pip install cartopy")
