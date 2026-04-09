"""
Civilian Cognitive State Machine Diagram
=========================================
Three-layer diagram:
  Top row    — Cognitive states (rational / confused / herding)
  Middle row — Life states (alive / evacuated / injured)
  Triggers & guards shown on edges

Output: civilian_fsm.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# ---------------------------------------------------------------------------
# Node definitions: (x, y, label, sublabel, facecolor, textcolor)
# ---------------------------------------------------------------------------

NODES = {
    # Cognitive row  (y = 5.5)
    'rational': (2.0,  5.5, 'RATIONAL',    'panic < 0.3\noptimal navigation',        '#1565C0', 'white'),
    'confused': (7.0,  5.5, 'CONFUSED',    '0.3 ≤ panic < 0.7\ndegraded pathfinding', '#F57C00', 'white'),
    'herding':  (12.0, 5.5, 'HERDING',     'panic ≥ 0.7\nfollows crowd',              '#C62828', 'white'),

    # Life-event row  (y = 2.0)
    'alerted':  (3.5,  2.0, 'ALERTED',     'WARNING\nreceived',                       '#1976D2', 'white'),
    'milling':  (7.0,  2.0, 'MILLING',     'pre-evacuation delay\nlog-normal(182, 0.6)', '#7B1FA2', 'white'),
    'evacuating':(10.5, 2.0,'EVACUATING',  'moving to safe zone\nA* pathfinding',      '#2E7D32', 'white'),
    'safe':     (12.5, 5.5, 'SAFE',        'reached safe zone\nis_evacuated=True',     '#388E3C', 'white'),
    'injured':  (12.5, 2.0, 'INJURED',     'smoke exposure\n> threshold',              '#616161', 'white'),
}

# Redefine overlapping safe/herding — shift safe to bottom-right
NODES['safe']    = (12.5, 0.5, 'SAFE',    'reached safe zone\nis_evacuated = True',  '#388E3C', 'white')
NODES['injured'] = (9.5,  0.5, 'INJURED', 'cumulative smoke\n> injury_threshold',    '#616161', 'white')

BOX_W, BOX_H = 2.2, 1.1

def draw_node(ax, key, nodes):
    x, y, label, sub, fc, tc = nodes[key]
    box = FancyBboxPatch(
        (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
        boxstyle='round,pad=0.08',
        facecolor=fc, edgecolor='#222222', linewidth=1.4,
        zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y + 0.18, label, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=tc, zorder=4)
    ax.text(x, y - 0.22, sub, ha='center', va='center',
            fontsize=6.8, color=tc, linespacing=1.3, zorder=4)
    return (x, y)


def arrow(ax, x0, y0, x1, y1, label='', color='#333333', rad=0.0, fontsize=7.5,
          label_x=None, label_y=None):
    style = f'arc3,rad={rad}'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                connectionstyle=style),
                zorder=2)
    if label:
        mx = label_x if label_x is not None else (x0 + x1) / 2
        my = label_y if label_y is not None else (y0 + y1) / 2
        ax.text(mx, my, label, ha='center', va='center', fontsize=fontsize,
                color=color, style='italic',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1),
                zorder=5)


# ---------------------------------------------------------------------------
# Draw nodes
# ---------------------------------------------------------------------------
positions = {k: draw_node(ax, k, NODES) for k in NODES}

# ---------------------------------------------------------------------------
# Cognitive transitions  (top row, y=5.5)
# ---------------------------------------------------------------------------
# rational → confused
arrow(ax, positions['rational'][0] + BOX_W/2, 5.5,
          positions['confused'][0] - BOX_W/2, 5.5,
      label='panic ≥ 0.3\n(fire proximity / AQI)',
      label_x=4.5, label_y=5.95, color='#E65100')

# confused → herding
arrow(ax, positions['confused'][0] + BOX_W/2, 5.5,
          positions['herding'][0]  - BOX_W/2, 5.5,
      label='panic ≥ 0.7',
      label_x=9.5, label_y=5.95, color='#B71C1C')

# herding → confused (panic decay)
arrow(ax, positions['herding'][0]  - BOX_W/2, 5.2,
          positions['confused'][0] + BOX_W/2, 5.2,
      label='panic < 0.7\n(no fire visible)', rad=-0.18,
      label_x=9.5, label_y=4.85, color='#555555')

# confused → rational (panic decay)
arrow(ax, positions['confused'][0] - BOX_W/2, 5.2,
          positions['rational'][0] + BOX_W/2, 5.2,
      label='panic < 0.3\n(decay / fire gone)', rad=-0.18,
      label_x=4.5, label_y=4.85, color='#555555')

# ---------------------------------------------------------------------------
# Vertical: rational → alerted (WARNING message)
# ---------------------------------------------------------------------------
arrow(ax, 2.0, 5.5 - BOX_H/2,
          3.5, 2.0 + BOX_H/2,
      label='Commander\nWARNING', color='#1565C0',
      label_x=2.4, label_y=3.8)

# ---------------------------------------------------------------------------
# Life-event row transitions  (y=2.0)
# ---------------------------------------------------------------------------
# alerted → milling  (EVACUATE order, fire not visible)
arrow(ax, positions['alerted'][0]   + BOX_W/2, 2.0,
          positions['milling'][0]   - BOX_W/2, 2.0,
      label='EVACUATE order\n(fire not visible)',
      label_x=5.25, label_y=2.45, color='#7B1FA2')

# milling → evacuating
arrow(ax, positions['milling'][0]    + BOX_W/2, 2.0,
          positions['evacuating'][0] - BOX_W/2, 2.0,
      label='milling complete\nOR fire visible',
      label_x=8.75, label_y=2.45, color='#2E7D32')

# alerted → evacuating directly (fire visible → immediate flight)
arrow(ax, positions['alerted'][0]   + BOX_W/2, 1.75,
          positions['evacuating'][0] - BOX_W/2, 1.75,
      label='fire visible\n(immediate flight)', rad=-0.25,
      label_x=7.0, label_y=1.2, color='#C62828')

# evacuating → safe
arrow(ax, positions['evacuating'][0], 2.0 - BOX_H/2,
          positions['safe'][0],       0.5 + BOX_H/2,
      label='reached\nsafe zone',
      label_x=11.8, label_y=1.25, color='#388E3C')

# evacuating → injured
arrow(ax, positions['evacuating'][0] - BOX_W/2, 1.75,
          positions['injured'][0]    + BOX_W/2, 1.75,
      label='smoke exposure\n> threshold', rad=0.0,
      label_x=None, label_y=None, color='#616161')

# rational/confused → injured directly (smoke while not evacuating)
arrow(ax, 3.5, 5.5 - BOX_H/2,
          9.5, 0.5 + BOX_H/2,
      label='smoke injury\n(cumulative PM2.5)',
      label_x=5.8, label_y=2.85, color='#616161', rad=0.0)

# ---------------------------------------------------------------------------
# Section labels
# ---------------------------------------------------------------------------
ax.text(0.3, 7.4, 'Cognitive layer\n(panic-driven)',
        fontsize=9, color='#444444', va='top', style='italic')
ax.text(0.3, 3.2, 'Life-event layer\n(communication-driven)',
        fontsize=9, color='#444444', va='top', style='italic')

ax.axhline(4.1, color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=1)

# ---------------------------------------------------------------------------
# Start marker
# ---------------------------------------------------------------------------
ax.plot(2.0, 7.0, 'o', color='black', markersize=10, zorder=5)
ax.annotate('', xy=(2.0, 5.5 + BOX_H/2), xytext=(2.0, 7.0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=4)
ax.text(2.3, 7.0, 'start', fontsize=8, va='center', color='#333333')

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor='#1565C0', label='Cognitive: rational'),
    mpatches.Patch(facecolor='#F57C00', label='Cognitive: confused'),
    mpatches.Patch(facecolor='#C62828', label='Cognitive: herding'),
    mpatches.Patch(facecolor='#1976D2', label='Alerted (warning)'),
    mpatches.Patch(facecolor='#7B1FA2', label='Milling (pre-evac delay)'),
    mpatches.Patch(facecolor='#2E7D32', label='Evacuating'),
    mpatches.Patch(facecolor='#388E3C', label='Safe (goal state)'),
    mpatches.Patch(facecolor='#616161', label='Injured (absorbing)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=7.5,
          framealpha=0.95, edgecolor='#CCCCCC', ncol=2)

ax.set_title(
    'Civilian Agent — Cognitive State Machine\n'
    'Panic model: Panic(t) = Panic(t\u22121) + \u03b1\u00b7(1/d\u209b\u1d35\u02b3\u1d49) + \u03b2\u00b7\u03a3\u209b\u2098\u2080\u2096\u1d49 \u2212 decay   '
    '(Rao & Georgeff 1995; Lindell & Perry 2012)',
    fontsize=11, fontweight='bold', color='#1A1A1A', pad=10
)

plt.tight_layout()
plt.savefig('civilian_fsm.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: civilian_fsm.png")
