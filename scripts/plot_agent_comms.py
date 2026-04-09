"""
Agent Communication Flow Diagram
==================================
Shows the inter-agent message topology for AIGIS.
Nodes = agents; directed edges = message flows (label = performative : content type).

Layout:
  Row 1 (top)    : Sentinel (reactive, sensor)  |  RiskMonitor (model-based)
  Row 2          : Analyst   (BDI, physics models)
  Row 3 (centre) : Commander (BDI, CNP auctioneer)
  Row 4 (bottom) : Firefighter | Rescuer | Ambulance | Civilian

Output: agent_comms.png
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(18, 11), facecolor='white')
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')

# ---------------------------------------------------------------------------
# Agent nodes  (x, y, label, arch, facecolor)
# ---------------------------------------------------------------------------
AGENTS = {
    'sentinel':    ( 5.5,  9.5, 'Sentinel',      'Reactive / SDT',            '#0D47A1'),
    'riskmonitor': (13.0,  9.5, 'RiskMonitor',   'Model-Based / FWI + FIRMS', '#00695C'),
    'analyst':     ( 5.5,  7.2, 'Analyst',       'BDI / Rothermel + Fuzzy',   '#1565C0'),
    'commander':   ( 9.0,  5.0, 'Commander',     'BDI / CNP Auctioneer',      '#4527A0'),
    'firefighter': ( 2.0,  2.2, 'Firefighter',   'Goal-based / CNP Bidder',   '#B71C1C'),
    'rescuer':     ( 6.0,  2.2, 'Rescuer',       'Goal-based / CNP Bidder',   '#E65100'),
    'ambulance':   (10.5,  2.2, 'Ambulance',     'BDI / CNP Bidder',          '#1B5E20'),
    'civilian':    (15.0,  2.2, 'Civilian',      'BDI / Panic + LWR',         '#4E342E'),
}

BW, BH = 2.8, 0.9

def draw_agent(ax, key):
    x, y, name, arch, fc = AGENTS[key]
    box = FancyBboxPatch(
        (x - BW/2, y - BH/2), BW, BH,
        boxstyle='round,pad=0.1',
        facecolor=fc, edgecolor='#111111', linewidth=1.6, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y + 0.16, name, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)
    ax.text(x, y - 0.2, arch, ha='center', va='center',
            fontsize=7, color='#DDDDDD', zorder=4)

for k in AGENTS:
    draw_agent(ax, k)

# ---------------------------------------------------------------------------
def get_pos(key):
    return AGENTS[key][0], AGENTS[key][1]

def edge(ax, src, dst, label, color='#333333', rad=0.0,
         src_off=(0, 0), dst_off=(0, 0), lx=None, ly=None, fs=7.0):
    x0, y0 = get_pos(src)
    x1, y1 = get_pos(dst)
    x0 += src_off[0]; y0 += src_off[1]
    x1 += dst_off[0]; y1 += dst_off[1]
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)
    if label:
        mx = lx if lx is not None else (x0 + x1) / 2
        my = ly if ly is not None else (y0 + y1) / 2
        ax.text(mx, my, label, ha='center', va='center', fontsize=fs,
                color=color, style='italic',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.5),
                zorder=5)

# ---------------------------------------------------------------------------
# Environment → Sentinel / RiskMonitor (implicit perception)
# ---------------------------------------------------------------------------
ax.annotate('', xy=(5.5, 9.5 + BH/2), xytext=(5.5, 10.7),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.2, linestyle='dashed'), zorder=2)
ax.text(6.2, 10.3, 'Environment\n(fire_grid, temperature_grid)',
        fontsize=7, color='#78909C', ha='left', va='center', style='italic')

ax.annotate('', xy=(13.0, 9.5 + BH/2), xytext=(13.0, 10.7),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.2, linestyle='dashed'), zorder=2)
ax.text(13.7, 10.3, 'Environment\n(FWI, fuel, FIRMS, SRTM)',
        fontsize=7, color='#78909C', ha='left', va='center', style='italic')

# ---------------------------------------------------------------------------
# Detection / Pre-ignition chain
# ---------------------------------------------------------------------------
# Sentinel → Analyst
edge(ax, 'sentinel', 'analyst',
     label='INFORM : FIRE_DETECTION\n(lat, lon, intensity)',
     color='#1565C0', lx=4.0, ly=8.4)

# RiskMonitor → Commander (pre-ignition risk forecast)
edge(ax, 'riskmonitor', 'commander',
     label='INFORM : RISK_FORECAST\n(fwi, max_risk, high_risk_zones)',
     color='#00695C', rad=0.0, lx=12.5, ly=7.1)

# Analyst → Commander
edge(ax, 'analyst', 'commander',
     label='INFORM : RISK_REPORT\n(max_risk, TTI, ROS, exits)',
     color='#1565C0', lx=6.5, ly=6.1)

# Firefighter → Analyst (suppression feedback)
edge(ax, 'firefighter', 'analyst',
     label='INFORM : SUPPRESSION_UPDATE\n(row, col)',
     color='#B71C1C', rad=-0.25, lx=2.5, ly=5.5)

# Commander → Analyst (phase update)
edge(ax, 'commander', 'analyst',
     label='INFORM : PHASE_UPDATE\n(phase)',
     color='#4527A0',
     src_off=(-0.3, 0), dst_off=(-0.3, 0),
     rad=0.35, lx=4.8, ly=6.1)

# ---------------------------------------------------------------------------
# CNP: Commander ↔ Firefighter / Rescuer / Ambulance
# ---------------------------------------------------------------------------
edge(ax, 'commander', 'firefighter',
     label='CFP : FIRE_SUPPRESSION_CFP',
     color='#B71C1C', rad=0.2, lx=4.0, ly=3.9)
edge(ax, 'firefighter', 'commander',
     label='PROPOSE / REFUSE / CONFIRM',
     color='#B71C1C', rad=0.2, lx=3.0, ly=3.4)

edge(ax, 'commander', 'rescuer',
     label='CFP : RESCUE_CFP',
     color='#E65100', rad=0.1, lx=6.8, ly=3.9)
edge(ax, 'rescuer', 'commander',
     label='PROPOSE / REFUSE / CONFIRM',
     color='#E65100', rad=0.1, lx=7.8, ly=3.4)

edge(ax, 'commander', 'ambulance',
     label='CFP : AMBULANCE_CFP',
     color='#1B5E20', rad=-0.1, lx=10.0, ly=3.9)
edge(ax, 'ambulance', 'commander',
     label='PROPOSE / REFUSE / CONFIRM',
     color='#1B5E20', rad=-0.1, lx=9.0, ly=3.4)

# ---------------------------------------------------------------------------
# Commander → Civilian (warnings & orders)
# ---------------------------------------------------------------------------
edge(ax, 'commander', 'civilian',
     label='INFORM : WARNING / FWI_WARNING\nREQUEST : EVACUATE / REDIRECT',
     color='#4E342E', rad=-0.15, lx=13.5, ly=3.9)

# Civilian → Ambulance (injury report)
edge(ax, 'civilian', 'ambulance',
     label='INFORM : INJURY_REPORT\n(direct dispatch)',
     color='#616161',
     src_off=(-BW/2, 0), dst_off=(BW/2, 0),
     lx=12.8, ly=2.2)

# ---------------------------------------------------------------------------
# Protocol labels
# ---------------------------------------------------------------------------
ax.text(0.2, 8.8, 'Pre-ignition\nchain', fontsize=8, color='#00695C',
        va='center', style='italic', fontweight='bold')
ax.text(0.2, 7.8, 'Detection\nchain', fontsize=8, color='#1565C0',
        va='center', style='italic', fontweight='bold')
ax.text(0.2, 3.1, 'Contract Net\nProtocol (CNP)\nSmith (1980)',
        fontsize=8, color='#4527A0', va='center', style='italic', fontweight='bold')

ax.plot([1.0, 1.0], [1.6, 4.5], color='#4527A0', lw=1.5)
ax.plot([1.0, 1.2], [1.6, 1.6], color='#4527A0', lw=1.5)
ax.plot([1.0, 1.2], [4.5, 4.5], color='#4527A0', lw=1.5)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor='#0D47A1', label='Reactive (Sentinel)'),
    mpatches.Patch(facecolor='#00695C', label='Model-Based (RiskMonitor)'),
    mpatches.Patch(facecolor='#1565C0', label='BDI — perception (Analyst)'),
    mpatches.Patch(facecolor='#4527A0', label='BDI — coordination (Commander)'),
    mpatches.Patch(facecolor='#B71C1C', label='Goal-based — CNP (Firefighter)'),
    mpatches.Patch(facecolor='#E65100', label='Goal-based — CNP (Rescuer)'),
    mpatches.Patch(facecolor='#1B5E20', label='BDI — CNP (Ambulance)'),
    mpatches.Patch(facecolor='#4E342E', label='BDI — panic model (Civilian)'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=7.5,
          framealpha=0.95, edgecolor='#CCCCCC', ncol=2,
          title='Agent architectures', title_fontsize=8)

ax.set_title(
    'AIGIS — Inter-Agent Communication Topology\n'
    'Performatives: INFORM · REQUEST · CFP · PROPOSE · REFUSE · CONFIRM   '
    '(FIPA-ACL [56]; Smith 1980 [16])',
    fontsize=11, fontweight='bold', color='#1A1A1A', pad=10
)

plt.tight_layout()
plt.savefig('agent_comms.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: agent_comms.png")
