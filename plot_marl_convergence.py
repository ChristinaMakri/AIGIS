"""
MARL Convergence Zoom
=====================
Plots the rolling-mean reward with the convergence annotation at episode 712.
Two panels:
  Left  — full 4000-episode training run
  Right — zoom into episodes 400–1000 showing the plateau detection

Output: marl_convergence.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONV_EP = 712
WINDOW  = 500

df = pd.read_csv('marl_training_log.csv')
df['reward_total'] = df['reward_ff'] + df['reward_rsc'] + df['reward_cmd']
df['roll_mean']    = df['reward_total'].rolling(window=WINDOW, min_periods=1).mean()

episodes = df['episode'].values
total    = df['reward_total'].values
roll     = df['roll_mean'].values

PHASE_BOUNDS = [
    (1,    800,  '#E3F2FD', 'Phase 1 (easy)'),
    (800,  2400, '#FFF3E0', 'Phase 2 (medium)'),
    (2400, 4001, '#FCE4EC', 'Phase 3 (hard)'),
]

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5),
                                        facecolor='white',
                                        gridspec_kw={'wspace': 0.28})

def draw_panel(ax, ep_min, ep_max):
    mask = (episodes >= ep_min) & (episodes <= ep_max)
    ep_s = episodes[mask]
    to_s = total[mask]
    ro_s = roll[mask]

    # Phase backgrounds
    seen_phases = set()
    for p_lo, p_hi, p_col, p_lbl in PHASE_BOUNDS:
        x0, x1 = max(p_lo, ep_min), min(p_hi, ep_max)
        if x0 < x1:
            label = p_lbl if p_lbl not in seen_phases else None
            ax.axvspan(x0, x1, color=p_col, alpha=0.5, zorder=0, label=label)
            seen_phases.add(p_lbl)

    ax.plot(ep_s, to_s, color='#90CAF9', linewidth=0.7, alpha=0.5, zorder=1,
            label='Episode reward')
    ax.plot(ep_s, ro_s, color='#1565C0', linewidth=2.0, zorder=2,
            label=f'{WINDOW}-ep rolling mean')

    if ep_min <= CONV_EP <= ep_max:
        ax.axvline(CONV_EP, color='#C62828', linewidth=1.5, linestyle='--', zorder=3)
        ylo, yhi = to_s.min(), to_s.max()
        y_ann    = ylo + (yhi - ylo) * 0.85
        x_off    = 60 if ep_max > 2000 else 15
        ax.annotate(
            f'Convergence\nep {CONV_EP}',
            xy=(CONV_EP, y_ann),
            xytext=(CONV_EP + x_off, y_ann),
            fontsize=8.5, color='#C62828', fontweight='bold',
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.0),
            zorder=4,
        )

    ax.set_xlim(ep_min, ep_max)
    ax.set_xlabel('Episode')
    ax.grid(axis='y', linewidth=0.4, color='#DDDDDD', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    seen, unique = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    ax.legend(*zip(*unique), fontsize=7.5, loc='lower right', framealpha=0.9)


draw_panel(ax_full, 1, 4000)
ax_full.set_title('Full training run (4 000 episodes)', fontweight='bold')
ax_full.set_ylabel('Total reward (firefighter + rescuer + commander)')

draw_panel(ax_zoom, 400, 1050)
ax_zoom.set_title('Convergence detail (episodes 400–1050)', fontweight='bold')

fig.suptitle(
    'AIGIS MARL Training — Reward Curve and Convergence Detection\n'
    'Rolling-window plateau criterion (de Witt et al. 2020; Yu et al. 2022)',
    fontsize=12, fontweight='bold', y=1.02, color='#1A1A1A'
)

plt.savefig('marl_convergence.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: marl_convergence.png")
