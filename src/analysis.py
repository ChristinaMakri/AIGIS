"""
Post-simulation analysis and visualization for AIGIS.

Provides:
- plot_simulation_summary: 4-panel static summary figure
- save_animation: MP4 animation of fire progression
- plot_monte_carlo_results: Mean±std bar charts for batch runs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Dict, Any


def plot_simulation_summary(
    results: Dict[str, Any],
    history: Optional[Dict] = None,
    env=None,
    output_path: Optional[str] = "output/summary.png"
) -> plt.Figure:
    """
    Generate a 4-panel simulation summary figure.

    Panels:
      1. Fire progression  (burning & burnt cells over time)
      2. Evacuation curve  (evacuated & casualties over time)
      3. Phase timeline    (commander phase over steps)
      4. Panic over time   (mean panic level)

    Args:
        results:     dict returned by simulation.get_results()
        history:     dict with per-step metric lists (from results['history'])
        env:         Environment instance (optional, for map snapshot)
        output_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    if history is None:
        history = results.get('history', {})

    steps = list(range(len(history.get('active_fires', []))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AIGIS Simulation Summary", fontsize=14, fontweight='bold')

    # --- Panel 1: Fire Progression ---
    ax1 = axes[0, 0]
    active_fires = history.get('active_fires', [])
    burnt_cells = history.get('burnt_cells', [])
    if active_fires:
        ax1.fill_between(steps, active_fires, alpha=0.5, color='orangered', label='Burning')
        ax1.plot(steps, active_fires, color='orangered', linewidth=1.5)
    if burnt_cells:
        ax1.fill_between(steps, burnt_cells, alpha=0.3, color='gray', label='Burnt')
        ax1.plot(steps, burnt_cells, color='gray', linewidth=1.0, linestyle='--')
    ax1.set_title("Fire Progression", fontweight='bold')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Grid Cells")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Evacuation Curve ---
    ax2 = axes[0, 1]
    casualties = history.get('casualties', [])
    evacuated = history.get('evacuated', [])
    if evacuated:
        ax2.plot(steps[:len(evacuated)], evacuated, color='green',
                 linewidth=2, label='Evacuated')
        ax2.fill_between(steps[:len(evacuated)], evacuated, alpha=0.3, color='green')
    if casualties:
        ax2.plot(steps[:len(casualties)], casualties, color='red',
                 linewidth=2, label='Casualties')
        ax2.fill_between(steps[:len(casualties)], casualties, alpha=0.3, color='red')
    ax2.set_title("Evacuation Curve", fontweight='bold')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Civilians")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Phase Timeline ---
    ax3 = axes[1, 0]
    phase_history = history.get('phase_history', [])
    phase_colors = {0: '#2196F3', 1: '#FF9800', 2: '#F44336', 3: '#9C27B0'}
    phase_names = {0: 'Monitor', 1: 'Pre-Alert', 2: 'Evacuate', 3: 'Shelter'}
    if phase_history:
        for phase, color in phase_colors.items():
            mask = [i for i, p in enumerate(phase_history) if p == phase]
            if mask:
                for i in mask:
                    ax3.axvspan(i, i + 1, facecolor=color, alpha=0.6)

        legend_patches = [
            mpatches.Patch(color=c, label=phase_names[p], alpha=0.8)
            for p, c in phase_colors.items()
        ]
        ax3.legend(handles=legend_patches, fontsize=8, loc='upper right')
    ax3.set_title("Commander Phase Timeline", fontweight='bold')
    ax3.set_xlabel("Step")
    ax3.set_yticks([])
    ax3.set_xlim(0, max(len(phase_history), 1))

    # --- Panel 4: Panic Over Time ---
    ax4 = axes[1, 1]
    panic_levels = history.get('panic_levels', [])
    if panic_levels:
        ax4.plot(steps[:len(panic_levels)], panic_levels, color='purple',
                 linewidth=2, label='Mean Panic')
        ax4.axhline(0.4, color='orange', linestyle='--', alpha=0.7, label='Confused threshold')
        ax4.axhline(0.7, color='red', linestyle='--', alpha=0.7, label='Herding threshold')
        ax4.set_ylim(0, 1)
    ax4.set_title("Average Panic Level", fontweight='bold')
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Panic Level")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Add summary text box
    total = results.get('total_civilians', 0)
    cas = results.get('casualties', 0)
    evac = results.get('evacuated', 0)
    summary_text = (
        f"Steps: {results.get('steps', 0)}  |  "
        f"Civilians: {total}  |  "
        f"Casualties: {cas} ({results.get('mortality_rate', 0):.1%})  |  "
        f"Evacuated: {evac} ({results.get('evacuation_success_rate', 0):.1%})"
    )
    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=100, bbox_inches='tight')
        print(f"  Summary figure saved to {out}")

    return fig


def save_animation(
    history_frames: list,
    env,
    output_path: str = "output/simulation.mp4",
    fps: int = 10
) -> None:
    """
    Save fire progression animation as MP4.

    Args:
        history_frames: List of fire_grid numpy arrays (one per step)
        env:            Environment instance
        output_path:    Path for output MP4 file
        fps:            Frames per second
    """
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
    except ImportError:
        print("  [Analysis] matplotlib.animation not available")
        return

    if not history_frames:
        print("  [Analysis] No frames to animate")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("AIGIS Fire Progression")

    def _grid_to_rgb(fire_grid):
        vis = np.zeros((*fire_grid.shape, 3), dtype=np.float32)
        vis[fire_grid == 3] = [0.13, 0.55, 0.13]   # fuel
        vis[fire_grid == 1] = [1.0, 0.27, 0.0]     # burning
        vis[fire_grid == 2] = [0.18, 0.31, 0.31]   # burnt
        return vis

    im = ax.imshow(_grid_to_rgb(history_frames[0]), origin='upper', animated=True)
    step_text = ax.text(0.02, 0.97, "Step 0", transform=ax.transAxes,
                        fontsize=10, color='white', va='top')

    def _update(i):
        im.set_array(_grid_to_rgb(history_frames[i]))
        step_text.set_text(f"Step {i}")
        return [im, step_text]

    anim = animation.FuncAnimation(fig, _update, frames=len(history_frames),
                                    interval=1000 // fps, blit=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(title='AIGIS Simulation'))
        anim.save(str(out), writer=writer)
        print(f"  Animation saved to {out}")
    except Exception as e:
        print(f"  [Analysis] Could not save animation (ffmpeg required): {e}")

    plt.close(fig)


def plot_monte_carlo_results(
    batch_df,
    output_path: Optional[str] = "output/monte_carlo.png"
) -> plt.Figure:
    """
    Plot mean±std bar charts from Monte Carlo batch results.

    Args:
        batch_df:    pandas DataFrame with batch run results
        output_path: If provided, save figure here

    Returns:
        matplotlib Figure
    """
    metrics = ['casualties', 'evacuated', 'mortality_rate', 'avg_panic_level']
    labels = ['Casualties', 'Evacuated', 'Mortality Rate', 'Avg Panic']

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.suptitle("Monte Carlo Results (Mean ± Std)", fontsize=13, fontweight='bold')

    for ax, col, label in zip(axes, metrics, labels):
        if col not in batch_df.columns:
            ax.set_visible(False)
            continue
        mean_val = batch_df[col].mean()
        std_val = batch_df[col].std()
        ax.bar([label], [mean_val], yerr=[std_val], capsize=8,
               color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        ax.text(0, mean_val + std_val * 0.5,
                f"{mean_val:.2f}±{std_val:.2f}",
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=100, bbox_inches='tight')
        print(f"  Monte Carlo figure saved to {out}")

    return fig
