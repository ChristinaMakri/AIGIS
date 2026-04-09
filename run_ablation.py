"""
AIGIS — Ablation Study
======================
Compares four conditions to isolate the contribution of the Contract Net
Protocol (CNP) and the panic cognitive model:

  Condition A — BASELINE:         CNP + panic model both active (default)
  Condition B — DISABLE_CNP:      Random task assignment; panic model active
  Condition C — DISABLE_PANIC:    CNP active; rational-agent baseline (no panic)
  Condition D — NO COORDINATION:  Both CNP and panic disabled (null baseline)

Academic grounding
------------------
Ablation design:
  Wilensky, U. & Rand, W. (2015).
  An Introduction to Agent-Based Modeling.
  MIT Press. ISBN 978-0-262-73189-8.
  [Section 8.3: ablation / component removal as standard ABM validation technique.]

CNP baseline (Condition B):
  Smith, R.G. (1980). "The Contract Net Protocol: High-Level Communication
  and Control in a Distributed Problem Solver."
  IEEE Transactions on Computers, C-29(12), pp. 1104–1113.
  DOI: 10.1109/TC.1980.1675516
  [CNP improves task allocation over random assignment by minimising bid cost.]

Panic model baseline (Condition C):
  Cova, T.J. & Johnson, J.P. (2002). "Microsimulation of neighborhood-level
  evacuation in the urban-wildland interface."
  Environment and Planning A, 34(12), pp. 2211–2229.
  DOI: 10.1068/a34251
  [Three-state cognitive machine; disabling panic yields purely rational agents.]

Statistical comparison:
  Mann-Whitney U test (non-parametric, appropriate for bounded [0,1] variables
  that may not be normally distributed):
    Mann, H.B. & Whitney, D.R. (1947). "On a Test of Whether One of Two Random
    Variables is Stochastically Larger than the Other."
    The Annals of Mathematical Statistics, 18(1), pp. 50–60.
    DOI: 10.1214/aoms/1177730491

Effect size (rank-biserial correlation r):
  Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics.
  SAGE Publications. ISBN 978-1-4462-4918-5.
  [r = 0.1 small, 0.3 medium, 0.5 large — Section 6.5.3]

Usage
-----
  python run_ablation.py [--runs N] [--output FILE]

Outputs
-------
  - CSV:     ablation_results.csv
  - PNG:     ablation_plot.png
  - Console: comparison table with p-values, effect sizes, and verdict
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

import src.config as _cfg
from src.simulation import AIGISSimulation

warnings.filterwarnings('ignore')

OUTPUTS = ['mortality_rate', 'evacuation_success_rate', 'steps', 'avg_panic_level']
BG, PANEL, FG = 'white', '#f5f5f5', '#222222'

CONDITIONS = [
    # (label, DISABLE_CNP, DISABLE_PANIC, colour)
    ('Baseline (CNP + Panic)',      False, False, '#ffd60a'),
    ('No CNP (random assign)',      True,  False, '#ff006e'),
    ('No Panic (rational agents)',  False, True,  '#06d6a0'),
    # Condition D — both components disabled: minimal-intelligence baseline.
    # This null condition has no coordination protocol and no panic model,
    # representing an uncoordinated random-assignment system with rational
    # agents only.  Comparison against Baseline quantifies the combined
    # contribution of CNP + panic-aware modelling — directly supporting the
    # thesis claim that coordinated MAS outperforms uncoordinated response.
    # Methodology: Wilensky & Rand (2015) §8.3 — multiple component removal
    # produces a lower bound on system performance.
    ('No Coordination (No CNP + No Panic)', True, True, '#8b5cf6'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_condition(label: str, disable_cnp: bool, disable_panic: bool,
                   num_runs: int, lat: float, lon: float, radius: int) -> pd.DataFrame:
    """
    Run `num_runs` simulations under one ablation condition.
    Patches _cfg directly so AIGISSimulation picks up flags at construction.
    """
    _cfg.DISABLE_CNP   = disable_cnp
    _cfg.DISABLE_PANIC = disable_panic

    rows = []
    for i in range(num_runs):
        sim = AIGISSimulation(lat=lat, lon=lon, radius=radius, mode='batch', run_id=i)
        result = sim.run_until_complete()
        rows.append({k: result[k] for k in OUTPUTS})

    _cfg.DISABLE_CNP   = False
    _cfg.DISABLE_PANIC = False

    df = pd.DataFrame(rows)
    df['condition'] = label
    return df


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation as effect size for Mann-Whitney U.
    Field (2013), Section 6.5.3:  r = 1 - (2U)/(n1*n2)
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def _sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation(
    num_runs: int = 30,
    lat: float = 38.090, lon: float = 23.920, radius: int = 3000,
    output_file: str = 'ablation_results.csv',
) -> pd.DataFrame:
    """
    Execute ablation study across four conditions (A=Baseline, B=No CNP, C=No Panic, D=No Coordination).

    30 runs per condition follows Grimm et al. (2020) ODD Protocol minimum
    for characterising stochastic ABM output distributions.
    """
    print('=' * 70)
    print('AIGIS — Ablation Study')
    print('=' * 70)
    print('Wilensky & Rand (2015)  |  Smith (1980)  |  Cova & Johnson (2002)')
    print(f'Conditions: {len(CONDITIONS)}  |  Runs per condition: {num_runs}')
    print(f'Statistical test: Mann-Whitney U  (Mann & Whitney 1947)')
    print('=' * 70 + '\n')

    all_dfs = []
    for label, dcnp, dpanic, _ in CONDITIONS:
        print(f'Running: {label} ...')
        df = _run_condition(label, dcnp, dpanic, num_runs, lat, lon, radius)
        all_dfs.append(df)
        print(f'  mort={df["mortality_rate"].mean():.3%}  '
              f'evac={df["evacuation_success_rate"].mean():.3%}  '
              f'steps={df["steps"].mean():.1f}')

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f'\nResults saved to: {output_file}')

    _print_comparison_table(df_all)
    _plot_ablation(df_all, output_file.replace('.csv', '.png'))
    return df_all


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print mean ± std per condition, then pairwise Mann-Whitney U vs. Baseline.
    Mann & Whitney (1947); effect size from Field (2013).
    """
    print('\n' + '=' * 70)
    print('ABLATION RESULTS')
    print('=' * 70)

    labels = [c[0] for c in CONDITIONS]
    baseline_label = labels[0]

    for out in OUTPUTS:
        print(f'\n{out}:')
        for label in labels:
            sub  = df[df['condition'] == label][out]
            mean = sub.mean()
            std  = sub.std()
            print(f'  {label:<40} {mean:.4f} ± {std:.4f}', end='')

            # Mann-Whitney U vs. Baseline (skip baseline vs. itself)
            if label != baseline_label:
                base = df[df['condition'] == baseline_label][out]
                u, p = stats.mannwhitneyu(sub, base, alternative='two-sided')
                r    = _rank_biserial(u, len(sub), len(base))
                stars = _sig_stars(p)
                print(f'   p={p:.4f} {stars}  r={r:.3f}', end='')
            print()

    print('\n' + '=' * 70)
    print('Significance: *** p<0.001  ** p<0.01  * p<0.05  ns not significant')
    print('Effect size r (Field 2013): |r|≥0.5 large, ≥0.3 medium, ≥0.1 small')
    print('=' * 70)
    print("""
Interpretation guide:
  Condition B (No CNP) vs. Baseline — significant increase in mortality_rate
    → confirms CNP improves resource allocation efficiency (Smith 1980)
  Condition C (No Panic) vs. Baseline — significant decrease in mortality_rate
    → confirms panic model raises casualties, validating Cova & Johnson (2002)
  Condition D (No Coordination) vs. Baseline — largest expected mortality gap
    → lower bound on system performance; directly supports thesis claim that
      coordinated MAS outperforms uncoordinated random-assignment response
  Non-significant results indicate the component has marginal impact under
    these simulation parameters.
""")


def _plot_ablation(df: pd.DataFrame, out_path: str) -> None:
    """
    2×2 violin + strip plot for each output metric.
    Violin shows full distribution; strip overlays individual run values.
    """
    metrics = OUTPUTS
    n_metrics = len(metrics)
    colours = [c[3] for c in CONDITIONS]
    labels  = [c[0] for c in CONDITIONS]
    short   = ['Baseline', 'No CNP', 'No Panic', 'No Coord.']

    fig = plt.figure(figsize=(5 * n_metrics, 6), facecolor=BG)
    fig.suptitle(
        'Ablation Study  |  Wilensky & Rand (2015)  |  Mann-Whitney U  (Mann & Whitney 1947)',
        color=FG, fontsize=10, fontweight='bold'
    )
    gs = gridspec.GridSpec(1, n_metrics, figure=fig, wspace=0.35)

    for col_i, out in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        data = [df[df['condition'] == lbl][out].values for lbl in labels]

        parts = ax.violinplot(data, positions=range(len(labels)), widths=0.7,
                              showmeans=True, showextrema=True)
        for pc, col in zip(parts['bodies'], colours):
            pc.set_facecolor(col)
            pc.set_alpha(0.5)
        for attr in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
            if attr in parts:
                parts[attr].set_color(FG)
                parts[attr].set_linewidth(1)

        # Overlay individual points (jittered)
        for xi, (series, col) in enumerate(zip(data, colours)):
            jitter = np.random.uniform(-0.08, 0.08, size=len(series))
            ax.scatter(xi + jitter, series, color=col, s=10, alpha=0.6, zorder=5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, color=FG, fontsize=7, rotation=15, ha='right')
        ax.set_title(out.replace('_', '\n'), color=FG, fontsize=8, fontweight='bold')
        ax.set_ylabel('Value', color=FG, fontsize=7)

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f'Ablation plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ablation study: CNP and panic model contribution'
    )
    parser.add_argument('--runs',   type=int,   default=30,
                        help='Monte Carlo runs per condition (default: 30)')
    parser.add_argument('--output', type=str,   default='ablation_results.csv')
    parser.add_argument('--lat',    type=float, default=38.090)
    parser.add_argument('--lon',    type=float, default=23.920)
    parser.add_argument('--radius', type=int,   default=3000)
    args = parser.parse_args()

    run_ablation(
        num_runs=args.runs,
        lat=args.lat, lon=args.lon, radius=args.radius,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
