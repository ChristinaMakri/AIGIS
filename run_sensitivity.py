"""
AIGIS — One-at-a-Time Sensitivity Analysis
==========================================
Sweeps six key parameters individually, holding all others at baseline,
to determine which inputs most influence casualty and evacuation outcomes.

Methodology
-----------
One-at-a-time (OAT) sensitivity analysis:
  Saltelli, A., Ratto, M., Andres, T., et al. (2008).
  Global Sensitivity Analysis: The Primer.
  Wiley. ISBN 978-0-470-05997-5.
  [Chapter 1: OAT is standard for ABM feasibility / diagnostic analysis.]

Monte Carlo ensemble per parameter value:
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
  Other Simulation Models: A Second Update to Improve Clarity, Replication,
  and Structural Realism." JASSS 23(2):7. DOI: 10.18564/jasss.4259
  [Recommends ≥ 30 stochastic runs per configuration point.]

Parameters swept
----------------
  FIRE_SPREAD_PROB_BASE    — base ignition probability per step
  ROTHERMEL_BASE_ROS       — base rate of spread (m/s)
  NUM_CIVILIANS            — population size in simulation zone
  WIND_OSCILLATION_AMPLITUDE — wind gust magnitude (degrees)
  CIVILIAN_PANIC_RATIONAL  — panic threshold for Rational→Confused transition
  CIVILIAN_V_FREE_FLOW     — free-flow evacuation speed (cells/step)

Primary outputs monitored
--------------------------
  mortality_rate            — fraction of civilians killed
  evacuation_success_rate   — fraction safely evacuated
  steps                     — simulation duration

Usage
-----
  python run_sensitivity.py [--runs N] [--output FILE]

Outputs
-------
  - CSV: sensitivity_results.csv
  - PNG: sensitivity_plot.png (normalised sensitivity index bar chart)
"""
import argparse
import copy
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import src.config as _cfg
from src.simulation import AIGISSimulation

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Baseline configuration (Mati-neutral defaults from src/config.py)
# ---------------------------------------------------------------------------
BASELINE = {
    'FIRE_SPREAD_PROB_BASE':      0.30,
    'ROTHERMEL_BASE_ROS':         0.5,
    'NUM_CIVILIANS':              60,
    'WIND_OSCILLATION_AMPLITUDE': 5.0,
    'CIVILIAN_PANIC_RATIONAL':    0.3,
    'CIVILIAN_V_FREE_FLOW':       3.0,
}

# ---------------------------------------------------------------------------
# Sweep ranges — ±50 % around baseline in 5 steps (Saltelli 2008, Ch. 1)
# ---------------------------------------------------------------------------
SWEEPS = {
    'FIRE_SPREAD_PROB_BASE':      np.linspace(0.10, 0.60, 5),
    'ROTHERMEL_BASE_ROS':         np.linspace(0.20, 1.00, 5),
    'NUM_CIVILIANS':              [30, 45, 60, 80, 100],
    'WIND_OSCILLATION_AMPLITUDE': np.linspace(1.0, 15.0, 5),
    'CIVILIAN_PANIC_RATIONAL':    np.linspace(0.1, 0.6, 5),
    'CIVILIAN_V_FREE_FLOW':       np.linspace(1.0, 6.0, 5),
}

OUTPUTS   = ['mortality_rate', 'evacuation_success_rate', 'steps']
BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_override(param: str, value) -> None:
    """Patch src.config in-process so AIGISSimulation picks up the new value."""
    setattr(_cfg, param, value)


def _reset_to_baseline() -> None:
    for k, v in BASELINE.items():
        setattr(_cfg, k, v)


def _run_batch(num_runs: int, lat: float, lon: float, radius: int) -> pd.DataFrame:
    """
    Run `num_runs` simulations with current config values; return DataFrame.
    Each simulation uses a fresh AIGISSimulation instance so it reads the
    patched _cfg values at construction time.
    """
    rows = []
    for i in range(num_runs):
        sim = AIGISSimulation(lat=lat, lon=lon, radius=radius, mode='batch', run_id=i)
        result = sim.run_until_complete()
        rows.append({k: result[k] for k in OUTPUTS})
    return pd.DataFrame(rows)


def _sensitivity_index(baseline_mean: float, varied_mean: float,
                        baseline_val, varied_val) -> float:
    """
    Normalised sensitivity index S:
      S = (ΔOutput / Output_baseline) / (ΔInput / Input_baseline)
    Saltelli et al. (2008), Eq. 1.7.
    Returns NaN when inputs are identical or baseline is zero.
    """
    if baseline_val == 0 or baseline_mean == 0:
        return float('nan')
    d_out = (varied_mean - baseline_mean) / baseline_mean
    d_in  = (varied_val  - baseline_val)  / baseline_val
    if d_in == 0:
        return float('nan')
    return d_out / d_in


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    num_runs: int = 30,
    lat:      float = 38.090,
    lon:      float = 23.920,
    radius:   int   = 3000,
    output_file: str = 'sensitivity_results.csv',
) -> pd.DataFrame:
    """
    Execute OAT sweep and collect results.

    For each parameter p in SWEEPS:
      1. Reset all parameters to BASELINE
      2. Patch p to each sweep value v_i
      3. Run num_runs simulations → record mean of each output
    Then compute normalised sensitivity index for all (param, output) pairs.
    """
    print('=' * 70)
    print('AIGIS — One-at-a-Time Sensitivity Analysis')
    print('=' * 70)
    print(f'Saltelli et al. (2008) OAT methodology  |  Grimm et al. (2020) n≥30')
    print(f'Parameters: {len(SWEEPS)}  |  Values per param: 5  |  Runs per value: {num_runs}')
    total = sum(len(v) for v in SWEEPS.values()) * num_runs
    print(f'Total simulations: {total}')
    print('=' * 70 + '\n')

    all_rows = []

    for param, values in SWEEPS.items():
        baseline_val = BASELINE[param]
        print(f'Sweeping {param} (baseline={baseline_val}) ...')

        for vi, val in enumerate(values):
            _reset_to_baseline()
            _apply_override(param, val if not isinstance(val, np.integer)
                            else int(val))

            print(f'  value {vi+1}/{len(values)}: {val:.4g}', end='  ')
            df = _run_batch(num_runs, lat, lon, radius)

            for out_col in OUTPUTS:
                mean_val = df[out_col].mean()
                std_val  = df[out_col].std()
                si = _sensitivity_index(
                    baseline_mean=BASELINE.get(f'__bm_{out_col}', np.nan),
                    varied_mean=mean_val,
                    baseline_val=baseline_val,
                    varied_val=val,
                )
                all_rows.append({
                    'parameter':   param,
                    'value':       float(val),
                    'baseline':    float(baseline_val),
                    'output':      out_col,
                    'mean':        mean_val,
                    'std':         std_val,
                    'si':          si,
                })
            print(f"mort={df['mortality_rate'].mean():.3%}", flush=True)

        _reset_to_baseline()

    # -----------------------------------------------------------------------
    # Re-run baselines now that we have collected all varied runs,
    # recompute SI properly
    # -----------------------------------------------------------------------
    print('\nRunning baseline ensemble ...')
    _reset_to_baseline()
    baseline_df = _run_batch(num_runs, lat, lon, radius)
    baseline_means = {c: baseline_df[c].mean() for c in OUTPUTS}

    # Recompute SI using correct baseline means
    for row in all_rows:
        param = row['parameter']
        bv    = row['baseline']
        vv    = row['value']
        out   = row['output']
        row['si'] = _sensitivity_index(
            baseline_mean=baseline_means[out],
            varied_mean=row['mean'],
            baseline_val=bv,
            varied_val=vv,
        )

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(output_file, index=False)
    print(f'\nResults saved to: {output_file}')

    _print_sensitivity_table(df_all, baseline_means)
    _plot_sensitivity(df_all, output_file.replace('.csv', '.png'))

    _reset_to_baseline()
    return df_all


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_sensitivity_table(df: pd.DataFrame, baseline_means: dict) -> None:
    print('\n' + '=' * 70)
    print('SENSITIVITY INDEX TABLE  (Saltelli et al. 2008, Eq. 1.7)')
    print('=' * 70)
    print(f'{"Parameter":<35} {"Output":<28} {"Max |SI|":>8}')
    print('-' * 70)

    for param in SWEEPS:
        sub = df[df['parameter'] == param]
        for out in OUTPUTS:
            s = sub[sub['output'] == out]
            max_si = s['si'].abs().max()
            flag = ''
            if max_si > 1.0:
                flag = ' *** HIGH'
            elif max_si > 0.5:
                flag = ' **  MODERATE'
            print(f'  {param:<33} {out:<28} {max_si:>8.3f}{flag}')
    print('=' * 70)
    print('Baseline means:', {k: f'{v:.4f}' for k, v in baseline_means.items()})


def _plot_sensitivity(df: pd.DataFrame, out_path: str) -> None:
    """
    3-row × 6-col grid: one column per parameter, one row per output metric.
    Each cell shows output mean vs. parameter value, with ±1σ shading.
    """
    params = list(SWEEPS.keys())
    n_params = len(params)

    fig = plt.figure(figsize=(4 * n_params, 4 * len(OUTPUTS)), facecolor=BG)
    fig.suptitle(
        'One-at-a-Time Sensitivity Analysis  |  Saltelli et al. (2008)\n'
        f'Grimm et al. (2020) n≥30 Monte Carlo runs per point',
        color=FG, fontsize=10, fontweight='bold'
    )
    gs = gridspec.GridSpec(len(OUTPUTS), n_params, figure=fig,
                           hspace=0.45, wspace=0.35)

    colours = ['#ff006e', '#06d6a0', '#ffd60a']

    for row_i, out in enumerate(OUTPUTS):
        for col_i, param in enumerate(params):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor('#3a3a5c')

            sub = df[(df['parameter'] == param) & (df['output'] == out)]
            xs  = sub['value'].values
            ys  = sub['mean'].values
            es  = sub['std'].values

            ax.plot(xs, ys, color=colours[row_i], linewidth=1.5, marker='o',
                    markersize=4)
            ax.fill_between(xs, ys - es, ys + es,
                            color=colours[row_i], alpha=0.20)
            ax.axvline(BASELINE[param], color='white', linestyle='--',
                       linewidth=1.0, alpha=0.6, label='baseline')

            if col_i == 0:
                ax.set_ylabel(out.replace('_', '\n'), color=FG, fontsize=7)
            if row_i == 0:
                ax.set_title(param.replace('_', '\n'), color=FG, fontsize=7,
                             fontweight='bold')
            ax.set_xlabel('', color=FG)

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f'Sensitivity plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='One-at-a-time sensitivity analysis for AIGIS'
    )
    parser.add_argument('--runs',   type=int,   default=30,
                        help='Monte Carlo runs per parameter value (default: 30)')
    parser.add_argument('--output', type=str,   default='sensitivity_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--lat',    type=float, default=38.090)
    parser.add_argument('--lon',    type=float, default=23.920)
    parser.add_argument('--radius', type=int,   default=3000)
    args = parser.parse_args()

    run_sensitivity(
        num_runs=args.runs,
        lat=args.lat, lon=args.lon, radius=args.radius,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
