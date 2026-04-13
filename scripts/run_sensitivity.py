"""
AIGIS — Sobol Global Sensitivity Analysis
==========================================
Replaces the previous one-at-a-time (OAT) method with Sobol variance-based
global sensitivity analysis, which correctly handles nonlinear interactions
between parameters.

Methodology
-----------
Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., &
Tarantola, S. (2010). "Variance based sensitivity analysis of model output.
Design and estimator for the total sensitivity index." Computer Physics
Communications, 181(2), pp. 259-270.  DOI: 10.1016/j.cpc.2009.09.018.

The Saltelli (2010) estimator is the operational standard for ABM
sensitivity analysis and outperforms OAT for nonlinear/non-monotone models:
  Saltelli, A., Ratto, M., Andres, T., et al. (2008). Global Sensitivity
  Analysis: The Primer. Wiley.  ISBN 978-0-470-05997-5.  [Chapter 4.]

Indices computed
----------------
  Si   — first-order Sobol index: direct variance contribution of parameter i
  STi  — total-effect index: direct + interaction contributions
  Si_conf, STi_conf — 95% bootstrap confidence intervals

Sample generation (Saltelli sampler)
--------------------------------------
N × (2D + 2) model evaluations where N is the base sample size and D is
the number of parameters.  N = 512 gives 512 × (2×6 + 2) = 7168 evaluations,
which satisfies the N ≥ 500 recommendation for D = 6 parameters
(Saltelli et al. 2010, Section 3.2).  Pass --workers <cpu_count> to
parallelise across cores and reduce wall-clock time proportionally.

Parameters analysed
-------------------
  FIRE_SPREAD_PROB_BASE    — base ignition probability per step [0.10, 0.60]
  ROTHERMEL_BASE_ROS       — base rate of spread (m/s)          [0.20, 1.00]
  NUM_CIVILIANS            — population size                     [30, 100]
  WIND_OSCILLATION_AMPLITUDE — wind gust magnitude (degrees)    [1.0, 15.0]
  CIVILIAN_PANIC_RATIONAL  — panic threshold Rational→Confused  [0.1, 0.6]
  CIVILIAN_V_FREE_FLOW     — free-flow evacuation speed         [1.0, 6.0]

Outputs monitored
-----------------
  mortality_rate            — fraction of civilians killed
  evacuation_success_rate   — fraction safely evacuated
  steps                     — simulation duration

Grimm et al. (2020) ODD Protocol recommends Sobol/variance-based sensitivity
for ABMs (Section 7.6 "Sensitivity Analysis"):
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
  Other Simulation Models." JASSS 23(2):7. DOI: 10.18564/jasss.4259.

Usage
-----
  python run_sensitivity.py [--N 128] [--output FILE]

Outputs
-------
  - CSV: sensitivity_results.csv  (Si, STi, confidence intervals)
  - PNG: sensitivity_plot.png     (Si / STi bar chart per output)
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import os
import warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from SALib.sample import saltelli
from SALib.analyze import sobol

import src.config as _cfg

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Problem definition (Saltelli et al. 2010 notation)
# ---------------------------------------------------------------------------
#
# Parameter bounds chosen to span the physically plausible range for
# Mediterranean wildfire conditions:
#   FIRE_SPREAD_PROB_BASE:     [0.10, 0.60]  — wet-spring to extreme-summer
#   ROTHERMEL_BASE_ROS:        [0.20, 1.00]  — light grass to dense shrub
#   NUM_CIVILIANS:             [30, 100]     — hamlet to small village
#   WIND_OSCILLATION_AMPLITUDE:[1.0, 15.0]  — calm to gusty Meltemi
#   CIVILIAN_PANIC_RATIONAL:   [0.1, 0.6]   — literature range from PADM
#     (Lindell & Perry 2012 — panic thresholds vary across population segments)
#   CIVILIAN_V_FREE_FLOW:      [1.0, 6.0]   — elderly walkers to healthy adults
#     (Cova & Johnson 2002 report 1.2–4.5 km/h foot speeds in WUI evacuations)
PROBLEM = {
    'num_vars': 6,
    'names': [
        'FIRE_SPREAD_PROB_BASE',
        'ROTHERMEL_BASE_ROS',
        'NUM_CIVILIANS',
        'WIND_OSCILLATION_AMPLITUDE',
        'CIVILIAN_PANIC_RATIONAL',
        'CIVILIAN_V_FREE_FLOW',
    ],
    'bounds': [
        [0.10, 0.60],   # FIRE_SPREAD_PROB_BASE
        [0.20, 1.00],   # ROTHERMEL_BASE_ROS
        [30,   100],    # NUM_CIVILIANS
        [1.0,  15.0],   # WIND_OSCILLATION_AMPLITUDE
        [0.1,  0.6],    # CIVILIAN_PANIC_RATIONAL
        [1.0,  6.0],    # CIVILIAN_V_FREE_FLOW
    ],
}

OUTPUTS = ['mortality_rate', 'evacuation_success_rate', 'steps']

BG, PANEL, FG = 'white', '#f5f5f5', '#222222'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple:
    """
    Self-contained worker for parallel execution.

    Each subprocess (fork or spawn) gets its own copy of src.config, so
    patching it here is safe — no shared state with other workers.

    Args:
        args: (sample_row, lat, lon, radius, run_id)

    Returns:
        (run_id, {output_name: value, ...})
    """
    sample_row, lat, lon, radius, run_id = args
    import src.config as _local_cfg
    from src.simulation import AIGISSimulation

    for i, name in enumerate(PROBLEM['names']):
        val = sample_row[i]
        if name == 'NUM_CIVILIANS':
            val = int(round(val))
        setattr(_local_cfg, name, val)

    sim = AIGISSimulation(lat=lat, lon=lon, radius=radius, mode='batch', run_id=run_id)
    result = sim.run_until_complete()
    return run_id, {k: result[k] for k in OUTPUTS}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    N:           int   = 512,
    lat:         float = 38.090,
    lon:         float = 23.920,
    radius:      int   = 3000,
    output_file: str   = 'sensitivity_results.csv',
    workers:     int   = 1,
) -> pd.DataFrame:
    """
    Execute Sobol global sensitivity analysis.

    Steps (Saltelli et al. 2010):
      1. Generate Saltelli sample matrix  (N*(2D+2) rows)
      2. Evaluate model at each sample point (parallel if workers > 1)
      3. Compute Sobol Si and STi for each (parameter, output) pair
      4. Report and plot results

    Args:
        N:           base sample size (Saltelli et al. 2010 recommend N >= 500
                     for D = 6 parameters)
        lat, lon:    simulation centre (default: Mati, Greece)
        radius:      map radius in metres
        output_file: CSV output path
        workers:     number of parallel worker processes (default 1 = serial)
    """
    n_params = PROBLEM['num_vars']
    total_runs = N * (2 * n_params + 2)

    print('=' * 70)
    print('AIGIS — Sobol Global Sensitivity Analysis')
    print('=' * 70)
    print('Saltelli, A. et al. (2010) variance-based estimator.')
    print('Grimm et al. (2020) ODD §7.6 — recommended for nonlinear ABMs.')
    print(f'N = {N}  |  D = {n_params}  |  Total model runs = {total_runs}  |  Workers = {workers}')
    print('=' * 70 + '\n')

    # ---- Step 1: Saltelli sample matrix -----------------------------------
    # Saltelli (2010) quasi-random sampling gives a well-distributed coverage
    # of parameter space and ensures unbiased Si/STi estimates.
    param_values = saltelli.sample(PROBLEM, N, calc_second_order=False)
    # shape: (N*(2D+2), D)

    # ---- Step 2: Model evaluations ----------------------------------------
    args_list = [
        (param_values[i], lat, lon, radius, i)
        for i in range(len(param_values))
    ]

    Y = {out: np.zeros(len(param_values)) for out in OUTPUTS}
    completed = 0

    if workers > 1:
        with Pool(processes=workers) as pool:
            for run_id, res in pool.imap_unordered(_worker, args_list):
                completed += 1
                if completed % 50 == 0 or completed == 1:
                    print(f'  Run {completed}/{total_runs}', end='\r', flush=True)
                for out in OUTPUTS:
                    Y[out][run_id] = res[out]
    else:
        for run_id, res in (_worker(a) for a in args_list):
            completed += 1
            if completed % 50 == 0 or completed == 1:
                print(f'  Run {completed}/{total_runs}', end='\r', flush=True)
            for out in OUTPUTS:
                Y[out][run_id] = res[out]

    print(f'  Run {total_runs}/{total_runs} — done.              ')

    # ---- Step 3: Sobol analysis -------------------------------------------
    all_rows = []
    si_results = {}
    for out in OUTPUTS:
        si = sobol.analyze(PROBLEM, Y[out], calc_second_order=False, print_to_console=False)
        si_results[out] = si
        for j, name in enumerate(PROBLEM['names']):
            all_rows.append({
                'parameter': name,
                'output':    out,
                'S1':        float(si['S1'][j]),
                'S1_conf':   float(si['S1_conf'][j]),
                'ST':        float(si['ST'][j]),
                'ST_conf':   float(si['ST_conf'][j]),
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(output_file, index=False)
    print(f'\nResults saved to: {output_file}')

    _print_sensitivity_table(df)
    _plot_sensitivity(df, output_file.replace('.csv', '.png'))

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_sensitivity_table(df: pd.DataFrame) -> None:
    print('\n' + '=' * 80)
    print('SOBOL SENSITIVITY INDICES  (Saltelli et al. 2010)')
    print('S1 = first-order (direct)  |  ST = total-effect (incl. interactions)')
    print('=' * 80)
    print(f'{"Parameter":<35} {"Output":<28} {"S1":>6} {"ST":>6}  Flag')
    print('-' * 80)

    for _, row in df.iterrows():
        flag = ''
        if row['ST'] > 0.5:
            flag = '*** HIGH'
        elif row['ST'] > 0.2:
            flag = '**  MOD'
        elif row['ST'] > 0.05:
            flag = '*   LOW'
        print(f"  {row['parameter']:<33} {row['output']:<28} "
              f"{row['S1']:>6.3f} {row['ST']:>6.3f}  {flag}")
    print('=' * 80)
    print('Interaction effects = ST - S1.  Large (ST - S1) indicates')
    print('parameter behaves differently depending on others.')


def _plot_sensitivity(df: pd.DataFrame, out_path: str) -> None:
    """
    One subplot per output metric — S1 (first-order) vs ST (total-effect)
    grouped bar chart.

    Visualization follows Saltelli et al. (2010) Fig. 1 convention:
      Blue bars = S1 (direct variance fraction)
      Orange bars = ST (total including interactions)
    """
    params  = PROBLEM['names']
    n_out   = len(OUTPUTS)
    x       = np.arange(len(params))
    width   = 0.35

    fig = plt.figure(figsize=(14, 4 * n_out), facecolor=BG)
    fig.suptitle(
        'Sobol Global Sensitivity Analysis  |  Saltelli et al. (2010)\n'
        'S1 = first-order  |  ST = total-effect',
        color=FG, fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(n_out, 1, figure=fig, hspace=0.55)

    colours_s1 = '#4cc9f0'
    colours_st = '#f77f00'

    for row_i, out in enumerate(OUTPUTS):
        ax = fig.add_subplot(gs[row_i, 0])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        sub  = df[df['output'] == out]
        s1s  = sub['S1'].values
        sts  = sub['ST'].values
        s1c  = sub['S1_conf'].values
        stc  = sub['ST_conf'].values

        ax.bar(x - width / 2, s1s, width, color=colours_s1, alpha=0.85,
               label='S1 (first-order)', yerr=s1c, capsize=3,
               error_kw={'color': FG, 'linewidth': 0.8})
        ax.bar(x + width / 2, sts, width, color=colours_st, alpha=0.85,
               label='ST (total-effect)', yerr=stc, capsize=3,
               error_kw={'color': FG, 'linewidth': 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', '\n') for p in params], fontsize=7)
        ax.set_ylabel(out.replace('_', '\n'), color=FG, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.axhline(0, color=FG, linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=FG, loc='upper right')

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f'Sensitivity plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sobol global sensitivity analysis for AIGIS'
    )
    parser.add_argument('--N',       type=int,   default=512,
                        help='Saltelli base sample size N (total runs = N*(2D+2), default 512)')
    parser.add_argument('--output',  type=str,   default='sensitivity_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--lat',     type=float, default=38.090)
    parser.add_argument('--lon',     type=float, default=23.920)
    parser.add_argument('--radius',  type=int,   default=3000)
    parser.add_argument('--workers', type=int,   default=os.cpu_count(),
                        help='Parallel worker processes (default: all CPU cores)')
    args = parser.parse_args()

    run_sensitivity(
        N=args.N,
        lat=args.lat, lon=args.lon, radius=args.radius,
        output_file=args.output,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
