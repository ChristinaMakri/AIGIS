"""
AIGIS — Runtime Benchmark
==========================
Measures wall-clock time per simulation run across all 32 scenarios and
produces a compact performance table for the thesis.

Why this step is included
-------------------------
Most disaster-response and evacuation simulation papers targeting Safety Science
or Natural Hazards do not include computational benchmarks, but the systematic
review of 134 pedestrian ABM studies (Ronchi & Nilsson 2024 IJDRR) identifies
computational complexity as a persistent challenge and notes it is rarely
quantified.  Including a runtime table:
  (a) provides reproducibility information for readers who wish to replicate the
      experiments on their own hardware;
  (b) demonstrates that AIGIS is tractable for the 640–7168 total simulation
      runs needed by the thesis pipeline; and
  (c) distinguishes AIGIS from papers that claim scalability without evidence.

For HPC-scale benchmarks with strong/weak scaling see:
  Richmond et al. (2023) FLAME GPU 2 — GPU vs CPU at varying agent counts.
  Gutenschwager et al. (2018) Scalable HPC Evacuation — 2048-process strong scaling.
AIGIS targets single-machine CPU execution; we report median wall-clock time and
inter-run coefficient of variation (CV) as the two key metrics.

References
----------
  Ronchi, E. & Nilsson, D. (2024). "Agent-based simulation for pedestrian
    evacuation: A systematic literature review."
    International Journal of Disaster Risk Reduction, 100, 104194.
    DOI: 10.1016/j.ijdrr.2024.104194.
    [Identified computational complexity as an under-reported challenge in
     134 pedestrian ABM studies.]

  Richmond, P. et al. (2023). "FLAME GPU 2: A framework for flexible and
    performant agent based simulation on GPUs."
    Software: Practice and Experience, 53(8), pp. 1564-1596.
    DOI: 10.1002/spe.3207.
    [GPU vs CPU wall-clock benchmarks at varying agent counts — motivates
     reporting runtime alongside simulation results.]

  Gutenschwager, K. et al. (2018). "Scalable HPC Enhanced Agent Based System
    for Simulating Mixed Mode Evacuation of Large Urban Areas."
    Procedia Computer Science, 130, pp. 402-409.
    DOI: 10.1016/j.procs.2018.04.059.
    [Strong scalability to 2048 processes — runtime per step as primary metric.]

Usage
-----
  python benchmark_runtime.py [--runs-per-scenario N] [--output FILE]

  Default: 5 runs per scenario (32 scenarios = 160 total runs).
  Each run is timed with time.perf_counter() for sub-millisecond resolution.

Outputs
-------
  CSV : per-scenario median time (s), std, CV, agent count, grid size
  PNG : horizontal bar chart sorted by median runtime
"""
from __future__ import annotations
import argparse
import contextlib
import io
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.config as _cfg
from src.simulation import AIGISSimulation
from train_models import TRAINING_LOCATIONS

warnings.filterwarnings('ignore')

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'

# ---------------------------------------------------------------------------
# Held-out scenarios (same parameters as validate_*.py / evaluate_marl.py)
# ---------------------------------------------------------------------------
HELD_OUT_SCENARIOS = [
    {'name': 'Mati 2018 (held-out)',
     'lat': 38.090, 'lon': 23.920, 'radius': 3000,
     'fire_locations': [(38.097, 23.940), (38.092, 23.932)],
     'params': {'WIND_SPEED': 11.0, 'NUM_CIVILIANS': 60}},
    {'name': 'Camp Fire 2018 (held-out)',
     'lat': 39.746, 'lon': -121.621, 'radius': 4000,
     'fire_locations': [(39.792, -121.437), (39.765, -121.501)],
     'params': {'WIND_SPEED': 14.0, 'NUM_CIVILIANS': 90}},
    {'name': 'Pedrogao Grande 2017 (held-out)',
     'lat': 39.921, 'lon': -8.133, 'radius': 3500,
     'fire_locations': [(39.938, -8.152)],
     'params': {'WIND_SPEED': 10.0, 'NUM_CIVILIANS': 50}},
    {'name': 'Alexandroupoli 2023 (held-out)',
     'lat': 40.820, 'lon': 26.050, 'radius': 5000,
     'fire_locations': [(40.860, 26.100)],
     'params': {'WIND_SPEED': 13.0, 'NUM_CIVILIANS': 100}},
    {'name': 'Lahaina 2023 (held-out)',
     'lat': 20.878, 'lon': -156.679, 'radius': 2000,
     'fire_locations': [(20.889, -156.669)],
     'params': {'WIND_SPEED': 27.0, 'NUM_CIVILIANS': 100}},
    {'name': 'Black Saturday 2009 (held-out)',
     'lat': -37.627, 'lon': 145.373, 'radius': 4000,
     'fire_locations': [(-37.600, 145.410)],
     'params': {'WIND_SPEED': 18.0, 'NUM_CIVILIANS': 90}},
    {'name': 'Tubbs Fire 2017 (held-out)',
     'lat': 38.531, 'lon': -122.694, 'radius': 3500,
     'fire_locations': [(38.560, -122.660)],
     'params': {'WIND_SPEED': 25.0, 'NUM_CIVILIANS': 70}},
    {'name': 'Peloponnese 2007 (held-out)',
     'lat': 37.488, 'lon': 21.632, 'radius': 3500,
     'fire_locations': [(37.510, 21.660)],
     'params': {'WIND_SPEED': 14.0, 'NUM_CIVILIANS': 55}},
    {'name': 'Valparaiso 2014 (held-out)',
     'lat': -33.036, 'lon': -71.628, 'radius': 2500,
     'fire_locations': [(-33.020, -71.600)],
     'params': {'WIND_SPEED': 12.0, 'NUM_CIVILIANS': 70}},
]


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _run_timed(lat, lon, radius, fire_locations, params, max_steps) -> float:
    """Run one simulation and return wall-clock seconds (time.perf_counter)."""
    orig = {}
    for k, v in params.items():
        orig[k] = getattr(_cfg, k, None)
        setattr(_cfg, k, v)

    t0 = time.perf_counter()
    with _quiet():
        sim = AIGISSimulation(lat=lat, lon=lon, radius=radius,
                              mode='batch', run_id=0,
                              fire_locations=fire_locations)
        while sim.step < max_steps and not sim.is_complete():
            sim.run_step()
    elapsed = time.perf_counter() - t0

    for k, v in orig.items():
        if v is not None:
            setattr(_cfg, k, v)

    return elapsed


def benchmark(
    runs_per_scenario: int = 5,
    output_file: str = 'benchmark_runtime.csv',
) -> pd.DataFrame:
    """
    Run N timed simulations per scenario and report median wall-clock time.

    Uses time.perf_counter() for sub-millisecond resolution (Python 3.3+).
    Coefficient of variation (CV = std/mean) quantifies run-to-run variability
    caused by stochastic fire spread and agent behaviour.
    """
    print('=' * 70)
    print('AIGIS — Runtime Benchmark')
    print('=' * 70)
    print('Ronchi & Nilsson (2024) IJDRR  |  Richmond et al. (2023) FLAME GPU 2')
    print(f'Runs per scenario: {runs_per_scenario}  |  '
          f'Scenarios: 32 (23 training + 9 held-out)')
    print(f'Total timed runs: {runs_per_scenario * 32}')
    print('=' * 70 + '\n')

    max_steps = _cfg.MAX_STEPS
    records = []
    total_scenarios = len(TRAINING_LOCATIONS) + len(HELD_OUT_SCENARIOS)
    done = 0

    all_scenarios = []
    for loc in TRAINING_LOCATIONS:
        all_scenarios.append({
            'name':           loc['name'],
            'lat':            loc['lat'],
            'lon':            loc['lon'],
            'radius':         loc['radius'],
            'fire_locations': loc.get('fire_locations'),
            'params':         {k: v for k, v in loc.items()
                               if k in ('WIND_SPEED', 'NUM_CIVILIANS',
                                        'FIRE_SPREAD_PROB_BASE', 'ROTHERMEL_BASE_ROS')},
            'split':          'training',
        })
    for sc in HELD_OUT_SCENARIOS:
        all_scenarios.append({
            'name':           sc['name'],
            'lat':            sc['lat'],
            'lon':            sc['lon'],
            'radius':         sc['radius'],
            'fire_locations': sc.get('fire_locations'),
            'params':         sc.get('params', {}),
            'split':          'held-out',
        })

    for sc in all_scenarios:
        done += 1
        print(f'  [{done}/{total_scenarios}] {sc["name"]}', end=' ', flush=True)
        times = []
        for r in range(runs_per_scenario):
            t = _run_timed(
                lat=sc['lat'], lon=sc['lon'], radius=sc['radius'],
                fire_locations=sc['fire_locations'],
                params=sc['params'],
                max_steps=max_steps,
            )
            times.append(t)
            print('.', end='', flush=True)
        print(f'  median={np.median(times):.2f}s')

        records.append({
            'scenario':           sc['name'],
            'split':              sc['split'],
            'radius_m':           sc['radius'],
            'num_civilians':      sc['params'].get('NUM_CIVILIANS',
                                                   _cfg.NUM_CIVILIANS),
            'max_steps':          max_steps,
            'runs':               runs_per_scenario,
            'median_time_s':      float(np.median(times)),
            'mean_time_s':        float(np.mean(times)),
            'std_time_s':         float(np.std(times)),
            'cv':                 float(np.std(times) / np.mean(times))
                                  if np.mean(times) > 0 else float('nan'),
            'min_time_s':         float(np.min(times)),
            'max_time_s':         float(np.max(times)),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f'\nBenchmark results saved to: {output_file}\n')

    _print_summary(df)
    _plot_benchmark(df, output_file.replace('.csv', '.png'))

    return df


def _print_summary(df: pd.DataFrame) -> None:
    print('=' * 70)
    print('RUNTIME SUMMARY')
    print('-' * 70)
    print(f"{'Scenario':<38} {'Split':<10} {'Median(s)':>9} {'CV':>6}")
    print('-' * 70)
    for _, row in df.sort_values('median_time_s', ascending=False).iterrows():
        print(f"  {row['scenario']:<36} {row['split']:<10} "
              f"{row['median_time_s']:>9.2f} {row['cv']:>6.3f}")
    print('-' * 70)
    print(f"  Overall median: {df['median_time_s'].median():.2f} s/run")
    print(f"  Overall range : {df['median_time_s'].min():.2f} – "
          f"{df['median_time_s'].max():.2f} s/run")
    total_pipeline = (
        df['median_time_s'].sum() * (
            # approximate total runs from thesis pipeline
            50 + 50 * 9 + 20 * 32 + 50 * 32
        ) / 32
    )
    print(f"  Estimated total pipeline wall-clock (approx): "
          f"{total_pipeline / 3600:.1f} hours (CPU, single thread)")
    print('=' * 70)


def _plot_benchmark(df: pd.DataFrame, out_path: str) -> None:
    """
    Horizontal bar chart of median runtime per scenario, sorted descending.
    Colour distinguishes training (blue) from held-out (orange) scenarios.
    Ronchi & Nilsson (2024) — runtime table as a thesis transparency measure.
    """
    df_sorted = df.sort_values('median_time_s')
    colours = ['#fb5607' if s == 'held-out' else '#8338ec'
               for s in df_sorted['split']]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df_sorted) * 0.28)),
                           facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#3a3a5c')

    y = np.arange(len(df_sorted))
    bars = ax.barh(y, df_sorted['median_time_s'], color=colours, alpha=0.85)

    # Error bars (std)
    ax.barh(y, df_sorted['std_time_s'],
            left=df_sorted['median_time_s'] - df_sorted['std_time_s'] / 2,
            color='white', alpha=0.25, height=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted['scenario'], color=FG, fontsize=6)
    ax.set_xlabel('Median wall-clock time (s)', color=FG, fontsize=9)
    ax.set_title(
        'AIGIS Runtime Benchmark — median wall-clock per simulation run\n'
        'Ronchi & Nilsson (2024) IJDRR  |  Richmond et al. (2023) FLAME GPU 2',
        color=FG, fontsize=9, fontweight='bold',
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8338ec', alpha=0.85, label='Training scenario'),
        Patch(facecolor='#fb5607', alpha=0.85, label='Held-out scenario'),
    ]
    ax.legend(handles=legend_elements, fontsize=7,
              facecolor=PANEL, labelcolor=FG, loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Benchmark plot saved to: {out_path}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description='Benchmark AIGIS wall-clock runtime across 32 scenarios'
    )
    p.add_argument('--runs-per-scenario', type=int, default=5,
                   help='Timed runs per scenario (default: 5)')
    p.add_argument('--output', type=str, default='benchmark_runtime.csv')
    args = p.parse_args()

    benchmark(
        runs_per_scenario=args.runs_per_scenario,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
