"""
AIGIS — Hybrid BDI+RL Evaluation
==================================
Evaluates the trained hybrid system on:
  1. All 15 training scenarios (in-distribution check, one per curriculum phase)
  2. 9 held-out real-incident scenarios (OOD generalisation):
       Mati 2018, Camp Fire 2018, Pedrogao Grande 2017, Alexandroupoli 2023,
       Lahaina 2023, Black Saturday 2009, Tubbs Fire 2017, Peloponnese 2007,
       Valparaiso 2014

Reports: mean, std, 95% CI across N independent runs per scenario.
  Bengio, Y. et al. (2009). "Curriculum learning." ICML.
  Schulman, J. et al. (2017). "Proximal Policy Optimization." arXiv:1707.06347.
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
    Other Simulation Models." JASSS, 23(2), 7.
  Filippi, J.B. et al. (2016). "Representation and evaluation of wildfire
    simulations." Environmental Modelling & Software, 80, pp. 262-276.

Usage
-----
  python evaluate_marl.py [--runs N] [--output FILE]
"""
from __future__ import annotations
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import src.config as _cfg
from train_marl import run_episode
from src.rl.ppo import PPOAgent
from src.rl.curriculum import SCENARIOS

warnings.filterwarnings('ignore')

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'

# ---------------------------------------------------------------------------
# Held-out scenarios — never seen during MARL training.
# Each entry mirrors the corresponding validate_*.py documented conditions.
# ---------------------------------------------------------------------------
HELD_OUT = [
    {
        # Lagouvardos et al. (2019) BAMS 100(11):2243-2257
        # Copernicus EMSR249 | 102 fatalities / ~6,000 population = 1.70 %
        'name': 'Mati 2018 (held-out)',
        'lat': 38.090, 'lon': 23.920, 'radius': 3000,
        'fire_locations': [(38.097, 23.940), (38.092, 23.932), (38.085, 23.925)],
        'params': {
            'WIND_SPEED': 11.0, 'WIND_INITIAL_DIRECTION': 295.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # CAL FIRE (2020); NWS Sacramento (2018)
        # 85 fatalities / ~27,000 population = 0.31 %
        'name': 'Camp Fire 2018 (held-out)',
        'lat': 39.759, 'lon': -121.622, 'radius': 3000,
        'fire_locations': [(39.793, -121.575), (39.779, -121.593)],
        'params': {
            'WIND_SPEED': 16.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Viegas et al. (2017) ADAI/CEIF; Guerreiro et al. (2018)
        # Copernicus EMSR218 | 66 fatalities / ~7,500 population = 0.88 %
        'name': 'Pedrogao Grande 2017 (held-out)',
        'lat': 39.947, 'lon': -8.148, 'radius': 3000,
        'fire_locations': [(39.958, -8.137), (39.953, -8.143), (39.948, -8.150)],
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.95,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Copernicus EMSR689 (2023); Greek Fire Service (2023); EMY (2023)
        # Largest fire in EU history: ~81,000 ha
        # 20 fatalities / ~5,000 in study zone = 0.40 %
        'name': 'Alexandroupoli 2023 (held-out)',
        'lat': 41.049, 'lon': 26.357, 'radius': 3000,
        'fire_locations': [(41.061, 26.369), (41.055, 26.363), (41.049, 26.357)],
        'params': {
            'WIND_SPEED': 16.0, 'WIND_INITIAL_DIRECTION': 170.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # NFPA (2024); Maui County (2024); NOAA (2023); USFA (2024)
        # Hurricane Dora downburst; ENE wind 27 m/s; 100 fatalities
        # 100 fatalities / ~12,800 residents = 0.78 %
        'name': 'Lahaina 2023 (held-out)',
        'lat': 20.888, 'lon': -156.673, 'radius': 3000,
        'fire_locations': [(20.896, -156.665), (20.891, -156.670), (20.885, -156.675)],
        'params': {
            'WIND_SPEED': 27.0, 'WIND_INITIAL_DIRECTION': 245.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 15.0,
            'FIRE_SPREAD_PROB_BASE': 0.58, 'ROTHERMEL_BASE_ROS': 1.25,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Teague et al. (2010) Royal Commission; Cruz et al. (2012); Blanchi et al. (2010)
        # NW wind 18 m/s; FFDI 190+; temperature 46.4 °C
        # 119 fatalities / ~12,000 population in Kinglake complex = 0.99 %
        'name': 'Black Saturday 2009 (held-out)',
        'lat': -37.515, 'lon': 145.365, 'radius': 3000,
        'fire_locations': [(-37.503, 145.377), (-37.509, 145.371), (-37.515, 145.365)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # CAL FIRE (2018); Nauslar et al. (2018) Weather and Forecasting 33(5):2123-2148
        # Diablo NE wind 25 m/s; 36,807 acres burned; 22 deaths / ~8,000 = 0.28 %
        'name': 'Tubbs Fire 2017 (held-out)',
        'lat': 38.479, 'lon': -122.728, 'radius': 3000,
        'fire_locations': [(38.491, -122.716), (38.485, -122.722), (38.479, -122.728)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.56, 'ROTHERMEL_BASE_ROS': 1.18,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Koutsias et al. (2012) Agric. Forest Meteorol. 156:41-53; EEA (2007)
        # Etesian NNE wind 14 m/s; 270,000 ha across Greece; 77 deaths
        # ~30 deaths / ~5,000 in Zacharo/Ilia = 0.60 %
        'name': 'Peloponnese 2007 (held-out)',
        'lat': 37.489, 'lon': 21.648, 'radius': 3000,
        'fire_locations': [(37.500, 21.659), (37.494, 21.653), (37.488, 21.648)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 28.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.98,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Encinas et al. (2015) Int J Disaster Risk Reduction 13:280-289; CONAF (2014)
        # SE wind 12 m/s (La Nina drought); 15 deaths / ~8,000 = 0.19 %
        'name': 'Valparaiso 2014 (held-out)',
        'lat': -33.047, 'lon': -71.613, 'radius': 3000,
        'fire_locations': [(-33.040, -71.606), (-33.045, -71.611), (-33.050, -71.616)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 315.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 60,
        },
    },
]


def _load_agents(policy_dir: str, device: str = 'cpu') -> dict:
    global_dim = _cfg.RL_GLOBAL_STATE_DIM
    agents = {
        'ff':  PPOAgent('firefighter', global_dim, device=device),
        'rsc': PPOAgent('rescuer',     global_dim, device=device),
        'cmd': PPOAgent('commander',   global_dim, device=device),
    }
    role_map = {'ff': 'firefighter', 'rsc': 'rescuer', 'cmd': 'commander'}
    for key, agent in agents.items():
        path = os.path.join(policy_dir, f'{role_map[key]}.pt')
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded: {path}")
        else:
            print(f"  WARNING: {path} not found — using random policy")
    return agents


def _run_n(scenario: dict, agents: dict, num_runs: int, max_steps: int) -> pd.DataFrame:
    rows = []
    for i in range(num_runs):
        s = run_episode(scenario, agents, max_steps=max_steps, training=False, run_id=i)
        rows.append(s)
    return pd.DataFrame(rows)


def _ci95(series: pd.Series) -> float:
    """95% confidence interval half-width (normal approximation)."""
    return 1.96 * series.std() / np.sqrt(len(series)) if len(series) > 1 else 0.0


def evaluate(
    num_runs: int = 30,
    policy_dir: str = 'models/rl',
    output_file: str = 'marl_evaluation_results.csv',
    device: str = 'cpu',
    max_steps: int = 500,
) -> pd.DataFrame:
    print('=' * 70)
    print('AIGIS — Hybrid BDI+RL Evaluation')
    print('=' * 70)
    print('Grimm et al. (2020) ODD  |  Schulman et al. (2017) PPO')
    print('15 training scenarios (phases 1-3) + 9 held-out real incidents')
    print(f'Runs per scenario: {num_runs}  |  Policy dir: {policy_dir}')
    print('=' * 70 + '\n')

    agents = _load_agents(policy_dir, device)

    all_rows = []
    # All 15 curriculum training scenarios + 9 held-out real incidents = 24 total.
    # Training split confirms in-distribution generalisation across phases 1–3.
    # Held-out split measures OOD generalisation to real documented events.
    eval_scenarios = list(SCENARIOS) + HELD_OUT  # 15 training + 9 held-out

    for scenario in eval_scenarios:
        name = scenario['name']
        split = 'held-out' if '(held-out)' in name else 'training'
        print(f"  [{split}] {name}")

        df = _run_n(scenario, agents, num_runs, max_steps)
        df['scenario_name'] = name
        df['split'] = split
        all_rows.append(df)

        for metric in ['mortality_rate', 'evacuation_success_rate', 'burned_area_pct']:
            m   = df[metric].mean()
            ci  = _ci95(df[metric])
            print(f"    {metric:<30}  mean={m:.3f}  95%CI=[{m-ci:.3f}, {m+ci:.3f}]")
        print()

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    _plot_evaluation(df_all, output_file.replace('.csv', '.png'))
    return df_all


def _plot_evaluation(df: pd.DataFrame, out_path: str) -> None:
    scenarios = df['scenario_name'].unique()
    metrics   = ['mortality_rate', 'evacuation_success_rate', 'burned_area_pct']
    labels    = ['Mortality Rate', 'Evacuation Success', 'Burned Area %']

    # Colour by split: training = blue, held-out = amber
    split_map = df.drop_duplicates('scenario_name').set_index('scenario_name')['split']
    colours = [('#4895ef' if split_map[s] == 'training' else '#f4a261') for s in scenarios]

    fig = plt.figure(figsize=(len(scenarios) * 2 + 2, 10), facecolor=BG)
    fig.suptitle(
        'AIGIS: Hybrid BDI+RL Evaluation  (Schulman et al. 2017 PPO)\n'
        'Blue = training scenarios  |  Amber = held-out (OOD)  |  error bars = 95% CI',
        color=FG, fontsize=9, fontweight='bold',
    )
    gs = gridspec.GridSpec(len(metrics), 1, figure=fig, hspace=0.55)

    for row_i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = fig.add_subplot(gs[row_i, 0])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        x = np.arange(len(scenarios))
        means = [df[df['scenario_name'] == s][metric].mean() for s in scenarios]
        cis   = [_ci95(df[df['scenario_name'] == s][metric]) for s in scenarios]

        ax.bar(x, means, width=0.6, color=colours, alpha=0.8,
               yerr=cis, capsize=3,
               error_kw={'ecolor': FG, 'elinewidth': 0.8})

        ax.set_xticks(x)
        if row_i == len(metrics) - 1:
            short = [s.split('(')[0].strip()[:18] for s in scenarios]
            ax.set_xticklabels(short, color=FG, fontsize=6.5, rotation=15, ha='right')
        else:
            ax.set_xticklabels([], color=FG)

        if metric == 'burned_area_pct':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.set_ylabel(label, color=FG, fontsize=8)

    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f"Evaluation plot saved to: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Evaluate Hybrid BDI+RL system')
    p.add_argument('--runs',    type=int, default=30)
    p.add_argument('--output',  type=str, default='marl_evaluation_results.csv')
    p.add_argument('--dir',     type=str, default='models/rl')
    p.add_argument('--device',  type=str, default='cpu')
    p.add_argument('--steps',   type=int, default=500)
    args = p.parse_args()
    evaluate(num_runs=args.runs, policy_dir=args.dir,
             output_file=args.output, device=args.device, max_steps=args.steps)


if __name__ == '__main__':
    main()
