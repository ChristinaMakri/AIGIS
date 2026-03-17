"""
AIGIS — Hybrid BDI+RL Evaluation
==================================
Evaluates the trained hybrid system on:
  1. Training scenarios (in-distribution check)
  2. Held-out Mati 2018 (OOD generalisation)
  3. Held-out Camp Fire 2018 (OOD generalisation)

Reports: mean, std, 95% CI across N independent runs.
  Bengio, Y. et al. (2009). "Curriculum learning." ICML.
  Schulman, J. et al. (2017). "Proximal Policy Optimization." arXiv:1707.06347.
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
    Other Simulation Models." JASSS, 23(2), 7.

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

# Held-out scenarios (never seen during training)
HELD_OUT = [
    {
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
    print(f'Runs per scenario: {num_runs}  |  Policy dir: {policy_dir}')
    print('=' * 70 + '\n')

    agents = _load_agents(policy_dir, device)

    all_rows = []
    eval_scenarios = SCENARIOS[:3] + HELD_OUT  # 3 training + 2 held-out

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
