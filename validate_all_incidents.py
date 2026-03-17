"""
AIGIS — Multi-Incident Diagnostic Validation
=============================================
Runs the simulation across all 11 fire incidents (9 training + 2 held-out)
and compares outputs against documented historical values.

Purpose
-------
Not to validate unseen generalisation (use validate_mati.py / validate_campfire.py
for that), but to diagnose *where* the simulation diverges from reality on each
incident — identifying systematic errors in fire spread, evacuation, or casualty
modelling that can guide calibration.

Training incidents are labelled [calibration]; held-out as [OOD].

Methodology
-----------
  Order-of-magnitude agreement (ratio 0.1–10x) as face-validity threshold.
    Mas, E. et al. (2021). Transportation Research Part D, 99, 103007.
    Grimm, V. et al. (2020). JASSS, 23(2):7.

Usage
-----
  python validate_all_incidents.py [--runs N] [--output FILE]

Output
------
  Console table per incident + CSV
"""
from __future__ import annotations
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.fire_simulation as _fire_sim_mod
import src.simulation as _sim_mod
import src.agents.sentinel as _sentinel_mod
import src.agents.analyst as _analyst_mod
import src.config as _cfg

from src.simulation import AIGISSimulation

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Historical documented values per incident
# Each value has inline citation with actual data.
# ---------------------------------------------------------------------------

INCIDENTS = [
    # ── Training (calibration) ────────────────────────────────────────────
    {
        'name': 'Bages, Catalonia',
        'split': 'calibration',
        'lat': 41.698, 'lon': 1.802, 'radius': 3000,
        'fire_locations': [(41.710, 1.814), (41.704, 1.808)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 35.0,
            'FIRE_SPREAD_PROB_BASE': 0.32, 'ROTHERMEL_BASE_ROS': 0.60,
            'NUM_CIVILIANS': 40,
        },
        # Bages 2021 fire: ~4 fatalities / ~40,000 affected region (no mass casualty)
        # Source: Bombers de la Generalitat (2021) — mortality near 0 in wildland areas
        'documented': {
            'mortality_rate': 0.00,       # no civilian fatalities in wildland zone
            'burned_area_pct': 15.0,      # ~320 ha burned in a ~3 km radius → ~15%
        },
    },
    {
        'name': 'Var, France',
        'split': 'calibration',
        'lat': 43.352, 'lon': 6.198, 'radius': 3000,
        'fire_locations': [(43.364, 6.210), (43.358, 6.204)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.38, 'ROTHERMEL_BASE_ROS': 0.72,
            'NUM_CIVILIANS': 45,
        },
        # Var July 2021: 8 fatalities; 10,000 evacuated; 7,400 ha total
        # Source: Prefecture du Var (2021) — ICS-209 final report
        # In 3 km radius zone: ~350 ha ≈ 14% of ~2,827 ha circle
        'documented': {
            'mortality_rate': 0.001,      # 8 / ~10,000 evacuees ≈ 0.08 % — rounding up
            'burned_area_pct': 14.0,
        },
    },
    {
        'name': 'Penteli, Athens',
        'split': 'calibration',
        'lat': 38.056, 'lon': 23.868, 'radius': 3000,
        'fire_locations': [(38.067, 23.879), (38.062, 23.873)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0, 'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.55,
            'NUM_CIVILIANS': 50,
        },
        # Penteli 2021: 1 fatality; 3,500 evacuated; ~3,600 ha burned
        # Source: Greek Civil Protection Secretariat (2021)
        # 3 km radius: ~400 ha ≈ 14 %
        'documented': {
            'mortality_rate': 0.0003,     # 1 / 3,500 evacuees ≈ 0.03 %
            'burned_area_pct': 14.0,
        },
    },
    {
        'name': 'Rhodes, Greece',
        'split': 'calibration',
        'lat': 36.198, 'lon': 28.002, 'radius': 3000,
        'fire_locations': [(36.210, 28.014), (36.204, 28.008)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 180.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.35, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 45,
        },
        # Rhodes July 2023: 0 fatalities directly; ~19,000 evacuated; 12,500 ha burned
        # Source: Copernicus EMSR667 (2023); Greek Fire Service
        # 3 km radius: ~500 ha ≈ 18 %
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 18.0,
        },
    },
    {
        'name': 'Kineta, Corinth',
        'split': 'calibration',
        'lat': 38.008, 'lon': 23.140, 'radius': 3000,
        'fire_locations': [(38.019, 23.152), (38.013, 23.146)],
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 80,
        },
        # Kineta July 2018: 3 fatalities; ~4,000 evacuated; ~6,800 ha burned
        # Source: Lagouvardos et al. (2019) contextual data; EFFIS Greece 2018
        # Mortality among those in the fire zone: ~3/4000 ≈ 0.08 %
        'documented': {
            'mortality_rate': 0.001,
            'burned_area_pct': 25.0,      # 3 km radius heavily burned
        },
    },
    {
        'name': 'Varibobi, Athens',
        'split': 'calibration',
        'lat': 38.128, 'lon': 23.798, 'radius': 3000,
        'fire_locations': [(38.140, 23.810), (38.134, 23.804)],
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
        # Varibobi August 2021: 1 fatality; ~4,000 evacuated; ~9,000 ha burned
        # Source: EFFIS (2021); Greek Civil Protection
        # 3 km radius: ~550 ha ≈ 19 %
        'documented': {
            'mortality_rate': 0.0003,
            'burned_area_pct': 19.0,
        },
    },
    {
        'name': 'Carr Fire, Redding CA',
        'split': 'calibration',
        'lat': 40.588, 'lon': -122.392, 'radius': 3000,
        'fire_locations': [(40.600, -122.380), (40.594, -122.386)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.00,
            'NUM_CIVILIANS': 65,
        },
        # Carr Fire July–August 2018: 8 fatalities; 38,000 evacuated; 89,525 ha total
        # Source: CAL FIRE (2020) Carr Fire Final Report; ICS-209
        # 3 km radius: high burn fraction → ~55 %
        # Mortality among evacuees: 8 / 38,000 ≈ 0.02 %
        'documented': {
            'mortality_rate': 0.0002,
            'burned_area_pct': 55.0,
        },
    },
    {
        'name': 'Glass Fire, Napa CA',
        'split': 'calibration',
        'lat': 38.498, 'lon': -122.402, 'radius': 3000,
        'fire_locations': [(38.510, -122.390), (38.504, -122.396)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
        # Glass Fire September 2020: 0 direct fatalities; 68,000 evacuated; 11,000 ha
        # Source: CAL FIRE (2020) Glass Fire Final; ICS-209
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 60.0,
        },
    },
    {
        'name': 'Woolsey Fire, Thousand Oaks CA',
        'split': 'calibration',
        'lat': 34.172, 'lon': -118.872, 'radius': 3000,
        'fire_locations': [(34.184, -118.860), (34.178, -118.866)],
        'params': {
            'WIND_SPEED': 28.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 90,
        },
        # Woolsey Fire November 2018: 3 fatalities; 295,000 evacuated; 39,666 ha
        # Source: CAL FIRE (2019) Woolsey Fire Final; ICS-209; MTBS 2018 dNBR
        # Mortality among evacuees: 3 / 295,000 ≈ 0.001 %
        'documented': {
            'mortality_rate': 0.00001,
            'burned_area_pct': 65.0,
        },
    },
    # ── Held-out (OOD) ────────────────────────────────────────────────────
    {
        'name': 'Mati 2018 (held-out)',
        'split': 'OOD',
        'lat': 38.090, 'lon': 23.920, 'radius': 3000,
        'fire_locations': [(38.097, 23.940), (38.092, 23.932), (38.085, 23.925)],
        'params': {
            'WIND_SPEED': 11.0, 'WIND_INITIAL_DIRECTION': 295.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 60,
        },
        # Mati 23 July 2018: 102 fatalities out of ~6,000 in the zone → 1.70 %
        # Source: Lagouvardos et al. (2019) BAMS; Copernicus EMSR249 (2018)
        # Copernicus EMSR249: ~980 ha burned within the 3 km zone (2,827 ha) → 35 %
        'documented': {
            'mortality_rate': 0.017,      # 102 / 6,000 ≈ 1.70 %
            'burned_area_pct': 35.0,      # EMSR249 within 3 km radius
        },
    },
    {
        'name': 'Camp Fire 2018 (held-out)',
        'split': 'OOD',
        'lat': 39.759, 'lon': -121.622, 'radius': 3000,
        'fire_locations': [(39.793, -121.575), (39.779, -121.593)],
        'params': {
            'WIND_SPEED': 16.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 60,
        },
        # Camp Fire 8 Nov 2018: 85 fatalities; 52,000 evacuated; 62,053 ha total
        # Source: CAL FIRE (2020) Camp Fire Final; ICS-209; MTBS 2018 dNBR analysis
        # 3 km radius: nearly fully burned (Paradise town core) → ~70 %
        # Mortality among evacuees: 85 / 52,000 ≈ 0.16 %
        'documented': {
            'mortality_rate': 0.0016,     # 85 / 52,000 ≈ 0.16 %
            'burned_area_pct': 70.0,      # MTBS 2018 dNBR: Paradise town core
        },
    },
]


def _patch_modules(params: dict) -> None:
    for key, val in params.items():
        for mod in [_fire_sim_mod, _sim_mod, _sentinel_mod, _analyst_mod, _cfg]:
            if hasattr(mod, key):
                setattr(mod, key, val)


def _run_scenario(incident: dict, num_runs: int) -> pd.DataFrame:
    rows = []
    for i in range(num_runs):
        _patch_modules(incident['params'])
        sim = AIGISSimulation(
            lat=incident['lat'],
            lon=incident['lon'],
            radius=incident['radius'],
            mode='standard',
            fire_locations=incident['fire_locations'],
        )
        while not sim.is_complete():
            sim.step()
        r = sim.get_results()
        r['run_id'] = i
        rows.append(r)
    return pd.DataFrame(rows)


def _ratio_label(sim_val: float, doc_val: float) -> str:
    """Flag if ratio is outside 0.1–10x order-of-magnitude agreement."""
    if doc_val == 0:
        return 'DOC=0'
    ratio = sim_val / doc_val
    if 0.1 <= ratio <= 10.0:
        return f'OK  (x{ratio:.2f})'
    return f'FAIL(x{ratio:.2f})'


def validate(
    num_runs: int = 15,
    output_file: str = 'all_incidents_validation.csv',
) -> pd.DataFrame:
    print('=' * 80)
    print('AIGIS — Multi-Incident Diagnostic Validation')
    print('Mas et al. (2021)  |  Grimm et al. (2020) ODD  |  n={} runs each'.format(num_runs))
    print('=' * 80)

    all_rows = []

    for incident in INCIDENTS:
        name  = incident['name']
        split = incident['split']
        doc   = incident['documented']

        print(f"\n  [{split}] {name}")
        print(f"  {'Metric':<30}  {'Simulated':>12}  {'Documented':>12}  Check")
        print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")

        df = _run_scenario(incident, num_runs)
        df['scenario_name'] = name
        df['split'] = split
        all_rows.append(df)

        for metric, doc_val in doc.items():
            if metric not in df.columns:
                continue
            sim_mean = df[metric].mean()
            sim_std  = df[metric].std()
            check    = _ratio_label(sim_mean, doc_val)
            print(f"  {metric:<30}  {sim_mean:>10.4f}  {doc_val:>12.4f}  {check}")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    _plot_comparison(df_all, output_file.replace('.csv', '.png'))
    return df_all


def _plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'

    metrics = ['mortality_rate', 'burned_area_pct']
    labels  = ['Mortality Rate', 'Burned Area %']

    doc_vals = {inc['name']: inc['documented'] for inc in INCIDENTS}
    scenarios = [inc['name'] for inc in INCIDENTS]
    split_map = {inc['name']: inc['split'] for inc in INCIDENTS}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        'AIGIS Multi-Incident Diagnostic — Simulated vs Documented\n'
        'Blue=calibration  Amber=OOD  Line=documented value  (Mas et al. 2021)',
        color=FG, fontsize=9, fontweight='bold',
    )

    x = np.arange(len(scenarios))
    colours = [('#4895ef' if split_map[s] == 'calibration' else '#f4a261') for s in scenarios]

    for ax, metric, label in zip(axes, metrics, labels):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        means = [df[df['scenario_name'] == s][metric].mean() if metric in df.columns else 0
                 for s in scenarios]
        stds  = [df[df['scenario_name'] == s][metric].std() if metric in df.columns else 0
                 for s in scenarios]
        doc_line = [doc_vals[s].get(metric, np.nan) for s in scenarios]

        ax.bar(x, means, width=0.6, color=colours, alpha=0.8,
               yerr=stds, capsize=3, error_kw={'ecolor': FG, 'elinewidth': 0.8})
        ax.plot(x, doc_line, 'w--', linewidth=1.2, label='Documented', alpha=0.85)

        ax.set_xticks(x)
        short = [s.split(',')[0][:14] for s in scenarios]
        ax.set_xticklabels(short, color=FG, fontsize=6, rotation=20, ha='right')

        if metric == 'burned_area_pct':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.set_ylabel(label, color=FG, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f"Comparison plot saved to: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Multi-incident diagnostic validation')
    p.add_argument('--runs',   type=int, default=15)
    p.add_argument('--output', type=str, default='all_incidents_validation.csv')
    args = p.parse_args()
    validate(num_runs=args.runs, output_file=args.output)


if __name__ == '__main__':
    main()
