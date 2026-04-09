"""
Incremental Retraining Validation
==================================
Validates AIGIS against each of the 9 held-out incidents sequentially.
After each incident, the ML hurdle model is retrained on the expanded
training dataset (original 23-scenario data + all incident runs so far).

Experiment design (online/continual learning):
  1. Start with models trained on 23 training scenarios.
  2. For each held-out incident (in chronological order):
     a. Run 50 simulations under documented conditions.
     b. Record pre-retrain model predictions on those runs (before retraining).
     c. Append feature+target rows to training_dataset.csv.
     d. Retrain the model on the expanded dataset.
     e. Record post-retrain predictions on the same runs.
  3. Output a CSV showing how AUC-ROC, MAE (casualties), R² (evacuation)
     evolve after each increment.

Motivation:
  The 5-fold CV (step 2b) revealed negative R² for evacuation when evaluated
  in eval-mode (no per-fold refit).  This experiment tests whether sequential
  retraining with real incident data corrects that deficiency, demonstrating
  the system's online adaptation capability — a key deployment scenario.

Primary reference for incremental/online learning evaluation:
  Losing, V., Hammer, B., & Wersing, H. (2018). "Incremental On-line Learning:
  A Review and Comparison of State of the Art Algorithms." Neurocomputing,
  275, pp. 1261-1274. DOI: 10.1016/j.neucom.2017.06.084.

Usage
-----
  python run_incremental_validation.py [--runs N] [--output FILE]
  (run AFTER all 17 steps of run_thesis_experiments.sh are complete)

Outputs
-------
  incremental_validation_results.csv — metric evolution across incidents
  incremental_validation_plot.png    — line plots of metric trajectories
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from __future__ import annotations
import argparse
import copy
import pickle
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, mean_absolute_error, r2_score, mean_squared_error,
)

from src.simulation import AIGISSimulation
from src.config import MAX_STEPS
from src.ml_predictor import RiskPredictor

# Reuse helpers from train_models without re-running __main__
from train_models import (
    _extract_state, _apply_overrides, _reset_overrides,
    _quiet, FEATURE_NAMES, MODEL_SPECS, train_and_save,
)

# ---------------------------------------------------------------------------
# Incident definitions (chronological order of event date)
# Configs imported from each validate_*.py to avoid duplication
# ---------------------------------------------------------------------------

from validate_pedrogao     import (PEDROGAO_LAT    as _LAT, PEDROGAO_LON    as _LON,
                                   PEDROGAO_RADIUS  as _RAD,
                                   PEDROGAO_FIRE_LOCATIONS as _FL,
                                   PEDROGAO_CONFIG_OVERRIDES as _CO)
PEDROGAO = dict(name='Pedrogao Grande 2017', lat=_LAT, lon=_LON, radius=_RAD,
                fire_locations=_FL, config_overrides=_CO)

from validate_mati         import (MATI_LAT, MATI_LON, MATI_RADIUS,
                                   MATI_FIRE_LOCATIONS, MATI_CONFIG_OVERRIDES)
MATI = dict(name='Mati 2018', lat=MATI_LAT, lon=MATI_LON, radius=MATI_RADIUS,
            fire_locations=MATI_FIRE_LOCATIONS, config_overrides=MATI_CONFIG_OVERRIDES)

from validate_campfire     import (CAMPFIRE_LAT, CAMPFIRE_LON, CAMPFIRE_RADIUS,
                                   CAMPFIRE_FIRE_LOCATIONS, CAMPFIRE_CONFIG_OVERRIDES)
CAMPFIRE = dict(name='Camp Fire 2018', lat=CAMPFIRE_LAT, lon=CAMPFIRE_LON,
                radius=CAMPFIRE_RADIUS, fire_locations=CAMPFIRE_FIRE_LOCATIONS,
                config_overrides=CAMPFIRE_CONFIG_OVERRIDES)

from validate_tubbs        import (TUBBS_LAT, TUBBS_LON, TUBBS_RADIUS,
                                   TUBBS_FIRE_LOCATIONS, TUBBS_CONFIG_OVERRIDES)
TUBBS = dict(name='Tubbs Fire 2017', lat=TUBBS_LAT, lon=TUBBS_LON,
             radius=TUBBS_RADIUS, fire_locations=TUBBS_FIRE_LOCATIONS,
             config_overrides=TUBBS_CONFIG_OVERRIDES)

from validate_black_saturday import (BS_LAT, BS_LON, BS_RADIUS,
                                     BS_FIRE_LOCATIONS, BS_CONFIG_OVERRIDES)
BLACK_SAT = dict(name='Black Saturday 2009', lat=BS_LAT, lon=BS_LON,
                 radius=BS_RADIUS, fire_locations=BS_FIRE_LOCATIONS,
                 config_overrides=BS_CONFIG_OVERRIDES)

from validate_peloponnese  import (PELOP_LAT, PELOP_LON, PELOP_RADIUS,
                                   PELOP_FIRE_LOCATIONS, PELOP_CONFIG_OVERRIDES)
PELOPONNESE = dict(name='Peloponnese 2007', lat=PELOP_LAT, lon=PELOP_LON,
                   radius=PELOP_RADIUS, fire_locations=PELOP_FIRE_LOCATIONS,
                   config_overrides=PELOP_CONFIG_OVERRIDES)

from validate_lahaina      import (LAHAINA_LAT, LAHAINA_LON, LAHAINA_RADIUS,
                                   LAHAINA_FIRE_LOCATIONS, LAHAINA_CONFIG_OVERRIDES)
LAHAINA = dict(name='Lahaina 2023', lat=LAHAINA_LAT, lon=LAHAINA_LON,
               radius=LAHAINA_RADIUS, fire_locations=LAHAINA_FIRE_LOCATIONS,
               config_overrides=LAHAINA_CONFIG_OVERRIDES)

from validate_alexandroupoli import (ALEX_LAT, ALEX_LON, ALEX_RADIUS,
                                     ALEX_FIRE_LOCATIONS, ALEX_CONFIG_OVERRIDES)
ALEXANDROUPOLI = dict(name='Alexandroupoli 2023', lat=ALEX_LAT, lon=ALEX_LON,
                      radius=ALEX_RADIUS, fire_locations=ALEX_FIRE_LOCATIONS,
                      config_overrides=ALEX_CONFIG_OVERRIDES)

from validate_valparaiso   import (VALP_LAT, VALP_LON, VALP_RADIUS,
                                   VALP_FIRE_LOCATIONS, VALP_CONFIG_OVERRIDES)
VALPARAISO = dict(name='Valparaiso 2014', lat=VALP_LAT, lon=VALP_LON,
                  radius=VALP_RADIUS, fire_locations=VALP_FIRE_LOCATIONS,
                  config_overrides=VALP_CONFIG_OVERRIDES)

# Chronological order (event date)
INCIDENTS = [
    PELOPONNESE,    # 2007
    BLACK_SAT,      # 2009
    PEDROGAO,       # 2017
    TUBBS,          # 2017
    MATI,           # 2018
    CAMPFIRE,       # 2018
    VALPARAISO,     # 2014 (included here for geographic diversity sweep)
    LAHAINA,        # 2023
    ALEXANDROUPOLI, # 2023
]

MODEL_DIR    = Path('models')
TRAIN_CSV    = Path('training_dataset.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_payload(name: str) -> dict:
    """Load a pkl payload from models/."""
    path = MODEL_DIR / name
    with open(path, 'rb') as f:
        return pickle.load(f)


def _evaluate_on_runs(
    X: np.ndarray,
    y_casualties: np.ndarray,
    y_evacuated:  np.ndarray,
) -> dict:
    """
    Evaluate the current saved models on a set of feature rows + targets.
    Returns AUC-ROC, MAE (casualties), R² (evacuation).
    """
    # Casualty hurdle model
    cas_payload = _load_model_payload('casualty_risk_model.pkl')
    scaler      = cas_payload['scaler']
    classifier  = cas_payload['classifier']
    regressor   = cas_payload.get('regressor')
    X_s         = scaler.transform(X)

    y_bin  = (y_casualties > 0).astype(int)
    proba  = classifier.predict_proba(X_s)[:, 1]
    auc    = roc_auc_score(y_bin, proba) if y_bin.sum() > 0 and y_bin.sum() < len(y_bin) else float('nan')

    if regressor is not None:
        cas_pred = proba * np.maximum(regressor.predict(X_s), 0)
    else:
        cas_pred = proba * y_casualties.mean()
    mae_cas = mean_absolute_error(y_casualties, cas_pred)

    # Evacuation model
    evac_payload = _load_model_payload('evacuation_count_model.pkl')
    scaler_e     = evac_payload['scaler']
    model_e      = evac_payload['model']
    X_se         = scaler_e.transform(X)
    evac_pred    = np.maximum(model_e.predict(X_se), 0)
    r2_evac      = r2_score(y_evacuated, evac_pred)
    mae_evac     = mean_absolute_error(y_evacuated, evac_pred)

    return {
        'auc_roc':          auc,
        'mae_casualties':   mae_cas,
        'r2_evacuation':    r2_evac,
        'mae_evacuation':   mae_evac,
    }


def _run_incident_sims(incident: dict, num_runs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run `num_runs` simulations for an incident.
    Returns (X, y_casualties, y_evacuated) arrays where X has FEATURE_NAMES columns.
    Mid-step feature extraction mirrors train_models.generate_dataset().
    """
    predictor = RiskPredictor.__new__(RiskPredictor)
    predictor.models     = {}
    predictor.is_trained = False

    rows_X  = []
    y_cas   = []
    y_evac  = []
    y_steps = []

    midpoint = MAX_STEPS // 2

    snapshot = _apply_overrides(incident['config_overrides'])
    try:
        for i in range(num_runs):
            print(f"  Run {i + 1}/{num_runs}", end="\r", flush=True)
            with _quiet():
                sim = AIGISSimulation(
                    lat=incident['lat'],
                    lon=incident['lon'],
                    radius=incident['radius'],
                    mode='batch',
                    run_id=i,
                    fire_locations=incident['fire_locations'],
                )

            with _quiet():
                while sim.step < midpoint and not sim.is_complete():
                    sim.run_step()

            state    = _extract_state(sim)
            features = predictor._extract_features(state)
            rows_X.append(features)

            with _quiet():
                while sim.step < MAX_STEPS and not sim.is_complete():
                    sim.run_step()

            result = sim.get_results()
            y_cas.append(result['casualties'])
            y_evac.append(result['evacuated'])
            y_steps.append(result['steps'])
    finally:
        _reset_overrides(snapshot)

    print()
    return (
        np.array(rows_X,  dtype=np.float32),
        np.array(y_cas,   dtype=np.float32),
        np.array(y_evac,  dtype=np.float32),
        np.array(y_steps, dtype=np.float32),
    )


def _retrain(df_train: pd.DataFrame) -> dict:
    """Retrain all models on df_train, save pkl files, return metrics."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        metrics = train_and_save(df_train, MODEL_DIR, '/tmp/incremental_train_plot.png')
    return metrics


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_incremental_validation(num_runs: int = 50, output_file: str = 'incremental_validation_results.csv'):
    print("=" * 70)
    print("AIGIS — Incremental Retraining Validation")
    print("Losing et al. (2018) DOI: 10.1016/j.neucom.2017.06.084")
    print(f"Incidents: {len(INCIDENTS)} | Runs per incident: {num_runs}")
    print("=" * 70 + "\n")

    # Load baseline training dataset
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Baseline training set: {len(df_train)} rows\n")

    records = []

    for idx, incident in enumerate(INCIDENTS):
        print(f"\n[{idx + 1}/{len(INCIDENTS)}] {incident['name']}")
        print("-" * 50)

        # Run simulations
        X, y_cas, y_evac, y_steps = _run_incident_sims(incident, num_runs)

        # Evaluate BEFORE retraining
        pre = _evaluate_on_runs(X, y_cas, y_evac)
        print(f"  Pre-retrain  | AUC-ROC: {pre['auc_roc']:.4f}  "
              f"MAE cas: {pre['mae_casualties']:.3f}  "
              f"R² evac: {pre['r2_evacuation']:.4f}")

        # Build new rows for training dataset
        new_rows = []
        for j in range(len(X)):
            row = {f: float(X[j, k]) for k, f in enumerate(FEATURE_NAMES)}
            row['run_id']            = j
            row['train_lat']         = incident['lat']
            row['train_lon']         = incident['lon']
            row['target_casualties'] = float(y_cas[j])
            row['target_evacuated']  = float(y_evac[j])
            row['target_steps']      = float(y_steps[j])
            # param columns not available; fill with NaN
            for col in df_train.columns:
                if col.startswith('param_') and col not in row:
                    row[col] = float('nan')
            new_rows.append(row)

        df_new   = pd.DataFrame(new_rows)
        df_train = pd.concat([df_train, df_new], ignore_index=True)
        df_train.to_csv(TRAIN_CSV, index=False)
        print(f"  Training set expanded to {len(df_train)} rows — retraining...")

        # Retrain
        _retrain(df_train)

        # Evaluate AFTER retraining
        post = _evaluate_on_runs(X, y_cas, y_evac)
        print(f"  Post-retrain | AUC-ROC: {post['auc_roc']:.4f}  "
              f"MAE cas: {post['mae_casualties']:.3f}  "
              f"R² evac: {post['r2_evacuation']:.4f}")

        records.append({
            'incident':              incident['name'],
            'incident_idx':          idx + 1,
            'training_set_size':     len(df_train),
            'pre_auc_roc':           pre['auc_roc'],
            'pre_mae_casualties':    pre['mae_casualties'],
            'pre_mae_evacuation':    pre['mae_evacuation'],
            'pre_r2_evacuation':     pre['r2_evacuation'],
            'post_auc_roc':          post['auc_roc'],
            'post_mae_casualties':   post['mae_casualties'],
            'post_mae_evacuation':   post['mae_evacuation'],
            'post_r2_evacuation':    post['r2_evacuation'],
            'delta_auc_roc':         post['auc_roc']        - pre['auc_roc'],
            'delta_mae_casualties':  post['mae_casualties']  - pre['mae_casualties'],
            'delta_r2_evacuation':   post['r2_evacuation']   - pre['r2_evacuation'],
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    _plot_results(df_out, output_file.replace('.csv', '.png'))
    return df_out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(df: pd.DataFrame, out_path: str) -> None:
    BG    = 'white'
    PANEL = '#f5f5f5'
    FG    = '#222222'
    C_PRE  = '#ff6b6b'
    C_POST = '#06d6a0'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle(
        "Incremental Retraining — Metric Evolution Across Held-Out Incidents\n"
        "Losing et al. (2018) DOI: 10.1016/j.neucom.2017.06.084",
        color=FG, fontsize=10, fontweight='bold',
    )

    metrics = [
        ('pre_auc_roc',        'post_auc_roc',        'AUC-ROC (Casualty Classifier)',    None),
        ('pre_mae_casualties',  'post_mae_casualties',  'MAE — Casualties',                 None),
        ('pre_r2_evacuation',   'post_r2_evacuation',   'R² — Evacuation Count',            0.0),
    ]

    x = df['incident_idx'].values
    labels = [n.split()[0] for n in df['incident']]

    for ax, (pre_col, post_col, title, hline) in zip(axes, metrics):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#cccccc')

        ax.plot(x, df[pre_col],  'o--', color=C_PRE,  linewidth=1.5, label='Before retrain', markersize=5)
        ax.plot(x, df[post_col], 'o-',  color=C_POST, linewidth=1.5, label='After retrain',  markersize=5)
        if hline is not None:
            ax.axhline(hline, color='white', linestyle=':', linewidth=1, alpha=0.5, label=f'y={hline}')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6, color=FG)
        ax.set_title(title, color=FG, fontsize=9)
        ax.set_xlabel('Incident (chronological)', color=FG, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Plot saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Incremental retraining validation across held-out incidents'
    )
    parser.add_argument('--runs',   type=int, default=50,
                        help='Simulation runs per incident (default: 50)')
    parser.add_argument('--output', type=str, default='incremental_validation_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    run_incremental_validation(num_runs=args.runs, output_file=args.output)


if __name__ == '__main__':
    main()
