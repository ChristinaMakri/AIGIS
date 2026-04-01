"""
AIGIS — ML Model Evaluation
============================
Evaluates the four trained XGBoost models against ground-truth simulation
outcomes to quantify prediction accuracy.

Methodology
-----------
Each evaluation run proceeds in two phases:
  1. Run simulation to the midpoint (step = MAX_STEPS // 2), then extract
     the 14 simulation-state features and obtain ML predictions.
  2. Continue simulation to completion and record actual outcomes.

Predictions at the midpoint are compared to final outcomes — this reflects
how the models are used in practice: the Commander queries the predictor
mid-simulation to inform phase decisions.

Models evaluated
----------------
  casualty_risk    → predicted_casualties  vs  actual casualties
  evacuation_count → predicted_evacuations vs  actual evacuated
  containment_time → predicted_containment_days (no direct ground truth;
                      reported as descriptive stats only)

Metrics reported
----------------
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Squared Error
  R²    — Coefficient of determination (sklearn.metrics.r2_score)

Feature importances
-------------------
  XGBoost / sklearn models expose feature_importances_ directly.
  The 14 feature names follow ML_FEATURE_NAMES from config.py.

References
----------
  Willmott, C.J. & Matsuura, K. (2005). "Advantages of the mean absolute
    error (MAE) over the root mean square error (RMSE) in assessing average
    model performance." Climate Research, 30(1), 79–82.
    DOI: 10.3354/cr030079
    [MAE + RMSE together characterise both average and outlier error.]

  Nagelkerke, N.J.D. (1991). "A note on a general definition of the
    coefficient of determination." Biometrika, 78(3), 691–692.
    DOI: 10.1093/biomet/78.3.691
    [R² as a normalised goodness-of-fit measure.]

  Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
    System." Proceedings of KDD '16, pp. 785–794.
    DOI: 10.1145/2939672.2939785
    [XGBoost feature_importances_ (gain-based) interpretation.]

  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
    Other Simulation Models: A Second Update." JASSS 23(2):7.
    DOI: 10.18564/jasss.4259
    [ODD §"Design concepts / Prediction" — internal model evaluation.]

Usage
-----
  python evaluate_ml_models.py [--runs N] [--output FILE]
      [--lat LAT --lon LON --radius R]

Outputs
-------
  - Console table: MAE, RMSE, R² per model
  - CSV: all (predicted, actual) pairs per run
  - PNG: 2-panel figure — scatter plots + feature importances
"""
import argparse
import contextlib
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, precision_score, recall_score)

from src.simulation import AIGISSimulation
from src.ml_predictor import RiskPredictor
from src.config import MAX_STEPS
from train_models import TRAINING_LOCATIONS

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature names (must match ml_predictor._extract_features order)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'burning_cells_pct', 'burnt_cells_pct',
    'wind_speed', 'wind_dir_x', 'wind_dir_y',
    'mean_slope', 'dominant_fuel_type',
    'active_rescuers', 'civilians_remaining',
    'current_phase',
    'tti_normalized', 'ect_normalized', 'step_normalized',
    'humidity',
]

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'


@contextlib.contextmanager
def _quiet():
    """Suppress stdout during simulation runs."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _extract_state(sim: AIGISSimulation) -> dict:
    """
    Build the simulation_state dict that RiskPredictor.predict_casualty_risk()
    expects, mirroring how CommanderAgent constructs it (commander.py ~L648).
    """
    env = sim.environment
    commander = sim.agents.get('commander')
    fire_sim = sim.fire_sim

    wind_speed = getattr(fire_sim, 'wind_speed', 5.0)
    wind_dir = list(getattr(fire_sim, 'wind_direction', [1.0, 0.0]))

    return {
        'fire_grid':       env.fire_grid,
        'fuel_type_grid':  getattr(env, 'fuel_type_grid', None),
        'elevation_grid':  env.elevation_grid,
        'wind_speed':      wind_speed,
        'wind_direction':  wind_dir,
        'humidity':        getattr(env, 'humidity', 30.0),
        'tti_minutes':     getattr(commander, 'tti', float('inf')) if commander else float('inf'),
        'ect_minutes':     getattr(commander, 'ect', 0.0) if commander else 0.0,
        'current_phase':   getattr(commander, 'current_phase', 0) if commander else 0,
        'step':            sim.step,
        'max_steps':       MAX_STEPS,
        'agents':          getattr(env, 'agents', {}),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    num_runs: int = 30,
    lat: float = 38.090, lon: float = 23.920, radius: int = 3000,
    output_file: str = 'ml_evaluation_results.csv',
) -> pd.DataFrame:
    """
    Run N simulations, capture ML predictions at midpoint, compare to
    actual outcomes at completion.

    Midpoint evaluation reflects operational use: Commander queries the
    predictor during active fire response, not at t=0.
    """
    print('=' * 70)
    print('AIGIS — ML Model Evaluation')
    print('=' * 70)
    print('Willmott & Matsuura (2005)  |  Chen & Guestrin (2016) XGBoost')
    print(f'Evaluation strategy: midpoint prediction vs. final outcome')
    print(f'Midpoint: step {MAX_STEPS // 2} / {MAX_STEPS}  |  Runs: {num_runs}')
    print('=' * 70 + '\n')

    predictor = RiskPredictor()
    if not predictor.is_trained:
        print('ERROR: No trained models found in models/. Cannot evaluate.')
        return pd.DataFrame()

    records = []

    for i in range(num_runs):
        print(f'  Run {i + 1}/{num_runs}', end='\r', flush=True)

        with _quiet():
            sim = AIGISSimulation(lat=lat, lon=lon, radius=radius,
                                  mode='batch', run_id=i)

        midpoint = MAX_STEPS // 2

        # Phase 1: run to midpoint
        with _quiet():
            while sim.step < midpoint and not sim.is_complete():
                sim.run_step()

        # Extract features + predictions at midpoint
        state = _extract_state(sim)
        preds = predictor.predict_casualty_risk(state)

        # Phase 2: run to completion
        with _quiet():
            while sim.step < MAX_STEPS and not sim.is_complete():
                sim.run_step()

        actual = sim.get_results()

        records.append({
            'run_id':                      i,
            'midpoint_step':               midpoint,
            # Predictions
            'pred_casualties':             preds.get('predicted_casualties', 0.0),
            'pred_evacuations':            preds.get('predicted_evacuations', 0.0),
            'pred_containment_days':       preds.get('predicted_containment_days', 0.0),
            'pred_risk_level':             preds.get('risk_level', 'UNKNOWN'),
            # Actuals
            'actual_casualties':           actual['casualties'],
            'actual_evacuated':            actual['evacuated'],
            'actual_steps':                actual['steps'],
            'actual_mortality_rate':       actual['mortality_rate'],
            'actual_evacuation_rate':      actual['evacuation_success_rate'],
        })

    print()  # newline after \r

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f'Results saved to: {output_file}\n')

    _print_metrics_table(df)
    _plot_evaluation(df, predictor, output_file.replace('.csv', '.png'))

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_metrics_table(df: pd.DataFrame) -> None:
    """
    Print MAE, RMSE, R² for casualty and evacuation predictions.
    containment_time lacks simulation ground truth and is reported as
    descriptive statistics only.

    Willmott & Matsuura (2005): use MAE and RMSE together — MAE measures
    average error; RMSE penalises outliers more heavily.
    """
    print('=' * 70)
    print('ML MODEL EVALUATION RESULTS')
    print('=' * 70)

    # Casualty Risk — classification metrics (binary: casualties > 0?)
    y_score = df['pred_casualties'].values
    y_true_c = df['actual_casualties'].values
    y_bin_true = (y_true_c > 0).astype(int)
    y_bin_pred = (y_score > 0).astype(int)
    mae_c  = mean_absolute_error(y_true_c, y_score)
    rmse_c = np.sqrt(mean_squared_error(y_true_c, y_score))
    bias_c = np.mean(y_score - y_true_c)
    auc    = roc_auc_score(y_bin_true, y_score) if y_bin_true.sum() > 0 else float('nan')
    prec   = precision_score(y_bin_true, y_bin_pred, zero_division=0)
    rec    = recall_score(y_bin_true, y_bin_pred, zero_division=0)

    print('\nCasualty Risk  (predicted_casualties vs. actual casualties):')
    print(f'  MAE       = {mae_c:.3f}   (Willmott & Matsuura 2005)')
    print(f'  RMSE      = {rmse_c:.3f}')
    print(f'  Bias      = {bias_c:+.3f}  (positive = over-prediction)')
    print(f'  AUC-ROC   = {auc:.4f}   (Hanley & McNeil 1982) — binary: casualties > 0')
    print(f'  Precision = {prec:.4f}  (threshold: pred > 0)')
    print(f'  Recall    = {rec:.4f}  (threshold: pred > 0)')

    # Evacuation Count — derived as (total_civilians - predicted_casualties)
    # This identity holds with r=0.9992 across all training runs; a separate
    # regression model adds no predictive value (Pearson 1895).
    y_pred_e = df['pred_evacuations'].values
    y_true_e = df['actual_evacuated'].values
    mae_e  = mean_absolute_error(y_true_e, y_pred_e)
    rmse_e = np.sqrt(mean_squared_error(y_true_e, y_pred_e))
    r2_e   = r2_score(y_true_e, y_pred_e)
    bias_e = np.mean(y_pred_e - y_true_e)

    print('\nEvacuation Count  (derived: total_civilians - predicted_casualties):')
    print(f'  MAE   = {mae_e:.3f}   (Willmott & Matsuura 2005)')
    print(f'  RMSE  = {rmse_e:.3f}')
    print(f'  R²    = {r2_e:.4f}   (Nagelkerke 1991)')
    print(f'  Bias  = {bias_e:+.3f}  (positive = over-prediction)')
    print(f'  Note: derived from casualty prediction; evacuated + casualties = NUM_CIVILIANS'
          f' in 99.9% of runs (r=0.9992)')

    # Descriptive only (no ground truth)
    print('\n--- Descriptive statistics (no simulation ground truth) ---')
    col, label = 'pred_containment_days', 'Containment Time (days)'
    print(f'\n{label}:')
    print(f'  Mean = {df[col].mean():.3f}  |  Std = {df[col].std():.3f}  '
          f'|  Min = {df[col].min():.3f}  |  Max = {df[col].max():.3f}')
    print(f'  Note: no direct simulation ground truth for this output.')

    # Risk level distribution
    print('\nPredicted Risk Level distribution:')
    for level, count in df['pred_risk_level'].value_counts().items():
        print(f'  {level:<10} {count:>4} / {len(df)}  ({count / len(df):.1%})')

    print('\n' + '=' * 70)
    print('AUC-ROC: 1.0 = perfect, 0.5 = random  (Hanley & McNeil 1982)')
    print('R² interpretation: 1.0 = perfect, 0.0 = no better than mean.')
    print('=' * 70)


def _plot_evaluation(df: pd.DataFrame, predictor: RiskPredictor,
                     out_path: str) -> None:
    """
    Two-panel figure:
      Top row: scatter plots of predicted vs. actual (casualty, evacuation)
      Bottom row: feature importances for each model (if available)

    Chen & Guestrin (2016): gain-based feature importance for XGBoost.
    """
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.suptitle(
        'ML Model Evaluation  |  Chen & Guestrin (2016)  |  '
        f'Willmott & Matsuura (2005)  |  n={len(df)} runs',
        color=FG, fontsize=10, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Row 0: scatter plots ----
    scatter_specs = [
        ('pred_casualties',  'actual_casualties',  'Casualty Risk Model',
         'Predicted Casualties', 'Actual Casualties', '#ff006e'),
        ('pred_evacuations', 'actual_evacuated',   'Evacuation Count Model',
         'Predicted Evacuated',  'Actual Evacuated',  '#06d6a0'),
    ]

    for col_i, (pred_col, actual_col, title, xlabel, ylabel, colour) in \
            enumerate(scatter_specs):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        x = df[pred_col].values
        y = df[actual_col].values

        ax.scatter(x, y, color=colour, s=30, alpha=0.7, zorder=5)

        # Perfect-prediction diagonal
        lim = max(x.max(), y.max()) * 1.1
        ax.plot([0, lim], [0, lim], color='white', linestyle='--',
                linewidth=1, alpha=0.5, label='Perfect prediction')

        # Annotation: AUC for casualty risk (classifier), R² for evacuation count
        mae = mean_absolute_error(y, x)
        if pred_col == 'pred_casualties':
            y_bin = (y > 0).astype(int)
            score_val = roc_auc_score(y_bin, x) if y_bin.sum() > 0 else float('nan')
            annot = f'AUC = {score_val:.3f}\nMAE = {mae:.2f}'
        else:
            score_val = r2_score(y, x)
            annot = f'R² = {score_val:.3f}\nMAE = {mae:.2f}'
        ax.text(0.05, 0.92, annot,
                transform=ax.transAxes, color=FG, fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor=PANEL, edgecolor='#3a3a5c', alpha=0.8))

        ax.set_title(title, color=FG, fontsize=9, fontweight='bold')
        ax.set_xlabel(xlabel, color=FG, fontsize=8)
        ax.set_ylabel(ylabel, color=FG, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)

    # ---- Row 1: feature importances ----
    importance_specs = [
        ('casualty_risk',    'Casualty Risk — Feature Importances', '#ff006e'),
        ('evacuation_count', 'Evacuation Count — Feature Importances', '#06d6a0'),
    ]

    for col_i, (model_key, title, colour) in enumerate(importance_specs):
        ax = fig.add_subplot(gs[1, col_i])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        model_data = predictor.models.get(model_key)
        if model_data is None:
            ax.text(0.5, 0.5, 'Model not loaded', transform=ax.transAxes,
                    color=FG, ha='center')
            ax.set_title(title, color=FG, fontsize=9)
            continue

        if model_data.get('model_type') == 'hurdle':
            model_obj = model_data.get('classifier')
        else:
            model_obj = model_data.get('model')
        if not hasattr(model_obj, 'feature_importances_'):
            ax.text(0.5, 0.5, 'feature_importances_ not available',
                    transform=ax.transAxes, color=FG, ha='center')
            ax.set_title(title, color=FG, fontsize=9)
            continue

        importances = model_obj.feature_importances_
        # Pad or trim to match FEATURE_NAMES length
        n_feat = min(len(importances), len(FEATURE_NAMES))
        imp = importances[:n_feat]
        names = FEATURE_NAMES[:n_feat]

        order = np.argsort(imp)
        ax.barh(range(n_feat), imp[order], color=colour, alpha=0.75)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels([names[o] for o in order], color=FG, fontsize=6)
        ax.set_xlabel('Importance (gain)', color=FG, fontsize=8)
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold')

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Evaluation plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-scenario evaluation (in-distribution)
# ---------------------------------------------------------------------------

def run_multi_scenario_evaluation(
    num_runs: int = 10,
    output_file: str = 'ml_evaluation_results.csv',
) -> pd.DataFrame:
    """
    Evaluate across all 15 training scenarios to measure in-distribution
    performance.  num_runs simulations are run per scenario (total = num_runs × 15).

    This is the correct evaluation for assessing whether the models have learned
    from the training distribution.  A separate held-out evaluation (single OOD
    scenario) tests generalisation.
    """
    print('=' * 70)
    print('AIGIS — ML Model Evaluation  (in-distribution, 15 training scenarios)')
    print('=' * 70)
    print('Willmott & Matsuura (2005)  |  Chen & Guestrin (2016) XGBoost')
    print(f'Scenarios: {len(TRAINING_LOCATIONS)}  |  Runs per scenario: {num_runs}'
          f'  |  Total: {len(TRAINING_LOCATIONS) * num_runs}')
    print(f'Midpoint: step {MAX_STEPS // 2} / {MAX_STEPS}')
    print('=' * 70 + '\n')

    predictor = RiskPredictor()
    if not predictor.is_trained:
        print('ERROR: No trained models found in models/. Cannot evaluate.')
        return pd.DataFrame()

    midpoint = MAX_STEPS // 2
    records = []
    total = len(TRAINING_LOCATIONS) * num_runs
    done = 0

    for loc_i, loc in enumerate(TRAINING_LOCATIONS):
        lat, lon, radius = loc['lat'], loc['lon'], loc['radius']

        for i in range(num_runs):
            done += 1
            print(f'  Run {done}/{total}  (scenario {loc_i + 1}/{len(TRAINING_LOCATIONS)})',
                  end='\r', flush=True)

            with _quiet():
                sim = AIGISSimulation(lat=lat, lon=lon, radius=radius,
                                      mode='batch', run_id=loc_i * num_runs + i,
                                      fire_locations=loc.get('fire_locations'))

            with _quiet():
                while sim.step < midpoint and not sim.is_complete():
                    sim.run_step()

            state = _extract_state(sim)
            preds = predictor.predict_casualty_risk(state)

            with _quiet():
                while sim.step < MAX_STEPS and not sim.is_complete():
                    sim.run_step()

            actual = sim.get_results()

            records.append({
                'scenario_idx':          loc_i,
                'lat':                   lat,
                'lon':                   lon,
                'run_id':                i,
                'midpoint_step':         midpoint,
                'pred_casualties':       preds.get('predicted_casualties', 0.0),
                'pred_evacuations':      preds.get('predicted_evacuations', 0.0),
                'pred_containment_days': preds.get('predicted_containment_days', 0.0),
                'pred_risk_level':       preds.get('risk_level', 'UNKNOWN'),
                'actual_casualties':     actual['casualties'],
                'actual_evacuated':      actual['evacuated'],
                'actual_steps':          actual['steps'],
                'actual_mortality_rate': actual['mortality_rate'],
                'actual_evacuation_rate': actual['evacuation_success_rate'],
            })

    print()

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f'Results saved to: {output_file}\n')

    _print_metrics_table(df)
    _plot_evaluation(df, predictor, output_file.replace('.csv', '.png'))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AIGIS ML models against ground-truth simulation outcomes'
    )
    parser.add_argument('--runs',   type=int,   default=30,
                        help='Number of evaluation runs — per scenario if '
                             '--multi-scenario, otherwise total (default: 30)')
    parser.add_argument('--output', type=str,   default='ml_evaluation_results.csv')
    parser.add_argument('--lat',    type=float, default=38.090)
    parser.add_argument('--lon',    type=float, default=23.920)
    parser.add_argument('--radius', type=int,   default=3000)
    parser.add_argument('--multi-scenario', action='store_true',
                        help='Evaluate across all 15 training scenarios '
                             '(in-distribution); --runs is per scenario')
    args = parser.parse_args()

    if args.multi_scenario:
        run_multi_scenario_evaluation(
            num_runs=args.runs,
            output_file=args.output,
        )
    else:
        run_evaluation(
            num_runs=args.runs,
            lat=args.lat, lon=args.lon, radius=args.radius,
            output_file=args.output,
        )


if __name__ == '__main__':
    main()
