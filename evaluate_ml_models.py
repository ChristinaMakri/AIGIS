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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.simulation import AIGISSimulation
from src.ml_predictor import RiskPredictor
from src.config import MAX_STEPS

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

    # Evaluable models (have ground truth)
    evaluable = [
        ('Casualty Risk',    'pred_casualties',  'actual_casualties',
         'predicted_casualties vs. actual casualties'),
        ('Evacuation Count', 'pred_evacuations', 'actual_evacuated',
         'predicted_evacuations vs. actual evacuated civilians'),
    ]

    for label, pred_col, actual_col, desc in evaluable:
        y_pred = df[pred_col].values
        y_true = df[actual_col].values
        mae    = mean_absolute_error(y_true, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        r2     = r2_score(y_true, y_pred)
        bias   = np.mean(y_pred - y_true)

        print(f'\n{label}  ({desc}):')
        print(f'  MAE   = {mae:.3f}   (Willmott & Matsuura 2005)')
        print(f'  RMSE  = {rmse:.3f}')
        print(f'  R²    = {r2:.4f}   (Nagelkerke 1991)')
        print(f'  Bias  = {bias:+.3f}  (positive = over-prediction)')

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
    print('R² interpretation: 1.0 = perfect, 0.0 = no better than mean,')
    print('  < 0 = worse than predicting the mean.')
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

        # R² annotation
        r2 = r2_score(y, x)
        mae = mean_absolute_error(y, x)
        ax.text(0.05, 0.92, f'R² = {r2:.3f}\nMAE = {mae:.2f}',
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

        model_obj = model_data['model']
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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AIGIS ML models against ground-truth simulation outcomes'
    )
    parser.add_argument('--runs',   type=int,   default=30,
                        help='Number of evaluation runs (default: 30)')
    parser.add_argument('--output', type=str,   default='ml_evaluation_results.csv')
    parser.add_argument('--lat',    type=float, default=38.090)
    parser.add_argument('--lon',    type=float, default=23.920)
    parser.add_argument('--radius', type=int,   default=3000)
    args = parser.parse_args()

    run_evaluation(
        num_runs=args.runs,
        lat=args.lat, lon=args.lon, radius=args.radius,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
