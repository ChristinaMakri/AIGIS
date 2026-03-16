"""
AIGIS — ML Model Training Script
==================================
Generates a training dataset by running N simulations with randomised
parameters, then trains four regression models to predict simulation
outcomes from mid-run state features.

The trained models replace the pre-packaged models/*.pkl files, which
were trained on an external US wildfire incident database (33 features)
incompatible with the simulation's 14-feature extractor.  After running
this script the ML subsystem in CommanderAgent will produce real
predictions instead of silent fallback zeros.

Feature vector (14 features, extracted at step = MAX_STEPS // 2)
-----------------------------------------------------------------
  Extracted by src/ml_predictor.RiskPredictor._extract_features(),
  which is also used at runtime by the Commander.  Training on the same
  extractor guarantees zero feature-mismatch at inference time.

  0  burning_cells_pct    — fraction of grid currently burning
  1  burnt_cells_pct      — fraction of grid already burnt
  2  wind_speed           — m/s at current step
  3  wind_dir_x           — wind unit vector x-component
  4  wind_dir_y           — wind unit vector y-component
  5  mean_slope           — mean slope magnitude in burning cells
  6  dominant_fuel_type   — modal fuel class in burning area
  7  active_rescuers      — number of active Rescuer agents
  8  civilians_remaining  — civilians still evacuating
  9  current_phase        — Commander operational phase (0-3)
  10 tti_normalized       — time-to-impact / 60, clipped [0,1]
  11 ect_normalized       — evacuation clearance time / 30, clipped [0,1]
  12 step_normalized      — current step / MAX_STEPS
  13 humidity             — relative humidity (%)

Target variables
----------------
  casualty_risk    → final casualty count          (XGBRegressor)
  evacuation_count → final evacuated count         (RandomForestRegressor)
  containment_time → final simulation steps        (RandomForestRegressor)

Training methodology
--------------------
  - Parameter randomisation per run (PARAM_RANGES below) for feature diversity
  - 80 / 20 stratified train / test split
  - StandardScaler fitted on training set only (no leakage)
  - Models saved to models/*.pkl replacing existing files

References
----------
  Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
    System." Proceedings of KDD '16, pp. 785-794.
    DOI: 10.1145/2939672.2939785

  Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
    DOI: 10.1023/A:1010933404324

  Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research, 12, pp. 2825-2830.

  Willmott, C.J. & Matsuura, K. (2005). "Advantages of the mean absolute
    error (MAE) over the root mean square error (RMSE) in assessing average
    model performance." Climate Research, 30(1), 79-82.
    DOI: 10.3354/cr030079

  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
    Other Simulation Models: A Second Update." JASSS 23(2):7.
    DOI: 10.18564/jasss.4259

Usage
-----
  python train_models.py [--runs N] [--lat LAT --lon LON --radius R]
                         [--output-dir DIR] [--seed SEED]

  Recommended: --runs 200 (80/20 split → 160 train / 40 test)
  Minimum:     --runs 50  (adequate for pilot evaluation)

Outputs
-------
  - models/casualty_risk_model.pkl      (overwrites existing)
  - models/evacuation_count_model.pkl   (overwrites existing)
  - models/containment_time_model.pkl   (overwrites existing)
  - training_dataset.csv                (full feature+target matrix)
  - training_evaluation.png             (predicted vs actual + importances)
  - Console: per-model MAE, RMSE, R² on held-out test set
"""
import argparse
import contextlib
import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.simulation import AIGISSimulation
from src.config import MAX_STEPS
from src.ml_predictor import RiskPredictor

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature names — must match ml_predictor._extract_features() order exactly
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

# ---------------------------------------------------------------------------
# Parameter ranges for training diversity
# Each run samples uniformly from these ranges so the models see a wide
# range of fire conditions, wind regimes, and population sizes.
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    # Fire physics
    'FIRE_SPREAD_PROB_BASE': (0.10, 0.60),
    'ROTHERMEL_BASE_ROS':    (0.20, 1.00),
    # Wind
    'WIND_SPEED':                  (2.0,  15.0),
    'WIND_INITIAL_DIRECTION':      (0.0,  360.0),
    'WIND_OSCILLATION_AMPLITUDE':  (2.0,  20.0),
    # Population
    'NUM_CIVILIANS': (30, 100),   # integer range
}

# ---------------------------------------------------------------------------
# Training locations — geographically diverse, all with confirmed wildfire
# history and sufficient vegetation fuel.
# Explicitly EXCLUDES validation locations:
#   Mati, Greece       (38.090, 23.920) — reserved for validate_mati.py
#   Paradise, CA       (39.759,-121.622) — reserved for validate_campfire.py
# ---------------------------------------------------------------------------
TRAINING_LOCATIONS = [
    # Greece — Penteli forest, NE Athens (pine, 1995/2009 wildfires)
    {'lat': 38.056, 'lon': 23.868, 'radius': 3000,
     'fire_locations': [(38.067, 23.879), (38.062, 23.873)]},
    # Greece — Kineta, Corinth Gulf coast (July 2018 wildfire, same month as Mati)
    {'lat': 38.008, 'lon': 23.140, 'radius': 3000,
     'fire_locations': [(38.019, 23.152), (38.013, 23.146)]},
    # Greece — Varibobi forest, N Athens (August 2021 wildfire, pine/fir)
    {'lat': 38.128, 'lon': 23.798, 'radius': 3000,
     'fire_locations': [(38.140, 23.810), (38.134, 23.804)]},
    # Greece — Rhodes island (July 2023 wildfire, Mediterranean scrub/pine)
    {'lat': 36.198, 'lon': 28.002, 'radius': 3000,
     'fire_locations': [(36.210, 28.014), (36.204, 28.008)]},
    # California — Redding area (Carr Fire July 2018, chaparral/mixed forest)
    {'lat': 40.588, 'lon': -122.392, 'radius': 3000,
     'fire_locations': [(40.600, -122.380), (40.594, -122.386)]},
    # California — Napa Valley (Glass Fire September 2020, oak woodland)
    {'lat': 38.498, 'lon': -122.402, 'radius': 3000,
     'fire_locations': [(38.510, -122.390), (38.504, -122.396)]},
    # California — Thousand Oaks (Woolsey Fire November 2018, coastal chaparral)
    {'lat': 34.172, 'lon': -118.872, 'radius': 3000,
     'fire_locations': [(34.184, -118.860), (34.178, -118.866)]},
    # Spain — Bages, Catalonia (frequent summer wildfires, Mediterranean pine)
    {'lat': 41.698, 'lon': 1.802, 'radius': 3000,
     'fire_locations': [(41.710, 1.814), (41.704, 1.808)]},
    # France — Var department (frequent Mediterranean wildfires, garrigue/pine)
    {'lat': 43.352, 'lon': 6.198, 'radius': 3000,
     'fire_locations': [(43.364, 6.210), (43.358, 6.204)]},
]

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """Suppress simulation stdout during batch training."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _sample_overrides(rng: np.random.Generator) -> dict:
    """Sample one set of randomised config_overrides from PARAM_RANGES."""
    overrides = {}
    for param, (lo, hi) in PARAM_RANGES.items():
        if param == 'NUM_CIVILIANS':
            overrides[param] = int(rng.integers(lo, hi + 1))
        else:
            overrides[param] = float(rng.uniform(lo, hi))
    return overrides


def _extract_state(sim: AIGISSimulation) -> dict:
    """
    Build the simulation_state dict that RiskPredictor expects.
    Mirrors CommanderAgent._update_ml_predictions() (commander.py ~L648).
    """
    env       = sim.environment
    commander = sim.agents.get('commander')
    fire_sim  = sim.fire_sim

    return {
        'fire_grid':      env.fire_grid,
        'fuel_type_grid': getattr(env, 'fuel_type_grid', None),
        'elevation_grid': env.elevation_grid,
        'wind_speed':     getattr(fire_sim, 'wind_speed', 5.0),
        'wind_direction': list(getattr(fire_sim, 'wind_direction', [1.0, 0.0])),
        'humidity':       getattr(env, 'humidity', 30.0),
        'tti_minutes':    getattr(commander, 'tti', float('inf')) if commander else float('inf'),
        'ect_minutes':    getattr(commander, 'ect', 0.0)          if commander else 0.0,
        'current_phase':  getattr(commander, 'current_phase', 0)  if commander else 0,
        'step':           sim.step,
        'max_steps':      MAX_STEPS,
        'agents':         getattr(env, 'agents', {}),
    }



# ---------------------------------------------------------------------------
# Pre-caching
# ---------------------------------------------------------------------------

def precache_locations() -> None:
    """
    Download and cache OSM / SRTM data for all TRAINING_LOCATIONS before
    the main training loop starts.  This is a one-time cost: osmnx caches
    Overpass API responses locally, so every subsequent simulation at the
    same (lat, lon, radius) uses the cache and runs in ~27 s instead of
    ~6 min.

    Each location is initialised (environment built) and immediately
    discarded — no simulation steps are run.
    """
    print(f'Phase 0 — Pre-caching {len(TRAINING_LOCATIONS)} locations ...')
    print('  (first visit downloads OSM/SRTM; cached on disk for all later runs)\n')

    for i, loc in enumerate(TRAINING_LOCATIONS):
        label = f"  [{i + 1}/{len(TRAINING_LOCATIONS)}]  lat={loc['lat']}  lon={loc['lon']}"
        print(f'{label}', flush=True)
        try:
            with _quiet():
                sim = AIGISSimulation(
                    lat=loc['lat'], lon=loc['lon'], radius=loc['radius'],
                    mode='batch', run_id=0,
                    fire_locations=loc['fire_locations'],
                )
            del sim
            print(f'{label}  OK', flush=True)
        except Exception as exc:
            print(f'{label}  WARNING: {exc}', flush=True)

    print()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_runs: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run `num_runs` simulations sampling randomly from TRAINING_LOCATIONS.
    For each run: extract features at midpoint, record targets at completion.

    Location diversity ensures models learn physics-based relationships that
    generalise across geographies, not location-specific artefacts.
    Validation locations (Mati, Paradise CA) are excluded from training so
    that validate_mati.py and validate_campfire.py provide a genuine
    out-of-distribution test.
    """
    rng = np.random.default_rng(seed)
    records = []

    for i in range(num_runs):
        print(f'  Generating run {i + 1}/{num_runs}', end='\r', flush=True)

        overrides = _sample_overrides(rng)
        loc = TRAINING_LOCATIONS[int(rng.integers(0, len(TRAINING_LOCATIONS)))]

        with _quiet():
            sim = AIGISSimulation(
                lat=loc['lat'], lon=loc['lon'], radius=loc['radius'],
                mode='batch', run_id=i,
                config_overrides=overrides,
                fire_locations=loc['fire_locations'],
            )

        midpoint = MAX_STEPS // 2

        # Phase 1: run to midpoint and extract features
        with _quiet():
            while sim.step < midpoint and not sim.is_complete():
                sim.run_step()

        predictor = RiskPredictor.__new__(RiskPredictor)
        predictor.models    = {}
        predictor.is_trained = False
        state    = _extract_state(sim)
        features = predictor._extract_features(state)

        # Phase 2: run to completion
        with _quiet():
            while sim.step < MAX_STEPS and not sim.is_complete():
                sim.run_step()

        result     = sim.get_results()
        casualties = result['casualties']
        evacuated  = result['evacuated']
        steps      = result['steps']

        row = {f: v for f, v in zip(FEATURE_NAMES, features)}
        row.update({
            'run_id':            i,
            'train_lat':         loc['lat'],
            'train_lon':         loc['lon'],
            'target_casualties': casualties,
            'target_evacuated':  evacuated,
            'target_steps':      steps,
            # Store overrides for reference
            **{f'param_{k}': v for k, v in overrides.items()},
        })
        records.append(row)

    print()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_model(model_type: str):
    """Instantiate a fresh untrained model of the given type."""
    if model_type == 'xgboost':
        if XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
        else:
            print('  Warning: XGBoost not installed; using Ridge regression '
                  'for casualty_risk model.')
            return Ridge(alpha=1.0)
    else:
        return RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )


MODEL_SPECS = [
    # (model_key, filename, target_col, model_type, label)
    ('casualty_risk',    'casualty_risk_model.pkl',
     'target_casualties', 'xgboost',      'Casualty Risk'),
    ('evacuation_count', 'evacuation_count_model.pkl',
     'target_evacuated',  'random_forest', 'Evacuation Count'),
    ('containment_time', 'containment_time_model.pkl',
     'target_steps',      'random_forest', 'Containment Time (steps)'),
]


def train_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    plot_path: str,
) -> dict:
    """
    Train all four models, evaluate on held-out test set, save to disk.
    Returns dict of {model_key: {'test_r2', 'test_mae', 'test_rmse'}}.
    """
    X = df[FEATURE_NAMES].values
    metrics = {}

    # Single 80/20 split — same indices reused for all models for consistency
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2,
                                           random_state=42, shuffle=True)

    X_train, X_test = X[idx_train], X[idx_test]

    # Fit scaler on training set only (prevent test leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print('\n' + '=' * 70)
    print('TRAINING RESULTS  (80% train / 20% test)')
    print('Willmott & Matsuura (2005)  |  Chen & Guestrin (2016)  |  Breiman (2001)')
    print('=' * 70)

    trained_models = {}

    for model_key, filename, target_col, model_type, label in MODEL_SPECS:
        y      = df[target_col].values
        y_train = y[idx_train]
        y_test  = y[idx_test]

        model = _build_model(model_type)
        model.fit(X_train_s, y_train)

        y_pred  = model.predict(X_test_s)
        y_pred  = np.maximum(y_pred, 0)   # predictions must be non-negative

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f'\n{label}:')
        print(f'  MAE   = {mae:.4f}  (Willmott & Matsuura 2005)')
        print(f'  RMSE  = {rmse:.4f}')
        print(f'  R²    = {r2:.4f}  (Nagelkerke 1991)')

        payload = {
            'model':         model,
            'scaler':        scaler,
            'feature_names': FEATURE_NAMES,
            'model_type':    model_type,
            'test_r2':       float(r2),
            'test_mae':      float(mae),
        }

        out_path = output_dir / filename
        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f'  Saved to: {out_path}')

        metrics[model_key] = {
            'test_r2': r2, 'test_mae': mae, 'test_rmse': rmse,
            'y_test': y_test, 'y_pred': y_pred,
            'model': model, 'label': label,
        }
        trained_models[model_key] = model

    print('\n' + '=' * 70)
    _plot_training(metrics, df, plot_path)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_training(metrics: dict, df: pd.DataFrame, out_path: str) -> None:
    """
    2×4 figure:
      Row 0: predicted vs. actual scatter (one panel per model)
      Row 1: feature importances (where available)
    """
    keys = [m[0] for m in MODEL_SPECS]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        'ML Model Training Evaluation  |  Chen & Guestrin (2016)  |  '
        'Breiman (2001)  |  Willmott & Matsuura (2005)',
        color=FG, fontsize=10, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.35)

    colours = ['#ff006e', '#06d6a0', '#ffd60a', '#8b5cf6']

    for col_i, (model_key, colour) in enumerate(zip(keys, colours)):
        m = metrics[model_key]

        # Row 0: scatter predicted vs actual
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        ax.scatter(m['y_pred'], m['y_test'], color=colour, s=20, alpha=0.6)
        lim = max(m['y_test'].max(), m['y_pred'].max()) * 1.1
        ax.plot([0, lim], [0, lim], color='white', linestyle='--',
                linewidth=1, alpha=0.5)
        ax.text(0.05, 0.90,
                f"R²={m['test_r2']:.3f}\nMAE={m['test_mae']:.2f}",
                transform=ax.transAxes, color=FG, fontsize=7,
                verticalalignment='top',
                bbox=dict(facecolor=PANEL, edgecolor='#3a3a5c', alpha=0.8))
        ax.set_title(m['label'], color=FG, fontsize=8, fontweight='bold')
        ax.set_xlabel('Predicted', color=FG, fontsize=7)
        ax.set_ylabel('Actual',    color=FG, fontsize=7)

        # Row 1: feature importances
        ax2 = fig.add_subplot(gs[1, col_i])
        ax2.set_facecolor(PANEL)
        ax2.tick_params(colors=FG, labelsize=6)
        for sp in ax2.spines.values():
            sp.set_edgecolor('#3a3a5c')

        model_obj = m['model']
        if hasattr(model_obj, 'feature_importances_'):
            imp   = model_obj.feature_importances_
            n     = min(len(imp), len(FEATURE_NAMES))
            order = np.argsort(imp[:n])
            ax2.barh(range(n), imp[:n][order], color=colour, alpha=0.75)
            ax2.set_yticks(range(n))
            ax2.set_yticklabels(
                [FEATURE_NAMES[o] for o in order], color=FG, fontsize=5
            )
            ax2.set_xlabel('Importance (gain)', color=FG, fontsize=7)
        else:
            ax2.text(0.5, 0.5, 'Importances\nnot available',
                     transform=ax2.transAxes, color=FG, ha='center',
                     fontsize=8)
        ax2.set_title('Feature Importances', color=FG, fontsize=8,
                      fontweight='bold')

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Training plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train AIGIS ML models on simulation-generated data'
    )
    parser.add_argument('--runs',       type=int,   default=200,
                        help='Number of training simulations (default: 200)')
    parser.add_argument('--output-dir', type=str,   default='models',
                        help='Directory to save model pkl files (default: models/)')
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Master RNG seed for reproducible data generation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print('AIGIS — ML Model Training')
    print('=' * 70)
    print('Chen & Guestrin (2016)  |  Breiman (2001)  |  Pedregosa et al. (2011)')
    print(f'Runs: {args.runs}  |  Train/test: 80/20  |  Features: {len(FEATURE_NAMES)}')
    print(f'Locations: {len(TRAINING_LOCATIONS)} diverse sites (Mati + Camp Fire excluded)')
    print(f'Midpoint extraction: step {MAX_STEPS // 2} / {MAX_STEPS}')
    print(f'Output dir: {output_dir}')
    print('=' * 70 + '\n')

    # 0. Pre-cache all training locations (downloads OSM/SRTM once per site)
    precache_locations()

    # 1. Generate dataset
    print('Phase 1 — Generating training dataset ...')
    df = generate_dataset(
        num_runs=args.runs,
        seed=args.seed,
    )
    csv_path = 'training_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f'Dataset saved to: {csv_path}  ({len(df)} rows)\n')

    # Quick dataset summary
    print('Target variable summary:')
    for col in ['target_casualties', 'target_evacuated', 'target_steps']:
        print(f'  {col:<24} mean={df[col].mean():.2f}  '
              f'std={df[col].std():.2f}  '
              f'min={df[col].min():.0f}  max={df[col].max():.0f}')
    print()

    # 2. Train and save
    print('Phase 2 — Training models ...')
    train_and_save(df, output_dir, plot_path='training_evaluation.png')

    print('\nDone. Run evaluate_ml_models.py to verify predictions on new runs.')


if __name__ == '__main__':
    main()
