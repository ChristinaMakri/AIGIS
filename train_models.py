"""
Train machine learning models for wildfire prediction.

Models trained:
1. Casualty Risk Predictor - Predicts number of casualties based on fire characteristics
2. Evacuation Count Predictor - Predicts number of evacuations needed
3. Containment Time Predictor - Predicts days to contain fire
4. Financial Cost Predictor - Predicts estimated financial damage

Uses XGBoost and Random Forest algorithms on historical fire data
or simulation-derived training data (--sim-data flag).
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
    print("   Using Random Forest for all models.")


def load_fire_data():
    """Load historical fire data from CSV"""
    data_file = Path("data/historical_fires.csv")

    if not data_file.exists():
        raise FileNotFoundError(
            f"\n❌ Fire data not found at {data_file}\n"
            f"   Run 'python collect_fire_data.py' first to download historical data."
        )

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} fire records")

    return df


def prepare_features(df):
    """
    Prepare feature matrix for ML training.

    Features used:
    - fire_size_acres: Size of fire (primary predictor)
    - latitude, longitude: Geographic location
    - month: Season (affects fire behavior)
    - day_of_year: Temporal pattern

    Targets:
    - estimated_casualties
    - estimated_evacuations
    - containment_days
    - estimated_cost
    """
    print("\n🔧 Preparing features...")

    features = [
        'fire_size_acres',
        'latitude',
        'longitude',
        'month',
        'day_of_year'
    ]

    # Remove rows with missing values
    required_cols = features + ['estimated_casualties', 'estimated_evacuations',
                               'containment_days', 'estimated_cost']
    df_clean = df.dropna(subset=required_cols)

    X = df_clean[features].copy()
    y_casualties = df_clean['estimated_casualties'].copy()
    y_evacuations = df_clean['estimated_evacuations'].copy()
    y_containment = df_clean['containment_days'].copy()
    y_cost = df_clean['estimated_cost'].copy()

    print(f"  Features: {features}")
    print(f"  Samples: {len(X)}")
    print(f"  Feature matrix shape: {X.shape}")

    return X, y_casualties, y_evacuations, y_containment, y_cost


def train_model(X, y, model_name, use_xgboost=True):
    """
    Train a regression model using XGBoost or Random Forest.

    Args:
        X: Feature matrix
        y: Target variable
        model_name: Name for display
        use_xgboost: Use XGBoost if available (otherwise Random Forest)

    Returns:
        Trained model, scaler, and evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"🤖 Training: {model_name}")
    print(f"{'='*70}")

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select and train model
    if use_xgboost and XGBOOST_AVAILABLE:
        print("  Algorithm: XGBoost")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("  Algorithm: Random Forest")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  📊 Performance Metrics:")
    print(f"     MAE (Mean Absolute Error):  {mae:.2f}")
    print(f"     RMSE (Root Mean Squared):    {rmse:.2f}")
    print(f"     R² Score:                    {r2:.3f}")

    # Interpret R² score
    if r2 > 0.9:
        print(f"     ✅ Excellent fit!")
    elif r2 > 0.7:
        print(f"     ✅ Good fit")
    elif r2 > 0.5:
        print(f"     ⚠️  Moderate fit")
    else:
        print(f"     ⚠️  Weak fit - consider more features")

    return model, scaler, {'mae': mae, 'rmse': rmse, 'r2': r2}


def generate_training_data(n_runs: int = 100) -> str:
    """
    Generate ML training data by running N headless AIGIS simulations.

    Records 14-feature vector + final outcomes per step per run.
    Saves to data/training/sim_derived.pkl

    Args:
        n_runs: Number of simulation runs (default 100)

    Returns:
        Path to saved training data file
    """
    print(f"\n{'='*70}")
    print(f"Generating simulation-derived training data ({n_runs} runs)...")
    print(f"{'='*70}")

    try:
        from src.simulation import AIGISSimulation
        from src.config import DEFAULT_MAP_CENTER_LAT, DEFAULT_MAP_CENTER_LON, DEFAULT_MAP_RADIUS
        from src.ml_predictor import RiskPredictor
    except ImportError as e:
        print(f"  ERROR: Could not import simulation modules: {e}")
        return ""

    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sim_derived.pkl"

    all_features = []
    all_casualties = []
    all_evacuations = []
    all_containment = []
    all_cost = []

    dummy_predictor = RiskPredictor.__new__(RiskPredictor)
    dummy_predictor.models = {}
    dummy_predictor.is_trained = False

    for run_id in range(n_runs):
        print(f"  Run {run_id + 1}/{n_runs}")
        try:
            sim = AIGISSimulation(
                DEFAULT_MAP_CENTER_LAT, DEFAULT_MAP_CENTER_LON, DEFAULT_MAP_RADIUS,
                mode='batch', run_id=run_id
            )

            step_features = []
            while sim.step < 500 and not sim.is_complete():
                sim.run_step()

                env = sim.environment
                agents = sim.agents
                fire_sim = sim.fire_sim
                commander = agents.get('commander')

                simulation_state = {
                    'fire_grid': env.fire_grid,
                    'fuel_type_grid': getattr(env, 'fuel_type_grid', None),
                    'elevation_grid': env.elevation_grid,
                    'wind_speed': fire_sim.wind_speed,
                    'wind_direction': list(fire_sim.wind_direction),
                    'humidity': getattr(env, 'humidity', 30.0),
                    'tti_minutes': getattr(commander, 'tti', float('inf')),
                    'ect_minutes': getattr(commander, 'ect', 0.0),
                    'current_phase': getattr(commander, 'current_phase', 0),
                    'step': sim.step,
                    'max_steps': 500,
                    'agents': agents,
                }

                features = dummy_predictor._extract_features(simulation_state)
                step_features.append(features)

            # Final outcomes for this run
            results = sim.get_results()
            final_casualties = results['casualties']
            final_evacuated = results['evacuated']
            total_civilians = results['total_civilians']
            # Derive proxy targets
            containment_days = results['steps'] / 50.0  # ~steps to "days" proxy
            cost_proxy = final_casualties * 1e6 + final_evacuated * 1e4

            for feat in step_features:
                all_features.append(feat)
                all_casualties.append(final_casualties)
                all_evacuations.append(final_evacuated)
                all_containment.append(containment_days)
                all_cost.append(cost_proxy)

        except Exception as e:
            print(f"  WARNING: Run {run_id} failed: {e}")
            continue

    if not all_features:
        print("  ERROR: No training data generated")
        return ""

    training_data = {
        'X': np.array(all_features, dtype=np.float32),
        'y_casualties': np.array(all_casualties, dtype=np.float32),
        'y_evacuations': np.array(all_evacuations, dtype=np.float32),
        'y_containment': np.array(all_containment, dtype=np.float32),
        'y_cost': np.array(all_cost, dtype=np.float32),
    }

    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"  Saved {len(all_features)} samples to {output_path}")
    return str(output_path)


def load_sim_derived_data(data_path: str):
    """Load simulation-derived training data from pickle file"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = pd.DataFrame(data['X'])
    y_casualties = pd.Series(data['y_casualties'])
    y_evacuations = pd.Series(data['y_evacuations'])
    y_containment = pd.Series(data['y_containment'])
    y_cost = pd.Series(data['y_cost'])

    print(f"  Loaded simulation-derived data: {len(X)} samples, {X.shape[1]} features")
    return X, y_casualties, y_evacuations, y_containment, y_cost


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description="Train ML models for AIGIS wildfire prediction"
    )
    parser.add_argument('--sim-data', action='store_true',
                        help='Use simulation-derived data instead of historical CSV')
    parser.add_argument('--n-runs', type=int, default=100,
                        help='Number of simulation runs for data generation (default: 100)')
    parser.add_argument('--sim-data-path', type=str, default='data/training/sim_derived.pkl',
                        help='Path to simulation-derived data (if already generated)')
    args = parser.parse_args()

    print("=" * 70)
    print("ML MODEL TRAINING FOR WILDFIRE PREDICTION")
    print("=" * 70)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load data
    if args.sim_data:
        sim_data_path = Path(args.sim_data_path)
        if not sim_data_path.exists():
            print(f"\nGenerating simulation data ({args.n_runs} runs)...")
            sim_data_path_str = generate_training_data(n_runs=args.n_runs)
            if not sim_data_path_str:
                print("ERROR: Failed to generate training data")
                return
            sim_data_path = Path(sim_data_path_str)

        print(f"\nLoading simulation-derived training data from {sim_data_path}...")
        X, y_casualties, y_evacuations, y_containment, y_cost = load_sim_derived_data(str(sim_data_path))
    else:
        print("\nLoading historical fire data...")
        df = load_fire_data()
        # Prepare features from historical data
        X, y_casualties, y_evacuations, y_containment, y_cost = prepare_features(df)

    # Train 4 models
    models = {}

    # Model 1: Casualty Risk Predictor
    casualty_model, casualty_scaler, casualty_metrics = train_model(
        X, y_casualties, "Casualty Risk Predictor", use_xgboost=True
    )
    models['casualty_risk'] = {
        'model': casualty_model,
        'scaler': casualty_scaler,
        'metrics': casualty_metrics
    }

    # Model 2: Evacuation Count Predictor
    evacuation_model, evacuation_scaler, evacuation_metrics = train_model(
        X, y_evacuations, "Evacuation Count Predictor", use_xgboost=False
    )
    models['evacuation_count'] = {
        'model': evacuation_model,
        'scaler': evacuation_scaler,
        'metrics': evacuation_metrics
    }

    # Model 3: Containment Time Predictor
    containment_model, containment_scaler, containment_metrics = train_model(
        X, y_containment, "Containment Time Predictor", use_xgboost=False
    )
    models['containment_time'] = {
        'model': containment_model,
        'scaler': containment_scaler,
        'metrics': containment_metrics
    }

    # Model 4: Financial Cost Predictor
    cost_model, cost_scaler, cost_metrics = train_model(
        X, y_cost, "Financial Cost Predictor", use_xgboost=False
    )
    models['financial_cost'] = {
        'model': cost_model,
        'scaler': cost_scaler,
        'metrics': cost_metrics
    }

    # Save all models
    print(f"\n{'='*70}")
    print("💾 Saving trained models...")

    for model_name, model_data in models.items():
        output_file = models_dir / f"{model_name}_model.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  ✅ {output_file}")

    # Print summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Dataset: {len(X)} fire incidents")
    print(f"  Features: {list(X.columns)}")
    print(f"  Models trained: {len(models)}")
    print(f"\n  Model Performance (R² scores):")
    for model_name, model_data in models.items():
        r2 = model_data['metrics']['r2']
        status = "✅" if r2 > 0.7 else "⚠️"
        print(f"    {status} {model_name:25s}: R²={r2:.3f}")

    print(f"\n{'='*70}")
    print("✅ Training complete! Models saved to ./models/")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  1. Run simulation: python main.py")
    print("  2. Commander agent will automatically use ML predictions")
    print("  3. Check logs for ML prediction outputs")


if __name__ == "__main__":
    main()
