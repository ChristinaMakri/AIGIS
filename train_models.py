"""
Train machine learning models for wildfire prediction.

Models trained:
1. Casualty Risk Predictor - Predicts number of casualties based on fire characteristics
2. Evacuation Count Predictor - Predicts number of evacuations needed
3. Containment Time Predictor - Predicts days to contain fire
4. Financial Cost Predictor - Predicts estimated financial damage

Uses XGBoost and Random Forest algorithms on historical fire data.
"""
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


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🔥 ML MODEL TRAINING FOR WILDFIRE PREDICTION")
    print("=" * 70)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load data
    print("\n📂 Loading historical fire data...")
    df = load_fire_data()

    # Prepare features
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
