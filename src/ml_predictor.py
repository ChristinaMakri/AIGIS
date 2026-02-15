"""
Machine Learning Predictor for AIGIS.

Integrates trained ML models to predict:
- Casualty risk
- Evacuation requirements
- Containment time
- Financial cost

Used by Commander agent for enhanced decision-making with real historical data.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class RiskPredictor:
    """ML-based risk prediction for wildfire scenarios"""

    def __init__(self):
        """Initialize predictor and load trained models"""
        self.models = {}
        self.is_trained = False
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk"""
        models_dir = Path(__file__).parent.parent / "models"

        model_files = {
            'casualty_risk': 'casualty_risk_model.pkl',
            'evacuation_count': 'evacuation_count_model.pkl',
            'containment_time': 'containment_time_model.pkl',
            'financial_cost': 'financial_cost_model.pkl'
        }

        loaded_count = 0
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")

        if loaded_count > 0:
            self.is_trained = True
            print(f"🤖 ML Predictor initialized with {loaded_count}/4 models")
        else:
            print("⚠️  No ML models found. Run 'python train_models.py' first.")
            print("   Simulation will use physics-based predictions only.")

    def predict_casualty_risk(
        self,
        fire_grid: np.ndarray,
        population_density: np.ndarray,
        wind_speed: float,
        temperature: float = 25.0,
        humidity: float = 30.0,
        evacuation_status: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Predict casualty risk and other metrics for current scenario.

        Args:
            fire_grid: Fire intensity grid (2D array)
            population_density: Population density grid (2D array)
            wind_speed: Current wind speed (m/s)
            temperature: Temperature (°C)
            humidity: Relative humidity (%)
            evacuation_status: Dict with 'evacuated' and 'total' counts

        Returns:
            Dictionary with predictions:
            - predicted_casualties: Expected number of casualties
            - predicted_evacuations: Required evacuation count
            - predicted_containment_days: Days to contain fire
            - predicted_cost: Estimated financial cost ($)
            - risk_level: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        if not self.is_trained:
            return self._get_default_predictions()

        try:
            # Extract features from current simulation state
            features = self._extract_features(
                fire_grid, population_density, wind_speed,
                temperature, humidity, evacuation_status
            )

            predictions = {}

            # Casualty prediction
            if 'casualty_risk' in self.models:
                model_data = self.models['casualty_risk']
                X_scaled = model_data['scaler'].transform([features])
                casualties = model_data['model'].predict(X_scaled)[0]
                predictions['predicted_casualties'] = max(0, float(casualties))
            else:
                predictions['predicted_casualties'] = 0.0

            # Evacuation prediction
            if 'evacuation_count' in self.models:
                model_data = self.models['evacuation_count']
                X_scaled = model_data['scaler'].transform([features])
                evacuations = model_data['model'].predict(X_scaled)[0]
                predictions['predicted_evacuations'] = max(0, float(evacuations))
            else:
                predictions['predicted_evacuations'] = 0.0

            # Containment time prediction
            if 'containment_time' in self.models:
                model_data = self.models['containment_time']
                X_scaled = model_data['scaler'].transform([features])
                containment = model_data['model'].predict(X_scaled)[0]
                predictions['predicted_containment_days'] = max(0, float(containment))
            else:
                predictions['predicted_containment_days'] = 0.0

            # Financial cost prediction
            if 'financial_cost' in self.models:
                model_data = self.models['financial_cost']
                X_scaled = model_data['scaler'].transform([features])
                cost = model_data['model'].predict(X_scaled)[0]
                predictions['predicted_cost'] = max(0, float(cost))
            else:
                predictions['predicted_cost'] = 0.0

            # Determine overall risk level
            predictions['risk_level'] = self._calculate_risk_level(predictions)

            return predictions

        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return self._get_default_predictions()

    def _extract_features(
        self,
        fire_grid: np.ndarray,
        population_density: np.ndarray,
        wind_speed: float,
        temperature: float,
        humidity: float,
        evacuation_status: Optional[Dict]
    ) -> list:
        """
        Extract ML features from simulation state.

        Features must match training data format:
        - fire_size_acres
        - latitude (placeholder: grid center)
        - longitude (placeholder: grid center)
        - month (placeholder: assume summer)
        - day_of_year (placeholder: assume peak season)
        """
        # Calculate fire size from grid
        fire_cells = np.sum(fire_grid > 0)
        # Convert grid cells to acres (assume each cell ~ 50m x 50m = 0.62 acres)
        fire_size_acres = fire_cells * 0.62

        # Use placeholder location features (these would ideally come from simulation config)
        # For now, use mid-latitude values common in fire-prone regions
        latitude = 38.0
        longitude = -120.0

        # Temporal features (assume peak fire season)
        month = 7  # July
        day_of_year = 200  # Mid-summer

        # Return features in the same order as training
        return [
            fire_size_acres,
            latitude,
            longitude,
            month,
            day_of_year
        ]

    def _calculate_risk_level(self, predictions: Dict) -> str:
        """
        Determine overall risk level from predictions.

        Combines casualty and evacuation predictions into categorical risk level.

        Args:
            predictions: Dictionary with prediction values

        Returns:
            Risk level: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        """
        casualties = predictions.get('predicted_casualties', 0)
        evacuations = predictions.get('predicted_evacuations', 0)

        # Risk thresholds based on impact severity
        if casualties > 10 or evacuations > 1000:
            return 'CRITICAL'
        elif casualties > 5 or evacuations > 500:
            return 'HIGH'
        elif casualties > 1 or evacuations > 100:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_default_predictions(self) -> Dict[str, Any]:
        """Return default predictions when models unavailable"""
        return {
            'predicted_casualties': 0.0,
            'predicted_evacuations': 0.0,
            'predicted_containment_days': 0.0,
            'predicted_cost': 0.0,
            'risk_level': 'UNKNOWN'
        }


# Global flag for ML availability
ML_AVAILABLE = True

try:
    _test_predictor = RiskPredictor()
    if not _test_predictor.is_trained:
        ML_AVAILABLE = False
except Exception:
    ML_AVAILABLE = False
