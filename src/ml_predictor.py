"""
Machine Learning Predictor for AIGIS.

Integrates trained ML models to predict:
- Casualty risk
- Evacuation requirements
- Containment time
- Financial cost

Uses 14 simulation-derived features (no hardcoded lat/lon/month).
Used by Commander agent for enhanced decision-making.
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
                    print(f"  Warning: Failed to load {model_name}: {e}")

        if loaded_count > 0:
            self.is_trained = True
            print(f"  ML Predictor initialized with {loaded_count}/4 models")
        else:
            print("  No ML models found. Run 'python train_models.py' first.")

    def predict_casualty_risk(self, simulation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict casualty risk and other metrics for current simulation state.

        Args:
            simulation_state: dict with keys:
                fire_grid, fuel_type_grid, elevation_grid,
                wind_speed, wind_direction, humidity,
                tti_minutes, ect_minutes, current_phase,
                step, max_steps, agents

        Returns:
            Dictionary with predictions:
            - predicted_casualties, predicted_evacuations,
              predicted_containment_days, predicted_cost, risk_level
        """
        if not self.is_trained:
            return self._get_default_predictions()

        try:
            features = self._extract_features(simulation_state)

            predictions = {}

            for model_key, out_key in [
                ('casualty_risk', 'predicted_casualties'),
                ('evacuation_count', 'predicted_evacuations'),
                ('containment_time', 'predicted_containment_days'),
                ('financial_cost', 'predicted_cost'),
            ]:
                if model_key in self.models:
                    model_data = self.models[model_key]
                    X_scaled = model_data['scaler'].transform([features])
                    value = model_data['model'].predict(X_scaled)[0]
                    predictions[out_key] = max(0, float(value))
                else:
                    predictions[out_key] = 0.0

            predictions['risk_level'] = self._calculate_risk_level(predictions)
            return predictions

        except Exception as e:
            return self._get_default_predictions()

    def _extract_features(self, simulation_state: Dict[str, Any]) -> list:
        """
        Extract 14 simulation-derived ML features.

        Feature order must match ML_FEATURE_NAMES in config.py:
        burning_cells_pct, burnt_cells_pct, wind_speed, wind_dir_x, wind_dir_y,
        mean_slope, dominant_fuel_type, active_rescuers, civilians_remaining,
        current_phase, tti_normalized, ect_normalized, step_normalized, humidity
        """
        fire_grid = simulation_state.get('fire_grid')
        fuel_type_grid = simulation_state.get('fuel_type_grid')
        elevation_grid = simulation_state.get('elevation_grid')
        wind_speed = float(simulation_state.get('wind_speed', 5.0))
        wind_direction = simulation_state.get('wind_direction', [1.0, 0.0])
        humidity = float(simulation_state.get('humidity', 30.0))
        tti_minutes = float(simulation_state.get('tti_minutes', float('inf')))
        ect_minutes = float(simulation_state.get('ect_minutes', 0.0))
        current_phase = int(simulation_state.get('current_phase', 0))
        step = int(simulation_state.get('step', 0))
        max_steps = int(simulation_state.get('max_steps', 500))
        agents = simulation_state.get('agents', {})

        total_cells = fire_grid.size if fire_grid is not None else 1

        # 1. burning_cells_pct
        burning_cells_pct = float(np.sum(fire_grid == 1)) / total_cells if fire_grid is not None else 0.0

        # 2. burnt_cells_pct
        burnt_cells_pct = float(np.sum(fire_grid == 2)) / total_cells if fire_grid is not None else 0.0

        # 3-4-5. wind_speed, wind_dir_x, wind_dir_y
        wind_dir = wind_direction if wind_direction is not None else [1.0, 0.0]
        wind_dir_x = float(wind_dir[0])
        wind_dir_y = float(wind_dir[1])

        # 6. mean_slope in burning area
        if elevation_grid is not None and fire_grid is not None:
            burning_mask = (fire_grid == 1)
            if np.any(burning_mask):
                # Compute gradient magnitude at burning cells
                grad_y, grad_x = np.gradient(elevation_grid)
                slope_mag = np.sqrt(grad_y**2 + grad_x**2)
                mean_slope = float(np.mean(slope_mag[burning_mask]))
            else:
                mean_slope = 0.0
        else:
            mean_slope = 0.0

        # 7. dominant_fuel_type in burning area
        if fuel_type_grid is not None and fire_grid is not None:
            burning_mask = (fire_grid == 1)
            if np.any(burning_mask):
                fuels = fuel_type_grid[burning_mask]
                values, counts = np.unique(fuels, return_counts=True)
                dominant_fuel_type = float(values[np.argmax(counts)])
            else:
                from .config import DEFAULT_FUEL_MODEL
                dominant_fuel_type = float(DEFAULT_FUEL_MODEL)
        else:
            dominant_fuel_type = 5.0

        # 8. active_rescuers
        rescuers = agents.get('rescuers', [])
        active_rescuers = float(len([r for r in rescuers if getattr(r, 'is_active', True)]))

        # 9. civilians_remaining
        civilians = agents.get('civilians', [])
        civilians_remaining = float(sum(1 for c in civilians if getattr(c, 'is_active', True)))

        # 10. current_phase
        phase = float(current_phase)

        # 11. tti_normalized (clip tti/60 to [0,1])
        if tti_minutes == float('inf') or tti_minutes != tti_minutes:
            tti_normalized = 0.0
        else:
            tti_normalized = float(np.clip(tti_minutes / 60.0, 0.0, 1.0))

        # 12. ect_normalized (clip ect/30 to [0,1])
        ect_normalized = float(np.clip(ect_minutes / 30.0, 0.0, 1.0))

        # 13. step_normalized
        step_normalized = float(step) / float(max_steps) if max_steps > 0 else 0.0

        # 14. humidity
        humidity_feat = float(humidity)

        return [
            burning_cells_pct, burnt_cells_pct,
            wind_speed, wind_dir_x, wind_dir_y,
            mean_slope, dominant_fuel_type,
            active_rescuers, civilians_remaining,
            phase,
            tti_normalized, ect_normalized, step_normalized,
            humidity_feat,
        ]

    def _calculate_risk_level(self, predictions: Dict) -> str:
        """Determine overall risk level from predictions."""
        casualties = predictions.get('predicted_casualties', 0)
        evacuations = predictions.get('predicted_evacuations', 0)

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
