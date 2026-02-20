"""
Step-Ahead Fire Predictor for AIGIS.

Uses an XGBoost binary classifier to predict, for each grid cell,
whether it will be burning/burnt in H steps given the current state.

Used by AnalystAgent to produce better TTI estimates than pure Rothermel.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from scipy.signal import convolve2d


_MODEL_PATH = Path(__file__).parent.parent / "models" / "step_ahead_predictor.pkl"


class StepAheadPredictor:
    """
    Predicts which cells will be burning/burnt H steps into the future.

    7 features per cell:
      0  fire_state              (0=no-fuel/empty, 1=burning, 2=burnt, 3=fuel)
      1  n_burning_neighbors     (0-8, via convolution)
      2  wind_alignment          (dot product of cell→wind-direction and spread direction)
      3  slope                   (gradient magnitude at cell)
      4  fuel_type               (NFFL fuel model integer, 0=no fuel)
      5  dist_to_burning_centroid (Euclidean distance in grid cells, normalised by max)
      6  humidity_proxy          (constant per-step, e.g. 30.0)
    """

    def __init__(self):
        self.model = None
        self._try_load()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _try_load(self):
        """Load cached model if it exists."""
        if _MODEL_PATH.exists():
            try:
                with open(_MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def _build_feature_matrix(
        self,
        fire_grid: np.ndarray,
        wind_vec: np.ndarray,
        slope_grid: np.ndarray,
        fuel_grid: np.ndarray,
        humidity: float = 30.0,
    ) -> np.ndarray:
        """
        Build (H*W, 7) feature matrix for every cell in the grid.

        Args:
            fire_grid:  (H, W) int8 — fire states (0-3)
            wind_vec:   (2,) float — normalised wind direction [dx, dy]
            slope_grid: (H, W) float — terrain slope magnitude per cell
            fuel_grid:  (H, W) int8 — fuel type per cell (0=no fuel)
            humidity:   float — ambient humidity (%)

        Returns:
            (H*W, 7) float32 feature matrix
        """
        H, W = fire_grid.shape

        # --- feature 0: fire_state (normalised to [0,1]) ---
        feat_state = (fire_grid / 3.0).astype(np.float32).ravel()

        # --- feature 1: burning-neighbour count via convolution ---
        burning_mask = (fire_grid == 1).astype(np.float32)
        kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=np.float32)
        n_neighbors = convolve2d(burning_mask, kernel, mode='same',
                                  boundary='fill', fillvalue=0)
        feat_neighbors = (n_neighbors / 8.0).astype(np.float32).ravel()

        # --- feature 2: wind alignment ---
        # For each cell compute an average "alignment" with wind_vec
        # using the gradient of the fire mask as a proxy spread direction
        grad_y, grad_x = np.gradient(burning_mask)
        # dot product with wind
        wx, wy = float(wind_vec[0]), float(wind_vec[1])
        wind_align = (grad_x * wx + grad_y * wy).astype(np.float32)
        # normalise to [-1, 1]
        max_wa = np.abs(wind_align).max()
        if max_wa > 0:
            wind_align = wind_align / max_wa
        feat_wind = wind_align.ravel()

        # --- feature 3: slope ---
        max_slope = slope_grid.max()
        if max_slope > 0:
            feat_slope = (slope_grid / max_slope).astype(np.float32).ravel()
        else:
            feat_slope = slope_grid.astype(np.float32).ravel()

        # --- feature 4: fuel type (normalised by 10) ---
        feat_fuel = (fuel_grid / 10.0).astype(np.float32).ravel()

        # --- feature 5: distance to centroid of burning cells (normalised) ---
        burning_positions = np.argwhere(fire_grid == 1)
        if len(burning_positions) > 0:
            centroid = burning_positions.mean(axis=0)  # (row, col)
            rows = np.arange(H)[:, None] * np.ones((1, W))
            cols = np.ones((H, 1)) * np.arange(W)[None, :]
            dist = np.sqrt((rows - centroid[0])**2 + (cols - centroid[1])**2)
            max_dist = dist.max()
            if max_dist > 0:
                dist = dist / max_dist
        else:
            dist = np.zeros((H, W), dtype=np.float32)
        feat_dist = dist.astype(np.float32).ravel()

        # --- feature 6: humidity proxy (constant) ---
        feat_humidity = np.full(H * W, humidity / 100.0, dtype=np.float32)

        X = np.column_stack([
            feat_state, feat_neighbors, feat_wind,
            feat_slope, feat_fuel, feat_dist, feat_humidity
        ])
        return X.astype(np.float32)

    def train(self, data_path: str) -> None:
        """
        Train XGBoost binary classifier from collected prediction data.

        Args:
            data_path: Path to .npz file with keys X (features) and y (labels)
        """
        try:
            import xgboost as xgb
        except ImportError:
            print("  [FirePredictor] xgboost not installed; trying sklearn GradientBoosting")
            xgb = None

        data = np.load(data_path)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)

        print(f"  [FirePredictor] Training on {len(X)} samples, "
              f"{y.mean()*100:.1f}% positive class")

        if xgb is not None:
            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            )

        clf.fit(X, y)

        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)

        self.model = clf
        print(f"  [FirePredictor] Model saved to {_MODEL_PATH}")

    def predict(
        self,
        fire_grid: np.ndarray,
        wind_vec: np.ndarray,
        slope_grid: np.ndarray,
        fuel_grid: np.ndarray,
        humidity: float = 30.0,
    ) -> np.ndarray:
        """
        Predict per-cell fire probability one horizon ahead.

        Args:
            fire_grid:  (H, W) current fire state grid
            wind_vec:   (2,) normalised wind direction vector
            slope_grid: (H, W) slope magnitude grid
            fuel_grid:  (H, W) fuel type grid
            humidity:   ambient humidity

        Returns:
            (H, W) float32 probability grid; zeros if not trained
        """
        if not self.is_trained:
            return np.zeros(fire_grid.shape, dtype=np.float32)

        H, W = fire_grid.shape
        try:
            X = self._build_feature_matrix(fire_grid, wind_vec, slope_grid,
                                            fuel_grid, humidity)
            probs = self.model.predict_proba(X)[:, 1]
            return probs.reshape(H, W).astype(np.float32)
        except Exception as e:
            return np.zeros(fire_grid.shape, dtype=np.float32)
