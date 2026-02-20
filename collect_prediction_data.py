"""
Collect training data for StepAheadPredictor.

Runs N headless simulations and records (state_t, state_t+10) pairs.
Label: cell was fuel at t AND (burning or burnt) at t+10.
Only keeps fuel cells to reduce class imbalance from empty/road cells.

Usage:
    python collect_prediction_data.py
    python collect_prediction_data.py --n-sims 50 --horizon 10
"""
import argparse
import numpy as np
from pathlib import Path

from src.simulation import AIGISSimulation
from src.fire_predictor import StepAheadPredictor
from src.config import DEFAULT_MAP_CENTER_LAT, DEFAULT_MAP_CENTER_LON, DEFAULT_MAP_RADIUS


def collect(n_sims: int = 50, horizon: int = 10, output_path: str = "data/training/fire_prediction_data.npz"):
    """Collect (features, label) pairs from N headless simulations."""
    print(f"Collecting fire prediction data: {n_sims} sims, horizon={horizon}")

    predictor = StepAheadPredictor()

    all_X = []
    all_y = []

    for run_id in range(n_sims):
        print(f"  Sim {run_id + 1}/{n_sims}")
        try:
            sim = AIGISSimulation(
                DEFAULT_MAP_CENTER_LAT, DEFAULT_MAP_CENTER_LON, DEFAULT_MAP_RADIUS,
                mode='batch', run_id=run_id
            )

            env = sim.environment
            fire_sim = sim.fire_sim

            # Record snapshots at every step up to horizon before completion
            snapshots = []  # list of (fire_grid_copy, wind_vec, slope)

            max_steps = 400
            while sim.step < max_steps and not sim.is_complete():
                grid_copy = env.fire_grid.copy()
                wind_vec = fire_sim.wind_direction.copy()

                # Compute slope grid
                grad_y, grad_x = np.gradient(env.elevation_grid)
                slope_grid = np.sqrt(grad_y**2 + grad_x**2).astype(np.float32)

                snapshots.append((sim.step, grid_copy, wind_vec, slope_grid))
                sim.run_step()

            # For each snapshot t, find snapshot t+horizon
            snap_dict = {s[0]: s for s in snapshots}

            for step_t, grid_t, wind_t, slope_t in snapshots:
                step_th = step_t + horizon
                if step_th not in snap_dict:
                    continue

                _, grid_th, _, _ = snap_dict[step_th]

                # Build feature matrix for state at t
                X = predictor._build_feature_matrix(
                    fire_grid=grid_t,
                    wind_vec=wind_t,
                    slope_grid=slope_t,
                    fuel_grid=env.fuel_type_grid,
                    humidity=getattr(env, 'humidity', 30.0),
                )

                # Label: fuel cell at t AND (burning or burnt) at t+horizon
                # Only keep fuel cells (grid_t == 3) to avoid class imbalance
                fuel_mask = (grid_t.ravel() == 3)
                y = ((grid_th.ravel() == 1) | (grid_th.ravel() == 2)).astype(np.float32)

                X_fuel = X[fuel_mask]
                y_fuel = y[fuel_mask]

                if len(X_fuel) > 0:
                    all_X.append(X_fuel)
                    all_y.append(y_fuel)

        except Exception as e:
            print(f"  WARNING: Sim {run_id} failed: {e}")
            continue

    if not all_X:
        print("ERROR: No data collected")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"\nTotal samples: {len(X_all)}")
    print(f"Positive class (will burn): {y_all.mean()*100:.1f}%")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X_all, y=y_all)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect fire prediction training data")
    parser.add_argument('--n-sims', type=int, default=50,
                        help='Number of simulation runs (default: 50)')
    parser.add_argument('--horizon', type=int, default=10,
                        help='Steps ahead to predict (default: 10)')
    parser.add_argument('--output', type=str, default='data/training/fire_prediction_data.npz',
                        help='Output path for collected data')
    args = parser.parse_args()

    collect(n_sims=args.n_sims, horizon=args.horizon, output_path=args.output)


if __name__ == "__main__":
    main()
