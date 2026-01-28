"""
AIGIS - AI for Guardian & Intervention Systems
Multi-Agent Wildfire Evacuation Simulation

CLI Interface supporting:
- Single simulation with live dashboard (GUI mode)
- Monte Carlo batch experiments (batch mode)
- Location-agnostic: works anywhere in the world
- Research-ready: exports CSV results for statistical analysis
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

from src.simulation import AIGISSimulation
from src.dashboard import Dashboard
from src.config import *


def run_single_simulation(lat: float, lon: float, radius: float, mode: str = 'gui') -> Dict[str, Any]:
    """
    Run a single simulation instance.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius: Map radius in meters
        mode: 'gui' (interactive dashboard) or 'batch' (headless)

    Returns:
        Dictionary of final metrics
    """
    sim = AIGISSimulation(lat, lon, radius, mode=mode)

    if mode == 'gui':
        # Run with live dashboard
        dashboard = Dashboard(sim)
        dashboard.run()
        return sim.get_results()
    else:
        # Run headless (batch mode)
        return sim.run_until_complete()


def run_monte_carlo(lat: float, lon: float, radius: float,
                   num_runs: int, output_file: str) -> pd.DataFrame:
    """
    Run Monte Carlo experiments (N iterations) and export results to CSV.

    Monte Carlo simulation runs the same scenario multiple times with different
    random seeds to assess statistical variability. This is critical for stochastic
    simulations where outcomes depend on random fire ignition, agent decisions, etc.

    Process:
    1. Run N independent simulations with same initial conditions
    2. Collect metrics from each run (casualties, evacuations, panic levels)
    3. Export to CSV for statistical analysis
    4. Print summary statistics (mean ± std, min/max)

    Use Cases:
    - Sensitivity analysis (vary panic thresholds, agent counts, etc.)
    - Validation of evacuation strategies
    - Statistical significance testing
    - Research publications requiring reproducible results

    Args:
        lat: Center latitude (location-agnostic, works anywhere)
        lon: Center longitude
        radius: Map radius in meters (larger = more area but slower)
        num_runs: Number of simulation runs (typical: 10-100 for quick tests, 100-1000 for research)
        output_file: Output CSV file path

    Returns:
        DataFrame with all results (columns: run_id, steps, casualties, evacuated, etc.)
    """
    print("=" * 70)
    print("🧪 MONTE CARLO BATCH MODE")
    print("=" * 70)
    print(f"  Location: ({lat:.4f}, {lon:.4f})")
    print(f"  Radius: {radius}m")
    print(f"  Runs: {num_runs}")
    print(f"  Output: {output_file}")
    print("=" * 70 + "\n")

    results_list = []

    # Run N independent simulations
    for run_id in range(num_runs):
        print(f"\n🔬 Run {run_id + 1}/{num_runs}")
        print("-" * 70)

        # Each run uses a different random seed for variability
        # Simulation runs in headless mode (no GUI) for speed
        sim = AIGISSimulation(lat, lon, radius, mode='batch', run_id=run_id)
        result = sim.run_until_complete()

        # Add metadata to results
        result['run_id'] = run_id
        result['lat'] = lat
        result['lon'] = lon
        result['radius'] = radius

        results_list.append(result)

        # Print summary for this run
        print(f"  ✅ Complete: {result['steps']} steps, "
              f"{result['casualties']} casualties, "
              f"{result['evacuated']} evacuated")

    # Convert to pandas DataFrame for easy analysis
    df = pd.DataFrame(results_list)

    # Save to CSV for external analysis (Excel, R, Python notebooks, etc.)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

    return df


def print_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics from Monte Carlo results.

    Calculates and displays:
    - Mean ± Standard Deviation (measure of central tendency and spread)
    - Min and Max (range of outcomes)

    Statistical Interpretation:
    - Low std dev: Consistent outcomes across runs
    - High std dev: High variability (sensitive to random factors)
    - Range (max-min): Worst-case vs best-case scenarios

    Args:
        df: DataFrame with Monte Carlo results (from run_monte_carlo)
    """
    print("\n" + "=" * 70)
    print("📊 MONTE CARLO SUMMARY STATISTICS")
    print("=" * 70)

    # Key metrics
    metrics = [
        ('steps', 'Simulation Steps'),
        ('casualties', 'Casualties'),
        ('evacuated', 'Successfully Evacuated'),
        ('mortality_rate', 'Mortality Rate'),
        ('evacuation_success_rate', 'Evacuation Success Rate'),
        ('avg_panic_level', 'Average Panic Level'),
        ('max_panic_level', 'Max Panic Level'),
        ('rescuer_refusals', 'Rescuer Refusals'),
        ('max_fire_cells', 'Max Active Fire Cells')
    ]

    for col, label in metrics:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()

            # Format based on metric type
            if 'rate' in col:
                print(f"\n{label}:")
                print(f"  Mean: {mean_val:.2%} ± {std_val:.2%}")
                print(f"  Range: [{min_val:.2%}, {max_val:.2%}]")
            elif col in ['avg_panic_level', 'max_panic_level']:
                print(f"\n{label}:")
                print(f"  Mean: {mean_val:.3f} ± {std_val:.3f}")
                print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
            else:
                print(f"\n{label}:")
                print(f"  Mean: {mean_val:.2f} ± {std_val:.2f}")
                print(f"  Range: [{int(min_val)}, {int(max_val)}]")

    print("\n" + "=" * 70)


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="AIGIS - Multi-Agent Wildfire Evacuation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation with live dashboard (default location)
  python main.py

  # Custom location with live dashboard
  python main.py --lat 40.7128 --lon -74.0060 --radius 2000

  # Monte Carlo batch mode (10 runs)
  python main.py --batch 10 --output results.csv

  # Custom location + batch mode
  python main.py --lat 37.7749 --lon -122.4194 --radius 1500 --batch 5
        """
    )

    # Location parameters
    parser.add_argument('--lat', type=float, default=MAP_CENTER_LAT,
                       help=f'Center latitude (default: {MAP_CENTER_LAT})')
    parser.add_argument('--lon', type=float, default=MAP_CENTER_LON,
                       help=f'Center longitude (default: {MAP_CENTER_LON})')
    parser.add_argument('--radius', type=float, default=MAP_RADIUS,
                       help=f'Map radius in meters (default: {MAP_RADIUS})')

    # Mode selection
    parser.add_argument('--batch', type=int, metavar='N',
                       help='Run N Monte Carlo experiments (batch mode)')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file for batch mode (default: results.csv)')
    parser.add_argument('--mode', type=str, choices=['gui', 'headless'],
                       default='gui',
                       help='Visualization mode (default: gui)')

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print("🛡️  AIGIS: AI for Guardian & Intervention Systems")
    print("    Multi-Agent Wildfire Evacuation Simulation")
    print("=" * 70 + "\n")

    try:
        if args.batch:
            # Monte Carlo batch mode
            df = run_monte_carlo(
                lat=args.lat,
                lon=args.lon,
                radius=args.radius,
                num_runs=args.batch,
                output_file=args.output
            )

            # Print statistics
            print_statistics(df)

        else:
            # Single simulation mode
            print(f"📍 Location: ({args.lat:.4f}, {args.lon:.4f})")
            print(f"📏 Radius: {args.radius}m")
            print(f"🎨 Mode: {args.mode}")
            print()

            result = run_single_simulation(
                lat=args.lat,
                lon=args.lon,
                radius=args.radius,
                mode=args.mode
            )

            # Print single-run results
            print("\n" + "=" * 70)
            print("📊 SIMULATION RESULTS")
            print("=" * 70)
            print(f"  Total Steps:              {result['steps']}")
            print(f"  Total Civilians:          {result['total_civilians']}")
            print(f"  Casualties:               {result['casualties']}")
            print(f"  Successfully Evacuated:   {result['evacuated']}")
            print(f"  Mortality Rate:           {result['mortality_rate']:.2%}")
            print(f"  Evacuation Success Rate:  {result['evacuation_success_rate']:.2%}")
            print(f"  Average Panic Level:      {result['avg_panic_level']:.2f}")
            print(f"  Max Panic Level:          {result['max_panic_level']:.2f}")
            print(f"  Rescuer Refusals:         {result['rescuer_refusals']}")
            print(f"  Max Active Fire Cells:    {result['max_fire_cells']}")
            print(f"  Final Commander Phase:    {result['final_phase']}")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ AIGIS session complete.\n")


if __name__ == "__main__":
    main()
