"""
AIGIS - AI for Guardian & Intervention Systems
Multi-Agent Wildfire Evacuation Simulation

CLI Interface supporting:
- Single headless simulation
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
from src.config import *
from src.parameter_adapter import ParameterAdapter


def run_single_simulation(lat: float, lon: float, radius: float,
                          fire_locations: list = None,
                          config_overrides: dict = None):
    """
    Run a single headless simulation instance.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius: Map radius in meters
        fire_locations: List of (lat, lon) tuples for real fire ignition points.
        config_overrides: Optional dict of config parameter overrides.

    Returns:
        Tuple of (results dict, AIGISSimulation instance)
    """
    sim = AIGISSimulation(lat, lon, radius, mode='batch',
                          fire_locations=fire_locations,
                          config_overrides=config_overrides or {})
    results = sim.run_until_complete()
    return results, sim


def run_monte_carlo(lat: float, lon: float, radius: float,
                   num_runs: int, output_file: str,
                   fire_locations: list = None,
                   config_overrides: dict = None) -> pd.DataFrame:
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

    adapter = ParameterAdapter()
    results_list = []

    # Run N independent simulations
    for run_id in range(num_runs):
        print(f"\n🔬 Run {run_id + 1}/{num_runs}")
        print("-" * 70)

        # Apply learned overrides from previous runs (skip on first run)
        overrides = adapter.get_overrides() if run_id > 0 else {}
        if config_overrides:
            overrides.update(config_overrides)

        # Each run uses a different random seed for variability
        # Simulation runs in headless mode (no GUI) for speed
        sim = AIGISSimulation(lat, lon, radius, mode='batch', run_id=run_id,
                              fire_locations=fire_locations,
                              config_overrides=overrides)
        result = sim.run_until_complete()

        # Feed outcome back to adapter for online learning
        adapter.update(result)

        # Add metadata to results (exclude complex nested objects for CSV)
        result_flat = {k: v for k, v in result.items()
                       if k not in ('history', 'reconsideration_log')}
        result_flat['run_id'] = run_id
        result_flat['lat'] = lat
        result_flat['lon'] = lon
        result_flat['radius'] = radius

        results_list.append(result_flat)
        result = result_flat  # for the print below

        # Print summary for this run
        print(f"  ✅ Complete: {result['steps']} steps, "
              f"{result['casualties']} casualties, "
              f"{result['evacuated']} evacuated")

    adapter.print_summary()

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


def save_visualization(sim: 'AIGISSimulation', result: dict,
                        out_path: str = "aigis_result.png") -> str:
    """
    Render a 2×2 grid of maps and save to PNG:
      Top-left:  Terrain / fuel type map
      Top-right: Ignition risk grid (pre-fire probability)
      Bottom-left: Final fire state
      Bottom-right: NASA FIRMS historical hotspot density
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display needed
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping visualization.")
        return ""

    env = sim.environment
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"AIGIS Simulation Results  |  ({sim.lat:.4f}, {sim.lon:.4f})  |  "
        f"Steps: {result['steps']}  Casualties: {result['casualties']}  "
        f"Evacuated: {result['evacuated']}/{result['total_civilians']}",
        fontsize=11, fontweight='bold'
    )

    # ── 1. Fuel / terrain map ──────────────────────────────────────────────
    ax = axes[0, 0]
    fuel_cmap = mcolors.ListedColormap([
        '#4a7c59',   # 0 = no fuel / urban (dark green)
        '#7fbf7f',   # 1 = light (shrub)
        '#c8a96e',   # 2 = medium (mixed)
        '#8b5a2b',   # 3 = heavy (forest)
        '#3a5f8a',   # 4 = water (blue)
        '#d3d3d3',   # 5 = road / impervious (grey)
    ])
    fuel_norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6], fuel_cmap.N)
    fuel_grid = getattr(env, 'fuel_grid', None)
    if fuel_grid is not None:
        im = ax.imshow(fuel_grid, cmap=fuel_cmap, norm=fuel_norm,
                       origin='upper', interpolation='nearest')
        legend_elements = [
            Patch(facecolor='#4a7c59', label='No fuel / Urban'),
            Patch(facecolor='#7fbf7f', label='Light (Shrub)'),
            Patch(facecolor='#c8a96e', label='Medium (Mixed)'),
            Patch(facecolor='#8b5a2b', label='Heavy (Forest)'),
            Patch(facecolor='#3a5f8a', label='Water'),
            Patch(facecolor='#d3d3d3', label='Road'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=6, framealpha=0.7)
    ax.set_title('Terrain / Fuel Type', fontsize=10)
    ax.axis('off')

    # ── 2. Ignition risk grid ──────────────────────────────────────────────
    ax = axes[0, 1]
    risk_grid = getattr(env, 'ignition_risk_grid', None)
    if risk_grid is not None and risk_grid.max() > 0:
        im2 = ax.imshow(risk_grid, cmap='YlOrRd', vmin=0, vmax=1,
                        origin='upper', interpolation='bilinear')
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04,
                     label='Ignition Risk [0-1]')
        fwi = env.fwi_data.get('fwi', 0.0) if hasattr(env, 'fwi_data') else 0.0
        risk_level = env.fwi_data.get('risk_level', 'N/A') if hasattr(env, 'fwi_data') else 'N/A'
        ax.set_title(f'Pre-Ignition Risk Grid  (FWI={fwi:.1f} — {risk_level})',
                     fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Risk grid not computed\n(RiskMonitor agent inactive)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('Pre-Ignition Risk Grid', fontsize=10)
    ax.axis('off')

    # ── 3. Final fire state ────────────────────────────────────────────────
    ax = axes[1, 0]
    fire_state = getattr(env, 'fire_grid', None)
    if fire_state is not None:
        fire_cmap = mcolors.ListedColormap(['#2c7bb6', '#fdae61', '#d7191c'])
        fire_norm = mcolors.BoundaryNorm([0, 1, 2, 3], fire_cmap.N)
        ax.imshow(fire_state, cmap=fire_cmap, norm=fire_norm,
                  origin='upper', interpolation='nearest')
        fire_legend = [
            Patch(facecolor='#2c7bb6', label='Unburned'),
            Patch(facecolor='#fdae61', label='Burning'),
            Patch(facecolor='#d7191c', label='Burned out'),
        ]
        ax.legend(handles=fire_legend, loc='lower right',
                  fontsize=7, framealpha=0.7)
    ax.set_title('Final Fire State', fontsize=10)
    ax.axis('off')

    # ── 4. FIRMS historical hotspot density ────────────────────────────────
    ax = axes[1, 1]
    firms_density = getattr(env, 'firms_density', None)
    if firms_density is not None and firms_density.max() > 0:
        im4 = ax.imshow(firms_density, cmap='hot_r', vmin=0, vmax=1,
                        origin='upper', interpolation='bilinear')
        fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04,
                     label='Historical Ignition Density')
        ax.set_title('NASA FIRMS Historical Hotspot Density (7d)', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No historical fire data\n(FIRMS: 0 hotspots)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('NASA FIRMS Historical Hotspot Density', fontsize=10)
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="AIGIS - Multi-Agent Wildfire Evacuation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation (default location)
  python main.py

  # Custom location
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

    # Fire ignition (real coordinates)
    parser.add_argument('--fire-lat', type=float, nargs='+', metavar='LAT',
                       help='Latitude(s) of real fire ignition point(s)')
    parser.add_argument('--fire-lon', type=float, nargs='+', metavar='LON',
                       help='Longitude(s) of real fire ignition point(s)')

    # Agent counts
    parser.add_argument('--ambulances', type=int, default=None,
                       help='Number of ambulance agents (default: from config)')

    # Mode selection
    parser.add_argument('--batch', type=int, metavar='N',
                       help='Run N Monte Carlo experiments (batch mode)')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file for batch mode (default: results.csv)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save PNG visualization of risk grid and fire spread after simulation')

    args = parser.parse_args()

    # Build fire locations list from paired --fire-lat / --fire-lon args
    fire_locations = None
    if args.fire_lat and args.fire_lon:
        if len(args.fire_lat) != len(args.fire_lon):
            print("❌ --fire-lat and --fire-lon must have the same number of values")
            sys.exit(1)
        fire_locations = list(zip(args.fire_lat, args.fire_lon))
        print(f"  🔥 Using {len(fire_locations)} real fire location(s) from CLI")

    # Build config overrides from CLI flags
    cli_overrides = {}
    if args.ambulances is not None:
        cli_overrides['NUM_AMBULANCES'] = args.ambulances

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
                output_file=args.output,
                fire_locations=fire_locations,
                config_overrides=cli_overrides,
            )

            # Print statistics
            print_statistics(df)

        else:
            # Single simulation mode
            print(f"Location: ({args.lat:.4f}, {args.lon:.4f})")
            print(f"Radius: {args.radius}m")
            print()

            result, sim = run_single_simulation(
                lat=args.lat,
                lon=args.lon,
                radius=args.radius,
                fire_locations=fire_locations,
                config_overrides=cli_overrides,
            )

            # Print single-run results
            print("\n" + "=" * 70)
            print("SIMULATION RESULTS")
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
            recon = result.get('reconsideration_log', [])
            print(f"  Reconsideration Events:   {len(recon)}")
            print("=" * 70)

            if getattr(args, 'visualize', False):
                out_path = save_visualization(sim, result)
                print(f"\n  📊 Visualization saved to: {out_path}")

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
