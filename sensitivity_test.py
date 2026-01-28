"""
Sensitivity Analysis: Panic Threshold vs Mortality Rate
Tests how varying the panic threshold affects evacuation outcomes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.simulation import AIGISSimulation
from src.config import *

# Save original panic thresholds
ORIGINAL_RATIONAL = CIVILIAN_PANIC_RATIONAL
ORIGINAL_CONFUSED = CIVILIAN_PANIC_CONFUSED

def run_sensitivity_analysis(lat=MAP_CENTER_LAT, lon=MAP_CENTER_LON, radius=MAP_RADIUS, runs_per_threshold=10):
    """
    Run sensitivity analysis on panic threshold.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius: Map radius in meters
        runs_per_threshold: Number of Monte Carlo runs per threshold value

    Returns:
        DataFrame with results
    """
    print("=" * 70)
    print("🧪 SENSITIVITY ANALYSIS: Panic Threshold vs Mortality Rate")
    print("=" * 70)
    print(f"  Location: ({lat:.4f}, {lon:.4f})")
    print(f"  Runs per threshold: {runs_per_threshold}")
    print("=" * 70 + "\n")

    # Test panic thresholds from 0.1 to 0.9
    panic_thresholds = np.arange(0.1, 1.0, 0.1)
    results_list = []

    for threshold in panic_thresholds:
        print(f"\n🔬 Testing Panic Threshold: {threshold:.1f}")
        print("-" * 70)

        # Temporarily modify panic thresholds
        import src.config as config
        config.CIVILIAN_PANIC_RATIONAL = threshold
        config.CIVILIAN_PANIC_CONFUSED = threshold + 0.2  # Keep 0.2 gap

        threshold_results = []

        for run_id in range(runs_per_threshold):
            print(f"  Run {run_id + 1}/{runs_per_threshold}...", end=" ")

            # Run simulation
            sim = AIGISSimulation(lat, lon, radius, mode='batch')
            result = sim.run_until_complete(max_steps=500)

            threshold_results.append({
                'panic_threshold': threshold,
                'run_id': run_id,
                'mortality_rate': result['mortality_rate'],
                'evacuation_success_rate': result['evacuation_success_rate'],
                'casualties': result['casualties'],
                'evacuated': result['evacuated'],
                'avg_panic_level': result['avg_panic_level'],
                'steps': result['steps']
            })

            print(f"Mortality: {result['mortality_rate']:.1%}")

        # Calculate statistics for this threshold
        df_threshold = pd.DataFrame(threshold_results)
        mean_mortality = df_threshold['mortality_rate'].mean()
        std_mortality = df_threshold['mortality_rate'].std()

        print(f"  ✅ Mean Mortality: {mean_mortality:.2%} ± {std_mortality:.2%}")

        results_list.extend(threshold_results)

    # Restore original values
    import src.config as config
    config.CIVILIAN_PANIC_RATIONAL = ORIGINAL_RATIONAL
    config.CIVILIAN_PANIC_CONFUSED = ORIGINAL_CONFUSED

    return pd.DataFrame(results_list)


def plot_sensitivity_results(df: pd.DataFrame, output_file='sensitivity_analysis.png'):
    """
    Plot sensitivity analysis results.

    Args:
        df: DataFrame with results
        output_file: Output filename for plot
    """
    # Calculate mean and std for each threshold
    grouped = df.groupby('panic_threshold').agg({
        'mortality_rate': ['mean', 'std'],
        'evacuation_success_rate': ['mean', 'std'],
        'avg_panic_level': ['mean', 'std']
    }).reset_index()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Mortality Rate vs Panic Threshold
    ax1 = axes[0]
    thresholds = grouped['panic_threshold']
    mortality_mean = grouped['mortality_rate']['mean']
    mortality_std = grouped['mortality_rate']['std']

    ax1.plot(thresholds, mortality_mean, 'ro-', linewidth=2, markersize=8)
    ax1.fill_between(thresholds,
                     mortality_mean - mortality_std,
                     mortality_mean + mortality_std,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Panic Threshold (Rational → Confused)', fontsize=12)
    ax1.set_ylabel('Mortality Rate', fontsize=12)
    ax1.set_title('Mortality Rate vs Panic Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Evacuation Success vs Panic Threshold
    ax2 = axes[1]
    evac_mean = grouped['evacuation_success_rate']['mean']
    evac_std = grouped['evacuation_success_rate']['std']

    ax2.plot(thresholds, evac_mean, 'go-', linewidth=2, markersize=8)
    ax2.fill_between(thresholds,
                     evac_mean - evac_std,
                     evac_mean + evac_std,
                     alpha=0.3, color='green')
    ax2.set_xlabel('Panic Threshold (Rational → Confused)', fontsize=12)
    ax2.set_ylabel('Evacuation Success Rate', fontsize=12)
    ax2.set_title('Evacuation Success vs Panic Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Average Panic Level vs Panic Threshold
    ax3 = axes[2]
    panic_mean = grouped['avg_panic_level']['mean']
    panic_std = grouped['avg_panic_level']['std']

    ax3.plot(thresholds, panic_mean, 'bo-', linewidth=2, markersize=8)
    ax3.fill_between(thresholds,
                     panic_mean - panic_std,
                     panic_mean + panic_std,
                     alpha=0.3, color='blue')
    ax3.set_xlabel('Panic Threshold (Rational → Confused)', fontsize=12)
    ax3.set_ylabel('Average Panic Level', fontsize=12)
    ax3.set_title('Panic Level vs Panic Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    plt.close()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Sensitivity Analysis: Panic Threshold vs Mortality")
    parser.add_argument('--lat', type=float, default=MAP_CENTER_LAT, help='Center latitude')
    parser.add_argument('--lon', type=float, default=MAP_CENTER_LON, help='Center longitude')
    parser.add_argument('--radius', type=float, default=MAP_RADIUS, help='Map radius in meters')
    parser.add_argument('--runs', type=int, default=10, help='Runs per threshold (default: 10)')
    parser.add_argument('--output-csv', type=str, default='sensitivity_results.csv', help='Output CSV file')
    parser.add_argument('--output-plot', type=str, default='sensitivity_analysis.png', help='Output plot file')

    args = parser.parse_args()

    # Run analysis
    df = run_sensitivity_analysis(
        lat=args.lat,
        lon=args.lon,
        radius=args.radius,
        runs_per_threshold=args.runs
    )

    # Save results
    df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Results saved to: {args.output_csv}")

    # Generate plots
    plot_sensitivity_results(df, args.output_plot)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("📊 SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)

    summary = df.groupby('panic_threshold').agg({
        'mortality_rate': ['mean', 'std'],
        'evacuation_success_rate': ['mean', 'std']
    })

    print("\nMortality Rate by Panic Threshold:")
    for threshold in sorted(df['panic_threshold'].unique()):
        subset = df[df['panic_threshold'] == threshold]
        mean = subset['mortality_rate'].mean()
        std = subset['mortality_rate'].std()
        print(f"  {threshold:.1f}: {mean:.2%} ± {std:.2%}")

    print("\nKey Findings:")
    best_threshold = df.groupby('panic_threshold')['mortality_rate'].mean().idxmin()
    worst_threshold = df.groupby('panic_threshold')['mortality_rate'].mean().idxmax()

    print(f"  ✅ Best Threshold: {best_threshold:.1f} (lowest mortality)")
    print(f"  ⚠️  Worst Threshold: {worst_threshold:.1f} (highest mortality)")

    print("\n" + "=" * 70)
    print("✅ Sensitivity analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
