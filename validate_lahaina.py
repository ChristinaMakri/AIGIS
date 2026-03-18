"""
Lahaina 2023 Wildfire Validation Script
========================================
Validates AIGIS outputs against the documented conditions of the 8 August 2023
Lahaina wildfire (Maui, Hawaii) — the deadliest US wildland-urban interface fire
in over a century and one of the most destructive single-day fires in US history.

Primary references:
  NFPA (2024). "Lahaina, Hawaii Wildfire: Fire Investigation Report."
    National Fire Protection Association, Quincy, MA.
    Key findings: Hurricane Dora-driven ENE wind 60-80+ mph; fire moved through
    residential Lahaina in < 2 hours; 101 confirmed deaths; 2,170 acres burned.

  Maui County (2024). "Maui Wildfires After-Action Report."
    County of Maui, Department of Management.
    Population of Lahaina: ~13,000 (2020 US Census).
    Mortality rate: 101 / 13,000 ≈ 0.78 %.

  USFA (2024). "Lahaina, Hawaii Wildfire: U.S. Fire Administration Technical Report."
    Federal Emergency Management Agency / USFA, Washington DC.

  NOAA (2023). "Hurricane Dora: Post-Storm Summary."
    National Weather Service Honolulu, HI.
    Documented ENE surface wind 50-80 mph (22-36 m/s) as Dora passed south of Maui.

Burn scar spatial reference:
  USGS (2023). Lahaina Fire Burn Severity Map (BAER assessment), August 2023.
  Burned extent within the Lahaina town area: ~878 ha (2,170 acres).
  In the AIGIS 3 km study zone (π × 3² ≈ 2827 ha): ~878 / 2827 ≈ 31 %.
  Wind-driven spread toward WSW (Hurricane Dora ENE → WSW flow).

ODD validation methodology:
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
  Other Simulation Models: A Second Update." JASSS 23(2):7.
  DOI: 10.18564/jasss.4259

Usage
-----
  python validate_lahaina.py [--runs N] [--output FILE]

Outputs
-------
  - Console table: mean ± std vs. documented Lahaina values
  - Saves CSV to lahaina_validation_results.csv (or --output)
  - Saves validation summary plot to lahaina_validation_results.png
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from src.simulation import AIGISSimulation
from src.config import MAX_STEPS


# ---------------------------------------------------------------------------
# Spatial validation helpers (Filippi et al. 2016)
# ---------------------------------------------------------------------------

def _build_reference_burn_grid(grid_shape: tuple, wind_dir_deg: float,
                                burned_area_frac: float) -> np.ndarray:
    """
    Approximate the USGS BAER burn scar as an anisotropic ellipse elongated
    in the dominant spread direction (WSW, driven by Hurricane Dora ENE wind).
    Aspect ratio ~2.5:1 reflecting the narrow coastal burn corridor.
    """
    rows, cols = grid_shape
    total_cells  = rows * cols
    target_cells = int(total_cells * burned_area_frac)

    spread_rad = np.radians(wind_dir_deg)
    spread_dx  = np.sin(spread_rad)
    spread_dy  = -np.cos(spread_rad)

    aspect = 2.5
    b = np.sqrt(target_cells / (np.pi * aspect))
    a = aspect * b

    theta = np.arctan2(spread_dx, -spread_dy)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    rr, cc = np.ogrid[:rows, :cols]
    dr = rr - rows // 2
    dc = cc - cols // 2

    dr_rot = dr * cos_t + dc * sin_t
    dc_rot = -dr * sin_t + dc * cos_t

    offset_r = int(0.10 * rows * (-spread_dy / max(abs(spread_dy), 1e-6)))
    offset_c = int(0.10 * cols * ( spread_dx / max(abs(spread_dx), 1e-6)))
    dr_rot = dr_rot - offset_r * cos_t - offset_c * sin_t
    dc_rot = dc_rot + offset_r * sin_t - offset_c * cos_t

    ellipse = ((dr_rot / a) ** 2 + (dc_rot / b) ** 2) <= 1.0
    return ellipse.astype(bool)


def jaccard_index(sim_burn_mask: np.ndarray, ref_burn_mask: np.ndarray) -> float:
    """Jaccard/IoU index (Filippi et al. 2016, Eq. 5)."""
    intersection = np.logical_and(sim_burn_mask, ref_burn_mask).sum()
    union        = np.logical_or( sim_burn_mask, ref_burn_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def dice_coefficient(sim_burn_mask: np.ndarray, ref_burn_mask: np.ndarray) -> float:
    """Sorensen-Dice coefficient (Filippi et al. 2016)."""
    intersection = np.logical_and(sim_burn_mask, ref_burn_mask).sum()
    denom = sim_burn_mask.sum() + ref_burn_mask.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


# ---------------------------------------------------------------------------
# Lahaina 2023 documented conditions
# ---------------------------------------------------------------------------
LAHAINA_LAT    = 20.888   # Lahaina town centre, Maui, Hawaii
LAHAINA_LON    = -156.673
LAHAINA_RADIUS = 3000     # metres — covers residential Lahaina and surrounds

# Fire ignition points: multiple starts due to downed power lines in
# the dry grass fields northeast and east of the town (NFPA 2024, Fig. 3).
LAHAINA_FIRE_LOCATIONS = [
    (20.898, -156.661),  # Primary ignition — dry grass field NE of town
    (20.893, -156.667),  # Secondary ignition — Lahainaluna Road corridor
    (20.887, -156.672),  # Tertiary front — approaching waterfront
]

# Wind conditions from NOAA (2023) and NFPA (2024):
# Hurricane Dora passed ~480 km south of Maui on 8 August 2023.
# Surface wind at Maui: ENE, 22-32 m/s (50-70 mph) sustained,
# gusts to 36+ m/s (80+ mph).
# AIGIS convention: WIND_INITIAL_DIRECTION is direction wind is going TO.
# ENE wind = FROM ENE (≈65°) = going TOWARD WSW (≈245°).
LAHAINA_CONFIG_OVERRIDES = {
    'WIND_INITIAL_DIRECTION': 245.0,    # Going WSW — ENE wind (NOAA 2023)
    'WIND_SPEED': 27.0,                 # m/s sustained (NFPA 2024; NOAA 2023)
    'WIND_OSCILLATION_AMPLITUDE': 10.0, # Gusting documented (NFPA 2024)
    'WIND_OSCILLATION_PERIOD': 20.0,
    # Extreme fire-weather: drought conditions (D4 exceptional drought),
    # dry grass fuel, very low humidity (~20-30 %).
    'FIRE_SPREAD_PROB_BASE': 0.58,      # Elevated for extreme wind + dry grass
    'ROTHERMEL_BASE_ROS': 1.25,         # High ROS (dry grass + hurricane wind)
    'NUM_CIVILIANS': 80,                # Representative of densely-settled Lahaina
}

# ---------------------------------------------------------------------------
# Documented real-event reference values
# ---------------------------------------------------------------------------
LAHAINA_DOCUMENTED = {
    # 101 confirmed deaths (NFPA 2024; Maui County 2024; Hawaii DOH 2023).
    # Population of Lahaina: ~13,000 (US Census Bureau 2020, Lahaina CDP).
    # Mortality rate = 101 / 13,000 ≈ 0.78 %.
    'mortality_rate':          0.0078,

    # Complement: (13,000 - 101) / 13,000 ≈ 99.22 %.
    'evacuation_success_rate': 0.9922,

    # Burned extent: 2,170 acres (≈ 878 ha) documented by USGS BAER (2023).
    # AIGIS 3 km zone = π × 3² ≈ 28.3 km² = 2,827 ha.
    # 878 / 2827 ≈ 31 % of study zone.
    'burned_area_3km_pct':     31.0,

    'fire_spread_note': "Fire consumed Lahaina in < 2 hours (NFPA 2024)",
}


def run_validation(num_runs: int = 30, output_file: str = "lahaina_validation_results.csv"):
    """
    Run AIGIS N times under Lahaina 2023 conditions and compare to documented values.
    """
    print("=" * 70)
    print("AIGIS — Lahaina 2023 Wildfire Validation")
    print("=" * 70)
    print("Reference: NFPA (2024); Maui County (2024); NOAA (2023)")
    print("  Wind: ENE 65°→245°, 27 m/s | Hurricane Dora-driven")
    print(f"  Documented mortality: ~0.78 % | Runs: {num_runs}")
    print("=" * 70 + "\n")

    _ref_grid_shape = (200, 200)
    _ref_burn_mask  = _build_reference_burn_grid(
        grid_shape       = _ref_grid_shape,
        wind_dir_deg     = 245.0,
        burned_area_frac = LAHAINA_DOCUMENTED['burned_area_3km_pct'] / 100.0,
    )

    results = []
    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}", end="\r", flush=True)
        sim = AIGISSimulation(
            lat=LAHAINA_LAT,
            lon=LAHAINA_LON,
            radius=LAHAINA_RADIUS,
            mode='batch',
            run_id=i,
            fire_locations=LAHAINA_FIRE_LOCATIONS,
            config_overrides=LAHAINA_CONFIG_OVERRIDES,
        )
        result = sim.run_until_complete()

        fire_grid = sim.environment.fire_grid if hasattr(sim, 'environment') else None
        jaccard = dice = 0.0
        if fire_grid is not None:
            sim_burn_mask = (fire_grid == 2)
            jaccard = jaccard_index(sim_burn_mask, _ref_burn_mask)
            dice    = dice_coefficient(sim_burn_mask, _ref_burn_mask)

        results.append({
            'run_id':                  i,
            'steps':                   result['steps'],
            'casualties':              result['casualties'],
            'evacuated':               result['evacuated'],
            'total_civilians':         result['total_civilians'],
            'mortality_rate':          result['mortality_rate'],
            'evacuation_success_rate': result['evacuation_success_rate'],
            'burned_area_pct':         result['burned_area_pct'],
            'burned_area_ha':          result['burned_area_ha'],
            'avg_panic_level':         result['avg_panic_level'],
            'max_panic_level':         result['max_panic_level'],
            'max_fire_cells':          result['max_fire_cells'],
            'final_phase':             result['final_phase'],
            'jaccard_iou':             jaccard,
            'dice_coefficient':        dice,
        })

    print()
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}\n")
    _print_validation_table(df)
    _plot_validation(df, output_file.replace('.csv', '.png'))
    return df


def _print_validation_table(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("VALIDATION RESULTS vs. NFPA (2024) / Maui County (2024)")
    print("=" * 70)

    checks = [
        ('mortality_rate',          'Mortality Rate',
         LAHAINA_DOCUMENTED['mortality_rate'],          True),
        ('evacuation_success_rate', 'Evacuation Success Rate',
         LAHAINA_DOCUMENTED['evacuation_success_rate'], False),
        ('burned_area_pct',         'Burned Area (% of 3 km zone)',
         LAHAINA_DOCUMENTED['burned_area_3km_pct'],     True),
    ]

    all_pass = True
    for col, label, target, lower_is_better in checks:
        mean  = df[col].mean()
        std   = df[col].std()
        n     = len(df)
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(df[col]))

        if target == 0:
            within_order = True
            ratio_str = 'N/A (doc=0)'
        else:
            ratio = mean / target
            within_order = 0.1 <= ratio <= 10.0
            ratio_str = f'{ratio:.2f}x'
        status = "PASS" if within_order else "FAIL"
        if not within_order:
            all_pass = False

        if col == 'burned_area_pct':
            print(f"\n{label}:")
            print(f"  Simulated:   {mean:.1f}% ± {std:.1f}%")
            print(f"  95% CI:      [{lo:.1f}%, {hi:.1f}%]")
            print(f"  Documented:  {target:.1f}%  (USGS BAER 2023)")
            print(f"  Ratio sim/doc: {ratio_str}  →  {status}")
        else:
            print(f"\n{label}:")
            print(f"  Simulated:   {mean:.3%} ± {std:.3%}")
            print(f"  95% CI:      [{lo:.3%}, {hi:.3%}]")
            print(f"  Documented:  {target:.3%}  (NFPA 2024)")
            print(f"  Ratio sim/doc: {ratio_str}  →  {status}")

    print(f"\n{LAHAINA_DOCUMENTED['fire_spread_note']}")

    if 'jaccard_iou' in df.columns:
        jac_mean = df['jaccard_iou'].mean()
        jac_std  = df['jaccard_iou'].std()
        jac_status = 'PASS' if jac_mean >= 0.30 else 'REVIEW'
        print(f"\nSpatial Jaccard/IoU (Filippi et al. 2016, Eq. 5):")
        print(f"  Simulated vs. USGS BAER ellipse: {jac_mean:.3f} ± {jac_std:.3f}")
        print(f"  Copernicus QA threshold: J >= 0.30  →  {jac_status}")
    if 'dice_coefficient' in df.columns:
        dice_mean = df['dice_coefficient'].mean()
        dice_std  = df['dice_coefficient'].std()
        print(f"\nSorensen-Dice Coefficient (Filippi et al. 2016):")
        print(f"  Simulated vs. USGS BAER ellipse: {dice_mean:.3f} ± {dice_std:.3f}")
        print(f"  (threshold equiv. to J>=0.30: Dice>=0.46)")

    print("\n" + "=" * 70)
    overall = "PASS — outputs consistent with documented Lahaina event" if all_pass \
              else "REVIEW — some metrics outside order-of-magnitude range"
    print(f"Overall: {overall}")
    print("=" * 70)
    print("""
Note: Order-of-magnitude agreement is the standard face-validity threshold
for evacuation ABMs (Mas et al. 2021, Transportation Research Part D).
The 0.78% documented mortality is derived from Lahaina's full population;
AIGIS models representative agents in a 3 km study zone under the same
wind/fire conditions.
""")


def _plot_validation(df: pd.DataFrame, out_path: str) -> None:
    BG = '#1a1a2e'; PANEL = '#16213e'; FG = '#e0e0e0'
    fig = plt.figure(figsize=(12, 10), facecolor=BG)
    fig.suptitle(
        f"AIGIS vs. Lahaina 2023  |  NFPA (2024)  |  n={len(df)} runs",
        color=FG, fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    panels = [
        (0, 0, 'mortality_rate',          'Mortality Rate',
         LAHAINA_DOCUMENTED['mortality_rate'],          '#ff006e', True),
        (0, 1, 'evacuation_success_rate', 'Evacuation Success Rate',
         LAHAINA_DOCUMENTED['evacuation_success_rate'], '#06d6a0', True),
        (1, 0, 'burned_area_pct',         'Burned Area (% of zone)',
         LAHAINA_DOCUMENTED['burned_area_3km_pct'],     '#ffd166', False),
        (1, 1, 'jaccard_iou',             'Jaccard / IoU  (Filippi et al. 2016)',
         0.30,                                          '#8338ec', False),
    ]

    for row, col_idx, col, label, target, colour, pct_fmt in panels:
        if col not in df.columns:
            continue
        ax = fig.add_subplot(gs[row, col_idx])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        ax.hist(df[col], bins=12, color=colour, alpha=0.75,
                edgecolor='white', linewidth=0.5)
        lbl_doc = (f"Documented: {target:.2%}" if pct_fmt
                   else (f"Documented: {target:.1f}%"
                         if col == 'burned_area_pct' else f"Threshold: {target:.2f}"))
        lbl_sim = (f"Simulated mean: {df[col].mean():.2%}" if pct_fmt
                   else (f"Simulated mean: {df[col].mean():.1f}%"
                         if col == 'burned_area_pct'
                         else f"Simulated mean: {df[col].mean():.3f}"))
        ax.axvline(target, color='white', linestyle='--', linewidth=1.5, label=lbl_doc)
        ax.axvline(df[col].mean(), color=colour, linestyle='-', linewidth=2, label=lbl_sim)
        ax.set_xlabel(label, color=FG, fontsize=9)
        ax.set_ylabel('Frequency', color=FG, fontsize=9)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)
        if pct_fmt:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Validation plot saved to: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Validate AIGIS against Lahaina 2023 wildfire event data"
    )
    parser.add_argument('--runs', type=int, default=30,
                        help='Number of Monte Carlo runs (default: 30)')
    parser.add_argument('--output', type=str, default='lahaina_validation_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    run_validation(num_runs=args.runs, output_file=args.output)


if __name__ == '__main__':
    main()
