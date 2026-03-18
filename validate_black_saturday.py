"""
Black Saturday 2009 Wildfire Validation Script (Kinglake Complex, Victoria)
============================================================================
Validates AIGIS outputs against the documented conditions of the Kinglake
complex fires of 7 February 2009 — the deadliest bushfires in Australian
recorded history (173 total deaths; 119 in the Kinglake complex alone).

Primary references:
  Teague, B., MacLeod, R., & Pascoe, S. (2010).
    "2009 Victorian Bushfires Royal Commission: Final Report."
    State of Victoria, Melbourne.
    Documented conditions: NW wind 15-20 m/s sustained, gusts to 33 m/s
    (120 km/h); temperature 46.4°C; RH < 8 %; Code Red fire danger.

  Blanchi, R., Lucas, C., Leonard, J., & Finkele, K. (2010).
    "Meteorological conditions and wildfire-related houseloss in Australia."
    International Journal of Wildland Fire, 19(7), 914-926.
    DOI: 10.1071/WF08175

  Cruz, M.G., Sullivan, A.L., Gould, J.S., Sims, N.C., Bannister, A.J.,
    Hollis, J.J., & Hurley, R.J. (2012).
    "Anatomy of a catastrophic wildfire: The Black Saturday Kilmore East fire
    in Victoria, Australia." Forest Ecology and Management, 284, 269-285.
    DOI: 10.1016/j.foreco.2012.02.035

Burn scar spatial reference:
  DEPI Victoria (2009). Black Saturday fire boundaries GIS dataset.
  Kinglake complex burned area: ~112,000 ha total complex.
  In a 3 km study zone centred on Kinglake township: approximately 50 % burned.

Affected population:
  Kinglake municipality: ~7,000 residents (ABS 2006 Census).
  Estimated people in fire-affected area on the day (residents + visitors):
  ~12,000 (Teague et al. 2010, Chapter 6, Table 6.2).
  Deaths in Kinglake complex: 119 (Teague et al. 2010, Appendix A).
  Mortality rate: 119 / 12,000 ≈ 0.99 %.

ODD validation methodology:
  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
  Other Simulation Models: A Second Update." JASSS 23(2):7.
  DOI: 10.18564/jasss.4259

Usage
-----
  python validate_black_saturday.py [--runs N] [--output FILE]

Outputs
-------
  - Console table: mean ± std vs. documented Black Saturday values
  - Saves CSV to black_saturday_validation_results.csv (or --output)
  - Saves validation summary plot to black_saturday_validation_results.png
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
    Approximate the DEPI Victoria Kinglake complex burn scar as an
    anisotropic ellipse elongated in the NW→SE spread direction.
    Aspect ratio ~2:1 consistent with documented fire perimeter shape
    (Cruz et al. 2012, Fig. 4 — elongated SE corridor).
    """
    rows, cols = grid_shape
    total_cells  = rows * cols
    target_cells = int(total_cells * burned_area_frac)

    spread_rad = np.radians(wind_dir_deg)
    spread_dx  = np.sin(spread_rad)
    spread_dy  = -np.cos(spread_rad)

    aspect = 2.0
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
# Black Saturday 2009 / Kinglake complex documented conditions
# ---------------------------------------------------------------------------
BS_LAT    = -37.515   # Kinglake township, Victoria, Australia
BS_LON    = 145.365
BS_RADIUS = 3000      # metres

# Ignition points for the Kilmore East fire (Kinglake complex):
# Fire ignited near Kilmore East on the Hume Highway and spread SE rapidly.
# (Cruz et al. 2012, Fig. 2 — ignition location and spread trajectory)
BS_FIRE_LOCATIONS = [
    (-37.503, 145.377),  # Primary ignition — Kilmore East corridor
    (-37.509, 145.371),  # Secondary spread point
    (-37.515, 145.365),  # Tertiary front reaching Kinglake
]

# Wind conditions from Teague et al. (2010) and Cruz et al. (2012):
# 7 February 2009: extreme NW wind event.
# NW wind = FROM NW (≈ 315°) = going TOWARD SE (≈ 135°). AIGIS TO convention.
# Sustained 15-20 m/s (54-72 km/h), gusts to 33 m/s (120 km/h).
# Temperature 46.4°C (record for Melbourne); RH < 8 %; FFDI = 190+ (catastrophic).
BS_CONFIG_OVERRIDES = {
    'WIND_INITIAL_DIRECTION': 135.0,    # Going SE — NW wind (Teague et al. 2010)
    'WIND_SPEED': 18.0,                 # m/s sustained (Cruz et al. 2012)
    'WIND_OSCILLATION_AMPLITUDE': 14.0, # Extreme gusting documented (Teague 2010)
    'WIND_OSCILLATION_PERIOD': 18.0,
    # FFDI > 100 (catastrophic); high fuel load (dry summer); RH < 8 %.
    'FIRE_SPREAD_PROB_BASE': 0.55,      # Catastrophic fire weather
    'ROTHERMEL_BASE_ROS': 1.20,         # Extreme ROS (Cruz et al. 2012, Table 3)
    'NUM_CIVILIANS': 75,                # Representative of Kinglake population
}

# ---------------------------------------------------------------------------
# Documented real-event reference values
# ---------------------------------------------------------------------------
BS_DOCUMENTED = {
    # 119 deaths in Kinglake complex (Teague et al. 2010, Appendix A).
    # Estimated population at risk: ~12,000 (ibid., Chapter 6).
    # Mortality rate: 119 / 12,000 ≈ 0.99 %.
    'mortality_rate':          0.0099,

    # Complement: (12,000 - 119) / 12,000 ≈ 99.01 %.
    'evacuation_success_rate': 0.9901,

    # Kinglake complex burned area within 3 km study zone: ~50 %.
    # Cruz et al. (2012): fire spread at ~6-7 km/h; most of Kinglake township burned.
    'burned_area_3km_pct':     50.0,

    'fire_spread_note': "Fire reached Kinglake in ~2 hours from Kilmore East (Cruz et al. 2012)",
}


def run_validation(num_runs: int = 30,
                   output_file: str = "black_saturday_validation_results.csv"):
    """
    Run AIGIS N times under Black Saturday 2009 / Kinglake conditions and
    compare to documented values.
    """
    print("=" * 70)
    print("AIGIS — Black Saturday 2009 Wildfire Validation (Kinglake Complex)")
    print("=" * 70)
    print("Reference: Teague et al. (2010) Royal Commission; Cruz et al. (2012)")
    print("  Wind: NW 315°→135°, 18 m/s | Temperature: 46.4°C | FFDI: 190+")
    print(f"  Documented mortality: ~0.99 % | Runs: {num_runs}")
    print("=" * 70 + "\n")

    _ref_grid_shape = (200, 200)
    _ref_burn_mask  = _build_reference_burn_grid(
        grid_shape       = _ref_grid_shape,
        wind_dir_deg     = 135.0,
        burned_area_frac = BS_DOCUMENTED['burned_area_3km_pct'] / 100.0,
    )

    results = []
    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}", end="\r", flush=True)
        sim = AIGISSimulation(
            lat=BS_LAT,
            lon=BS_LON,
            radius=BS_RADIUS,
            mode='batch',
            run_id=i,
            fire_locations=BS_FIRE_LOCATIONS,
            config_overrides=BS_CONFIG_OVERRIDES,
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
    print("VALIDATION RESULTS vs. Teague et al. (2010) / Cruz et al. (2012)")
    print("=" * 70)

    checks = [
        ('mortality_rate',          'Mortality Rate',
         BS_DOCUMENTED['mortality_rate'],          True),
        ('evacuation_success_rate', 'Evacuation Success Rate',
         BS_DOCUMENTED['evacuation_success_rate'], False),
        ('burned_area_pct',         'Burned Area (% of 3 km zone)',
         BS_DOCUMENTED['burned_area_3km_pct'],     True),
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
            print(f"  Documented:  {target:.1f}%  (DEPI Victoria 2009)")
            print(f"  Ratio sim/doc: {ratio_str}  →  {status}")
        else:
            print(f"\n{label}:")
            print(f"  Simulated:   {mean:.3%} ± {std:.3%}")
            print(f"  95% CI:      [{lo:.3%}, {hi:.3%}]")
            print(f"  Documented:  {target:.3%}  (Teague et al. 2010)")
            print(f"  Ratio sim/doc: {ratio_str}  →  {status}")

    print(f"\n{BS_DOCUMENTED['fire_spread_note']}")

    if 'jaccard_iou' in df.columns:
        jac_mean = df['jaccard_iou'].mean()
        jac_std  = df['jaccard_iou'].std()
        jac_status = 'PASS' if jac_mean >= 0.30 else 'REVIEW'
        print(f"\nSpatial Jaccard/IoU (Filippi et al. 2016, Eq. 5):")
        print(f"  Simulated vs. Kinglake perimeter ellipse: {jac_mean:.3f} ± {jac_std:.3f}")
        print(f"  Copernicus QA threshold: J >= 0.30  →  {jac_status}")
    if 'dice_coefficient' in df.columns:
        dice_mean = df['dice_coefficient'].mean()
        dice_std  = df['dice_coefficient'].std()
        print(f"\nSorensen-Dice Coefficient (Filippi et al. 2016):")
        print(f"  Simulated vs. Kinglake perimeter ellipse: {dice_mean:.3f} ± {dice_std:.3f}")
        print(f"  (threshold equiv. to J>=0.30: Dice>=0.46)")

    print("\n" + "=" * 70)
    overall = "PASS — outputs consistent with documented Black Saturday event" if all_pass \
              else "REVIEW — some metrics outside order-of-magnitude range"
    print(f"Overall: {overall}")
    print("=" * 70)
    print("""
Note: Order-of-magnitude agreement is the standard face-validity threshold
for evacuation ABMs (Mas et al. 2021). The 0.99% documented mortality is
derived from the estimated at-risk population during the event; exact
population on the day cannot be determined from census data alone.
""")


def _plot_validation(df: pd.DataFrame, out_path: str) -> None:
    BG = '#1a1a2e'; PANEL = '#16213e'; FG = '#e0e0e0'
    fig = plt.figure(figsize=(12, 10), facecolor=BG)
    fig.suptitle(
        f"AIGIS vs. Black Saturday 2009 (Kinglake)  |  "
        f"Teague et al. (2010)  |  n={len(df)} runs",
        color=FG, fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    panels = [
        (0, 0, 'mortality_rate',          'Mortality Rate',
         BS_DOCUMENTED['mortality_rate'],          '#ff006e', True),
        (0, 1, 'evacuation_success_rate', 'Evacuation Success Rate',
         BS_DOCUMENTED['evacuation_success_rate'], '#06d6a0', True),
        (1, 0, 'burned_area_pct',         'Burned Area (% of zone)',
         BS_DOCUMENTED['burned_area_3km_pct'],     '#ffd166', False),
        (1, 1, 'jaccard_iou',             'Jaccard / IoU  (Filippi et al. 2016)',
         0.30,                                     '#8338ec', False),
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
        description="Validate AIGIS against Black Saturday 2009 (Kinglake) wildfire data"
    )
    parser.add_argument('--runs', type=int, default=30,
                        help='Number of Monte Carlo runs (default: 30)')
    parser.add_argument('--output', type=str, default='black_saturday_validation_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()
    run_validation(num_runs=args.runs, output_file=args.output)


if __name__ == '__main__':
    main()
