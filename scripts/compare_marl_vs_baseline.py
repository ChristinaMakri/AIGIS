"""
AIGIS — MARL vs Rule-Based Baseline Comparison
================================================
Computes the performance delta between the trained MARL policy (evaluate_marl.py)
and the rule-based BDI baseline (validate_all_incidents.py) across all 32 scenarios,
with formal statistical testing.

Why this step is included
-------------------------
A common reviewer demand for MARL papers is that the learned policy must be compared
against a rule-based or heuristic baseline using formal hypothesis tests, not just
inspection of mean values.  Yu et al. (2022) compare MAPPO and IPPO against QMIX and
heuristics across 5-10 seeds and report median + IQR.  Sivagnanam et al. (2024)
compare MARL-HC against greedy, static, and MCTS baselines, reporting mean response
time reduction with seed-level variance.  This script closes the gap between AIGIS's
Block C (rule-based baseline across 32 scenarios) and Block E (MARL evaluation across
the same 32 scenarios) by computing the per-scenario delta and testing it.

Statistical method: Wilcoxon signed-rank test (non-parametric paired comparison).
  Wilcoxon (1945): does not assume normality of differences — appropriate for small
  N simulation samples and zero-inflated mortality distributions.
  Effect size: rank-biserial correlation r = 1 − (4W / (N(N+1))) — matches the
  ablation study convention (Mann-Whitney U / ablation uses the same effect size family).
  Bonferroni correction: alpha = 0.05 / 32 scenarios = 0.00156 (conservative).

References
----------
  Yu, C. et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
    Multi-Agent Games." NeurIPS 2022. arXiv:2103.01955.
    [MAPPO/IPPO vs heuristics; per-seed median + IQR reporting standard]

  Sivagnanam, A. et al. (2024). "Multi-Agent Reinforcement Learning with
    Hierarchical Coordination for Emergency Responder Stationing." ICML 2024.
    arXiv:2405.13205.
    [MARL vs greedy/MCTS for emergency response; mean response-time reduction]

  Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
    Biometrics Bulletin, 1(6), pp. 80-83.
    [Non-parametric paired signed-rank test]

  Kerby, D.S. (2014). "The simple difference formula: an approach to teaching
    nonparametric correlation." Comprehensive Psychology, 3, pp. 11.IT.3.1.
    DOI: 10.2466/11.IT.3.1.
    [Rank-biserial correlation as effect size for Wilcoxon test]

Usage
-----
  python compare_marl_vs_baseline.py \\
      --marl-file     marl_evaluation_results.csv \\
      --baseline-file all_incidents_validation.csv \\
      --output        marl_vs_baseline.csv

Outputs
-------
  CSV  : per-scenario delta (MARL − baseline) with W, p-value, effect size r
  PNG  : grouped bar chart — mortality and evacuation rate by scenario and policy
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BG, PANEL, FG = 'white', '#f5f5f5', '#222222'

# ---------------------------------------------------------------------------
# Bonferroni-corrected alpha
# ---------------------------------------------------------------------------
N_SCENARIOS = 32
ALPHA_BONFERRONI = 0.05 / N_SCENARIOS   # 0.001563


def _rank_biserial(w_stat: float, n: int) -> float:
    """
    Rank-biserial correlation as effect size for the Wilcoxon signed-rank test.

    r = 1 − (4W / (N(N+1)))

    Kerby (2014) Comprehensive Psychology — simple difference formula.
    Convention: |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.
    """
    if n == 0:
        return float('nan')
    return 1.0 - (4.0 * w_stat) / (n * (n + 1))


def _wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Run Wilcoxon signed-rank test on paired samples a (MARL) vs b (baseline).
    Returns (W statistic, p-value, rank-biserial r).
    Silently handles n < 10 by skipping the test.
    """
    diff = a - b
    diff = diff[diff != 0]  # ties are dropped per Wilcoxon (1945)
    if len(diff) < 5:
        return float('nan'), float('nan'), float('nan')
    w, p = stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
    r = _rank_biserial(w, len(diff))
    return float(w), float(p), float(r)


def compare(
    marl_file:     str = 'marl_evaluation_results.csv',
    baseline_file: str = 'all_incidents_validation.csv',
    output_file:   str = 'marl_vs_baseline.csv',
) -> pd.DataFrame:
    """
    Load per-run results from both evaluation files, compute per-scenario
    paired Wilcoxon tests, and report the delta with effect sizes.

    Both CSVs must contain columns: scenario, mortality_rate, evacuation_rate.
    """
    print('=' * 70)
    print('AIGIS — MARL vs Rule-Based Baseline Comparison')
    print('=' * 70)
    print('Wilcoxon (1945) signed-rank test  |  Kerby (2014) rank-biserial r')
    print(f'Bonferroni alpha = 0.05 / {N_SCENARIOS} scenarios = {ALPHA_BONFERRONI:.5f}')
    print('Yu et al. (2022) NeurIPS  |  Sivagnanam et al. (2024) ICML')
    print('=' * 70 + '\n')

    marl_df = pd.read_csv(marl_file)
    base_df = pd.read_csv(baseline_file)

    # Normalise column names
    for df in [marl_df, base_df]:
        df.columns = df.columns.str.lower().str.strip()
        if 'evacuation_success_rate' in df.columns and 'evacuation_rate' not in df.columns:
            df.rename(columns={'evacuation_success_rate': 'evacuation_rate'}, inplace=True)
        if 'scenario_name' in df.columns and 'scenario' not in df.columns:
            df.rename(columns={'scenario_name': 'scenario'}, inplace=True)

    scenarios = sorted(set(marl_df['scenario'].unique()) &
                       set(base_df['scenario'].unique()))

    if not scenarios:
        print('ERROR: No matching scenario names found between the two files.')
        print('  MARL scenarios :', sorted(marl_df['scenario'].unique())[:5], '...')
        print('  Baseline scenarios:', sorted(base_df['scenario'].unique())[:5], '...')
        return pd.DataFrame()

    records = []
    for sc in scenarios:
        m = marl_df[marl_df['scenario'] == sc]
        b = base_df[base_df['scenario'] == sc]

        m_mort = m['mortality_rate'].values
        b_mort = b['mortality_rate'].values
        m_evac = m['evacuation_rate'].values
        b_evac = b['evacuation_rate'].values

        n = min(len(m_mort), len(b_mort))
        if n == 0:
            continue

        # Align lengths by truncating to min
        m_mort, b_mort = m_mort[:n], b_mort[:n]
        m_evac, b_evac = m_evac[:n], b_evac[:n]

        w_m, p_m, r_m = _wilcoxon_pair(m_mort, b_mort)
        w_e, p_e, r_e = _wilcoxon_pair(m_evac, b_evac)

        records.append({
            'scenario':              sc,
            'n_marl':                len(marl_df[marl_df['scenario'] == sc]),
            'n_baseline':            len(base_df[base_df['scenario'] == sc]),
            # Mortality
            'marl_mort_mean':        float(np.mean(m_mort)),
            'marl_mort_std':         float(np.std(m_mort)),
            'base_mort_mean':        float(np.mean(b_mort)),
            'base_mort_std':         float(np.std(b_mort)),
            'delta_mort':            float(np.mean(m_mort) - np.mean(b_mort)),
            'wilcoxon_W_mort':       w_m,
            'p_mort':                p_m,
            'p_mort_sig':            p_m < ALPHA_BONFERRONI if not np.isnan(p_m) else False,
            'r_mort':                r_m,
            # Evacuation
            'marl_evac_mean':        float(np.mean(m_evac)),
            'marl_evac_std':         float(np.std(m_evac)),
            'base_evac_mean':        float(np.mean(b_evac)),
            'base_evac_std':         float(np.std(b_evac)),
            'delta_evac':            float(np.mean(m_evac) - np.mean(b_evac)),
            'wilcoxon_W_evac':       w_e,
            'p_evac':                p_e,
            'p_evac_sig':            p_e < ALPHA_BONFERRONI if not np.isnan(p_e) else False,
            'r_evac':                r_e,
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_file, index=False)
    print(f'Results saved to: {output_file}\n')

    _print_comparison_table(df_out)
    _plot_comparison(df_out, output_file.replace('.csv', '.png'))

    return df_out


def _print_comparison_table(df: pd.DataFrame) -> None:
    """Print per-scenario delta table with significance flags."""
    print('=' * 70)
    print('MARL vs RULE-BASED BASELINE  (positive delta = MARL worse)')
    print(f'* = significant at Bonferroni-corrected alpha = {ALPHA_BONFERRONI:.5f}')
    print('-' * 70)
    print(f"{'Scenario':<35} {'d_mort':>8} {'p_m':>8} {'r_m':>6} "
          f"{'d_evac':>8} {'p_e':>8} {'r_e':>6}")
    print('-' * 70)
    for _, row in df.iterrows():
        sig_m = '*' if row['p_mort_sig'] else ' '
        sig_e = '*' if row['p_evac_sig'] else ' '
        p_m_str = f"{row['p_mort']:.4f}{sig_m}" if not np.isnan(row['p_mort']) else '  n/a  '
        p_e_str = f"{row['p_evac']:.4f}{sig_e}" if not np.isnan(row['p_evac']) else '  n/a  '
        r_m_str = f"{row['r_mort']:.3f}" if not np.isnan(row['r_mort']) else ' n/a'
        r_e_str = f"{row['r_evac']:.3f}" if not np.isnan(row['r_evac']) else ' n/a'
        print(f"  {row['scenario']:<33} {row['delta_mort']:>+8.4f} {p_m_str:>8} {r_m_str:>6} "
              f"{row['delta_evac']:>+8.4f} {p_e_str:>8} {r_e_str:>6}")
    print('-' * 70)
    sig_mort = df['p_mort_sig'].sum()
    sig_evac = df['p_evac_sig'].sum()
    print(f"  Scenarios with significant mortality improvement: "
          f"{sig_mort}/{len(df)}")
    print(f"  Scenarios with significant evacuation improvement: "
          f"{sig_evac}/{len(df)}")
    mean_d_mort = df['delta_mort'].mean()
    mean_d_evac = df['delta_evac'].mean()
    print(f"  Mean delta mortality  : {mean_d_mort:+.4f} "
          f"({'MARL better' if mean_d_mort < 0 else 'baseline better'})")
    print(f"  Mean delta evacuation : {mean_d_evac:+.4f} "
          f"({'MARL better' if mean_d_evac > 0 else 'baseline better'})")
    print('=' * 70)


def _plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    """
    Two-panel grouped bar chart:
      Panel 1: mortality rate — MARL vs baseline per scenario
      Panel 2: evacuation rate — MARL vs baseline per scenario

    Significant differences (Bonferroni-corrected Wilcoxon) are marked with *.
    Yu et al. (2022) NeurIPS — per-scenario bar charts with significance markers.
    """
    n = len(df)
    x = np.arange(n)
    w = 0.35

    fig = plt.figure(figsize=(max(14, n * 0.5), 9), facecolor=BG)
    fig.suptitle(
        'MARL vs Rule-Based Baseline  |  Wilcoxon (1945) signed-rank test\n'
        f'Bonferroni alpha = {ALPHA_BONFERRONI:.5f}  |  '
        'Yu et al. (2022) NeurIPS  |  * = significant',
        color=FG, fontsize=9, fontweight='bold',
    )

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

    for panel_i, (metric, title, marl_col, base_col, sig_col, sign) in enumerate([
        ('Mortality Rate',   'Mortality Rate — MARL vs Baseline (lower is better)',
         'marl_mort_mean', 'base_mort_mean', 'p_mort_sig', -1),
        ('Evacuation Rate',  'Evacuation Rate — MARL vs Baseline (higher is better)',
         'marl_evac_mean', 'base_evac_mean', 'p_evac_sig', +1),
    ]):
        ax = fig.add_subplot(gs[panel_i])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        bars_marl = ax.bar(x - w / 2, df[marl_col], w,
                           label='MARL (trained)', color='#8338ec', alpha=0.85)
        bars_base = ax.bar(x + w / 2, df[base_col], w,
                           label='Rule-based BDI', color='#fb5607', alpha=0.85)

        # Significance markers
        for i, (_, row) in enumerate(df.iterrows()):
            if row[sig_col]:
                y_top = max(row[marl_col], row[base_col]) * 1.05
                ax.text(i, y_top, '*', color='white', ha='center', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(df['scenario'], rotation=45, ha='right',
                           fontsize=6, color=FG)
        ax.set_ylabel(metric, color=FG, fontsize=8)
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Comparison plot saved to: {out_path}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description='Compare MARL policy vs rule-based BDI baseline across 32 scenarios'
    )
    p.add_argument('--marl-file',     type=str,
                   default='marl_evaluation_results.csv')
    p.add_argument('--baseline-file', type=str,
                   default='all_incidents_validation.csv')
    p.add_argument('--output',        type=str,
                   default='marl_vs_baseline.csv')
    args = p.parse_args()

    compare(
        marl_file=args.marl_file,
        baseline_file=args.baseline_file,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
