#!/bin/bash
# =============================================================================
# AIGIS Thesis Experiment Runner
# =============================================================================
# Runs all thesis experiments sequentially and logs full output.
# Expected runtime: ~14 hours (CPU-only).
# Run with: bash run_thesis_experiments.sh
#
# SCENARIO INVENTORY
# ------------------
# Training (23 scenarios — MARL curriculum, never used for validation):
#   Phase 1 (easy,    5): Bages (Spain), Var (France), Penteli (Greece),
#                         Corsica (France), Tuscany (Italy)
#   Phase 2 (medium,  8): Manavgat (Turkey), Rhodes (Greece),
#                         Kineta (Greece), Varibobi (Greece), Dadia/Evros (Greece),
#                         Carmel (Israel), Dwellingup (W. Australia), Monchique (Portugal)
#   Phase 3 (hard,   10): Fort McMurray (Canada), Gospers Mountain (Australia),
#                         Carr Fire (USA), Glass Fire (USA), Woolsey Fire (USA),
#                         Thomas Fire (USA), Evia (Greece), Oristano/Sardinia (Italy),
#                         Lytton Creek (Canada), Knysna (South Africa)
#
# Validation / held-out (9 scenarios — real incidents, never seen in training):
#   Mati 2018 (Greece)              — Lagouvardos et al. 2019 / EMSR249
#   Camp Fire 2018 (USA)            — CAL FIRE 2020
#   Pedrogao Grande 2017 (Portugal) — Viegas et al. 2017 / EMSR218
#   Alexandroupoli 2023 (Greece)    — Greek Fire Service 2023 / EMSR689
#   Lahaina 2023 (USA)              — NFPA 2024 / Maui County 2024
#   Black Saturday 2009 (Australia) — Teague et al. 2010 Royal Commission
#   Tubbs Fire 2017 (USA)           — CAL FIRE 2018 / Nauslar et al. 2018
#   Peloponnese 2007 (Greece)       — Koutsias et al. 2012 / EEA 2007
#   Valparaiso 2014 (Chile)         — Encinas et al. 2015 / CONAF 2014
#
# EXPERIMENT PIPELINE
# -------------------
# Block A — ML models       (steps  1–2)
# Block B — Physics validation against 9 held-out real incidents (steps 3–11)
# Block C — Behavioural baselines and stress tests (steps 12–13)
# Block D — Sensitivity and ablation (steps 14–15)
# Block E — MARL training and full evaluation (steps 16–17)
# =============================================================================

set -e  # abort on any non-zero exit

LOGFILE="thesis_experiments.log"
echo "Started: $(date)" | tee "$LOGFILE"

run_step() {
    local step="$1"
    local cmd="$2"
    echo ""                                          | tee -a "$LOGFILE"
    echo "=============================================" | tee -a "$LOGFILE"
    echo "[$step] $(date)"                          | tee -a "$LOGFILE"
    echo "CMD: $cmd"                                | tee -a "$LOGFILE"
    echo "=============================================" | tee -a "$LOGFILE"
    eval "$cmd" 2>&1 | tee -a "$LOGFILE"
    echo "[$step] DONE: $(date)"                    | tee -a "$LOGFILE"
}

# =============================================================================
# BLOCK A — ML MODEL TRAINING AND EVALUATION
# Two-stage hurdle model (Mullahy 1986; Cameron & Trivedi 2013) for casualty
# risk; RandomForest for evacuation count and containment time.
# Trained on 2000 Monte Carlo runs across 23 historical locations (80/20 split).
# Evaluated with 50-run hold-out split.
# =============================================================================

run_step "0/17 [Pre-run] Dataset Diversity Chart — 32 scenarios (23 training + 9 held-out)" \
    "python3 plot_dataset_diversity.py --output dataset_diversity.png"

run_step "1/17 [Block A] Train ML Models — 2000 Monte Carlo runs, 23 locations" \
    "python3 train_models.py --runs 2000"

run_step "2/17 [Block A] Evaluate ML Models — 10 runs x 23 training scenarios (in-distribution)" \
    "python3 evaluate_ml_models.py --runs 10 --multi-scenario --output ml_evaluation_results.csv"

# =============================================================================
# BLOCK B — PHYSICS / AGENT VALIDATION AGAINST REAL INCIDENTS (held-out)
# Each script runs AIGIS 50 times under documented real-event conditions and
# compares mortality rate, evacuation success, burned area, and Jaccard/IoU
# against Copernicus EMS burn scars (Filippi et al. 2016).
# 50 runs per scenario provides tighter 95% CI on mean metrics.
# Order-of-magnitude agreement is the accepted face-validity standard
# for evacuation ABMs (Mas et al. 2021; Grimm et al. 2020).
# =============================================================================

run_step "3/17 [Block B] Validate Mati 2018 — 50 runs [held-out]" \
    "python3 validate_mati.py --runs 50 --output mati_validation_results.csv"
#   Reference : Lagouvardos et al. (2019) BAMS 100(11):2243-2257
#   Burn scar : Copernicus EMSR249
#   Target    : mortality ~1.70 %, burned area ~35 % of 3 km zone, Jaccard >= 0.30

run_step "4/17 [Block B] Validate Camp Fire 2018 — 50 runs [held-out]" \
    "python3 validate_campfire.py --runs 50 --output campfire_validation_results.csv"
#   Reference : CAL FIRE (2020); NWS Sacramento (2018)
#   Target    : mortality ~0.31 %, Jaccard >= 0.30

run_step "5/17 [Block B] Validate Pedrogao Grande 2017 — 50 runs [held-out]" \
    "python3 validate_pedrogao.py --runs 50 --output pedrogao_validation_results.csv"
#   Reference : Viegas et al. (2017) ADAI/CEIF; Guerreiro et al. (2018)
#   Burn scar : Copernicus EMSR218
#   Target    : mortality ~0.88 %, burned area ~40 % of 3 km zone, Jaccard >= 0.30

run_step "6/17 [Block B] Validate Alexandroupoli 2023 — 50 runs [held-out]" \
    "python3 validate_alexandroupoli.py --runs 50 --output alexandroupoli_validation_results.csv"
#   Reference : Greek Fire Service (2023); Copernicus EMSR689; EMY (2023)
#   Burn scar : Copernicus EMSR689 — largest fire in EU recorded history
#   Target    : mortality ~0.40 %, burned area ~45 % of 3 km zone, Jaccard >= 0.30

run_step "7/17 [Block B] Validate Lahaina 2023 — 50 runs [held-out]" \
    "python3 validate_lahaina.py --runs 50 --output lahaina_validation_results.csv"
#   Reference : NFPA (2024); Maui County (2024); NOAA (2023); USFA (2024)
#   Hurricane Dora downburst: ENE wind 27 m/s; 100 fatalities / ~12,800 residents
#   Target    : mortality ~0.78 %, burned area ~31 % of 3 km zone, Jaccard >= 0.30

run_step "8/17 [Block B] Validate Black Saturday 2009 — 50 runs [held-out]" \
    "python3 validate_black_saturday.py --runs 50 --output black_saturday_validation_results.csv"
#   Reference : Teague et al. (2010) Royal Commission; Cruz et al. (2012)
#   NW wind 18 m/s; FFDI 190+; 119 fatalities / ~12,000 population = 0.99 %
#   Target    : mortality ~0.99 %, burned area ~50 % of 3 km zone, Jaccard >= 0.30

run_step "9/17 [Block B] Validate Tubbs Fire 2017 — 50 runs [held-out]" \
    "python3 validate_tubbs.py --runs 50 --output tubbs_validation_results.csv"
#   Reference : CAL FIRE (2018); Nauslar et al. (2018) Weather and Forecasting
#   Diablo NE wind 25 m/s; 22 deaths / ~8,000 residents = 0.28 %
#   Target    : mortality ~0.28 %, burned area ~41 % of 3 km zone, Jaccard >= 0.30

run_step "10/17 [Block B] Validate Peloponnese 2007 — 50 runs [held-out]" \
    "python3 validate_peloponnese.py --runs 50 --output peloponnese_validation_results.csv"
#   Reference : Koutsias et al. (2012) Agric. Forest Meteorol. 156:41-53; EEA (2007)
#   Etesian NNE wind 14 m/s; 77 deaths total Greece; 0.60 % in Zacharo study zone
#   Target    : mortality ~0.60 %, burned area ~45 % of 3 km zone, Jaccard >= 0.30

run_step "11/17 [Block B] Validate Valparaiso 2014 — 50 runs [held-out]" \
    "python3 validate_valparaiso.py --runs 50 --output valparaiso_validation_results.csv"
#   Reference : Encinas et al. (2015) Int J Disaster Risk Reduction 13:280-289; CONAF (2014)
#   SE wind 12 m/s (La Nina drought); 15 deaths / ~8,000 = 0.19 %; South America coverage
#   Target    : mortality ~0.19 %, burned area ~30 % of 3 km zone, Jaccard >= 0.30

# =============================================================================
# BLOCK C — BASELINE MONTE CARLO AND MULTI-INCIDENT COVERAGE
# Establishes baseline performance distribution across all 23 training
# scenarios and provides inter-scenario comparison data.
# =============================================================================

run_step "12/17 [Block C] Baseline Monte Carlo — 50 runs, default scenario" \
    "python3 main.py --batch 50 --output baseline.csv"

run_step "13/17 [Block C] Multi-Incident Diagnostic — 20 runs x 32 incidents (23 training + 9 held-out)" \
    "python3 validate_all_incidents.py --runs 20 --output all_incidents_validation.csv"
#   Runs all 32 scenarios at 20 runs each = 640 total runs.
#   Outputs per-scenario mean +/- 95 % CI for mortality, evacuation, burned area.

# =============================================================================
# BLOCK D — SENSITIVITY ANALYSIS AND ABLATION
# Sobol global variance-based sensitivity (Saltelli et al. 2010):
#   N=512 base -> 7168 total model evaluations.
#   Outputs: S1 (first-order) and ST (total-effect) indices with 95 % CI.
# Ablation (4 conditions x 50 runs = 200 total):
#   Baseline | No-CNP | No-Panic | No-Coordination
#   Mann-Whitney U + rank-biserial effect size (alpha=0.017 Bonferroni).
# =============================================================================

run_step "14/17 [Block D] Sobol Sensitivity Analysis — N=512 (7168 model runs)" \
    "python3 run_sensitivity.py --N 512 --output sensitivity_results.csv"
#   N=512 satisfies Saltelli et al. (2010) Section 3.2 recommendation N >= 500/D
#   for D=6 parameters.  N=128 is exploratory only; N=512 gives publication-quality
#   95 % CI on total-effect indices STi.

run_step "15/17 [Block D] Ablation Study — 100 runs per condition (4 conditions)" \
    "python3 run_ablation.py --runs 100 --output ablation_results.csv"

# =============================================================================
# BLOCK E — MARL TRAINING AND FULL EVALUATION
# Train: 4,000 episodes x 500 steps over 15 curriculum scenarios (Bengio et al. 2009).
# Evaluate: 50 runs x 32 scenarios (23 training + 9 held-out) = 1600 total.
#   Training split  -> in-distribution generalisation check (phases 1-3)
#   Held-out split  -> OOD generalisation to real documented incidents
# =============================================================================

run_step "16/17 [Block E] Train MARL — 4,000 episodes x 500 steps, 23-scenario curriculum" \
    "python3 train_marl.py --episodes 4000 --steps 500 --phase1-end 800 --phase2-end 2400 --output models/rl"
#   4000 episodes x 500 steps = 2,000,000 total environment steps — same as the
#   original 10,000 x 200 budget, but episode length now matches evaluation (500 steps).
#   Phase boundaries scaled proportionally from original (20% / 60% / 40%):
#     Phase 1 easy   :   0 –  800 episodes (20%)
#     Phase 2 medium :  800 – 2400 episodes (40%)
#     Phase 3 hard   : 2400 – 4000 episodes (40%)
#   Bengio et al. (2009); PPO clip eps=0.2, GAE lambda=0.95 (Schulman et al. 2016)

run_step "17/17 [Block E] Evaluate MARL — 50 runs x 32 scenarios (23 train + 9 held-out)" \
    "python3 evaluate_marl.py --runs 50 --output marl_evaluation_results.csv"
#   Held-out: Mati, Camp Fire, Pedrogao, Alexandroupoli, Lahaina, Black Saturday,
#             Tubbs Fire 2017, Peloponnese 2007, Valparaiso 2014
#   Metric: mortality rate, evacuation success, burned area % — mean +/- 95 % CI

# =============================================================================
# SUMMARY
# =============================================================================
echo ""                                              | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"
echo "ALL EXPERIMENTS COMPLETE: $(date)"            | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"
echo ""                                              | tee -a "$LOGFILE"
echo "Output files:"                                 | tee -a "$LOGFILE"
ls -lh \
    training_dataset.csv \
    ml_evaluation_results.csv \
    mati_validation_results.csv \
    campfire_validation_results.csv \
    pedrogao_validation_results.csv \
    alexandroupoli_validation_results.csv \
    lahaina_validation_results.csv \
    black_saturday_validation_results.csv \
    tubbs_validation_results.csv \
    peloponnese_validation_results.csv \
    valparaiso_validation_results.csv \
    baseline.csv \
    all_incidents_validation.csv \
    sensitivity_results.csv \
    ablation_results.csv \
    marl_training_log.csv \
    marl_evaluation_results.csv \
    training_evaluation.png \
    ml_evaluation_results.png \
    mati_validation_results.png \
    campfire_validation_results.png \
    pedrogao_validation.png \
    alexandroupoli_validation.png \
    lahaina_validation_results.png \
    black_saturday_validation_results.png \
    tubbs_validation_results.png \
    peloponnese_validation_results.png \
    valparaiso_validation_results.png \
    sensitivity_plot.png \
    ablation_plot.png \
    all_incidents_validation.png \
    marl_evaluation_results.png \
    2>/dev/null | tee -a "$LOGFILE"
