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
# Training (12 scenarios — MARL curriculum, never used for validation):
#   Phase 1 (easy)   : Bages (Spain), Var (France), Penteli (Greece)
#   Phase 2 (medium) : Manavgat (Turkey), Rhodes (Greece),
#                      Kineta (Greece), Varibobi (Greece)
#   Phase 3 (hard)   : Fort McMurray (Canada), Gospers Mountain (Australia),
#                      Carr Fire (USA), Glass Fire (USA), Woolsey Fire (USA)
#
# Validation / held-out (4 scenarios — real incidents, never seen in training):
#   Mati 2018 (Greece)              — Lagouvardos et al. 2019 / EMSR249
#   Camp Fire 2018 (USA)            — CAL FIRE 2020
#   Pedrogao Grande 2017 (Portugal) — Viegas et al. 2017 / EMSR218
#   Alexandroupoli 2023 (Greece)    — Greek Fire Service 2023 / EMSR689
#
# EXPERIMENT PIPELINE
# -------------------
# Block A — ML models       (steps  1–2)
# Block B — Physics validation against 4 held-out real incidents (steps 3–6)
# Block C — Behavioural baselines and stress tests (steps 7–8)
# Block D — Sensitivity and ablation (steps 9–10)
# Block E — MARL training and full evaluation (steps 11–13)
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
# XGBoost fire-danger predictors trained on 100 Monte Carlo runs.
# Evaluated with 30-run hold-out split.
# =============================================================================

run_step "1/12 [Block A] Train ML Models — 100 Monte Carlo runs" \
    "python3 train_models.py --runs 100"

run_step "2/12 [Block A] Evaluate ML Models — 30 runs, hold-out split" \
    "python3 evaluate_ml_models.py --runs 30 --output ml_evaluation_results.csv"

# =============================================================================
# BLOCK B — PHYSICS / AGENT VALIDATION AGAINST REAL INCIDENTS (held-out)
# Each script runs AIGIS 30 times under documented real-event conditions and
# compares mortality rate, evacuation success, burned area, and Jaccard/IoU
# against Copernicus EMS burn scars (Filippi et al. 2016).
# Order-of-magnitude agreement is the accepted face-validity standard
# for evacuation ABMs (Mas et al. 2021; Grimm et al. 2020).
# =============================================================================

run_step "3/12 [Block B] Validate Mati 2018 — 30 runs [held-out]" \
    "python3 validate_mati.py --runs 30 --output mati_validation_results.csv"
#   Reference : Lagouvardos et al. (2019) BAMS 100(11):2243-2257
#   Burn scar : Copernicus EMSR249
#   Target    : mortality ~1.70 %, burned area ~35 % of 3 km zone, Jaccard >= 0.30

run_step "4/12 [Block B] Validate Camp Fire 2018 — 30 runs [held-out]" \
    "python3 validate_campfire.py --runs 30 --output campfire_validation_results.csv"
#   Reference : CAL FIRE (2020); NWS Sacramento (2018)
#   Target    : mortality ~0.31 %, Jaccard >= 0.30

run_step "5/12 [Block B] Validate Pedrogao Grande 2017 — 30 runs [held-out]" \
    "python3 validate_pedrogao.py --runs 30 --output pedrogao_validation_results.csv"
#   Reference : Viegas et al. (2017) ADAI/CEIF; Guerreiro et al. (2018)
#   Burn scar : Copernicus EMSR218
#   Target    : mortality ~0.88 %, burned area ~40 % of 3 km zone, Jaccard >= 0.30

run_step "6/12 [Block B] Validate Alexandroupoli 2023 — 30 runs [held-out]" \
    "python3 validate_alexandroupoli.py --runs 30 --output alexandroupoli_validation_results.csv"
#   Reference : Greek Fire Service (2023); Copernicus EMSR689; EMY (2023)
#   Burn scar : Copernicus EMSR689 — largest fire in EU recorded history
#   Target    : mortality ~0.40 %, burned area ~45 % of 3 km zone, Jaccard >= 0.30

# =============================================================================
# BLOCK C — BASELINE MONTE CARLO AND MULTI-INCIDENT COVERAGE
# Establishes baseline performance distribution across all 12 training
# scenarios and provides inter-scenario comparison data.
# =============================================================================

run_step "7/12 [Block C] Baseline Monte Carlo — 50 runs, default scenario" \
    "python3 main.py --batch 50 --output baseline.csv"

run_step "8/12 [Block C] Multi-Incident Diagnostic — 15 runs x 16 incidents (12 training + 4 held-out)" \
    "python3 validate_all_incidents.py --runs 15 --output all_incidents_validation.csv"
#   Runs all 12 curriculum training scenarios at 15 runs each = 180 total runs.
#   Outputs per-scenario mean +/- 95 % CI for mortality, evacuation, burned area.

# =============================================================================
# BLOCK D — SENSITIVITY ANALYSIS AND ABLATION
# Sobol global variance-based sensitivity (Saltelli et al. 2010):
#   N=512 base -> 7168 total model evaluations.
#   Outputs: S1 (first-order) and ST (total-effect) indices with 95 % CI.
# Ablation (4 conditions x 30 runs = 120 total):
#   Baseline | No-CNP | No-Panic | No-Coordination
#   Mann-Whitney U + rank-biserial effect size (alpha=0.017 Bonferroni).
# =============================================================================

run_step "9/12 [Block D] Sobol Sensitivity Analysis — N=512 (7168 model runs)" \
    "python3 run_sensitivity.py --runs 512 --output sensitivity_results.csv"
#   N=512 satisfies Saltelli et al. (2010) Section 3.2 recommendation N >= 500/D
#   for D=6 parameters.  N=128 is exploratory only; N=512 gives publication-quality
#   95 % CI on total-effect indices STi.

run_step "10/12 [Block D] Ablation Study — 30 runs per condition (4 conditions)" \
    "python3 run_ablation.py --runs 30 --output ablation_results.csv"

# =============================================================================
# BLOCK E — MARL TRAINING AND FULL EVALUATION
# Train: 10,000 episodes over 12 curriculum scenarios (Bengio et al. 2009).
# Evaluate: 30 runs x 16 scenarios (12 training + 4 held-out) = 480 total.
#   Training split  -> in-distribution generalisation check (phases 1-3)
#   Held-out split  -> OOD generalisation to real documented incidents
# =============================================================================

run_step "11/12 [Block E] Train MARL — 10,000 episodes, 12-scenario curriculum" \
    "python3 train_marl.py --episodes 10000 --output models/rl"
#   Curriculum phases: 1 easy (0-2000 ep) -> 2 medium (2000-6000) -> 3 hard (6000+)
#   Bengio et al. (2009); PPO clip eps=0.2, GAE lambda=0.95 (Schulman et al. 2016)

run_step "12/12 [Block E] Evaluate MARL — 30 runs x 16 scenarios (12 train + 4 held-out)" \
    "python3 evaluate_marl.py --runs 30 --output marl_evaluation_results.csv"
#   Held-out: Mati 2018, Camp Fire 2018, Pedrogao Grande 2017, Alexandroupoli 2023
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
    sensitivity_plot.png \
    ablation_plot.png \
    all_incidents_validation.png \
    marl_evaluation_results.png \
    2>/dev/null | tee -a "$LOGFILE"
