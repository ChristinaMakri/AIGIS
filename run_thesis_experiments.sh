#!/bin/bash
# AIGIS Thesis Experiment Runner
# Runs all experiments sequentially and logs output.
# Expected runtime: ~8 hours (CPU-only).
# Run with: bash run_thesis_experiments.sh

set -e  # stop on any error

LOGFILE="thesis_experiments.log"
echo "Started: $(date)" | tee "$LOGFILE"

run_step() {
    local step="$1"
    local cmd="$2"
    echo "" | tee -a "$LOGFILE"
    echo "========================================" | tee -a "$LOGFILE"
    echo "[$step] $(date)" | tee -a "$LOGFILE"
    echo "CMD: $cmd" | tee -a "$LOGFILE"
    echo "========================================" | tee -a "$LOGFILE"
    eval "$cmd" 2>&1 | tee -a "$LOGFILE"
    echo "[$step] DONE: $(date)" | tee -a "$LOGFILE"
}

run_step "1/10 Train ML Models (100 runs)" \
    "python3 train_models.py --runs 100"

run_step "2/10 Evaluate ML Models (30 runs)" \
    "python3 evaluate_ml_models.py --runs 30 --output ml_evaluation_results.csv"

run_step "3/10 Validate Mati 2018 (30 runs)" \
    "python3 validate_mati.py --runs 30 --output mati_validation_results.csv"

run_step "4/10 Validate Camp Fire 2018 (30 runs)" \
    "python3 validate_campfire.py --runs 30 --output campfire_validation_results.csv"

run_step "5/10 Baseline Monte Carlo (50 runs)" \
    "python3 main.py --batch 50 --output baseline.csv"

run_step "6/10 Sensitivity Analysis (15 runs per value)" \
    "python3 run_sensitivity.py --runs 15 --output sensitivity_results.csv"

run_step "7/10 Ablation Study (30 runs per condition)" \
    "python3 run_ablation.py --runs 30 --output ablation_results.csv"

run_step "8/10 Multi-Incident Validation (15 runs x 11 incidents)" \
    "python3 validate_all_incidents.py --runs 15 --output all_incidents_validation.csv"

run_step "9/10 Train MARL (10000 episodes, curriculum)" \
    "python3 train_marl.py --episodes 10000 --output models/rl"

run_step "10/10 Evaluate MARL (30 runs per scenario)" \
    "python3 evaluate_marl.py --runs 30 --output marl_evaluation_results.csv"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "ALL EXPERIMENTS COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Output files:" | tee -a "$LOGFILE"
ls -lh training_dataset.csv ml_evaluation_results.csv \
       mati_validation_results.csv campfire_validation_results.csv \
       baseline.csv sensitivity_results.csv ablation_results.csv \
       all_incidents_validation.csv \
       marl_training_log.csv marl_evaluation_results.csv \
       training_evaluation.png ml_evaluation_results.png \
       mati_validation_results.png campfire_validation_results.png \
       sensitivity_plot.png ablation_plot.png \
       all_incidents_validation.png marl_evaluation_results.png 2>/dev/null | tee -a "$LOGFILE"
