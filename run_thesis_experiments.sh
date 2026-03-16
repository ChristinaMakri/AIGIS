#!/bin/bash
# AIGIS Thesis Experiment Runner
# Runs all experiments sequentially and logs output.
# Expected runtime: ~6 hours.
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

run_step "1/7 Train ML Models (100 runs)" \
    "python3 train_models.py --runs 100"

run_step "2/7 Evaluate ML Models (30 runs)" \
    "python3 evaluate_ml_models.py --runs 30 --output ml_evaluation_results.csv"

run_step "3/7 Validate Mati 2018 (30 runs)" \
    "python3 validate_mati.py --runs 30 --output mati_validation_results.csv"

run_step "4/7 Validate Camp Fire 2018 (30 runs)" \
    "python3 validate_campfire.py --runs 30 --output campfire_validation_results.csv"

run_step "5/7 Baseline Monte Carlo (50 runs)" \
    "python3 main.py --batch 50 --output baseline.csv"

run_step "6/7 Sensitivity Analysis (15 runs per value)" \
    "python3 run_sensitivity.py --runs 15 --output sensitivity_results.csv"

run_step "7/7 Ablation Study (30 runs per condition)" \
    "python3 run_ablation.py --runs 30 --output ablation_results.csv"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "ALL EXPERIMENTS COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Output files:" | tee -a "$LOGFILE"
ls -lh training_dataset.csv ml_evaluation_results.csv \
       mati_validation_results.csv campfire_validation_results.csv \
       baseline.csv sensitivity_results.csv ablation_results.csv \
       training_evaluation.png ml_evaluation.png \
       mati_validation.png campfire_validation.png \
       sensitivity_plot.png ablation_plot.png 2>/dev/null | tee -a "$LOGFILE"
