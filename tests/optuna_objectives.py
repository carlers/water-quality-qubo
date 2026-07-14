#@title 🧪 OBJECTIVE ABLATION v6 – Refactored (Normalized QUBO)
"""
================================================================================
OBJECTIVE ABLATION v6 – Refactored (Normalized QUBO)
================================================================================

This script tests 9 objective functions on the normalized QUBO environment
using the winning schedule from schedule_comparison_v5.py.

Key features:
    - Uses get_environment() for normalized QUBO (fixed L_c).
    - Uses run_ablation_experiment() for tuning + validation.
    - Uses plotting.py for visualizations.
    - FORCE_RERUN flag: deletes all existing results for a clean slate.

Objectives tested:
    - BestOnly    : maximize best_sqr
    - Lin         : best_sqr * feas_rate
    - Sq          : best_sqr * feas_rate^2
    - Cube        : best_sqr * feas_rate^3
    - Avg         : mean feasible SQR
    - Pctl10      : 10th percentile of feasible SQRs
    - Penalty-0.1 : best_sqr - 0.1*(1-feas_rate)
    - Penalty-0.5 : best_sqr - 0.5*(1-feas_rate)
    - Multi       : Pareto [1-best_sqr, 1-feas_rate] (NSGA-II)

Configuration:
    - SEED, K_new, L_c, CONNECTIVITY_RANGE are fixed physical parameters.
    - WINNING_SCHEDULE: set to the winner from schedule_comparison_v5.py.
    - 150 trials per objective (balanced speed/statistics).
    - Validation: top 15 trials with 256 reads.

Results saved to: results/ablation_v6/
================================================================================
"""

import sys
import os
import shutil
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure the repository is in the path ---
sys.path.insert(0, "/content")
os.chdir("/content")

# Import refactored modules
from src.environment import get_environment
from src.experiment import run_ablation_experiment
from src.plotting import plot_experiment_comparison_table, display_saved_plots
from src.utils import safe_save_pickle, NumpyEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --- Physical parameters ---
SEED = 42
K_NEW = 5
L_C = 5.0                        # FIXED (km)
CONNECTIVITY_RANGE = 8.0         # FIXED (km)
L_W = 1.0
CURRENT_VECTOR = (1.0, 0.0)
BETA = 1.0
DELTA = 1.0

# --- Winning schedule from schedule_comparison_v5 ---
WINNING_SCHEDULE = "new"         # 'new' or 'old' – update after running comparison

# --- Experiment parameters ---
OBJECTIVE_TYPES = [
    "BestOnly",
    "Lin",
    "Sq",
    "Cube",
    "Avg",
    "Pctl10",
    "Penalty-0.1",
    "Penalty-0.5",
    "Multi",
]
N_TRIALS = 150                   # Per objective
TUNING_READS = 100
VAL_READS = 256
VAL_TOP_K = 15
TUNING_SEED = 42
VAL_SEED = 43
USE_SEED_NONE = True

# --- Force rerun flag ---
FORCE_RERUN = True               # If True, delete all existing ablation results

# --- Output ---
SAVE_DIR = Path("results/ablation_v6")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print("=" * 80)
print("🧪 OBJECTIVE ABLATION v6 – Refactored (Normalized QUBO)")
print("=" * 80)
print(f"\n[Configuration]")
print(f"  Seed: {SEED}")
print(f"  K_new: {K_NEW}")
print(f"  L_c: {L_C} km (FIXED)")
print(f"  Connectivity range: {CONNECTIVITY_RANGE} km (FIXED)")
print(f"  Winning schedule: {WINNING_SCHEDULE}")
print(f"  Objectives: {OBJECTIVE_TYPES}")
print(f"  Trials per objective: {N_TRIALS}")
print(f"  Tuning reads: {TUNING_READS}")
print(f"  Validation reads: {VAL_READS}")
print(f"  Val top K: {VAL_TOP_K}")
print(f"  USE_SEED_NONE: {USE_SEED_NONE}")
print(f"  FORCE_RERUN: {FORCE_RERUN}")
print(f"  Results directory: {SAVE_DIR.resolve()}")

# ============================================================================
# CLEANUP IF FORCE_RERUN
# ============================================================================

if FORCE_RERUN:
    print("\n" + "-" * 80)
    print("[Cleanup] FORCE_RERUN enabled – deleting existing files...")
    print("-" * 80)
    
    # Remove entire ablation subdirectories
    for obj in OBJECTIVE_TYPES:
        obj_dir = SAVE_DIR / obj
        if obj_dir.exists():
            shutil.rmtree(obj_dir)
            print(f"  Deleted: {obj_dir}")
    
    # Remove any top-level comparison files
    for file in SAVE_DIR.glob("ablation_*"):
        file.unlink()
        print(f"  Deleted: {file}")
    
    # Remove progress pickle if exists
    progress_file = SAVE_DIR / "ablation_progress.pkl"
    if progress_file.exists():
        progress_file.unlink()
        print(f"  Deleted: {progress_file}")
    
    print("  ✅ Cleanup complete.")

# ============================================================================
# PHASE 1: Load/Create Environment
# ============================================================================

print("\n" + "-" * 80)
print("[1] Loading/Creating normalized QUBO environment...")
print("-" * 80)

env = get_environment(
    seed=SEED,
    K_new=K_NEW,
    L_c=L_C,
    L_w=L_W,
    beta=BETA,
    delta=DELTA,
    connectivity_range=CONNECTIVITY_RANGE,
    current_vector=CURRENT_VECTOR,
    force_recompute=FORCE_RERUN,
    verbose=True,
    gdrive_base=None,              # Set to GDrive path if needed
)

print(f"\n  ✅ Environment loaded.")
print(f"     L_c = {env['L_c']:.2f} km (FIXED)")
print(f"     Q_sum = {env['Q_sum']:.4f}")
print(f"     Gurobi baseline = {env['gurobi_miqp']:.8f}")
print(f"     Cache hash: {env['hash']}")

# ============================================================================
# PHASE 2: Run ablation experiments for each objective
# ============================================================================

print("\n" + "-" * 80)
print(f"[2] Running objective ablation (winning schedule: {WINNING_SCHEDULE})...")
print("-" * 80)

all_results = {}
study_objects = {}

for objective_type in OBJECTIVE_TYPES:
    print(f"\n  🔹 Running: {objective_type}")

    try:
        result, study = run_ablation_experiment(
            experiment_name=f"ablation_{objective_type}",
            env=env,
            schedule_type=WINNING_SCHEDULE,
            objective_type=objective_type,
            n_trials=N_TRIALS,
            tuning_reads=TUNING_READS,
            val_reads=VAL_READS,
            val_seed=VAL_SEED,
            storage_dir=SAVE_DIR / objective_type,
            val_top_k=VAL_TOP_K,
            tuning_seed=TUNING_SEED,
            use_seed_none=USE_SEED_NONE,
            compute_esr_mcr=True,
            force_retune=True,                # Always retune (fresh run)
            show_plots=True,                  # Display plots inline
            verbose=True,
        )
        all_results[objective_type] = result
        study_objects[objective_type] = study
    except Exception as e:
        print(f"  ❌ {objective_type} failed: {e}")
        all_results[objective_type] = {
            'best_sqr': np.nan,
            'avg_top5_sqr': np.nan,
            'feas_rate': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'n_validated': 0,
            'trials': [],
            'error': str(e),
        }
    # Save progress after each objective
    safe_save_pickle(SAVE_DIR / "ablation_progress.pkl", all_results)

# ============================================================================
# PHASE 3: Comparison and Ranking
# ============================================================================

print("\n" + "-" * 80)
print("[3] Generating comparison summary...")
print("-" * 80)

# Build comparison DataFrame
comparison_rows = []
for obj_name, res in all_results.items():
    comparison_rows.append({
        'Objective': obj_name,
        'Best SQR': res.get('best_sqr', np.nan),
        'Avg Top5 SQR': res.get('avg_top5_sqr', np.nan),
        'Feas Rate': res.get('feas_rate', np.nan),
        'Spearman ρ': res.get('spearman_rho', np.nan),
        'ρ p-value': res.get('spearman_p', np.nan),
        'N Validated': res.get('n_validated', 0),
    })

df = pd.DataFrame(comparison_rows)

# Sort by Spearman ρ (descending)
df_sorted = df.sort_values('Spearman ρ', ascending=False, na_position='last')

# Save to CSV
csv_path = SAVE_DIR / "ablation_results.csv"
df_sorted.to_csv(csv_path, index=False)
print(f"  ✅ Results saved to {csv_path}")

# Print rich table to terminal
print("\n" + "=" * 80)
print("🏆 OBJECTIVE ABLATION RESULTS (v6)")
print("=" * 80)
print(df_sorted.to_string(index=False, float_format="%.4f"))

# Determine winner
valid_rows = df_sorted[df_sorted['Spearman ρ'].notna()]
if len(valid_rows) > 0:
    winner_row = valid_rows.iloc[0]
    winner = winner_row['Objective']
    best_rho = winner_row['Spearman ρ']
    best_sqr = winner_row['Best SQR']
    best_feas = winner_row['Feas Rate']

    print("\n" + "=" * 80)
    print("🏆 WINNER")
    print("=" * 80)
    print(f"  🥇 Winner (by Spearman ρ): {winner}")
    print(f"     Spearman ρ = {best_rho:.4f}")
    print(f"     Best SQR = {best_sqr:.4f}")
    print(f"     Feas Rate = {best_feas:.4f}")

    # Check for ties (within 0.02)
    tie_threshold = 0.02
    ties = valid_rows[valid_rows['Spearman ρ'] >= best_rho - tie_threshold]
    if len(ties) > 1:
        # Tie-break by SQR
        tie_winner_row = ties.loc[ties['Best SQR'].idxmax()]
        tie_winner = tie_winner_row['Objective']
        if tie_winner != winner:
            print(f"\n  ℹ️ Tie detected (within {tie_threshold}).")
            print(f"     Tie-breaker (highest SQR): {tie_winner}")
            print(f"     Spearman ρ = {tie_winner_row['Spearman ρ']:.4f}")
            print(f"     Best SQR = {tie_winner_row['Best SQR']:.4f}")
            winner = tie_winner
            best_rho = tie_winner_row['Spearman ρ']
            best_sqr = tie_winner_row['Best SQR']
            best_feas = tie_winner_row['Feas Rate']

    # Save winner
    with open(SAVE_DIR / "ablation_winner.txt", "w") as f:
        f.write(winner)

    print(f"\n  ✅ Recommended objective: {winner}")
    print(f"  Reason: Highest Spearman correlation ({best_rho:.4f})")
    if best_sqr > 0.95:
        print(f"          Maintains high validation SQR ({best_sqr:.4f})")
    if best_feas > 0.3:
        print(f"          Achieves good feasibility rate ({best_feas:.1%})")

    # Print ranking table
    print("\n" + "=" * 80)
    print("📊 RANKING (by Spearman ρ)")
    print("=" * 80)
    for idx, row in df_sorted.iterrows():
        rho = row['Spearman ρ']
        if np.isnan(rho):
            rho_str = "N/A"
        else:
            rho_str = f"{rho:.4f}"
        print(f"  {row['Objective']:12s} | ρ={rho_str:8s} | SQR={row['Best SQR']:.4f} | Feas={row['Feas Rate']:.4f}")

else:
    print("\n  ⚠️ No valid results to determine a winner.")
    with open(SAVE_DIR / "ablation_winner.txt", "w") as f:
        f.write("No winner")

# ============================================================================
# PHASE 4: Generate comparison bar chart
# ============================================================================

print("\n" + "-" * 80)
print("[4] Generating comparison bar chart...")
print("-" * 80)

# Use plotting.py function
plot_experiment_comparison_table(
    results_dict=all_results,
    save_path=SAVE_DIR / "ablation_barchart.png",
    metrics=['best_sqr', 'spearman_rho', 'feas_rate'],
    show_fig=True,
)

# Also display saved plots from the winner
winner_name = winner if 'winner' in locals() else "Pctl10"  # default
winner_dir = SAVE_DIR / winner_name / "plots"
if winner_dir.exists():
    print(f"\n  Displaying plots from winning objective: {winner_name}")
    display_saved_plots(winner_dir)
else:
    print(f"\n  ⚠️ No plots found for {winner_name}")

# ============================================================================
# PHASE 5: Extract best hyperparameters from winner
# ============================================================================

print("\n" + "-" * 80)
print("[5] Extracting best hyperparameters from winner...")
print("-" * 80)

if winner_name in study_objects:
    winner_result = all_results[winner_name]
    if winner_result.get('trials'):
        # The first trial in the list is the best (we sorted by best_sqr in validation)
        best_trial = winner_result['trials'][0]
        print(f"  Best validated trial for {winner_name}:")
        print(f"    λ₁ = {best_trial.get('lam1', np.nan):.4f}")
        print(f"    λ₂ = {best_trial.get('lam2', np.nan):.4f}")
        print(f"    num_sweeps = {best_trial.get('num_sweeps', 'N/A')}")
        if 'ESR' in best_trial and not np.isnan(best_trial['ESR']):
            print(f"    ESR = {best_trial['ESR']:.4f}")
        if 'MCR' in best_trial and not np.isnan(best_trial['MCR']):
            print(f"    MCR = {best_trial['MCR']:.4f}")
    else:
        print(f"  ⚠️ No validated trials found for {winner_name}")

    # Also get best from Optuna study directly (tuning)
    study = study_objects[winner_name]
    if study:
        best_trial_optuna = study.best_trial
        print(f"\n  Best Optuna trial for {winner_name} (tuning):")
        print(f"    λ₁ = {best_trial_optuna.params.get('lam1', 'N/A')}")
        print(f"    λ₂ = {best_trial_optuna.params.get('lam2', 'N/A')}")
        print(f"    num_sweeps = {best_trial_optuna.params.get('num_sweeps', 'N/A')}")

# ============================================================================
# PHASE 6: Final summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ OBJECTIVE ABLATION v6 COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved to: {SAVE_DIR.resolve()}")
print("   - ablation_results.csv (ranked summary table)")
print("   - ablation_barchart.png (comparison bar chart)")
print(f"   - ablation_winner.txt (winner name: {winner_name if 'winner_name' in locals() else 'unknown'})")
print("   - {objective}/ (detailed results + plots for each objective)")

# Save a summary JSON with all metrics
summary = {
    'experiment': 'objective_ablation_v6',
    'seed': SEED,
    'winning_schedule': WINNING_SCHEDULE,
    'winner': winner_name if 'winner_name' in locals() else None,
    'results': df_sorted.to_dict('records'),
}
with open(SAVE_DIR / "ablation_summary.json", "w") as f:
    import json
    json.dump(summary, f, indent=2, cls=NumpyEncoder)

# Clean up
gc.collect()
print("\n🧹 Memory cleanup complete.")
print("=" * 80)