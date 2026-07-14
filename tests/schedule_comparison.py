#@title 🧪 SCHEDULE COMPARISON v5 – Refactored (Normalized QUBO)
"""
================================================================================
SCHEDULE COMPARISON v5 – Refactored (Normalized QUBO)
================================================================================

This script compares the "old" vs "new" annealing schedules using the
refactored codebase and the normalized QUBO environment.

Key features:
    - Uses get_environment() for normalized QUBO (with fixed L_c).
    - Uses run_ablation_experiment() for tuning + validation.
    - Uses plotting.py for all visualizations.
    - FORCE_RERUN flag: deletes existing study DBs and results for a clean slate.

Configuration:
    - SEED, K_new, L_c, CONNECTIVITY_RANGE are fixed physical parameters.
    - Schedules: 'old' and 'new'.
    - Objective: Pctl10 (fixed for fair comparison).
    - 100 trials per schedule (quick comparison).

Results saved to: results/schedule_compare_v5/
================================================================================
"""

import sys
import os
import time
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
from src.utils import safe_save_pickle, safe_load_pickle, NumpyEncoder

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

# --- Experiment parameters ---
SCHEDULES = ['old', 'new']       # Schedules to compare
N_TRIALS = 100                   # Per schedule
TUNING_READS = 100
VAL_READS = 256
VAL_TOP_K = 15
TUNING_SEED = 42
VAL_SEED = 43
USE_SEED_NONE = True

# --- Force rerun flag ---
FORCE_RERUN = True               # If True, delete existing DBs and results

# --- Output ---
SAVE_DIR = Path("results/schedule_compare_v5")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print("=" * 80)
print("🧪 SCHEDULE COMPARISON v5 – Refactored (Normalized QUBO)")
print("=" * 80)
print(f"\n[Configuration]")
print(f"  Seed: {SEED}")
print(f"  K_new: {K_NEW}")
print(f"  L_c: {L_C} km (FIXED)")
print(f"  Connectivity range: {CONNECTIVITY_RANGE} km (FIXED)")
print(f"  Schedules: {SCHEDULES}")
print(f"  Trials per schedule: {N_TRIALS}")
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
    
    # Remove entire schedule subdirectories
    for schedule in SCHEDULES:
        schedule_dir = SAVE_DIR / schedule
        if schedule_dir.exists():
            shutil.rmtree(schedule_dir)
            print(f"  Deleted: {schedule_dir}")
    
    # Remove any top-level comparison files
    for file in SAVE_DIR.glob("comparison_*"):
        file.unlink()
        print(f"  Deleted: {file}")
    
    # Remove progress pickle if exists
    progress_file = SAVE_DIR / "comparison_progress.pkl"
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
    force_recompute=FORCE_RERUN,   # Rebuild if forcing rerun
    verbose=True,
    gdrive_base=None,              # Set to GDrive path if needed
)

print(f"\n  ✅ Environment loaded.")
print(f"     L_c = {env['L_c']:.2f} km (FIXED)")
print(f"     Q_sum = {env['Q_sum']:.4f}")
print(f"     Gurobi baseline = {env['gurobi_miqp']:.8f}")
print(f"     Cache hash: {env['hash']}")

# ============================================================================
# PHASE 2: Run ablation experiments for each schedule
# ============================================================================

print("\n" + "-" * 80)
print("[2] Running schedule ablation experiments...")
print("-" * 80)

all_results = {}

for schedule_type in SCHEDULES:
    print(f"\n  🔹 Running: {schedule_type} schedule")

    try:
        # run_ablation_experiment will automatically use its own force_retune flag
        # but we also want to ensure its subdirectories are clean.
        # Since we already deleted the whole schedule subdir, we can rely on
        # run_ablation_experiment's force_retune to skip loading old DBs.
        result, study = run_ablation_experiment(
            experiment_name=f"schedule_{schedule_type}",
            env=env,
            schedule_type=schedule_type,
            objective_type="Pctl10",              # fixed for fair comparison
            n_trials=N_TRIALS,
            tuning_reads=TUNING_READS,
            val_reads=VAL_READS,
            val_seed=VAL_SEED,
            storage_dir=SAVE_DIR / schedule_type,
            val_top_k=VAL_TOP_K,
            tuning_seed=TUNING_SEED,
            use_seed_none=USE_SEED_NONE,
            compute_esr_mcr=True,
            force_retune=True,                    # Always retune (to ensure fresh)
            show_plots=True,                      # Display plots inline
            verbose=True,
        )
        all_results[schedule_type] = result
    except Exception as e:
        print(f"  ❌ {schedule_type} failed: {e}")
        all_results[schedule_type] = {
            'best_sqr': np.nan,
            'avg_top5_sqr': np.nan,
            'feas_rate': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'n_validated': 0,
            'trials': [],
            'error': str(e),
        }

# ============================================================================
# PHASE 3: Comparison and Visualization
# ============================================================================

print("\n" + "-" * 80)
print("[3] Generating comparison summary...")
print("-" * 80)

# Build comparison DataFrame
comparison_rows = []
for name, res in all_results.items():
    comparison_rows.append({
        'Schedule': name,
        'Best SQR': res.get('best_sqr', np.nan),
        'Avg Top5 SQR': res.get('avg_top5_sqr', np.nan),
        'Feas Rate': res.get('feas_rate', np.nan),
        'Spearman ρ': res.get('spearman_rho', np.nan),
        'ρ p-value': res.get('spearman_p', np.nan),
        'N Validated': res.get('n_validated', 0),
    })

df = pd.DataFrame(comparison_rows)

# Save to CSV
csv_path = SAVE_DIR / "comparison_results.csv"
df.to_csv(csv_path, index=False)
print(f"  ✅ Results saved to {csv_path}")

# Print rich table to terminal
print("\n" + "=" * 80)
print("🏆 SCHEDULE COMPARISON RESULTS (v5)")
print("=" * 80)
print(df.to_string(index=False, float_format="%.4f"))

# Determine winner
valid_rows = df[df['Spearman ρ'].notna()]
if len(valid_rows) > 0:
    winner_idx = valid_rows['Spearman ρ'].idxmax()
    winner = valid_rows.loc[winner_idx, 'Schedule']
    best_rho = valid_rows.loc[winner_idx, 'Spearman ρ']
    best_sqr = valid_rows.loc[winner_idx, 'Best SQR']

    print("\n" + "=" * 80)
    print("🏆 WINNER")
    print("=" * 80)
    print(f"  🥇 Winner (by Spearman ρ): {winner}")
    print(f"     Spearman ρ = {best_rho:.4f}")
    print(f"     Best SQR = {best_sqr:.4f}")

    # Check for ties (within 0.02)
    tie_threshold = 0.02
    ties = valid_rows[valid_rows['Spearman ρ'] >= best_rho - tie_threshold]
    if len(ties) > 1:
        tie_winner_row = ties.loc[ties['Best SQR'].idxmax()]
        tie_winner = tie_winner_row['Schedule']
        if tie_winner != winner:
            print(f"\n  ℹ️ Tie detected (within {tie_threshold}).")
            print(f"     Tie-breaker (highest SQR): {tie_winner}")
            print(f"     Spearman ρ = {tie_winner_row['Spearman ρ']:.4f}")
            print(f"     Best SQR = {tie_winner_row['Best SQR']:.4f}")
            winner = tie_winner

    # Save winner
    with open(SAVE_DIR / "comparison_winner.txt", "w") as f:
        f.write(winner)

    print(f"\n  ✅ Recommended schedule: {winner}")

    # Additional insight: Feasibility comparison
    feas_old = df[df['Schedule'] == 'old']['Feas Rate'].values[0] if 'old' in df['Schedule'].values else np.nan
    feas_new = df[df['Schedule'] == 'new']['Feas Rate'].values[0] if 'new' in df['Schedule'].values else np.nan
    if not np.isnan(feas_old) and not np.isnan(feas_new):
        print(f"\n  Feasibility: old = {feas_old*100:.1f}%, new = {feas_new*100:.1f}%")
        if feas_new > feas_old:
            print(f"    ✅ New schedule has {feas_new-feas_old:.1%} higher feasibility")
        elif feas_old > feas_new:
            print(f"    ✅ Old schedule has {feas_old-feas_new:.1%} higher feasibility")
        else:
            print("    ✅ Feasibility is equal.")
else:
    print("\n  ⚠️ No valid results to determine a winner.")
    with open(SAVE_DIR / "comparison_winner.txt", "w") as f:
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
    save_path=SAVE_DIR / "comparison_barchart.png",
    metrics=['best_sqr', 'spearman_rho', 'feas_rate'],
    show_fig=True,
)

# Also display saved plots from the winning schedule
winner_name = winner if 'winner' in locals() else 'new'  # default
winner_dir = SAVE_DIR / winner_name / "plots"
if winner_dir.exists():
    print(f"\n  Displaying plots from winning schedule: {winner_name}")
    display_saved_plots(winner_dir)
else:
    print(f"\n  ⚠️ No plots found for {winner_name}")

# ============================================================================
# PHASE 5: Final summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ SCHEDULE COMPARISON v5 COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved to: {SAVE_DIR.resolve()}")
print("   - comparison_results.csv (summary table)")
print("   - comparison_barchart.png (comparison bar chart)")
print(f"   - comparison_winner.txt (winner name: {winner_name})")
print(f"   - {winner_name}/ (detailed results + plots)")

# Clean up
gc.collect()
print("\n🧹 Memory cleanup complete.")
print("=" * 80)