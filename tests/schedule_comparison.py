#@title 🧪 SCHEDULE COMPARISON v4 – OLD vs NEW (Normalized QUBO)
"""
================================================================================
SCHEDULE COMPARISON v4 – OLD vs NEW (Normalized QUBO)
================================================================================

This script compares the "old" geometric cooling schedule against the "new"
physics‑informed schedule on the normalized QUBO environment.

Key changes from previous versions:
    - Uses the normalized QUBO (via environment.py).
    - Uses the refactored experiment.py (run_ablation_experiment).
    - Uses plotting.py for all visualizations.

Schedule types:
    - 'old': build_schedule with beta_min, beta_max, cooling_power, num_steps.
    - 'new': Physics‑informed beta range with beta_min_mult, beta_max_mult.

Objective: Pctl10 (fixed for fair comparison)

Trial budget: 100 trials per schedule (quick comparison).
Validation: 256 reads on top 15 trials.

Results are saved to: results/schedule_compare_v4/
================================================================================
"""

import sys
import os
import time
import gc
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure the repository is in the path ---
sys.path.insert(0, "/content")
os.chdir("/content")

# Import Level 2 modules (refactored)
from src.environment import get_environment
from src.experiment import run_ablation_experiment
from src.plotting import plot_experiment_comparison_table, display_saved_plots
from src.utils import safe_save_pickle, safe_load_pickle, NumpyEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment config
SEED = 42
K_NEW = 5
N_TRIALS = 100          # Quick comparison
TUNING_READS = 100
VAL_READS = 256
VAL_TOP_K = 15
TUNING_SEED = 42
VAL_SEED = 43
USE_SEED_NONE = True

# Schedule types to compare
SCHEDULE_TYPES = [
    ('old', 'old'),
    ('new', 'new'),
]

# Storage
SAVE_DIR = Path("results/schedule_compare_v4")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🧪 SCHEDULE COMPARISON v4 – OLD vs NEW (Normalized QUBO)")
print("=" * 80)
print(f"\n[Configuration]")
print(f"  Seed: {SEED}")
print(f"  K_new: {K_NEW}")
print(f"  Trials per schedule: {N_TRIALS}")
print(f"  Tuning reads: {TUNING_READS}")
print(f"  Validation reads: {VAL_READS}")
print(f"  Val top K: {VAL_TOP_K}")
print(f"  USE_SEED_NONE: {USE_SEED_NONE}")
print(f"  Schedules: {[s[0] for s in SCHEDULE_TYPES]}")
print(f"  Results directory: {SAVE_DIR.resolve()}")

# ============================================================================
# PHASE 1: Load/Create Environment
# ============================================================================

print("\n" + "-" * 80)
print("[1] Loading/Creating normalized QUBO environment...")
print("-" * 80)

env = get_environment(
    seed=SEED,
    K_new=K_NEW,
    L_w=1.0,
    L_c=5.0,
    beta=1.0,
    delta=1.0,
    conn_multiplier=2.0,
    qir_target=0.25,
    calibrate_alpha=1.5,
    force_recompute=False,
    verbose=True,
)

print(f"\n  ✅ Environment loaded.")
print(f"     L_c* = {env['L_c_star']:.2f} km")
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

for schedule_name, schedule_type in SCHEDULE_TYPES:
    print(f"\n  🔹 Running: {schedule_name} schedule")

    result, study = run_ablation_experiment(
        experiment_name=f"schedule_{schedule_name}",
        env=env,
        schedule_type=schedule_type,
        objective_type="Pctl10",  # fixed for comparison
        n_trials=N_TRIALS,
        tuning_reads=TUNING_READS,
        val_reads=VAL_READS,
        val_seed=VAL_SEED,
        storage_dir=SAVE_DIR / schedule_name,
        val_top_k=VAL_TOP_K,
        tuning_seed=TUNING_SEED,
        use_seed_none=USE_SEED_NONE,
        compute_esr_mcr=True,
        force_retune=True,   # Always retune to ensure fresh comparison
        show_plots=True,     # Show plots inline in Colab
        verbose=True,
    )

    all_results[schedule_name] = result

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
print("🏆 SCHEDULE COMPARISON RESULTS (v4)")
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

    # Check if tie
    tie_threshold = 0.02
    ties = valid_rows[valid_rows['Spearman ρ'] >= best_rho - tie_threshold]
    if len(ties) > 1:
        tie_winner = ties.loc[ties['Best SQR'].idxmax()]
        if tie_winner['Schedule'] != winner:
            print(f"\n  ℹ️ Tie detected (within {tie_threshold}).")
            print(f"     Tie-breaker (highest SQR): {tie_winner['Schedule']}")
            print(f"     Spearman ρ = {tie_winner['Spearman ρ']:.4f}")
            print(f"     Best SQR = {tie_winner['Best SQR']:.4f}")

    # Save winner
    with open(SAVE_DIR / "comparison_winner.txt", "w") as f:
        f.write(winner)

    print(f"\n  ✅ Recommended schedule: {winner}")

    # Additional insight: Which schedule has higher feasibility?
    feas_old = df[df['Schedule'] == 'old']['Feas Rate'].values[0] if 'old' in df['Schedule'].values else np.nan
    feas_new = df[df['Schedule'] == 'new']['Feas Rate'].values[0] if 'new' in df['Schedule'].values else np.nan
    if not np.isnan(feas_old) and not np.isnan(feas_new):
        print(f"\n  Feasibility: old = {feas_old*100:.1f}%, new = {feas_new*100:.1f}%")
        if feas_new > feas_old:
            print(f"    ✅ New schedule has {feas_new-feas_old:.1%} higher feasibility")
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
print("✅ SCHEDULE COMPARISON v4 COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved to: {SAVE_DIR.resolve()}")
print("   - comparison_results.csv (summary table)")
print("   - comparison_barchart.png (comparison bar chart)")
print(f"   - comparison_winner.txt (winner name)")
print(f"   - {winner_name}/ (detailed results + plots)" if 'winner' in locals() else "   - old/ and new/ (detailed results)")

# Clean up
gc.collect()
print("\n🧹 Memory cleanup complete.")
print("=" * 80)