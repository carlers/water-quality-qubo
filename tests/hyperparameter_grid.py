#@title 🧪 HYPERPARAMETER GRID SEARCH – Unified Factorial Design
"""
================================================================================
HYPERPARAMETER GRID SEARCH – Unified Factorial Design
================================================================================

This script runs a comprehensive grid search across the core hyperparameters
of the QUBO tuning pipeline (Tuning → Validation). It tests interactions
between objectives, trial budgets, and validation Top K across multiple seeds.

Factors tested:
    - Objective: Sq, Pctl10, Penalty-0.1
    - Tuning trials: 100, 200
    - Tuning reads: 50, 100
    - Validation Top K: 3, 10, 20

Metrics compared:
    - Best Validation SQR (primary)
    - Feasibility Rate
    - Spearman ρ
    - Total Runtime

Seeds: 42, 43, 44 (3 seeds for robustness)

Outputs:
    - Ranked summary table (CSV)
    - Heatmap (SQR vs factors)
    - Pareto front (SQR vs samples used)
    - Boxplots for top configurations
================================================================================
"""

import sys
import os
import time
import gc
import json
import warnings
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "/content")
os.chdir("/content")

from src.environment import get_environment
from src.experiment import run_optuna_study, make_objective, validate_study
from src.utils import safe_save_pickle, NumpyEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Fixed parameters ---
SCHEDULE_TYPE = "old"
USE_SEED_NONE = True
TUNING_SEED = 42
VAL_SEED = 43
K_NEW = 5
L_C = 5.0
CONNECTIVITY_RANGE = 8.0
VAL_READS = 512  # Fixed for all validation runs

# --- Grid factors ---
OBJECTIVES = ["Sq", "Pctl10", "Penalty-0.1"]
TUNING_TRIALS_LIST = [100, 200]
TUNING_READS_LIST = [50, 100]
VAL_TOP_K_LIST = [3, 10, 20]

# --- Seeds for robustness ---
SEEDS = [42, 43, 44]

# --- Output ---
SAVE_DIR = Path("results/hyperparameter_grid_search")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER: Run a single configuration
# ============================================================================

def run_single_config(seed, obj, n_trials, tuning_reads, val_top_k):
    """Run a single (tuning + validation) experiment for a given config and seed."""
    print(f"\n  🧪 Seed {seed} | Obj: {obj} | Trials: {n_trials} | TReads: {tuning_reads} | TopK: {val_top_k}")

    # Load environment (cached per seed)
    env = get_environment(
        seed=seed,
        K_new=K_NEW,
        L_c=L_C,
        connectivity_range=CONNECTIVITY_RANGE,
        force_recompute=False,
        verbose=False,
    )

    start_time = time.time()

    # Build objective
    objective = make_objective(
        env=env,
        schedule_type=SCHEDULE_TYPE,
        objective_type=obj,
        tuning_reads=tuning_reads,
        tuning_seed=TUNING_SEED,
        use_seed_none=USE_SEED_NONE,
        compute_esr_mcr=True,
    )

    # Run tuning
    study = run_optuna_study(
        experiment_name=f"grid_{seed}_{obj}_{n_trials}_{tuning_reads}_{val_top_k}",
        objective=objective,
        n_trials=n_trials,
        storage_dir=SAVE_DIR / "studies",
        directions=['minimize'],
        sampler_type='TPE',
        seed=TUNING_SEED,
        load_if_exists=False,
        verbose=False,
    )

    # Run validation
    val_results = validate_study(
        study=study,
        env=env,
        schedule_type=SCHEDULE_TYPE,
        val_reads=VAL_READS,
        val_seed=VAL_SEED,
        top_k=val_top_k,
        use_seed_none=USE_SEED_NONE,
        compute_esr_mcr=True,
        verbose=False,
    )

    elapsed = time.time() - start_time

    # Extract metrics
    best_sqr = val_results.get('best_sqr', np.nan)
    feas_rate = val_results.get('feas_rate', np.nan)
    spearman_rho = val_results.get('spearman_rho', np.nan)
    n_validated = val_results.get('n_validated', 0)

    # Total samples consumed
    total_samples = n_trials * tuning_reads + val_top_k * VAL_READS

    return {
        'seed': seed,
        'objective': obj,
        'tuning_trials': n_trials,
        'tuning_reads': tuning_reads,
        'val_top_k': val_top_k,
        'val_reads': VAL_READS,
        'best_sqr': best_sqr,
        'feas_rate': feas_rate,
        'spearman_rho': spearman_rho,
        'n_validated': n_validated,
        'total_samples': total_samples,
        'time_seconds': elapsed,
    }


# ============================================================================
# MAIN LOOP: Grid Search
# ============================================================================

print("=" * 80)
print("🧪 HYPERPARAMETER GRID SEARCH – Unified Factorial Design")
print("=" * 80)
print(f"\n[Grid Factors]")
print(f"  Objectives: {OBJECTIVES}")
print(f"  Tuning Trials: {TUNING_TRIALS_LIST}")
print(f"  Tuning Reads: {TUNING_READS_LIST}")
print(f"  Validation Top K: {VAL_TOP_K_LIST}")
print(f"  Validation Reads: {VAL_READS} (fixed)")
print(f"  Seeds: {SEEDS}")
print(f"  Total configs: {len(OBJECTIVES) * len(TUNING_TRIALS_LIST) * len(TUNING_READS_LIST) * len(VAL_TOP_K_LIST)}")
print(f"  Total runs: {len(OBJECTIVES) * len(TUNING_TRIALS_LIST) * len(TUNING_READS_LIST) * len(VAL_TOP_K_LIST) * len(SEEDS)}")
print(f"\n📁 Results directory: {SAVE_DIR.resolve()}")

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

all_results = []
progress_path = SAVE_DIR / "grid_progress.pkl"
completed_configs = set()

# Load progress if exists
if progress_path.exists():
    try:
        all_results = safe_load_pickle(progress_path, [])
        # Build set of completed configs (seed, obj, n_trials, tuning_reads, val_top_k)
        for res in all_results:
            key = (res['seed'], res['objective'], res['tuning_trials'], 
                   res['tuning_reads'], res['val_top_k'])
            completed_configs.add(key)
        print(f"\n✅ Resuming from progress: {len(all_results)} runs already completed.")
    except Exception as e:
        print(f"  ⚠️ Failed to load progress: {e}")

# Generate all combinations
configs = list(itertools.product(SEEDS, OBJECTIVES, TUNING_TRIALS_LIST, 
                                  TUNING_READS_LIST, VAL_TOP_K_LIST))

total_configs = len(configs)

for idx, (seed, obj, n_trials, tuning_reads, val_top_k) in enumerate(configs):
    key = (seed, obj, n_trials, tuning_reads, val_top_k)
    
    if key in completed_configs:
        print(f"  ⏭️ Skipping config {idx+1}/{total_configs} (already done)")
        continue
    
    print(f"\n{'=' * 60}")
    print(f"Progress: {idx+1}/{total_configs} ({100*(idx+1)/total_configs:.1f}%)")
    print(f"{'=' * 60}")
    
    try:
        result = run_single_config(seed, obj, n_trials, tuning_reads, val_top_k)
        all_results.append(result)
        completed_configs.add(key)
        
        # Save progress after each run
        safe_save_pickle(progress_path, all_results, verbose=False)
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        # Store error placeholder
        all_results.append({
            'seed': seed,
            'objective': obj,
            'tuning_trials': n_trials,
            'tuning_reads': tuning_reads,
            'val_top_k': val_top_k,
            'val_reads': VAL_READS,
            'best_sqr': np.nan,
            'feas_rate': np.nan,
            'spearman_rho': np.nan,
            'n_validated': 0,
            'total_samples': n_trials * tuning_reads + val_top_k * VAL_READS,
            'time_seconds': np.nan,
            'error': str(e),
        })
        safe_save_pickle(progress_path, all_results, verbose=False)

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("📊 AGGREGATING RESULTS")
print("=" * 80)

df = pd.DataFrame(all_results)

# Aggregate across seeds
agg_df = df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k']).agg({
    'best_sqr': ['mean', 'std', 'count'],
    'feas_rate': ['mean', 'std'],
    'spearman_rho': ['mean', 'std'],
    'time_seconds': ['mean', 'std'],
    'total_samples': ['mean'],
}).reset_index()

# Flatten multi-index columns
agg_df.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k',
                  'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                  'feas_rate_mean', 'feas_rate_std',
                  'spearman_rho_mean', 'spearman_rho_std',
                  'time_mean', 'time_std',
                  'total_samples']

# Sort by best_sqr_mean (descending)
agg_df_sorted = agg_df.sort_values('best_sqr_mean', ascending=False)

# Save aggregated results
agg_df_sorted.to_csv(SAVE_DIR / "grid_results_aggregated.csv", index=False)
print(f"✅ Aggregated results saved to {SAVE_DIR / 'grid_results_aggregated.csv'}")

# ============================================================================
# RANKED TABLE (Terminal Output)
# ============================================================================

print("\n" + "=" * 80)
print("🏆 RANKED CONFIGURATIONS (by Mean SQR)")
print("=" * 80)

print(agg_df_sorted[['objective', 'tuning_trials', 'tuning_reads', 'val_top_k',
                     'best_sqr_mean', 'best_sqr_std', 'feas_rate_mean', 
                     'total_samples']].to_string(
    index=False, float_format="%.4f", max_rows=36))

# ============================================================================
# FIND WINNER
# ============================================================================

winner_row = agg_df_sorted.iloc[0]
print("\n" + "=" * 80)
print("🥇 WINNER")
print("=" * 80)
print(f"  Objective:     {winner_row['objective']}")
print(f"  Tuning Trials: {int(winner_row['tuning_trials'])}")
print(f"  Tuning Reads:  {int(winner_row['tuning_reads'])}")
print(f"  Validation Top K: {int(winner_row['val_top_k'])}")
print(f"  Mean SQR:      {winner_row['best_sqr_mean']:.6f} ± {winner_row['best_sqr_std']:.6f}")
print(f"  Mean Feas:     {winner_row['feas_rate_mean']:.4f}")
print(f"  Total Samples: {int(winner_row['total_samples'])}")

# Save winner
with open(SAVE_DIR / "grid_winner.txt", "w") as f:
    f.write(f"{winner_row['objective']}\n")
    f.write(f"Tuning Trials: {int(winner_row['tuning_trials'])}\n")
    f.write(f"Tuning Reads: {int(winner_row['tuning_reads'])}\n")
    f.write(f"Validation Top K: {int(winner_row['val_top_k'])}\n")
    f.write(f"Mean SQR: {winner_row['best_sqr_mean']:.6f}\n")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "-" * 80)
print("📊 GENERATING PLOTS")
print("-" * 80)

plot_dir = SAVE_DIR / "plots"
plot_dir.mkdir(exist_ok=True)

# 1. Heatmap (SQR vs Val Top K and Tuning Trials/Reads, faceted by Objective)
for obj in OBJECTIVES:
    sub_df = agg_df[agg_df['objective'] == obj]
    # Pivot for heatmap: rows = tuning_trials * tuning_reads, cols = val_top_k
    # Create a combined row label
    sub_df['trials_reads'] = sub_df['tuning_trials'].astype(str) + "T," + sub_df['tuning_reads'].astype(str) + "R"
    pivot = sub_df.pivot(index='trials_reads', columns='val_top_k', values='best_sqr_mean')
    if pivot.empty:
        continue
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', cbar_kws={'label': 'Mean SQR'})
    plt.title(f'Mean SQR Heatmap (Objective: {obj})')
    plt.xlabel('Validation Top K')
    plt.ylabel('Tuning Trials, Tuning Reads')
    plt.tight_layout()
    plt.savefig(plot_dir / f"heatmap_{obj}.png", dpi=150)
    plt.close()
    print(f"  ✅ Heatmap: {obj}")

# 2. Pareto Front: SQR vs Total Samples
plt.figure(figsize=(10, 6))
# Color by objective
for obj in OBJECTIVES:
    sub_df = agg_df[agg_df['objective'] == obj]
    plt.scatter(sub_df['total_samples'], sub_df['best_sqr_mean'], 
                label=obj, s=60, alpha=0.7)
plt.xlabel('Total Samples (Tuning + Validation)')
plt.ylabel('Mean Validation SQR')
plt.title('Pareto Front: SQR vs Total Samples')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / "pareto_front.png", dpi=150)
plt.close()
print("  ✅ Pareto Front")

# 3. Boxplots for Top 5 Configurations
top5 = agg_df_sorted.head(5)
top5_configs = top5[['objective', 'tuning_trials', 'tuning_reads', 'val_top_k']]
# Merge to get raw data for top 5
raw_top5 = df.merge(top5_configs, on=['objective', 'tuning_trials', 'tuning_reads', 'val_top_k'])
# Create a label for each config
raw_top5['config_label'] = (
    raw_top5['objective'] + " T" + raw_top5['tuning_trials'].astype(str) + 
    " R" + raw_top5['tuning_reads'].astype(str) + " K" + raw_top5['val_top_k'].astype(str)
)
plt.figure(figsize=(12, 6))
sns.boxplot(data=raw_top5, x='config_label', y='best_sqr')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Configuration')
plt.ylabel('Validation SQR')
plt.title('Top 5 Configurations – Distribution Across Seeds')
plt.tight_layout()
plt.savefig(plot_dir / "top5_boxplots.png", dpi=150)
plt.close()
print("  ✅ Top 5 Boxplots")

# 4. Feasibility vs SQR scatter
plt.figure(figsize=(10, 6))
for obj in OBJECTIVES:
    sub_df = agg_df[agg_df['objective'] == obj]
    plt.scatter(sub_df['feas_rate_mean'], sub_df['best_sqr_mean'], 
                label=obj, s=50, alpha=0.7)
plt.xlabel('Mean Feasibility Rate')
plt.ylabel('Mean Validation SQR')
plt.title('SQR vs Feasibility')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / "sqr_vs_feas.png", dpi=150)
plt.close()
print("  ✅ SQR vs Feasibility")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ HYPERPARAMETER GRID SEARCH COMPLETE")
print("=" * 80)
print(f"\n📁 Results saved to: {SAVE_DIR.resolve()}")
print("   - grid_results_aggregated.csv (ranked table)")
print("   - grid_winner.txt (best config)")
print("   - plots/ (heatmaps, Pareto front, boxplots)")

gc.collect()
print("\n🧹 Memory cleanup complete.")
print("=" * 80)