#@title 🧪 STRATEGY COMPARISON – Validation Resource Allocation Grid
"""
================================================================================
STRATEGY COMPARISON – Validation Resource Allocation Grid
================================================================================

This script systematically compares strategies for allocating compute budget
between tuning (Optuna) and validation. It answers:
    1. Is validation worth the extra cost?
    2. What is the optimal split between tuning trials/reads and validation top-K/reads?
    3. How does this optimal strategy scale with problem size (N)?

Design:
    - Stage 1: Full factorial grid on N=100 to find the best strategy.
    - Stage 2: Fix the winner strategy, sweep N=[150,200] and VAL_READS=[256,512,1024]
      to test scalability and validation-read sensitivity.

Real-time feedback:
    - After every single config: print tuning summary, validation summary,
      single-run summary, and update cross-strategy summary and plots.
    - Crash-proof: progress saved after every run; on resume, regenerate plots
      and print cross-strategy summary.

Outputs:
    - results/strategy_comparison_{test|full}/stage1_results.pkl
    - results/strategy_comparison_{test|full}/stage2_results.pkl (if Stage 2 run)
    - results/strategy_comparison_{test|full}/plots/ (progress plots updated live)
    - results/strategy_comparison_{test|full}/final_plots/ (publication-ready)
    - results/strategy_comparison_{test|full}/winner.txt
    - results/strategy_comparison_{test|full}/final_ranked_table.csv

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
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, "/content")
os.chdir("/content")

from src.environment import get_environment
from src.experiment import run_optuna_study, make_objective, validate_study, evaluate_qubo
from src.utils import (
    safe_save_pickle, safe_load_pickle, NumpyEncoder,
    suppress_optuna_trial_logs,
    print_tuning_summary, print_validation_summary,
    print_single_run_summary, print_cross_strategy_summary,
    cleanup_tqdm
)
from src.plotting import (
    plot_single_config_result,
    plot_cross_strategy_progress,
    regenerate_plots,
    display_saved_plots
)

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# --- Mode ---
CURRENT_MODE = 'test'  # 'test' or 'full'
FORCE_RECOMPUTE_ENV = False  # Set True if you want to rebuild environments from scratch

# --- Fixed physical parameters (kept constant across all runs) ---
K_NEW = 5
L_C = 5.0
CONNECTIVITY_RANGE = 8.0
L_W = 1.0
BETA = 1.0
DELTA = 1.0
SCHEDULE_TYPE = "old"  # confirmed winner
USE_SEED_NONE = True
TUNING_SEED = 42
VAL_SEED = 43

# --- Stage 1: Full grid on N=100 ---
# These lists define the grid. If a list has only one element, it's fixed.
if CURRENT_MODE == 'test':
    OBJECTIVES = ["Sq"]
    TUNING_TRIALS_LIST = [50]
    TUNING_READS_LIST = [50]
    VAL_TOP_K_LIST = [1, 3]          # 1 = minimal (re-eval best trial), 3, 10, 20
    VAL_READS_LIST = [128, 256]
    SEEDS = [42]
    N_LIST = [100]                    # Stage 1 only N=100
elif CURRENT_MODE == 'full':
    OBJECTIVES = ["Sq", "Pctl10", "Penalty-0.1"]
    TUNING_TRIALS_LIST = [100, 200]
    TUNING_READS_LIST = [50, 100]
    VAL_TOP_K_LIST = [1, 3, 10, 20]
    VAL_READS_LIST = [128, 256, 512]
    SEEDS = [42, 43, 44]
    N_LIST = [100]
else:
    raise ValueError(f"Unknown CURRENT_MODE: {CURRENT_MODE}")

# --- Stage 2: Scaling & validation-read sensitivity ---
RUN_STAGE2 = True
if RUN_STAGE2:
    SCALING_N_LIST = [150, 200]
    SCALING_VAL_READS_LIST = [256, 512, 1024]  # sweep around winner's val_reads
    # The winner's other params (objective, tuning_trials, tuning_reads, val_top_k)
    # will be fixed after Stage 1.

# --- Output directories ---
SAVE_DIR = Path(f"results/strategy_comparison_{CURRENT_MODE}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STUDIES_DIR = SAVE_DIR / "studies"
STUDIES_DIR.mkdir(exist_ok=True)
PLOTS_DIR = SAVE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
FINAL_PLOTS_DIR = SAVE_DIR / "final_plots"
FINAL_PLOTS_DIR.mkdir(exist_ok=True)

# --- GDrive (if in Colab) ---
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    GDRIVE_BASE = "/content/drive/MyDrive/water_quality_results"
    GDRIVE_ENABLED = True
    GDRIVE_SAVE_DIR = Path(GDRIVE_BASE) / f"strategy_comparison_{CURRENT_MODE}"
    GDRIVE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
except ImportError:
    GDRIVE_ENABLED = False
    GDRIVE_SAVE_DIR = None

# ============================================================================
# 2. PROGRESS MANAGEMENT
# ============================================================================

def load_progress():
    """
    Load saved results and completed set from disk.
    Returns:
        results_list: list of result dicts
        completed_set: set of (seed, objective, tuning_trials, tuning_reads, val_top_k, val_reads, N)
    """
    results_file = SAVE_DIR / "stage1_results.pkl"
    completed_file = SAVE_DIR / "stage1_completed.pkl"

    results = safe_load_pickle(results_file, [])
    completed = safe_load_pickle(completed_file, set())

    return results, completed


def save_progress(results, completed):
    """Save results and completed set to disk."""
    safe_save_pickle(SAVE_DIR / "stage1_results.pkl", results, verbose=False)
    safe_save_pickle(SAVE_DIR / "stage1_completed.pkl", completed, verbose=False)
    # Also save a CSV for easy inspection
    if results:
        df = pd.DataFrame(results)
        df.to_csv(SAVE_DIR / "stage1_results.csv", index=False)

    # Mirror to GDrive if enabled
    if GDRIVE_ENABLED:
        safe_save_pickle(GDRIVE_SAVE_DIR / "stage1_results.pkl", results, verbose=False)
        safe_save_pickle(GDRIVE_SAVE_DIR / "stage1_completed.pkl", completed, verbose=False)
        if results:
            df.to_csv(GDRIVE_SAVE_DIR / "stage1_results.csv", index=False)


def load_stage2_progress():
    """Same for Stage 2."""
    results_file = SAVE_DIR / "stage2_results.pkl"
    completed_file = SAVE_DIR / "stage2_completed.pkl"
    results = safe_load_pickle(results_file, [])
    completed = safe_load_pickle(completed_file, set())
    return results, completed


def save_stage2_progress(results, completed):
    safe_save_pickle(SAVE_DIR / "stage2_results.pkl", results, verbose=False)
    safe_save_pickle(SAVE_DIR / "stage2_completed.pkl", completed, verbose=False)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(SAVE_DIR / "stage2_results.csv", index=False)
    if GDRIVE_ENABLED:
        safe_save_pickle(GDRIVE_SAVE_DIR / "stage2_results.pkl", results, verbose=False)
        safe_save_pickle(GDRIVE_SAVE_DIR / "stage2_completed.pkl", completed, verbose=False)
        if results:
            df.to_csv(GDRIVE_SAVE_DIR / "stage2_results.csv", index=False)


# ============================================================================
# 3. WORKER FUNCTION: run_one_config
# ============================================================================

def run_one_config(
    seed: int,
    objective: str,
    n_trials: int,
    tuning_reads: int,
    val_top_k: int,
    val_reads: int,
    N: int,
    env: Dict,
) -> Dict:
    """
    Run a single configuration (tuning + optional validation) and return results.

    Args:
        seed: dataset seed (for environment)
        objective: objective function name (e.g., "Sq")
        n_trials: number of Optuna trials
        tuning_reads: reads per trial during tuning
        val_top_k: number of top trials to validate (1 = minimal, re-eval best only)
        val_reads: reads per validation trial
        N: number of candidate sites
        env: environment dictionary (must already be loaded)

    Returns:
        dict with keys:
            - seed, objective, tuning_trials, tuning_reads, val_top_k, val_reads, N
            - best_sqr: validation SQR of the best trial (or tuning SQR if val_top_k=0)
            - feas_rate: feasibility rate from validation
            - spearman_rho: Spearman correlation (if validation run)
            - total_samples: total QUBO samples consumed
            - best_solution: binary array (list)
            - tuning_best_sqr: best SQR from tuning (cheap)
            - tuning_feas_rate: feasibility from tuning
            - runtime: wall-clock time in seconds
    """
    start_time = time.time()

    # Build objective
    obj_func = make_objective(
        env=env,
        schedule_type=SCHEDULE_TYPE,
        objective_type=objective,
        tuning_reads=tuning_reads,
        tuning_seed=TUNING_SEED,
        use_seed_none=USE_SEED_NONE,
        compute_esr_mcr=True,
    )

    # Create study name
    study_name = f"strategy_seed{seed}_{objective}_T{n_trials}_R{tuning_reads}_K{val_top_k}_V{val_reads}_N{N}"
    # Run tuning (with progress bar visible, per-trial logs suppressed)
    study = run_optuna_study(
        experiment_name=study_name,
        objective=obj_func,
        n_trials=n_trials,
        storage_dir=STUDIES_DIR,
        directions=['minimize'],
        sampler_type='TPE',
        seed=TUNING_SEED,
        load_if_exists=False,
        verbose=False,  # We'll print our own summary
    )

    # Print tuning summary
    print_tuning_summary(study, f"Tuning: {study_name}")

    # Extract best tuning trial info
    best_trial = study.best_trial
    tuning_best_sqr = best_trial.user_attrs.get('best_sqr', np.nan)
    tuning_feas_rate = best_trial.user_attrs.get('feas_rate', np.nan)

    # --- Validation ---
    if val_top_k >= 1:
        # Run validation on top K trials
        val_results = validate_study(
            study=study,
            env=env,
            schedule_type=SCHEDULE_TYPE,
            val_reads=val_reads,
            val_seed=VAL_SEED,
            top_k=val_top_k,
            use_seed_none=USE_SEED_NONE,
            compute_esr_mcr=True,
            verbose=True,  # This will print its own summary
        )
        best_sqr = val_results.get('best_sqr', np.nan)
        feas_rate = val_results.get('feas_rate', np.nan)
        spearman_rho = val_results.get('spearman_rho', np.nan)
        # Get best solution from validation
        if val_results.get('trials'):
            best_solution = val_results['trials'][0].get('solution')
        else:
            best_solution = None
        # Print the validation summary
        print_validation_summary(val_results, f"Validation: {study_name}")
    else:
        # Minimal: re-evaluate the best trial with val_reads (this is essentially val_top_k=1)
        # We'll just call evaluate_qubo on the best trial's params
        lam1 = best_trial.params.get('lam1', 0.01)
        lam2 = best_trial.params.get('lam2', 0.01)
        num_sweeps = best_trial.params.get('num_sweeps', 10000)
        # Need schedule params depending on schedule type
        if SCHEDULE_TYPE == 'old':
            beta_min = best_trial.params.get('beta_min', 0.01)
            beta_max = best_trial.params.get('beta_max', 40.0)
            cooling_power = best_trial.params.get('cooling_power', 1.8)
            num_steps = max(10, num_sweeps // 150)
            extra_kwargs = {
                'beta_min': beta_min,
                'beta_max': beta_max,
                'cooling_power': cooling_power,
                'num_steps': num_steps,
            }
        else:
            beta_min_mult = best_trial.params.get('beta_min_mult', 1.0)
            beta_max_mult = best_trial.params.get('beta_max_mult', 1.0)
            extra_kwargs = {
                'beta_min_mult': beta_min_mult,
                'beta_max_mult': beta_max_mult,
            }

        result = evaluate_qubo(
            env=env,
            lam1=lam1,
            lam2=lam2,
            num_sweeps=num_sweeps,
            schedule_type=SCHEDULE_TYPE,
            num_reads=val_reads,
            seed=VAL_SEED,
            return_all=True,
            use_seed_none=USE_SEED_NONE,
            compute_esr_mcr=True,
            verbose=False,
            **extra_kwargs,
        )
        best_sqr = result.get('best_sqr', np.nan)
        feas_rate = result.get('feas_rate', np.nan)
        spearman_rho = np.nan
        best_solution = result.get('best_solution')
        # Print a mini summary for this "top-1 validation"
        print("\n" + "=" * 80)
        print("📊 TOP-1 VALIDATION (minimal)")
        print("=" * 80)
        print(f"  Best Trial #: {best_trial.number}")
        print(f"  Re-evaluated SQR: {_safe_format(best_sqr, '.6f')}")
        print(f"  Feasibility: {_safe_format(feas_rate, '.4f')}")
        print("=" * 80)

    # Total samples consumed
    total_samples = n_trials * tuning_reads + val_top_k * val_reads

    elapsed = time.time() - start_time

    result_dict = {
        'seed': seed,
        'objective': objective,
        'tuning_trials': n_trials,
        'tuning_reads': tuning_reads,
        'val_top_k': val_top_k,
        'val_reads': val_reads,
        'N': N,
        'best_sqr': best_sqr,
        'feas_rate': feas_rate,
        'spearman_rho': spearman_rho,
        'total_samples': total_samples,
        'best_solution': best_solution.tolist() if best_solution is not None else None,
        'tuning_best_sqr': tuning_best_sqr,
        'tuning_feas_rate': tuning_feas_rate,
        'runtime': elapsed,
    }
    return result_dict


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("🧪 STRATEGY COMPARISON – Validation Resource Allocation Grid")
    print("=" * 80)
    print(f"  Mode: {CURRENT_MODE.upper()}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Objectives: {OBJECTIVES}")
    print(f"  Tuning Trials: {TUNING_TRIALS_LIST}")
    print(f"  Tuning Reads: {TUNING_READS_LIST}")
    print(f"  Val Top K: {VAL_TOP_K_LIST}")
    print(f"  Val Reads: {VAL_READS_LIST}")
    print(f"  N (Stage 1): {N_LIST}")
    if RUN_STAGE2:
        print(f"  Stage 2: Scaling N={SCALING_N_LIST}, Val Reads sweep={SCALING_VAL_READS_LIST}")
    else:
        print("  Stage 2: Disabled")
    print("=" * 80)

    # Suppress Optuna per-trial logs
    suppress_optuna_trial_logs()

    # ========================================================================
    # STAGE 1: Full grid on N=100
    # ========================================================================
    print("\n" + "=" * 80)
    print("📌 STAGE 1: Grid Search on N=100")
    print("=" * 80)

    # Load progress
    results, completed = load_progress()

    # Generate all configs for Stage 1
    stage1_configs = list(itertools.product(
        SEEDS,
        OBJECTIVES,
        TUNING_TRIALS_LIST,
        TUNING_READS_LIST,
        VAL_TOP_K_LIST,
        VAL_READS_LIST,
        N_LIST  # N is fixed to 100
    ))

    # If resuming, print cross-summary and regenerate plots
    if results:
        df_existing = pd.DataFrame(results)
        print(f"\n✅ Resuming from saved progress: {len(results)} runs completed.")
        print_cross_strategy_summary(df_existing, title="Stage 1 Progress So Far")
        regenerate_plots(df_existing, PLOTS_DIR, show_fig=True)
        print("Resuming...\n")

    # Outer loop: seeds
    total_configs = len(stage1_configs)
    # We'll use a tqdm for the entire Stage 1 (global)
    global_pbar = tqdm(total=total_configs, desc="Stage 1 Overall", position=0, leave=True)

    # We'll also want per-seed progress bars, but we need to handle them inside the seed loop.
    # Since tqdm can be nested, we'll create a per-seed bar inside.
    # However, we need to update the global bar manually.

    for seed in SEEDS:
        # Filter configs for this seed
        seed_configs = [c for c in stage1_configs if c[0] == seed]
        seed_pbar = tqdm(total=len(seed_configs), desc=f"Seed {seed}", position=1, leave=False)

        # Load environment once per seed (cached)
        env = get_environment(
            seed=seed,
            K_new=K_NEW,
            L_c=L_C,
            L_w=L_W,
            beta=BETA,
            delta=DELTA,
            connectivity_range=CONNECTIVITY_RANGE,
            current_vector=(1.0, 0.0),
            force_recompute=FORCE_RECOMPUTE_ENV,
            verbose=False,
            gdrive_base=GDRIVE_BASE if GDRIVE_ENABLED else None,
        )

        for config in seed_configs:
            (seed_c, objective, n_trials, tuning_reads, val_top_k, val_reads, N) = config
            key = (seed_c, objective, n_trials, tuning_reads, val_top_k, val_reads, N)

            if key in completed:
                seed_pbar.update(1)
                global_pbar.update(1)
                continue

            # Run the config
            try:
                result = run_one_config(
                    seed=seed_c,
                    objective=objective,
                    n_trials=n_trials,
                    tuning_reads=tuning_reads,
                    val_top_k=val_top_k,
                    val_reads=val_reads,
                    N=N,
                    env=env,
                )
                results.append(result)
                completed.add(key)
                # Save progress
                save_progress(results, completed)

                # Print single-run summary
                print_single_run_summary(result, result, result['runtime'])

                # Update plots and cross-summary with ALL completed runs
                df_all = pd.DataFrame(results)
                print_cross_strategy_summary(df_all, title="Stage 1 Progress (All Configs So Far)")
                plot_cross_strategy_progress(df_all, PLOTS_DIR, show_fig=True)
                # Also try to plot single config result (use the last result)
                try:
                    # Need the solution from the result
                    sol = result.get('best_solution')
                    if sol is not None:
                        # Reconstruct solution as numpy array
                        sol_array = np.array(sol)
                        # We need to pass env and config and result
                        plot_single_config_result(
                            env=env,
                            config=result,
                            result=result,
                            save_dir=PLOTS_DIR,
                            show_fig=True
                        )
                except Exception as e:
                    print(f"  ⚠️ Single-run plot failed: {e}")

            except Exception as e:
                print(f"  ❌ Config {key} failed: {e}")
                # Store error placeholder
                error_result = {
                    'seed': seed_c,
                    'objective': objective,
                    'tuning_trials': n_trials,
                    'tuning_reads': tuning_reads,
                    'val_top_k': val_top_k,
                    'val_reads': val_reads,
                    'N': N,
                    'best_sqr': np.nan,
                    'feas_rate': np.nan,
                    'spearman_rho': np.nan,
                    'total_samples': n_trials * tuning_reads + val_top_k * val_reads,
                    'best_solution': None,
                    'tuning_best_sqr': np.nan,
                    'tuning_feas_rate': np.nan,
                    'runtime': -1,
                    'error': str(e),
                }
                results.append(error_result)
                completed.add(key)
                save_progress(results, completed)

            seed_pbar.update(1)
            global_pbar.update(1)

        seed_pbar.close()
        # After finishing a seed, print its summary (optional)
        print(f"\n✅ Seed {seed} complete. Total completed: {len(completed)} / {total_configs}")

    global_pbar.close()

    # ---- Stage 1 final summary and winner ----
    df_stage1 = pd.DataFrame(results)
    # Filter out errors (where best_sqr is nan)
    df_valid = df_stage1[df_stage1['best_sqr'].notna()]
    if not df_valid.empty:
        # Find winner: group by config (excluding seed) and take mean SQR across seeds
        winner_agg = df_valid.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
            'best_sqr': ['mean', 'std', 'count'],
            'feas_rate': ['mean'],
            'spearman_rho': ['mean'],
            'total_samples': ['mean']
        }).reset_index()
        winner_agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                              'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                              'feas_mean', 'rho_mean', 'samples_mean']
        winner_row = winner_agg.loc[winner_agg['best_sqr_mean'].idxmax()]
        winner = {
            'objective': winner_row['objective'],
            'tuning_trials': int(winner_row['tuning_trials']),
            'tuning_reads': int(winner_row['tuning_reads']),
            'val_top_k': int(winner_row['val_top_k']),
            'val_reads': int(winner_row['val_reads']),
            'N': int(winner_row['N']),
            'best_sqr_mean': winner_row['best_sqr_mean'],
            'best_sqr_std': winner_row['best_sqr_std'],
            'feas_mean': winner_row['feas_mean'],
            'rho_mean': winner_row['rho_mean'],
            'samples_mean': winner_row['samples_mean'],
        }

        print("\n" + "=" * 80)
        print("🏆 STAGE 1 WINNER")
        print("=" * 80)
        for k, v in winner.items():
            print(f"  {k}: {v}")
        print("=" * 80)

        # Save winner
        with open(SAVE_DIR / "stage1_winner.json", "w") as f:
            json.dump(winner, f, indent=2, cls=NumpyEncoder)
        if GDRIVE_ENABLED:
            with open(GDRIVE_SAVE_DIR / "stage1_winner.json", "w") as f:
                json.dump(winner, f, indent=2, cls=NumpyEncoder)

        # Also save ranked table
        winner_agg_sorted = winner_agg.sort_values('best_sqr_mean', ascending=False)
        winner_agg_sorted.to_csv(SAVE_DIR / "stage1_ranked_table.csv", index=False)
        if GDRIVE_ENABLED:
            winner_agg_sorted.to_csv(GDRIVE_SAVE_DIR / "stage1_ranked_table.csv", index=False)

        # Generate final publication-ready plots for Stage 1
        print("\n🎨 Generating final Stage 1 plots...")
        plot_cross_strategy_progress(df_valid, FINAL_PLOTS_DIR, show_fig=False)  # save only, not show
        print("  ✅ Final plots saved to", FINAL_PLOTS_DIR)
    else:
        print("\n⚠️ No valid Stage 1 results to determine winner.")
        winner = None

    # ========================================================================
    # STAGE 2: Scaling and validation-read sensitivity
    # ========================================================================
    if RUN_STAGE2 and winner is not None:
        print("\n" + "=" * 80)
        print("📌 STAGE 2: Scaling to larger N and VAL_READS sensitivity")
        print("=" * 80)

        # Load Stage 2 progress
        stage2_results, stage2_completed = load_stage2_progress()

        # Generate configs for Stage 2:
        # Fixed: objective, tuning_trials, tuning_reads, val_top_k (from winner)
        # Sweep: N (from SCALING_N_LIST) and val_reads (from SCALING_VAL_READS_LIST)
        # Also include the winner's original N and val_reads for comparison.
        fixed_obj = winner['objective']
        fixed_trials = winner['tuning_trials']
        fixed_reads = winner['tuning_reads']
        fixed_topk = winner['val_top_k']

        # Create a list of (N, val_reads) pairs
        sweep_pairs = []
        # Include the winner's own N and val_reads for baseline
        sweep_pairs.append((winner['N'], winner['val_reads']))
        for N in SCALING_N_LIST:
            for vr in SCALING_VAL_READS_LIST:
                sweep_pairs.append((N, vr))

        # Generate configs for all seeds
        stage2_configs = list(itertools.product(
            SEEDS,
            [fixed_obj],
            [fixed_trials],
            [fixed_reads],
            [fixed_topk],
            [vr for _, vr in sweep_pairs],
            [N for N, _ in sweep_pairs]
        ))

        # Remove duplicates (the winner's config might appear twice)
        stage2_configs = list(set(stage2_configs))

        total_stage2 = len(stage2_configs)
        print(f"  Stage 2 configs: {total_stage2}")

        # We'll reuse the same progress bar structure
        global_pbar2 = tqdm(total=total_stage2, desc="Stage 2 Overall", position=0, leave=True)

        for seed in SEEDS:
            seed_configs = [c for c in stage2_configs if c[0] == seed]
            seed_pbar2 = tqdm(total=len(seed_configs), desc=f"Seed {seed} (Stage2)", position=1, leave=False)

            # Load environment per seed and N (this might be slow if N changes)
            # We'll load a new env for each N, but we can cache by (seed, N)
            env_cache = {}
            for config in seed_configs:
                (seed_c, obj, n_trials, tuning_reads, val_top_k, val_reads, N) = config
                if (seed_c, N) not in env_cache:
                    env_cache[(seed_c, N)] = get_environment(
                        seed=seed_c,
                        K_new=K_NEW,
                        L_c=L_C,
                        L_w=L_W,
                        beta=BETA,
                        delta=DELTA,
                        connectivity_range=CONNECTIVITY_RANGE,
                        current_vector=(1.0, 0.0),
                        force_recompute=FORCE_RECOMPUTE_ENV,
                        verbose=False,
                        gdrive_base=GDRIVE_BASE if GDRIVE_ENABLED else None,
                    )
                env = env_cache[(seed_c, N)]

                key = (seed_c, obj, n_trials, tuning_reads, val_top_k, val_reads, N)
                if key in stage2_completed:
                    seed_pbar2.update(1)
                    global_pbar2.update(1)
                    continue

                try:
                    result = run_one_config(
                        seed=seed_c,
                        objective=obj,
                        n_trials=n_trials,
                        tuning_reads=tuning_reads,
                        val_top_k=val_top_k,
                        val_reads=val_reads,
                        N=N,
                        env=env,
                    )
                    stage2_results.append(result)
                    stage2_completed.add(key)
                    save_stage2_progress(stage2_results, stage2_completed)

                    print_single_run_summary(result, result, result['runtime'])
                    # Update cross-summary for Stage 2
                    df_stage2 = pd.DataFrame(stage2_results)
                    print_cross_strategy_summary(df_stage2, title="Stage 2 Progress")
                    # Plot progress (using the same plotting function, but we'll save in PLOTS_DIR)
                    plot_cross_strategy_progress(df_stage2, PLOTS_DIR, show_fig=True)

                except Exception as e:
                    print(f"  ❌ Stage2 config {key} failed: {e}")
                    error_result = {
                        'seed': seed_c,
                        'objective': obj,
                        'tuning_trials': n_trials,
                        'tuning_reads': tuning_reads,
                        'val_top_k': val_top_k,
                        'val_reads': val_reads,
                        'N': N,
                        'best_sqr': np.nan,
                        'feas_rate': np.nan,
                        'spearman_rho': np.nan,
                        'total_samples': n_trials * tuning_reads + val_top_k * val_reads,
                        'best_solution': None,
                        'tuning_best_sqr': np.nan,
                        'tuning_feas_rate': np.nan,
                        'runtime': -1,
                        'error': str(e),
                    }
                    stage2_results.append(error_result)
                    stage2_completed.add(key)
                    save_stage2_progress(stage2_results, stage2_completed)

                seed_pbar2.update(1)
                global_pbar2.update(1)

            seed_pbar2.close()

        global_pbar2.close()

        # Stage 2 final summary
        df_stage2 = pd.DataFrame(stage2_results)
        df_stage2_valid = df_stage2[df_stage2['best_sqr'].notna()]
        if not df_stage2_valid.empty:
            print("\n" + "=" * 80)
            print("📊 STAGE 2 FINAL SUMMARY")
            print("=" * 80)
            # Group by N and val_reads, show mean SQR and feasibility
            stage2_agg = df_stage2_valid.groupby(['N', 'val_reads']).agg({
                'best_sqr': ['mean', 'std', 'count'],
                'feas_rate': ['mean'],
                'total_samples': ['mean']
            }).reset_index()
            stage2_agg.columns = ['N', 'val_reads', 'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                                  'feas_mean', 'samples_mean']
            print(stage2_agg.to_string(index=False, float_format="%.4f"))
            # Save
            stage2_agg.to_csv(SAVE_DIR / "stage2_summary.csv", index=False)
            if GDRIVE_ENABLED:
                stage2_agg.to_csv(GDRIVE_SAVE_DIR / "stage2_summary.csv", index=False)

            # Create scaling plot: SQR vs N for different val_reads
            plt.figure(figsize=(10, 6))
            for vr in stage2_agg['val_reads'].unique():
                sub = stage2_agg[stage2_agg['val_reads'] == vr]
                plt.plot(sub['N'], sub['best_sqr_mean'], marker='o', label=f'val_reads={vr}')
            plt.xlabel('Number of Candidate Sites (N)')
            plt.ylabel('Mean Validation SQR')
            plt.title('Scaling Performance with N')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FINAL_PLOTS_DIR / "scaling_plot.png", dpi=150)
            plt.show()
            print("  ✅ Scaling plot saved to", FINAL_PLOTS_DIR / "scaling_plot.png")

    # ========================================================================
    # FINAL CLEANUP
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ STRATEGY COMPARISON COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {SAVE_DIR.resolve()}")
    if GDRIVE_ENABLED:
        print(f"📁 GDrive backup: {GDRIVE_SAVE_DIR.resolve()}")
    print("  - Stage 1 results: stage1_results.pkl, stage1_ranked_table.csv")
    if RUN_STAGE2 and winner is not None:
        print("  - Stage 2 results: stage2_results.pkl, stage2_summary.csv")
    print("  - Plots: plots/ (progress), final_plots/ (publication-ready)")
    print("  - Winner: stage1_winner.json")
    print("=" * 80)

    cleanup_tqdm()
    gc.collect()
    print("🧹 Memory cleanup complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()