#@title 🧪 STRATEGY COMPARISON – Validation Resource Allocation Grid (with W&B)
"""
================================================================================
STRATEGY COMPARISON – Validation Resource Allocation Grid
================================================================================

This script systematically compares strategies for allocating compute budget
between tuning (Optuna) and validation. It answers:
    1. Is validation worth the extra cost?
    2. What is the optimal split between tuning trials/reads and validation top-K/reads?
    3. How does this optimal strategy scale with problem size (N)?

Features:
    - Real‑time summaries and plots after every single config.
    - Crash‑proof: progress saved after each run; resumes with regenerated plots.
    - W&B integration for live dashboard and artifact tracking.
    - Two‑stage design: Stage 1 grid on N=100; Stage 2 scales N and val_reads.
    - Metrics: SQR, feasibility, runtime, samples, efficiency, Spearman ρ, CV.
    - Rankings: by Efficiency, by SQR, by CV.

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
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

sys.path.insert(0, "/content")
os.chdir("/content")

# ---- W&B ----
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Install with: !pip install wandb")

from src.environment import get_environment
from src.experiment import run_optuna_study, make_objective, validate_study, evaluate_qubo
from src.utils import (
    safe_save_pickle, safe_load_pickle, NumpyEncoder,
    suppress_optuna_trial_logs,
    print_tuning_summary, print_validation_summary,
    print_single_run_summary, print_cross_strategy_summary, print_cross_strategy_metrics,
    print_config_header,
    cleanup_tqdm
)
from src.plotting import (
    plot_deployment_with_qubo,
    plot_gurobi_baseline,
    plot_optuna_learning_curve_enhanced,
    plot_cross_strategy_progress,
    plot_sqr_vs_runtime,
    regenerate_plots,
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
if CURRENT_MODE == 'test':
    OBJECTIVES = ["Sq"]
    TUNING_TRIALS_LIST = [50]
    TUNING_READS_LIST = [50]
    VAL_TOP_K_LIST = [1, 3]          # 1 = minimal (re-eval best trial), 3, 10
    VAL_READS_LIST = [128, 256]
    SEEDS = [42]
    N_LIST = [100]                    # Stage 1 only N=100
elif CURRENT_MODE == 'full':
    OBJECTIVES = ["Sq", "Pctl10", "Penalty-0.1"]
    TUNING_TRIALS_LIST = [100, 200]
    TUNING_READS_LIST = [50, 100]
    VAL_TOP_K_LIST = [1, 3, 10]       # no 20
    VAL_READS_LIST = [128, 256, 512]
    SEEDS = [42, 43, 44]
    N_LIST = [100]
else:
    raise ValueError(f"Unknown CURRENT_MODE: {CURRENT_MODE}")

# --- Stage 2: Scaling & validation-read sensitivity ---
RUN_STAGE2 = True
if RUN_STAGE2:
    # Sweep only N and val_reads; exclude N=100 (already done in Stage 1)
    SCALING_N_LIST = [150] if CURRENT_MODE == 'test' else [150, 200]
    SCALING_VAL_READS_LIST = [512] if CURRENT_MODE == 'test' else [256, 512, 1024]
else:
    SCALING_N_LIST = []
    SCALING_VAL_READS_LIST = []

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

# --- W&B Configuration ---
PROJECT_NAME = "wqm-placement-optimization"
if WANDB_AVAILABLE:
    # Will be initialized after config
    pass

# ============================================================================
# 2. PROGRESS MANAGEMENT
# ============================================================================

def load_progress(stage=1):
    """Load saved results and completed set for a given stage."""
    if stage == 1:
        results_file = SAVE_DIR / "stage1_results.pkl"
        completed_file = SAVE_DIR / "stage1_completed.pkl"
    else:
        results_file = SAVE_DIR / "stage2_results.pkl"
        completed_file = SAVE_DIR / "stage2_completed.pkl"
    results = safe_load_pickle(results_file, [])
    completed = safe_load_pickle(completed_file, set())
    return results, completed

def save_progress(results, completed, stage=1):
    """Save results and completed set for a given stage."""
    if stage == 1:
        results_file = SAVE_DIR / "stage1_results.pkl"
        completed_file = SAVE_DIR / "stage1_completed.pkl"
        csv_file = SAVE_DIR / "stage1_results.csv"
    else:
        results_file = SAVE_DIR / "stage2_results.pkl"
        completed_file = SAVE_DIR / "stage2_completed.pkl"
        csv_file = SAVE_DIR / "stage2_results.csv"

    safe_save_pickle(results_file, results, verbose=False)
    safe_save_pickle(completed_file, completed, verbose=False)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)

    # Mirror to GDrive if enabled
    if GDRIVE_ENABLED:
        safe_save_pickle(GDRIVE_SAVE_DIR / results_file.name, results, verbose=False)
        safe_save_pickle(GDRIVE_SAVE_DIR / completed_file.name, completed, verbose=False)
        if results:
            df.to_csv(GDRIVE_SAVE_DIR / csv_file.name, index=False)

# ============================================================================
# 3. W&B INITIALIZATION
# ============================================================================

def init_wandb(stage, run_name, config):
    """Initialize W&B run if available."""
    if not WANDB_AVAILABLE:
        return None
    try:
        wandb.init(
            project=PROJECT_NAME,
            config=config,
            tags=[CURRENT_MODE, "strategy_comparison", f"stage{stage}"],
            group=f"stage{stage}",
            name=run_name,
            reinit=True,
        )
        return wandb
    except Exception as e:
        print(f"  ⚠️ W&B initialization failed: {e}")
        return None

# ============================================================================
# 4. WORKER FUNCTION: run_one_config
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
    # Run tuning
    study = run_optuna_study(
        experiment_name=study_name,
        objective=obj_func,
        n_trials=n_trials,
        storage_dir=STUDIES_DIR,
        directions=['minimize'],
        sampler_type='TPE',
        seed=TUNING_SEED,
        load_if_exists=True,  # crash recovery
        verbose=False,
    )

    # Print tuning summary (horizontal compact)
    print_tuning_summary(study, f"Tuning: {study_name}", compact=True, width=200)

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
        if val_results.get('trials'):
            best_solution = val_results['trials'][0].get('solution')
            # Extract hyperparameters from the best validation trial
            lam1 = val_results['trials'][0].get('lam1', np.nan)
            lam2 = val_results['trials'][0].get('lam2', np.nan)
            num_sweeps = val_results['trials'][0].get('num_sweeps', np.nan)
        else:
            best_solution = None
            lam1, lam2, num_sweeps = np.nan, np.nan, np.nan
        # Print the validation summary (horizontal)
        print_validation_summary(val_results, f"Validation: {study_name}", compact=True, width=200)
    else:
        # Minimal: re-evaluate the best trial with val_reads (this is essentially val_top_k=1)
        # We'll just call evaluate_qubo on the best trial's params
        lam1 = best_trial.params.get('lam1', 0.01)
        lam2 = best_trial.params.get('lam2', 0.01)
        num_sweeps = best_trial.params.get('num_sweeps', 10000)
        # Need schedule params
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

    # Effective SQR = max(tuning, validation)
    effective_sqr = max(tuning_best_sqr, best_sqr) if not (np.isnan(tuning_best_sqr) or np.isnan(best_sqr)) else (tuning_best_sqr if not np.isnan(tuning_best_sqr) else best_sqr)

    # Extract SA schedule parameters from the best trial (if available)
    # They are stored in the user attrs or params
    if SCHEDULE_TYPE == 'old':
        beta_min = best_trial.params.get('beta_min', np.nan)
        beta_max = best_trial.params.get('beta_max', np.nan)
        cooling_power = best_trial.params.get('cooling_power', np.nan)
        num_steps = best_trial.params.get('num_steps_actual', np.nan)
    else:
        beta_min = np.nan
        beta_max = np.nan
        cooling_power = np.nan
        num_steps = np.nan

    result_dict = {
        'seed': seed,
        'objective': objective,
        'tuning_trials': n_trials,
        'tuning_reads': tuning_reads,
        'val_top_k': val_top_k,
        'val_reads': val_reads,
        'N': N,
        'best_sqr': best_sqr,
        'effective_sqr': effective_sqr,
        'tuning_best_sqr': tuning_best_sqr,
        'tuning_feas_rate': tuning_feas_rate,
        'feas_rate': feas_rate,
        'spearman_rho': spearman_rho,
        'total_samples': total_samples,
        'best_solution': best_solution.tolist() if best_solution is not None else None,
        'runtime': elapsed,
        'lam1': lam1,
        'lam2': lam2,
        'num_sweeps': num_sweeps,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'cooling_power': cooling_power,
        'num_steps': num_steps,
        'miqp_energy': result.get('best_miqp', np.nan) if 'result' in locals() else np.nan,
    }
    return result_dict

# Helper for formatting
def _safe_format(val, fmt=".4f"):
    if isinstance(val, (int, float)) and np.isfinite(val):
        return f"{val:{fmt}}"
    return "N/A"

# ============================================================================
# 5. MAIN LOOP: Run Stage 1 and Stage 2
# ============================================================================

def run_stage(stage, configs, stage_name, desc):
    """
    Run a list of configs (each a tuple of parameters) for a given stage.
    Returns aggregated results DataFrame.
    """
    # Load progress
    results, completed = load_progress(stage)
    total_configs = len(configs)

    if results:
        print(f"\n✅ Resuming {stage_name}: {len(results)} runs already completed.")
        df_existing = pd.DataFrame(results)
        print_cross_strategy_summary(df_existing, title=f"{stage_name} Progress So Far")
        regenerate_plots(df_existing, PLOTS_DIR, show_fig=True)
        # Also print metrics
        print_cross_strategy_metrics(df_existing, title=f"{stage_name} Metrics")

    # ---- Initialize progress bars ----
    # Global bar
    global_bar = tqdm(total=total_configs, desc=f"{stage_name} Overall", position=0, leave=True)
    # Per-seed bars
    seed_bars = {}
    for seed in SEEDS:
        seed_bars[seed] = tqdm(total=len([c for c in configs if c[0] == seed]),
                               desc=f"Seed {seed}", position=SEEDS.index(seed)+1, leave=False)

    # For W&B, we'll log per config, not per seed

    # ---- Outer loop: seeds ----
    for seed in SEEDS:
        # Filter configs for this seed
        seed_configs = [c for c in configs if c[0] == seed]
        if not seed_configs:
            continue

        # Load environment once per seed (cached)
        # N is the first element of the config that is variable; but we assume N is fixed for Stage 1.
        # For Stage 2, N varies. We'll load env per N as well.
        # We'll load env for the first config's N (if stage 2, we'll load per N later)
        # To simplify, we'll load per (seed, N) and cache.
        env_cache = {}
        for config_item in seed_configs:
            (seed_c, objective, n_trials, tuning_reads, val_top_k, val_reads, N) = config_item
            key = (seed_c, N)
            if key not in env_cache:
                env = get_environment(
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
                    N=N,
                )
                env_cache[key] = env
            else:
                env = env_cache[key]

            # At seed start for the first N, print Gurobi baseline
            # We'll do it once per seed (when the first config runs)
            if config_item == seed_configs[0]:
                print("\n" + "=" * 80)
                print(f"🌱 SEED {seed} START (N={N})")
                print("=" * 80)
                print(f"Gurobi MIQP baseline: {env['gurobi_miqp']:.6f} (SQR=1.0000)")
                # Generate and display Gurobi deployment plot
                gurobi_path = PLOTS_DIR / f"gurobi_baseline_seed{seed}_N{N}.png"
                plot_gurobi_baseline(env, gurobi_path, show_fig=True)

            # Check if already completed
            key = (seed_c, objective, n_trials, tuning_reads, val_top_k, val_reads, N)
            if key in completed:
                seed_bars[seed].update(1)
                global_bar.update(1)
                continue

            # ---- Run the config ----
            print_config_header({
                'seed': seed_c,
                'objective': objective,
                'tuning_trials': n_trials,
                'tuning_reads': tuning_reads,
                'val_top_k': val_top_k,
                'val_reads': val_reads,
                'N': N,
            })

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
                save_progress(results, completed, stage)

                # ---- Print single-run summary ----
                print_single_run_summary(result, result, result['runtime'])

                # ---- Update cross-strategy summaries and plots ----
                df_all = pd.DataFrame(results)
                print_cross_strategy_summary(df_all, title=f"{stage_name} Progress (All Configs So Far)")
                print_cross_strategy_metrics(df_all, title=f"{stage_name} Metrics")

                # Update cross-strategy progress plots
                plot_cross_strategy_progress(df_all, PLOTS_DIR, show_fig=True)
                plot_sqr_vs_runtime(df_all, PLOTS_DIR / "sqr_vs_runtime.png", show_fig=True)

                # ---- Generate per-config deployment+QUBO plot ----
                # We need lam1, lam2 from the result
                lam1 = result.get('lam1', np.nan)
                lam2 = result.get('lam2', np.nan)
                solution = result.get('best_solution')
                if solution is not None and not np.isnan(lam1) and not np.isnan(lam2):
                    solution_arr = np.array(solution)
                    selected_new = [i for i in range(len(solution_arr)) if solution_arr[i] == 1 and i not in env['M_indices']]
                    title = f"Seed {seed_c}, Obj {objective}, N={N}, SQR={result['effective_sqr']:.4f}"
                    deploy_path = PLOTS_DIR / f"deployment_seed{seed_c}_{objective}_N{N}_T{n_trials}_R{tuning_reads}_K{val_top_k}_V{val_reads}.png"
                    plot_deployment_with_qubo(
                        env=env,
                        lam1=lam1,
                        lam2=lam2,
                        solution=solution_arr,
                        M_indices=env['M_indices'],
                        selected_new=selected_new,
                        U=env['U'],
                        coords=env['coords'],
                        DOMAIN_SIZE=50.0,
                        current_vector=(1.0, 0.0),
                        CONNECTIVITY_RANGE=env['connectivity_range'],
                        title=title,
                        save_path=deploy_path,
                        show_fig=True,
                    )

                # ---- Log learning curve ----
                # We need to get the study from the run_one_config? It's not returned.
                # We can re-fetch the study from the DB or we can already have the study object.
                # To simplify, we'll skip logging learning curve for now, or we can re-create it.
                # For the MVP, we'll just log the plots we already have.

                # ---- Log to W&B ----
                if WANDB_AVAILABLE and wandb.run is not None:
                    try:
                        # Log metrics
                        wandb.log({
                            "best_sqr": result['best_sqr'],
                            "effective_sqr": result['effective_sqr'],
                            "feas_rate": result['feas_rate'],
                            "spearman_rho": result['spearman_rho'],
                            "runtime_seconds": result['runtime'],
                            "total_samples": result['total_samples'],
                            "efficiency_time": result['effective_sqr'] / result['runtime'] if result['runtime'] > 0 else 0,
                            "efficiency_samples": result['effective_sqr'] / result['total_samples'] if result['total_samples'] > 0 else 0,
                            "lambda1": result['lam1'],
                            "lambda2": result['lam2'],
                            "num_sweeps": result['num_sweeps'],
                            "beta_min": result['beta_min'],
                            "beta_max": result['beta_max'],
                            "cooling_power": result['cooling_power'],
                            "num_steps": result['num_steps'],
                            "seed": seed_c,
                            "objective": objective,
                            "tuning_trials": n_trials,
                            "tuning_reads": tuning_reads,
                            "val_top_k": val_top_k,
                            "val_reads": val_reads,
                            "N": N,
                            "miqp_energy": result.get('miqp_energy', np.nan),
                            "stage": stage,
                        })
                        # Log images if they exist
                        if deploy_path.exists():
                            wandb.log({"deployment_map": wandb.Image(str(deploy_path))})
                        # Log cross-strategy plots
                        for fname in ["heatmap_Sq_N100.png", "pareto_front_samples.png", "sqr_vs_feas.png", "sqr_vs_runtime.png"]:
                            p = PLOTS_DIR / fname
                            if p.exists():
                                wandb.log({f"cross_{fname}": wandb.Image(str(p))})
                    except Exception as e:
                        print(f"  ⚠️ W&B logging failed: {e}")

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
                    'effective_sqr': np.nan,
                    'tuning_best_sqr': np.nan,
                    'tuning_feas_rate': np.nan,
                    'feas_rate': np.nan,
                    'spearman_rho': np.nan,
                    'total_samples': n_trials * tuning_reads + val_top_k * val_reads,
                    'best_solution': None,
                    'runtime': -1,
                    'lam1': np.nan,
                    'lam2': np.nan,
                    'num_sweeps': np.nan,
                    'beta_min': np.nan,
                    'beta_max': np.nan,
                    'cooling_power': np.nan,
                    'num_steps': np.nan,
                    'miqp_energy': np.nan,
                    'error': str(e),
                }
                results.append(error_result)
                completed.add(key)
                save_progress(results, completed, stage)

            seed_bars[seed].update(1)
            global_bar.update(1)

        seed_bars[seed].close()

    global_bar.close()

    # Return aggregated DataFrame
    df = pd.DataFrame(results)
    return df

# ============================================================================
# 6. MAIN EXECUTION
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

    # ---- Initialize W&B ----
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            config = {
                "mode": CURRENT_MODE,
                "schedule_type": SCHEDULE_TYPE,
                "K_new": K_NEW,
                "L_c": L_C,
                "connectivity_range": CONNECTIVITY_RANGE,
                "objectives": OBJECTIVES,
                "tuning_trials_list": TUNING_TRIALS_LIST,
                "tuning_reads_list": TUNING_READS_LIST,
                "val_top_k_list": VAL_TOP_K_LIST,
                "val_reads_list": VAL_READS_LIST,
                "seeds": SEEDS,
                "stage1_N": N_LIST,
                "stage2_N": SCALING_N_LIST if RUN_STAGE2 else None,
                "stage2_val_reads": SCALING_VAL_READS_LIST if RUN_STAGE2 else None,
            }
            run_name = f"stage1_{CURRENT_MODE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = init_wandb(1, run_name, config)
        except Exception as e:
            print(f"  ⚠️ W&B init failed: {e}")

    # ========================================================================
    # STAGE 1: Full grid on N=100
    # ========================================================================
    print("\n" + "=" * 80)
    print("📌 STAGE 1: Grid Search on N=100")
    print("=" * 80)

    # Generate configs
    stage1_configs = list(itertools.product(
        SEEDS,
        OBJECTIVES,
        TUNING_TRIALS_LIST,
        TUNING_READS_LIST,
        VAL_TOP_K_LIST,
        VAL_READS_LIST,
        N_LIST  # N is fixed to 100
    ))

    df_stage1 = run_stage(1, stage1_configs, "Stage 1", "Stage 1")

    # ---- Stage 1 final aggregation and winner ----
    if not df_stage1.empty:
        # Filter out errors
        df_valid = df_stage1[df_stage1['best_sqr'].notna() & df_stage1['best_sqr'] > 0]
        if not df_valid.empty:
            # Group by config (excluding seed) to get mean SQR, runtime, etc.
            winner_agg = df_valid.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
                'best_sqr': ['mean', 'std', 'count'],
                'effective_sqr': ['mean', 'std'],
                'feas_rate': ['mean'],
                'spearman_rho': ['mean'],
                'total_samples': ['mean'],
                'runtime': ['mean', 'std']
            }).reset_index()
            winner_agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                                  'sqr_mean', 'sqr_std', 'n_seeds',
                                  'eff_sqr_mean', 'eff_sqr_std',
                                  'feas_mean', 'rho_mean',
                                  'samples_mean',
                                  'time_mean', 'time_std']

            # Compute efficiency
            winner_agg['efficiency_time'] = winner_agg['eff_sqr_mean'] / winner_agg['time_mean']
            winner_agg['efficiency_samples'] = winner_agg['eff_sqr_mean'] / winner_agg['samples_mean']

            # Find winners
            winner_eff = winner_agg.loc[winner_agg['efficiency_time'].idxmax()]
            winner_sqr = winner_agg.loc[winner_agg['sqr_mean'].idxmax()]
            # For CV, we need std; but we might not have enough seeds for std.
            # We'll just use the SQR winner.

            print("\n" + "=" * 80)
            print("🏆 STAGE 1 WINNERS")
            print("=" * 80)
            print("\n🥇 By Efficiency (SQR / Runtime):")
            print(f"  {winner_eff['objective']} T{winner_eff['tuning_trials']} R{winner_eff['tuning_reads']} K{winner_eff['val_top_k']} V{winner_eff['val_reads']} N{winner_eff['N']}")
            print(f"  Eff: {winner_eff['efficiency_time']:.4f}, SQR: {winner_eff['sqr_mean']:.4f}, Runtime: {winner_eff['time_mean']:.2f}s")

            print("\n🥇 By Best SQR:")
            print(f"  {winner_sqr['objective']} T{winner_sqr['tuning_trials']} R{winner_sqr['tuning_reads']} K{winner_sqr['val_top_k']} V{winner_sqr['val_reads']} N{winner_sqr['N']}")
            print(f"  SQR: {winner_sqr['sqr_mean']:.4f}, Runtime: {winner_sqr['time_mean']:.2f}s")

            # Save winner (default: efficiency winner)
            winner_dict = {
                'objective': winner_eff['objective'],
                'tuning_trials': int(winner_eff['tuning_trials']),
                'tuning_reads': int(winner_eff['tuning_reads']),
                'val_top_k': int(winner_eff['val_top_k']),
                'val_reads': int(winner_eff['val_reads']),
                'N': int(winner_eff['N']),
                'sqr_mean': winner_eff['sqr_mean'],
                'time_mean': winner_eff['time_mean'],
                'efficiency': winner_eff['efficiency_time'],
            }
            with open(SAVE_DIR / "stage1_winner.json", "w") as f:
                json.dump(winner_dict, f, indent=2, cls=NumpyEncoder)

    else:
        winner_dict = None

    # ========================================================================
    # STAGE 2: Scaling and validation-read sensitivity
    # ========================================================================
    if RUN_STAGE2 and winner_dict is not None:
        print("\n" + "=" * 80)
        print("📌 STAGE 2: Scaling to larger N and VAL_READS sensitivity")
        print("=" * 80)

        # Fix winner params
        fixed_obj = winner_dict['objective']
        fixed_trials = winner_dict['tuning_trials']
        fixed_reads = winner_dict['tuning_reads']
        fixed_topk = winner_dict['val_top_k']
        # Use all seeds and sweep N and val_reads
        stage2_configs = list(itertools.product(
            SEEDS,
            [fixed_obj],
            [fixed_trials],
            [fixed_reads],
            [fixed_topk],
            SCALING_VAL_READS_LIST,
            SCALING_N_LIST
        ))
        # Remove any duplicates (e.g., if N=100 somehow still there)
        stage2_configs = list(set(stage2_configs))

        # Run Stage 2
        df_stage2 = run_stage(2, stage2_configs, "Stage 2", "Stage 2")

        # Aggregate Stage 2 results
        if not df_stage2.empty:
            df_valid2 = df_stage2[df_stage2['best_sqr'].notna() & df_stage2['best_sqr'] > 0]
            if not df_valid2.empty:
                stage2_agg = df_valid2.groupby(['N', 'val_reads']).agg({
                    'best_sqr': ['mean', 'std', 'count'],
                    'feas_rate': ['mean'],
                    'runtime': ['mean']
                }).reset_index()
                stage2_agg.columns = ['N', 'val_reads', 'sqr_mean', 'sqr_std', 'n_seeds',
                                      'feas_mean', 'time_mean']
                print("\n📊 STAGE 2 SUMMARY")
                print(stage2_agg.to_string(index=False, float_format="%.4f"))
                # Save
                stage2_agg.to_csv(SAVE_DIR / "stage2_summary.csv", index=False)

                # Scaling plot
                plt.figure(figsize=(10, 6))
                for vr in stage2_agg['val_reads'].unique():
                    sub = stage2_agg[stage2_agg['val_reads'] == vr]
                    plt.plot(sub['N'], sub['sqr_mean'], marker='o', label=f'val_reads={vr}')
                plt.xlabel('Number of Candidate Sites (N)')
                plt.ylabel('Mean Validation SQR')
                plt.title('Scaling Performance with N')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(FINAL_PLOTS_DIR / "scaling_plot.png", dpi=150)
                plt.show()
                plt.close()

    # ========================================================================
    # FINAL CLEANUP
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ STRATEGY COMPARISON COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {SAVE_DIR.resolve()}")
    if GDRIVE_ENABLED:
        print(f"📁 GDrive backup: {GDRIVE_SAVE_DIR.resolve()}")
    print("  - Stage 1 results: stage1_results.pkl, stage1_results.csv")
    print("  - Stage 1 winner: stage1_winner.json")
    if RUN_STAGE2 and winner_dict is not None:
        print("  - Stage 2 results: stage2_results.pkl, stage2_summary.csv")
    print("  - Plots: plots/ (progress), final_plots/ (publication-ready)")
    print("=" * 80)

    if wandb_run is not None:
        try:
            wandb.finish()
        except:
            pass

    cleanup_tqdm()
    gc.collect()
    print("🧹 Memory cleanup complete.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()