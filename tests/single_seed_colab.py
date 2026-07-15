#@title 🚀 MAIN PIPELINE v7 – Normalized QUBO with Winning Schedule & Objective
"""
================================================================================
MAIN PIPELINE v7 – Normalized QUBO with Winning Schedule & Objective
================================================================================

This script runs the full water quality monitoring pipeline using the
refactored codebase and the winning schedule/objective from the ablation studies.

Pipeline steps:
    1. Load/Create normalized QUBO environment (cached).
    2. Optuna tuning (using winning objective).
    3. Validation of top K trials.
    4. Sharpening: Re-run top champions with high reads, track convergence.
    5. Select best deployment solution.
    6. Generate final visualizations (deployment map, QUBO matrix, convergence profile).
    7. Print rich summaries to terminal for easy overnight interpretation.
    8. Save all results to GDrive and local.

TEST_MODE:
    - If True, uses reduced parameters (10 trials, 10 reads, etc.) for rapid testing.
    - Results are saved to results/main_pipeline_test/ to avoid overwriting full runs.

Inputs (set below):
    - TEST_MODE: Boolean – True for rapid testing, False for full production.
    - SEED_DATA: random seed for data
    - K_new: number of new stations to select
    - L_c: correlation length (km) – FIXED PHYSICAL PARAMETER
    - CONNECTIVITY_RANGE: communication range (km) – FIXED
    - WINNING_SCHEDULE: from schedule_comparison_v4 (e.g., 'new' or 'old')
    - WINNING_OBJECTIVE: from optuna_tests_v4 (e.g., 'Sq' or 'Pctl10')
================================================================================
"""

import sys
import os
import time
import json
import pickle
import gc
import warnings
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ensure the repository is in the path ---
sys.path.insert(0, "/content")
os.chdir("/content")

# Import refactored modules (Level 1 & 2)
from src.environment import get_environment
from src.experiment import run_optuna_study, make_objective, validate_study, evaluate_qubo
from src.plotting import (
    plot_validation_grid,
    plot_convergence_profile,
    plot_final_deployment,
    plot_qubo_matrix_heatmap,
    plot_optuna_learning_curve_enhanced,
    plot_deployment_with_qubo,
    plot_tuning_vs_validation_deployment,
    display_saved_plots,
)
from src.utils import (
    safe_save_pickle,
    safe_load_pickle,
    safe_load_json,
    NumpyEncoder,
    extract_top3_champions,
    print_loaded_seed_summary,
    print_tuning_summary,
    print_validation_summary,
    print_sharpening_summary,
    print_global_summary,
    execute_phase,
    PHASE_TIMES,
    cleanup_tqdm,
)

warnings.filterwarnings('ignore')

# ============================================================================
# 0. USER CONFIGURATION – SET THESE
# ============================================================================

# --- Mode ---
TEST_MODE = True  # If True: 10 trials, 10 reads, 15 val, 20 sharpen (rapid test)
CURRENT_MODE = "test" if TEST_MODE else "full"

# --- Data ---
SEED_DATA = 42
K_new = 5
L_c = 5.0                          # FIXED PHYSICAL PARAMETER (km)
CONNECTIVITY_RANGE = 8.0           # FIXED (km)
L_w = 1.0
CURRENT_VECTOR = (1.0, 0.0)
BETA = 1.0
DELTA = 1.0

# --- Results from ablation ---
WINNING_SCHEDULE = "old"           # 'new' or 'old' – from schedule_comparison_v4
WINNING_OBJECTIVE = "Sq"           # from optuna_tests_v4

# --- Flags ---
FORCE_RETUNE = True                # If True, delete existing Optuna DB and re-tune
FORCE_REVALIDATE = True            # If True, re-run validation even if results exist
FORCE_RESHARPEN = True             # If True, re-run sharpening even if results exist
USE_SEED_NONE = True               # OpenJij workaround (keep True)

# --- Mode-specific parameters ---
if TEST_MODE:
    print("\n⚠️  TEST_MODE ACTIVE – Using reduced parameters for rapid testing.")
    TUNING_TRIALS = 10
    TUNING_READS = 10
    VAL_READS = 15
    VAL_TOP_K = 3
    SHARPEN_TOP_K = 3
    SHARPEN_READS = 20
    SHARPEN_REPEATS = 1
else:
    TUNING_TRIALS = 200
    TUNING_READS = 128
    VAL_READS = 512
    VAL_TOP_K = 20
    SHARPEN_TOP_K = 3
    SHARPEN_READS = 2048
    SHARPEN_REPEATS = 3

TUNING_SEED = 42
VAL_SEED = 43

# --- Plots ---
PLOT_DPI = 150
SHOW_PLOTS = True                  # Display plots in Colab

# --- Storage (mode-specific) ---
LOCAL_SAVE_DIR = Path(f"results/main_pipeline_{CURRENT_MODE}")
LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# GDrive base path (if mounted)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    GDRIVE_BASE = "/content/drive/MyDrive/water_quality_results"
    GDRIVE_ENABLED = True
except ImportError:
    GDRIVE_BASE = None
    GDRIVE_ENABLED = False
    print("  ⚠️ Not running in Colab, GDrive disabled.")

# Drive subdirectory for this seed (mode-specific)
if GDRIVE_ENABLED:
    SEED_DRIVE_DIR = Path(GDRIVE_BASE) / f"seed_{SEED_DATA}" / f"main_pipeline_{CURRENT_MODE}"
    SEED_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
else:
    SEED_DRIVE_DIR = LOCAL_SAVE_DIR

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print("=" * 80)
print(f"🚀 MAIN PIPELINE v7 – Normalized QUBO ({CURRENT_MODE.upper()} MODE)")
print("=" * 80)
print(f"\n[Configuration]")
print(f"  Mode: {CURRENT_MODE.upper()}")
print(f"  Seed: {SEED_DATA}")
print(f"  K_new: {K_new}")
print(f"  L_c: {L_c} km (FIXED)")
print(f"  Connectivity range: {CONNECTIVITY_RANGE} km (FIXED)")
print(f"  Winning schedule: {WINNING_SCHEDULE}")
print(f"  Winning objective: {WINNING_OBJECTIVE}")
print(f"  Tuning trials: {TUNING_TRIALS}")
print(f"  Tuning reads: {TUNING_READS}")
print(f"  Validation reads: {VAL_READS}")
print(f"  Sharpening reads: {SHARPEN_READS}")
print(f"  Sharpening repeats: {SHARPEN_REPEATS}")
print(f"  USE_SEED_NONE: {USE_SEED_NONE}")
print(f"  Force retune: {FORCE_RETUNE}")
print(f"  GDrive enabled: {GDRIVE_ENABLED}")
print(f"  Results directory: {LOCAL_SAVE_DIR.resolve()}")
if GDRIVE_ENABLED:
    print(f"  GDrive directory: {SEED_DRIVE_DIR.resolve()}")

# ============================================================================
# PHASE 0: Load/Create Environment
# ============================================================================

with execute_phase("0. Environment Loading"):
    print("  Loading/Creating normalized QUBO environment...")
    env = get_environment(
        seed=SEED_DATA,
        K_new=K_new,
        L_c=L_c,
        L_w=L_w,
        beta=BETA,
        delta=DELTA,
        connectivity_range=CONNECTIVITY_RANGE,
        current_vector=CURRENT_VECTOR,
        force_recompute=FORCE_RETUNE,  # Rebuild if forcing retune
        verbose=True,
        gdrive_base=GDRIVE_BASE if GDRIVE_ENABLED else None,
    )

    print(f"\n  ✅ Environment loaded.")
    print(f"     L_c = {env['L_c']:.2f} km")
    print(f"     Q_sum = {env['Q_sum']:.4f}")
    print(f"     Gurobi baseline = {env['gurobi_miqp']:.8f}")
    print(f"     Cache hash: {env['hash']}")

# ============================================================================
# PHASE 1: Tuning (Optuna)
# ============================================================================

with execute_phase("1. Optuna Tuning"):
    print(f"  Running tuning with objective: {WINNING_OBJECTIVE}")
    print(f"  Schedule: {WINNING_SCHEDULE}")

    # Build objective function
    objective = make_objective(
        env=env,
        schedule_type=WINNING_SCHEDULE,
        objective_type=WINNING_OBJECTIVE,
        tuning_reads=TUNING_READS,
        tuning_seed=TUNING_SEED,
        use_seed_none=USE_SEED_NONE,
        compute_esr_mcr=True,
    )

    # Run study
    study = run_optuna_study(
        experiment_name=f"main_pipeline_seed{SEED_DATA}_{CURRENT_MODE}",
        objective=objective,
        n_trials=TUNING_TRIALS,
        storage_dir=LOCAL_SAVE_DIR / "study",
        directions=['minimize'] if WINNING_OBJECTIVE != 'Multi' else ['minimize', 'minimize'],
        sampler_type='TPE' if WINNING_OBJECTIVE != 'Multi' else 'NSGAII',
        seed=TUNING_SEED,
        load_if_exists=not FORCE_RETUNE,
        verbose=True,
    )

    print(f"  ✅ Tuning complete. Trials: {len(study.trials)}")

    # --- Print tuning summary ---
    print_tuning_summary(study, "Main Pipeline")

    # --- Generate enhanced learning curve ---
    plot_dir = LOCAL_SAVE_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_optuna_learning_curve_enhanced(
        study,
        plot_dir / "learning_curve_enhanced.png",
        show_fig=SHOW_PLOTS,
    )

    # Backup study to GDrive (skip in test mode)
    if GDRIVE_ENABLED and not TEST_MODE:
        study_db = LOCAL_SAVE_DIR / "study" / f"main_pipeline_seed{SEED_DATA}_{CURRENT_MODE}.db"
        if study_db.exists():
            drive_study_dir = SEED_DRIVE_DIR / "study"
            drive_study_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(study_db, drive_study_dir / study_db.name)
            print(f"  ✅ Study DB backed up to {drive_study_dir}")

# ============================================================================
# PHASE 2: Validation
# ============================================================================

with execute_phase("2. Validation"):
    print(f"  Validating top {VAL_TOP_K} trials with {VAL_READS} reads...")

    # Check if validation results already exist and we are not forcing revalidation
    val_path = LOCAL_SAVE_DIR / "validation_results.pkl"
    if not FORCE_REVALIDATE and val_path.exists():
        validation_results = safe_load_pickle(val_path)
        print(f"  ✅ Loaded existing validation results ({len(validation_results.get('trials', []))} trials)")
    else:
        validation_results = validate_study(
            study=study,
            env=env,
            schedule_type=WINNING_SCHEDULE,
            val_reads=VAL_READS,
            val_seed=VAL_SEED,
            top_k=VAL_TOP_K,
            use_seed_none=USE_SEED_NONE,
            compute_esr_mcr=True,
            verbose=True,
        )
        safe_save_pickle(val_path, validation_results)
        print(f"  ✅ Validation complete. Best SQR = {validation_results['best_sqr']:.4f}")

    # --- Print validation summary ---
    print_validation_summary(validation_results, "Main Pipeline")

    # Extract champion and top K unique champions
    champion = None
    if validation_results.get('trials'):
        # The first trial in the list (after sorting) is the best
        # But we need to find the trial with highest SQR among those with feas_rate > 0
        valid_trials = [t for t in validation_results['trials'] if t.get('feas_rate', 0) > 0 and np.isfinite(t.get('best_sqr', np.nan))]
        if valid_trials:
            best_trial = max(valid_trials, key=lambda x: x['best_sqr'])
            champion = {
                'trial': best_trial.get('trial_number', -1),
                'lam1': best_trial.get('lam1', np.nan),
                'lam2': best_trial.get('lam2', np.nan),
                'num_sweeps': best_trial.get('num_sweeps', np.nan),
                'best_sqr': best_trial.get('best_sqr', np.nan),
                'feas_rate': best_trial.get('feas_rate', 0.0),
                'ESR': best_trial.get('ESR', np.nan),
                'MCR': best_trial.get('MCR', np.nan),
            }
        else:
            # Fallback: use the best overall from summary
            champion = {
                'trial': -1,
                'lam1': np.nan,
                'lam2': np.nan,
                'num_sweeps': np.nan,
                'best_sqr': validation_results.get('best_sqr', np.nan),
                'feas_rate': validation_results.get('feas_rate', 0.0),
                'ESR': np.nan,
                'MCR': np.nan,
            }

    # Extract top K unique champions
    top3_champions = extract_top3_champions(
        validation_results=validation_results.get('trials', []),
        champion=champion,
        sharpen_top_k=SHARPEN_TOP_K,
    )

    print(f"\n  🏆 Champion: Trial {champion['trial']} | SQR: {champion['best_sqr']:.4f}")
    print(f"     Top {SHARPEN_TOP_K} champions: {[t.get('trial', '?') for t in top3_champions[:SHARPEN_TOP_K]]}")

# ============================================================================
# PHASE 3: Sharpening
# ============================================================================

with execute_phase("3. Sharpening"):
    print(f"  Sharpening top {SHARPEN_TOP_K} champions with {SHARPEN_READS} reads, {SHARPEN_REPEATS} repeats...")

    sharpen_path = LOCAL_SAVE_DIR / "sharpening_results.pkl"
    if not FORCE_RESHARPEN and sharpen_path.exists():
        sharpen_data = safe_load_pickle(sharpen_path)
        sharpening_results = sharpen_data.get('results', [])
        convergence_data = sharpen_data.get('convergence_data', {})
        print(f"  ✅ Loaded existing sharpening results ({len(sharpening_results)} runs)")
    else:
        sharpening_results = []
        convergence_data = {}

        # For each champion, run multiple repeats
        for champ_idx, champ in enumerate(top3_champions[:SHARPEN_TOP_K]):
            trial_num = champ.get('trial', -1)
            if trial_num == -1:
                print(f"  ⚠️ Skipping fallback champion (no valid trial).")
                continue

            # Get parameters from validation result (or fallback to champion dict)
            lam1 = champ.get('lam1', np.nan)
            lam2 = champ.get('lam2', np.nan)
            num_sweeps = champ.get('num_sweeps', 15000)
            if np.isnan(lam1) or np.isnan(lam2):
                print(f"  ⚠️ Champion {trial_num} has invalid lam1/lam2; skipping.")
                continue

            print(f"\n  [Sharpening Trial {trial_num}] (lam1={lam1:.4f}, lam2={lam2:.4f})")

            for run_idx in range(SHARPEN_REPEATS):
                seed_run = np.random.randint(0, 2**31)  # different seed each run
                print(f"    Run {run_idx+1}/{SHARPEN_REPEATS} (seed={seed_run})...")

                try:
                    result = evaluate_qubo(
                        env=env,
                        lam1=lam1,
                        lam2=lam2,
                        num_sweeps=num_sweeps,
                        schedule_type=WINNING_SCHEDULE,
                        num_reads=SHARPEN_READS,
                        seed=seed_run,
                        return_all=True,
                        use_seed_none=USE_SEED_NONE,
                        compute_esr_mcr=True,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"      ❌ Run failed: {e}")
                    continue

                # Extract data
                best_miqp = result.get('best_miqp', float('inf'))
                best_sqr = result.get('best_sqr', 0.0)
                best_solution = result.get('best_solution', None)
                feas_rate = result.get('feas_rate', 0.0)
                all_samples = result.get('all_samples', [])

                # Build convergence data
                cumulative_best = []
                best_energy_so_far = float('inf')
                for sample in all_samples:
                    if sample.get('violations', {}).get('feasible', False):
                        miqp = sample.get('miqp_energy')
                        if miqp is not None and np.isfinite(miqp) and miqp < best_energy_so_far:
                            best_energy_so_far = miqp
                    cumulative_best.append(best_energy_so_far)

                # Store convergence data
                run_key = f"trial_{trial_num}_run_{run_idx+1}"
                convergence_data[run_key] = {
                    'cumulative_best': cumulative_best,
                    'seed': seed_run,
                    'reads': len(all_samples),
                    'trial': trial_num,
                    'run': run_idx + 1,
                    'best_miqp': best_energy_so_far,
                    'best_sqr': best_energy_so_far / env['gurobi_miqp'] if best_energy_so_far != float('inf') else 0.0,
                    'feasible_count': sum(1 for s in all_samples if s.get('violations', {}).get('feasible', False)),
                }

                # Store result
                sharpening_results.append({
                    'trial': trial_num,
                    'lam1': lam1,
                    'lam2': lam2,
                    'num_sweeps': num_sweeps,
                    'run': run_idx + 1,
                    'seed': seed_run,
                    'best_miqp': best_miqp,
                    'best_sqr': best_sqr,
                    'feas_rate': feas_rate,
                    'solution': best_solution,
                    'ESR': result.get('ESR', np.nan),
                    'MCR': result.get('MCR', np.nan),
                    'convergence_key': run_key,
                    'time': 0.0,  # placeholders (we don't have time in evaluate_qubo)
                })

                print(f"      Best SQR: {best_sqr:.6f} | Feas: {feas_rate*100:.1f}%")

        # Save sharpening results
        safe_save_pickle(sharpen_path, {
            'results': sharpening_results,
            'convergence_data': convergence_data,
        })
        print(f"\n  ✅ Sharpening complete. {len(sharpening_results)} runs saved.")

    # Find best sharpening result
    if sharpening_results:
        best_sharpen = max(sharpening_results, key=lambda x: x['best_sqr'])
        print(f"\n  🏆 Best sharpening: Trial {best_sharpen['trial']} | Run {best_sharpen['run']}")
        print(f"     Best SQR: {best_sharpen['best_sqr']:.6f} | Feas: {best_sharpen['feas_rate']*100:.1f}%")
    else:
        print("  ⚠️ No sharpening results. Using champion as fallback.")
        best_sharpen = {
            'trial': champion.get('trial', -1),
            'lam1': champion.get('lam1', np.nan),
            'lam2': champion.get('lam2', np.nan),
            'num_sweeps': champion.get('num_sweeps', 15000),
            'run': 0,
            'seed': 'fallback',
            'best_miqp': champion.get('best_miqp', env['gurobi_miqp']),
            'best_sqr': champion.get('best_sqr', 0.0),
            'feas_rate': champion.get('feas_rate', 0.0),
            'solution': None,
            'ESR': champion.get('ESR', np.nan),
            'MCR': champion.get('MCR', np.nan),
            'convergence_key': None,
        }

    # --- Print sharpening summary ---
    print_sharpening_summary(best_sharpen, "Main Pipeline")

# ============================================================================
# PHASE 4: Final Visualization
# ============================================================================

with execute_phase("4. Final Visualization"):
    print("  Generating final plots...")

    # Plot directory
    plot_dir = LOCAL_SAVE_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation grid (Gurobi + Top 3 champions)
    print("    Generating validation grid...")
    # Reconstruct solutions for top 3 champions from validation results
    top3_solutions = []
    top3_trials_info = []
    for champ in top3_champions[:SHARPEN_TOP_K]:
        trial_num = champ.get('trial', -1)
        if trial_num == -1:
            # Use fallback: re-run evaluation with champion params
            print(f"      Re-solving fallback champion...")
            sol = np.zeros(len(env['coords']), dtype=int)
            for m in env['M_indices']:
                sol[m] = 1
            if best_sharpen.get('solution') is not None:
                sol = best_sharpen['solution']
            top3_solutions.append(sol)
            top3_trials_info.append({'trial': trial_num, 'best_sqr': champ.get('best_sqr', 0.0)})
        else:
            # Find in validation results
            found = False
            for vt in validation_results.get('trials', []):
                if vt.get('trial_number') == trial_num:
                    sol = vt.get('solution')
                    if sol is not None:
                        top3_solutions.append(sol)
                        top3_trials_info.append({'trial': trial_num, 'best_sqr': vt.get('best_sqr', 0.0)})
                        found = True
                        break
            if not found:
                # Reconstruct solution from selected indices (if available)
                # Fallback: use zeros (will be plotted as empty)
                print(f"      No solution for trial {trial_num}; using empty.")
                sol = np.zeros(len(env['coords']), dtype=int)
                for m in env['M_indices']:
                    sol[m] = 1
                top3_solutions.append(sol)
                top3_trials_info.append({'trial': trial_num, 'best_sqr': champ.get('best_sqr', 0.0)})

    plot_validation_grid(
        coords=env['coords'],
        U=env['U'],
        gurobi_solution=env['gurobi_solution'],
        top3_solutions=top3_solutions,
        top3_trials=top3_trials_info,
        M_indices=env['M_indices'],
        DOMAIN_SIZE=50.0,
        save_path=plot_dir / "validation_grid.png",
        dpi=PLOT_DPI,
        show_fig=SHOW_PLOTS,
    )

    # 2. Convergence profile (best sharpening run)
    print("    Generating convergence profile...")
    if best_sharpen.get('convergence_key') and best_sharpen['convergence_key'] in convergence_data:
        plot_convergence_profile(
            best_sharpen=best_sharpen,
            convergence_data=convergence_data,
            GUROBI_MIQP=env['gurobi_miqp'],
            save_path=plot_dir / "convergence.png",
            dpi=PLOT_DPI,
            show_fig=SHOW_PLOTS,
        )
    else:
        print("      Skipping convergence profile (no data).")

    # 3. Final deployment map (legacy, kept for compatibility)
    print("    Generating final deployment map (legacy)...")
    if best_sharpen.get('solution') is not None:
        solution = best_sharpen['solution']
        selected_new = [i for i in range(len(solution)) if solution[i] == 1 and i not in env['M_indices']]
        sharpening_means = {
            'best_sqr': np.mean([r['best_sqr'] for r in sharpening_results if r['best_sqr'] > 0]) if sharpening_results else 0.0,
            'feas_rate': np.mean([r['feas_rate'] for r in sharpening_results]) if sharpening_results else 0.0,
        }
        plot_final_deployment(
            coords=env['coords'],
            U=env['U'],
            best_solution=solution,
            M_indices=env['M_indices'],
            selected_new=selected_new,
            DOMAIN_SIZE=50.0,
            current_vector=CURRENT_VECTOR,
            CONNECTIVITY_RANGE=CONNECTIVITY_RANGE,
            champion=champion,
            best_sharpen=best_sharpen,
            sharpening_means=sharpening_means,
            save_path=plot_dir / "deployment.png",
            dpi=PLOT_DPI,
            show_fig=SHOW_PLOTS,
        )
    else:
        print("      Skipping deployment map (no solution).")

    # 4. NEW: Deployment + QUBO side-by-side for final champion
    print("    Generating deployment + QUBO side-by-side...")
    if best_sharpen.get('solution') is not None:
        solution = best_sharpen['solution']
        selected_new = [i for i in range(len(solution)) if solution[i] == 1 and i not in env['M_indices']]
        lam1 = best_sharpen.get('lam1', np.nan)
        lam2 = best_sharpen.get('lam2', np.nan)
        if not np.isnan(lam1) and not np.isnan(lam2):
            title = f"Sharpening Champion: Trial #{best_sharpen['trial']} | Run {best_sharpen['run']} | SQR={best_sharpen['best_sqr']:.4f}"
            plot_deployment_with_qubo(
                env=env,
                lam1=lam1,
                lam2=lam2,
                solution=solution,
                M_indices=env['M_indices'],
                selected_new=selected_new,
                U=env['U'],
                coords=env['coords'],
                DOMAIN_SIZE=50.0,
                current_vector=CURRENT_VECTOR,
                CONNECTIVITY_RANGE=CONNECTIVITY_RANGE,
                title=title,
                save_path=plot_dir / "deployment_with_qubo.png",
                dpi=PLOT_DPI,
                show_fig=SHOW_PLOTS,
            )
        else:
            print("      Skipping deployment+QUBO (invalid lam1/lam2).")
    else:
        print("      Skipping deployment+QUBO (no solution).")

    # 5. NEW: Tuning vs Validation deployment comparison (2x2)
    print("    Generating tuning vs validation deployment comparison...")
    # Get best tuning trial solution from study
    best_tuning_trial = study.best_trial
    tuning_solution = best_tuning_trial.user_attrs.get('best_solution')
    if tuning_solution is not None:
        tuning_solution = np.array(tuning_solution)
    tuning_trial_info = {
        'trial_number': best_tuning_trial.number,
        'lam1': best_tuning_trial.params.get('lam1'),
        'lam2': best_tuning_trial.params.get('lam2'),
        'best_sqr': best_tuning_trial.user_attrs.get('best_sqr', np.nan),
        'solution': tuning_solution,
    }

    # Get best validation trial
    if validation_results.get('trials'):
        best_val_trial = validation_results['trials'][0]
        val_trial_info = {
            'trial_number': best_val_trial.get('trial_number'),
            'lam1': best_val_trial.get('lam1'),
            'lam2': best_val_trial.get('lam2'),
            'best_sqr': best_val_trial.get('best_sqr'),
            'solution': best_val_trial.get('solution'),
        }
    else:
        val_trial_info = None

    if tuning_solution is not None and val_trial_info is not None and val_trial_info['solution'] is not None:
        plot_tuning_vs_validation_deployment(
            env=env,
            tuning_trial=tuning_trial_info,
            val_trial=val_trial_info,
            coords=env['coords'],
            U=env['U'],
            M_indices=env['M_indices'],
            DOMAIN_SIZE=50.0,
            current_vector=CURRENT_VECTOR,
            CONNECTIVITY_RANGE=CONNECTIVITY_RANGE,
            experiment_name="Main Pipeline",
            save_path=plot_dir / "tuning_vs_validation_deployment.png",
            dpi=PLOT_DPI,
            show_fig=SHOW_PLOTS,
        )
    else:
        print("      Skipping tuning vs validation deployment (missing solutions).")

    # 6. QUBO matrix heatmap for best sharpening hyperparameters (legacy)
    print("    Generating QUBO matrix heatmap (legacy)...")
    if not np.isnan(best_sharpen.get('lam1', np.nan)) and not np.isnan(best_sharpen.get('lam2', np.nan)):
        plot_qubo_matrix_heatmap(
            env=env,
            lam1=best_sharpen['lam1'],
            lam2=best_sharpen['lam2'],
            K_new=K_new,
            save_path=plot_dir / f"qubo_heatmap_lam1_{best_sharpen['lam1']:.4f}_lam2_{best_sharpen['lam2']:.4f}.png",
            show_fig=SHOW_PLOTS,
        )
    else:
        print("      Skipping QUBO heatmap (invalid lam1/lam2).")

    # Display all plots inline
    display_saved_plots(plot_dir)

    # Copy plots to GDrive (skip in test mode)
    if GDRIVE_ENABLED and not TEST_MODE:
        drive_plot_dir = SEED_DRIVE_DIR / "plots"
        drive_plot_dir.mkdir(parents=True, exist_ok=True)
        for png in plot_dir.glob("*.png"):
            shutil.copy2(png, drive_plot_dir / png.name)
        print(f"  ✅ Plots copied to {drive_plot_dir}")

# ============================================================================
# PHASE 5: Save Final Results
# ============================================================================

with execute_phase("5. Save Results"):
    print("  Saving final results...")

    # Compute summary statistics
    total_time = sum(PHASE_TIMES.values())

    # Package results
    results = {
        'champion': champion,
        'top3_champions': top3_champions[:SHARPEN_TOP_K],
        'best_sharpen': best_sharpen,
        'sharpening_results': sharpening_results,
        'sharpening_convergence_data': convergence_data,
        'validation_results': validation_results,
        'gurobi_baseline': env['gurobi_miqp'],
        'phase_times': PHASE_TIMES,
        'total_time': total_time,
        'seed': SEED_DATA,
        'mode': CURRENT_MODE,
        'config': {
            'schedule': WINNING_SCHEDULE,
            'objective': WINNING_OBJECTIVE,
            'L_c': L_c,
            'connectivity_range': CONNECTIVITY_RANGE,
            'K_new': K_new,
            'tuning_trials': TUNING_TRIALS,
            'tuning_reads': TUNING_READS,
            'val_reads': VAL_READS,
            'sharpen_reads': SHARPEN_READS,
            'sharpen_repeats': SHARPEN_REPEATS,
            'use_seed_none': USE_SEED_NONE,
            'test_mode': TEST_MODE,
        },
        'env_cache_hash': env['hash'],
        'version': 'v7',
    }

    # Save locally
    local_results_path = LOCAL_SAVE_DIR / "results.json"
    with open(local_results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  ✅ Local results saved to {local_results_path}")

    # Also save as pickle (for full object storage)
    safe_save_pickle(LOCAL_SAVE_DIR / "results.pkl", results)

    # Save to GDrive (skip in test mode)
    if GDRIVE_ENABLED and not TEST_MODE:
        drive_results_path = SEED_DRIVE_DIR / "results.json"
        with open(drive_results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        safe_save_pickle(SEED_DRIVE_DIR / "results.pkl", results)
        print(f"  ✅ GDrive results saved to {drive_results_path}")

# ============================================================================
# PHASE 6: Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("📊 FINAL SUMMARY")
print("=" * 80)

# --- Print global summary ---
print_global_summary(results, "Main Pipeline")

# --- Also print the legacy loaded seed summary ---
print_loaded_seed_summary(results, SEED_DATA, "main_v7")

print("\n" + "=" * 80)
print(f"✅ MAIN PIPELINE v7 COMPLETE! ({CURRENT_MODE.upper()} MODE)")
print("=" * 80)
print(f"📁 Results saved to: {LOCAL_SAVE_DIR.resolve()}")
if GDRIVE_ENABLED and not TEST_MODE:
    print(f"📁 GDrive backup: {SEED_DRIVE_DIR.resolve()}")
print(f"🎯 Deployment SQR: {best_sharpen['best_sqr']:.4f} ({100*best_sharpen['best_sqr']:.1f}% of optimal)")
if not np.isnan(best_sharpen.get('lam1', np.nan)):
    print(f"🔧 Winning λ₁: {best_sharpen['lam1']:.4f}, λ₂: {best_sharpen['lam2']:.4f}")
print("=" * 80)

# Clean up
cleanup_tqdm()
gc.collect()
print("🧹 Memory cleanup complete.")