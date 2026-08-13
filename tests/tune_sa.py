#@title 🔬 SA TUNE (Real Data - Multi-Tier, Decoupled Violations)
"""
================================================================================
SA TUNING WITH OPTUNA – REAL DATA (MULTI-TIER)
================================================================================
- Decoupled constraints: budget_dev and isolated_count are passed separately.
- Objectives: minimize elite_avg_energy, maximize feasibility_rate.
- Multi‑tier: tunes each instance_data_N{}.pkl independently.
- W&B table logging and backup included.
================================================================================
"""

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_SA = {
    # ---- Which tier(s) to tune ----
    "TARGET_N": 35,              # None = tune all; or int (e.g., 35)

    # ---- SA parameters ----
    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    "PATIENCE": 20,                # Stop after N trials without Pareto improvement
    "WARMUP_TRIALS": 10,
    "USE_BEST_SA": False,          # False: best feasible post‑select; True: response.first
    "MIN_FEASIBILITY_FLOOR": 0.15, # Used in objective for viability tracking, not as constraint

    # ---- Lambda bounds ----
    "LAMBDA_LOWER": 0.0001,
    # LAMBDA_UPPER auto‑computed as qsum per tier

    # ---- W&B ----
    "USE_WANDB": True,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,       # Log mutable table with trial data

    # ---- General ----
    "RUN_ID": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "FORCE_RETUNE": True,
    "BACKUP_FREQ": 10,
    "SAVE_PLOTS": True,
    "SHOW_PLOTS": True,
    "PLOT_DPI": 150,
}

# Extract config
TARGET_N = CONFIG_SA["TARGET_N"]
NUM_SWEEPS = CONFIG_SA["NUM_SWEEPS"]
NUM_READS = CONFIG_SA["NUM_READS"]
PATIENCE = CONFIG_SA["PATIENCE"]
WARMUP_TRIALS = CONFIG_SA["WARMUP_TRIALS"]
USE_BEST_SA = CONFIG_SA["USE_BEST_SA"]
MIN_FEASIBILITY_FLOOR = CONFIG_SA["MIN_FEASIBILITY_FLOOR"]
LAMBDA_LOWER = CONFIG_SA["LAMBDA_LOWER"]
USE_WANDB = CONFIG_SA["USE_WANDB"]
WANDB_PROJECT = CONFIG_SA["WANDB_PROJECT"]
LOG_WANDB_TABLE = CONFIG_SA["LOG_WANDB_TABLE"]
RUN_ID = CONFIG_SA["RUN_ID"]
FORCE_RETUNE = CONFIG_SA["FORCE_RETUNE"]
BACKUP_FREQ = CONFIG_SA["BACKUP_FREQ"]
SAVE_PLOTS = CONFIG_SA["SAVE_PLOTS"]
SHOW_PLOTS = CONFIG_SA["SHOW_PLOTS"]
PLOT_DPI = CONFIG_SA["PLOT_DPI"]

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import sys
import os
import json
import time
import pickle
import re
import shutil
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import optuna
import wandb

# Repo utilities
try:
    from src.jij_solvers import solve_sa_jij, compute_energy, solve_greedy_jij
    from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
    from src.utils import safe_save_pickle, safe_load_pickle, NumpyEncoder
except ImportError:
    print("⚠️ Ensure src modules are in your sys.path before running.")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
GDRIVE_BASE.mkdir(parents=True, exist_ok=True)

# Run-specific folder
RUN_DIR = GDRIVE_BASE / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def compute_qsum(instance_data):
    """Compute qsum = sum(|a_i|) + sum(|Q_ij|) for i<j."""
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    N = len(a)
    qsum = np.sum(np.abs(a))
    for i in range(N):
        for j in range(i + 1, N):
            qsum += np.abs(Q[i, j])
    return max(qsum, 1.0)

def compute_decoupled_violations(x, neigh, K, fixed_neighbors=None):
    """Return budget_dev and isolated_count separately."""
    num_selected = np.sum(x)
    budget_dev = float(abs(num_selected - K))

    isolated_count = 0
    selected = np.where(x == 1)[0]
    for i in selected:
        has_free_neighbor = np.sum(neigh[i] * x) > 0
        has_fixed_neighbor = False
        if not has_free_neighbor and fixed_neighbors is not None:
            if len(fixed_neighbors.get(i, [])) > 0:
                has_fixed_neighbor = True
        if not (has_free_neighbor or has_fixed_neighbor):
            isolated_count += 1

    return budget_dev, float(isolated_count)

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
class ParetoFrontPatienceCallback:
    def __init__(self, patience: int = 50, warmup_trials: int = 20):
        self.patience = patience
        self.warmup_trials = warmup_trials
        self.current_pareto_trials = set()
        self.trials_without_improvement = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        latest_pareto_trials = {t.number for t in study.best_trials}

        if trial.number < self.warmup_trials:
            self.current_pareto_trials = latest_pareto_trials
            return

        if not latest_pareto_trials:
            return

        if latest_pareto_trials != self.current_pareto_trials:
            self.current_pareto_trials = latest_pareto_trials
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        if self.trials_without_improvement >= self.patience:
            print(f"\n🛑 [Early Stopping] No Pareto front improvement for {self.patience} trials. Halting.")
            study.stop()

class DriveBackupCallback:
    def __init__(self, local_path: Path, drive_path: Path, backup_freq: int = 10):
        self.local_path = local_path
        self.drive_path = drive_path
        self.backup_freq = backup_freq

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.number > 0 and trial.number % self.backup_freq == 0:
            try:
                self.drive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.local_path, self.drive_path)
                print(f"  [Backup] Synced study database to Drive (Trial {trial.number}).")
            except Exception as e:
                print(f"  [Backup] WARNING: Could not backup to Drive: {e}")

# -----------------------------------------------------------------------------
# OBJECTIVE FACTORY
# -----------------------------------------------------------------------------
def make_objective(instance_data, neigh, K, fixed_neighbors,
                   NUM_READS, NUM_SWEEPS, LAMBDA_LOWER, LAMBDA_UPPER,
                   USE_BEST_SA, MIN_FEASIBILITY_FLOOR, wandb_run=None, wandb_table=None):
    """Return an objective function that closes over the tier's data."""

    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    def objective(trial):
        lambda_budget = trial.suggest_float("lambda_budget", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", LAMBDA_LOWER, LAMBDA_UPPER, log=True)

        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)

        result = solve_sa_jij(
            instance_data,
            penalty_weights,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            return_all=True,
            verbose=False,
        )

        all_samples = result.get("all_samples", [])
        if not all_samples:
            trial.set_user_attr("budget_dev", 1e9)
            trial.set_user_attr("isolated_count", 1e9)
            trial.set_user_attr("feasibility_rate", 0.0)
            return 1e9, 0.0  # worst energy, zero feasibility

        # 1. Compute decoupled violations for all samples
        for s in all_samples:
            if "budget_dev" not in s or "isolated_count" not in s:
                b_dev, i_cnt = compute_decoupled_violations(
                    s["solution"], neigh, K, fixed_neighbors
                )
                s["budget_dev"] = b_dev
                s["isolated_count"] = i_cnt
                s["total_violation"] = b_dev + i_cnt

        # 2. Separate feasible samples (both budget_dev == 0 and isolated_count == 0)
        feasible_samples = [s for s in all_samples if s["budget_dev"] == 0 and s["isolated_count"] == 0]
        num_feasible = len(feasible_samples)
        feasibility_rate = float(num_feasible / len(all_samples))

        # 3. Compute elite_avg_energy (top 10% of feasible samples)
        if num_feasible > 0:
            feasible_samples.sort(key=lambda x: x["energy"])
            trial_best_feasible_energy = float(feasible_samples[0]["energy"])
            target_k = max(1, int(len(all_samples) * 0.10))
            actual_k = min(num_feasible, target_k)
            elite_cohort = feasible_samples[:actual_k]
            elite_avg_energy = float(np.mean([s["energy"] for s in elite_cohort]))
            best_sample_for_logging = feasible_samples[0]
            best_budget_dev = 0.0
            best_isolated_count = 0.0
        else:
            # No feasible samples: use least‑violated sample for energy, but keep feasibility 0
            trial_best_feasible_energy = np.nan
            all_samples.sort(key=lambda s: (s["total_violation"], s["energy"]))
            best_sample_for_logging = all_samples[0]
            elite_avg_energy = float(best_sample_for_logging["energy"])
            best_budget_dev = float(best_sample_for_logging["budget_dev"])
            best_isolated_count = float(best_sample_for_logging["isolated_count"])

        # 4. Global best trackers (over all completed trials)
        completed_trials = trial.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        # Track best feasible energy (only if feasible)
        past_single_best = [
            t.user_attrs["trial_best_feasible_energy"]
            for t in completed_trials
            if "trial_best_feasible_energy" in t.user_attrs and not np.isnan(t.user_attrs["trial_best_feasible_energy"])
        ]
        if not np.isnan(trial_best_feasible_energy):
            past_single_best.append(trial_best_feasible_energy)
        global_best_feasible_energy = float(np.min(past_single_best)) if past_single_best else np.nan

        # Track best elite_avg_energy (only for viable trials with feasibility ≥ floor)
        past_elite = [
            t.user_attrs["elite_avg_energy"]
            for t in completed_trials
            if t.user_attrs.get("feasibility_rate", 0.0) >= MIN_FEASIBILITY_FLOOR
        ]
        if feasibility_rate >= MIN_FEASIBILITY_FLOOR:
            past_elite.append(elite_avg_energy)
        global_best_elite = float(np.min(past_elite)) if past_elite else np.nan

        # Store global records in study
        if not np.isnan(global_best_feasible_energy):
            trial.study.set_user_attr("global_best_feasible_energy", global_best_feasible_energy)
        if not np.isnan(global_best_elite):
            trial.study.set_user_attr("global_best_elite_avg", global_best_elite)

        # 5. Store user attributes
        trial.set_user_attr("trial_best_feasible_energy", trial_best_feasible_energy)
        trial.set_user_attr("global_best_feasible_energy", global_best_feasible_energy)
        trial.set_user_attr("global_best_elite_avg", global_best_elite)
        trial.set_user_attr("budget_dev", best_budget_dev)
        trial.set_user_attr("isolated_count", best_isolated_count)
        trial.set_user_attr("feasibility_rate", feasibility_rate)
        trial.set_user_attr("elite_avg_energy", elite_avg_energy)
        trial.set_user_attr("lambda_budget", lambda_budget)
        trial.set_user_attr("lambda_conn", lambda_conn)
        trial.set_user_attr("num_selected", int(np.sum(best_sample_for_logging["solution"])))
        trial.set_user_attr("best_solution", best_sample_for_logging["solution"].tolist())
        trial.set_user_attr("runtime", result["runtime"])

        # 6. W&B logging
        if wandb_run is not None:
            log_dict = {
                "trial_number": trial.number,
                "elite_avg_energy": elite_avg_energy,
                "feasibility_rate": feasibility_rate,
                "budget_dev": best_budget_dev,
                "isolated_count": best_isolated_count,
                "lambda_budget": lambda_budget,
                "lambda_conn": lambda_conn,
                "runtime": result["runtime"],
                "num_selected": int(np.sum(best_sample_for_logging["solution"])),
                "best_solution": best_sample_for_logging["solution"].tolist(),
            }
            if not np.isnan(global_best_feasible_energy):
                log_dict["global_best_feasible_energy"] = global_best_feasible_energy
            if not np.isnan(global_best_elite):
                log_dict["global_best_elite_avg"] = global_best_elite
            if not np.isnan(trial_best_feasible_energy):
                log_dict["trial_best_feasible_energy"] = trial_best_feasible_energy

            if wandb_table is not None:
                wandb_table.add_data(
                    trial.number,
                    elite_avg_energy,
                    feasibility_rate,
                    best_budget_dev,
                    best_isolated_count,
                    lambda_budget,
                    lambda_conn,
                    int(np.sum(best_sample_for_logging["solution"])),
                    result["runtime"],
                    global_best_feasible_energy if not np.isnan(global_best_feasible_energy) else None,
                    global_best_elite if not np.isnan(global_best_elite) else None,
                    trial_best_feasible_energy if not np.isnan(trial_best_feasible_energy) else None,
                )
                log_dict["live_trials_table"] = wandb_table

            wandb_run.log(log_dict)

        # Cleanup
        del result, all_samples, feasible_samples, completed_trials, past_single_best, past_elite

        # Return: (minimize elite_avg_energy, maximize feasibility_rate)
        return elite_avg_energy, feasibility_rate

    return objective

# -----------------------------------------------------------------------------
# CONSTRAINT FUNCTION (Decoupled)
# -----------------------------------------------------------------------------
def constraint_func(trial):
    budget_dev = trial.user_attrs.get("budget_dev", 1e9)
    isolated_count = trial.user_attrs.get("isolated_count", 1e9)
    # Both must be ≤ 0 → i.e., exactly 0
    return [budget_dev, isolated_count]

# -----------------------------------------------------------------------------
# DISCOVER INSTANCE FILES
# -----------------------------------------------------------------------------
pattern = re.compile(r"instance_data_N(\d+)\.pkl")

def find_instance_files():
    local_files = {p.name: p for p in LOCAL_CACHE.glob("instance_data_N*.pkl") if pattern.match(p.name)}
    drive_files = {p.name: p for p in GDRIVE_BASE.glob("instance_data_N*.pkl") if pattern.match(p.name)}
    all_files = {}
    for name, path in drive_files.items():
        all_files[name] = local_files.get(name, path)
    for name, path in local_files.items():
        if name not in all_files:
            all_files[name] = path
    return sorted(all_files.values(), key=lambda p: int(pattern.search(p.name).group(1)))

instance_files = find_instance_files()

if not instance_files:
    raise FileNotFoundError("No instance_data_N*.pkl files found in local or Drive.")

if TARGET_N is not None:
    instance_files = [p for p in instance_files if int(pattern.search(p.name).group(1)) == TARGET_N]
    if not instance_files:
        raise ValueError(f"No instance data found for N={TARGET_N}")

print(f"Found {len(instance_files)} instance data files: {[p.name for p in instance_files]}")

# -----------------------------------------------------------------------------
# TUNE EACH TIER
# -----------------------------------------------------------------------------
all_tuning_results = []

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print("\n" + "=" * 80)
    print(f"🔬 Tuning SA for tier N={N_true}")
    print("=" * 80)

    # Load instance
    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)

    N_free = instance_data["N"]
    K = instance_data["K"]
    dmax = instance_data.get("D_max", 0.0)
    fixed_count = len(instance_data["fixed_indices"])
    neigh = instance_data["neigh"]
    fixed_neighbors = instance_data.get("fixed_neighbors", None)

    # Compute qsum for this tier
    qsum = compute_qsum(instance_data)
    LAMBDA_UPPER = qsum
    print(f"  qsum = {qsum:.4f}  →  Lambda upper bound = {LAMBDA_UPPER:.4f}")

    # Study name and paths
    study_name = f"sa_tuning_N{N_true}_{RUN_ID}"
    local_db_path = LOCAL_CACHE / f"{study_name}.db"
    drive_db_path = RUN_DIR / f"{study_name}.db"

    # Delete existing study if FORCE_RETUNE
    if FORCE_RETUNE:
        try:
            optuna.delete_study(study_name=study_name, storage=f"sqlite:///{local_db_path}")
            print(f"  Deleted existing study '{study_name}'")
        except KeyError:
            pass

    # W&B init per tier
    wandb_run = None
    wandb_table = None
    if USE_WANDB:
        try:
            from google.colab import userdata
            os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                config={
                    "study_name": study_name,
                    "run_id": RUN_ID,
                    "N_true": N_true,
                    "N_free": N_free,
                    "K": K,
                    "num_sweeps": NUM_SWEEPS,
                    "num_reads": NUM_READS,
                    "patience": PATIENCE,
                    "use_best_sa": USE_BEST_SA,
                    "lambda_range": [LAMBDA_LOWER, LAMBDA_UPPER],
                    "force_retune": FORCE_RETUNE,
                    "min_feasibility_floor": MIN_FEASIBILITY_FLOOR,
                    "backup_freq": BACKUP_FREQ,
                    "optimization_objectives": "minimize elite_avg_energy, maximize feasibility_rate",
                    "constraints": "budget_dev == 0, isolated_count == 0",
                },
                name=f"{study_name}_{datetime.now().strftime('%H%M%S')}",
            )
            if LOG_WANDB_TABLE:
                wandb_table = wandb.Table(
                    columns=[
                        "trial", "elite_avg_energy", "feasibility_rate",
                        "budget_dev", "isolated_count", "lambda_budget",
                        "lambda_conn", "num_selected", "runtime",
                        "global_best_feasible_energy", "global_best_elite_avg"
                    ],
                    log_mode="MUTABLE"
                )
            print("  ✅ W&B logging enabled for this tier.")
        except Exception as e:
            print(f"  ⚠️ W&B init failed: {e}")
            wandb_run = None

    # Create objective for this tier
    objective = make_objective(
        instance_data, neigh, K, fixed_neighbors,
        NUM_READS, NUM_SWEEPS, LAMBDA_LOWER, LAMBDA_UPPER,
        USE_BEST_SA, MIN_FEASIBILITY_FLOOR,
        wandb_run, wandb_table
    )

    # Sampler and study
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=WARMUP_TRIALS, constraints_func=constraint_func)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{local_db_path}",
        sampler=sampler,
        directions=["minimize", "maximize"],  # [elite_avg_energy, feasibility_rate]
        load_if_exists=not FORCE_RETUNE,
    )

    callbacks = [
        ParetoFrontPatienceCallback(patience=PATIENCE, warmup_trials=WARMUP_TRIALS),
        DriveBackupCallback(local_path=local_db_path, drive_path=drive_db_path, backup_freq=BACKUP_FREQ)
    ]

    print(f"\n  Running SA tuning ({NUM_SWEEPS} sweeps, {NUM_READS} reads)")
    print(f"  Lambda range: [{LAMBDA_LOWER:.4f}, {LAMBDA_UPPER:.4f}] (log scale)")
    print(f"  Patience: {PATIENCE} trials")
    print(f"  Use best SA (response.first): {USE_BEST_SA}")
    study.optimize(objective, callbacks=callbacks, gc_after_trial=True)

    # Extract best trial
    pareto_trials = study.best_trials
    if not pareto_trials:
        print(f"  ⚠️ No Pareto trials found for N={N_true}. Skipping.")
        all_tuning_results.append({
            "N_true": N_true,
            "success": False,
            "reason": "No Pareto trials",
        })
        continue

    # Select best trial: sort by elite_avg_energy (lower better), then feasibility_rate (higher better)
    best_trial = sorted(pareto_trials, key=lambda t: (t.values[0], -t.values[1]))[0]

    best_params = best_trial.params.copy()
    best_params["num_sweeps"] = NUM_SWEEPS
    best_params["num_reads"] = NUM_READS

    best_elite = best_trial.values[0]
    best_feasibility = best_trial.values[1]
    best_budget_dev = best_trial.user_attrs.get("budget_dev", np.nan)
    best_isolated_count = best_trial.user_attrs.get("isolated_count", np.nan)

    print("\n" + "-" * 65)
    print(f"🏆 BEST TRIAL FOR N={N_true}: Trial #{best_trial.number}")
    print(f"  Elite Avg Energy: {best_elite:.4f}")
    print(f"  Feasibility Rate: {best_feasibility:.2%}")
    print(f"  Budget Deviation: {best_budget_dev:.2f}")
    print(f"  Isolated Stations: {best_isolated_count:.2f}")
    print("  Optimized Hyperparameters:")
    for pname, pval in best_params.items():
        print(f"    • {pname:<18}: {pval:.4f}")
    print("-" * 65)

    # Save best params
    best_params_path = RUN_DIR / f"best_sa_N{N_true}.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2, cls=NumpyEncoder)
    safe_save_pickle(LOCAL_CACHE / f"best_sa_N{N_true}.pkl", best_params, verbose=False)
    print(f"  ✅ Best params saved to {best_params_path}")

    # ---- Save Optuna plots ----
    if SAVE_PLOTS:
        print("\n  📈 Generating Optuna plots...")
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_slice,
            plot_param_importances,
            plot_pareto_front,
        )

        def save_figure(fig, filename, format="png"):
            try:
                fig.write_image(str(filename))
                print(f"    Saved {filename}")
                return True
            except ValueError as e:
                if "kaleido" in str(e):
                    print(f"    ⚠️ kaleido not available, saving as HTML: {filename.with_suffix('.html')}")
                    fig.write_html(str(filename.with_suffix(".html")))
                    return True
                else:
                    raise

        tier_plot_dir = RUN_DIR / f"plots_N{N_true}"
        tier_plot_dir.mkdir(parents=True, exist_ok=True)

        # Pareto Front (elite vs feasibility)
        fig_pareto = plot_pareto_front(study, target_names=["Elite Avg Energy (min)", "Feasibility Rate (max)"])
        fig_pareto.update_layout(title=f"Pareto Front (SA) – N={N_true}")
        if SHOW_PLOTS:
            fig_pareto.show()
        if SAVE_PLOTS:
            save_figure(fig_pareto, tier_plot_dir / f"pareto_N{N_true}.png")

        # Optimization history for elite energy
        fig_hist_elite = plot_optimization_history(study, target=lambda t: t.values[0], target_name="Elite Avg Energy (min)")
        fig_hist_elite.update_layout(title=f"Optimization History – N={N_true} (Elite Energy)")
        if SHOW_PLOTS:
            fig_hist_elite.show()
        if SAVE_PLOTS:
            save_figure(fig_hist_elite, tier_plot_dir / f"history_elite_N{N_true}.png")

        # Optimization history for feasibility
        fig_hist_feas = plot_optimization_history(study, target=lambda t: t.values[1], target_name="Feasibility Rate (max)")
        fig_hist_feas.update_layout(title=f"Optimization History – N={N_true} (Feasibility)")
        if SHOW_PLOTS:
            fig_hist_feas.show()
        if SAVE_PLOTS:
            save_figure(fig_hist_feas, tier_plot_dir / f"history_feasibility_N{N_true}.png")

        # Parallel coordinates (elite energy)
        fig_par = plot_parallel_coordinate(study, params=["lambda_budget", "lambda_conn"],
                                           target=lambda t: t.values[0], target_name="Elite Avg Energy (min)")
        fig_par.update_layout(title=f"Parallel Coordinates – N={N_true} (Elite Energy)")
        if SHOW_PLOTS:
            fig_par.show()
        if SAVE_PLOTS:
            save_figure(fig_par, tier_plot_dir / f"parallel_elite_N{N_true}.png")

        # Slice plots
        fig_slice = plot_slice(study, params=["lambda_budget", "lambda_conn"],
                               target=lambda t: t.values[0], target_name="Elite Avg Energy (min)")
        fig_slice.update_layout(title=f"Slice Plot – N={N_true} (Elite Energy)")
        if SHOW_PLOTS:
            fig_slice.show()
        if SAVE_PLOTS:
            save_figure(fig_slice, tier_plot_dir / f"slice_elite_N{N_true}.png")

        # Importance
        fig_imp = plot_param_importances(study, params=["lambda_budget", "lambda_conn"],
                                         target=lambda t: t.values[0], target_name="Elite Avg Energy (min)")
        fig_imp.update_layout(title=f"Hyperparameter Importance – N={N_true} (Elite Energy)")
        if SHOW_PLOTS:
            fig_imp.show()
        if SAVE_PLOTS:
            save_figure(fig_imp, tier_plot_dir / f"importance_elite_N{N_true}.png")

    # W&B finish
    if wandb_run is not None:
        wandb_run.finish()

    # Store summary
    all_tuning_results.append({
        "N_true": N_true,
        "N_free": N_free,
        "K": K,
        "success": True,
        "best_elite": best_elite,
        "best_feasibility": best_feasibility,
        "best_params": best_params,
        "best_trial_number": best_trial.number,
        "qsum": qsum,
        "D_max": dmax,
    })

# -----------------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 110)
print("📊 SA TUNING SUMMARY (ALL TIERS)")
print("=" * 110)
print(f"{'N_true':<10} | {'N_free':<10} | {'K':<5} | {'Best Elite':<14} | {'Feasibility':<12} | {'qsum':<10} | {'Status':<10}")
print("-" * 110)
for res in all_tuning_results:
    if res.get("success", False):
        print(f"{res['N_true']:<10} | {res['N_free']:<10} | {res['K']:<5} | {res['best_elite']:<14.4f} | {res['best_feasibility']:<12.2%} | {res['qsum']:<10.2f} | {'✅ OK':<10}")
    else:
        print(f"{res['N_true']:<10} | {'-':<10} | {'-':<5} | {'-':<14} | {'-':<12} | {'-':<10} | {'❌ Failed':<10}")
print("=" * 110)

print("\n✅ SA tuning cell complete.")