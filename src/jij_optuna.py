# =============================================================================
# src/jij_optuna.py – Optuna Hyperparameter Tuning for SA and SQA
# =============================================================================
# This module provides functions to tune hyperparameters for:
#   - Simulated Annealing (SA)
#   - Simulated Quantum Annealing (SQA)
#
# It uses the solvers from jij_solvers.py and returns the best parameters.
#
# SPECIFICATION (from mentor):
#   - SA: tune only lambda_budget, lambda_conn (fixed sweeps=10000, reads=1024)
#   - SQA: tune lambda_budget, lambda_conn, trotter (fixed sweeps=10000, reads=1024)
#   - Multi-objective: [MIQP_energy, violation_rate] with directions ['minimize', 'minimize']
#   - Use MultiObjectivePatienceCallback (patience=50) – no n_trials or timeout
#   - Lambda bounds: 0.01 to Qsum (log scale)
#   - Trotter bounds: 1.0 to 64.0 (log scale, float)
# =============================================================================

import optuna
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable, Union

from .jij_model import build_augmented_model, compile_instance, get_penalty_weights
from .jij_solvers import solve_sa_jij, solve_sqa_jij
from .utils import (
    compute_violation_rate,
    select_best_from_pareto,
    print_multiobjective_tuning_summary,
    safe_save_pickle,
    safe_load_pickle,
    NumpyEncoder,
)

# Try importing wandb for optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

warnings.filterwarnings('ignore')

__all__ = [
    "tune_sa",
    "tune_sqa",
    "MultiObjectivePatienceCallback",
    "compute_qsum",
]


# -----------------------------------------------------------------------------
# Helper: compute Qsum for penalty scaling
# -----------------------------------------------------------------------------
def compute_qsum(instance_data: Dict) -> float:
    """
    Compute Qsum = sum(|a_i|) + sum(|Q_ij|) over i<j.
    Used to scale penalty ranges.
    """
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    N = len(a)
    qsum = np.sum(np.abs(a))
    for i in range(N):
        for j in range(i + 1, N):
            qsum += np.abs(Q[i, j])
    return max(qsum, 1.0)  # ensure positive


# -----------------------------------------------------------------------------
# MultiObjectivePatienceCallback – Early stopping based on Pareto front
# -----------------------------------------------------------------------------
class MultiObjectivePatienceCallback:
    """
    Early stopping callback for multi-objective Optuna studies.
    
    Tracks the evolution of the Pareto front and stops the study if the front
    has not improved for `patience` consecutive trials.
    
    Usage:
        callback = MultiObjectivePatienceCallback(patience=50)
        study.optimize(objective, callbacks=[callback])
    """
    
    def __init__(self, patience: int = 50):
        """
        Args:
            patience: Number of trials without Pareto front improvement before stopping.
        """
        self.patience = patience
        self.trials_without_improvement = 0
        self.best_pareto_set = set()
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Callback called after each trial.
        """
        # 1. Skip if the trial didn't complete successfully
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        
        # 2. Get the unique numbers of the current non-dominated trials (Pareto front)
        current_pareto_set = {t.number for t in study.best_trials}
        
        # 3. Check if the Pareto front has evolved
        if current_pareto_set != self.best_pareto_set:
            self.best_pareto_set = current_pareto_set
            self.trials_without_improvement = 0  # Reset patience
        else:
            self.trials_without_improvement += 1  # Increment patience count
        
        # 4. Trigger study termination if patience expires
        if self.trials_without_improvement >= self.patience:
            print(f"\n[Early Stopping] The Pareto front hasn't improved for {self.patience} trials. Stopping study.")
            study.stop()  # Aborts the entire optimization loop smoothly


# -----------------------------------------------------------------------------
# Tuning for SA
# -----------------------------------------------------------------------------

def tune_sa(
    instance_data: Dict,
    qsum: Optional[float] = None,
    num_sweeps: int = 10000,
    num_reads: int = 1024,
    patience: int = 50,
    study_name: str = "sa_tuning",
    storage_dir: Optional[Union[str, Path]] = None,
    force_retune: bool = False,
    wandb_run: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Dict, optuna.Study]:
    """
    Tune SA hyperparameters using Optuna (multi‑objective).
    
    Objectives:
        1. Minimize MIQP energy
        2. Minimize continuous violation (budget deviation + isolated count)
    
    Hyperparameters (log scale):
        - lambda_budget: 0.01 to 1000.0
        - lambda_conn: 0.01 to 1000.0
    """
    if qsum is None:
        qsum = compute_qsum(instance_data)
    
    if storage_dir is not None:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = storage_dir / f"{study_name}.db"
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={'connect_args': {'timeout': 60, 'check_same_thread': False}}
        )
    else:
        storage = None
    
    if force_retune:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            if verbose:
                print(f"  Deleted existing study '{study_name}' (force_retune)")
        except optuna.exceptions.StudyNotFoundError:
            pass  # no existing study, fine
    
    if verbose:
        print(f"\n🔬 Tuning SA (sweeps={num_sweeps}, reads={num_reads})")
        print(f"  Qsum = {qsum:.4f}")
        print(f"  Lambda range: [0.01, {qsum:.2f}] (log scale)")
        print(f"  Patience: {patience} trials")

    # Build model once for penalty ID mapping
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    # Extract fixed_neighbors for violation computation
    fixed_neighbors = instance_data.get("fixed_neighbors", None)
    K = instance_data["K"]
    neigh = instance_data["neigh"]

    def objective(trial):
        # Hyperparameters (log scale, wider range)
        lambda_budget = trial.suggest_float("lambda_budget", 0.01, qsum, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", 0.01, qsum, log=True)
        
        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)
        
        result = solve_sa_jij(
            instance_data,
            penalty_weights,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            return_all=False,
            verbose=False,
        )
        
        # Compute violation metrics (continuous and discrete)
        if result["solution"] is not None:
            energy = float(result["energy"])
            viol_cont = compute_violation_rate(
                result["solution"], neigh, K, fixed_neighbors, continuous=True
            )
            viol_disc = compute_violation_rate(
                result["solution"], neigh, K, fixed_neighbors, continuous=False
            )
            feasible = bool(result["feasible"])
            num_selected = int(result["num_selected"])
            isolated = int(len(result.get("isolated_indices", [])))
            runtime = float(result["runtime"])
            status = str(result["status"])
        else:
            energy = np.nan
            viol_cont = np.nan
            viol_disc = np.nan
            feasible = False
            num_selected = 0
            isolated = 0
            runtime = float(result["runtime"])
            status = "error"
        
        # Store user attributes for later reference (e.g., best trial selection)
        trial.set_user_attr("energy", energy)
        trial.set_user_attr("violation_continuous", viol_cont)
        trial.set_user_attr("violation_discrete", viol_disc)
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr("num_selected", num_selected)
        trial.set_user_attr("isolated", isolated)
        trial.set_user_attr("runtime", runtime)
        trial.set_user_attr("status", status)
        if result["solution"] is not None:
            trial.set_user_attr("best_solution", result["solution"].tolist())
        
        # --- Manual W&B logging (per trial) ---
        if wandb_run is not None:
            wandb_run.log({
                "trial_number": trial.number,
                "energy": energy,
                "violation_continuous": viol_cont,
                "violation_discrete": viol_disc,
                "feasible": feasible,
                "num_selected": num_selected,
                "isolated": isolated,
                "lambda_budget": lambda_budget,
                "lambda_conn": lambda_conn,
                "runtime": runtime,
                "status": status,
            })
        
        # Return multi‑objective values (energy, continuous violation)
        return [energy, viol_cont]
    
    # Create study (multi‑objective)
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        directions=["minimize", "minimize"],
        load_if_exists=not force_retune,
    )
    
    # Setup patience callback (no W&B callback)
    callback = MultiObjectivePatienceCallback(patience=patience)
    callbacks = [callback]
    
    if verbose:
        print(f"  Running optimization (stopping after {patience} trials without Pareto improvement)...")
    
    study.optimize(objective, callbacks=callbacks)
    
    # Select best parameters from Pareto front
    best_trial = select_best_from_pareto(study)
    if best_trial is None:
        if verbose:
            print("  ⚠️ No valid trials found. Using fallback parameters.")
        best_params = {"lambda_budget": 0.1 * qsum, "lambda_conn": 0.1 * qsum}
    else:
        best_params = best_trial.params
    
    # Add fixed parameters
    best_params["num_sweeps"] = num_sweeps
    best_params["num_reads"] = num_reads
    
    if verbose:
        print_multiobjective_tuning_summary(study, title=f"SA Tuning: {study_name}")
        print("\n  Best parameters selected:")
        for key, val in best_params.items():
            print(f"    {key}: {val}")
    
    return best_params, study


# -----------------------------------------------------------------------------
# Tuning for SQA
# -----------------------------------------------------------------------------
def tune_sqa(
    instance_data: Dict,
    qsum: Optional[float] = None,
    num_sweeps: int = 10000,
    num_reads: int = 1024,
    patience: int = 50,
    study_name: str = "sqa_tuning",
    storage_dir: Optional[Union[str, Path]] = None,
    force_retune: bool = False,
    wandb_run: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Dict, optuna.Study]:
    """
    Tune SQA hyperparameters using Optuna (multi‑objective).
    
    Objectives:
        1. Minimize MIQP energy
        2. Minimize continuous violation (budget deviation + isolated count)
    
    Hyperparameters (log scale):
        - lambda_budget: 0.01 to 1000.0
        - lambda_conn: 0.01 to 1000.0
        - trotter: 1.0 to 64.0 (log scale)
    """
    if qsum is None:
        qsum = compute_qsum(instance_data)
    
    if storage_dir is not None:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = storage_dir / f"{study_name}.db"
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={'connect_args': {'timeout': 60, 'check_same_thread': False}}
        )
    else:
        storage = None

    if force_retune:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            if verbose:
                print(f"  Deleted existing study '{study_name}' (force_retune)")
        except optuna.exceptions.StudyNotFoundError:
            pass  # no existing study, fine
    
    if verbose:
        print(f"\n🔬 Tuning SQA (sweeps={num_sweeps}, reads={num_reads})")
        print(f"  Qsum = {qsum:.4f}")
        print(f"  Lambda range: [0.01, {qsum:.2f}] (log scale)")
        print(f"  Trotter range: [1.0, 64.0] (log scale)")
        print(f"  Patience: {patience} trials")

    # Build model once for penalty ID mapping
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    fixed_neighbors = instance_data.get("fixed_neighbors", None)
    K = instance_data["K"]
    neigh = instance_data["neigh"]

    def objective(trial):
        lambda_budget = trial.suggest_float("lambda_budget", 0.01, qsum, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", 0.01, qsum, log=True)
        trotter_float = trial.suggest_float("trotter", 1.0, 64.0, log=True)
        trotter = int(round(trotter_float))
        
        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)
        
        result = solve_sqa_jij(
            instance_data,
            penalty_weights,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            return_all=False,
            verbose=False,
        )
        
        if result["solution"] is not None:
            energy = float(result["energy"])
            viol_cont = compute_violation_rate(
                result["solution"], neigh, K, fixed_neighbors, continuous=True
            )
            viol_disc = compute_violation_rate(
                result["solution"], neigh, K, fixed_neighbors, continuous=False
            )
            feasible = bool(result["feasible"])
            num_selected = int(result["num_selected"])
            isolated = int(len(result.get("isolated_indices", [])))
            runtime = float(result["runtime"])
            status = str(result["status"])
        else:
            energy = np.nan
            viol_cont = np.nan
            viol_disc = np.nan
            feasible = False
            num_selected = 0
            isolated = 0
            runtime = float(result["runtime"])
            status = "error"
        
        trial.set_user_attr("energy", energy)
        trial.set_user_attr("violation_continuous", viol_cont)
        trial.set_user_attr("violation_discrete", viol_disc)
        trial.set_user_attr("feasible", feasible)
        trial.set_user_attr("num_selected", num_selected)
        trial.set_user_attr("isolated", isolated)
        trial.set_user_attr("runtime", runtime)
        trial.set_user_attr("status", status)
        trial.set_user_attr("trotter_used", trotter)
        if result["solution"] is not None:
            trial.set_user_attr("best_solution", result["solution"].tolist())
        
        # Manual W&B logging
        if wandb_run is not None:
            wandb_run.log({
                "trial_number": trial.number,
                "energy": energy,
                "violation_continuous": viol_cont,
                "violation_discrete": viol_disc,
                "feasible": feasible,
                "num_selected": num_selected,
                "isolated": isolated,
                "lambda_budget": lambda_budget,
                "lambda_conn": lambda_conn,
                "trotter": trotter,
                "runtime": runtime,
                "status": status,
            })
        
        # Return multi‑objective values
        return [energy, viol_cont]
    
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        directions=["minimize", "minimize"],
        load_if_exists=not force_retune,
    )
    
    callback = MultiObjectivePatienceCallback(patience=patience)
    callbacks = [callback]
    
    if verbose:
        print(f"  Running optimization (stopping after {patience} trials without Pareto improvement)...")
    
    study.optimize(objective, callbacks=callbacks)
    
    best_trial = select_best_from_pareto(study)
    if best_trial is None:
        if verbose:
            print("  ⚠️ No valid trials found. Using fallback parameters.")
        best_params = {
            "lambda_budget": 0.1 * qsum,
            "lambda_conn": 0.1 * qsum,
            "trotter": 16,
        }
    else:
        best_params = best_trial.params
        if "trotter" in best_params:
            best_params["trotter"] = int(round(best_params["trotter"]))
    
    best_params["num_sweeps"] = num_sweeps
    best_params["num_reads"] = num_reads
    
    if verbose:
        print_multiobjective_tuning_summary(study, title=f"SQA Tuning: {study_name}")
        print("\n  Best parameters selected:")
        for key, val in best_params.items():
            print(f"    {key}: {val}")
    
    return best_params, study


# -----------------------------------------------------------------------------
# For testing (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    N, K = 10, 3
    a = -np.random.rand(N)
    Q = np.random.rand(N, N) * 0.1
    Q = (Q + Q.T) / 2
    np.fill_diagonal(Q, 0)
    
    neigh = np.zeros((N, N), dtype=int)
    for i in range(N - 1):
        neigh[i, i + 1] = 1
        neigh[i + 1, i] = 1
    
    data = {
        "N": N,
        "K": K,
        "a": a.tolist(),
        "Q": Q.tolist(),
        "neigh": neigh.tolist(),
    }
    
    qsum = compute_qsum(data)
    
    print("=" * 80)
    print("Testing SA tuning...")
    print("=" * 80)
    sa_params, sa_study = tune_sa(
        data,
        qsum=qsum,
        num_sweeps=100,
        num_reads=10,
        patience=3,
        verbose=True,
    )
    
    print("\n" + "=" * 80)
    print("Testing SQA tuning...")
    print("=" * 80)
    sqa_params, sqa_study = tune_sqa(
        data,
        qsum=qsum,
        num_sweeps=100,
        num_reads=10,
        patience=3,
        verbose=True,
    )
    
    print("\n" + "=" * 80)
    print("Tuning complete!")
    print("SA params:", sa_params)
    print("SQA params:", sqa_params)
    print("=" * 80)