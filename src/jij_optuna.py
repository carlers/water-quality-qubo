# =============================================================================
# src/jij_optuna.py – Optuna Hyperparameter Tuning for SA and SQA
# =============================================================================
# This module provides functions to tune hyperparameters for:
#   - Simulated Annealing (SA)
#   - Simulated Quantum Annealing (SQA)
#
# It uses the solvers from jij_solvers.py and returns the best parameters.
# =============================================================================

import optuna
import numpy as np
from typing import Dict, Optional

from .jij_model import build_augmented_model, compile_instance, get_penalty_weights
from .jij_solvers import solve_sa_jij, solve_sqa_jij

__all__ = [
    "tune_sa",
    "tune_sqa",
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
    a = instance_data["a"]
    Q = instance_data["Q"]
    N = len(a)
    qsum = np.sum(np.abs(a))
    for i in range(N):
        for j in range(i+1, N):
            qsum += np.abs(Q[i, j])
    return max(qsum, 1.0)  # ensure positive


# -----------------------------------------------------------------------------
# Tuning for SA
# -----------------------------------------------------------------------------
def tune_sa(
    instance_data: Dict,
    n_trials: int = 20,
    timeout: Optional[int] = 300,
    verbose: bool = True,
) -> Dict:
    """
    Tune SA hyperparameters using Optuna.
    Returns dict with best parameters:
        lambda_budget, lambda_conn, num_sweeps, num_reads
    """
    qsum = compute_qsum(instance_data)
    lb_min = 0.1 * qsum
    lb_max = 20.0 * qsum

    if verbose:
        print(f"Qsum = {qsum:.4f}")
        print(f"Lambda range: [{lb_min:.2f}, {lb_max:.2f}]")

    # Build model once for penalty ID mapping
    model = build_augmented_model()
    instance = compile_instance(model, instance_data)

    def objective(trial):
        lambda_budget = trial.suggest_float("lambda_budget", lb_min, lb_max, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", lb_min, lb_max, log=True)
        num_sweeps = trial.suggest_int("num_sweeps", 500, 3000, step=100)
        num_reads = trial.suggest_int("num_reads", 50, 200, step=50)

        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)

        result = solve_sa_jij(
            instance_data,
            penalty_weights,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            verbose=False,
        )

        if not result["feasible"]:
            return 1e6  # large penalty for infeasible

        return result["energy"]  # raw MIQP energy (minimise)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    if verbose:
        print("\nBest SA parameters:")
        print(f"  Value (energy): {best.value:.6f}")
        for key, val in best.params.items():
            print(f"    {key}: {val}")

    return best.params


# -----------------------------------------------------------------------------
# Tuning for SQA
# -----------------------------------------------------------------------------
def tune_sqa(
    instance_data: Dict,
    n_trials: int = 15,
    timeout: Optional[int] = 600,
    verbose: bool = True,
) -> Dict:
    """
    Tune SQA hyperparameters using Optuna.
    Returns dict with best parameters:
        lambda_budget, lambda_conn, num_sweeps, num_reads, trotter
    """
    qsum = compute_qsum(instance_data)
    lb_min = 0.01
    lb_max = qsum

    if verbose:
        print(f"Qsum = {qsum:.4f}")
        print(f"Lambda range: [{lb_min:.2f}, {lb_max:.2f}]")

    model = build_augmented_model()
    instance = compile_instance(model, instance_data)

    def objective(trial):
        lambda_budget = trial.suggest_float("lambda_budget", lb_min, lb_max, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", lb_min, lb_max, log=True)
        num_sweeps = trial.suggest_int("num_sweeps", 500, 3000, step=100)
        num_reads = trial.suggest_int("num_reads", 50, 200, step=50)
        trotter = trial.suggest_int("trotter", 4, 32, step=4)

        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)

        result = solve_sqa_jij(
            instance_data,
            penalty_weights,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            verbose=False,
        )

        if not result["feasible"]:
            return 1e6

        return result["energy"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    if verbose:
        print("\nBest SQA parameters:")
        print(f"  Value (energy): {best.value:.6f}")
        for key, val in best.params.items():
            print(f"    {key}: {val}")

    return best.params


# -----------------------------------------------------------------------------
# For testing (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    N, K = 10, 3
    a = -np.random.rand(N)
    Q = np.random.rand(N, N) * 0.1
    neigh = np.zeros((N, N), dtype=int)
    for i in range(N-1):
        neigh[i, i+1] = 1
        neigh[i+1, i] = 1
    data = {
        "N": N,
        "K": K,
        "a": a.tolist(),
        "Q": Q.tolist(),
        "neigh": neigh.tolist(),
    }

    print("Tuning SA...")
    best_sa = tune_sa(data, n_trials=5, timeout=60)
    print("\nTuning SQA...")
    best_sqa = tune_sqa(data, n_trials=5, timeout=60)