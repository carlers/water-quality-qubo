# =============================================================================
# src/jij_solvers.py – SA and SQA Solvers using JijModeling + OpenJij
# =============================================================================
# This module provides solver functions for:
#   - Simulated Annealing (SA) via OMMXOpenJijSAAdapter
#   - Simulated Quantum Annealing (SQA) via oj.SQASampler
#
# Both return a standard dictionary with:
#   - solution: binary numpy array of length N
#   - energy: raw MIQP energy (objective without penalties)
#   - runtime: wall-clock time in seconds
#   - feasible: boolean indicating if all constraints are satisfied
#   - status: string (e.g., "optimal", "infeasible", "error")
#
# They rely on jij_model for the problem definition and penalty mapping.
# =============================================================================

import time
import numpy as np
import jijmodeling as jm
import openjij as oj
from ommx_openjij_adapter import OMMXOpenJijSAAdapter
from typing import Dict, Optional, Tuple

from .jij_model import build_augmented_model, get_penalty_weights, compile_instance

__all__ = [
    "solve_sa_jij",
    "solve_sqa_jij",
    "decode_solution",
    "check_feasibility",
    "compute_energy",
]


# -----------------------------------------------------------------------------
# Helper: decode solution from OMMX solution or OpenJij response
# -----------------------------------------------------------------------------
def decode_solution(solution_obj, N: int) -> np.ndarray:
    """Decode a binary solution vector from an OMMX Solution or OpenJij response."""
    x_sol = np.zeros(N, dtype=int)

    # OMMX Solution case
    if hasattr(solution_obj, "decision_variables_df"):
        df = solution_obj.decision_variables_df
        # Filter only variables with name 'x'
        x_df = df[df["name"] == "x"]
        selected = x_df[x_df["value"] == 1]["subscripts"].tolist()
        for s in selected:
            # subscripts is a tuple, e.g., (i,)
            idx = s[0] if isinstance(s, tuple) else s
            if idx < N:
                x_sol[idx] = 1
        return x_sol

    # OpenJij Response with .first.sample
    elif hasattr(solution_obj, "first"):
        best_sample = solution_obj.first.sample
        for idx, val in best_sample.items():
            if idx < N:
                x_sol[idx] = int(val)
        return x_sol

    # Fallback: if it's a dict directly
    elif isinstance(solution_obj, dict):
        for idx, val in solution_obj.items():
            if idx < N:
                x_sol[idx] = int(val)
        return x_sol

    else:
        raise TypeError(f"Unsupported solution object type: {type(solution_obj)}")


# -----------------------------------------------------------------------------
# Helper: check feasibility (budget + connectivity)
# -----------------------------------------------------------------------------
def check_feasibility(x_sol: np.ndarray, neigh: np.ndarray, K: int) -> bool:
    """Check budget and connectivity constraints."""
    if x_sol is None:
        return False
    selected = np.where(x_sol == 1)[0]
    if len(selected) != K:
        return False
    for i in selected:
        if not np.any(neigh[i, selected] == 1):
            return False
    return True


# -----------------------------------------------------------------------------
# Helper: compute raw MIQP energy (objective only, no penalties)
# -----------------------------------------------------------------------------
def compute_energy(x_sol: np.ndarray, a: np.ndarray, Q: np.ndarray) -> float:
    """Return raw MIQP energy: ∑ a_i x_i + ∑_{i<j} Q_ij x_i x_j."""
    N = len(x_sol)
    energy = np.dot(a, x_sol)
    # Quadratic part: sum over i<j
    for i in range(N):
        for j in range(i+1, N):
            if Q[i, j] != 0:
                energy += Q[i, j] * x_sol[i] * x_sol[j]
    return energy


# -----------------------------------------------------------------------------
# SA solver using OMMX adapter
# -----------------------------------------------------------------------------
def solve_sa_jij(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
    num_reads: int,
    num_sweeps: int,
    verbose: bool = False,
) -> Dict:
    """
    Solve using Simulated Annealing via OMMXOpenJijSAAdapter.
    """
    N = instance_data["N"]
    K = instance_data["K"]
    a = instance_data["a"]
    Q = instance_data["Q"]
    neigh = instance_data["neigh"]

    # Compile instance
    model = build_augmented_model()
    instance = compile_instance(model, instance_data)

    start = time.perf_counter()
    try:
        solution = OMMXOpenJijSAAdapter.solve(
            instance,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            penalty_weights=penalty_weights,
        )
        runtime = time.perf_counter() - start
        x_sol = decode_solution(solution, N)
        feasible = check_feasibility(x_sol, neigh, K)
        energy = compute_energy(x_sol, a, Q)
        status = "optimal" if feasible else "infeasible"
        if verbose:
            print(f"    SA: energy={energy:.6f}, runtime={runtime:.4f}s, feasible={feasible}")
        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "status": status,
        }
    except Exception as e:
        runtime = time.perf_counter() - start
        if verbose:
            print(f"    SA error: {e}")
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": runtime,
            "feasible": False,
            "status": f"error: {e}",
        }


# -----------------------------------------------------------------------------
# SQA solver using OpenJij SQASampler directly
# -----------------------------------------------------------------------------
def solve_sqa_jij(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
    num_reads: int,
    num_sweeps: int,
    trotter: int,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> Dict:
    """
    Solve using Simulated Quantum Annealing via OpenJij SQASampler.
    """
    N = instance_data["N"]
    K = instance_data["K"]
    a = instance_data["a"]
    Q = instance_data["Q"]
    neigh = instance_data["neigh"]

    # Compile instance
    model = build_augmented_model()
    instance = compile_instance(model, instance_data)

    # Build QUBO with penalties
    qubo_dict, constant = instance.to_qubo(penalty_weights=penalty_weights)

    start = time.perf_counter()
    try:
        sampler = oj.SQASampler()
        response = sampler.sample_qubo(
            qubo_dict,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            sparse=True,
            seed=seed,
        )
        runtime = time.perf_counter() - start

        # Decode best solution from response.first.sample
        x_sol = decode_solution(response, N)
        feasible = check_feasibility(x_sol, neigh, K)
        energy = compute_energy(x_sol, a, Q)
        status = "optimal" if feasible else "infeasible"
        if verbose:
            print(f"    SQA: energy={energy:.6f}, runtime={runtime:.4f}s, feasible={feasible}")
        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "status": status,
        }
    except Exception as e:
        runtime = time.perf_counter() - start
        if verbose:
            print(f"    SQA error: {e}")
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": runtime,
            "feasible": False,
            "status": f"error: {e}",
        }


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
    # simple connectivity: each site connected to next
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
    # penalty weights (dummy)
    model = build_augmented_model()
    instance = compile_instance(model, data)
    pw = get_penalty_weights(instance, lambda_budget=1.0, lambda_conn=1.0)

    print("Testing SA...")
    res_sa = solve_sa_jij(data, pw, num_reads=10, num_sweeps=100, verbose=True)
    print("Testing SQA...")
    res_sqa = solve_sqa_jij(data, pw, num_reads=10, num_sweeps=100, trotter=4, verbose=True)