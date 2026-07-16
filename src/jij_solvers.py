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
#   - violation_rate: 0.0, 0.5, or 1.0 (budget + connectivity)
#   - status: string (e.g., "optimal", "infeasible", "error")
#   - all_samples: (if return_all=True) list of dicts with solution, energy, violation_rate
#
# They rely on jij_model for the problem definition and penalty mapping.
# =============================================================================

import time
import numpy as np
import jijmodeling as jm
import openjij as oj
from ommx_openjij_adapter import OMMXOpenJijSAAdapter
from typing import Dict, Optional, Tuple, List, Any

from .jij_model import build_augmented_model, get_penalty_weights, compile_instance
from .utils import compute_violation_rate

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
    """
    Decode a binary solution vector from an OMMX Solution or OpenJij response.
    
    Handles:
        - OMMX Solution with decision_variables_df
        - OpenJij Response with .first.sample
        - Direct dict of {idx: value}
    """
    x_sol = np.zeros(N, dtype=int)

    # OMMX Solution case
    if hasattr(solution_obj, "decision_variables_df"):
        df = solution_obj.decision_variables_df
        # Filter only variables with name 'x'
        x_df = df[df["name"] == "x"]
        for _, row in x_df.iterrows():
            if row["value"] == 1:
                # subscripts can be tuple or list; extract first element
                subs = row["subscripts"]
                if isinstance(subs, (tuple, list)) and len(subs) > 0:
                    idx = subs[0]
                else:
                    # fallback: if subs is int or something else
                    idx = int(subs)
                if 0 <= idx < N:
                    x_sol[idx] = 1
        return x_sol

    # OpenJij Response with .first.sample
    elif hasattr(solution_obj, "first"):
        best_sample = solution_obj.first.sample
        for idx, val in best_sample.items():
            # idx can be int or tuple; convert to int if needed
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(val)
        return x_sol

    # Fallback: if it's a dict directly
    elif isinstance(solution_obj, dict):
        for idx, val in solution_obj.items():
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(val)
        return x_sol

    else:
        raise TypeError(f"Unsupported solution object type: {type(solution_obj)}")


# -----------------------------------------------------------------------------
# Helper: check feasibility (budget + connectivity)
# -----------------------------------------------------------------------------
def check_feasibility(
    x_sol: np.ndarray,
    neigh: np.ndarray,
    K: int
) -> Dict[str, Any]:
    """
    Check budget and connectivity constraints.
    
    Returns:
        dict with:
            - budget_ok: bool
            - connectivity_ok: bool
            - feasible: bool
            - num_selected: int
            - isolated_indices: List[int]
    """
    if x_sol is None or neigh is None:
        return {
            "budget_ok": False,
            "connectivity_ok": False,
            "feasible": False,
            "num_selected": 0,
            "isolated_indices": [],
        }
    
    x_sol = np.asarray(x_sol)
    neigh = np.asarray(neigh)
    selected = np.where(x_sol == 1)[0]
    num_selected = len(selected)
    
    # Budget constraint
    budget_ok = (num_selected == K)
    
    # Connectivity constraint
    isolated_indices = []
    if num_selected == 0:
        connectivity_ok = False
    else:
        for i in selected:
            if not np.any(neigh[i, selected] == 1):
                isolated_indices.append(int(i))
        connectivity_ok = (len(isolated_indices) == 0)
    
    return {
        "budget_ok": budget_ok,
        "connectivity_ok": connectivity_ok,
        "feasible": budget_ok and connectivity_ok,
        "num_selected": num_selected,
        "isolated_indices": isolated_indices,
    }


# -----------------------------------------------------------------------------
# Helper: compute raw MIQP energy (objective only, no penalties)
# -----------------------------------------------------------------------------
def compute_energy(x_sol: np.ndarray, a: np.ndarray, Q: np.ndarray) -> float:
    """Return raw MIQP energy: ∑ a_i x_i + ∑_{i<j} Q_ij x_i x_j."""
    x_sol = np.asarray(x_sol)
    N = len(x_sol)
    a = np.asarray(a)
    Q = np.asarray(Q)
    
    energy = np.dot(a, x_sol)
    # Quadratic part: sum over i<j
    for i in range(N):
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                energy += Q[i, j] * x_sol[i] * x_sol[j]
    return float(energy)


# -----------------------------------------------------------------------------
# Helper: extract MIQP objective from instance_data
# -----------------------------------------------------------------------------
def _get_miqp_arrays(instance_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a and Q arrays from instance_data."""
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    return a, Q


# -----------------------------------------------------------------------------
# SA solver using OMMX adapter
# -----------------------------------------------------------------------------
def solve_sa_jij(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
    num_reads: int = 1024,
    num_sweeps: int = 10000,
    return_all: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Solve using Simulated Annealing via OMMXOpenJijSAAdapter.
    
    Args:
        instance_data: dict with N, K, a, Q, neigh, coords, etc.
        penalty_weights: dict mapping constraint ID -> penalty weight.
        num_reads: number of independent runs.
        num_sweeps: number of sweeps per run.
        return_all: if True, return all samples with their energies and violation rates.
        verbose: print progress.
    
    Returns:
        dict with solution, energy, runtime, feasible, violation_rate, status, all_samples.
    """
    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    neigh = np.asarray(instance_data["neigh"])

    # Compile instance
    model = build_augmented_model()
    # Filter instance data to only model keys
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    start = time.perf_counter()
    try:
        solution = OMMXOpenJijSAAdapter.solve(
            instance,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            penalty_weights=penalty_weights,
        )
        runtime = time.perf_counter() - start
        
        # Decode best solution
        x_sol = decode_solution(solution, N)
        
        # Compute energy and feasibility
        energy = compute_energy(x_sol, a, Q)
        feas_detail = check_feasibility(x_sol, neigh, K)
        feasible = feas_detail["feasible"]
        violation_rate = compute_violation_rate(x_sol, neigh, K)
        status = "optimal" if feasible else "infeasible"
        
        # Build all_samples if requested
        all_samples = []
        if return_all:
            # OMMX solution may not expose all samples easily.
            # We'll create a single sample with the best solution.
            all_samples.append({
                "solution": x_sol.copy(),
                "energy": energy,
                "violation_rate": violation_rate,
                "feasible": feasible,
                "budget_ok": feas_detail["budget_ok"],
                "connectivity_ok": feas_detail["connectivity_ok"],
                "num_selected": feas_detail["num_selected"],
            })
        
        if verbose:
            print(f"    SA: energy={energy:.6f}, runtime={runtime:.4f}s, "
                  f"feasible={feasible}, violation_rate={violation_rate:.1f}")
        
        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "violation_rate": violation_rate,
            "status": status,
            "all_samples": all_samples if return_all else None,
            "budget_ok": feas_detail["budget_ok"],
            "connectivity_ok": feas_detail["connectivity_ok"],
            "num_selected": feas_detail["num_selected"],
            "isolated_indices": feas_detail["isolated_indices"],
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
            "violation_rate": 1.0,
            "status": f"error: {e}",
            "all_samples": None,
            "budget_ok": False,
            "connectivity_ok": False,
            "num_selected": 0,
            "isolated_indices": [],
        }


# -----------------------------------------------------------------------------
# SQA solver using OpenJij SQASampler directly
# -----------------------------------------------------------------------------
def solve_sqa_jij(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
    num_reads: int = 1024,
    num_sweeps: int = 10000,
    trotter: int = 16,
    seed: Optional[int] = 42,
    return_all: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Solve using Simulated Quantum Annealing via OpenJij SQASampler.
    
    Args:
        instance_data: dict with N, K, a, Q, neigh, coords, etc.
        penalty_weights: dict mapping constraint ID -> penalty weight.
        num_reads: number of independent runs.
        num_sweeps: number of sweeps per run.
        trotter: number of Trotter slices (quantum replicas).
        seed: random seed for reproducibility.
        return_all: if True, return all samples with their energies and violation rates.
        verbose: print progress.
    
    Returns:
        dict with solution, energy, runtime, feasible, violation_rate, status, all_samples.
    """
    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    neigh = np.asarray(instance_data["neigh"])

    # Compile instance
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    # Build QUBO with penalties
    qubo_dict, _ = instance.to_qubo(penalty_weights=penalty_weights)

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
        
        # Compute energy and feasibility
        energy = compute_energy(x_sol, a, Q)
        feas_detail = check_feasibility(x_sol, neigh, K)
        feasible = feas_detail["feasible"]
        violation_rate = compute_violation_rate(x_sol, neigh, K)
        status = "optimal" if feasible else "infeasible"
        
        # Build all_samples if requested
        all_samples = []
        if return_all:
            # Extract all samples from response
            for idx in range(response.record.shape[0]):
                sample_arr = response.record['sample'][idx]
                # Build full solution from sample
                x_sample = np.zeros(N, dtype=int)
                for var_idx, val in zip(response.indices, sample_arr):
                    if var_idx < N:
                        x_sample[var_idx] = int(round(val))
                
                # Compute energy and violation for this sample
                e_sample = compute_energy(x_sample, a, Q)
                v_sample = compute_violation_rate(x_sample, neigh, K)
                f_sample = check_feasibility(x_sample, neigh, K)
                
                all_samples.append({
                    "solution": x_sample.copy(),
                    "energy": e_sample,
                    "violation_rate": v_sample,
                    "feasible": f_sample["feasible"],
                    "budget_ok": f_sample["budget_ok"],
                    "connectivity_ok": f_sample["connectivity_ok"],
                    "num_selected": f_sample["num_selected"],
                })
        
        if verbose:
            print(f"    SQA: energy={energy:.6f}, runtime={runtime:.4f}s, "
                  f"feasible={feasible}, violation_rate={violation_rate:.1f}, trotter={trotter}")
        
        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "violation_rate": violation_rate,
            "status": status,
            "all_samples": all_samples if return_all else None,
            "budget_ok": feas_detail["budget_ok"],
            "connectivity_ok": feas_detail["connectivity_ok"],
            "num_selected": feas_detail["num_selected"],
            "isolated_indices": feas_detail["isolated_indices"],
            "trotter_used": trotter,
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
            "violation_rate": 1.0,
            "status": f"error: {e}",
            "all_samples": None,
            "budget_ok": False,
            "connectivity_ok": False,
            "num_selected": 0,
            "isolated_indices": [],
            "trotter_used": trotter,
        }


# -----------------------------------------------------------------------------
# Greedy solver (for baseline comparison)
# -----------------------------------------------------------------------------
def solve_greedy_jij(
    instance_data: Dict,
    verbose: bool = False,
) -> Dict:
    """
    Greedy solver: select K sites with highest utility (ignores connectivity).
    This is a baseline for comparison.
    """
    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    neigh = np.asarray(instance_data["neigh"])
    
    start = time.perf_counter()
    
    # Select K sites with lowest a (highest utility)
    # a is negative utility, so smallest a = highest utility
    indices = np.argsort(a)[:K]
    x_sol = np.zeros(N, dtype=int)
    x_sol[indices] = 1
    
    energy = compute_energy(x_sol, a, Q)
    runtime = time.perf_counter() - start
    
    feas_detail = check_feasibility(x_sol, neigh, K)
    feasible = feas_detail["feasible"]
    violation_rate = compute_violation_rate(x_sol, neigh, K)
    
    if verbose:
        print(f"    Greedy: energy={energy:.6f}, runtime={runtime:.4f}s, "
              f"feasible={feasible}, violation_rate={violation_rate:.1f}")
    
    return {
        "solution": x_sol,
        "energy": energy,
        "runtime": runtime,
        "feasible": feasible,
        "violation_rate": violation_rate,
        "status": "feasible" if feasible else "infeasible",
        "budget_ok": feas_detail["budget_ok"],
        "connectivity_ok": feas_detail["connectivity_ok"],
        "num_selected": feas_detail["num_selected"],
        "isolated_indices": feas_detail["isolated_indices"],
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
    # Make Q symmetric
    Q = (Q + Q.T) / 2
    np.fill_diagonal(Q, 0)
    
    neigh = np.zeros((N, N), dtype=int)
    # simple connectivity: each site connected to next
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
    
    # Build penalty weights (dummy)
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)
    penalty_weights = get_penalty_weights(instance, lambda_budget=1.0, lambda_conn=1.0)
    
    print("Testing SA...")
    res_sa = solve_sa_jij(data, penalty_weights, num_reads=10, num_sweeps=100, verbose=True)
    print(f"  SA feasible: {res_sa['feasible']}, violation_rate: {res_sa['violation_rate']}")
    
    print("\nTesting SQA...")
    res_sqa = solve_sqa_jij(data, penalty_weights, num_reads=10, num_sweeps=100, trotter=4, verbose=True)
    print(f"  SQA feasible: {res_sqa['feasible']}, violation_rate: {res_sqa['violation_rate']}")
    
    print("\nTesting Greedy...")
    res_greedy = solve_greedy_jij(data, verbose=True)
    print(f"  Greedy feasible: {res_greedy['feasible']}, violation_rate: {res_greedy['violation_rate']}")