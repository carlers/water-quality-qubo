"""
src/solvers.py

Classical solvers for the water quality monitoring QUBO problem.

This module provides:
    1. Greedy selection (simple top-K and marginal gain)
    2. Simulated Annealing (SA) via OpenJij
    3. Simulated Quantum Annealing (SQA) via OpenJij
    4. Constraint violation metrics for debugging
    5. Annealing schedule builder

All solvers accept the same QUBO representation (h, J, constant) and return
a standardized result dictionary for fair comparison.

Usage:
    from src.solvers import solve_all_classical
    
    results = solve_all_classical(
        pairwise_data=pairwise,
        K_new=5,
        lambda1=10.0,
        lambda2=10.0,
        verbose=True
    )
    print(results['SA']['miqp_energy'])
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# -----------------------------------------------------------------------------
# ANNEALING SCHEDULE BUILDER
# -----------------------------------------------------------------------------


def build_schedule(
    beta_min: float,
    beta_max: float,
    num_sweeps: int,
    num_steps: int,
    cooling_power: float,
) -> List[List[float]]:
    """
    Build annealing schedule for SA/SQA.

    Args:
        beta_min: Initial inverse temperature.
        beta_max: Final inverse temperature.
        num_sweeps: Total number of sweeps.
        num_steps: Number of schedule steps.
        cooling_power: Power for temperature progression (0.5-3.0).

    Returns:
        List of [beta, sweeps_per_step] pairs.

    Example:
        schedule = build_schedule(0.01, 40.0, 15000, 120, 1.8)
        # Returns [[0.01, 125], [0.02, 125], ...] where each step has 125 sweeps.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_sweeps <= 0:
        raise ValueError("num_sweeps must be positive")
    
    # Progress from 0 to 1, then raised to cooling_power
    progress = np.linspace(0.0, 1.0, num_steps) ** cooling_power
    betas = beta_min + (beta_max - beta_min) * progress
    
    # Distribute sweeps evenly
    sweeps_per_step = num_sweeps // num_steps
    schedule = [[float(b), sweeps_per_step] for b in betas]
    
    # Add remaining sweeps to the last step
    remaining = num_sweeps - (num_steps * sweeps_per_step)
    if remaining > 0:
        schedule[-1][1] += remaining
    
    return schedule


# -----------------------------------------------------------------------------
# CONSTRAINT VIOLATION METRICS
# -----------------------------------------------------------------------------


def compute_violations(
    x_full: np.ndarray,
    free_indices: List[int],
    M_indices: List[int],
    neighbors: Dict[int, List[int]],
    K_new: int,
) -> Dict[str, Union[int, float, bool]]:
    """
    Compute constraint violations for a solution.

    Args:
        x_full: Binary vector of length N_total.
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        neighbors: Dict {i: list_of_j} from pairwise_data.
        K_new: Required number of new stations.

    Returns:
        dict with:
            'budget_violation': |Σx_free - K_new| (absolute error)
            'num_isolated': number of selected free stations with no neighbors
            'max_isolated_penalty': max(1 - sum_neighbors, 0) for selected free stations
            'feasible': True if budget_violation == 0 and num_isolated == 0
            'total_selected_new': int, actual number of new stations selected
    """
    # Budget: count selected free stations
    selected_free = [i for i in free_indices if x_full[i] == 1]
    sum_selected = len(selected_free)
    budget_violation = abs(sum_selected - K_new)

    # Connectivity: check each selected free station
    isolated_count = 0
    max_isolated_penalty = 0.0
    isolated_indices = []
    
    for i in selected_free:
        # Count selected neighbors (free + fixed)
        selected_neighbors = 0
        for j in neighbors.get(i, []):
            if x_full[j] == 1:
                selected_neighbors += 1
        
        if selected_neighbors == 0:
            isolated_count += 1
            isolated_indices.append(i)
            max_isolated_penalty = max(max_isolated_penalty, 1.0)
        else:
            # Penalty term: x_i * (1 - sum_neighbors)
            # If sum_neighbors >= 1, penalty is <= 0 (reward for clustering)
            # We report the actual penalty value
            penalty = 1.0 - selected_neighbors
            max_isolated_penalty = max(max_isolated_penalty, penalty)

    return {
        "budget_violation": budget_violation,
        "num_isolated": isolated_count,
        "max_isolated_penalty": max_isolated_penalty,
        "feasible": (budget_violation == 0 and isolated_count == 0),
        "total_selected_new": sum_selected,
        "isolated_indices": isolated_indices,
    }


# -----------------------------------------------------------------------------
# INTERNAL UTILITIES
# -----------------------------------------------------------------------------


def _build_full_solution(
    sample: Union[Dict[int, int], List[int], np.ndarray],
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
) -> np.ndarray:
    """
    Convert solver sample (dict or list) to full binary vector length N_total.

    Args:
        sample: Sample from solver. Can be:
            - dict: {index: value} (OpenJij .first.sample)
            - list: [value, value, ...] (OpenJij .samples[0])
            - np.ndarray: array of values
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        N_total: Total number of candidates.

    Returns:
        np.ndarray of length N_total with 1/0 values.
    """
    x_full = np.zeros(N_total, dtype=int)

    # Fixed stations are always selected
    for m in M_indices:
        x_full[m] = 1

    # Handle different sample types
    if isinstance(sample, dict):
        # OpenJij .first.sample returns {index: value}
        for idx, val in sample.items():
            # The index is the original variable index (not position in free_indices)
            if idx < N_total:
                x_full[idx] = int(round(val))
            else:
                warnings.warn(f"Sample index {idx} out of range (N_total={N_total})")
    elif isinstance(sample, (list, np.ndarray)):
        # List/array in same order as free_indices
        for pos, val in enumerate(sample):
            if pos < len(free_indices):
                idx = free_indices[pos]
                x_full[idx] = int(round(val))
            else:
                warnings.warn(f"Sample position {pos} out of range (len(free_indices)={len(free_indices)})")
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    return x_full


def _compute_qubo_energy(
    x_full: np.ndarray,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
) -> float:
    """
    Compute QUBO energy from full binary vector.

    Energy = constant + Σ_i h[i] * x[i] + Σ_{i<j} J[(i,j)] * x[i] * x[j]

    Args:
        x_full: Binary vector of length N_total.
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term.

    Returns:
        Energy (float).
    """
    energy = constant

    # Linear terms
    for i, coeff in h.items():
        if i < len(x_full):
            energy += coeff * x_full[i]
        else:
            warnings.warn(f"Linear index {i} out of range (len(x_full)={len(x_full)})")

    # Quadratic terms
    for (i, j), coeff in J.items():
        if i < len(x_full) and j < len(x_full):
            energy += coeff * x_full[i] * x_full[j]
        else:
            warnings.warn(f"Quadratic index ({i},{j}) out of range (len(x_full)={len(x_full)})")

    return energy


def _compute_miqp_energy(
    x_full: np.ndarray,
    pairwise_data: Dict,
) -> float:
    """
    Compute MIQP energy (no penalties) from full binary vector.

    This is the original objective without QUBO penalties.
    Used for fair SQR comparison across solvers.

    Args:
        x_full: Binary vector of length N_total.
        pairwise_data: Output from compute_pairwise_terms().

    Returns:
        MIQP energy (float).
    """
    if pairwise_data is None:
        raise ValueError("pairwise_data is required for MIQP energy computation")
    
    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]

    energy = 0.0

    # Linear terms (only free variables have linear coefficients)
    for i, coeff in linear.items():
        if i < len(x_full):
            energy += coeff * x_full[i]

    # Quadratic terms (only free-free pairs)
    for (i, j), coeff in quad.items():
        if i < len(x_full) and j < len(x_full):
            energy += coeff * x_full[i] * x_full[j]

    return energy


def _build_openjij_dict(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
) -> Dict[Tuple[int, int], float]:
    """
    Build QUBO dict for OpenJij from (h, J, constant).

    OpenJij expects:
        Q[(i, i)] = h[i]        # linear terms
        Q[(i, j)] = J[(i, j)]   # quadratic terms (i < j)

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term (ignored by OpenJij).

    Returns:
        QUBO dict for OpenJij.
    """
    Q = {}

    # Linear terms
    for i, coeff in h.items():
        Q[(i, i)] = Q.get((i, i), 0.0) + coeff

    # Quadratic terms (already i < j)
    for (i, j), coeff in J.items():
        Q[(i, j)] = Q.get((i, j), 0.0) + coeff

    return Q


# -----------------------------------------------------------------------------
# SOLVER: GREEDY
# -----------------------------------------------------------------------------

def solve_greedy(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    K_new: int,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
    pairwise_data: Dict,
    mode: str = "marginal",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Greedy selection of K_new stations.

    Two modes:
        - "simple": Select K_new sites with largest -h[i] (highest utility).
        - "marginal": Select sites with largest marginal energy decrease,
          accounting for quadratic terms (redundancy, wake, connectivity penalties).

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term (included in QUBO energy).
        K_new: Number of new stations to select.
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        N_total: Total number of candidates.
        pairwise_data: Output from compute_pairwise_terms() (for MIQP energy).
        mode: "simple" or "marginal".
        seed: Random seed (for tie-breaking in marginal mode).
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "solution": np.ndarray of length N_total (binary).
            - "qubo_energy": float (with penalties).
            - "miqp_energy": float (without penalties, for SQR).
            - "time": float (wall-clock seconds).
            - "solver": "Greedy".
            - "status": "OPTIMAL", "FEASIBLE", or "FAILED".
            - "violations": dict from compute_violations().
            - "details": dict with mode, selected_indices.
    """
    start_time = time.time()

    if verbose:
        print(f"\n[Greedy] Mode: {mode}")
        print(f"[Greedy] Selecting {K_new} stations from {len(free_indices)} candidates")

    # Simple mode: select top K by linear coefficient
    if mode == "simple":
        # Sort free indices by -h[i] (descending utility)
        sorted_indices = sorted(free_indices, key=lambda i: -h.get(i, 0.0))
        selected_new = sorted_indices[:K_new]

        if verbose:
            selected_vals = [h.get(i, 0.0) for i in selected_new]
            print(f"[Greedy] Selected: {selected_new}")
            print(f"[Greedy] Selected h values: {[f'{v:.4f}' for v in selected_vals]}")

    # Marginal mode: greedily add site with best marginal gain
    elif mode == "marginal":
        # Use seed for tie-breaking
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        selected_new = []
        # Start with existing stations as selected
        selected_set = set(M_indices)

        for step in range(K_new):
            best_delta = float("inf")
            best_candidate = None
            candidates = []

            for i in free_indices:
                if i in selected_set:
                    continue

                # Compute marginal change if we add i
                # ΔE = h[i] + Σ_{j in selected} (J[(i,j)] + J[(j,i)])
                # Since J is stored with i < j, we need to check both orders
                delta = h.get(i, 0.0)

                for j in selected_set:
                    if i < j:
                        delta += J.get((i, j), 0.0)
                    else:
                        delta += J.get((j, i), 0.0)

                candidates.append((delta, i))

                if delta < best_delta - 1e-12:
                    best_delta = delta
                    best_candidate = i

            # Tie-breaking: if multiple candidates have same delta, choose randomly
            if best_candidate is not None:
                # Check for ties within tolerance
                tied = [(d, idx) for d, idx in candidates if abs(d - best_delta) < 1e-10]
                if len(tied) > 1:
                    # Random tie-break
                    _, best_candidate = tied[rng.randint(len(tied))]
                    if verbose:
                        print(f"[Greedy] Step {step+1}: Tie among {len(tied)} candidates, randomly selected {best_candidate}")

            if best_candidate is None:
                if verbose:
                    print(f"[Greedy] Warning: No candidate found at step {step+1}")
                break

            selected_new.append(best_candidate)
            selected_set.add(best_candidate)

            if verbose:
                print(f"[Greedy] Step {step+1}: Selected {best_candidate}, ΔE={best_delta:.6f}")

        if verbose:
            print(f"[Greedy] Final selected: {selected_new}")

    else:
        raise ValueError(f"mode must be 'simple' or 'marginal', got '{mode}'")

    # Build full solution vector
    x_full = np.zeros(N_total, dtype=int)
    for m in M_indices:
        x_full[m] = 1
    for i in selected_new:
        x_full[i] = 1

    # Compute energies
    qubo_energy = _compute_qubo_energy(x_full, h, J, constant)
    miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

    # Compute violations
    neighbors = pairwise_data.get("neighbors", {})
    violations = compute_violations(
        x_full=x_full,
        free_indices=free_indices,
        M_indices=M_indices,
        neighbors=neighbors,
        K_new=K_new,
    )

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"[Greedy] QUBO energy: {qubo_energy:.8f}")
        print(f"[Greedy] MIQP energy: {miqp_energy:.8f}")
        print(f"[Greedy] Time: {elapsed_time:.4f}s")
        print(f"[Greedy] Violations: feasible={violations['feasible']}, "
              f"budget_err={violations['budget_violation']}, "
              f"isolated={violations['num_isolated']}")

    return {
        "solution": x_full,
        "qubo_energy": qubo_energy,
        "miqp_energy": miqp_energy,
        "time": elapsed_time,
        "solver": "Greedy",
        "status": "FEASIBLE" if len(selected_new) == K_new else "FAILED",
        "violations": violations,
        "details": {
            "mode": mode,
            "selected_indices": selected_new,
            "K_new_actual": len(selected_new),
        },
    }


# -----------------------------------------------------------------------------
# SOLVER: SIMULATED ANNEALING (SA) with return_all support
# -----------------------------------------------------------------------------

def solve_sa(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    K_new: int,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
    pairwise_data: Dict,
    num_reads: int = 100,
    schedule: Optional[List] = None,
    seed: Optional[int] = None,
    return_all: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve QUBO using OpenJij Simulated Annealing (SA).

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term (included in QUBO energy).
        K_new: Number of new stations (for validation only).
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        N_total: Total number of candidates.
        pairwise_data: Output from compute_pairwise_terms() (for MIQP energy).
        num_reads: Number of annealing runs.
        schedule: Optional custom annealing schedule as list of [beta, steps] or (beta, steps)
                  pairs, where beta = inverse temperature. If provided, num_sweeps is ignored.
                  Example: [[0.1, 10], [1.0, 20], [5.0, 20], [10.0, 10]]
        seed: Random seed for reproducibility.
        return_all: If True, also return a list of all samples with their MIQP energies
                    and violations (useful for feasibility rate computation).
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "solution": np.ndarray of length N_total (binary) of the best feasible solution.
            - "qubo_energy": float (with penalties) of the best solution.
            - "miqp_energy": float (without penalties) of the best solution.
            - "time": float (wall-clock seconds).
            - "solver": "SA".
            - "status": "OPTIMAL", "FEASIBLE", or "FAILED".
            - "violations": dict from compute_violations() for the best solution.
            - "details": dict with num_reads, schedule, seed, response_info.
            - If return_all=True: "all_samples": List[Dict] with keys:
                "solution", "miqp_energy", "violations", "qubo_energy", "is_feasible"
    """
    try:
        import openjij as oj
    except ImportError as e:
        raise ImportError(f"OpenJij is required for SA solver: {e}")

    start_time = time.time()

    if verbose:
        print(f"\n[SA] num_reads={num_reads}")
        if schedule is not None:
            print(f"[SA] Custom schedule: {schedule}")
        else:
            print(f"[SA] Warning: No schedule provided; using OpenJij default.")
        if seed is not None:
            print(f"[SA] Seed: {seed}")
        if return_all:
            print(f"[SA] return_all=True: will return all samples.")

    # Build OpenJij QUBO dict
    Q = _build_openjij_dict(h, J, constant)

    # Run SA
    try:
        sampler = oj.SASampler()
        if schedule is not None:
            # Convert to list of tuples (beta, steps)
            schedule_converted = [(float(beta), int(steps)) for beta, steps in schedule]
            response = sampler.sample_qubo(
                Q,
                num_reads=num_reads,
                schedule=schedule_converted,
                seed=seed,
            )
        else:
            # Use default schedule
            response = sampler.sample_qubo(
                Q,
                num_reads=num_reads,
                seed=seed,
            )
    except Exception as e:
        if verbose:
            print(f"[SA] ERROR: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": "SA",
            "status": "FAILED",
            "violations": {
                "budget_violation": float("inf"),
                "num_isolated": float("inf"),
                "max_isolated_penalty": float("inf"),
                "feasible": False,
                "total_selected_new": 0,
                "isolated_indices": [],
            },
            "details": {"error": str(e)},
        }

    # --- Process all samples ---
    all_samples = []
    best_feasible_miqp = float('inf')
    best_feasible_solution = None
    best_feasible_violations = None
    best_feasible_qubo = float('inf')

    # Neighbors for violations
    neighbors = pairwise_data.get("neighbors", {})

    for idx in range(response.record.shape[0]):
        sample_arr = response.record['sample'][idx]
        # Build full solution
        x_full = np.zeros(N_total, dtype=int)
        for m in M_indices:
            x_full[m] = 1
        for var_idx, val in zip(response.indices, sample_arr):
            if var_idx < N_total:
                x_full[var_idx] = int(round(val))

        # Compute violations and MIQP energy
        viol = compute_violations(x_full, free_indices, M_indices, neighbors, K_new)
        miqp = _compute_miqp_energy(x_full, pairwise_data)
        qubo = _compute_qubo_energy(x_full, h, J, constant)

        sample_info = {
            "solution": x_full.copy(),
            "miqp_energy": miqp,
            "qubo_energy": qubo,
            "violations": viol,
            "is_feasible": viol['feasible'],
        }
        all_samples.append(sample_info)

        # Track best feasible solution
        if viol['feasible'] and miqp < best_feasible_miqp:
            best_feasible_miqp = miqp
            best_feasible_solution = x_full.copy()
            best_feasible_violations = viol
            best_feasible_qubo = qubo

    # If no feasible solution, use the best overall (might be infeasible)
    if best_feasible_solution is None:
        # Fallback: use the sample with lowest MIQP (could be infeasible)
        # Find the sample with minimum miqp
        min_miqp_idx = min(range(len(all_samples)), key=lambda i: all_samples[i]['miqp_energy'])
        best_sample = all_samples[min_miqp_idx]
        best_feasible_solution = best_sample["solution"]
        best_feasible_miqp = best_sample["miqp_energy"]
        best_feasible_violations = best_sample["violations"]
        best_feasible_qubo = best_sample["qubo_energy"]
        if verbose:
            print("[SA] No feasible solution found; returning best overall (infeasible).")

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"[SA] Best MIQP energy: {best_feasible_miqp:.8f}")
        print(f"[SA] Time: {elapsed_time:.4f}s")
        print(f"[SA] Violations: feasible={best_feasible_violations['feasible']}, "
              f"budget_err={best_feasible_violations['budget_violation']}, "
              f"isolated={best_feasible_violations['num_isolated']}")

    result = {
        "solution": best_feasible_solution,
        "qubo_energy": best_feasible_qubo,
        "miqp_energy": best_feasible_miqp,
        "time": elapsed_time,
        "solver": "SA",
        "status": "OPTIMAL" if best_feasible_violations['feasible'] else "FEASIBLE" if best_feasible_solution is not None else "FAILED",
        "violations": best_feasible_violations,
        "details": {
            "num_reads": num_reads,
            "schedule": schedule,
            "seed": seed,
            "response_info": getattr(response, "info", {}),
            "feasible_count": sum(1 for s in all_samples if s['is_feasible']),
            "total_samples": len(all_samples),
        },
    }

    if return_all:
        result["all_samples"] = all_samples

    return result


# -----------------------------------------------------------------------------
# SOLVER: SIMULATED QUANTUM ANNEALING (SQA)
# -----------------------------------------------------------------------------

def solve_sqa(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    K_new: int,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
    pairwise_data: Dict,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    trotter: int = 32,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve QUBO using OpenJij Simulated Quantum Annealing (SQA).

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term (included in QUBO energy).
        K_new: Number of new stations (for validation only).
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        N_total: Total number of candidates.
        pairwise_data: Output from compute_pairwise_terms() (for MIQP energy).
        num_reads: Number of annealing runs.
        num_sweeps: Number of sweeps per run.
        trotter: Number of Trotter slices (quantum replica dimension).
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "solution": np.ndarray of length N_total (binary).
            - "qubo_energy": float (with penalties).
            - "miqp_energy": float (without penalties, for SQR).
            - "time": float (wall-clock seconds).
            - "solver": "SQA".
            - "status": "OPTIMAL", "FEASIBLE", or "FAILED".
            - "violations": dict from compute_violations().
            - "details": dict with num_reads, num_sweeps, trotter, response_info.
    """
    try:
        import openjij as oj
    except ImportError as e:
        raise ImportError(f"OpenJij is required for SQA solver: {e}")

    start_time = time.time()

    if verbose:
        print(f"\n[SQA] num_reads={num_reads}, num_sweeps={num_sweeps}, trotter={trotter}")
        if seed is not None:
            print(f"[SQA] Seed: {seed}")

    # Build OpenJij QUBO dict
    Q = _build_openjij_dict(h, J, constant)

    # Run SQA
    try:
        sampler = oj.SQASampler()
        response = sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            seed=seed,
        )
    except TypeError as e:
        # If 'trotter' is not accepted directly, try as **parameters
        if verbose:
            print(f"[SQA] 'trotter' as direct parameter failed: {e}")
            print("[SQA] Trying with **parameters...")
        try:
            sampler = oj.SQASampler()
            response = sampler.sample_qubo(
                Q,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                **{"trotter": trotter, "seed": seed},
            )
        except Exception as e2:
            if verbose:
                print(f"[SQA] ERROR: {e2}")
                import traceback
                traceback.print_exc()
            return {
                "solution": np.zeros(N_total, dtype=int),
                "qubo_energy": float("inf"),
                "miqp_energy": float("inf"),
                "time": time.time() - start_time,
                "solver": "SQA",
                "status": "FAILED",
                "violations": {
                    "budget_violation": float("inf"),
                    "num_isolated": float("inf"),
                    "max_isolated_penalty": float("inf"),
                    "feasible": False,
                    "total_selected_new": 0,
                    "isolated_indices": [],
                },
                "details": {"error": str(e2)},
            }
    except Exception as e:
        if verbose:
            print(f"[SQA] ERROR: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": "SQA",
            "status": "FAILED",
            "violations": {
                "budget_violation": float("inf"),
                "num_isolated": float("inf"),
                "max_isolated_penalty": float("inf"),
                "feasible": False,
                "total_selected_new": 0,
                "isolated_indices": [],
            },
            "details": {"error": str(e)},
        }

    # Extract best sample
    try:
        best_sample = response.first.sample
        best_energy = response.first.energy

        if verbose:
            print(f"[SQA] Best energy: {best_energy:.8f}")
            print(f"[SQA] Best sample: {best_sample}")

        # Build full solution
        x_full = _build_full_solution(best_sample, free_indices, M_indices, N_total)

        # Validate energy
        computed_energy = _compute_qubo_energy(x_full, h, J, constant)
        energy_diff = abs(computed_energy - best_energy)

        if energy_diff > 1e-6:
            warnings.warn(
                f"[SQA] Energy mismatch: computed={computed_energy:.8f}, "
                f"reported={best_energy:.8f}, diff={energy_diff:.2e}"
            )

        # Compute MIQP energy (without penalties)
        miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

        # Compute violations
        neighbors = pairwise_data.get("neighbors", {})
        violations = compute_violations(
            x_full=x_full,
            free_indices=free_indices,
            M_indices=M_indices,
            neighbors=neighbors,
            K_new=K_new,
        )

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"[SQA] Computed QUBO energy: {computed_energy:.8f}")
            print(f"[SQA] MIQP energy: {miqp_energy:.8f}")
            print(f"[SQA] Time: {elapsed_time:.4f}s")
            print(f"[SQA] Violations: feasible={violations['feasible']}, "
                  f"budget_err={violations['budget_violation']}, "
                  f"isolated={violations['num_isolated']}")

        return {
            "solution": x_full,
            "qubo_energy": computed_energy,
            "miqp_energy": miqp_energy,
            "time": elapsed_time,
            "solver": "SQA",
            "status": "OPTIMAL",
            "violations": violations,
            "details": {
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "trotter": trotter,
                "seed": seed,
                "response_info": getattr(response, "info", {}),
            },
        }

    except Exception as e:
        if verbose:
            print(f"[SQA] Extraction error: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": "SQA",
            "status": "FAILED",
            "violations": {
                "budget_violation": float("inf"),
                "num_isolated": float("inf"),
                "max_isolated_penalty": float("inf"),
                "feasible": False,
                "total_selected_new": 0,
                "isolated_indices": [],
            },
            "details": {"error": str(e)},
        }


# -----------------------------------------------------------------------------
# ORCHESTRATOR: RUN ALL SOLVERS
# -----------------------------------------------------------------------------

def solve_all_classical(
    pairwise_data: Dict,
    K_new: int,
    lambda1: float,
    lambda2: float,
    greedy_mode: str = "marginal",
    num_reads: int = 100,
    num_sweeps: int = 1000,
    trotter: int = 32,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Orchestrator: Build QUBO once, run all classical solvers, return results.

    Args:
        pairwise_data: Output from compute_pairwise_terms().
        K_new: Number of new stations to select.
        lambda1: Budget penalty coefficient.
        lambda2: Connectivity penalty coefficient.
        greedy_mode: "simple" or "marginal".
        num_reads: Number of runs for SA/SQA.
        num_sweeps: Number of sweeps for SA/SQA.
        trotter: Trotter slices for SQA.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Dict with keys "Greedy", "SA", "SQA", each containing the result dict.
    """
    # Import here to avoid circular imports
    from src.model import build_qubo

    if verbose:
        print("=" * 60)
        print("SOLVING ALL CLASSICAL SOLVERS")
        print("=" * 60)
        print(f"K_new={K_new}, lambda1={lambda1}, lambda2={lambda2}")
        print(f"Seed: {seed}")
        print("=" * 60)

    # Build QUBO once
    if verbose:
        print("\n[Build QUBO]")
    qubo_result = build_qubo(
        pairwise_data=pairwise_data,
        K_new=K_new,
        lambda1=lambda1,
        lambda2=lambda2,
        use_jijmodeling=False,
        verbose=verbose,
    )

    h = qubo_result["h"]
    J = qubo_result["J"]
    constant = qubo_result["constant"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]
    N_total = pairwise_data["N_total"]

    if verbose:
        print(f"  h: {len(h)} terms")
        print(f"  J: {len(J)} terms")
        print(f"  constant: {constant:.4f}")

    # Run Greedy
    if verbose:
        print("\n" + "=" * 60)
    greedy_result = solve_greedy(
        h=h,
        J=J,
        constant=constant,
        K_new=K_new,
        free_indices=free_indices,
        M_indices=M_indices,
        N_total=N_total,
        pairwise_data=pairwise_data,
        mode=greedy_mode,
        seed=seed,
        verbose=verbose,
    )

    # Build a default schedule for SA if not provided
    # Use a simple geometric schedule: 100 steps, total sweeps = num_sweeps
    # Here we use num_sweeps as total sweeps
    schedule = build_schedule(0.01, 40.0, num_sweeps, 120, 1.8)

    # Run SA
    if verbose:
        print("\n" + "=" * 60)
    sa_result = solve_sa(
        h=h,
        J=J,
        constant=constant,
        K_new=K_new,
        free_indices=free_indices,
        M_indices=M_indices,
        N_total=N_total,
        pairwise_data=pairwise_data,
        num_reads=num_reads,
        schedule=schedule,
        seed=seed,
        verbose=verbose,
    )

    # Run SQA
    if verbose:
        print("\n" + "=" * 60)
    sqa_result = solve_sqa(
        h=h,
        J=J,
        constant=constant,
        K_new=K_new,
        free_indices=free_indices,
        M_indices=M_indices,
        N_total=N_total,
        pairwise_data=pairwise_data,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        trotter=trotter,
        seed=seed,
        verbose=verbose,
    )

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SOLVER SUMMARY")
        print("=" * 60)
        for name, result in [("Greedy", greedy_result), ("SA", sa_result), ("SQA", sqa_result)]:
            status = result["status"]
            if status == "FAILED":
                status_display = f"❌ {status}"
            elif status == "OPTIMAL":
                status_display = f"✅ {status}"
            else:
                status_display = f"⚠️ {status}"
            
            # Show violation summary
            v = result.get("violations", {})
            feasible = v.get("feasible", False)
            feasible_str = "✅" if feasible else "❌"
            budget_err = v.get("budget_violation", "?")
            isolated = v.get("num_isolated", "?")
            
            print(f"{name:8s} | {status_display:15s} | QUBO: {result['qubo_energy']:10.6f} | MIQP: {result['miqp_energy']:10.6f} | Time: {result['time']:.4f}s | Feasible: {feasible_str} | BudgetErr: {budget_err} | Isolated: {isolated}")

    return {
        "Greedy": greedy_result,
        "SA": sa_result,
        "SQA": sqa_result,
    }


# -----------------------------------------------------------------------------
# MODULE EXPORTS
# -----------------------------------------------------------------------------

__all__ = [
    "build_schedule",
    "compute_violations",
    "solve_greedy",
    "solve_sa",
    "solve_sqa",
    "solve_all_classical",
]