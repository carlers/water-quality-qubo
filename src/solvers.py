"""
src/solvers.py

Classical solvers for the water quality monitoring QUBO problem.

This module provides:
    1. Greedy selection (simple top-K and marginal gain)
    2. Simulated Annealing (SA) via OpenJij
    3. Simulated Quantum Annealing (SQA) via OpenJij

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
    # For greedy, we don't have a sample dict, so we build directly
    x_full = np.zeros(N_total, dtype=int)
    for m in M_indices:
        x_full[m] = 1
    for i in selected_new:
        x_full[i] = 1

    # Compute energies
    qubo_energy = _compute_qubo_energy(x_full, h, J, constant)
    miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"[Greedy] QUBO energy: {qubo_energy:.8f}")
        print(f"[Greedy] MIQP energy: {miqp_energy:.8f}")
        print(f"[Greedy] Time: {elapsed_time:.4f}s")

    return {
        "solution": x_full,
        "qubo_energy": qubo_energy,
        "miqp_energy": miqp_energy,
        "time": elapsed_time,
        "solver": "Greedy",
        "status": "FEASIBLE" if len(selected_new) == K_new else "FAILED",
        "details": {
            "mode": mode,
            "selected_indices": selected_new,
            "K_new_actual": len(selected_new),
        },
    }


# -----------------------------------------------------------------------------
# SOLVER: SIMULATED ANNEALING (SA)
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
    num_sweeps: int = 1000,
    seed: Optional[int] = None,
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
        num_sweeps: Number of sweeps per run.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "solution": np.ndarray of length N_total (binary).
            - "qubo_energy": float (with penalties).
            - "miqp_energy": float (without penalties, for SQR).
            - "time": float (wall-clock seconds).
            - "solver": "SA".
            - "status": "OPTIMAL", "FEASIBLE", or "FAILED".
            - "details": dict with num_reads, num_sweeps, response_info.
    """
    try:
        import openjij as oj
    except ImportError as e:
        raise ImportError(f"OpenJij is required for SA solver: {e}")

    start_time = time.time()

    if verbose:
        print(f"\n[SA] num_reads={num_reads}, num_sweeps={num_sweeps}")
        if seed is not None:
            print(f"[SA] Seed: {seed}")

    # Build OpenJij QUBO dict
    Q = _build_openjij_dict(h, J, constant)

    # Run SA
    try:
        sampler = oj.SASampler()
        response = sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
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
            "details": {"error": str(e)},
        }

    # Extract best sample
    try:
        best_sample = response.first.sample
        best_energy = response.first.energy

        if verbose:
            print(f"[SA] Best energy: {best_energy:.8f}")
            print(f"[SA] Best sample: {best_sample}")

        # Build full solution
        x_full = _build_full_solution(best_sample, free_indices, M_indices, N_total)

        # Validate energy
        computed_energy = _compute_qubo_energy(x_full, h, J, constant)
        energy_diff = abs(computed_energy - best_energy)

        if energy_diff > 1e-6:
            warnings.warn(
                f"[SA] Energy mismatch: computed={computed_energy:.8f}, "
                f"reported={best_energy:.8f}, diff={energy_diff:.2e}"
            )

        # Compute MIQP energy (without penalties)
        miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"[SA] Computed QUBO energy: {computed_energy:.8f}")
            print(f"[SA] MIQP energy: {miqp_energy:.8f}")
            print(f"[SA] Time: {elapsed_time:.4f}s")

        return {
            "solution": x_full,
            "qubo_energy": computed_energy,
            "miqp_energy": miqp_energy,
            "time": elapsed_time,
            "solver": "SA",
            "status": "OPTIMAL",
            "details": {
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "seed": seed,
                "response_info": getattr(response, "info", {}),
            },
        }

    except Exception as e:
        if verbose:
            print(f"[SA] Extraction error: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": "SA",
            "status": "FAILED",
            "details": {"error": str(e)},
        }


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

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"[SQA] Computed QUBO energy: {computed_energy:.8f}")
            print(f"[SQA] MIQP energy: {miqp_energy:.8f}")
            print(f"[SQA] Time: {elapsed_time:.4f}s")

        return {
            "solution": x_full,
            "qubo_energy": computed_energy,
            "miqp_energy": miqp_energy,
            "time": elapsed_time,
            "solver": "SQA",
            "status": "OPTIMAL",
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
        num_sweeps=num_sweeps,
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
            print(f"{name:8s} | {status_display:15s} | QUBO: {result['qubo_energy']:10.6f} | MIQP: {result['miqp_energy']:10.6f} | Time: {result['time']:.4f}s")

    return {
        "Greedy": greedy_result,
        "SA": sa_result,
        "SQA": sqa_result,
    }

# ============================================================================
# QAOA SOLVER (PHASE 3.5)
# ============================================================================

def _project_qubo_to_free_variables(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    M_indices: List[int],
    free_indices: List[int],
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Project QUBO onto free variables only (remove fixed M_indices).

    For fixed stations m ∈ M, x_m = 1. We absorb their contributions into
    linear terms of free variables and update the constant.

    Args:
        h: Original linear coefficients {i: coeff}.
        J: Original quadratic coefficients {(i,j): coeff}.
        constant: Original constant.
        M_indices: List of fixed station indices.
        free_indices: List of free variable indices.

    Returns:
        (h_proj, J_proj, constant_proj) for free variables only.
    """
    M_set = set(M_indices)
    free_set = set(free_indices)

    # Start with original constant
    const_proj = constant

    # Add linear contributions from fixed stations
    h_proj = {}
    for i in free_indices:
        h_proj[i] = h.get(i, 0.0)

    # Add fixed station contributions to constant and free variables
    for m in M_indices:
        # Add h_m to constant (since x_m = 1)
        const_proj += h.get(m, 0.0)

        # For each free variable i, add J_{(m,i)} + J_{(i,m)} to h_i
        for i in free_indices:
            if m < i:
                j_coeff = J.get((m, i), 0.0)
            else:
                j_coeff = J.get((i, m), 0.0)
            if j_coeff != 0:
                h_proj[i] = h_proj.get(i, 0.0) + j_coeff

    # Add fixed-fixed quadratic terms to constant
    for idx_m, m in enumerate(M_indices):
        for n in M_indices[idx_m + 1:]:
            if m < n:
                const_proj += J.get((m, n), 0.0)
            else:
                const_proj += J.get((n, m), 0.0)

    # Project quadratic terms (only free-free pairs)
    J_proj = {}
    for (i, j), coeff in J.items():
        if i in free_set and j in free_set:
            J_proj[(i, j)] = coeff

    return h_proj, J_proj, const_proj


def _qubo_to_ising(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    n_qubits: int,
) -> Tuple[Any, float]:
    """
    Convert QUBO (h, J, constant) to Ising Hamiltonian (SparsePauliOp).

    Mapping: x_i = (1 - Z_i) / 2
    For QUBO: E = Σ_i h_i x_i + Σ_{i<j} J_ij x_i x_j + constant

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term.
        n_qubits: Number of qubits (variables).

    Returns:
        (SparsePauliOp, float) where float is the Ising constant offset.
    """
    from qiskit.quantum_info import SparsePauliOp

    pauli_list = []

    # Linear terms: h_i * (1 - Z_i)/2 = h_i/2 - (h_i/2) Z_i
    for i, coeff in h.items():
        if i >= n_qubits:
            continue
        z_list = ['I'] * n_qubits
        z_list[i] = 'Z'
        pauli_list.append((''.join(z_list), -coeff / 2.0))

    # Quadratic terms: J_ij * (1 - Z_i)(1 - Z_j)/4
    # = J_ij/4 - J_ij/4 Z_i - J_ij/4 Z_j + J_ij/4 Z_i Z_j
    for (i, j), coeff in J.items():
        if i >= n_qubits or j >= n_qubits:
            continue
        # Z_i part: -coeff/4
        z_i_list = ['I'] * n_qubits
        z_i_list[i] = 'Z'
        pauli_list.append((''.join(z_i_list), -coeff / 4.0))

        # Z_j part: -coeff/4
        z_j_list = ['I'] * n_qubits
        z_j_list[j] = 'Z'
        pauli_list.append((''.join(z_j_list), -coeff / 4.0))

        # Z_i Z_j part: coeff/4
        z_ij_list = ['I'] * n_qubits
        z_ij_list[i] = 'Z'
        z_ij_list[j] = 'Z'
        pauli_list.append((''.join(z_ij_list), coeff / 4.0))

    # Combine terms with same Pauli string
    combined = {}
    for pauli_str, coeff in pauli_list:
        combined[pauli_str] = combined.get(pauli_str, 0.0) + coeff

    # Constant term: sum of all constant contributions
    # From linear: h_i/2
    # From quadratic: J_ij/4
    ising_const = constant
    for i, coeff in h.items():
        if i < n_qubits:
            ising_const += coeff / 2.0
    for (i, j), coeff in J.items():
        if i < n_qubits and j < n_qubits:
            ising_const += coeff / 4.0

    # Build SparsePauliOp
    if combined:
        pauli_strings = list(combined.keys())
        coeffs = list(combined.values())
        ham = SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))
    else:
        # No Pauli terms (all constants)
        ham = SparsePauliOp.from_list([('I' * n_qubits, 0.0)])

    return ham, ising_const


def _qaoa_objective(
    params: np.ndarray,
    qaoa: Any,
    ham: Any,
    ising_const: float,
) -> float:
    """
    Compute QAOA expectation value for given parameters.

    Args:
        params: QAOA parameters (β, γ).
        qaoa: QAOAAnsatz circuit.
        ham: Ising Hamiltonian (SparsePauliOp).
        ising_const: Ising constant offset.

    Returns:
        Expectation value (float).
    """
    from qiskit.quantum_info import Statevector

    circuit = qaoa.assign_parameters(params)
    state = Statevector.from_instruction(circuit)
    expectation = state.expectation_value(ham).real + ising_const
    return expectation


def _qaoa_state_to_binary(
    state: Any,
    n_qubits: int,
    threshold: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Extract binary solution from QAOA statevector via marginal probabilities.

    Args:
        state: Statevector.
        n_qubits: Number of qubits.
        threshold: Probability threshold for bit = 1 (default 0.5).
        verbose: Print marginal probabilities.

    Returns:
        np.ndarray of binary values (length n_qubits).
    """
    probs = state.probabilities()

    # Marginal probability for each qubit being 1
    p1 = np.zeros(n_qubits)
    for state_int, prob in enumerate(probs):
        for q in range(n_qubits):
            if (state_int >> q) & 1:
                p1[q] += prob

    if verbose:
        print(f"    Marginal P(1): {p1}")

    binary = (p1 >= threshold).astype(int)
    return binary


def solve_qaoa(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    K_new: int,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
    pairwise_data: Dict,
    p: int = 1,
    maxiter: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve QUBO using QAOA with Statevector exact simulation.

    This is the Phase 3.5 QAOA solver. It uses the Statevector approach
    for exact expectation value computation (no sampling noise).

    NOTE: QAOA operates only on free variables. Fixed stations (M_indices)
    are projected out before building the Hamiltonian.

    Args:
        h: Linear coefficients dict {i: coeff}.
        J: Quadratic coefficients dict {(i,j): coeff} with i < j.
        constant: Constant term (included in QUBO energy).
        K_new: Number of new stations (for validation only).
        free_indices: List of free variable indices.
        M_indices: List of fixed existing station indices.
        N_total: Total number of candidates.
        pairwise_data: Output from compute_pairwise_terms() (for MIQP energy).
        p: QAOA depth (number of layers).
        maxiter: Maximum iterations for COBYLA. If None, auto-set based on p.
        seed: Random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - "solution": np.ndarray of length N_total (binary).
            - "qubo_energy": float (with penalties).
            - "miqp_energy": float (without penalties, for SQR).
            - "time": float (wall-clock seconds).
            - "solver": "QAOA(p={p})".
            - "status": "OPTIMAL", "FEASIBLE", or "FAILED".
            - "details": dict with p, maxiter, nfev, success, final_params.
    """
    # -------------------------------------------------------------------------
    # 1. Import checks
    # -------------------------------------------------------------------------
    try:
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit.circuit.library import QAOAAnsatz
        from scipy.optimize import minimize
    except ImportError as e:
        if verbose:
            print(f"[QAOA] Import error: {e}")
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": 0.0,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {
                "error": f"Missing import: {e}. Install qiskit and scipy.",
                "p": p,
            },
        }

    start_time = time.time()

    if verbose:
        print(f"\n[QAOA] p={p}, maxiter={maxiter}")
        if seed is not None:
            print(f"[QAOA] Seed: {seed}")

    # -------------------------------------------------------------------------
    # 2. PROJECT QUBO TO FREE VARIABLES ONLY
    # -------------------------------------------------------------------------
    # This is CRITICAL: QAOA should only operate on free variables.
    # Fixed stations M are clamped to 1 and absorbed into the objective.
    h_proj, J_proj, const_proj = _project_qubo_to_free_variables(
        h=h,
        J=J,
        constant=constant,
        M_indices=M_indices,
        free_indices=free_indices,
    )

    # Number of qubits = number of free variables
    n_qubits = len(free_indices)

    if n_qubits == 0:
        if verbose:
            print("[QAOA] ERROR: No free variables.")
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": "No free variables.", "p": p},
        }

    if verbose:
        print(f"[QAOA] Projected QUBO: {n_qubits} free variables")
        print(f"[QAOA]   h_proj: {len(h_proj)} terms")
        print(f"[QAOA]   J_proj: {len(J_proj)} terms")
        print(f"[QAOA]   const_proj: {const_proj:.6f}")

    # Auto-set maxiter if not provided
    if maxiter is None:
        # p=1 converges in ~80-100 iterations, p=2 in ~100-150
        maxiter = 100 + (p - 1) * 80

    if verbose:
        print(f"[QAOA] n_qubits={n_qubits}, maxiter={maxiter}")

    # Check if n_qubits is too large for Statevector
    if n_qubits > 20:
        warnings.warn(
            f"[QAOA] n_qubits={n_qubits} > 20. Statevector simulation will be "
            f"extremely slow or impossible. Consider reducing N or using a sampler."
        )

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # -------------------------------------------------------------------------
    # 3. Build Ising Hamiltonian
    # -------------------------------------------------------------------------
    try:
        ham, ising_const = _qubo_to_ising(h_proj, J_proj, const_proj, n_qubits)

        if verbose:
            print(f"[QAOA] Hamiltonian built: {len(ham.paulis)} Pauli terms")
            print(f"[QAOA] Ising constant: {ising_const:.6f}")
            if verbose and len(ham.paulis) <= 10:
                for pauli, coeff in zip(ham.paulis, ham.coeffs):
                    print(f"    {pauli}: {coeff:.6f}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] Hamiltonian build failed: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": f"Hamiltonian build failed: {e}", "p": p},
        }

    # -------------------------------------------------------------------------
    # 4. Build QAOAAnsatz
    # -------------------------------------------------------------------------
    try:
        qaoa = QAOAAnsatz(cost_operator=ham, reps=p)

        if verbose:
            print(f"[QAOA] QAOAAnsatz built: {qaoa.num_parameters} parameters, {qaoa.num_qubits} qubits")
            print(f"[QAOA] Parameter names: {qaoa.parameters}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] QAOAAnsatz build failed: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": f"QAOAAnsatz build failed: {e}", "p": p},
        }

    # -------------------------------------------------------------------------
    # 5. Initialize parameters
    # -------------------------------------------------------------------------
    # Use random initialization (avoids getting stuck at zero)
    initial_params = np.random.randn(qaoa.num_parameters) * 0.1

    if verbose:
        print(f"[QAOA] Initial params: {initial_params}")

    # -------------------------------------------------------------------------
    # 6. COBYLA Optimization
    # -------------------------------------------------------------------------
    def objective(params):
        return _qaoa_objective(params, qaoa, ham, ising_const)

    # Test initial objective
    initial_obj = objective(initial_params)
    if verbose:
        print(f"[QAOA] Initial objective: {initial_obj:.8f}")

    try:
        result = minimize(
            objective,
            initial_params,
            method='COBYLA',
            options={
                'maxiter': maxiter,
                'disp': False,
                'tol': 1e-6,
            },
        )

        best_params = result.x
        best_obj = result.fun
        nfev = result.nfev
        success = result.success

        if verbose:
            print(f"[QAOA] Optimization complete")
            print(f"[QAOA]   Success: {success}")
            print(f"[QAOA]   Iterations: {nfev}")
            print(f"[QAOA]   Final objective: {best_obj:.8f}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": f"Optimization failed: {e}", "p": p},
        }

    # -------------------------------------------------------------------------
    # 7. Extract binary solution
    # -------------------------------------------------------------------------
    try:
        best_circuit = qaoa.assign_parameters(best_params)
        best_state = Statevector.from_instruction(best_circuit)

        if verbose:
            print(f"[QAOA] Statevector: {best_state}")

        # Get binary for free variables (ordered by free_indices)
        binary_free = _qaoa_state_to_binary(best_state, n_qubits, verbose=verbose)

        if verbose:
            print(f"[QAOA] Binary (free variables): {binary_free}")

        # Build full solution vector
        x_full = np.zeros(N_total, dtype=int)
        for m in M_indices:
            x_full[m] = 1

        # Map binary (ordered by free_indices) to original indices
        for pos, idx in enumerate(free_indices):
            if pos < len(binary_free):
                if binary_free[pos] == 1:
                    x_full[idx] = 1

        # Count selected
        selected_new = [i for i in range(N_total) if x_full[i] == 1 and i not in M_indices]
        if verbose:
            print(f"[QAOA] Selected new stations: {selected_new}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] Extraction failed: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": f"Extraction failed: {e}", "p": p},
        }

    # -------------------------------------------------------------------------
    # 8. Compute energies and validate
    # -------------------------------------------------------------------------
    try:
        computed_qubo_energy = _compute_qubo_energy(x_full, h, J, constant)
        miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

        # Validate against best_obj
        energy_diff = abs(computed_qubo_energy - best_obj)

        if energy_diff > 1e-4:
            warnings.warn(
                f"[QAOA] Energy mismatch: computed={computed_qubo_energy:.8f}, "
                f"optimized={best_obj:.8f}, diff={energy_diff:.2e}"
            )
        else:
            if verbose:
                print(f"[QAOA] Energy validation passed (diff={energy_diff:.2e})")

        if verbose:
            print(f"[QAOA] QUBO energy: {computed_qubo_energy:.8f}")
            print(f"[QAOA] MIQP energy: {miqp_energy:.8f}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] Energy computation failed: {e}")
            import traceback
            traceback.print_exc()
        return {
            "solution": x_full,
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": f"Energy computation failed: {e}", "p": p},
        }

    # -------------------------------------------------------------------------
    # 9. Return
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time

    if verbose:
        print(f"[QAOA] Time: {elapsed_time:.4f}s")

    return {
        "solution": x_full,
        "qubo_energy": computed_qubo_energy,
        "miqp_energy": miqp_energy,
        "time": elapsed_time,
        "solver": f"QAOA(p={p})",
        "status": "OPTIMAL" if success else "FEASIBLE",
        "details": {
            "p": p,
            "maxiter": maxiter,
            "nfev": nfev,
            "success": success,
            "initial_params": initial_params.tolist(),
            "final_params": best_params.tolist(),
            "final_objective": best_obj,
            "seed": seed,
            "n_qubits": n_qubits,
        },
    }


def solve_qaoa_all_p(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    constant: float,
    K_new: int,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int,
    pairwise_data: Dict,
    p_values: List[int] = [1, 2, 3, 4, 5],
    maxiter: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Run QAOA for multiple p values.

    This is a convenience wrapper for running QAOA across a range of depths.

    Args:
        Same as solve_qaoa, plus:
        p_values: List of p values to test.

    Returns:
        Dict mapping p -> solve_qaoa result dict.
    """
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("QAOA SWEEP: p values", p_values)
        print("=" * 60)

    for p in p_values:
        if verbose:
            print(f"\n[QAOA Sweep] Running p={p}...")
            print("-" * 40)

        result = solve_qaoa(
            h=h,
            J=J,
            constant=constant,
            K_new=K_new,
            free_indices=free_indices,
            M_indices=M_indices,
            N_total=N_total,
            pairwise_data=pairwise_data,
            p=p,
            maxiter=maxiter,
            seed=seed,
            verbose=verbose,
        )
        results[p] = result

    if verbose:
        print("\n" + "=" * 60)
        print("QAOA SWEEP SUMMARY")
        print("=" * 60)
        for p, r in results.items():
            print(f"  p={p}: MIQP={r['miqp_energy']:.6f}, status={r['status']}, time={r['time']:.4f}s")

    return results

# -----------------------------------------------------------------------------
# MODULE EXPORTS
# -----------------------------------------------------------------------------

__all__ = [
    "solve_greedy",
    "solve_sa",
    "solve_sqa",
    "solve_qaoa",
    "solve_qaoa_all_p",
    "solve_all_classical",
    "solve_all_with_qaoa",
]