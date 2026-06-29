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
# QAOA SOLVER (PHASE 3.5) – FIXED VERSION
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
    """
    M_set = set(M_indices)
    free_set = set(free_indices)

    const_proj = constant
    h_proj = {}
    for i in free_indices:
        h_proj[i] = h.get(i, 0.0)

    # Absorb fixed stations
    for m in M_indices:
        const_proj += h.get(m, 0.0)
        for i in free_indices:
            if m < i:
                j_coeff = J.get((m, i), 0.0)
            else:
                j_coeff = J.get((i, m), 0.0)
            if j_coeff != 0:
                h_proj[i] = h_proj.get(i, 0.0) + j_coeff

    # Fixed-fixed quadratic terms to constant
    for idx_m, m in enumerate(M_indices):
        for n in M_indices[idx_m + 1:]:
            if m < n:
                const_proj += J.get((m, n), 0.0)
            else:
                const_proj += J.get((n, m), 0.0)

    # Quadratic terms (only free-free)
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
    Convert QUBO to Ising Hamiltonian. Returns (SparsePauliOp, ising_const).
    The ising_const is returned as 0.0; the constant is not included in the
    Hamiltonian to keep the landscape well-conditioned.
    """
    from qiskit.quantum_info import SparsePauliOp

    # We intentionally do NOT add the constant term to the Hamiltonian.
    # The constant will be added back later when computing the final energy.
    pauli_list = []

    # Linear terms: h_i * (1 - Z_i)/2
    for i, coeff in h.items():
        if i >= n_qubits:
            continue
        z_list = ['I'] * n_qubits
        z_list[i] = 'Z'
        pauli_list.append((''.join(z_list), -coeff / 2.0))

    # Quadratic terms: J_ij * (1 - Z_i)(1 - Z_j)/4
    for (i, j), coeff in J.items():
        if i >= n_qubits or j >= n_qubits:
            continue
        # Z_i
        z_i_list = ['I'] * n_qubits
        z_i_list[i] = 'Z'
        pauli_list.append((''.join(z_i_list), -coeff / 4.0))
        # Z_j
        z_j_list = ['I'] * n_qubits
        z_j_list[j] = 'Z'
        pauli_list.append((''.join(z_j_list), -coeff / 4.0))
        # Z_i Z_j
        z_ij_list = ['I'] * n_qubits
        z_ij_list[i] = 'Z'
        z_ij_list[j] = 'Z'
        pauli_list.append((''.join(z_ij_list), coeff / 4.0))

    # Combine terms with same Pauli string
    combined = {}
    for pauli_str, coeff in pauli_list:
        combined[pauli_str] = combined.get(pauli_str, 0.0) + coeff

    # Build SparsePauliOp
    if combined:
        pauli_strings = list(combined.keys())
        coeffs = list(combined.values())
        ham = SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))
    else:
        ham = SparsePauliOp.from_list([('I' * n_qubits, 0.0)])

    # Return ising_const = 0.0 (we'll add constant later)
    return ham, 0.0


def _qaoa_objective(
    params: np.ndarray,
    qaoa: Any,
    ham: Any,
) -> float:
    """Compute QAOA expectation value (without constant)."""
    from qiskit.quantum_info import Statevector
    circuit = qaoa.assign_parameters(params)
    state = Statevector.from_instruction(circuit)
    return state.expectation_value(ham).real


def _qaoa_state_to_binary(
    state: Any,
    n_qubits: int,
    threshold: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    probs = state.probabilities()
    p1 = np.zeros(n_qubits)
    for state_int, prob in enumerate(probs):
        for q in range(n_qubits):
            if (state_int >> q) & 1:
                p1[q] += prob
    if verbose:
        print(f"    Marginal P(1): {p1}")
    return (p1 >= threshold).astype(int)


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
    warm_start: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve QUBO using QAOA with Statevector exact simulation.

    Args:
        ... (same as before)
        warm_start: Optional binary array for free variables (length n_qubits)
                    to initialize QAOA parameters from (not used in this version).
    """
    try:
        from qiskit.quantum_info import Statevector
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
            "details": {"error": f"Missing import: {e}", "p": p},
        }

    start_time = time.time()
    if verbose:
        print(f"\n[QAOA] p={p}, maxiter={maxiter}")
        if seed is not None:
            print(f"[QAOA] Seed: {seed}")

    # Project to free variables
    h_proj, J_proj, const_proj = _project_qubo_to_free_variables(
        h, J, constant, M_indices, free_indices
    )
    n_qubits = len(free_indices)

    if n_qubits == 0:
        return {"solution": np.zeros(N_total, dtype=int), "qubo_energy": float("inf"),
                "miqp_energy": float("inf"), "time": 0.0,
                "solver": f"QAOA(p={p})", "status": "FAILED",
                "details": {"error": "No free variables.", "p": p}}

    if maxiter is None:
        maxiter = 100 + (p - 1) * 80

    if verbose:
        print(f"[QAOA] n_qubits={n_qubits}, maxiter={maxiter}")

    if n_qubits > 20:
        warnings.warn(f"[QAOA] n_qubits={n_qubits} > 20. Statevector simulation may be slow.")

    if seed is not None:
        np.random.seed(seed)

    # Build Hamiltonian (constant removed)
    ham, _ = _qubo_to_ising(h_proj, J_proj, 0.0, n_qubits)
    if verbose:
        print(f"[QAOA] Hamiltonian: {len(ham.paulis)} Pauli terms")

    # Build QAOAAnsatz
    qaoa = QAOAAnsatz(cost_operator=ham, reps=p)
    if verbose:
        print(f"[QAOA] QAOAAnsatz: {qaoa.num_parameters} params, {qaoa.num_qubits} qubits")

    # Initial parameters (random)
    initial_params = np.random.randn(qaoa.num_parameters) * 0.1
    if verbose:
        print(f"[QAOA] Initial params: {initial_params}")

    # Objective (without constant)
    def objective(params):
        return _qaoa_objective(params, qaoa, ham)

    # Warm start: if warm_start is provided, we could set initial params accordingly
    # (For simplicity, we keep random initialization; but we'll add support if needed.)

    # Optimize
    try:
        result = minimize(objective, initial_params, method='COBYLA',
                          options={'maxiter': maxiter, 'tol': 1e-6})
        best_params = result.x
        best_obj = result.fun
        nfev = result.nfev
        success = result.success
        if verbose:
            print(f"[QAOA] COBYLA finished: success={success}, nfev={nfev}, final objective={best_obj:.6f}")
    except Exception as e:
        if verbose:
            print(f"[QAOA] Optimization failed: {e}")
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": str(e), "p": p},
        }

    # Extract binary solution
    try:
        best_circuit = qaoa.assign_parameters(best_params)
        best_state = Statevector.from_instruction(best_circuit)
        binary_free = _qaoa_state_to_binary(best_state, n_qubits, verbose=verbose)
        if verbose:
            print(f"[QAOA] Binary (free): {binary_free}")

        x_full = np.zeros(N_total, dtype=int)
        for m in M_indices:
            x_full[m] = 1
        for pos, idx in enumerate(free_indices):
            if pos < len(binary_free) and binary_free[pos]:
                x_full[idx] = 1

        selected_new = [i for i in range(N_total) if x_full[i] == 1 and i not in M_indices]
        if verbose:
            print(f"[QAOA] Selected new: {selected_new}")

    except Exception as e:
        if verbose:
            print(f"[QAOA] Extraction failed: {e}")
        return {
            "solution": np.zeros(N_total, dtype=int),
            "qubo_energy": float("inf"),
            "miqp_energy": float("inf"),
            "time": time.time() - start_time,
            "solver": f"QAOA(p={p})",
            "status": "FAILED",
            "details": {"error": str(e), "p": p},
        }

    # Compute final energies (full QUBO and MIQP)
    qubo_energy = _compute_qubo_energy(x_full, h, J, constant)
    miqp_energy = _compute_miqp_energy(x_full, pairwise_data)

    elapsed = time.time() - start_time
    if verbose:
        print(f"[QAOA] QUBO energy: {qubo_energy:.6f}")
        print(f"[QAOA] MIQP energy: {miqp_energy:.6f}")
        print(f"[QAOA] Time: {elapsed:.2f}s")

    return {
        "solution": x_full,
        "qubo_energy": qubo_energy,
        "miqp_energy": miqp_energy,
        "time": elapsed,
        "solver": f"QAOA(p={p})",
        "status": "FEASIBLE" if success else "FEASIBLE",
        "details": {
            "p": p,
            "maxiter": maxiter,
            "nfev": nfev,
            "success": success,
            "final_objective": best_obj,
            "seed": seed,
            "n_qubits": n_qubits,
        },
    }