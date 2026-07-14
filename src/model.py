#@title Updated model.py with Two-Step Normalization
"""
src/model.py

Core mathematical formulation and solver builders for the water quality
monitoring optimization problem.

This module provides:
    1. Pairwise term computation (redundancy + wake) and neighbor finding.
    2. MIQP builder (Gurobi) with explicit constraints for exact baseline.
    3. QUBO builder (JijModeling + transpiler, with manual fallback).
    4. Utility functions for solution extraction and energy evaluation.
    5. Two-step QUBO normalization (Lee et al. 2025 + unit variance scaling).

NEW in this version:
    - compute_objective_variance: Var(h + J) over random binary assignments
    - balance_quadratic_variance: Scale J to match Var(h) (Lee et al. 2025)
    - normalize_objective_to_unit_variance: Normalize total objective to unit variance
    - prepare_normalized_qubo: Orchestrator for both steps

All functions are designed to be:
    - Fully deterministic (seeded).
    - Extensively logged for debugging.
    - Type-hinted for clarity.
    - Self-contained with graceful import fallbacks.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial import cKDTree

# ============================================================================
# Core Pairwise Term Computation (UNCHANGED)
# ============================================================================

def compute_pairwise_terms(
    coords: np.ndarray,
    U: np.ndarray,
    M_indices: List[int],
    L_c: float,
    L_w: float,
    current_vector: Tuple[float, float],
    beta: float = 1.0,
    delta: float = 1.0,
    connectivity_range: Optional[float] = None,
    hex_side: Optional[float] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compute all pairwise terms: linear coefficients, quadratic coefficients,
    and neighbor sets.

    Args:
        coords: (N_total, 2) array of coordinates in km.
        U: (N_total,) utility scores.
        M_indices: List of indices that are fixed existing stations.
        L_c: Spatial correlation length (km).
        L_w: Wake persistence length (km).
        current_vector: (dx, dy) direction of dominant current.
        beta: Redundancy penalty weight.
        delta: Wake penalty weight.
        connectivity_range: D_max (km). If None, computed from hex_side.
        hex_side: Hexagon side length (km). Needed if connectivity_range is None.
        verbose: Print progress and statistics.

    Returns:
        Dict with keys:
            'linear': dict {i: a_i} for i in free_indices.
            'quad': dict {(i,j): b_tilde_ij} for i<j.
            'neighbors': dict {i: list_of_j} for i in free_indices.
            'N_total': int,
            'N_free': int,
            'free_indices': List[int],
            'M_indices': List[int],
            'coords': np.ndarray,
            'U': np.ndarray,
            'L_c': float,
            'L_w': float,
            'current_vector': Tuple,
            'beta': float,
            'delta': float,
            'D_max': float,
    """
    N_total = coords.shape[0]
    free_indices = [i for i in range(N_total) if i not in M_indices]
    N_free = len(free_indices)
    free_set = set(free_indices)
    M_set = set(M_indices)

    # Connectivity range
    if connectivity_range is not None:
        D_max = connectivity_range
    elif hex_side is not None:
        D_max = 2.0 * np.sqrt(3.0) * hex_side
    else:
        raise ValueError("Either connectivity_range or hex_side must be provided.")

    if verbose:
        print(f"[Pairwise] N_total={N_total}, N_free={N_free}, |M|={len(M_indices)}")
        print(f"[Pairwise] L_c={L_c}, L_w={L_w}, D_max={D_max:.3f} km")

    # Build KD-tree for efficient neighbor queries
    tree = cKDTree(coords)

    # ------------------------------------------------------------------------
    # 1. Compute all pairwise raw terms (within 2*L_c)
    # ------------------------------------------------------------------------
    raw_quad = {}  # (i, j) -> coefficient (symmetrized later)
    all_pairs = set()
    neighbor_dict = {i: [] for i in free_indices}

    # Pre-compute normalization
    v_norm = np.linalg.norm(current_vector)
    if v_norm == 0:
        raise ValueError("current_vector cannot be zero.")
    v_unit = np.array(current_vector) / v_norm

    # Query all pairs within 2*L_c
    pairs_within = tree.query_pairs(r=2.0 * L_c)

    if verbose:
        print(f"[Pairwise] Found {len(pairs_within)} pairs within 2*L_c={2*L_c} km")

    for i, j in pairs_within:
        if i == j:
            continue
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        if d == 0:
            continue

        # Redundancy (symmetric)
        R_ij = max(0.0, 1.0 - d / L_c)

        # Wake (directional, depends on flow direction)
        cos_theta = (dx * v_unit[0] + dy * v_unit[1]) / d
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Wake half-angle = 45 degrees -> cos(45°) = 0.7071
        if cos_theta > 0.7071:
            W_ij = np.exp(-d / L_w) * cos_theta
        else:
            W_ij = 0.0

        # Compute W_ji (reverse direction)
        cos_theta_ji = (-dx * v_unit[0] + -dy * v_unit[1]) / d
        cos_theta_ji = np.clip(cos_theta_ji, -1.0, 1.0)
        if cos_theta_ji > 0.7071:
            W_ji = np.exp(-d / L_w) * cos_theta_ji
        else:
            W_ji = 0.0

        # Symmetrized coupling: b̃_ij = β*R_ij + δ*(W_ij + W_ji)
        raw_quad[(i, j)] = beta * R_ij + delta * (W_ij + W_ji)

        # Also store reverse for later (we'll symmetrize)
        raw_quad[(j, i)] = raw_quad[(i, j)]

    # ------------------------------------------------------------------------
    # 2. Build neighbor sets (for connectivity)
    # ------------------------------------------------------------------------
    # Query all pairs within D_max
    neighbor_pairs = tree.query_pairs(r=D_max)
    for i, j in neighbor_pairs:
        if i in free_set and j in free_set:
            neighbor_dict[i].append(j)
            neighbor_dict[j].append(i)
        elif i in free_set and j in M_set:
            neighbor_dict[i].append(j)
        elif j in free_set and i in M_set:
            neighbor_dict[j].append(i)

    # Remove duplicates and sort
    for i in neighbor_dict:
        neighbor_dict[i] = sorted(set(neighbor_dict[i]))

    if verbose:
        avg_neighbors = np.mean([len(neighbor_dict[i]) for i in neighbor_dict])
        print(f"[Pairwise] Avg connectivity neighbors (within {D_max} km): {avg_neighbors:.2f}")

    # ------------------------------------------------------------------------
    # 3. Build linear coefficients a_i (including interactions with fixed M)
    # ------------------------------------------------------------------------
    linear = {}
    for i in free_indices:
        # Base: negative utility (we are minimizing)
        val = -U[i]

        # Add interactions with fixed stations M
        for m in M_indices:
            key = (i, m) if i < m else (m, i)
            if key in raw_quad:
                val += raw_quad[key]
        linear[i] = val

    # ------------------------------------------------------------------------
    # 4. Build quadratic coefficients b̃_ij (only free-free pairs, i<j)
    # ------------------------------------------------------------------------
    quad = {}
    for i in free_indices:
        for j in free_indices:
            if i < j:
                key = (i, j)
                if key in raw_quad:
                    quad[key] = raw_quad[key]

    if verbose:
        print(f"[Pairwise] Linear terms: {len(linear)}")
        print(f"[Pairwise] Quadratic terms: {len(quad)}")
        if linear:
            vals = list(linear.values())
            print(f"  Linear: min={min(vals):.4f}, max={max(vals):.4f}")
        if quad:
            vals = list(quad.values())
            print(f"  Quad: min={min(vals):.4f}, max={max(vals):.4f}")

    # ------------------------------------------------------------------------
    # 5. Package and return
    # ------------------------------------------------------------------------
    return {
        "linear": linear,
        "quad": quad,
        "neighbors": neighbor_dict,
        "N_total": N_total,
        "N_free": N_free,
        "free_indices": free_indices,
        "M_indices": M_indices,
        "coords": coords,
        "U": U,
        "L_c": L_c,
        "L_w": L_w,
        "current_vector": current_vector,
        "beta": beta,
        "delta": delta,
        "D_max": D_max,
    }


# ============================================================================
# NEW: Two-Step QUBO Normalization (Lee et al. 2025 + Unit Variance)
# ============================================================================

def compute_objective_variance(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    n_samples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute variance of total objective energy E = Σ h_i x_i + Σ J_ij x_i x_j
    over random binary assignments.

    Args:
        h: dict {i: coeff} for linear terms
        J: dict {(i,j): coeff} for quadratic terms (i < j)
        n_samples: number of random assignments
        seed: random seed

    Returns:
        var_total: float (variance of the total objective energy)
        scale: float = sqrt(var_total)
    """
    rng = np.random.RandomState(seed)

    # Determine dimension from keys
    max_idx = max(h.keys()) if h else 0
    max_idx = max(max_idx, max([max(k) for k in J.keys()]) if J else 0)
    N = max_idx + 1

    # Generate random binary vectors
    samples = rng.randint(0, 2, size=(n_samples, N))

    # Compute energies for each sample
    energies = np.zeros(n_samples)

    # Linear terms
    for idx, coeff in h.items():
        energies += coeff * samples[:, idx]

    # Quadratic terms
    for (i, j), coeff in J.items():
        energies += coeff * samples[:, i] * samples[:, j]

    var_total = np.var(energies)
    scale = np.sqrt(var_total) if var_total > 0 else 1.0

    return var_total, scale


def balance_quadratic_variance(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    n_samples: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[Tuple[int, int], float], float, float, float, float]:
    """
    Scale J to match variance of h (Lee et al. 2025).

    Args:
        h: dict {i: coeff} for linear terms
        J: dict {(i,j): coeff} for quadratic terms (i < j)
        n_samples: number of random assignments
        seed: random seed

    Returns:
        J_balanced: dict with scaled coefficients
        j_scale: float (the scaling factor applied)
        var_h: float (original Var(h))
        var_J_original: float (original Var(J))
        var_J_balanced: float (Var(J) after scaling)
    """
    # Compute Var(h) and Var(J) separately
    var_h, _ = compute_objective_variance(h, {}, n_samples, seed)
    var_J_original, _ = compute_objective_variance({}, J, n_samples, seed)

    # Compute scaling factor
    if var_J_original > 0:
        j_scale = np.sqrt(var_h / var_J_original)
    else:
        j_scale = 1.0

    # Scale J
    J_balanced = {(i, j): val * j_scale for (i, j), val in J.items()}

    # Verify Var(J_balanced) ≈ Var(h)
    var_J_balanced, _ = compute_objective_variance({}, J_balanced, n_samples, seed)

    return J_balanced, j_scale, var_h, var_J_original, var_J_balanced


def normalize_objective_to_unit_variance(
    h: Dict[int, float],
    J_balanced: Dict[Tuple[int, int], float],
    target_variance: float = 1.0,
    n_samples: int = 10000,
    seed: int = 42,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float, float]:
    """
    Normalize h and J so that Var(h + J) = target_variance.

    Args:
        h: dict {i: coeff} for linear terms
        J_balanced: dict {(i,j): coeff} for quadratic terms (already balanced)
        target_variance: desired variance (default: 1.0)
        n_samples: number of random assignments
        seed: random seed

    Returns:
        h_norm: dict with scaled linear coefficients
        J_norm: dict with scaled quadratic coefficients
        total_scale: float (the scaling factor applied)
        var_total_original: float (Var(h + J_balanced) before normalization)
    """
    # Compute total variance
    var_total_original, scale = compute_objective_variance(h, J_balanced, n_samples, seed)

    # Compute normalization factor
    if var_total_original > 0:
        norm_scale = np.sqrt(var_total_original / target_variance)
    else:
        norm_scale = 1.0

    # Scale both h and J
    h_norm = {i: val / norm_scale for i, val in h.items()}
    J_norm = {(i, j): val / norm_scale for (i, j), val in J_balanced.items()}

    return h_norm, J_norm, norm_scale, var_total_original


def prepare_normalized_qubo(
    pairwise_data: Dict,
    target_variance: float = 1.0,
    n_samples: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], Dict]:
    """
    Prepare h and J for QUBO by:
    1. Balancing Var(h) and Var(J) (Lee et al. 2025).
    2. Normalizing total objective to target_variance.

    Args:
        pairwise_data: Output from compute_pairwise_terms()
        target_variance: Desired variance of the total objective (default: 1.0)
        n_samples: Number of random assignments for variance estimation
        seed: Random seed
        verbose: Print progress

    Returns:
        h_norm: dict {i: coeff} for linear terms (normalized)
        J_norm: dict {(i,j): coeff} for quadratic terms (balanced + normalized)
        scales: dict with intermediate values for debugging
    """
    h = pairwise_data["linear"]
    J = pairwise_data["quad"]

    if verbose:
        print("\n[prepare_normalized_qubo] Starting two-step normalization...")
        print(f"  target_variance = {target_variance}")
        print(f"  n_samples = {n_samples}")

    # Step 1: Balance Var(h) and Var(J) (Lee et al. 2025)
    J_balanced, j_scale, var_h, var_J_original, var_J_balanced = balance_quadratic_variance(
        h, J, n_samples, seed
    )

    if verbose:
        print(f"\n  Step 1: Quadratic variance balancing")
        print(f"    Var(h)           : {var_h:.6f}")
        print(f"    Var(J) original  : {var_J_original:.6f}")
        print(f"    j_scale          : {j_scale:.4f}")
        print(f"    Var(J) balanced  : {var_J_balanced:.6f}")
        print(f"    Ratio (J/h)      : {var_J_balanced / var_h:.4f}")

    # Step 2: Normalize total objective to target_variance
    h_norm, J_norm, total_scale, var_total_original = normalize_objective_to_unit_variance(
        h, J_balanced, target_variance, n_samples, seed
    )

    if verbose:
        print(f"\n  Step 2: Total objective normalization")
        print(f"    Var(h + J) original: {var_total_original:.6f}")
        print(f"    total_scale         : {total_scale:.4f}")
        print(f"    target_variance     : {target_variance:.6f}")

    # Compute final variance for verification
    var_final, _ = compute_objective_variance(h_norm, J_norm, n_samples, seed)
    if verbose:
        print(f"    Var(h_norm + J_norm): {var_final:.6f} (should be {target_variance:.6f})")

    # Build scales dict
    scales = {
        "var_h_original": var_h,
        "var_J_original": var_J_original,
        "j_scale": j_scale,
        "var_J_balanced": var_J_balanced,
        "var_total_original": var_total_original,
        "total_scale": total_scale,
        "var_total_final": var_final,
        "target_variance": target_variance,
    }

    if verbose:
        print("\n  ✅ Normalization complete.")
        print(f"    h range: {min(h_norm.values()):.4f} to {max(h_norm.values()):.4f}")
        print(f"    J range: {min(J_norm.values()):.4f} to {max(J_norm.values()):.4f}")

    return h_norm, J_norm, scales


# ============================================================================
# Gurobi MIQP Builder (UNCHANGED)
# ============================================================================

def build_miqp(
    pairwise_data: Dict,
    K_new: int,
    time_limit: float = 60.0,
    mip_gap: float = 1e-6,
    verbose: bool = True,
) -> Dict:
    """
    Build and solve (or return) a Gurobi MIQP model with explicit constraints.

    This is the exact baseline solver. It uses:
        - Binary variables x_i for free indices.
        - Objective: linear + quadratic terms (no penalties).
        - Budget constraint: sum(x_i) == K_new.
        - Connectivity constraints: x_i <= sum_{j in N_i} x_j.

    Args:
        pairwise_data: Output from compute_pairwise_terms.
        K_new: Number of new stations to select.
        time_limit: Gurobi time limit (seconds).
        mip_gap: Relative MIP gap tolerance.
        verbose: Print Gurobi log.

    Returns:
        Dict with keys:
            'model': gp.Model (solved or unsolved).
            'x_vars': dict {i: gurobi Var} for free indices.
            'free_indices': List[int],
            'M_indices': List[int],
            'N_total': int,
            'N_free': int,
            'status': str (optimization status if solved).
            'solve_time': float,
            'objective_value': float (if solved).
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        raise ImportError("Gurobi is required for build_miqp. Install gurobipy.")

    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]
    N_free = pairwise_data["N_free"]

    if N_free == 0:
        raise ValueError("No free variables. Cannot build MIQP.")

    # Create model
    model = gp.Model("MIQP_WaterQuality")
    if not verbose:
        model.setParam("OutputFlag", 0)
    else:
        model.setParam("OutputFlag", 1)

    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("NonConvex", 2)

    # Add binary variables
    x_vars = {}
    for i in free_indices:
        x_vars[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    model.update()

    # Build objective: linear + quadratic
    lin_expr = gp.quicksum(linear[i] * x_vars[i] for i in free_indices)
    quad_expr = gp.QuadExpr()
    for (i, j), coeff in quad.items():
        quad_expr += coeff * x_vars[i] * x_vars[j]

    model.setObjective(lin_expr + quad_expr, GRB.MINIMIZE)

    # Budget constraint: sum(x_i) == K_new
    model.addConstr(gp.quicksum(x_vars[i] for i in free_indices) == K_new, name="budget")

    # Connectivity constraints
    for i in free_indices:
        if i not in neighbors:
            model.addConstr(x_vars[i] == 0, name=f"conn_isolated_{i}")
            continue

        rhs_free = gp.quicksum(x_vars[j] for j in neighbors[i] if j in free_indices)
        rhs_fixed = sum(1 for j in neighbors[i] if j in M_indices)

        model.addConstr(
            x_vars[i] <= rhs_free + rhs_fixed,
            name=f"conn_{i}"
        )

    model.update()

    if verbose:
        print(f"[MIQP] Model built. Variables: {model.NumVars}, Constraints: {model.NumConstrs}")

    return {
        "model": model,
        "x_vars": x_vars,
        "free_indices": free_indices,
        "M_indices": M_indices,
        "N_total": pairwise_data["N_total"],
        "N_free": N_free,
        "status": None,
        "solve_time": None,
        "objective_value": None,
    }


# ============================================================================
# QUBO Builders (UNCHANGED)
# ============================================================================

def build_qubo_jijmodeling(
    pairwise_data: Dict,
    K_new: int,
    lambda1: float,
    lambda2: float,
    verbose: bool = True,
) -> Dict:
    """
    Build QUBO using JijModeling + transpiler.

    This constructs the QUBO symbolically and then uses
    jijmodeling_transpiler.core.pubo.transpile_to_pubo to extract
    h, J, and constant.
    """
    try:
        import jijmodeling as jm
        from jijmodeling_transpiler.core import compile_model
        from jijmodeling_transpiler.core.pubo import transpile_to_pubo
    except ImportError as e:
        warnings.warn(f"JijModeling/transpiler not available: {e}. Falling back to manual QUBO builder.")
        return build_qubo_manual(pairwise_data, K_new, lambda1, lambda2, verbose)

    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    N_free = pairwise_data["N_free"]

    if N_free == 0:
        raise ValueError("No free variables. Cannot build QUBO.")

    # Map free index -> position
    pos = {idx: p for p, idx in enumerate(free_indices)}

    a_arr = np.zeros(N_free)
    for i, coeff in linear.items():
        a_arr[pos[i]] = coeff

    b_arr = np.zeros((N_free, N_free))
    for (i, j), coeff in quad.items():
        p_i = pos[i]
        p_j = pos[j]
        b_arr[p_i, p_j] = coeff
        b_arr[p_j, p_i] = coeff

    neigh_mask = np.zeros((N_free, N_free), dtype=int)
    for i, neigh_list in neighbors.items():
        p_i = pos[i]
        for j in neigh_list:
            if j in pos:
                p_j = pos[j]
                neigh_mask[p_i, p_j] = 1

    a = jm.Placeholder("a", ndim=1, shape=(N_free,))
    b = jm.Placeholder("b", ndim=2, shape=(N_free, N_free))
    mask = jm.Placeholder("mask", ndim=2, shape=(N_free, N_free))
    K = jm.Placeholder("K", ndim=0)
    lam1 = jm.Placeholder("lambda1", ndim=0)
    lam2 = jm.Placeholder("lambda2", ndim=0)

    x = jm.Variable("x", shape=(N_free,), binary=True)

    linear_term = jm.sum(i, a[i] * x[i])
    quad_term = jm.sum([i, j], b[i, j] * x[i] * x[j], i < j)
    budget_penalty = lam1 * (jm.sum(i, x[i]) - K) ** 2
    conn_penalty = lam2 * (
        jm.sum(i, x[i]) - jm.sum([i, j], mask[i, j] * x[i] * x[j])
    )

    total_objective = linear_term + quad_term + budget_penalty + conn_penalty

    problem = jm.Problem("QUBO_WaterQuality")
    problem += jm.Objective("minimize", total_objective)

    if verbose:
        print("[QUBO_Jij] Problem defined. Compiling...")

    compiled = compile_model(problem)
    data = {
        "a": a_arr,
        "b": b_arr,
        "mask": neigh_mask,
        "K": np.array(K_new),
        "lambda1": np.array(lambda1),
        "lambda2": np.array(lambda2),
    }

    if verbose:
        print("[QUBO_Jij] Transpiling to PUBO...")

    pubo_model = transpile_to_pubo(compiled, data)

    h = {}
    J = {}
    constant = 0.0

    for term, coeff in pubo_model.pubo.items():
        if len(term) == 0:
            constant += coeff
        elif len(term) == 1:
            i = term[0]
            h[i] = h.get(i, 0.0) + coeff
        elif len(term) == 2:
            i, j = term[0], term[1]
            if i > j:
                i, j = j, i
            J[(i, j)] = J.get((i, j), 0.0) + coeff
        else:
            warnings.warn(f"Higher-order term found: {term} -> {coeff}. Ignoring.")

    if verbose:
        print(f"[QUBO_Jij] Extracted: {len(h)} linear, {len(J)} quadratic, constant={constant:.4f}")

    h_orig = {}
    for p, coeff in h.items():
        i = free_indices[p]
        h_orig[i] = coeff

    J_orig = {}
    for (p_i, p_j), coeff in J.items():
        i = free_indices[p_i]
        j = free_indices[p_j]
        J_orig[(i, j)] = coeff

    return {
        "h": h_orig,
        "J": J_orig,
        "constant": constant,
        "problem": problem,
        "status": "success",
        "method": "jijmodeling",
    }


def build_qubo_manual(
    pairwise_data: Dict,
    K_new: int,
    lambda1: float,
    lambda2: float,
    verbose: bool = True,
) -> Dict:
    """
    Manually construct QUBO (h, J, constant) without JijModeling.

    This is a fallback that expands the QUBO formula explicitly.
    """
    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]
    N_free = pairwise_data["N_free"]

    if N_free == 0:
        raise ValueError("No free variables. Cannot build QUBO.")

    h = {}
    J = {}
    constant = 0.0

    def add_linear(i, coeff):
        h[i] = h.get(i, 0.0) + coeff

    def add_quad(i, j, coeff):
        if i == j:
            add_linear(i, coeff)
            return
        if i > j:
            i, j = j, i
        J[(i, j)] = J.get((i, j), 0.0) + coeff

    # Base terms: linear + quadratic
    for i, coeff in linear.items():
        add_linear(i, coeff)

    for (i, j), coeff in quad.items():
        add_quad(i, j, coeff)

    # Budget penalty
    for i in free_indices:
        add_linear(i, lambda1 * (1.0 - 2.0 * K_new))

    for idx_i, i in enumerate(free_indices):
        for j in free_indices[idx_i + 1:]:
            add_quad(i, j, 2.0 * lambda1)

    constant += lambda1 * (K_new ** 2)

    # Connectivity penalty
    for i in free_indices:
        add_linear(i, lambda2)

    for i in free_indices:
        if i not in neighbors:
            continue
        for j in neighbors[i]:
            if j in M_indices:
                add_linear(i, -lambda2)
            elif j in free_indices and i != j:
                add_quad(i, j, -lambda2)

    if verbose:
        print(f"[QUBO_Manual] Built: {len(h)} linear, {len(J)} quadratic, constant={constant:.4f}")

    # DEBUG: Print penalty structure
    if verbose:
        # Compute penalty for selecting K_new vs K_new+1 stations
        # Take the first few free indices as an example
        sample_indices = free_indices[:K_new+1]
    
        # Energy for selecting K_new stations (using only these indices)
        E_K = 0.0
        for i in sample_indices[:K_new]:
            E_K += h.get(i, 0.0)
        for idx_i in range(K_new):
            for idx_j in range(idx_i+1, K_new):
                i = sample_indices[idx_i]
                j = sample_indices[idx_j]
                E_K += J.get((min(i,j), max(i,j)), 0.0)
    
        # Energy for selecting K_new+1 stations
        E_K1 = 0.0
        for i in sample_indices[:K_new+1]:
            E_K1 += h.get(i, 0.0)
        for idx_i in range(K_new+1):
            for idx_j in range(idx_i+1, K_new+1):
                i = sample_indices[idx_i]
                j = sample_indices[idx_j]
                E_K1 += J.get((min(i,j), max(i,j)), 0.0)
    
        print(f"[QUBO_Manual] DEBUG: Energy for K={K_new}: {E_K:.6f}")
        print(f"[QUBO_Manual] DEBUG: Energy for K={K_new+1}: {E_K1:.6f}")
        print(f"[QUBO_Manual] DEBUG: Difference (should be POSITIVE): {E_K1 - E_K:.6f}")

    return {
        "h": h,
        "J": J,
        "constant": constant,
        "status": "success",
        "method": "manual",
    }


def build_qubo(
    pairwise_data: Dict,
    K_new: int,
    lambda1: float,
    lambda2: float,
    use_jijmodeling: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Main QUBO builder entry point.

    Tries JijModeling first, then falls back to manual if requested or fails.
    """
    if use_jijmodeling:
        try:
            return build_qubo_jijmodeling(pairwise_data, K_new, lambda1, lambda2, verbose)
        except Exception as e:
            warnings.warn(f"JijModeling build failed: {e}. Falling back to manual.")
            return build_qubo_manual(pairwise_data, K_new, lambda1, lambda2, verbose)
    else:
        return build_qubo_manual(pairwise_data, K_new, lambda1, lambda2, verbose)


# ============================================================================
# Solution Extraction & Energy Evaluation (UNCHANGED)
# ============================================================================

def extract_solution_gurobi(model_result: Dict) -> np.ndarray:
    """Extract binary solution vector from Gurobi model result."""
    N_total = model_result["N_total"]
    x_vars = model_result["x_vars"]
    free_indices = model_result["free_indices"]
    M_indices = model_result["M_indices"]

    x_full = np.zeros(N_total, dtype=int)
    for m in M_indices:
        x_full[m] = 1

    for i, var in x_vars.items():
        x_full[i] = int(round(var.X))

    return x_full


def extract_solution_openjij(
    sampleset,
    free_indices: List[int],
    M_indices: List[int],
    N_total: int
) -> np.ndarray:
    """Extract binary solution vector from OpenJij sampleset."""
    x_full = np.zeros(N_total, dtype=int)
    for m in M_indices:
        x_full[m] = 1

    if hasattr(sampleset, 'record'):
        best_sample = sampleset.record.solution[0]
    elif hasattr(sampleset, 'samples'):
        best_sample = sampleset.samples[0]
    else:
        best_sample = sampleset[0]

    for p, val in enumerate(best_sample):
        if val == 1:
            x_full[free_indices[p]] = 1

    return x_full


def compute_energy(
    x: np.ndarray,
    pairwise_data: Dict,
    K_new: int,
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
) -> float:
    """Compute the energy of a given solution vector."""
    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]

    energy = 0.0

    for i, coeff in linear.items():
        energy += coeff * x[i]

    for (i, j), coeff in quad.items():
        energy += coeff * x[i] * x[j]

    if lambda1 is not None:
        budget = sum(x[i] for i in free_indices)
        energy += lambda1 * (budget - K_new) ** 2

    if lambda2 is not None:
        for i in free_indices:
            if x[i] == 1:
                selected_free_neighbors = sum(1 for j in neighbors.get(i, []) if j in free_indices and x[j] == 1)
                fixed_neighbors = sum(1 for j in neighbors.get(i, []) if j in M_indices)
                total_selected_neighbors = selected_free_neighbors + fixed_neighbors
                energy += lambda2 * (1 - total_selected_neighbors)

    return energy


# ============================================================================
# Save/Load Pairwise Data (UNCHANGED)
# ============================================================================

def save_pairwise_data(pairwise_data: Dict, filepath: Union[str, Path]) -> None:
    """Save pairwise_data to disk (pickle + JSON metadata)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "N_total": pairwise_data["N_total"],
        "N_free": pairwise_data["N_free"],
        "free_indices": pairwise_data["free_indices"],
        "M_indices": pairwise_data["M_indices"],
        "L_c": pairwise_data["L_c"],
        "L_w": pairwise_data["L_w"],
        "current_vector": pairwise_data["current_vector"],
        "beta": pairwise_data["beta"],
        "delta": pairwise_data["delta"],
        "D_max": pairwise_data["D_max"],
        "num_linear": len(pairwise_data["linear"]),
        "num_quad": len(pairwise_data["quad"]),
    }

    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(pairwise_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Save] Pairwise data saved to {path.with_suffix('.pkl')}")


def load_pairwise_data(filepath: Union[str, Path]) -> Dict:
    """Load pairwise_data from disk."""
    path = Path(filepath)
    with open(path.with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)


def print_pairwise_summary(pairwise_data: Dict) -> None:
    """Print a human-readable summary of pairwise_data."""
    print("=" * 60)
    print("PAIRWISE DATA SUMMARY")
    print("=" * 60)
    print(f"N_total: {pairwise_data['N_total']}")
    print(f"N_free: {pairwise_data['N_free']}")
    print(f"Free indices: {pairwise_data['free_indices'][:10]}..." if len(pairwise_data['free_indices']) > 10 else f"Free indices: {pairwise_data['free_indices']}")
    print(f"M_indices: {pairwise_data['M_indices']}")
    print(f"L_c: {pairwise_data['L_c']} km")
    print(f"L_w: {pairwise_data['L_w']} km")
    print(f"D_max: {pairwise_data['D_max']:.3f} km")
    print(f"Linear terms: {len(pairwise_data['linear'])}")
    print(f"Quad terms: {len(pairwise_data['quad'])}")
    print(f"Neighbors: {len(pairwise_data['neighbors'])} free nodes")
    if pairwise_data['linear']:
        vals = list(pairwise_data['linear'].values())
        print(f"  Linear: min={min(vals):.4f}, max={max(vals):.4f}")
    if pairwise_data['quad']:
        vals = list(pairwise_data['quad'].values())
        print(f"  Quad: min={min(vals):.4f}, max={max(vals):.4f}")
    print("=" * 60)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core
    "compute_pairwise_terms",
    # New normalization functions
    "compute_objective_variance",
    "balance_quadratic_variance",
    "normalize_objective_to_unit_variance",
    "prepare_normalized_qubo",
    # QUBO / MIQP
    "build_miqp",
    "build_qubo",
    "build_qubo_jijmodeling",
    "build_qubo_manual",
    # Solution utilities
    "extract_solution_gurobi",
    "extract_solution_openjij",
    "compute_energy",
    # I/O
    "save_pairwise_data",
    "load_pairwise_data",
    "print_pairwise_summary",
]