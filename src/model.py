"""
src/model.py

Core mathematical formulation and solver builders for the water quality
monitoring optimization problem.

This module provides:
    1. Pairwise term computation (redundancy + wake) and neighbor finding.
    2. MIQP builder (Gurobi) with explicit constraints for exact baseline.
    3. QUBO builder (JijModeling + transpiler, with manual fallback).
    4. Utility functions for solution extraction and energy evaluation.

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

# -----------------------------------------------------------------------------
# Core Pairwise Term Computation
# -----------------------------------------------------------------------------

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
        cos_theta_rev = (dx * v_unit[0] + dy * v_unit[1]) / d  # Actually same as above since dx is from i to j.
        # For W_ji, we need vector from j to i: (-dx, -dy)
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
                    # raw_quad is already symmetrized, so we can just use it.
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


# -----------------------------------------------------------------------------
# Gurobi MIQP Builder (Exact Baseline)
# -----------------------------------------------------------------------------

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
    model.setParam("NonConvex", 2)  # Important for quadratic objective

    # Add binary variables
    x_vars = {}
    for i in free_indices:
        x_vars[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    model.update()

    # Build objective: linear + quadratic
    lin_expr = gp.quicksum(linear[i] * x_vars[i] for i in free_indices)
    quad_expr = gp.QuadExpr()
    for (i, j), coeff in quad.items():
        # i and j are guaranteed free and i < j
        quad_expr += coeff * x_vars[i] * x_vars[j]

    model.setObjective(lin_expr + quad_expr, GRB.MINIMIZE)

    # Budget constraint: sum(x_i) == K_new
    model.addConstr(gp.quicksum(x_vars[i] for i in free_indices) == K_new, name="budget")

    # Connectivity constraints: x_i <= sum_{j in N_i} x_j
    # For fixed stations M, they are always 1, so they contribute a constant to RHS.
    for i in free_indices:
        if i not in neighbors:
            # Isolated point: no neighbors -> constraint is x_i <= 0 -> x_i = 0
            # But this is a soft constraint in QUBO; in MIQP we enforce strictly.
            # Since we have a budget constraint, we can just force x_i = 0.
            # However, connectivity is supposed to be a guarantee, so we add:
            # x_i <= 0  (which forces x_i = 0)
            model.addConstr(x_vars[i] == 0, name=f"conn_isolated_{i}")
            continue

        # RHS sum over free neighbors
        rhs_free = gp.quicksum(x_vars[j] for j in neighbors[i] if j in free_indices)
        # RHS constant from fixed neighbors
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


# -----------------------------------------------------------------------------
# QUBO Builders (for OpenJij, SA, SQA, QAOA)
# -----------------------------------------------------------------------------

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

    Args:
        pairwise_data: Output from compute_pairwise_terms.
        K_new: Number of new stations to select.
        lambda1: Budget penalty coefficient.
        lambda2: Connectivity penalty coefficient.
        verbose: Print progress.

    Returns:
        Dict with keys:
            'h': dict {i: coeff} for free indices.
            'J': dict {(i,j): coeff} for i<j.
            'constant': float,
            'problem': jm.Problem (if available),
            'status': str,
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

    # Prepare arrays for JijModeling
    # We need 1D arrays for linear, 2D for quad and neighbor mask
    # Map free index -> position 0..N_free-1
    pos = {idx: p for p, idx in enumerate(free_indices)}

    a_arr = np.zeros(N_free)
    for i, coeff in linear.items():
        a_arr[pos[i]] = coeff

    b_arr = np.zeros((N_free, N_free))
    for (i, j), coeff in quad.items():
        p_i = pos[i]
        p_j = pos[j]
        b_arr[p_i, p_j] = coeff
        b_arr[p_j, p_i] = coeff  # ensure symmetry (though quad only has i<j)

    neigh_mask = np.zeros((N_free, N_free), dtype=int)
    for i, neigh_list in neighbors.items():
        p_i = pos[i]
        for j in neigh_list:
            if j in pos:  # only free neighbors
                p_j = pos[j]
                neigh_mask[p_i, p_j] = 1

    # JijModeling placeholders
    a = jm.Placeholder("a", ndim=1, shape=(N_free,))
    b = jm.Placeholder("b", ndim=2, shape=(N_free, N_free))
    mask = jm.Placeholder("mask", ndim=2, shape=(N_free, N_free))
    K = jm.Placeholder("K", ndim=0)
    lam1 = jm.Placeholder("lambda1", ndim=0)
    lam2 = jm.Placeholder("lambda2", ndim=0)

    # Variables
    x = jm.Variable("x", shape=(N_free,), binary=True)

    # Build objective
    linear_term = jm.sum(i, a[i] * x[i])
    quad_term = jm.sum([i, j], b[i, j] * x[i] * x[j], i < j)
    budget_penalty = lam1 * (jm.sum(i, x[i]) - K) ** 2

    # Connectivity penalty: sum_i x_i * (1 - sum_{j in N_i} x_j)
    # Expand: sum_i x_i - sum_i sum_{j in N_i} x_i x_j
    conn_penalty = lam2 * (
        jm.sum(i, x[i]) - jm.sum([i, j], mask[i, j] * x[i] * x[j])
    )

    total_objective = linear_term + quad_term + budget_penalty + conn_penalty

    problem = jm.Problem("QUBO_WaterQuality")
    problem += jm.Objective("minimize", total_objective)

    if verbose:
        print("[QUBO_Jij] Problem defined. Compiling...")

    # Compile
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

    # Extract h, J, constant
    h = {}
    J = {}
    constant = 0.0

    # PUBO model contains linear and quadratic terms
    for term, coeff in pubo_model.pubo.items():
        # term is a tuple of indices (i, j, ...)
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
            # Higher-order terms (should not happen for QUBO, but warn)
            warnings.warn(f"Higher-order term found: {term} -> {coeff}. Ignoring.")
            continue

    if verbose:
        print(f"[QUBO_Jij] Extracted: {len(h)} linear, {len(J)} quadratic, constant={constant:.4f}")

    # Map positions back to original indices
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

    This is a fallback that expands the QUBO formula (Eq. 13) explicitly.
    It handles both free-free and free-fixed interactions.

    Args:
        pairwise_data: Output from compute_pairwise_terms.
        K_new: Number of new stations to select.
        lambda1: Budget penalty coefficient.
        lambda2: Connectivity penalty coefficient.
        verbose: Print progress.

    Returns:
        Dict with keys:
            'h': dict {i: coeff} for free indices.
            'J': dict {(i,j): coeff} for i<j.
            'constant': float,
            'status': str,
            'method': 'manual',
    """
    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]
    N_free = pairwise_data["N_free"]

    if N_free == 0:
        raise ValueError("No free variables. Cannot build QUBO.")

    # Initialize
    h = {}
    J = {}
    constant = 0.0

    # Helper to add to h or J
    def add_linear(i, coeff):
        h[i] = h.get(i, 0.0) + coeff

    def add_quad(i, j, coeff):
        if i == j:
            add_linear(i, coeff)
            return
        if i > j:
            i, j = j, i
        J[(i, j)] = J.get((i, j), 0.0) + coeff

    # ------------------------------------------------------------------------
    # 1. Base terms: linear + quadratic (from original objective)
    # ------------------------------------------------------------------------
    for i, coeff in linear.items():
        add_linear(i, coeff)

    for (i, j), coeff in quad.items():
        add_quad(i, j, coeff)

    # ------------------------------------------------------------------------
    # 2. Budget penalty: lambda1 * (sum x_i - K)^2
    #    Expand: lambda1 * (sum x_i)^2 - 2*lambda1*K * sum x_i + lambda1*K^2
    # ------------------------------------------------------------------------
    # Linear: -2 * lambda1 * K
    for i in free_indices:
        add_linear(i, lambda1 * (1.0 - 2.0 * K_new))

    # Quadratic: 2 * lambda1 for each pair (i, j)
    for idx_i, i in enumerate(free_indices):
        for idx_j, j in enumerate(free_indices):
            if i < j:
                add_quad(i, j, 2.0 * lambda1)

    # Constant: lambda1 * K^2
    constant += lambda1 * (K_new ** 2)

    # ------------------------------------------------------------------------
    # 3. Connectivity penalty: lambda2 * sum_i x_i * (1 - sum_{j in N_i} x_j)
    #    Expand: lambda2 * sum_i x_i - lambda2 * sum_i sum_{j in N_i} x_i x_j
    # ------------------------------------------------------------------------
    # Linear: lambda2 for each free i
    for i in free_indices:
        add_linear(i, lambda2)

    # Quadratic: -lambda2 for each (i, j) where j is in N_i AND j is free
    # For fixed neighbors, they contribute to linear: -lambda2 * x_i (since x_j = 1)
    for i in free_indices:
        if i not in neighbors:
            continue
        for j in neighbors[i]:
            if j in M_indices:
                # j is fixed: contributes -lambda2 * x_i
                add_linear(i, -lambda2)
            elif j in free_indices and i != j:
                # both free: contributes -lambda2 * x_i * x_j
                # We only add once when i < j to maintain symmetry
                if i < j:
                    add_quad(i, j, -lambda2)
                else:
                    # i > j, we add when i < j, so skip this direction.
                    # But we need to ensure it's added. Let's use a simpler approach:
                    # Add directly with min/max ordering
                    add_quad(i, j, -lambda2)

    if verbose:
        print(f"[QUBO_Manual] Built: {len(h)} linear, {len(J)} quadratic, constant={constant:.4f}")

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

    Args:
        pairwise_data: Output from compute_pairwise_terms.
        K_new: Number of new stations to select.
        lambda1: Budget penalty coefficient.
        lambda2: Connectivity penalty coefficient.
        use_jijmodeling: If True, try JijModeling first. If False, use manual.
        verbose: Print progress.

    Returns:
        Dict with keys:
            'h': dict {i: coeff} for free indices.
            'J': dict {(i,j): coeff} for i<j.
            'constant': float,
            'status': str,
            'method': str ('jijmodeling' or 'manual'),
    """
    if use_jijmodeling:
        try:
            return build_qubo_jijmodeling(pairwise_data, K_new, lambda1, lambda2, verbose)
        except Exception as e:
            warnings.warn(f"JijModeling build failed: {e}. Falling back to manual.")
            return build_qubo_manual(pairwise_data, K_new, lambda1, lambda2, verbose)
    else:
        return build_qubo_manual(pairwise_data, K_new, lambda1, lambda2, verbose)


# -----------------------------------------------------------------------------
# Solution Extraction & Energy Evaluation
# -----------------------------------------------------------------------------

def extract_solution_gurobi(model_result: Dict) -> np.ndarray:
    """
    Extract binary solution vector from Gurobi model result.

    Args:
        model_result: Output from build_miqp.

    Returns:
        np.ndarray of length N_total with 1/0 values.
    """
    N_total = model_result["N_total"]
    x_vars = model_result["x_vars"]
    free_indices = model_result["free_indices"]
    M_indices = model_result["M_indices"]

    x_full = np.zeros(N_total, dtype=int)
    # Fixed stations are always 1
    for m in M_indices:
        x_full[m] = 1

    # Free variables from Gurobi
    for i, var in x_vars.items():
        x_full[i] = int(round(var.X))

    return x_full


def extract_solution_openjij(sampleset, free_indices: List[int], M_indices: List[int], N_total: int) -> np.ndarray:
    """
    Extract binary solution vector from OpenJij sampleset.

    Args:
        sampleset: Output from oj.SASampler.sample_qubo or SQASampler.sample_qubo.
        free_indices: List of free indices.
        M_indices: List of fixed indices.
        N_total: Total number of original indices.

    Returns:
        np.ndarray of length N_total with 1/0 values.
    """
    x_full = np.zeros(N_total, dtype=int)
    for m in M_indices:
        x_full[m] = 1

    # Get best sample (first row)
    if hasattr(sampleset, 'record'):
        # OpenJij returns a SampleSet with .record
        best_sample = sampleset.record.solution[0]
    elif hasattr(sampleset, 'samples'):
        best_sample = sampleset.samples[0]
    else:
        # Fallback: iterate
        best_sample = sampleset[0]

    # The sample is indexed by the QUBO variable indices (0..N_free-1)
    # We need to map back to original indices.
    # The QUBO variables are in the same order as free_indices.
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
    """
    Compute the energy of a given solution vector.

    If lambda1 and lambda2 are provided, computes the QUBO energy (with penalties).
    If not, computes the MIQP energy (without penalties).

    Args:
        x: Binary vector of length N_total.
        pairwise_data: Output from compute_pairwise_terms.
        K_new: Number of new stations to select.
        lambda1: Budget penalty (optional, for QUBO).
        lambda2: Connectivity penalty (optional, for QUBO).

    Returns:
        Energy (float).
    """
    linear = pairwise_data["linear"]
    quad = pairwise_data["quad"]
    neighbors = pairwise_data["neighbors"]
    free_indices = pairwise_data["free_indices"]
    M_indices = pairwise_data["M_indices"]

    energy = 0.0

    # Linear terms (only free variables have linear coefficients)
    for i, coeff in linear.items():
        energy += coeff * x[i]

    # Quadratic terms (only free-free pairs)
    for (i, j), coeff in quad.items():
        energy += coeff * x[i] * x[j]

    # Penalties (if provided)
    if lambda1 is not None:
        budget = sum(x[i] for i in free_indices)
        energy += lambda1 * (budget - K_new) ** 2

    if lambda2 is not None:
        for i in free_indices:
            if x[i] == 1:
                # Count free neighbors selected
                selected_free_neighbors = sum(1 for j in neighbors.get(i, []) if j in free_indices and x[j] == 1)
                # Fixed neighbors are always selected (since they are 1)
                fixed_neighbors = sum(1 for j in neighbors.get(i, []) if j in M_indices)
                total_selected_neighbors = selected_free_neighbors + fixed_neighbors
                # Penalty: x_i * (1 - sum_neighbors)
                energy += lambda2 * (1 - total_selected_neighbors)

    return energy


# -----------------------------------------------------------------------------
# Utility: Save/Load Pairwise Data
# -----------------------------------------------------------------------------

def save_pairwise_data(pairwise_data: Dict, filepath: Union[str, Path]) -> None:
    """Save pairwise_data to disk (pickle + JSON metadata)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Separate metadata (JSON-safe) from large arrays
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

    # Save full data as pickle
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(pairwise_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Save] Pairwise data saved to {path.with_suffix('.pkl')}")


def load_pairwise_data(filepath: Union[str, Path]) -> Dict:
    """Load pairwise_data from disk."""
    path = Path(filepath)
    with open(path.with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Quick Debug Helper
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Module Exports (for cleaner imports)
# -----------------------------------------------------------------------------

__all__ = [
    "compute_pairwise_terms",
    "build_miqp",
    "build_qubo",
    "build_qubo_jijmodeling",
    "build_qubo_manual",
    "extract_solution_gurobi",
    "extract_solution_openjij",
    "compute_energy",
    "save_pairwise_data",
    "load_pairwise_data",
    "print_pairwise_summary",
]