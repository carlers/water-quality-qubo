"""
src/environment.py

Single source of truth for the normalized QUBO environment.
Provides a cached, reproducible environment for all experiments.

The environment includes:
    - Coordinates, utility, existing station indices.
    - Normalized pairwise terms (h_norm, J_norm).
    - Pre‑computed objective matrix Q_obj (for ESR/MCR computation).
    - Gurobi baseline (MIQP value and solution).
    - Fixed physical parameters (L_c, connectivity_range).

Caching:
    A deterministic hash is generated from all parameters that affect
    the QUBO (seed, K_new, L_c, L_w, beta, delta, connectivity_range).
    If the cache exists locally (or on GDrive), it is loaded.
    Otherwise, it is built and saved.

Usage:
    from src.environment import get_environment
    env = get_environment(seed=42, K_new=5, L_c=5.0, connectivity_range=8.0)
    # env is a dict with all needed data.
"""

import hashlib
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy.spatial.distance import cdist

from data.synthetic_data import load_master_data, generate_and_save_all
from src.model import (
    compute_pairwise_terms,
    build_qubo,
    build_miqp,
    prepare_normalized_qubo,
    extract_solution_gurobi,
)
from src.solvers import solve_greedy
from src.utils import safe_load_pickle, safe_save_pickle

warnings.filterwarnings('ignore')


# ============================================================================
# Helper: Build full Q matrix from h and J
# ============================================================================

def _build_full_Q(h: Dict[int, float], J: Dict[Tuple[int, int], float], N: int) -> np.ndarray:
    """
    Build symmetric full QUBO matrix from linear (h) and quadratic (J) coefficients.
    """
    Q = np.zeros((N, N))
    for i, val in h.items():
        if i < N:
            Q[i, i] += val
    for (i, j), val in J.items():
        if i >= N or j >= N:
            continue
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += val / 2.0
            Q[j, i] += val / 2.0
    return Q


# ============================================================================
# Core function: get_environment
# ============================================================================

def get_environment(
    seed: int = 42,
    K_new: int = 5,
    L_c: float = 5.0,
    L_w: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    connectivity_range: Optional[float] = None,
    current_vector: Tuple[float, float] = (1.0, 0.0),
    force_recompute: bool = False,
    verbose: bool = True,
    gdrive_base: Optional[str] = None,
) -> Dict:
    """
    Build or load the cached normalized QUBO environment.

    Args:
        seed: Random seed for data generation.
        K_new: Number of new stations to select.
        L_c: Spatial correlation length (km) – PHYSICAL PARAMETER, FIXED.
        L_w: Wake persistence length (km).
        beta: Redundancy penalty weight (for pairwise computation).
        delta: Wake penalty weight.
        connectivity_range: Communication range (km). If None, computed as 2 * mean NN distance.
        current_vector: (dx, dy) direction of dominant current.
        force_recompute: If True, ignore cache and rebuild.
        verbose: Print progress.
        gdrive_base: If provided, also look for cache in GDrive and save there.

    Returns:
        dict with keys:
            - 'coords': (N_total, 2) array
            - 'U': (N_total,) utility scores
            - 'M_indices': list of existing station indices
            - 'pairwise_norm': dict with normalized linear and quadratic terms
            - 'Q_obj': (N_total, N_total) objective matrix (h + J only, no penalties)
            - 'gurobi_miqp': float (baseline objective value)
            - 'gurobi_solution': np.ndarray (binary solution vector)
            - 'L_c': float (fixed correlation length)
            - 'Q_sum': float (sum of absolute normalized coefficients)
            - 'connectivity_range': float
            - 'config': dict of all parameters used for this environment
            - 'hash': str (cache hash)
            - 'scales': dict (normalization scales)
    """
    # Build a deterministic config dict (all parameters that affect the QUBO)
    config = {
        'seed': seed,
        'K_new': K_new,
        'L_c': L_c,
        'L_w': L_w,
        'beta': beta,
        'delta': delta,
        'connectivity_range': connectivity_range,
        'current_vector': current_vector,
    }
    # Sort keys for reproducibility
    config_str = json.dumps(config, sort_keys=True)
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    # Determine cache directories
    local_data_dir = Path(f"data_seed{seed}")
    local_cache_path = local_data_dir / f"env_{cache_hash}.pkl"

    # If gdrive_base is given, also check there
    gdrive_cache_path = None
    if gdrive_base:
        gdrive_root = Path(gdrive_base)
        gdrive_cache_path = gdrive_root / "env_cache" / f"env_{cache_hash}.pkl"

    # Try to load from cache
    if not force_recompute:
        # Try local first, then GDrive
        for cache_path in [local_cache_path, gdrive_cache_path]:
            if cache_path and cache_path.exists():
                if verbose:
                    print(f"  ⚡ Loading cached environment from {cache_path}")
                with open(cache_path, 'rb') as f:
                    env = pickle.load(f)
                # Verify that the stored config matches current config (safety)
                if env.get('config') == config:
                    return env
                else:
                    if verbose:
                        print(f"  ⚠️ Cache config mismatch. Rebuilding.")
                    break

    if verbose:
        print(f"  ⏳ Building environment (hash={cache_hash})...")

    # ------------------------------------------------------------------------
    # 1. Generate / load synthetic data
    # ------------------------------------------------------------------------
    if not local_data_dir.exists():
        generate_and_save_all(seed=seed, output_dir=str(local_data_dir))
    coords, factors, U, subsets, meta = load_master_data(str(local_data_dir))
    M_indices = meta['existing_indices']
    N_total = len(coords)

    if verbose:
        print(f"  ✓ Loaded data: N_total={N_total}, |M|={len(M_indices)}")

    # ------------------------------------------------------------------------
    # 2. Compute connectivity_range if not provided
    # ------------------------------------------------------------------------
    if connectivity_range is None:
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)
        mean_nn_dist = np.mean(np.min(dist_matrix, axis=1))
        connectivity_range = 2.0 * mean_nn_dist
        if verbose:
            print(f"  ✓ Auto-computed connectivity_range = {connectivity_range:.2f} km (2 × mean NN dist)")

    # Update config with the actual connectivity_range used
    config['connectivity_range'] = connectivity_range

    # ------------------------------------------------------------------------
    # 3. Build raw pairwise terms with fixed physical parameters
    # ------------------------------------------------------------------------
    if verbose:
        print(f"  Building pairwise terms (L_c={L_c} km, L_w={L_w} km, D_max={connectivity_range:.2f} km)...")

    pairwise_raw = compute_pairwise_terms(
        coords=coords,
        U=U,
        M_indices=M_indices,
        L_c=L_c,
        L_w=L_w,
        current_vector=current_vector,
        beta=beta,
        delta=delta,
        connectivity_range=connectivity_range,
        verbose=verbose,
    )

    # ------------------------------------------------------------------------
    # 4. Normalize the QUBO (two-step)
    # ------------------------------------------------------------------------
    if verbose:
        print("  Normalizing QUBO...")

    h_norm, J_norm, scales = prepare_normalized_qubo(
        pairwise_raw,
        target_variance=1.0,
        n_samples=10000,
        seed=42,
        verbose=verbose,
    )

    # Create normalized pairwise dict
    pairwise_norm = pairwise_raw.copy()
    pairwise_norm['linear'] = h_norm
    pairwise_norm['quad'] = J_norm

    # Compute Q_sum from normalized coefficients
    Q_sum = sum(abs(v) for v in h_norm.values()) + sum(abs(v) for v in J_norm.values())
    Q_sum = max(Q_sum, 1.0)  # safety floor

    # Pre-compute objective matrix Q_obj (no penalties)
    Q_obj = _build_full_Q(h_norm, J_norm, N_total)

    if verbose:
        print(f"  ✓ Normalization complete. Q_sum = {Q_sum:.4f}")
        print(f"    h range: {min(h_norm.values()):.4f} to {max(h_norm.values()):.4f}")
        print(f"    J range: {min(J_norm.values()) if J_norm else 0:.4f} to {max(J_norm.values()) if J_norm else 0:.4f}")

    # ------------------------------------------------------------------------
    # 5. Compute Gurobi baseline (or greedy fallback)
    # ------------------------------------------------------------------------
    gurobi_miqp = None
    gurobi_solution = None

    if verbose:
        print("  [Baseline] Computing exact MIQP solution (Gurobi or greedy fallback)...")

    try:
        import gurobipy as gp
        from gurobipy import GRB
        miqp_result = build_miqp(pairwise_raw, K_new, time_limit=30.0, mip_gap=1e-6, verbose=False)
        model = miqp_result['model']
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            gurobi_miqp = model.ObjVal
            gurobi_solution = extract_solution_gurobi(miqp_result)
            if verbose:
                print(f"    ✓ Gurobi optimal: MIQP = {gurobi_miqp:.8f}")
        else:
            raise Exception(f"Gurobi status {model.Status}")
    except Exception as e:
        if verbose:
            print(f"    ⚠️ Gurobi failed: {e}. Using greedy fallback.")
        free_indices = pairwise_raw['free_indices']
        M_indices_local = pairwise_raw['M_indices']
        N_total_local = pairwise_raw['N_total']
        greedy_result = solve_greedy(
            h=pairwise_raw['linear'],
            J=pairwise_raw['quad'],
            constant=0.0,
            K_new=K_new,
            free_indices=free_indices,
            M_indices=M_indices_local,
            N_total=N_total_local,
            pairwise_data=pairwise_raw,
            mode='marginal',
            verbose=False,
        )
        gurobi_miqp = greedy_result['miqp_energy']
        gurobi_solution = greedy_result['solution']
        if verbose:
            print(f"    ✓ Greedy fallback: MIQP = {gurobi_miqp:.8f}")

    # ------------------------------------------------------------------------
    # 6. Package environment
    # ------------------------------------------------------------------------
    env = {
        'coords': coords,
        'U': U,
        'M_indices': M_indices,
        'pairwise_norm': pairwise_norm,
        'Q_obj': Q_obj,
        'gurobi_miqp': gurobi_miqp,
        'gurobi_solution': gurobi_solution,
        'L_c': L_c,
        'Q_sum': Q_sum,
        'connectivity_range': connectivity_range,
        'config': config,
        'hash': cache_hash,
        'scales': scales,  # for reference
    }

    # ------------------------------------------------------------------------
    # 7. Save cache
    # ------------------------------------------------------------------------
    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_cache_path, 'wb') as f:
        pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  ✓ Saved local cache: {local_cache_path}")

    if gdrive_cache_path:
        try:
            gdrive_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gdrive_cache_path, 'wb') as f:
                pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"  ✓ Saved GDrive cache: {gdrive_cache_path}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Failed to save GDrive cache: {e}")

    if verbose:
        print("  ✅ Environment ready.")
    return env


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    'get_environment',
]