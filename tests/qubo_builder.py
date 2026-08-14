#@title GENERATE QUBO INSTANCES PER TIER (compute_pairwise_terms)
# -----------------------------------------------------------------------------
# (Assuming you have src.model.compute_pairwise_terms available)
# If not, I'll include a placeholder – you can replace with your actual import.


CONFIG_QUBO = {
    "L_c": 7500,
    "L_w": 1000,
    "Beta": 1.0,
    "Delta": 1.0,
    "Current_vector": (1.0, 0.0),
    "K": 5,
}

from typing import Dict, List, Optional, Tuple, Union, Any
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

if compute_pairwise_terms is not None:
    print("\n🔗 Generating QUBO instances for each tier...")
    for N, res in scaling_results.items():
        # Run pairwise terms (original full N)
        pairwise = compute_pairwise_terms(
            coords=res["coords"],
            U=res["U"],
            M_indices=res["M_indices"],
            L_c=CONFIG_QUBO["L_c"],
            L_w=CONFIG_QUBO["L_w"],
            current_vector=CONFIG_QUBO["Current_vector"],
            beta=CONFIG_QUBO["Beta"],
            delta=CONFIG_QUBO["Delta"],
            connectivity_range=res["dmax"],
            verbose=False,
        )
        # Build a, Q, neigh
        N_total = pairwise["N_total"]
        a = np.zeros(N_total)
        for i, coeff in pairwise["linear"].items():
            a[i] = coeff
        Q = np.zeros((N_total, N_total))
        for (i, j), coeff in pairwise["quad"].items():
            Q[i, j] = coeff
            Q[j, i] = coeff
        neigh = np.zeros((N_total, N_total), dtype=int)
        for i, nbrs in pairwise["neighbors"].items():
            for j in nbrs:
                neigh[i, j] = 1

        # Remove existing stations from decision space
        free_indices = [i for i in range(N_total) if i not in res["M_indices"]]
        M_set = set(res["M_indices"])
        a_new = np.zeros(len(free_indices))
        for new_i, orig_i in enumerate(free_indices):
            a_new[new_i] = pairwise["linear"].get(orig_i, 0.0)
            for m in M_set:
                if (orig_i, m) in pairwise["quad"]:
                    a_new[new_i] += pairwise["quad"][(orig_i, m)]
                elif (m, orig_i) in pairwise["quad"]:
                    a_new[new_i] += pairwise["quad"][(m, orig_i)]

        Q_new = np.zeros((len(free_indices), len(free_indices)))
        for a_idx, orig_a in enumerate(free_indices):
            for b_idx, orig_b in enumerate(free_indices):
                if a_idx < b_idx:
                    val = pairwise["quad"].get((orig_a, orig_b), 0.0)
                    if val != 0:
                        Q_new[a_idx, b_idx] = val
                        Q_new[b_idx, a_idx] = val

        neigh_new = np.zeros((len(free_indices), len(free_indices)), dtype=int)
        for i, orig_i in enumerate(free_indices):
            for j, orig_j in enumerate(free_indices):
                if i != j and pairwise["neighbors"][orig_i].count(orig_j) > 0:
                    neigh_new[i, j] = 1

        fixed_neighbors = {
            i: [m for m in M_set if pairwise["neighbors"][orig_i].count(m) > 0]
            for i, orig_i in enumerate(free_indices)
        }
        has_fixed_neighbor = np.array([len(fixed_neighbors[i]) > 0 for i in range(len(free_indices))], dtype=int)

        instance_data = {
            "N": len(free_indices),
            "K": CONFIG_QUBO["K"],
            "a": a_new,
            "Q": Q_new,
            "neigh": neigh_new,
            "coords": res["coords"][free_indices],
            "U": res["U"][free_indices],
            "M_indices": [],
            "D_max": res["dmax"],
            "original_indices": free_indices,
            "fixed_indices": res["M_indices"],
            "fixed_neighbors": fixed_neighbors,
            "has_fixed_neighbor": has_fixed_neighbor,
            "original_coords": res["coords"],
            "original_U": res["U"],
            "L_c": CONFIG_QUBO["L_c"],
            "L_w": CONFIG_QUBO["L_w"],
        }

        # Save tier instance to Drive
        instance_path_drive = DRIVE_BASE / f"instance_data_N{N}.pkl"
        with open(instance_path_drive, "wb") as f:
            pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"   ✅ Saved QUBO instance for N={N} to Drive.")