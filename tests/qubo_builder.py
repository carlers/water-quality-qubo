#@title 🧩 GENERATE SHAPE-AWARE QUBO INSTANCES PER TIER (Cell 3)
# =============================================================================
# Key features:
#   1. Line-of-Sight Filtering for Pairwise Terms: Redundancy (R_ij) and 
#      Wake (W_ij) penalties are zeroed out if the connecting line crosses land.
#   2. Inherits Water Graph: Adjacency/connectivity matrix 'neigh' is built
#      directly from Cell 2's pre-filtered water-only edges.
#   3. Clean Free/Fixed Decoupling: Correctly absorbs interactions with 
#      existing stations into linear terms (a_i) for free decision variables.
#   4. Preserves Master Index Tracking: Stores 'snapped_indices' to enable 
#      ground-truth evaluation against N=5417 master set downstream.
# =============================================================================
import os
import pickle
import time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LineString

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_QUBO = {
    "L_c": 7500.0,             # Spatial correlation length (meters)
    "L_w": 1000.0,             # Wake persistence length (meters)
    "Beta": 1.0,               # Redundancy penalty weight
    "Delta": 1.0,              # Wake penalty weight
    "Current_vector": (1.0, 0.0), # Flow vector (dx, dy)
    "K": 5,                    # Number of new stations to place per tier
}

# -----------------------------------------------------------------------------
# PAIRWISE TERM CALCULATION (SHAPE-AWARE)
# -----------------------------------------------------------------------------
def compute_pairwise_terms_shape_aware(
    coords: np.ndarray,
    U: np.ndarray,
    M_indices: list,
    valid_edges: set,
    water_polygon: shapely.geometry.Polygon,
    L_c: float,
    L_w: float,
    current_vector: tuple,
    beta: float = 1.0,
    delta: float = 1.0,
    D_max: float = 10000.0,
    verbose: bool = False,
) -> dict:
    """
    Computes linear/quadratic QUBO terms and connectivity neighbors, strictly
    enforcing water-only paths through the water polygon.
    """
    N_total = coords.shape[0]
    free_indices = [i for i in range(N_total) if i not in M_indices]
    N_free = len(free_indices)
    free_set = set(free_indices)
    M_set = set(M_indices)

    if verbose:
        print(f"[Pairwise] N_total={N_total}, N_free={N_free}, |M|={len(M_indices)}")
        print(f"[Pairwise] L_c={L_c}m, L_w={L_w}m, D_max={D_max:.1f}m")

    # Fast Shapely preparation
    shapely.prepare(water_polygon)

    # KD-tree for spatial interaction range (2 * L_c)
    tree = cKDTree(coords)
    v_norm = np.linalg.norm(current_vector)
    if v_norm == 0:
        raise ValueError("current_vector cannot be zero.")
    v_unit = np.array(current_vector) / v_norm

    # ------------------------------------------------------------------------
    # 1. Compute Pairwise Quadratic Interactions (Within 2 * L_c)
    # ------------------------------------------------------------------------
    raw_quad = {}
    pairs_within = tree.query_pairs(r=2.0 * L_c)

    for i, j in pairs_within:
        if i == j:
            continue
        
        p1 = coords[i]
        p2 = coords[j]
        edge_line = LineString([p1, p2])

        # 🟢 GEOMETRIC FILTER: If line crosses land, ignore spatial interaction
        if not water_polygon.contains(edge_line):
            continue

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = np.sqrt(dx * dx + dy * dy)
        if d == 0:
            continue

        # Redundancy penalty (symmetric)
        R_ij = max(0.0, 1.0 - d / L_c)

        # Directional Wake penalty (i -> j)
        cos_theta = (dx * v_unit[0] + dy * v_unit[1]) / d
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        W_ij = (np.exp(-d / L_w) * cos_theta) if cos_theta > 0.7071 else 0.0

        # Reverse Directional Wake penalty (j -> i)
        cos_theta_ji = (-dx * v_unit[0] + -dy * v_unit[1]) / d
        cos_theta_ji = np.clip(cos_theta_ji, -1.0, 1.0)
        W_ji = (np.exp(-d / L_w) * cos_theta_ji) if cos_theta_ji > 0.7071 else 0.0

        # Combined interaction weight
        coeff = beta * R_ij + delta * (W_ij + W_ji)
        raw_quad[(i, j)] = coeff
        raw_quad[(j, i)] = coeff

    # ------------------------------------------------------------------------
    # 2. Build Connectivity Neighbor Dict Directly from Cell 2 Valid Edges
    # ------------------------------------------------------------------------
    neighbor_dict = {i: [] for i in free_indices}
    for i, j in valid_edges:
        if i in free_set and j in free_set:
            neighbor_dict[i].append(j)
            neighbor_dict[j].append(i)
        elif i in free_set and j in M_set:
            neighbor_dict[i].append(j)
        elif j in free_set and i in M_set:
            neighbor_dict[j].append(i)

    for i in neighbor_dict:
        neighbor_dict[i] = sorted(set(neighbor_dict[i]))

    # ------------------------------------------------------------------------
    # 3. Build Linear Terms a_i (Including Absorbed Fixed Station Effects)
    # ------------------------------------------------------------------------
    linear = {}
    for i in free_indices:
        val = -U[i]  # Base utility minimization
        for m in M_indices:
            key = (i, m) if i < m else (m, i)
            if key in raw_quad:
                val += raw_quad[key]
        linear[i] = val

    # ------------------------------------------------------------------------
    # 4. Build Free-Free Quadratic Coupling Terms
    # ------------------------------------------------------------------------
    quad = {}
    for i in free_indices:
        for j in free_indices:
            if i < j:
                key = (i, j)
                if key in raw_quad:
                    quad[key] = raw_quad[key]

    return {
        "linear": linear,
        "quad": quad,
        "neighbors": neighbor_dict,
        "N_total": N_total,
        "N_free": N_free,
        "free_indices": free_indices,
        "M_indices": M_indices,
    }

# -----------------------------------------------------------------------------
# BUILD & SAVE QUBO INSTANCES PER TIER
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("🔗 GENERATING SHAPE-AWARE QUBO INSTANCES PER RESOLUTION TIER")
print("=" * 100)

for N, res in scaling_results.items():
    t0 = time.perf_counter()

    # Compute pairwise terms with shape awareness
    pairwise = compute_pairwise_terms_shape_aware(
        coords=res["coords"],
        U=res["U"],
        M_indices=res["M_indices"],
        valid_edges=res["edges"],
        water_polygon=water_polygon,
        L_c=CONFIG_QUBO["L_c"],
        L_w=CONFIG_QUBO["L_w"],
        current_vector=CONFIG_QUBO["Current_vector"],
        beta=CONFIG_QUBO["Beta"],
        delta=CONFIG_QUBO["Delta"],
        D_max=res["dmax"],
        verbose=False,
    )

    free_indices = pairwise["free_indices"]
    N_free = len(free_indices)
    M_set = set(res["M_indices"])

    # 1. Linear vector (a) for free variables
    a_new = np.zeros(N_free, dtype=np.float64)
    for new_i, orig_i in enumerate(free_indices):
        a_new[new_i] = pairwise["linear"].get(orig_i, 0.0)

    # 2. Symmetric Quadratic matrix (Q) for free variables
    Q_new = np.zeros((N_free, N_free), dtype=np.float64)
    for a_idx, orig_a in enumerate(free_indices):
        for b_idx, orig_b in enumerate(free_indices):
            if a_idx < b_idx:
                val = pairwise["quad"].get((orig_a, orig_b), 0.0)
                if val != 0.0:
                    Q_new[a_idx, b_idx] = val
                    Q_new[b_idx, a_idx] = val

    # 3. Connectivity Adjacency Matrix (neigh) for free variables
    neigh_new = np.zeros((N_free, N_free), dtype=int)
    for i, orig_i in enumerate(free_indices):
        for j, orig_j in enumerate(free_indices):
            if i != j and orig_j in pairwise["neighbors"][orig_i]:
                neigh_new[i, j] = 1

    # 4. Map fixed existing neighbors connected to each free node
    fixed_neighbors = {
        i: [m for m in M_set if m in pairwise["neighbors"][orig_i]]
        for i, orig_i in enumerate(free_indices)
    }
    has_fixed_neighbor = np.array([len(fixed_neighbors[i]) > 0 for i in range(N_free)], dtype=int)

    # 5. Build comprehensive instance_data dictionary
    instance_data = {
        "N": N_free,
        "K": CONFIG_QUBO["K"],
        "a": a_new,
        "Q": Q_new,
        "neigh": neigh_new,
        "coords": res["coords"][free_indices],
        "U": res["U"][free_indices],
        "M_indices": [],  # Cleaned: fixed nodes removed from decision space
        "D_max": res["dmax"],
        "original_indices": free_indices,
        "fixed_indices": res["M_indices"],
        "fixed_neighbors": fixed_neighbors,
        "has_fixed_neighbor": has_fixed_neighbor,
        "original_coords": res["coords"],
        "original_U": res["U"],
        "snapped_indices": res["snapped_indices"],  # Crucial for N=5417 ground-truth mapping
        "L_c": CONFIG_QUBO["L_c"],
        "L_w": CONFIG_QUBO["L_w"],
    }

    # Save to both Local Cache and Google Drive if available
    save_filename = f"instance_data_N{N}.pkl"
    
    # 🟢 FIX: Ensure the directory actually exists before writing!
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    
    local_save_path = LOCAL_CACHE / save_filename
    with open(local_save_path, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "OUTPUT_DIR" in globals():
        # Ensure drive output dir exists too, just in case
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        drive_save_path = OUTPUT_DIR / save_filename
        with open(drive_save_path, "wb") as f:
            pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = (time.perf_counter() - t0) * 1000
    n_nonzero_q = np.count_nonzero(Q_new) // 2
    print(f"✅ Tier N={N:<3d} (Free Variables: {N_free:<3d}) | Non-zero Q terms: {n_nonzero_q:<5d} | Saved in {dt:.1f} ms")

print("=" * 100)
print("🚀 All QUBO instances compiled and saved successfully.")