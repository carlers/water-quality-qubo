#@title 🧩 GENERATE SHAPE-AWARE QUBO INSTANCES PER TIER (Cell 3)
# =============================================================================
# Key features:
#   1. Resolution-Invariant D_max: Computed dynamically from the max nearest-
#      neighbor distance of EXISTING baseline stations on the master grid.
#   2. Graph Ownership: Constructs candidate adjacency edges strictly filtered 
#      by Shapely Line-of-Sight against the water polygon to block land hops.
#   3. Line-of-Sight Interactions: Redundancy (R_ij) and Wake (W_ij) penalties 
#      are zeroed out if the spatial correlation line crosses land.
#   4. Clean Free/Fixed Decoupling: Correctly absorbs interactions with 
#      existing stations into linear terms (a_i) for free decision variables.
# =============================================================================
import os
import pickle
import time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LineString
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_QUBO = {
    "L_c": 7500.0,                # Spatial correlation length (meters)
    "L_w": 1000.0,                # Wake persistence length (meters)
    "Beta": 1.0,                  # Redundancy penalty weight
    "Delta": 1.0,                 # Wake penalty weight
    "Current_vector": (1.0, 0.0), # Flow vector (dx, dy)
    "K": 5,                       # Number of new stations to place per tier
    "D_max_buffer": 1.15,         # Multiplier for max existing NN distance
}

# -----------------------------------------------------------------------------
# 1. CALCULATE GLOBAL RESOLUTION-INVARIANT D_MAX
# -----------------------------------------------------------------------------
# Extract coordinates of existing stations from the global master set
master_M_coords = coords[list(M_set_global)]

if len(master_M_coords) > 1:
    M_tree = cKDTree(master_M_coords)
    # Query nearest neighbors (k=2 because the closest to itself is distance 0)
    dists, _ = M_tree.query(master_M_coords, k=2)
    max_nn_dist = np.max(dists[:, 1])
    global_D_max = max_nn_dist * CONFIG_QUBO["D_max_buffer"]
else:
    print("⚠️ Less than 2 existing stations found. Falling back to default D_max.")
    global_D_max = 15000.0

print(f"🌍 Global D_max calculated: {global_D_max:.2f} m (based on existing stations)")

# -----------------------------------------------------------------------------
# PAIRWISE TERM CALCULATION (SHAPE-AWARE & GRAPH BUILDER)
# -----------------------------------------------------------------------------
def compute_pairwise_terms_shape_aware(
    coords: np.ndarray,
    U: np.ndarray,
    M_indices: list,
    water_polygon: shapely.geometry.Polygon,
    L_c: float,
    L_w: float,
    current_vector: tuple,
    global_D_max: float,
    beta: float = 1.0,
    delta: float = 1.0,
) -> dict:
    
    N_total = coords.shape[0]
    free_indices = [i for i in range(N_total) if i not in M_indices]
    N_free = len(free_indices)
    free_set = set(free_indices)
    M_set = set(M_indices)

    shapely.prepare(water_polygon)
    tree = cKDTree(coords)
    
    v_norm = np.linalg.norm(current_vector)
    if v_norm == 0:
        raise ValueError("current_vector cannot be zero.")
    v_unit = np.array(current_vector) / v_norm

    # ------------------------------------------------------------------------
    # A. Build Connectivity Graph (Within D_max)
    # ------------------------------------------------------------------------
    valid_edges = set()
    pairs_dmax = tree.query_pairs(r=global_D_max)

    for i, j in pairs_dmax:
        p1, p2 = coords[i], coords[j]
        edge_line = LineString([p1, p2])
        # Only keep edge if it completely stays within water polygon
        if water_polygon.contains(edge_line):
            valid_edges.add((i, j))
            
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
    # B. Compute Pairwise Quadratic Interactions (Within 2 * L_c)
    # ------------------------------------------------------------------------
    raw_quad = {}
    pairs_within = tree.query_pairs(r=2.0 * L_c)

    for i, j in pairs_within:
        p1, p2 = coords[i], coords[j]
        edge_line = LineString([p1, p2])

        # GEOMETRIC FILTER: If line crosses land, ignore spatial interaction
        if not water_polygon.contains(edge_line):
            continue

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = np.sqrt(dx * dx + dy * dy)
        if d == 0:
            continue

        R_ij = max(0.0, 1.0 - d / L_c)
        cos_theta = (dx * v_unit[0] + dy * v_unit[1]) / d
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        W_ij = (np.exp(-d / L_w) * cos_theta) if cos_theta > 0.7071 else 0.0

        cos_theta_ji = (-dx * v_unit[0] + -dy * v_unit[1]) / d
        cos_theta_ji = np.clip(cos_theta_ji, -1.0, 1.0)
        W_ji = (np.exp(-d / L_w) * cos_theta_ji) if cos_theta_ji > 0.7071 else 0.0

        coeff = beta * R_ij + delta * (W_ij + W_ji)
        raw_quad[(i, j)] = coeff
        raw_quad[(j, i)] = coeff

    # ------------------------------------------------------------------------
    # C. Build Linear Terms (Including Absorbed Fixed Station Effects)
    # ------------------------------------------------------------------------
    linear = {}
    for i in free_indices:
        val = -U[i]  
        for m in M_indices:
            key = (i, m) if i < m else (m, i)
            if key in raw_quad:
                val += raw_quad[key]
        linear[i] = val

    # ------------------------------------------------------------------------
    # D. Build Free-Free Quadratic Coupling Terms
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
        "valid_edges": valid_edges,
        "N_total": N_total,
        "N_free": N_free,
        "free_indices": free_indices,
        "M_indices": M_indices,
    }

# -----------------------------------------------------------------------------
# 2. BUILD & SAVE QUBO INSTANCES PER TIER
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("🔗 GENERATING SHAPE-AWARE QUBO INSTANCES PER RESOLUTION TIER")
print("=" * 100)

qubo_results = {}

for N, res in scaling_results.items():
    t0 = time.perf_counter()

    pairwise = compute_pairwise_terms_shape_aware(
        coords=res["coords"],
        U=res["U"],
        M_indices=res["M_indices"],
        water_polygon=water_polygon,
        L_c=CONFIG_QUBO["L_c"],
        L_w=CONFIG_QUBO["L_w"],
        current_vector=CONFIG_QUBO["Current_vector"],
        global_D_max=global_D_max,
        beta=CONFIG_QUBO["Beta"],
        delta=CONFIG_QUBO["Delta"],
    )

    free_indices = pairwise["free_indices"]
    N_free = len(free_indices)
    M_set = set(res["M_indices"])

    a_new = np.zeros(N_free, dtype=np.float64)
    for new_i, orig_i in enumerate(free_indices):
        a_new[new_i] = pairwise["linear"].get(orig_i, 0.0)

    Q_new = np.zeros((N_free, N_free), dtype=np.float64)
    for a_idx, orig_a in enumerate(free_indices):
        for b_idx, orig_b in enumerate(free_indices):
            if a_idx < b_idx:
                val = pairwise["quad"].get((orig_a, orig_b), 0.0)
                if val != 0.0:
                    Q_new[a_idx, b_idx] = val
                    Q_new[b_idx, a_idx] = val

    neigh_new = np.zeros((N_free, N_free), dtype=int)
    for i, orig_i in enumerate(free_indices):
        for j, orig_j in enumerate(free_indices):
            if i != j and orig_j in pairwise["neighbors"][orig_i]:
                neigh_new[i, j] = 1

    fixed_neighbors = {
        i: [m for m in M_set if m in pairwise["neighbors"][orig_i]]
        for i, orig_i in enumerate(free_indices)
    }
    has_fixed_neighbor = np.array([len(fixed_neighbors[i]) > 0 for i in range(N_free)], dtype=int)

    instance_data = {
        "N": N_free,
        "K": CONFIG_QUBO["K"],
        "a": a_new,
        "Q": Q_new,
        "neigh": neigh_new,
        "coords": res["coords"][free_indices],
        "U": res["U"][free_indices],
        "M_indices": [], 
        "D_max": global_D_max,
        "original_indices": free_indices,
        "fixed_indices": res["M_indices"],
        "fixed_neighbors": fixed_neighbors,
        "has_fixed_neighbor": has_fixed_neighbor,
        "original_coords": res["coords"],
        "original_U": res["U"],
        "snapped_indices": res["snapped_indices"],
        "L_c": CONFIG_QUBO["L_c"],
        "L_w": CONFIG_QUBO["L_w"],
    }
    
    qubo_results[N] = {
        "edges": pairwise["valid_edges"],
        "coords": res["coords"],
        "M_indices": res["M_indices"]
    }

    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    save_filename = f"instance_data_N{N}.pkl"
    local_save_path = LOCAL_CACHE / save_filename
    
    with open(local_save_path, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "OUTPUT_DIR" in globals():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        drive_save_path = OUTPUT_DIR / save_filename
        with open(drive_save_path, "wb") as f:
            pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = (time.perf_counter() - t0) * 1000
    n_nonzero_q = np.count_nonzero(Q_new) // 2
    avg_degree = np.sum(neigh_new) / N_free if N_free > 0 else 0
    
    print(f"✅ Tier N={N:<3d} (Free: {N_free:<3d}) | Avg Deg: {avg_degree:<5.1f} | Non-zero Q: {n_nonzero_q:<5d} | Saved in {dt:.1f} ms")

print("=" * 100)
print("🚀 All QUBO instances compiled and saved successfully.")

# -----------------------------------------------------------------------------
# 3. PLOTTING LANDMASS-AWARE CONNECTIVITY GRAPHS
# -----------------------------------------------------------------------------
print("\n📊 Plotting landmass-aware graph edges per tier...")
tiers = sorted(qubo_results.keys())
n_tiers = len(tiers)
cols = 2
rows = int(np.ceil(n_tiers / cols))

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = np.atleast_1d(axes).flatten()

for ax, N in zip(axes, tiers):
    tier_coords = qubo_results[N]["coords"]
    edges = qubo_results[N]["edges"]
    M_tier = qubo_results[N]["M_indices"]

    # Master map backdrop
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=4, alpha=0.35, zorder=0)

    # Plot water-only validated edges
    for i, j in edges:
        ax.plot([tier_coords[i, 0], tier_coords[j, 0]],
                [tier_coords[i, 1], tier_coords[j, 1]],
                color="steelblue", alpha=0.3, linewidth=0.6, zorder=1)

    # Plot Nodes
    ax.scatter(tier_coords[:, 0], tier_coords[:, 1], c="black", s=15, alpha=0.8, zorder=2)

    # Plot fixed stations
    if len(M_tier) > 0:
        ax.scatter(tier_coords[M_tier, 0], tier_coords[M_tier, 1],
                   c="red", marker="*", s=200, edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_title(f"Graph N={N} | Global D_max={global_D_max:.0f}m", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

for ax in axes[n_tiers:]:
    ax.axis("off")

plt.suptitle("Landmass-Aware Network Topology by Tier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()