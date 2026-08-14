#@title CELL 4: GENERATE SPARSE QUBO + PRE‑BUILT MIQP PER TIER
# =============================================================================
# Key features:
#   1. Sparse storage: Q_edges (i,j,val) for i<j, neighbors as list-of-lists.
#   2. qsum computed from only non‑zero edges – used for penalty scaling in SA.
#   3. Pre‑built MIQP model (jijmodeling instance) saved, so Cell 5 only solves.
#   4. Single‑pass geometry, endpoint pre‑filter, tqdm progress.
# =============================================================================
import os
import pickle
import time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import jijmodeling as jm
from tqdm.notebook import tqdm

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

LOCAL_CACHE = Path("/content/wqm_data")
OUTPUT_DIR = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (for global D_max and polygon)
# -----------------------------------------------------------------------------
master_path = OUTPUT_DIR / "master_real.pkl"
if not master_path.exists():
    master_path = LOCAL_CACHE / "master_real.pkl"
with open(master_path, "rb") as f:
    master = pickle.load(f)

coords_master = master["coords"]
U_master = master["U"]
M_indices_master = master["M_indices"]
water_polygon = master["water_polygon"]
M_set_global = set(M_indices_master)

# Compute global D_max from master existing stations
master_M_coords = coords_master[list(M_set_global)]
if len(master_M_coords) > 1:
    M_tree = cKDTree(master_M_coords)
    dists, _ = M_tree.query(master_M_coords, k=2)
    max_nn_dist = np.max(dists[:, 1])
    global_D_max = max_nn_dist * CONFIG_QUBO["D_max_buffer"]
else:
    global_D_max = 15000.0
print(f"🌍 Global D_max = {global_D_max:.2f} m")

shapely.prepare(water_polygon)

# -----------------------------------------------------------------------------
# PAIRWISE TERM CALCULATION (SPARSE)
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
    """
    Returns:
      - linear: dict {orig_idx: coeff}
      - quad_edges: list of (orig_i, orig_j, coeff) with orig_i < orig_j
      - neighbors: dict {orig_idx: list_of_neighbor_orig_indices} (free+fixed)
      - valid_edges: set of (orig_i, orig_j) for connectivity plotting
    """
    N_total = coords.shape[0]
    free_indices = [i for i in range(N_total) if i not in M_indices]
    free_set = set(free_indices)
    M_set = set(M_indices)

    tree = cKDTree(coords)
    v_norm = np.linalg.norm(current_vector)
    v_unit = np.array(current_vector) / v_norm

    # ---- Build connectivity graph (within D_max) ----
    valid_edges = set()
    pairs_dmax = tree.query_pairs(r=global_D_max)
    for i, j in pairs_dmax:
        p1, p2 = coords[i], coords[j]
        # endpoint pre‑filter
        if not (water_polygon.contains(Point(p1)) and water_polygon.contains(Point(p2))):
            continue
        if water_polygon.contains(LineString([p1, p2])):
            valid_edges.add((i, j))

    # Build neighbors dict for free vertices only (including fixed neighbors)
    neighbors = {i: [] for i in free_indices}
    for i, j in valid_edges:
        if i in free_set and j in free_set:
            neighbors[i].append(j)
            neighbors[j].append(i)
        elif i in free_set and j in M_set:
            neighbors[i].append(j)
        elif j in free_set and i in M_set:
            neighbors[j].append(i)
    # Remove duplicates and sort
    for i in neighbors:
        neighbors[i] = sorted(set(neighbors[i]))

    # ---- Compute quadratic interactions (within 2*L_c) ----
    raw_quad = {}
    pairs_within = tree.query_pairs(r=2.0 * L_c)
    for i, j in pairs_within:
        p1, p2 = coords[i], coords[j]
        if not (water_polygon.contains(Point(p1)) and water_polygon.contains(Point(p2))):
            continue
        line = LineString([p1, p2])
        if not water_polygon.contains(line):
            continue

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = np.sqrt(dx*dx + dy*dy)
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
        if abs(coeff) > 1e-12:
            raw_quad[(i, j)] = coeff
            raw_quad[(j, i)] = coeff   # keep symmetric dict for easier absorption

    # ---- Build linear terms (absorb fixed stations) ----
    linear = {}
    for i in free_indices:
        val = -U[i]
        for m in M_indices:
            key = (i, m) if i < m else (m, i)
            if key in raw_quad:
                val += raw_quad[key]
        linear[i] = val

    # ---- Build free‑free quadratic edges (i < j) ----
    quad_edges = []
    for i in free_indices:
        for j in free_indices:
            if i < j:
                key = (i, j)
                if key in raw_quad:
                    val = raw_quad[key]
                    if abs(val) > 1e-12:
                        quad_edges.append((i, j, val))

    return {
        "linear": linear,
        "quad_edges": quad_edges,
        "neighbors": neighbors,
        "valid_edges": valid_edges,
    }

# -----------------------------------------------------------------------------
# MIQP MODEL BUILDER (same as Cell 5, now used here)
# -----------------------------------------------------------------------------
def build_miqp_problem_sparse(N: int, max_degree: int, num_edges: int) -> jm.Problem:
    problem = jm.Problem("WQM_MIQP_sparse", sense=jm.ProblemSense.MINIMIZE)
    a = problem.Placeholder("a", shape=(N,), dtype=jm.DataType.FLOAT)
    neighbor_indices = problem.Placeholder("neighbor_indices", shape=(N, max_degree), dtype=jm.DataType.NATURAL)
    neighbor_mask = problem.Placeholder("neighbor_mask", shape=(N, max_degree), dtype=jm.DataType.BINARY)
    fixed_neighbors = problem.Placeholder("fixed_neighbors", shape=(N,), dtype=jm.DataType.FLOAT)
    K_total = problem.Placeholder("K_total", ndim=0, dtype=jm.DataType.INTEGER)
    x = problem.BinaryVar("x", shape=(N,))
    linear = jm.sum(jm.product(N), lambda i: a[i[0]] * x[i[0]])
    problem += linear
    if num_edges > 0:
        edges = problem.Placeholder("edges", shape=(num_edges, 2), dtype=jm.DataType.NATURAL)
        Q_vals = problem.Placeholder("Q_vals", shape=(num_edges,), dtype=jm.DataType.FLOAT)
        quad = jm.sum(jm.product(num_edges), lambda e: Q_vals[e[0]] * x[edges[e[0], 0]] * x[edges[e[0], 1]])
        problem += quad
    problem += problem.Constraint("budget", jm.sum(jm.product(N), lambda i: x[i[0]]) == K_total)
    problem += problem.Constraint(
        "connectivity",
        lambda i: x[i] <= fixed_neighbors[i] + jm.sum(
            jm.product(max_degree),
            lambda k: neighbor_mask[i, k[0]] * x[neighbor_indices[i, k[0]]]
        ),
        domain=N
    )
    return problem

# -----------------------------------------------------------------------------
# LOAD SCALING RESULTS (from Cell 3)
# -----------------------------------------------------------------------------
scaling_path = OUTPUT_DIR / "scaling_results.pkl"
if not scaling_path.exists():
    scaling_path = LOCAL_CACHE / "scaling_results.pkl"
with open(scaling_path, "rb") as f:
    scaling_results = pickle.load(f)

print("\n" + "=" * 100)
print("🔗 GENERATING SPARSE QUBO + MIQP INSTANCES PER RESOLUTION TIER")
print("=" * 100)

# -----------------------------------------------------------------------------
# MAIN LOOP OVER TIERS
# -----------------------------------------------------------------------------
qubo_results = {}  # for plotting edges

for N, res in tqdm(scaling_results.items(), desc="Tiers"):
    t0 = time.perf_counter()
    coords_tier = res["coords"]
    U_tier = res["U"]
    M_tier = res["M_indices"]          # indices within this tier (already compressed)
    # The tier's M_indices refer to indices in the tier's coordinate array.
    # But we need original indices for the pairwise computation? Actually the helper expects original indices.
    # However, our helper expects indices matching coords array. The tier has its own coords and U.
    # We'll use the tier's own indices directly (the helper will treat them as "original").
    # That's fine because we only need the internal geometry of the tier.
    
    # We'll call helper on the tier's data
    pair_data = compute_pairwise_terms_shape_aware(
        coords=coords_tier,
        U=U_tier,
        M_indices=M_tier,
        water_polygon=water_polygon,
        L_c=CONFIG_QUBO["L_c"],
        L_w=CONFIG_QUBO["L_w"],
        current_vector=CONFIG_QUBO["Current_vector"],
        global_D_max=global_D_max,
        beta=CONFIG_QUBO["Beta"],
        delta=CONFIG_QUBO["Delta"],
    )
    
    linear = pair_data["linear"]
    quad_edges_raw = pair_data["quad_edges"]      # list (orig_i, orig_j, val)
    neighbors_dict = pair_data["neighbors"]        # dict orig_i -> list of orig_j
    valid_edges = pair_data["valid_edges"]         # set of (orig_i, orig_j)
    
    # ---- Compress to free variables only ----
    free_indices_orig = list(neighbors_dict.keys())  # these are the free vertices
    free_set = set(free_indices_orig)
    N_free = len(free_indices_orig)
    orig_to_comp = {orig: idx for idx, orig in enumerate(free_indices_orig)}
    
    # a_new
    a_new = np.array([linear.get(orig, 0.0) for orig in free_indices_orig], dtype=float)
    
    # Q_edges_new (compressed)
    Q_edges_new = []
    for i, j, val in quad_edges_raw:
        if i in free_set and j in free_set:
            Q_edges_new.append((orig_to_comp[i], orig_to_comp[j], val))
    
    # neighbors_new (compressed, free-free only)
    neighbors_new = [[] for _ in range(N_free)]
    for orig_i, nbrs in neighbors_dict.items():
        ci = orig_to_comp[orig_i]
        for orig_j in nbrs:
            if orig_j in free_set:
                neighbors_new[ci].append(orig_to_comp[orig_j])
    # Sort and remove duplicates
    for i in range(N_free):
        neighbors_new[i] = sorted(set(neighbors_new[i]))
    
    # fixed_neighbors (boolean: 1 if any fixed neighbor)
    fixed_neighbors_arr = np.zeros(N_free, dtype=int)
    M_set_tier = set(M_tier)
    for ci, orig_i in enumerate(free_indices_orig):
        if any(nbr in M_set_tier for nbr in neighbors_dict[orig_i]):
            fixed_neighbors_arr[ci] = 1
    
    # ---- Compute qsum for penalty scaling ----
    qsum = np.sum(np.abs(a_new)) + sum(abs(val) for _, _, val in Q_edges_new)
    
    # ---- Build MIQP model ----
    K = CONFIG_QUBO["K"]
    max_degree = max(1, max((len(nbrs) for nbrs in neighbors_new), default=1))
    num_edges = len(Q_edges_new)
    
    # Build data dict for MIQP
    neighbor_indices = np.zeros((N_free, max_degree), dtype=np.int32)
    neighbor_mask = np.zeros((N_free, max_degree), dtype=np.int8)
    for i in range(N_free):
        for k, j in enumerate(neighbors_new[i][:max_degree]):
            neighbor_indices[i, k] = j
            neighbor_mask[i, k] = 1
    
    # For the linear term, we need a_effective = a + diag(Q) but our Q_edges only has off-diagonal, so diag=0.
    # We'll use a_new directly.
    a_effective = a_new.copy()
    # No diagonal Q entries, so no adjustment.
    
    # Build problem
    problem = build_miqp_problem_sparse(N_free, max_degree, num_edges)
    data_dict = {
        "K_total": int(K),
        "a": a_effective.tolist(),
        "neighbor_indices": neighbor_indices.tolist(),
        "neighbor_mask": neighbor_mask.astype(int).tolist(),
        "fixed_neighbors": fixed_neighbors_arr.tolist(),
    }
    if num_edges > 0:
        edges_list = [[i, j] for i, j, _ in Q_edges_new]
        qvals_list = [float(v) for _, _, v in Q_edges_new]
        data_dict["edges"] = edges_list
        data_dict["Q_vals"] = qvals_list
    
    miqp_instance = problem.eval(data_dict)
    
    # ---- Assemble instance data ----
    instance_data = {
        "N": N_free,
        "K": K,
        "a": a_new,                     # linear coefficients (not including diag)
        "Q_edges": Q_edges_new,         # list of (i,j,val)
        "neighbors": neighbors_new,     # list of lists
        "fixed_neighbors": fixed_neighbors_arr,  # boolean array
        "coords": coords_tier[free_indices_orig],
        "U": U_tier[free_indices_orig],
        "original_coords": coords_tier,
        "original_U": U_tier,
        "original_indices": free_indices_orig,
        "fixed_indices": M_tier,
        "snapped_indices": res["snapped_indices"],
        "D_max": global_D_max,
        "L_c": CONFIG_QUBO["L_c"],
        "L_w": CONFIG_QUBO["L_w"],
        "qsum": qsum,                   # for penalty scaling in SA
        "miqp_instance": miqp_instance, # pre-built MIQP model (solved in Cell 5)
        "miqp_data": data_dict,         # optional, for debugging
    }
    
    # Save to local cache and Drive
    save_name = f"instance_data_N{N}.pkl"
    local_path = LOCAL_CACHE / save_name
    drive_path = OUTPUT_DIR / save_name
    with open(local_path, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(drive_path, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Store edges for plotting
    qubo_results[N] = {
        "edges": valid_edges,
        "coords": coords_tier,
        "M_indices": M_tier,
    }
    
    dt = (time.perf_counter() - t0) * 1000
    avg_deg = np.mean([len(nbrs) for nbrs in neighbors_new]) if N_free > 0 else 0
    print(f"✅ N={N:<3d} (Free: {N_free:<3d}) | Avg Deg: {avg_deg:<5.1f} | Q_edges: {len(Q_edges_new):<6d} | qsum: {qsum:.2f} | {dt:.1f} ms")

print("=" * 100)
print("🚀 All QUBO + MIQP instances saved successfully.")

# -----------------------------------------------------------------------------
# PLOT CONNECTIVITY GRAPHS (same as before)
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
    ax.scatter(coords_master[:, 0], coords_master[:, 1], c="lightgray", s=4, alpha=0.35, zorder=0)

    # Plot water-valid edges
    for i, j in edges:
        ax.plot([tier_coords[i, 0], tier_coords[j, 0]],
                [tier_coords[i, 1], tier_coords[j, 1]],
                color="steelblue", alpha=0.3, linewidth=0.6, zorder=1)

    # Nodes
    ax.scatter(tier_coords[:, 0], tier_coords[:, 1], c="black", s=15, alpha=0.8, zorder=2)

    # Fixed stations
    if len(M_tier) > 0:
        ax.scatter(tier_coords[M_tier, 0], tier_coords[M_tier, 1],
                   c="red", marker="*", s=200, edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_title(f"Graph N={N} | D_max={global_D_max:.0f}m", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

for ax in axes[n_tiers:]:
    ax.axis("off")

plt.suptitle("Landmass-Aware Network Topology by Tier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()