#@title CELL 4: GENERATE SPARSE QUBO + MIQP DATA (Resumable, Fixed Plotting)
"""
================================================================================
REVISION: Resumable – loads per‑tier instance_data_N{N}.pkl if available.
- Uses FORCE_RECOMPUTE_QUBO flag from Cell 1.
- Dual storage: local and Drive.
- Stores valid_edges inside the instance data for connectivity plots.
- Plotting correctly uses original_coords (full combined array) from loaded data.
================================================================================
"""
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
import shutil
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION (reuse from Cell 2)
# -----------------------------------------------------------------------------
# MASTER_QUBO_CONFIG is defined in Cell 2; we use it directly.
# If it's not defined, fall back to a default (should not happen).
try:
    CONFIG_QUBO = MASTER_QUBO_CONFIG
except NameError:
    print("⚠️ MASTER_QUBO_CONFIG not found; using default config.")
    CONFIG_QUBO = {
        "L_c": 7500.0,
        "L_w": 1000.0,
        "Beta": 1.0,
        "Delta": 1.0,
        "Current_vector": (1.0, 0.0),
        "K": 5,
        "D_max_buffer": 1.15,
    }

LOCAL_CACHE = Path("/content/wqm_data")
OUTPUT_DIR = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (already in memory from Cell 2)
# -----------------------------------------------------------------------------
try:
    # Variables from Cell 2
    coords_master = coords
    U_master = U
    M_indices_master = M_indices
    water_polygon = water_polygon
    global_D_max = metadata["master_D_max"]
except NameError:
    print("⚠️ Master data not in namespace; loading from file...")
    master_path_local = LOCAL_CACHE / "master_real.pkl"
    if not master_path_local.exists():
        master_path_local = OUTPUT_DIR / "master_real.pkl"
    with open(master_path_local, "rb") as f:
        master_data = pickle.load(f)
    coords_master = master_data["coords"]
    U_master = master_data["U"]
    M_indices_master = master_data["M_indices"]
    water_polygon = master_data["water_polygon"]
    a_master = master_data["a"]
    Q_master_edges = master_data["Q_edges"]
    metadata = master_data["metadata"]
    global_D_max = metadata["master_D_max"]
    # also define coords for backward compatibility
    coords = coords_master
    U = U_master
    M_indices = M_indices_master

M_set_global = set(M_indices_master)

# -----------------------------------------------------------------------------
# LOAD SCALING RESULTS (already in memory from Cell 3)
# -----------------------------------------------------------------------------
try:
    if 'scaling_results' not in dir():
        raise NameError
except NameError:
    print("⚠️ Scaling results not in namespace; loading from file...")
    scaling_path_local = LOCAL_CACHE / "scaling_results.pkl"
    if not scaling_path_local.exists():
        scaling_path_local = OUTPUT_DIR / "scaling_results.pkl"
    with open(scaling_path_local, "rb") as f:
        scaling_results = pickle.load(f)

# -----------------------------------------------------------------------------
# PAIRWISE TERM CALCULATION (unchanged)
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
            raw_quad[(j, i)] = coeff

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
# MIQP MODEL BUILDER (unchanged)
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
# MAIN LOOP OVER TIERS (Resumable)
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("🔗 GENERATING SPARSE QUBO + MIQP DATA PER RESOLUTION TIER (Resumable)")
print("=" * 100)

qubo_results = {}  # for plotting
tiers = sorted(scaling_results.keys())

for N in tqdm(tiers, desc="Tiers"):
    instance_name = f"instance_data_N{N}.pkl"
    instance_local = LOCAL_CACHE / instance_name
    instance_drive = OUTPUT_DIR / instance_name

    # 1. Check if instance already exists and we are allowed to load
    loaded = False
    if not FORCE_RECOMPUTE_QUBO:
        if instance_local.exists():
            with open(instance_local, "rb") as f:
                instance_data = pickle.load(f)
            loaded = True
        elif instance_drive.exists():
            print(f"📂 Loading {instance_name} from Drive (copying to local)...")
            shutil.copy(instance_drive, instance_local)
            with open(instance_local, "rb") as f:
                instance_data = pickle.load(f)
            loaded = True

    if loaded:
        print(f"✅ Loaded instance data for N={N}")
        # Extract data for plotting
        # Use original_coords (full combined array) for plotting
        # Fallback: if original_coords missing, reconstruct from coords and fixed_indices
        if "original_coords" in instance_data:
            coords_plot = instance_data["original_coords"]
        else:
            # Reconstruct full coordinates: free + fixed
            free_coords = instance_data["coords"]
            fixed_indices = instance_data["fixed_indices"]
            fixed_coords = instance_data.get("original_coords_full", None)
            if fixed_coords is None:
                # approximate: use the original master coordinates for fixed stations? Not ideal.
                # but we can't easily reconstruct; warn and use free coords only.
                print(f"⚠️ Cannot reconstruct full coords for N={N}; using free coords only.")
                coords_plot = free_coords
            else:
                coords_plot = np.vstack([free_coords, fixed_coords])
        M_tier = instance_data["fixed_indices"]
        valid_edges = instance_data.get("valid_edges", set())
        qubo_results[N] = {
            "edges": valid_edges,
            "coords": coords_plot,
            "M_indices": M_tier,
        }
        continue

    # 2. Compute from scratch
    print(f"🔄 Computing instance for N={N}...")
    t0 = time.perf_counter()

    res = scaling_results[N]
    coords_tier = res["coords"]
    U_tier = res["U"]
    M_tier = res["M_indices"]          # indices within this tier (compressed)
    snapped_indices = res["snapped_indices"]

    # Call helper on the tier's data
    pair_data = compute_pairwise_terms_shape_aware(
        coords=coords_tier,
        U=U_tier,
        M_indices=M_tier,
        water_polygon=water_polygon,
        L_c=MASTER_QUBO_CONFIG["L_c"],
        L_w=MASTER_QUBO_CONFIG["L_w"],
        current_vector=MASTER_QUBO_CONFIG["Current_vector"],
        global_D_max=global_D_max,
        beta=MASTER_QUBO_CONFIG["Beta"],
        delta=MASTER_QUBO_CONFIG["Delta"],
    )

    linear = pair_data["linear"]
    quad_edges_raw = pair_data["quad_edges"]
    neighbors_dict = pair_data["neighbors"]
    valid_edges = pair_data["valid_edges"]

    # Compress to free variables only
    free_indices_orig = list(neighbors_dict.keys())
    free_set = set(free_indices_orig)
    N_free = len(free_indices_orig)
    orig_to_comp = {orig: idx for idx, orig in enumerate(free_indices_orig)}

    a_new = np.array([linear.get(orig, 0.0) for orig in free_indices_orig], dtype=float)

    Q_edges_new = []
    for i, j, val in quad_edges_raw:
        if i in free_set and j in free_set:
            Q_edges_new.append((orig_to_comp[i], orig_to_comp[j], val))

    neighbors_new = [[] for _ in range(N_free)]
    for orig_i, nbrs in neighbors_dict.items():
        ci = orig_to_comp[orig_i]
        for orig_j in nbrs:
            if orig_j in free_set:
                neighbors_new[ci].append(orig_to_comp[orig_j])
    for i in range(N_free):
        neighbors_new[i] = sorted(set(neighbors_new[i]))

    fixed_neighbors_arr = np.zeros(N_free, dtype=int)
    M_set_tier = set(M_tier)
    for ci, orig_i in enumerate(free_indices_orig):
        if any(nbr in M_set_tier for nbr in neighbors_dict[orig_i]):
            fixed_neighbors_arr[ci] = 1

    # qsum for penalty scaling
    qsum = np.sum(np.abs(a_new)) + sum(abs(val) for _, _, val in Q_edges_new)

    K = MASTER_QUBO_CONFIG["K"]
    max_degree = max(1, max((len(nbrs) for nbrs in neighbors_new), default=1))
    num_edges = len(Q_edges_new)

    neighbor_indices = np.zeros((N_free, max_degree), dtype=np.int32)
    neighbor_mask = np.zeros((N_free, max_degree), dtype=np.int8)
    for i in range(N_free):
        for k, j in enumerate(neighbors_new[i][:max_degree]):
            neighbor_indices[i, k] = j
            neighbor_mask[i, k] = 1

    a_effective = a_new.copy()

    # Build MIQP data dict (for Cell 5)
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

    # Assemble full instance_data
    instance_data = {
        "N": N_free,
        "K": K,
        "a": a_new,
        "Q_edges": Q_edges_new,
        "neighbors": neighbors_new,
        "fixed_neighbors": fixed_neighbors_arr,
        "coords": coords_tier[free_indices_orig],
        "U": U_tier[free_indices_orig],
        "original_coords": coords_tier,          # full combined coordinates (free + fixed)
        "original_U": U_tier,
        "original_indices": free_indices_orig,
        "fixed_indices": M_tier,
        "snapped_indices": snapped_indices,
        "D_max": global_D_max,
        "L_c": MASTER_QUBO_CONFIG["L_c"],
        "L_w": MASTER_QUBO_CONFIG["L_w"],
        "qsum": qsum,
        "valid_edges": valid_edges,          # store for connectivity plot
        "miqp_data": data_dict,
        "miqp_params": {
            "max_degree": max_degree,
            "num_edges": num_edges,
        }
    }

    # Save to local and Drive
    with open(instance_local, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(instance_drive, "wb") as f:
        pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Store for plotting
    qubo_results[N] = {
        "edges": valid_edges,
        "coords": coords_tier,
        "M_indices": M_tier,
    }

    dt = (time.perf_counter() - t0) * 1000
    avg_deg = np.mean([len(nbrs) for nbrs in neighbors_new]) if N_free > 0 else 0
    print(f"✅ N={N:<3d} (Free: {N_free:<3d}) | Avg Deg: {avg_deg:<5.1f} | Q_edges: {len(Q_edges_new):<6d} | qsum: {qsum:.2f} | {dt:.1f} ms")

print("=" * 100)
print("🚀 All QUBO + MIQP data saved successfully.")

# -----------------------------------------------------------------------------
# PLOT CONNECTIVITY GRAPHS (from loaded/computed data)
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

    ax.scatter(coords_master[:, 0], coords_master[:, 1], c="lightgray", s=4, alpha=0.35, zorder=0)

    for i, j in edges:
        ax.plot([tier_coords[i, 0], tier_coords[j, 0]],
                [tier_coords[i, 1], tier_coords[j, 1]],
                color="steelblue", alpha=0.3, linewidth=0.6, zorder=1)

    ax.scatter(tier_coords[:, 0], tier_coords[:, 1], c="black", s=15, alpha=0.8, zorder=2)
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

print("✅ Cell 4 complete.")