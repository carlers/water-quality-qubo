#@title 🗺️ HEXAGONAL RESOLUTION SCALING V2 (SHAPE-AWARE, DENSITY-CONTROLLED)
# =============================================================================
# Key changes vs. the original cell:
#   1. Cells snap to the CENTROID of their real member points (not the ideal
#      hex-grid center), so aggregation never "bridges" across land / narrow
#      waists in an irregular, branching shape like Laguna Bay.
#   2. Existing stations are RESERVED as the representative point of their own
#      cell (and globally deduplicated), so they can never be silently dropped.
#   3. D_max is no longer an arbitrary sqrt(base_N/N) formula. It is derived
#      per-tier from the actual k-th nearest-neighbor distance of the snapped
#      points, targeting a fixed AVERAGE DEGREE. This keeps graph density
#      sane and comparable across all resolutions.
#   4. Hex target size is corrected using an occupancy-grid estimate of how
#      much of the bounding box is actually water, so true_N lands much
#      closer to target_N even for a highly irregular / branching footprint.
# =============================================================================
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    # ---- Resolution tiers ----
    "target_sizes": [20, 50, 100, 200, 500, 800],

    # ---- Graph density control ----
    "target_avg_degree": 8,       # <-- directly controls how dense the adjacency
                                  #     graph looks; replaces the old base_D_max/
                                  #     base_N formula. Raise for denser graphs,
                                  #     lower for sparser ones.
    "min_dmax": 500,
    "fill_frac_grid_n": 80,

    # ---- Existing stations ----
    "N_existing": None,
    "simulate_existing_override": False,
    "min_existing_dist": 12000,
}

# -----------------------------------------------------------------------------
# LOAD MASTER DATA
# -----------------------------------------------------------------------------
master_path = OUTPUT_DIR / "master_real.pkl"
with open(master_path, "rb") as f:
    master = pickle.load(f)

coords = master["coords"]
U = master["U"]
factors = master["factors"]
M_indices_real = master["M_indices"]
bbox = master["metadata"]["bbox"]
n_sites = len(coords)

print("📦 Master data loaded.")
print(f"   Total sites: {n_sites}")
print(f"   Real existing stations: {len(M_indices_real)}")
print(f"   BBox: {bbox}")

# -----------------------------------------------------------------------------
# DETERMINE M_indices FOR THIS RUN (unchanged logic)
# -----------------------------------------------------------------------------
if CONFIG["N_existing"] is None:
    M_indices = M_indices_real
    print(f"   Using real existing stations (n={len(M_indices)})")
else:
    if CONFIG["simulate_existing_override"]:
        print(f"   Simulating {CONFIG['N_existing']} stations (ignoring real ones)...")
        from scipy.spatial.distance import cdist
        n_existing = CONFIG["N_existing"]
        min_dist = CONFIG["min_existing_dist"]
        selected = [int(np.argmax(U))]
        for _ in range(1, n_existing):
            dists = cdist(coords, coords[selected]).min(axis=1)
            mask = (dists >= min_dist) & (~np.isin(np.arange(n_sites), selected))
            valid = np.where(mask)[0]
            if len(valid) == 0:
                best = np.argmax(dists)
            else:
                best = valid[np.argmax(U[valid])]
            selected.append(int(best))
        M_indices = selected
    else:
        n_existing = CONFIG["N_existing"]
        if n_existing > len(M_indices_real):
            print(f"   ⚠️  Override N={n_existing} > real existing ({len(M_indices_real)}). Using all real.")
            M_indices = M_indices_real
        else:
            sorted_real = sorted(M_indices_real, key=lambda i: U[i], reverse=True)
            M_indices = sorted_real[:n_existing]
            print(f"   Using top {len(M_indices)} highest-U real existing stations.")

M_set_global = set(M_indices)

# -----------------------------------------------------------------------------
# HELPER: GENERATE UNIFORM HEX GRID OVER BBOX
# -----------------------------------------------------------------------------
def generate_hex_centroids_bbox(bbox, target_n):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    hex_area = area / target_n
    spacing = np.sqrt(hex_area / (np.sqrt(3) / 2.0))
    dx = spacing
    dy = spacing * (np.sqrt(3) / 2.0)

    cols = int(np.ceil(width / dx)) + 1
    rows = int(np.ceil(height / dy)) + 1

    centroids = []
    for r in range(rows):
        y = ymin + r * dy
        if y > ymax:
            continue
        shift = 0.5 * dx if (r % 2 == 1) else 0.0
        for c in range(cols):
            x = xmin + c * dx + shift
            if x > xmax:
                continue
            centroids.append([x, y])
    return np.array(centroids)

# -----------------------------------------------------------------------------
# HELPER: ESTIMATE WHAT FRACTION OF THE BBOX IS ACTUALLY WATER
# -----------------------------------------------------------------------------
def estimate_fill_fraction(coords, bbox, grid_n=80):
    """
    Occupancy-grid estimate of how much of the bounding box the real,
    irregular/branching footprint actually covers. Used to correct the hex
    grid's target size so true_N lands close to the requested target_N even
    when the shape (e.g. a bay with a narrow waist) is far from a rectangle.
    A convex hull would over-estimate this for concave shapes, so we use a
    direct occupancy count instead.
    """
    xmin, ymin, xmax, ymax = bbox
    xs = np.linspace(xmin, xmax, grid_n + 1)
    ys = np.linspace(ymin, ymax, grid_n + 1)
    H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[xs, ys])
    occupied = np.sum(H > 0)
    total = grid_n * grid_n
    return float(np.clip(occupied / total, 0.03, 1.0))

fill_frac = estimate_fill_fraction(coords, bbox, CONFIG["fill_frac_grid_n"])
print(f"   Estimated water-coverage fraction of bbox: {fill_frac:.3f}")

# -----------------------------------------------------------------------------
# RESOLUTION SCALING LOOP
# -----------------------------------------------------------------------------
print("\n" + "=" * 100)
print("🚀 Running Hexagonal Resolution Scaling V2 (Shape-Aware, Density-Controlled)")
print("=" * 100)
print(f"{'Target N':<10} | {'Actual N':<10} | {'D_max (m)':<12} | {'Avg Degree':<10} | {'Target Deg':<10} | {'Utility Mass':<12}")
print("=" * 100)

scaling_results = {}
target_sizes = CONFIG["target_sizes"]
target_avg_degree = CONFIG["target_avg_degree"]
min_dmax = CONFIG["min_dmax"]

real_tree = cKDTree(coords)

for target_N in sorted(target_sizes):
    t0 = time.perf_counter()

    # 1. Correct the hex-grid target using the estimated water fraction, so
    #    the number of OCCUPIED cells (true_N) lands close to target_N.
    target_N_grid = int(np.ceil(target_N / fill_frac))
    target_N_grid = min(target_N_grid, target_N * 25)  # sanity cap

    hex_coords = generate_hex_centroids_bbox(bbox, target_N_grid)

    # 2. Assign each REAL coordinate to its nearest hex center
    hex_tree = cKDTree(hex_coords)
    _, assigned_hex_idx = hex_tree.query(coords)
    unique_cells = np.unique(assigned_hex_idx)
    true_N = len(unique_cells)

    # 3. Representative-point selection per cell:
    #    - if the cell contains a real existing station, that station IS the
    #      representative (reserved + deduped globally)
    #    - otherwise, snap to the nearest UNUSED real coordinate to the
    #      cell's actual member centroid (not the empty hex center)
    used_master_idx = set()
    snapped_indices = np.zeros(true_N, dtype=int)
    agg_factors = np.zeros((true_N, 8), dtype=np.float64)
    agg_U = np.zeros(true_N, dtype=np.float64)

    for i, cell in enumerate(unique_cells):
        member_idx = np.where(assigned_hex_idx == cell)[0]
        agg_factors[i] = np.nanmean(factors[member_idx], axis=0)
        agg_U[i] = np.sum(U[member_idx])

        station_members = [m for m in member_idx if m in M_set_global]
        if station_members:
            rep = int(station_members[0])
        else:
            centroid = coords[member_idx].mean(axis=0)
            k_search = 8
            rep = None
            while rep is None:
                k = min(k_search, n_sites)
                dists, cand_idx = real_tree.query(centroid, k=k)
                cand_idx = np.atleast_1d(cand_idx)
                for c in cand_idx:
                    if c not in used_master_idx:
                        rep = int(c)
                        break
                if rep is None:
                    if k_search >= n_sites:
                        rep = int(np.atleast_1d(cand_idx)[0])  # give up on dedup
                        break
                    k_search *= 2

        used_master_idx.add(rep)
        snapped_indices[i] = rep

    coords_tier = coords[snapped_indices]

    # 4. Map existing stations into this tier
    M_tier = [i for i, orig_idx in enumerate(snapped_indices) if orig_idx in M_set_global]
    M_orig_tier = [int(orig_idx) for orig_idx in snapped_indices if orig_idx in M_set_global]

    # 5. Data-driven D_max: target a fixed AVERAGE DEGREE using the actual
    #    k-th nearest-neighbor distance of the snapped points, instead of an
    #    arbitrary sqrt(base_N/N) formula unrelated to real spacing.
    tier_tree = cKDTree(coords_tier)
    k = min(target_avg_degree, true_N - 1)
    if k >= 1:
        knn_dists, _ = tier_tree.query(coords_tier, k=k + 1)  # includes self at col 0
        scaled_dmax = max(min_dmax, float(np.median(knn_dists[:, -1])))
    else:
        scaled_dmax = min_dmax

    # 6. Build adjacency at that D_max
    pairs = tier_tree.query_pairs(r=scaled_dmax)
    num_edges = len(pairs)
    avg_degree = (2.0 * num_edges) / true_N if true_N > 0 else 0

    adj_matrix = np.zeros((true_N, true_N), dtype=int)
    for i, j in pairs:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    total_utility = agg_U.sum()

    print(f"{target_N:<10} | {true_N:<10} | {scaled_dmax:<12.2f} | {avg_degree:<10.2f} | "
          f"{target_avg_degree:<10} | {total_utility:<12.4f}")

    scaling_results[true_N] = {
        "coords": coords_tier,
        "U": agg_U,
        "factors": agg_factors,
        "adj_matrix": adj_matrix,
        "dmax": scaled_dmax,
        "edges": pairs,
        "avg_degree": avg_degree,
        "M_indices": M_tier,
        "M_orig_indices": M_orig_tier,
        "snapped_indices": snapped_indices,
    }

print("=" * 100)

# -----------------------------------------------------------------------------
# GENERATE QUBO INSTANCES PER TIER (unchanged from original)
# -----------------------------------------------------------------------------
try:
    from src.model import compute_pairwise_terms
    from src.utils import compute_energy  # optional
    print("\n✅ Imported QUBO builder functions.")
except ImportError:
    print("\n⚠️  Could not import compute_pairwise_terms. Skipping QUBO generation.")
    compute_pairwise_terms = None

if compute_pairwise_terms is not None:
    print("\n🔗 Generating QUBO instances for each tier...")
    for N, res in scaling_results.items():
        pairwise = compute_pairwise_terms(
            coords=res["coords"],
            U=res["U"],
            M_indices=res["M_indices"],
            L_c=CONFIG["L_c"],
            L_w=CONFIG["L_w"],
            current_vector=CONFIG["Current_vector"],
            beta=CONFIG["Beta"],
            delta=CONFIG["Delta"],
            connectivity_range=res["dmax"],
            verbose=False,
        )
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
            "K": CONFIG["K"],
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
            "L_c": CONFIG["L_c"],
            "L_w": CONFIG["L_w"],
        }

        instance_path = OUTPUT_DIR / f"instance_data_N{N}.pkl"
        with open(instance_path, "wb") as f:
            pickle.dump(instance_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✅ Saved QUBO instance for N={N} to {instance_path}")
# -----------------------------------------------------------------------------
# DYNAMIC PLOTTING (all tiers) — real-data footprint shown faintly underneath
# so you can visually confirm the tiers follow the true shape
# -----------------------------------------------------------------------------
print("\n📊 Plotting resolution scaling results...")
tiers = sorted(scaling_results.keys())
n_tiers = len(tiers)
cols = 2
rows = int(np.ceil(n_tiers / cols))

fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
axes = np.atleast_1d(axes).flatten()

for ax, N in zip(axes, tiers):
    res = scaling_results[N]
    coords_tier = res["coords"]
    U_tier = res["U"]
    edges = res["edges"]
    dmax = res["dmax"]
    M_tier = res["M_indices"]

    # Faint real-data footprint for shape reference
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=4, alpha=0.35, zorder=0)

    for i, j in edges:
        ax.plot([coords_tier[i, 0], coords_tier[j, 0]],
                 [coords_tier[i, 1], coords_tier[j, 1]],
                 color="gray", alpha=0.15, linewidth=0.5, zorder=1)

    # MODIFIED: Removed dynamic size array. Set s=60 for uniform blob sizes.
    sc = ax.scatter(coords_tier[:, 0], coords_tier[:, 1],
                     c=U_tier, cmap="viridis", s=60,
                     edgecolor="k", linewidth=0.3, alpha=0.9, zorder=2)

    if len(M_tier) > 0:
        ax.scatter(coords_tier[M_tier, 0], coords_tier[M_tier, 1],
                    c="red", marker="*", s=250, edgecolor="black",
                    linewidth=0.5, zorder=5, label="Existing")

    ax.set_title(f"N={N} | Dmax={dmax:.0f}m | Avg Deg={res['avg_degree']:.1f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.1)

for ax in axes[n_tiers:]:
    ax.axis("off")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Utility U_i (mass conserved)")

plt.suptitle("Real Data: Hexagonal Resolution Scaling V2 (Shape-Aware)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

print("✅ Done.")