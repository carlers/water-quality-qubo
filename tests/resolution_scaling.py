#@title CELL 3: HEXAGONAL RESOLUTION SCALING V3 (Idempotent + tqdm)
# =============================================================================
# Key features:
#   1. Idempotent: Loads cached scaling_results.pkl if available.
#   2. Cells snap to the CENTROID of their real member points.
#   3. Existing stations are RESERVED as representative points of their cells.
#   4. Utility (U) is aggregated using np.mean() and normalized against total master 
#      sites (n_sites / true_N) to conserve total master energy scale across tiers.
#   5. Graph connectivity (D_max and landmass-aware edge generation) is deferred
#      entirely to the QUBO Instance Builder cell.
#   6. tqdm progress bars for transparency.
# =============================================================================
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import shapely
from tqdm.notebook import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "target_sizes": [20, 50, 100, 200, 500, 800],  # Desired candidate counts
    "fill_frac_grid_n": 80,                         # Grid resolution for water fraction estimate
    "N_existing": None,                             # None = use real existing; or int to override
    "simulate_existing_override": False,            # If True, simulate stations even if real exist
    "min_existing_dist": 12000,                     # For fallback simulation
}

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (from the new Cell 2)
# -----------------------------------------------------------------------------
master_path = OUTPUT_DIR / "master_real.pkl"
if not master_path.exists():
    raise FileNotFoundError(f"❌ Master data not found at {master_path}. Run Cell 2 first.")

with open(master_path, "rb") as f:
    master = pickle.load(f)

coords = master["coords"]
U = master["U"]
factors = master["factors"]
M_indices_real = master["M_indices"]
water_polygon = master["water_polygon"]
bbox = master["metadata"]["bbox"]
n_sites = len(coords)

shapely.prepare(water_polygon)

print("📦 Master data loaded.")
print(f"   Total sites (N_master): {n_sites}")
print(f"   Real existing stations: {len(M_indices_real)}")
print(f"   Water polygon loaded: {type(water_polygon)}")
print(f"   BBox: {bbox}")

# -----------------------------------------------------------------------------
# DETERMINE M_indices FOR THIS RUN
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
            print(f"   ⚠️ Override N={n_existing} > real existing ({len(M_indices_real)}). Using all real.")
            M_indices = M_indices_real
        else:
            sorted_real = sorted(M_indices_real, key=lambda i: U[i], reverse=True)
            M_indices = sorted_real[:n_existing]
            print(f"   Using top {len(M_indices)} highest-U real existing stations.")

M_set_global = set(M_indices)

# -----------------------------------------------------------------------------
# CHECK FOR CACHED RESULT (Idempotency)
# -----------------------------------------------------------------------------
cached_path = OUTPUT_DIR / "scaling_results.pkl"
if cached_path.exists():
    print(f"📂 Found cached scaling results at {cached_path}. Loading...")
    with open(cached_path, "rb") as f:
        scaling_results = pickle.load(f)
    print(f"   ✅ Loaded {len(scaling_results)} tiers: {sorted(scaling_results.keys())}")
else:
    print("🔄 No cache found. Generating grids from scratch...")
    
    # -------------------------------------------------------------------------
    # HELPER: GENERATE UNIFORM HEX GRID OVER BBOX
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # HELPER: ESTIMATE WATER FRACTION
    # -------------------------------------------------------------------------
    def estimate_fill_fraction(coords, bbox, grid_n=80):
        xmin, ymin, xmax, ymax = bbox
        xs = np.linspace(xmin, xmax, grid_n + 1)
        ys = np.linspace(ymin, ymax, grid_n + 1)
        H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[xs, ys])
        occupied = np.sum(H > 0)
        total = grid_n * grid_n
        return float(np.clip(occupied / total, 0.03, 1.0))

    fill_frac = estimate_fill_fraction(coords, bbox, CONFIG["fill_frac_grid_n"])
    print(f"   Estimated water-coverage fraction of bbox: {fill_frac:.3f}")

    # -------------------------------------------------------------------------
    # RESOLUTION SCALING LOOP (with tqdm)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("🚀 Running Hexagonal Resolution Scaling (Shape-Aware, Energy-Conserved)")
    print("=" * 90)
    print(f"{'Target N':<10} | {'Actual N':<10} | {'Mean Cell U':<12} | {'Total System Utility':<20}")
    print("=" * 90)

    scaling_results = {}
    target_sizes = CONFIG["target_sizes"]
    real_tree = cKDTree(coords)

    for target_N in tqdm(sorted(target_sizes), desc="Generating hex grids"):
        t0 = time.perf_counter()

        # 1. Correct the hex-grid target using estimated water fraction
        target_N_grid = int(np.ceil(target_N / fill_frac))
        target_N_grid = min(target_N_grid, target_N * 25)

        hex_coords = generate_hex_centroids_bbox(bbox, target_N_grid)

        # 2. Assign REAL coordinates to nearest hex center
        hex_tree = cKDTree(hex_coords)
        _, assigned_hex_idx = hex_tree.query(coords)
        unique_cells = np.unique(assigned_hex_idx)
        true_N = len(unique_cells)

        # 3. Representative-point selection per cell & Energy scaling
        used_master_idx = set()
        snapped_indices = np.zeros(true_N, dtype=int)
        agg_factors = np.zeros((true_N, 8), dtype=np.float64)
        agg_U = np.zeros(true_N, dtype=np.float64)

        # Scaling factor to conserve master total energy scale
        tier_energy_scale = n_sites / true_N

        for i, cell in enumerate(unique_cells):
            member_idx = np.where(assigned_hex_idx == cell)[0]
            agg_factors[i] = np.nanmean(factors[member_idx], axis=0)
            agg_U[i] = np.mean(U[member_idx]) * tier_energy_scale

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
                            rep = int(np.atleast_1d(cand_idx)[0])
                            break
                        k_search *= 2

            used_master_idx.add(rep)
            snapped_indices[i] = rep

        coords_tier = coords[snapped_indices]

        # 4. Map existing stations into this tier
        M_tier = [i for i, orig_idx in enumerate(snapped_indices) if orig_idx in M_set_global]
        M_orig_tier = [int(orig_idx) for orig_idx in snapped_indices if orig_idx in M_set_global]

        total_utility = agg_U.sum()
        mean_utility = agg_U.mean()

        print(f"{target_N:<10} | {true_N:<10} | {mean_utility:<12.4f} | {total_utility:<20.4f}")

        scaling_results[true_N] = {
            "coords": coords_tier,
            "U": agg_U,
            "factors": agg_factors,
            "M_indices": M_tier,
            "M_orig_indices": M_orig_tier,
            "snapped_indices": snapped_indices,
        }

    print("=" * 90)

    # -------------------------------------------------------------------------
    # SAVE CACHED RESULTS
    # -------------------------------------------------------------------------
    with open(cached_path, "wb") as f:
        pickle.dump(scaling_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Cached scaling results to {cached_path}")

# -----------------------------------------------------------------------------
# PLOTTING (Candidate grids per tier)
# -----------------------------------------------------------------------------
print("\n📊 Plotting resolution scaling candidate grids...")
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
    M_tier = res["M_indices"]

    # Faint master footprint
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=4, alpha=0.35, zorder=0)

    sc = ax.scatter(coords_tier[:, 0], coords_tier[:, 1],
                     c=U_tier, cmap="viridis", s=60,
                     edgecolor="k", linewidth=0.3, alpha=0.9, zorder=2)

    if len(M_tier) > 0:
        ax.scatter(coords_tier[M_tier, 0], coords_tier[M_tier, 1],
                    c="red", marker="*", s=250, edgecolor="black",
                    linewidth=0.5, zorder=5, label="Existing")

    ax.set_title(f"N={N} | Total U={U_tier.sum():.1f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.1)

for ax in axes[n_tiers:]:
    ax.axis("off")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min(r['U'].min() for r in scaling_results.values()), 
                                                              vmax=max(r['U'].max() for r in scaling_results.values())))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Scaled Utility U_i")

plt.suptitle("Real Data: Hexagonal Candidate Grids (Energy-Conserved)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

print("✅ Candidate grid generation complete.")