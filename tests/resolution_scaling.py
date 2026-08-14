#@title CELL 3: NESTED RESOLUTION SCALING (Farthest Point Sampling)
"""
================================================================================
NESTED RESOLUTION SCALING VIA FARTHEST POINT SAMPLING (FPS)
================================================================================
- Generates a dense pool of candidate points over the lake.
- Maps each pool point to the nearest master centroid (from the full 5417 grid).
- Uses Farthest Point Sampling (seeded with existing stations) to select exactly 
  1000 well‑spread free candidates in a strict ordering.
- Nested subsets: N=20 ⊂ N=50 ⊂ N=100 ⊂ N=200 ⊂ N=500 ⊂ N=1000.
- Exact target sizes, monotonic master energy, clean spatial layout.
================================================================================
"""

import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import shapely
from shapely.geometry import Point
from tqdm.notebook import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "target_sizes": [20, 50, 100, 200, 500, 1000],   # Exact N for each tier
    "pool_size_estimate": 10000,                     # Rough number of hex centroids to generate
    "N_existing": None,                              # None = use real existing stations
}

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (from Cell 2)
# -----------------------------------------------------------------------------
master_path = OUTPUT_DIR / "master_real.pkl"
if not master_path.exists():
    master_path = LOCAL_CACHE / "master_real.pkl"
with open(master_path, "rb") as f:
    master = pickle.load(f)

coords = master["coords"]          # (5417, 2)
U = master["U"]                    # (5417,)
water_polygon = master["water_polygon"]
M_indices = master["M_indices"]    # list of existing station indices
bbox = master["metadata"]["bbox"]
n_sites = len(coords)
M_set_global = set(M_indices)

print("📦 Master data loaded.")
print(f"   Total master sites: {n_sites}")
print(f"   Existing stations: {len(M_indices)}")
print(f"   BBox: {bbox}")

# -----------------------------------------------------------------------------
# STEP 1: GENERATE A DENSE POOL OF HEX CENTROIDS OVER THE LAKE
# -----------------------------------------------------------------------------
def generate_hex_centroids_bbox(bbox, target_n):
    """Generate hex grid centroids covering the bounding box."""
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

print("\n🔹 Generating dense hex pool...")
pool_raw = generate_hex_centroids_bbox(bbox, CONFIG["pool_size_estimate"])
print(f"   Raw pool size: {len(pool_raw):,}")

# Filter to keep only points strictly inside the water polygon
shapely.prepare(water_polygon)
pool_water = []
for pt in tqdm(pool_raw, desc="Filtering water points"):
    if water_polygon.contains(Point(pt)):
        pool_water.append(pt)
pool_coords = np.array(pool_water, dtype=float)
print(f"   Pool points inside water: {len(pool_coords):,}")

# -----------------------------------------------------------------------------
# STEP 2: MAP EACH POOL POINT TO THE NEAREST MASTER CENTROID
# -----------------------------------------------------------------------------
master_tree = cKDTree(coords)
dist, pool_to_master = master_tree.query(pool_coords, k=1)
pool_to_master = pool_to_master.flatten().astype(int)

# Exclude pool points that map to an existing station
free_mask = ~np.isin(pool_to_master, list(M_set_global))
pool_coords_free = pool_coords[free_mask]
pool_to_master_free = pool_to_master[free_mask]
print(f"   Pool points mapped to free master indices: {len(pool_coords_free):,}")

# Retrieve the master coordinates and utility for each pool point
pool_master_coords = coords[pool_to_master_free]
pool_U = U[pool_to_master_free]

# -----------------------------------------------------------------------------
# STEP 3: FARTHEST POINT SAMPLING (Seeded with existing stations)
# -----------------------------------------------------------------------------
print("\n🔹 Running Farthest Point Sampling (seeded with existing stations)...")

# Seeds: coordinates of existing stations
seed_coords = coords[list(M_set_global)]

# Number of free points to select = max target size (1000)
n_select = 1000

# KDTree for seeds to compute initial distances
seed_tree = cKDTree(seed_coords)
# For each pool point, compute distance to the nearest seed
min_dist = seed_tree.query(pool_coords_free, k=1)[0].flatten()

selected_indices = []       # indices in pool_coords_free
selected_coords = []        # corresponding master coordinates
selected_master_indices = []  # master indices

# We'll also keep track of all pool points and update distances
for _ in tqdm(range(n_select), desc="FPS iterations"):
    # Find the pool point with largest minimum distance
    idx = np.argmax(min_dist)
    selected_indices.append(idx)
    selected_coords.append(pool_master_coords[idx])
    selected_master_indices.append(pool_to_master_free[idx])
    
    # Update min_dist for all remaining points
    # Compute distance from newly selected point to all others
    new_point = pool_master_coords[idx]
    # We could compute distances from new_point to all pool points using KDTree,
    # but for simplicity we compute Euclidean distances directly (pool size ~6000, OK)
    dist_to_new = np.linalg.norm(pool_master_coords - new_point, axis=1)
    # Update min_dist = min(min_dist, dist_to_new)
    min_dist = np.minimum(min_dist, dist_to_new)
    
    # Set the distance of selected point to -1 to avoid reselection
    min_dist[idx] = -1.0

# The order of selection gives the nested ordering.
# selected_master_indices now contains exactly 1000 free master indices,
# ordered by farthest-point sampling.

print(f"   Selected {len(selected_master_indices)} free points via FPS.")

# -----------------------------------------------------------------------------
# STEP 4: BUILD NESTED TIERS FOR EACH TARGET N
# -----------------------------------------------------------------------------
print("\n🔹 Building nested tiers for exact target sizes...")

scaling_results = {}

for N in CONFIG["target_sizes"]:
    # Take the first N free points from the FPS order
    free_master_indices = selected_master_indices[:N]
    free_coords = coords[free_master_indices]
    free_U = U[free_master_indices]
    
    # Combine with fixed stations (existing)
    # We'll create a combined list: first all free, then fixed stations
    all_master_indices = list(free_master_indices) + M_indices
    all_coords = np.vstack([free_coords, coords[M_indices]])
    all_U = np.concatenate([free_U, U[M_indices]])
    
    # M_indices for this tier: indices of the fixed stations in the combined array
    # Free indices are 0..N-1, fixed are N..N+len(M_indices)-1
    M_tier = list(range(N, N + len(M_indices)))
    
    # snapped_indices: master indices corresponding to each entry in all_coords
    snapped_indices = np.array(all_master_indices, dtype=int)
    
    # Store
    scaling_results[N] = {
        "coords": all_coords,
        "U": all_U,
        "M_indices": M_tier,
        "M_orig_indices": M_indices,   # original master indices of fixed stations
        "snapped_indices": snapped_indices,
        "free_master_indices": free_master_indices,   # for reference
    }
    
    print(f"   N={N:4d}: free={N}, fixed={len(M_indices)}, total={len(all_coords)}")

# -----------------------------------------------------------------------------
# STEP 5: PLOT THE NESTED CANDIDATE GRIDS
# -----------------------------------------------------------------------------
print("\n📊 Plotting nested candidate grids...")
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
    
    # Candidates (free + fixed)
    sc = ax.scatter(coords_tier[:, 0], coords_tier[:, 1],
                    c=U_tier, cmap="viridis", s=60,
                    edgecolor="k", linewidth=0.3, alpha=0.9, zorder=2)
    
    # Existing stations (fixed)
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
fig.colorbar(sm, cax=cbar_ax, label="Utility U_i")

plt.suptitle("Nested Resolution Scaling (Farthest Point Sampling)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

# -----------------------------------------------------------------------------
# STEP 6: SAVE RESULTS (CACHED)
# -----------------------------------------------------------------------------
cached_path = OUTPUT_DIR / "scaling_results.pkl"
if not cached_path.exists():
    cached_path = LOCAL_CACHE / "scaling_results.pkl"
with open(cached_path, "wb") as f:
    pickle.dump(scaling_results, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Cached scaling results to {cached_path}")

print("\n✅ Nested resolution scaling complete.")