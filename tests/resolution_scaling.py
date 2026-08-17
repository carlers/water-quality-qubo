#@title CELL 3: NESTED RESOLUTION SCALING (Resumable)
"""
================================================================================
NESTED RESOLUTION SCALING VIA FARTHEST POINT SAMPLING (FPS) – RESUMABLE
================================================================================
- Uses FORCE_RECOMPUTE_SCALING flag from Cell 1.
- Dual storage: local and Drive.
- If scaling_results.pkl exists, loads and skips FPS.
- Generates nested grid plots from loaded data.
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
import shutil

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "target_sizes": [20, 50, 100, 200, 500, 1000],
    "pool_size_estimate": 10000,
    "N_existing": None,  # None = use real existing stations
}

# -----------------------------------------------------------------------------
# RESUME LOGIC
# -----------------------------------------------------------------------------
scaling_path_local = LOCAL_CACHE / "scaling_results.pkl"
scaling_path_drive = OUTPUT_DIR / "scaling_results.pkl"
scaling_results = None

if not FORCE_RECOMPUTE_SCALING:
    if scaling_path_local.exists():
        print("📂 Loading scaling results from local cache...")
        with open(scaling_path_local, "rb") as f:
            scaling_results = pickle.load(f)
    elif scaling_path_drive.exists():
        print("📂 Loading scaling results from Drive (copying to local)...")
        shutil.copy(scaling_path_drive, scaling_path_local)
        with open(scaling_path_local, "rb") as f:
            scaling_results = pickle.load(f)

if scaling_results is not None:
    print(f"✅ Loaded scaling results for tiers: {sorted(scaling_results.keys())}")
    # Unpack the first tier to get metadata (coords, U, etc.)
    first_N = next(iter(scaling_results))
    first_res = scaling_results[first_N]
    n_free = len(first_res["free_master_indices"])
    n_fixed = len(first_res["M_orig_indices"])
    print(f"   Free sites in smallest tier: {n_free}, Fixed: {n_fixed}")
    print("   Tiers summary:")
    for N, res in sorted(scaling_results.items()):
        print(f"      N={N}: free={len(res['free_master_indices'])}, fixed={len(res['M_orig_indices'])}, total={len(res['coords'])}")

    # Load master data if needed (should already be defined from Cell 2)
    # We need coords and U from master to plot the background.
    # They are already in namespace from Cell 2.

    # Recreate plots from loaded data
    print("\n📊 Plotting nested candidate grids from loaded data...")
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
    fig.colorbar(sm, cax=cbar_ax, label="Utility U_i")

    plt.suptitle("Nested Resolution Scaling (Farthest Point Sampling – Deduplicated)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # We are done – skip recomputation
    print("✅ Scaling results loaded and ready. Skipping recomputation.")
    # Ensure scaling_results is defined
    pass

else:
    # -------------------------------------------------------------------------
    # COMPUTE FROM SCRATCH (original code)
    # -------------------------------------------------------------------------
    print("🔄 Scaling results not found or FORCE_RECOMPUTE_SCALING=True. Computing from scratch...")

    # ---- LOAD MASTER DATA (already in memory from Cell 2) ----
    # coords, U, water_polygon, M_indices, bbox, n_sites, M_set_global are already defined.

    print("📦 Master data loaded.")
    print(f"   Total master sites: {n_sites}")
    print(f"   Existing stations: {len(M_indices)}")
    print(f"   BBox: {bbox}")

    # ---- STEP 1: GENERATE DENSE HEX POOL ----
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

    print("\n🔹 Generating dense hex pool...")
    pool_raw = generate_hex_centroids_bbox(bbox, CONFIG["pool_size_estimate"])
    print(f"   Raw pool size: {len(pool_raw):,}")

    shapely.prepare(water_polygon)
    pool_water = []
    for pt in tqdm(pool_raw, desc="Filtering water points"):
        if water_polygon.contains(Point(pt)):
            pool_water.append(pt)
    pool_coords = np.array(pool_water, dtype=float)
    print(f"   Pool points inside water: {len(pool_coords):,}")

    # ---- STEP 2: MAP POOL POINTS TO NEAREST MASTER CENTROID ----
    master_tree = cKDTree(coords)
    dist, pool_to_master = master_tree.query(pool_coords, k=1)
    pool_to_master = pool_to_master.flatten().astype(int)

    free_mask = ~np.isin(pool_to_master, list(M_set_global))
    pool_to_master_free = pool_to_master[free_mask]

    unique_master_indices, unique_inverse = np.unique(pool_to_master_free, return_inverse=True)
    unique_master_coords = coords[unique_master_indices]
    unique_U = U[unique_master_indices]

    print(f"   Pool points mapped to free master indices (raw): {len(pool_to_master_free):,}")
    print(f"   Unique free master indices after deduplication: {len(unique_master_indices):,}")

    # ---- STEP 3: FARTHEST POINT SAMPLING ----
    print("\n🔹 Running Farthest Point Sampling on deduplicated master sites (seeded with existing stations)...")
    seed_coords = coords[list(M_set_global)]
    n_select = 1000

    seed_tree = cKDTree(seed_coords)
    min_dist = seed_tree.query(unique_master_coords, k=1)[0].flatten()

    selected_indices = []
    selected_master_indices = []

    for _ in tqdm(range(n_select), desc="FPS iterations (unique sites)"):
        idx = np.argmax(min_dist)
        selected_indices.append(idx)
        selected_master_indices.append(unique_master_indices[idx])

        new_point = unique_master_coords[idx]
        dist_to_new = np.linalg.norm(unique_master_coords - new_point, axis=1)
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[idx] = -1.0

    print(f"   Selected {len(selected_master_indices)} unique free points via FPS.")

    # ---- STEP 4: BUILD NESTED TIERS ----
    print("\n🔹 Building nested tiers for exact target sizes...")
    scaling_results = {}

    for N in CONFIG["target_sizes"]:
        free_master_indices = selected_master_indices[:N]
        free_coords = coords[free_master_indices]
        free_U = U[free_master_indices]

        all_master_indices = list(free_master_indices) + M_indices
        all_coords = np.vstack([free_coords, coords[M_indices]])
        all_U = np.concatenate([free_U, U[M_indices]])

        M_tier = list(range(N, N + len(M_indices)))
        snapped_indices = np.array(all_master_indices, dtype=int)

        scaling_results[N] = {
            "coords": all_coords,
            "U": all_U,
            "M_indices": M_tier,
            "M_orig_indices": M_indices,
            "snapped_indices": snapped_indices,
            "free_master_indices": free_master_indices,
        }

        print(f"   N={N:4d}: free={N}, fixed={len(M_indices)}, total={len(all_coords)}")

    # ---- STEP 5: PLOT ----
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
    fig.colorbar(sm, cax=cbar_ax, label="Utility U_i")

    plt.suptitle("Nested Resolution Scaling (Farthest Point Sampling – Deduplicated)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # ---- STEP 6: SAVE ----
    with open(scaling_path_local, "wb") as f:
        pickle.dump(scaling_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(scaling_path_drive, "wb") as f:
        pickle.dump(scaling_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Cached deduplicated scaling results to {scaling_path_local} and {scaling_path_drive}")

print("\n✅ Nested resolution scaling complete.")