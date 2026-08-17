#@title CELL 2: REAL DATA INGESTOR + MASTER MIQP (Resumable, Config‑Aware)
# =============================================================================
# REVISION: Resumable – loads master_real_{config_hash}.pkl if available.
# - Uses FORCE_RECOMPUTE_MASTER flag from Cell 1.
# - Dual storage: local and Drive.
# - Filename includes config_hash to invalidate on QUBO parameter changes.
# - Backward‑compatible: falls back to master_real.pkl if hashed version missing.
# =============================================================================
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import shapely
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from tqdm.notebook import tqdm
import warnings
import hashlib
import shutil

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# USER CONFIGURATION (unchanged)
# -----------------------------------------------------------------------------
DRIVE_BASE = Path("/content/drive/MyDrive")
LOCAL_CACHE = Path("/content/wqm_data")

GEOJSON_PATH = os.path.join(DRIVE_BASE, "water_quality_results", "LDB_centroids_clean.geojson")
OUTPUT_DIR = Path(os.path.join(DRIVE_BASE, "wqm_data"))
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"❌ GeoJSON not found at: {GEOJSON_PATH}")

print(f"✅ GeoJSON found: {GEOJSON_PATH}")

WEIGHTS = np.array([0.3671, 0.1224, 0.2508, 0.0502, 0.1278, 0.0183, 0.0476, 0.0159], dtype=np.float64)

FACTOR_NAMES = [
    "a1_river_proximity",
    "a2_runoff_proximity",
    "b1_fishpen_density",
    "b2_hypoxic_proximity",
    "c1_bathymetric_depressions",
    "c2_outlet_proximity",
    "d1_boatramp_proximity",
    "d2_road_proximity",
]

HULL_RATIO = 0.05
N_EXISTING_FALLBACK = 3
MIN_EXISTING_DIST_FALLBACK = 12000

MASTER_QUBO_CONFIG = {
    "L_c": 7500.0,
    "L_w": 1000.0,
    "Beta": 1.0,
    "Delta": 1.0,
    "Current_vector": (1.0, 0.0),
    "K": 5,
    "D_max_buffer": 1.15,
}

# ---- Compute config hash ----
config_str = json.dumps(MASTER_QUBO_CONFIG, sort_keys=True)
config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
print(f"🔑 Master config hash: {config_hash}")

# -----------------------------------------------------------------------------
# RESUME LOGIC (with fallback to unhashed)
# -----------------------------------------------------------------------------
master_path_local = LOCAL_CACHE / f"master_real_{config_hash}.pkl"
master_path_drive = OUTPUT_DIR / f"master_real_{config_hash}.pkl"
master_data = None

# 1. Try hashed version
if not FORCE_RECOMPUTE_MASTER:
    if master_path_local.exists():
        print("📂 Loading hashed master data from local cache...")
        with open(master_path_local, "rb") as f:
            master_data = pickle.load(f)
    elif master_path_drive.exists():
        print("📂 Loading hashed master data from Drive (copying to local)...")
        shutil.copy(master_path_drive, master_path_local)
        with open(master_path_local, "rb") as f:
            master_data = pickle.load(f)

# 2. If not found, try unhashed version (backward compatibility)
if master_data is None and not FORCE_RECOMPUTE_MASTER:
    unhashed_local = LOCAL_CACHE / "master_real.pkl"
    unhashed_drive = OUTPUT_DIR / "master_real.pkl"
    if unhashed_local.exists():
        print("📂 Loading unhashed master data from local cache (migrating to hashed)...")
        with open(unhashed_local, "rb") as f:
            master_data = pickle.load(f)
        # Save with hash for future
        with open(master_path_local, "wb") as f:
            pickle.dump(master_data, f)
        with open(master_path_drive, "wb") as f:
            pickle.dump(master_data, f)
        print(f"✅ Migrated master data to hashed version: {master_path_local}")
    elif unhashed_drive.exists():
        print("📂 Loading unhashed master data from Drive (migrating to hashed)...")
        shutil.copy(unhashed_drive, unhashed_local)
        with open(unhashed_local, "rb") as f:
            master_data = pickle.load(f)
        # Save with hash
        with open(master_path_local, "wb") as f:
            pickle.dump(master_data, f)
        with open(master_path_drive, "wb") as f:
            pickle.dump(master_data, f)
        print(f"✅ Migrated master data to hashed version: {master_path_local}")

if master_data is not None:
    # Unpack loaded data
    coords = master_data["coords"]
    factors = master_data["factors"]
    U = master_data["U"]
    M_indices = master_data["M_indices"]
    has_existing = master_data["has_existing"]
    water_polygon = master_data["water_polygon"]
    a_master = master_data["a"]
    Q_master_edges = master_data["Q_edges"]
    metadata = master_data["metadata"]

    n_sites = len(coords)
    print(f"✅ Loaded master data: N_sites={n_sites}, Q_edges={len(Q_master_edges)}")
    print(f"   Existing stations: {len(M_indices)}")
    print(f"   U range: {U.min():.4f} to {U.max():.4f}")

    # Recreate plot from loaded data
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(water_polygon, Polygon):
        x, y = water_polygon.exterior.xy
        ax.fill(x, y, alpha=0.15, fc="skyblue", ec="blue", linewidth=1.2, label="Water Polygon Mask", zorder=0)
        for interior in water_polygon.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=0.4, fc="bisque", ec="saddlebrown", linewidth=1.2, label="Landmass Hole", zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=U, cmap="viridis",
                    s=30, alpha=0.8, edgecolor="k", linewidth=0.2, label="Candidate sites", zorder=2)
    ax.scatter(coords[M_indices, 0], coords[M_indices, 1],
               c="red", marker="*", s=200, edgecolor="black", linewidth=0.8,
               label=f"Existing station (n={len(M_indices)})", zorder=5)
    ax.set_xlabel("Easting (m) – EPSG:32651")
    ax.set_ylabel("Northing (m) – EPSG:32651")
    ax.set_title("Real Centroids: Utility Score (U_i), Water Mask, and Existing Stations", fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax, label="Utility U_i (weighted average)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # We are done – skip the rest
    print("✅ Master data loaded and ready. Skipping recomputation.")
    # Ensure variables are defined for downstream
    pass
else:
    # -------------------------------------------------------------------------
    # COMPUTE FROM SCRATCH (original code)
    # -------------------------------------------------------------------------
    print("🔄 Master data not found or FORCE_RECOMPUTE_MASTER=True. Computing from scratch...")

    # ---- LOAD & PARSE GEOJSON ----
    print("📂 Loading real centroids from GeoJSON...")
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)

    features = data["features"]
    n_sites = len(features)

    coords = np.zeros((n_sites, 2), dtype=np.float64)
    factors = np.full((n_sites, 8), np.nan, dtype=np.float64)
    has_existing = np.zeros(n_sites, dtype=bool)
    ids = np.zeros(n_sites, dtype=int)
    row_idx = np.zeros(n_sites, dtype=int)
    col_idx = np.zeros(n_sites, dtype=int)

    for i, feat in enumerate(features):
        geom = feat["geometry"]["coordinates"]
        coords[i] = [geom[0], geom[1]]
        props = feat["properties"]
        ids[i] = props["id"]
        row_idx[i] = props["row_index"]
        col_idx[i] = props["col_index"]
        has_existing[i] = props.get("has_existing_station", False)
        for j, name in enumerate(FACTOR_NAMES):
            val = props.get(name)
            if val is not None:
                factors[i, j] = val

    print(f"   ✅ Loaded {n_sites} centroids.")
    print(f"   ✅ Found {np.sum(has_existing)} sites with 'has_existing_station': true")

    # ---- COMPUTE UTILITY ----
    def compute_utility(factors, weights):
        valid_mask = ~np.isnan(factors)
        numerator = np.nansum(factors * weights, axis=1)
        denominator = np.nansum(weights * valid_mask, axis=1)
        denominator = np.where(denominator == 0, 1.0, denominator)
        U = numerator / denominator
        return np.clip(U, 0.0, 1.0)

    U = compute_utility(factors, WEIGHTS)
    print(f"   ✅ Utility U_i range: {U.min():.4f} to {U.max():.4f}")

    # ---- DETERMINE EXISTING STATIONS ----
    if np.any(has_existing):
        M_indices = np.where(has_existing)[0].tolist()
        print(f"   📍 Using {len(M_indices)} real existing stations from GeoJSON.")
    else:
        print("   ⚠️  No 'has_existing_station: true' found. Falling back to simulation...")
        from scipy.spatial.distance import cdist
        n_existing = N_EXISTING_FALLBACK
        min_dist = MIN_EXISTING_DIST_FALLBACK
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
        print(f"   📍 Simulated {len(M_indices)} existing stations: {M_indices}")

    M_set_global = set(M_indices)

    # ---- GENERATE CONCAVE HULL ----
    print("🗺️ Generating Concave Hull water boundary mask...")
    multi_pt = MultiPoint(coords)
    laguna_water_polygon = shapely.concave_hull(multi_pt, ratio=HULL_RATIO, allow_holes=True)
    shapely.prepare(laguna_water_polygon)
    print("   ✅ Laguna de Bay land/water boundary mask created.")
    water_polygon = laguna_water_polygon

    # ---- BUILD MASTER MIQP (SEQUENTIAL) ----
    print("\n🔧 Building Master MIQP (full grid) with sequential optimized loop...")

    master_M_coords = coords[list(M_set_global)]
    if len(master_M_coords) > 1:
        M_tree = cKDTree(master_M_coords)
        dists, _ = M_tree.query(master_M_coords, k=2)
        max_nn_dist = np.max(dists[:, 1])
        global_D_max = max_nn_dist * MASTER_QUBO_CONFIG["D_max_buffer"]
    else:
        global_D_max = 15000.0
    print(f"   Global D_max = {global_D_max:.2f} m")

    v_norm = np.linalg.norm(MASTER_QUBO_CONFIG["Current_vector"])
    v_unit = np.array(MASTER_QUBO_CONFIG["Current_vector"]) / v_norm

    tree = cKDTree(coords)
    print("   Querying spatial pairs...")
    pairs = np.array(list(tree.query_pairs(r=global_D_max)))
    print(f"   Total candidate pairs: {len(pairs):,}")

    L_c = MASTER_QUBO_CONFIG["L_c"]
    L_w = MASTER_QUBO_CONFIG["L_w"]
    beta = MASTER_QUBO_CONFIG["Beta"]
    delta = MASTER_QUBO_CONFIG["Delta"]
    L_c_2 = 2.0 * L_c
    poly = water_polygon

    all_valid_edges = []
    all_quad_edges = []

    for i, j in tqdm(pairs, desc="🔹 Processing geometry pairs", unit="pairs"):
        p1 = coords[i]
        p2 = coords[j]

        if not (poly.contains(Point(p1)) and poly.contains(Point(p2))):
            continue

        line = LineString([p1, p2])
        if not poly.contains(line):
            continue

        all_valid_edges.append((i, j))

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        d = np.sqrt(dx*dx + dy*dy)
        if d <= L_c_2 and d > 0:
            R_ij = max(0.0, 1.0 - d / L_c)

            cos_theta = (dx * v_unit[0] + dy * v_unit[1]) / d
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            W_ij = (np.exp(-d / L_w) * cos_theta) if cos_theta > 0.7071 else 0.0

            cos_theta_ji = (-dx * v_unit[0] + -dy * v_unit[1]) / d
            cos_theta_ji = np.clip(cos_theta_ji, -1.0, 1.0)
            W_ji = (np.exp(-d / L_w) * cos_theta_ji) if cos_theta_ji > 0.7071 else 0.0

            coeff = beta * R_ij + delta * (W_ij + W_ji)
            if abs(coeff) > 1e-12:
                all_quad_edges.append((i, j, coeff))

    print(f"   ✅ Connectivity edges: {len(all_valid_edges):,}")
    print(f"   ✅ Non-zero quadratic edges: {len(all_quad_edges):,}")

    # ---- Absorb fixed stations ----
    free_indices = [i for i in range(n_sites) if i not in M_set_global]
    M_indices_list = list(M_set_global)

    a_master = np.zeros(n_sites, dtype=np.float64)

    for i in free_indices:
        a_master[i] = -U[i]

    quad_dict = {(i, j): val for i, j, val in all_quad_edges}
    for i in tqdm(free_indices, desc="Absorbing fixed station interactions"):
        for m in M_indices_list:
            key = (i, m) if i < m else (m, i)
            if key in quad_dict:
                a_master[i] += quad_dict[key]

    free_set = set(free_indices)
    Q_master_edges = []
    for i, j, val in all_quad_edges:
        if i in free_set and j in free_set:
            Q_master_edges.append((i, j, val))

    print(f"   ✅ Final Q_master_edges (free-free): {len(Q_master_edges):,}")

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(water_polygon, Polygon):
        x, y = water_polygon.exterior.xy
        ax.fill(x, y, alpha=0.15, fc="skyblue", ec="blue", linewidth=1.2, label="Water Polygon Mask", zorder=0)
        for interior in water_polygon.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=0.4, fc="bisque", ec="saddlebrown", linewidth=1.2, label="Landmass Hole", zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=U, cmap="viridis",
                    s=30, alpha=0.8, edgecolor="k", linewidth=0.2, label="Candidate sites", zorder=2)
    ax.scatter(coords[M_indices, 0], coords[M_indices, 1],
               c="red", marker="*", s=200, edgecolor="black", linewidth=0.8,
               label=f"Existing station (n={len(M_indices)})", zorder=5)
    ax.set_xlabel("Easting (m) – EPSG:32651")
    ax.set_ylabel("Northing (m) – EPSG:32651")
    ax.set_title("Real Centroids: Utility Score (U_i), Water Mask, and Existing Stations", fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax, label="Utility U_i (weighted average)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # ---- SAVE MASTER DATA (hashed) ----
    master_data = {
        "coords": coords,
        "factors": factors,
        "U": U,
        "M_indices": M_indices,
        "has_existing": has_existing,
        "water_polygon": water_polygon,
        "a": a_master,
        "Q_edges": Q_master_edges,
        "metadata": {
            "n_sites": n_sites,
            "weights": WEIGHTS.tolist(),
            "factor_names": FACTOR_NAMES,
            "bbox": [coords[:, 0].min(), coords[:, 1].min(),
                     coords[:, 0].max(), coords[:, 1].max()],
            "n_existing_real": int(np.sum(has_existing)),
            "n_existing_used": len(M_indices),
            "hull_ratio": HULL_RATIO,
            "master_D_max": global_D_max,
            "master_L_c": MASTER_QUBO_CONFIG["L_c"],
            "master_L_w": MASTER_QUBO_CONFIG["L_w"],
            "master_edges_count": len(Q_master_edges),
            "master_valid_edges_count": len(all_valid_edges),
        }
    }

    with open(master_path_local, "wb") as f:
        pickle.dump(master_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(master_path_drive, "wb") as f:
        pickle.dump(master_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Master data cached to: {master_path_local} and {master_path_drive}")
    print("=" * 70)

print("✅ Cell 2 complete. Master data ready.")