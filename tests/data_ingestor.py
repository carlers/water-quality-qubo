#@title CELL 2: REAL DATA INGESTOR + MASTER MIQP BUILDER (Full Sparse)
# =============================================================================
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import shapely
from shapely.geometry import MultiPoint, Polygon, LineString   # <-- added LineString

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
DRIVE_BASE = Path("/content/drive/MyDrive")
LOCAL_CACHE = Path("./cache")

GEOJSON_PATH = os.path.join(DRIVE_BASE, "water_quality_results", "LDB_centroids_clean.geojson")
OUTPUT_DIR = Path(os.path.join(DRIVE_BASE, "wqm_data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verify file exists
if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"❌ GeoJSON not found at: {GEOJSON_PATH}\n"
                            f"Please check the filename and folder path.")

print(f"✅ GeoJSON found: {GEOJSON_PATH}")

# 8 weights for the descriptive factors (sum to 1.0)
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

# Concave Hull tightness
HULL_RATIO = 0.05

# Fallback existing station simulation
N_EXISTING_FALLBACK = 3
MIN_EXISTING_DIST_FALLBACK = 12000  # meters

# -----------------------------------------------------------------------------
# MASTER QUBO CONFIG (mirrors Cell 4 exactly)
# -----------------------------------------------------------------------------
MASTER_QUBO_CONFIG = {
    "L_c": 7500.0,                # Spatial correlation length (meters)
    "L_w": 1000.0,                # Wake persistence length (meters)
    "Beta": 1.0,                  # Redundancy penalty weight
    "Delta": 1.0,                 # Wake penalty weight
    "Current_vector": (1.0, 0.0), # Flow vector (dx, dy)
    "D_max_buffer": 1.15,         # Multiplier for max existing NN distance
}

# -----------------------------------------------------------------------------
# LOAD & PARSE GEOJSON
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# COMPUTE UTILITY U_i
# -----------------------------------------------------------------------------
def compute_utility(factors, weights):
    valid_mask = ~np.isnan(factors)
    numerator = np.nansum(factors * weights, axis=1)
    denominator = np.nansum(weights * valid_mask, axis=1)
    denominator = np.where(denominator == 0, 1.0, denominator)
    U = numerator / denominator
    return np.clip(U, 0.0, 1.0)

U = compute_utility(factors, WEIGHTS)
print(f"   ✅ Utility U_i range: {U.min():.4f} to {U.max():.4f}")

# -----------------------------------------------------------------------------
# DETERMINE EXISTING STATIONS (M_indices)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# GENERATE CONCAVE HULL LAND/WATER MASK
# -----------------------------------------------------------------------------
print("🗺️ Generating Concave Hull water boundary mask...")
multi_pt = MultiPoint(coords)
laguna_water_polygon = shapely.concave_hull(multi_pt, ratio=HULL_RATIO, allow_holes=True)
shapely.prepare(laguna_water_polygon)
print("   ✅ Laguna de Bay land/water boundary mask created.")

# -----------------------------------------------------------------------------
# BUILD MASTER MIQP (Full N=5417) – Sparse Representation
# -----------------------------------------------------------------------------
print("\n🔧 Building Master MIQP (full grid) for global objective evaluation...")

# ----- Helper: compute global D_max from existing stations -----
master_M_coords = coords[list(M_set_global)]
if len(master_M_coords) > 1:
    M_tree = cKDTree(master_M_coords)
    dists, _ = M_tree.query(master_M_coords, k=2)
    max_nn_dist = np.max(dists[:, 1])
    global_D_max = max_nn_dist * MASTER_QUBO_CONFIG["D_max_buffer"]
else:
    print("⚠️ Less than 2 existing stations. Using default D_max.")
    global_D_max = 15000.0
print(f"   Global D_max = {global_D_max:.2f} m")

# ----- Pairwise term function (identical to Cell 4 logic) -----
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
    free_set = set(free_indices)
    M_set = set(M_indices)

    shapely.prepare(water_polygon)
    tree = cKDTree(coords)
    
    v_norm = np.linalg.norm(current_vector)
    if v_norm == 0:
        raise ValueError("current_vector cannot be zero.")
    v_unit = np.array(current_vector) / v_norm

    # Build connectivity graph (within D_max)
    valid_edges = set()
    pairs_dmax = tree.query_pairs(r=global_D_max)
    for i, j in pairs_dmax:
        p1, p2 = coords[i], coords[j]
        edge_line = LineString([p1, p2])
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

    # Compute raw quadratic interactions (within 2*L_c)
    raw_quad = {}
    pairs_within = tree.query_pairs(r=2.0 * L_c)
    for i, j in pairs_within:
        p1, p2 = coords[i], coords[j]
        edge_line = LineString([p1, p2])
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

    # Build linear terms (absorb fixed stations)
    linear = {}
    for i in free_indices:
        val = -U[i]
        for m in M_indices:
            key = (i, m) if i < m else (m, i)
            if key in raw_quad:
                val += raw_quad[key]
        linear[i] = val

    # Build free-free quadratic edges (only i<j, non-zero)
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
        "neighbors": neighbor_dict,
        "valid_edges": valid_edges,
        "N_total": N_total,
        "N_free": len(free_indices),
        "free_indices": free_indices,
        "M_indices": M_indices,
    }

# ----- Compute master pairwise terms -----
master_pairwise = compute_pairwise_terms_shape_aware(
    coords=coords,
    U=U,
    M_indices=M_indices,
    water_polygon=laguna_water_polygon,
    L_c=MASTER_QUBO_CONFIG["L_c"],
    L_w=MASTER_QUBO_CONFIG["L_w"],
    current_vector=MASTER_QUBO_CONFIG["Current_vector"],
    global_D_max=global_D_max,
    beta=MASTER_QUBO_CONFIG["Beta"],
    delta=MASTER_QUBO_CONFIG["Delta"],
)

# Build a_master as a full-length array (default 0)
a_master = np.zeros(n_sites, dtype=np.float64)
for idx, val in master_pairwise["linear"].items():
    a_master[idx] = val

# Q_master stored as sparse edges (i, j, val) with i < j
Q_master_edges = master_pairwise["quad_edges"]  # list of (i, j, val)

print(f"   ✅ Master a_master size: {len(a_master)}")
print(f"   ✅ Master Q_edges: {len(Q_master_edges)} non-zero pairs")
print(f"   ✅ Master free variables: {master_pairwise['N_free']} (fixed: {len(M_indices)})")

# -----------------------------------------------------------------------------
# PLOT: Utility Heatmap + Water Boundary Mask + Existing Stations
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

if isinstance(laguna_water_polygon, Polygon):
    x, y = laguna_water_polygon.exterior.xy
    ax.fill(x, y, alpha=0.15, fc="skyblue", ec="blue", linewidth=1.2, label="Water Polygon Mask", zorder=0)
    for interior in laguna_water_polygon.interiors:
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

# -----------------------------------------------------------------------------
# SAVE MASTER DATA (NOW WITH a_master AND Q_master_edges)
# -----------------------------------------------------------------------------
master_data = {
    "coords": coords,
    "factors": factors,
    "U": U,
    "M_indices": M_indices,
    "has_existing": has_existing,
    "water_polygon": laguna_water_polygon,
    "a": a_master,                    # <-- NEW: linear coefficients for full master
    "Q_edges": Q_master_edges,        # <-- NEW: sparse quadratic edges (i,j,val)
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
    }
}

master_path = OUTPUT_DIR / "master_real.pkl"
with open(master_path, "wb") as f:
    pickle.dump(master_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Master data cached to: {master_path}")
print("=" * 70)