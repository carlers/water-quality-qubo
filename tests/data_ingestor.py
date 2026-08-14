#@title REAL DATA INGESTOR + UTILITY HEATMAP
# =============================================================================
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------
DRIVE_BASE = "/content/drive/MyDrive"
from pathlib import Path

# Convert BOTH to Path objects
DRIVE_BASE = Path("/content/drive/MyDrive")  # <-- ADD Path() here
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
# Order: a1, a2, b1, b2, c1, c2, d1, d2
WEIGHTS = np.array([0.3671, 0.1224, 0.2508, 0.0502, 0.1278, 0.0183, 0.0476, 0.0159], dtype=np.float64)
#WEIGHTS = np.array([0.05, 0.04, 0.85, 0.02, 0.01, 0.01, 0.01, 0.01], dtype=np.float64)

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

# Fallback existing station simulation (if no real ones exist)
N_EXISTING_FALLBACK = 3
MIN_EXISTING_DIST_FALLBACK = 12000  # meters

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
# COMPUTE UTILITY U_i (weighted average of available factors)
# -----------------------------------------------------------------------------
def compute_utility(factors, weights):
    """Weighted average of non-null factors, normalized to [0,1]."""
    valid_mask = ~np.isnan(factors)
    numerator = np.nansum(factors * weights, axis=1)
    denominator = np.nansum(weights * valid_mask, axis=1)
    # If all factors are null, set to 0
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
    # Use your synthetic `select_existing_stations_spaced` logic
    # (I'll inline a simplified version here to avoid dependency)
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

# -----------------------------------------------------------------------------
# PLOT: Utility Heatmap + Existing Stations
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=U, cmap="viridis",
                s=30, alpha=0.8, edgecolor="k", linewidth=0.2, label="Candidate sites")

# Overlay existing stations
ax.scatter(coords[M_indices, 0], coords[M_indices, 1],
           c="red", marker="*", s=200, edgecolor="black", linewidth=0.8,
           label=f"Existing station (n={len(M_indices)})", zorder=5)

ax.set_xlabel("Easting (m) – EPSG:32651")
ax.set_ylabel("Northing (m) – EPSG:32651")
ax.set_title("Real Centroids: Utility Score (U_i) and Existing Stations", fontweight="bold")
cbar = plt.colorbar(sc, ax=ax, label="Utility U_i (weighted average)")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# SAVE MASTER DATA (CACHED)
# -----------------------------------------------------------------------------
master_data = {
    "coords": coords,
    "factors": factors,
    "U": U,
    "M_indices": M_indices,
    "has_existing": has_existing,
    "metadata": {
        "n_sites": n_sites,
        "weights": WEIGHTS.tolist(),
        "factor_names": FACTOR_NAMES,
        "bbox": [coords[:, 0].min(), coords[:, 1].min(),
                 coords[:, 0].max(), coords[:, 1].max()],
        "n_existing_real": int(np.sum(has_existing)),
        "n_existing_used": len(M_indices),
    }
}

master_path = OUTPUT_DIR / "master_real.pkl"
with open(master_path, "wb") as f:
    pickle.dump(master_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Master data cached to: {master_path}")
print("=" * 70)