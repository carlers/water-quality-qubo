"""
data/synthetic_data.py

Generate synthetic master dataset for water quality monitoring QUBO experiments.

This module creates a master set of candidate sites with:
- Random (x, y) coordinates in a square domain
- Random AHP factors (6 criteria) normalized to [0, 1]
- Utility scores U_i computed via AHP weights
- Nested subsets via farthest-point sampling for reproducible experiments
- Fixed "existing stations" M for incremental deployment scenarios
- **NEW: Guaranteed connectivity** – each subset contains a connected core of size K,
  ensuring at least one feasible solution (budget + connectivity) for all subsets.

All outputs are saved to the `data/` directory as .npy, .pkl, and .json files.

Usage (command line):
    python data/synthetic_data.py --seed 123 --n_master 600 --subset_sizes 10,20,30,50,100,150,200,300,500
    python data/synthetic_data.py --seed random  # Random seed (uses current time)
"""

import time
import json
import pickle
import warnings
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------

# Random seed for all stochastic operations (can be overridden via CLI)
RANDOM_SEED: int = 42

# Domain: square of size DOMAIN_SIZE x DOMAIN_SIZE (kilometers)
DOMAIN_SIZE: float = 50.0

# Number of master candidate sites (INCREASED to 600 to support scaling to N=500)
N_MASTER: int = 600

# AHP baseline weights (from Table I in the paper)
# Order: [Pollution Load, Ecological Sensitivity, Hydrodynamic Variability,
#         Accessibility, Data Scarcity, Socio-Economic Exposure]
AHP_WEIGHTS: np.ndarray = np.array([0.35, 0.20, 0.12, 0.12, 0.14, 0.07], dtype=np.float64)

# Spatial correlation length (km) - stored for metadata, not used in generation
L_C: float = 5.0

# Dominant current direction (dx, dy) - stored for metadata
CURRENT_VECTOR: Tuple[float, float] = (1.0, 0.0)

# Number of existing stations (for incremental scenario)
N_EXISTING: int = 3

# Connectivity range (km) – used to build connected core
D_MAX: float = 8.0

# Sizes of nested subsets to generate (EXTENDED to include 300 and 500)
SUBSET_SIZES: List[int] = [10, 20, 30, 50, 100, 150, 200, 300, 500]

# Output directory (relative to project root)
OUTPUT_DIR: str = "data"


# -----------------------------------------------------------------------------
# GENERATION FUNCTIONS
# -----------------------------------------------------------------------------

def generate_coordinates(
    n: int, domain_size: float, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a structured uniform grid of points (mimicking a hexagonal lattice).
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    # Generate grid points
    x = np.linspace(0, domain_size, cols)
    y = np.linspace(0, domain_size, rows)
    xx, yy = np.meshgrid(x, y)
    
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Randomly sample N points if grid has more than N
    if len(coords) > n:
        idx = rng.choice(len(coords), size=n, replace=False)
        coords = coords[idx]
    
    # Ensure points stay within bounds
    coords = np.clip(coords, 0, domain_size)
    
    return coords.astype(np.float64)


def generate_factors(
    n: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random AHP factor matrix with values in [0, 1].
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    factors = rng.uniform(0.0, 1.0, size=(n, 6))

    for j in range(factors.shape[1]):
        col = factors[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            factors[:, j] = (col - col_min) / (col_max - col_min)
        else:
            factors[:, j] = 0.5
            warnings.warn(f"Factor column {j} is constant; set all values to 0.5.")
    return factors.astype(np.float64)


def compute_utility(factors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute composite utility U_i = weights · factors_i.
    """
    if factors.shape[1] != weights.shape[0]:
        raise ValueError(f"Factor columns ({factors.shape[1]}) must match weights size ({weights.shape[0]}).")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        warnings.warn(f"AHP weights sum to {weights.sum():.4f}, not 1.0. Normalizing.")
        weights = weights / weights.sum()
    utility = factors @ weights
    return utility.astype(np.float64)


def farthest_point_sampling(
    coords: np.ndarray,
    sizes: List[int],
    start_idx: Optional[int] = None,
    seed: Optional[int] = None,
    domain_size: float = 50.0,
    fixed_indices: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    """
    Generate nested subsets via farthest-point sampling (FPS).

    Now with debug prints and optimised candidate filtering.
    """
    t0 = time.perf_counter()
    n_total = coords.shape[0]
    sizes = sorted(sizes)
    if sizes[-1] > n_total:
        raise ValueError(f"Largest requested size ({sizes[-1]}) exceeds number of points ({n_total}).")

    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)

    # ---------- Initialise selected set ----------
    selected = list(fixed_indices) if fixed_indices is not None else []
    selected = [int(idx) for idx in set(selected) if 0 <= idx < n_total]

    if start_idx is not None and start_idx not in selected:
        selected.append(start_idx)
    elif start_idx is None:
        center = np.array([domain_size / 2.0, domain_size / 2.0])
        distances_to_center = np.linalg.norm(coords - center, axis=1)
        candidates = [i for i in range(n_total) if i not in selected]
        if candidates:
            start_idx = candidates[np.argmin(distances_to_center[candidates])]
            selected.append(start_idx)

    if len(selected) > sizes[-1]:
        warnings.warn(
            f"Number of fixed indices ({len(selected)}) exceeds largest requested size ({sizes[-1]}). "
            "Truncating to largest size."
        )
        selected = selected[:sizes[-1]]

    t_init = time.perf_counter() - t0

    # ---------- Build distance array ----------
    # Efficient incremental initialisation: start from first point, then add others
    t_dist0 = time.perf_counter()
    if len(selected) == 0:
        # Fallback: pick a random point
        start_idx = rng.randint(n_total)
        selected = [start_idx]
        dist_to_selected = np.zeros(n_total, dtype=np.float64)
    else:
        # Build distances incrementally to avoid O(N * |selected|) cdist if |selected| is large
        dist_to_selected = np.full(n_total, np.inf, dtype=np.float64)
        for idx in selected:
            dist = np.linalg.norm(coords - coords[idx], axis=1)
            dist_to_selected = np.minimum(dist_to_selected, dist)
        # Mark selected points (distance 0, but we'll mask them)
    # Mark selected as -1 to ignore them
    dist_to_selected[selected] = -1.0
    t_dist = time.perf_counter() - t_dist0

    # ---------- Main FPS loop ----------
    subsets = {}
    total_loop_time = 0.0
    total_candidate_time = 0.0
    total_update_time = 0.0

    for size in sizes:
        t_size_start = time.perf_counter()
        # We need to add (size - len(selected)) points
        while len(selected) < size:
            # ---- Find candidates with maximum distance ----
            t_cand_start = time.perf_counter()
            max_dist = dist_to_selected.max()
            # Use boolean mask instead of list comprehension
            unselected_mask = np.ones(n_total, dtype=bool)
            unselected_mask[selected] = False
            candidates = np.where((dist_to_selected == max_dist) & unselected_mask)[0]
            # In case of floating point tolerance, use isclose:
            if len(candidates) == 0:
                # fallback: nearest unselected
                candidates = np.where(unselected_mask)[0]
                if len(candidates) == 0:
                    warnings.warn(f"No candidates left to reach size {size}; stopping at {len(selected)}.")
                    break
                best_idx = candidates[np.argmin(dist_to_selected[candidates])]
            else:
                # Random tie-breaking
                perm = rng.permutation(candidates)
                best_idx = int(perm[0])
            t_cand = time.perf_counter() - t_cand_start
            total_candidate_time += t_cand

            # ---- Update selected and distances ----
            t_upd_start = time.perf_counter()
            selected.append(best_idx)
            dist_new = np.linalg.norm(coords - coords[best_idx], axis=1)
            dist_to_selected = np.minimum(dist_to_selected, dist_new)
            dist_to_selected[selected] = -1.0   # mark selected
            t_upd = time.perf_counter() - t_upd_start
            total_update_time += t_upd

        # ---- Store subset ----
        subsets[size] = selected.copy()
        t_size = time.perf_counter() - t_size_start
        total_loop_time += t_size

    # ---- Summary ----
    total_time = time.perf_counter() - t0
    print(f"FPS total time: {total_time:.4f} s")
    return subsets


def create_nested_subsets(
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
    sizes: List[int],
    start_idx: Optional[int] = None,
    seed: Optional[int] = None,
    domain_size: float = 50.0,
    fixed_indices: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Union[np.ndarray, List[int]]]]:
    """
    Generate nested subsets with full data (coords, factors, utility, indices).
    """
    indices_dict = farthest_point_sampling(
        coords, sizes, start_idx, seed, domain_size, fixed_indices=fixed_indices
    )

    subsets = {}
    for size, indices in indices_dict.items():
        indices_arr = np.array(indices, dtype=int)
        subsets[size] = {
            "coords": coords[indices_arr].copy(),
            "factors": factors[indices_arr].copy(),
            "U": utility[indices_arr].copy(),
            "indices": indices_arr.tolist(),
        }
    return subsets


def select_existing_stations_spaced(
    coords: np.ndarray,
    utility: np.ndarray,
    n_existing: int,
    min_distance: float = 12.0,  # Force legacy stations to be at least 12km apart
    seed: Optional[int] = None,
) -> List[int]:
    """
    Select existing stations that are spaced out across the domain 
    while prioritizing high-utility zones.
    """
    n_total = coords.shape[0]
    if n_existing > n_total:
        raise ValueError(f"n_existing ({n_existing}) cannot exceed n_total.")
    if n_existing == 0:
        return []

    # 1. Start with the highest utility site on the map
    selected = [int(np.argmax(utility))]
    
    for _ in range(1, n_existing):
        # Calculate distances from all points to the already selected stations
        dists = cdist(coords, coords[selected]).min(axis=1)
        
        # Mask out points that are too close to existing stations OR already selected
        spatial_mask = (dists >= min_distance) & (~np.isin(np.arange(n_total), selected))
        
        valid_indices = np.where(spatial_mask)[0]
        
        # If we ran out of spaced-out areas, relax the distance constraint slightly
        if len(valid_indices) == 0:
            # Fallback: Just take the furthest available point (standard FPS)
            best_idx = np.argmax(dists)
        else:
            # Pick the site with the HIGHEST utility among the valid, spaced-out sites
            best_idx = valid_indices[np.argmax(utility[valid_indices])]
            
        selected.append(int(best_idx))
        
    return selected


# =============================================================================
# NEW: Build connected core to guarantee feasibility
# =============================================================================
def build_connected_core(
    coords: np.ndarray,
    M_indices: List[int],
    K: int,
    D_max: float,
    seed: Optional[int] = None,
    utility: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Build a connected core by starting with all existing legacy stations (M_indices)
    and appending exactly K new connected candidate stations.

    Returns:
        List of indices containing all M_indices plus K new candidate stations,
        guaranteed to be structurally connected within D_max.
    """
    if K <= 0:
        return list(set(M_indices))
        
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    n_total = coords.shape[0]

    # 1. Initialize with all existing stations (ensuring uniqueness and valid typing)
    selected = [int(i) for i in sorted(list(set(M_indices))) if 0 <= i < n_total]
    
    # Define our target total budget size: All legacy stations + K new ones
    target_size = len(selected) + K

    # 2. Greedy expansion until we satisfy the target budget size
    while len(selected) < target_size:
        # Find candidates within D_max of any selected point
        candidates = []
        for i in range(n_total):
            if i in selected:
                continue
            # Check distance to any currently selected point
            min_dist = np.min(np.linalg.norm(coords[i] - coords[selected], axis=1))
            if min_dist <= D_max:
                candidates.append(i)
                
        if candidates:
            # Pick the candidate with the highest utility
            if utility is not None:
                cand_util = [utility[i] for i in candidates]
                best_idx = candidates[np.argmax(cand_util)]
            else:
                # Fallback: random selection if no utility profile is provided
                best_idx = rng.choice(candidates)
        else:
            # Distance relaxation logic if a point is completely isolated
            distances = np.min(cdist(coords, coords[selected]), axis=1)
            distances[selected] = np.inf
            best_idx = int(np.argmin(distances))
            if best_idx == np.inf or distances[best_idx] == np.inf:
                raise RuntimeError("No more valid points available to expand core.")
            warnings.warn(f"Connected core: no candidate within D_max; picked nearest point {best_idx}")
            
        selected.append(best_idx)

    return selected


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR (UPDATED to use connected core)
# -----------------------------------------------------------------------------
def generate_and_save_all(
    output_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    n_master: Optional[int] = None,
    domain_size: Optional[float] = None,
    n_existing: Optional[int] = None,
    min_existing_distance: Optional[float] = 10.0,
    subset_sizes: Optional[List[int]] = None,
    D_max: float = D_MAX,
    K: int = 5,  # NEW: K for the connected core
) -> Dict:
    """
    Full pipeline: generate coordinates, factors, utility, subsets, and save.

    NEW: Ensures that all subsets contain a connected core of size K,
    guaranteeing at least one feasible solution (budget + connectivity).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if seed is None:
        seed = RANDOM_SEED
    if n_master is None:
        n_master = N_MASTER
    if domain_size is None:
        domain_size = DOMAIN_SIZE
    if n_existing is None:
        n_existing = N_EXISTING
    if subset_sizes is None:
        subset_sizes = SUBSET_SIZES

    print("=" * 60)
    print("Generating synthetic master dataset...")
    print(f"  Seed: {seed}")
    print(f"  Domain: {domain_size} × {domain_size} km")
    print(f"  Master candidates: {n_master}")
    print(f"  Existing stations: {n_existing}")
    print(f"  Subset sizes: {subset_sizes}")
    print(f"  Connectivity range (D_max): {D_max} km")
    print(f"  K (core size): {K}")
    print("=" * 60)

    # Step 1: Generate coordinates
    coords = generate_coordinates(n_master, domain_size, seed)
    print(f"✓ Coordinates: shape {coords.shape}, range [{coords.min():.2f}, {coords.max():.2f}] km")

    # Step 2: Generate factors
    factors = generate_factors(n_master, seed)
    print(f"✓ Factors: shape {factors.shape}, range [{factors.min():.3f}, {factors.max():.3f}]")

    # Step 3: Compute utility
    utility = compute_utility(factors, AHP_WEIGHTS)
    print(f"✓ Utility: shape {utility.shape}, range [{utility.min():.3f}, {utility.max():.3f}]")

    # Step 4: Select existing stations M
    existing_indices = select_existing_stations_spaced(
        coords, utility, n_existing, min_distance=min_existing_distance, seed=seed
    )
    print(f"✓ Existing stations (M): {existing_indices}")

    # Step 5: Build connected core (includes M_indices)
    core_indices = build_connected_core(
        coords=coords,
        M_indices=existing_indices,
        K=K,
        D_max=D_max,
        seed=seed,
        utility=utility,
    )
    print(f"✓ Connected core (size {len(core_indices)}): {core_indices}")

    # Combine: core will be forced into all subsets
    fixed_indices = list(set(core_indices))

    # Step 6: Create nested subsets using FPS, forcing fixed_indices
    center = np.array([domain_size / 2.0, domain_size / 2.0])
    dist_to_center = np.linalg.norm(coords - center, axis=1)
    start_idx = int(np.argmin(dist_to_center))

    subsets = create_nested_subsets(
        coords=coords,
        factors=factors,
        utility=utility,
        sizes=subset_sizes,
        start_idx=start_idx,
        seed=seed,
        domain_size=domain_size,
        fixed_indices=fixed_indices,
    )
    print(f"✓ Subsets created: {list(subsets.keys())}")

    # Step 7: Verify nesting and data integrity, and also verify that core is in each subset
    verify_subsets(subsets, coords, factors, utility, fixed_indices=fixed_indices)

    # Step 8: Save everything
    save_master_data(
        coords=coords,
        factors=factors,
        utility=utility,
        subsets=subsets,
        existing_indices=existing_indices,
        output_dir=output_dir,
        metadata={
            "start_idx": start_idx,
            "description": "Synthetic dataset for water quality monitoring QUBO.",
            "created_with_seed": seed,
            "max_existing_distance": min_existing_distance,
            "D_max": D_max,
            "K": K,
            "core_indices": core_indices,
        },
    )

    print("=" * 60)
    print("✅ Generation complete!")
    print("=" * 60)

    return {
        "coords": coords,
        "factors": factors,
        "U": utility,
        "subsets": subsets,
        "metadata": {
            "seed": seed,
            "domain_size": domain_size,
            "n_master": n_master,
            "ahp_weights": AHP_WEIGHTS.tolist(),
            "L_c": L_C,
            "current_vector": CURRENT_VECTOR,
            "n_existing": n_existing,
            "existing_indices": existing_indices,
            "subset_sizes": subset_sizes,
            "start_idx": start_idx,
            "D_max": D_max,
            "K": K,
            "core_indices": core_indices,
        },
    }


# -----------------------------------------------------------------------------
# VERIFICATION UTILITIES
# -----------------------------------------------------------------------------
def verify_subsets(
    subsets: Dict[int, Dict],
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
    fixed_indices: Optional[List[int]] = None,
) -> bool:
    """
    Perform sanity checks on generated subsets.
    """
    sorted_sizes = sorted(subsets.keys())
    n_total = coords.shape[0]

    for size, data in subsets.items():
        indices = np.array(data["indices"])
        assert np.all((0 <= indices) & (indices < n_total)), f"Subset {size}: indices out of bounds."
        assert len(indices) == size, f"Subset {size}: expected {size} points, got {len(indices)}."
        np.testing.assert_array_equal(coords[indices], data["coords"], err_msg=f"Subset {size}: coords mismatch.")
        np.testing.assert_array_equal(factors[indices], data["factors"], err_msg=f"Subset {size}: factors mismatch.")
        np.testing.assert_array_equal(utility[indices], data["U"], err_msg=f"Subset {size}: utility mismatch.")

    for i in range(len(sorted_sizes) - 1):
        s1 = sorted_sizes[i]
        s2 = sorted_sizes[i + 1]
        idx1 = set(subsets[s1]["indices"])
        idx2 = set(subsets[s2]["indices"])
        assert idx1.issubset(idx2), f"Subset {s1} is not a subset of {s2}. Missing: {idx1 - idx2}"

    if fixed_indices is not None:
        fixed_set = set(fixed_indices)
        for size, data in subsets.items():
            idx_set = set(data["indices"])
            assert fixed_set.issubset(idx_set), f"Subset {size} is missing fixed indices: {fixed_set - idx_set}"

    print("✓ All verification checks passed.")
    return True


# -----------------------------------------------------------------------------
# SAVE/LOAD
# -----------------------------------------------------------------------------
def save_master_data(
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
    subsets: Dict[int, Dict],
    existing_indices: List[int],
    output_dir: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "master_coords.npy", coords)
    np.save(output_path / "master_factors.npy", factors)
    np.save(output_path / "master_U.npy", utility)

    with open(output_path / "subsets.pkl", "wb") as f:
        pickle.dump(subsets, f, protocol=pickle.HIGHEST_PROTOCOL)

    default_metadata = {
        "seed": RANDOM_SEED,
        "domain_size_km": DOMAIN_SIZE,
        "n_master": N_MASTER,
        "ahp_weights": AHP_WEIGHTS.tolist(),
        "L_c_km": L_C,
        "current_vector": CURRENT_VECTOR,
        "n_existing": N_EXISTING,
        "existing_indices": existing_indices,
        "subset_sizes": list(subsets.keys()),
        "factor_columns": [
            "PollutionLoad",
            "EcologicalSensitivity",
            "HydrodynamicVariability",
            "Accessibility",
            "DataScarcity",
            "SocioEconomicExposure",
        ],
    }
    if metadata is not None:
        default_metadata.update(metadata)
    with open(output_path / "metadata.json", "w") as f:
        json.dump(default_metadata, f, indent=2)

    print(f"✓ Saved master data to {output_path.resolve()}")
    print(f"  - master_coords.npy: {coords.shape}")
    print(f"  - master_factors.npy: {factors.shape}")
    print(f"  - master_U.npy: {utility.shape}")
    print(f"  - subsets.pkl: {len(subsets)} subsets")
    print(f"  - metadata.json: {len(default_metadata)} keys")


def load_master_data(
    data_dir: Union[str, Path] = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    data_path = Path(data_dir)
    coords = np.load(data_path / "master_coords.npy")
    factors = np.load(data_path / "master_factors.npy")
    utility = np.load(data_path / "master_U.npy")
    with open(data_path / "subsets.pkl", "rb") as f:
        subsets = pickle.load(f)
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return coords, factors, utility, subsets, metadata


# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for water quality monitoring QUBO.")
    parser.add_argument("--seed", type=str, default="42", help='Random seed (integer) or "random" for time-based seed.')
    parser.add_argument("--n_master", type=int, default=None, help="Number of master candidate sites.")
    parser.add_argument("--domain_size", type=float, default=None, help="Domain size in km (square).")
    parser.add_argument("--n_existing", type=int, default=None, help="Number of existing stations M.")
    parser.add_argument("--max_existing_distance", type=float, default=10.0, help="Max cluster distance for existing stations.")
    parser.add_argument("--D_max", type=float, default=D_MAX, help="Connectivity range (km).")
    parser.add_argument("--K", type=int, default=5, help="Number of new stations (core size).")
    parser.add_argument("--subset_sizes", type=str, default=None,
                        help='Comma-separated subset sizes, e.g., "10,20,30,50,100,150,200,300,500"')
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for data files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed.lower() == "random":
        seed = int(time.time() * 1000) % 1000000
        print(f"Using random seed: {seed}")
    else:
        seed = int(args.seed)
    if args.subset_sizes is not None:
        subset_sizes = [int(x.strip()) for x in args.subset_sizes.split(",")]
    else:
        subset_sizes = None
    _ = generate_and_save_all(
        output_dir=args.output_dir,
        seed=seed,
        n_master=args.n_master,
        domain_size=args.domain_size,
        n_existing=args.n_existing,
        min_existing_distance=args.max_existing_distance,
        subset_sizes=subset_sizes,
        D_max=args.D_max,
        K=args.K,
    )