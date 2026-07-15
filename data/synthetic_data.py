"""
data/synthetic_data.py

Generate synthetic master dataset for water quality monitoring QUBO experiments.

This module creates a master set of candidate sites with:
- Random (x, y) coordinates in a square domain
- Random AHP factors (6 criteria) normalized to [0, 1]
- Utility scores U_i computed via AHP weights
- Nested subsets via farthest-point sampling for reproducible experiments
- Fixed "existing stations" M for incremental deployment scenarios

All outputs are saved to the `data/` directory as .npy, .pkl, and .json files.

Usage (command line):
    python data/synthetic_data.py --seed 123 --n_master 200 --subset_sizes 10,20,30,50,100,150,200
    python data/synthetic_data.py --seed random  # Random seed (uses current time)
"""

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

# Number of master candidate sites (INCREASED to 200 to support scaling)
N_MASTER: int = 200

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

# Sizes of nested subsets to generate (EXTENDED to include 150 and 200)
SUBSET_SIZES: List[int] = [10, 20, 30, 50, 100, 150, 200]

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
    
    This creates an approx sqrt(N) x sqrt(N) grid with a small random jitter
    to simulate real-world data, but maintaining a clear uniform structure.
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
    
    # # Add a tiny amount of jitter (e.g., up to 5% of spacing) to avoid exact alignment
    # # This mimics real GPS / QGIS noise without breaking the structure.
    # spacing = domain_size / max(cols, rows)
    # jitter = rng.uniform(-0.3 * spacing, 0.3 * spacing, size=coords.shape)
    # coords += jitter
    
    # Ensure points stay within bounds
    coords = np.clip(coords, 0, domain_size)
    
    return coords.astype(np.float64)


def generate_factors(
    n: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random AHP factor matrix with values in [0, 1].

    Each column corresponds to one of the six criteria and is independently
    sampled from a uniform distribution, then min-max normalized to [0, 1].

    Args:
        n: Number of candidate sites.
        seed: Random seed. If None, uses the global RANDOM_SEED.

    Returns:
        (n, 6) array of factors, each column in [0, 1].
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    factors = rng.uniform(0.0, 1.0, size=(n, 6))

    # Min-max normalize each column to exactly [0, 1]
    # (In case of degenerate column with all equal values, set to 0.5)
    for j in range(factors.shape[1]):
        col = factors[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            factors[:, j] = (col - col_min) / (col_max - col_min)
        else:
            factors[:, j] = 0.5
            warnings.warn(
                f"Factor column {j} is constant; set all values to 0.5."
            )

    return factors.astype(np.float64)


def compute_utility(factors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute composite utility U_i = weights · factors_i.

    Args:
        factors: (n, 6) array of normalized factors.
        weights: (6,) array of AHP weights (must sum to 1).

    Returns:
        (n,) array of utility scores in [0, 1].
    """
    if factors.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Factor columns ({factors.shape[1]}) must match weights size ({weights.shape[0]})."
        )

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
) -> Dict[int, List[int]]:
    """
    Generate nested subsets via farthest-point sampling (FPS).

    Starting from `start_idx`, greedily add the point farthest (in Euclidean
    distance) from the current selected set, until the largest requested size
    is reached. The resulting subsets are nested: size_1 ⊂ size_2 ⊂ ... ⊂ size_k.

    Args:
        coords: (N, 2) array of coordinates.
        sizes: List of requested subset sizes (must be sorted ascending).
        start_idx: Index to start with. If None, uses the point closest to the
                   geometric center of the domain.
        seed: Random seed for tie-breaking. If None, uses global RANDOM_SEED.
        domain_size: Domain size for center calculation.

    Returns:
        Dict mapping size -> list of indices (nested).
    """
    n_total = coords.shape[0]
    sizes = sorted(sizes)
    if sizes[-1] > n_total:
        raise ValueError(
            f"Largest requested size ({sizes[-1]}) exceeds number of points ({n_total})."
        )

    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)

    # Determine starting point
    if start_idx is None:
        center = np.array([domain_size / 2.0, domain_size / 2.0])
        distances_to_center = np.linalg.norm(coords - center, axis=1)
        # Tie-breaking: pick the first occurrence with minimal distance
        start_idx = int(np.argmin(distances_to_center))

    # Greedy FPS
    selected = [start_idx]
    # For speed, maintain a distance array to the nearest selected point
    # Initialize distances from all points to the starting point
    dist_to_selected = np.linalg.norm(coords - coords[start_idx], axis=1)

    subsets = {}
    target_idx = 0

    for size in sizes:
        while len(selected) < size:
            # Pick the point with the maximum distance to the selected set
            # Tie-breaking: choose randomly among ties (if any)
            max_dist = dist_to_selected.max()
            candidates = np.where(np.isclose(dist_to_selected, max_dist))[0]

            # Exclude already selected points
            candidates = [c for c in candidates if c not in selected]

            if not candidates:
                # Should not happen if size <= n_total
                warnings.warn(
                    f"No candidates left to reach size {size}; stopping at {len(selected)}."
                )
                break

            # Pick the first candidate (tie-breaking by random permutation)
            # This ensures reproducibility while handling ties gracefully
            perm = rng.permutation(candidates)
            new_idx = int(perm[0])

            selected.append(new_idx)

            # Update distances to the new point
            dist_to_new = np.linalg.norm(coords - coords[new_idx], axis=1)
            dist_to_selected = np.minimum(dist_to_selected, dist_to_new)

        # Store a copy of the current selected list for this size
        subsets[size] = selected.copy()

    return subsets


def create_nested_subsets(
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
    sizes: List[int],
    start_idx: Optional[int] = None,
    seed: Optional[int] = None,
    domain_size: float = 50.0,
) -> Dict[int, Dict[str, Union[np.ndarray, List[int]]]]:
    """
    Generate nested subsets with full data (coords, factors, utility, indices).

    Args:
        coords: (N, 2) master coordinates.
        factors: (N, 6) master factors.
        utility: (N,) master utility scores.
        sizes: Requested subset sizes (will be sorted).
        start_idx: Starting index for FPS. If None, uses point closest to center.
        seed: Random seed for FPS tie-breaking.
        domain_size: Domain size for center calculation.

    Returns:
        Dict mapping size -> {
            'coords': np.ndarray (size, 2),
            'factors': np.ndarray (size, 6),
            'U': np.ndarray (size,),
            'indices': List[int] (original master indices)
        }
    """
    indices_dict = farthest_point_sampling(coords, sizes, start_idx, seed, domain_size)

    subsets = {}
    for size, indices in indices_dict.items():
        indices_arr = np.array(indices, dtype=int)
        subsets[size] = {
            "coords": coords[indices_arr].copy(),
            "factors": factors[indices_arr].copy(),
            "U": utility[indices_arr].copy(),
            "indices": indices_arr.tolist(),  # store as list for JSON compatibility
        }

    return subsets


def select_existing_stations(
    n_existing: int, n_total: int, seed: Optional[int] = None
) -> List[int]:
    """
    Select random indices to serve as fixed existing stations M.

    Args:
        n_existing: Number of existing stations to select.
        n_total: Total number of master candidates.
        seed: Random seed. If None, uses global RANDOM_SEED.

    Returns:
        List of indices (length n_existing).
    """
    if n_existing > n_total:
        raise ValueError(
            f"n_existing ({n_existing}) cannot exceed n_total ({n_total})."
        )

    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    indices = rng.choice(n_total, size=n_existing, replace=False).tolist()
    return indices

def select_existing_stations_clustered(
    coords: np.ndarray,
    utility: np.ndarray,
    n_existing: int,
    max_distance: float = 10.0,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Select existing stations that are spatially coherent (clustered).

    Strategy: Pick the highest-utility point as an anchor, then greedily add
    the next highest-utility point within max_distance of any already selected point.
    This mimics real-world historical deployment (they start near pollution sources).

    Args:
        coords: (N, 2) array of coordinates.
        utility: (N,) array of utility scores.
        n_existing: Number of existing stations to select.
        max_distance: Maximum distance (km) for a station to be considered "in the cluster".
        seed: Random seed for tie-breaking.

    Returns:
        List of indices (length n_existing).
    """
    rng = np.random.RandomState(seed if seed is not None else RANDOM_SEED)
    n_total = coords.shape[0]
    
    if n_existing > n_total:
        raise ValueError(
            f"n_existing ({n_existing}) cannot exceed n_total ({n_total})."
        )
    
    if n_existing == 0:
        return []
    
    # Start with the single highest utility site
    anchor_idx = int(np.argmax(utility))
    selected = [anchor_idx]
    
    if n_existing == 1:
        return selected
    
    # Greedily add the best utility point within max_distance of the current cluster
    for _ in range(1, n_existing):
        # Calculate distances from all points to the current cluster
        dists = cdist(coords, coords[selected]).min(axis=1)
        
        # Create a mask: points not selected and within max_distance
        mask = (dists <= max_distance) & (~np.isin(np.arange(n_total), selected))
        
        # If no points are within max_distance, relax the constraint
        if not np.any(mask):
            warnings.warn(
                f"No points within {max_distance} km. Falling back to random selection for station {_+1}."
            )
            candidates = [i for i in range(n_total) if i not in selected]
            if not candidates:
                break
            new_idx = rng.choice(candidates)
            selected.append(int(new_idx))
            continue
        
        # Among valid points, pick the one with highest utility
        valid_indices = np.where(mask)[0]
        best_idx = valid_indices[np.argmax(utility[valid_indices])]
        selected.append(int(best_idx))
    
    return selected


def save_master_data(
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
    subsets: Dict[int, Dict],
    existing_indices: List[int],
    output_dir: Union[str, Path],
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save all generated data to disk.

    Files written:
        - master_coords.npy
        - master_factors.npy
        - master_U.npy
        - subsets.pkl
        - metadata.json

    Args:
        coords: (N, 2) master coordinates.
        factors: (N, 6) master factors.
        utility: (N,) master utility.
        subsets: Dict from create_nested_subsets().
        existing_indices: List of indices for existing stations M.
        output_dir: Directory to save files.
        metadata: Optional extra metadata to merge into metadata.json.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(output_path / "master_coords.npy", coords)
    np.save(output_path / "master_factors.npy", factors)
    np.save(output_path / "master_U.npy", utility)

    # Save subsets as pickle (contains nested dicts with numpy arrays)
    with open(output_path / "subsets.pkl", "wb") as f:
        pickle.dump(subsets, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Build metadata
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
    """
    Load all master data from disk.

    Args:
        data_dir: Directory containing the saved files.

    Returns:
        (coords, factors, utility, subsets, metadata)
    """
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
# VERIFICATION UTILITIES
# -----------------------------------------------------------------------------

def verify_subsets(
    subsets: Dict[int, Dict],
    coords: np.ndarray,
    factors: np.ndarray,
    utility: np.ndarray,
) -> bool:
    """
    Perform sanity checks on generated subsets.

    Checks:
        - All indices are within bounds.
        - Each subset's data matches the master data.
        - Subsets are properly nested (size_1 ⊂ size_2 ⊂ ...).
        - All subset sizes are correct.

    Returns:
        True if all checks pass, raises AssertionError otherwise.
    """
    sorted_sizes = sorted(subsets.keys())
    n_total = coords.shape[0]

    # Verify each subset's data integrity
    for size, data in subsets.items():
        indices = np.array(data["indices"])
        # Bounds check
        assert np.all((0 <= indices) & (indices < n_total)), (
            f"Subset {size}: indices out of bounds."
        )
        # Length check
        assert len(indices) == size, f"Subset {size}: expected {size} points, got {len(indices)}."

        # Data consistency
        np.testing.assert_array_equal(
            coords[indices], data["coords"],
            err_msg=f"Subset {size}: coords mismatch."
        )
        np.testing.assert_array_equal(
            factors[indices], data["factors"],
            err_msg=f"Subset {size}: factors mismatch."
        )
        np.testing.assert_array_equal(
            utility[indices], data["U"],
            err_msg=f"Subset {size}: utility mismatch."
        )

    # Verify nesting: each smaller subset should be a subset of the next larger one
    for i in range(len(sorted_sizes) - 1):
        s1 = sorted_sizes[i]
        s2 = sorted_sizes[i + 1]
        idx1 = set(subsets[s1]["indices"])
        idx2 = set(subsets[s2]["indices"])
        assert idx1.issubset(idx2), (
            f"Subset {s1} is not a subset of {s2}. "
            f"Missing indices: {idx1 - idx2}"
        )

    print("✓ All verification checks passed.")
    return True


# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -----------------------------------------------------------------------------

def generate_and_save_all(
    output_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    n_master: Optional[int] = None,
    domain_size: Optional[float] = None,
    n_existing: Optional[int] = None,
    max_existing_distance: Optional[float] = 10.0,
    subset_sizes: Optional[List[int]] = None,
) -> Dict:
    """
    Full pipeline: generate coordinates, factors, utility, subsets, and save.

    Args:
        output_dir: Directory to save data. If None, uses OUTPUT_DIR.
        seed: Random seed. If None, uses RANDOM_SEED.
        n_master: Number of master candidates. If None, uses N_MASTER.
        domain_size: Domain size in km. If None, uses DOMAIN_SIZE.
        n_existing: Number of existing stations. If None, uses N_EXISTING.
        max_existing_distance: Max cluster distance for existing stations.
        subset_sizes: List of subset sizes. If None, uses SUBSET_SIZES.

    Returns:
        Dict with keys: 'coords', 'factors', 'U', 'subsets', 'metadata'
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
    existing_indices = select_existing_stations_clustered(
        coords, utility, n_existing, max_existing_distance, seed=seed
    )
    print(f"✓ Existing stations (M): {existing_indices}")

    # Step 5: Create nested subsets using FPS starting from center
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
    )
    print(f"✓ Subsets created: {list(subsets.keys())}")

    # Step 6: Verify nesting and data integrity
    verify_subsets(subsets, coords, factors, utility)

    # Step 7: Save everything
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
            "max_existing_distance": max_existing_distance,
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
        },
    }


# -----------------------------------------------------------------------------
# COMMAND LINE INTERFACE
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for water quality monitoring QUBO."
    )
    parser.add_argument(
        "--seed", 
        type=str, 
        default="42",
        help='Random seed (integer) or "random" for time-based seed.'
    )
    parser.add_argument(
        "--n_master", 
        type=int, 
        default=None,
        help="Number of master candidate sites."
    )
    parser.add_argument(
        "--domain_size", 
        type=float, 
        default=None,
        help="Domain size in km (square)."
    )
    parser.add_argument(
        "--n_existing", 
        type=int, 
        default=None,
        help="Number of existing stations M."
    )
    parser.add_argument(
        "--max_existing_distance",
        type=float, 
        default=10.0,
        help="Maximum cluster distance (km) for existing stations. Default: 10.0"
    )
    parser.add_argument(
        "--subset_sizes", 
        type=str, 
        default=None,
        help='Comma-separated subset sizes, e.g., "10,20,30,50,100,150,200"'
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for data files."
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    
    # Parse seed
    if args.seed.lower() == "random":
        seed = int(time.time() * 1000) % 1000000
        print(f"Using random seed: {seed}")
    else:
        seed = int(args.seed)
    
    # Parse subset sizes
    if args.subset_sizes is not None:
        subset_sizes = [int(x.strip()) for x in args.subset_sizes.split(",")]
    else:
        subset_sizes = None
    
    # Generate data
    _ = generate_and_save_all(
        output_dir=args.output_dir,
        seed=seed,
        n_master=args.n_master,
        domain_size=args.domain_size,
        n_existing=args.n_existing,
        max_existing_distance=args.max_existing_distance,
        subset_sizes=subset_sizes,
    )