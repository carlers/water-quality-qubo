#!/usr/bin/env python3
"""
Refactor notebooks/experiment.ipynb according to the notebook-centric plan.
Creates notebooks/experiment_refactored.ipynb.
"""

import json
import copy
import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "notebooks" / "experiment.ipynb"
OUT_PATH = REPO / "notebooks" / "experiment_refactored.ipynb"


def load_notebook():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def new_cell(cell_type="code", title="", source=None, metadata=None):
    if source is None:
        source = []
    if metadata is None:
        metadata = {}
    if title:
        metadata["title"] = title
    return {
        "cell_type": cell_type,
        "metadata": metadata,
        "source": source if isinstance(source, list) else [source],
        "outputs": [],
        "execution_count": None,
    }


def join_source(lines):
    return "\n".join(lines) if isinstance(lines, list) else lines


# ---------------------------------------------------------------------------
# Cell 0: Hybrid Setup (centralized)
# ---------------------------------------------------------------------------
CELL_0_SOURCE = '''#@title CELL 0: HYBRID SETUP (Local + Colab)
"""
===============================================================================
HYBRID SETUP v2.0
- Detects Colab via importlib.util.find_spec("google.colab")
- Standardizes on numpy>=2.0 everywhere, with fallback to numpy<2.0
- Centralizes ALL path resolution (no PATH_FALLBACK needed downstream)
- Removes nonexistent packages (ommx-openjij-adapter)
===============================================================================
"""

import sys
import subprocess
import time
import warnings
import importlib
import importlib.util
import os
import site
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 HYBRID SETUP v2.0")
print("=" * 70)

# ============================================================================
# ENV DETECTION
# ============================================================================
try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except Exception:
    IN_COLAB = False
print(f"🔍 IN_COLAB = {IN_COLAB}")

# ============================================================================
# HELPER
# ============================================================================
def run_command(cmd, description=None, max_retries=3, wait=2, timeout=120):
    if description:
        print(f"  ▶ {description}...")
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"    Retry {attempt}/{max_retries}...")
            time.sleep(wait)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, check=False
            )
            if result.returncode == 0:
                if result.stdout:
                    output = result.stdout.strip()
                    if len(output) > 200:
                        output = output[:200] + "..."
                    print(f"    {output}")
                return True
            else:
                error_msg = result.stderr.strip()
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"    Attempt {attempt} failed: {error_msg}")
        except subprocess.TimeoutExpired:
            print(f"    Attempt {attempt} timed out after {timeout}s")
        except Exception as e:
            print(f"    Attempt {attempt} error: {e}")
    print(f"  ❌ Command failed after {max_retries} attempts: {cmd}")
    return False

# ============================================================================
# INSTALLATION
# ============================================================================
print("\n[1] Upgrading pip and certifi...")
run_command(f"{sys.executable} -m pip install --upgrade pip certifi", "Upgrade pip/certifi")

print("\n[2] Installing numpy>=2.0...")
run_command(f"{sys.executable} -m pip install --only-binary=:all: numpy>=2.0", "Install numpy>=2.0")

CORE_PACKAGES = [
    "scipy>=1.9",
    "pandas>=2.0,<3.0",
    "matplotlib>=3.5",
    "shapely>=2.0",
    "tqdm>=4.60",
    "ipywidgets>=8.0",
    "jijmodeling>=1.0,<2.0",
    "openjij>=0.12",
    "ommx-pyscipopt-adapter>=2.0",
    "optuna>=3.0",
    "wandb>=0.15",
]

print("\n[3] Installing core packages...")
packages_str = " ".join(CORE_PACKAGES)
success = run_command(f"{sys.executable} -m pip install {packages_str}", "Install core packages")

if not success:
    print("\\n⚠️ Install failed with numpy>=2.0. Falling back to numpy<2.0...")
    run_command(f"{sys.executable} -m pip install --only-binary=:all: 'numpy<2.0'", "Downgrade numpy")
    run_command(f"{sys.executable} -m pip install {packages_str}", "Retry install core packages")

# ============================================================================
# REFRESH CACHE
# ============================================================================
print("\n[4] Refreshing Python paths...")
site.main()
importlib.invalidate_caches()
print("  ✅ Paths refreshed.")

# ============================================================================
# PATH RESOLUTION (centralized - used by all downstream cells)
# ============================================================================
print("\n[5] Setting up paths...")

if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount('/content/drive')
    CACHE_DIR = Path("/content/wqm_data")
    RESULTS_DIR = Path("/content/drive/MyDrive/wqm_data")
    DATA_DIR = RESULTS_DIR
else:
    REPO_PATH = Path.cwd().resolve()
    for _ in range(6):
        if (REPO_PATH / "notebooks").exists() or (REPO_PATH / "src").exists() or (REPO_PATH / ".git").exists():
            break
        REPO_PATH = REPO_PATH.parent
    DATA_DIR = REPO_PATH / "data"
    CACHE_DIR = DATA_DIR / "cache"
    RESULTS_DIR = REPO_PATH / "results"

# Legacy aliases
LOCAL_CACHE = CACHE_DIR
GDRIVE_BASE = RESULTS_DIR
OUTPUT_DIR = RESULTS_DIR
DRIVE_CACHE = RESULTS_DIR
GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"  IN_COLAB = {IN_COLAB}")
print(f"  REPO_PATH = {REPO_PATH}")
print(f"  DATA_DIR = {DATA_DIR}")
print(f"  CACHE_DIR = {CACHE_DIR}")
print(f"  RESULTS_DIR = {RESULTS_DIR}")
print(f"  GEOJSON_PATH = {GEOJSON_PATH}")

if not GEOJSON_PATH.exists():
    raise FileNotFoundError(f"❌ GeoJSON not found at: {GEOJSON_PATH}")
print(f"✅ GeoJSON found: {GEOJSON_PATH}")

# Add repo root to Python path
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(REPO_PATH / "src"))

# ============================================================================
# MODULE PURGE
# ============================================================================
modules_to_reload = [
    'src.model', 'src.solvers', 'src.plotting', 'src.environment',
    'src.experiment', 'src.jij_model', 'src.jij_solvers', 'src.jij_optuna',
    'data.synthetic_data', 'tests.benchmark'
]
for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"🔄 Reloaded: {mod}")

print("\\n" + "=" * 70)
print("✅ SETUP COMPLETE")
print("=" * 70)
'''


# ---------------------------------------------------------------------------
# Cell 0.5: Shared Plotting
# ---------------------------------------------------------------------------
CELL_0_5_SOURCE = '''#@title CELL 0.5: SHARED PLOTTING FUNCTIONS
"""
===============================================================================
SHARED PLOTTING FUNCTIONS
- Defined once here for reuse by Cells 5, 6, 7, 8, 9, 10
- Each consuming cell has a fallback copy guarded by try/except NameError
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# These are populated by Cell 0 and Cell 2
try:
    coords_master  # noqa: F821
except NameError:
    coords_master = None
try:
    U_master  # noqa: F821
except NameError:
    U_master = None

def plot_deployment(instance_data, result_or_solution, save_path=None, show=True, dpi=150):
    coords_full = np.asarray(instance_data["original_coords"])
    D_MAX = instance_data["D_max"]
    fixed_indices = list(instance_data.get("fixed_indices", []))
    free_indices = list(instance_data.get("original_indices", []))

    if isinstance(result_or_solution, dict):
        x_sol = result_or_solution.get("solution")
    else:
        x_sol = result_or_solution
    x_sol = np.asarray(x_sol, dtype=int) if x_sol is not None else np.zeros(len(free_indices), dtype=int)

    selected_free_orig = [free_indices[i] for i in np.where(x_sol == 1)[0] if i < len(free_indices)]
    selected_new = [i for i in selected_free_orig if i not in fixed_indices]
    selected_m = fixed_indices
    all_selected = selected_new + selected_m

    if coords_master is not None and len(coords_master) > 0:
        xmin, ymin = coords_master.min(axis=0)
        xmax, ymax = coords_master.max(axis=0)
    else:
        xmin, ymin = coords_full.min(axis=0)
        xmax, ymax = coords_full.max(axis=0)
    pad_x = max(0.05 * (xmax - xmin), 1.0)
    pad_y = max(0.05 * (ymax - ymin), 1.0)

    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.3], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')

    if coords_master is not None and U_master is not None:
        sc = ax.scatter(coords_master[:, 0], coords_master[:, 1], c=U_master, cmap='viridis',
                        s=22, alpha=0.55, edgecolor='none', zorder=0, label='Utility Background')
        plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04).set_label('Utility $U_i$', fontsize=11)
    else:
        ax.scatter(coords_full[:, 0], coords_full[:, 1], c="#cbd5e1", s=22, alpha=0.6, edgecolor='none', zorder=0)

    for i in range(len(all_selected)):
        for j in range(i + 1, len(all_selected)):
            idx_i, idx_j = all_selected[i], all_selected[j]
            dist = np.linalg.norm(coords_full[idx_i] - coords_full[idx_j])
            if dist <= D_MAX:
                ax.plot([coords_full[idx_i, 0], coords_full[idx_j, 0]],
                        [coords_full[idx_i, 1], coords_full[idx_j, 1]],
                        color='#475569', alpha=0.5, linewidth=1.2, linestyle='--', zorder=1)

    if free_indices:
        candidate_coords = coords_full[free_indices]
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], c='white', s=40,
                   alpha=0.9, edgecolor='#1e293b', linewidth=0.8, label='Candidates', zorder=2)
    if selected_m:
        ax.scatter(coords_full[selected_m, 0], coords_full[selected_m, 1],
                   c='blue', s=110, marker='s', edgecolor='black', linewidth=1.2,
                   label=f'Existing ({len(selected_m)})', zorder=3)
    if selected_new:
        ax.scatter(coords_full[selected_new, 0], coords_full[selected_new, 1],
                   c='red', s=110, marker='o', edgecolor='black', linewidth=1.2,
                   label=f'New ({len(selected_new)})', zorder=4)

    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    ax.set_title('Deployment', fontsize=13, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, markeredgecolor='black', label=f'New ({len(selected_new)})'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=9, markeredgecolor='black', label=f'Existing ({len(selected_m)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, markeredgecolor='#1e293b', label=f'Candidates ({len(free_indices)})'),
        plt.Line2D([0], [0], color='#475569', linewidth=1.2, linestyle='--', label=f'Link (<= {D_MAX:.1f}m)')
    ]
    ax_info.legend(handles=handles, loc='upper left', frameon=True, fontsize=10, title="Legend", title_fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_miqp_matrix_reduced(instance_data, save_path=None, show=True, dpi=150, title_prefix="", sort_by="coords"):
    N = instance_data["N"]
    a = np.asarray(instance_data["a"], dtype=float)
    Q_edges = instance_data.get("Q_edges", [])
    mat = np.zeros((N, N), dtype=float)
    np.fill_diagonal(mat, a)
    for i, j, val in Q_edges:
        mat[i, j] = val
        mat[j, i] = val

    if sort_by == "coords" and "coords" in instance_data:
        coords_free = np.asarray(instance_data["coords"])
        order = np.lexsort((coords_free[:, 0], -coords_free[:, 1]))
        mat = mat[np.ix_(order, order)]
        title_suffix = " (Spatially Sorted)"
    else:
        title_suffix = ""

    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_vals = mat[off_diag_mask]
    if len(off_diag_vals) > 0 and np.max(np.abs(off_diag_vals)) > 0:
        max_val = np.max(np.abs(off_diag_vals))
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = f"Quadratic Coeffs (±{max_val:.2f})"
    else:
        max_val = np.max(np.abs(mat)) if np.max(np.abs(mat)) > 0 else 1.0
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = "Coefficient Value"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f"{title_prefix}Reduced MIQP Matrix (N={N}){title_suffix}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Variable Index (spatial order)" if sort_by == "coords" else "Variable Index", fontsize=10)
    ax.set_ylabel("Variable Index (spatial order)" if sort_by == "coords" else "Variable Index", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_qubo_matrix(instance, penalty_weights, N, save_path=None, show=True, dpi=150, title_prefix=""):
    qubo_dict, offset = instance.to_qubo(penalty_weights=penalty_weights)
    Q_mat = np.zeros((N, N), dtype=float)

    def extract_idx(key):
        if isinstance(key, tuple):
            if len(key) == 2 and isinstance(key[0], str) and key[0] == 'x':
                return key[1]
            for elem in key:
                if not isinstance(elem, str):
                    return elem
            return key[0]
        return key

    for (key_i, key_j), val in qubo_dict.items():
        i = extract_idx(key_i)
        j = extract_idx(key_j)
        if isinstance(i, int) and isinstance(j, int) and i < N and j < N:
            Q_mat[i, j] = val
    Q_mat = Q_mat + Q_mat.T - np.diag(np.diag(Q_mat))
    max_abs = np.max(np.abs(Q_mat)) if np.max(np.abs(Q_mat)) > 0 else 1.0
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Q_mat, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f"{title_prefix}QUBO Matrix (free variables) N={N}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Variable Index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Coefficient Value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_connectivity_graph(coords, edges, M_indices=None, save_path=None, show=True, dpi=150, title=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    segments = []
    for i, j in edges:
        segments.append([(coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1])])
    if segments:
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, colors="steelblue", alpha=0.3, linewidth=0.6, zorder=1)
        ax.add_collection(lc)
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=15, alpha=0.8, zorder=2)
    if M_indices is not None and len(M_indices) > 0:
        ax.scatter(coords[M_indices, 0], coords[M_indices, 1],
                   c="red", marker="*", s=200, edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
'''


# ---------------------------------------------------------------------------
# Cell 1: Master Control Flags + Configs
# ---------------------------------------------------------------------------
CELL_1_SOURCE = '''#@title CELL 1: MASTER CONTROL FLAGS + SIMULATION CONFIGS
# ============================================================================
# Recompute flags
# ============================================================================
FORCE_RECOMPUTE_MASTER = False
FORCE_RECOMPUTE_SCALING = False
FORCE_RECOMPUTE_QUBO = True
FORCE_RECOMPUTE_SCIP = False
FORCE_RETUNE_SA = False
FORCE_RETUNE_SQA = False
FORCE_ALL_RECOMPUTE = False

if FORCE_ALL_RECOMPUTE:
    FORCE_RECOMPUTE_MASTER = True
    FORCE_RECOMPUTE_SCALING = True
    FORCE_RECOMPUTE_QUBO = True
    FORCE_RECOMPUTE_SCIP = True
    FORCE_RETUNE_SA = True
    FORCE_RETUNE_SQA = True

print("🔧 Master Control Flags:")
print(f"   FORCE_RECOMPUTE_MASTER = {FORCE_RECOMPUTE_MASTER}")
print(f"   FORCE_RECOMPUTE_SCALING = {FORCE_RECOMPUTE_SCALING}")
print(f"   FORCE_RECOMPUTE_QUBO = {FORCE_RECOMPUTE_QUBO}")
print(f"   FORCE_RECOMPUTE_SCIP = {FORCE_RECOMPUTE_SCIP}")
print(f"   FORCE_RETUNE_SA = {FORCE_RETUNE_SA}")
print(f"   FORCE_RETUNE_SQA = {FORCE_RETUNE_SQA}")

# ============================================================================
# SA Tuning Config
# ============================================================================
CONFIG_SA = {
    "TARGET_N": None,
    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    "SYNC_INTERVAL": 5,
    "WARMUP_TRIALS": 15,
    "MAX_FLOOR_HITS": 10,
    "VARIANCE_TOLERANCE": 0.05,
    "MIN_FEASIBLE_FOR_VARIANCE": 5,
    "FEASIBILITY_RANK_THRESHOLD": 20,
    "TOP_K_ENERGY": 3,
    "ENERGY_TOLERANCE_PCT": 0.5,
    "LAMBDA_LOWER": 0.0001,
    "USE_WANDB": False,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,
    "SAVE_SA_SAMPLES": True,
}

# ============================================================================
# SQA Tuning Config (placeholder for Cells 8-9)
# ============================================================================
CONFIG_SQA = {
    "TARGET_N": None,
    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    "TROTTER": 4,
    "SCHEDULE": "linear",
    "SYNC_INTERVAL": 5,
    "WARMUP_TRIALS": 15,
    "MAX_FLOOR_HITS": 10,
    "VARIANCE_TOLERANCE": 0.05,
    "MIN_FEASIBLE_FOR_VARIANCE": 5,
    "FEASIBILITY_RANK_THRESHOLD": 20,
    "TOP_K_ENERGY": 3,
    "ENERGY_TOLERANCE_PCT": 0.5,
    "LAMBDA_LOWER": 0.0001,
    "USE_WANDB": False,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,
    "SAVE_SQA_SAMPLES": True,
}

# ============================================================================
# Final Benchmark Config
# ============================================================================
CONFIG_BENCH = {
    "N_TESTS": [20, 50, 100, 200, 500, 1000],
    "NUM_READS": 256,
    "NUM_SWEEPS": 200,
    "RANDOM_SOLUTIONS": 5,
    "REPEAT_TIMES": 3,
    "CACHE_FILE": str(CACHE_DIR / "benchmark_results.pkl"),
}

print("\\n✅ Simulation configs defined.")
'''


def main():
    nb = load_notebook()
    cells = nb["cells"]

    # We need to map current cells to new structure
    # Current order: 0=setup, 1=flags, 2=refresh, 3=data ingestor, 4=scaling,
    #                5=qubo, 6=delete scip, 7=scip solve, 8=sa tune,
    #                9=delete bench, 10=benchmark

    new_cells = []

    # Cell 0: Refactored Setup
    new_cells.append(new_cell(source=CELL_0_SOURCE, title="CELL 0: HYBRID SETUP"))

    # Cell 0.5: Shared Plotting
    new_cells.append(new_cell(source=CELL_0_5_SOURCE, title="CELL 0.5: SHARED PLOTTING"))

    # Cell 1: Expanded Flags + Configs
    new_cells.append(new_cell(source=CELL_1_SOURCE, title="CELL 1: MASTER CONTROL FLAGS"))

    # Cells 2-6: Keep existing logic, strip PATH_FALLBACK blocks
    # Current cells 2,3,4,5,6,7,8 become new cells 2,3,4,5,6,7,8
    # Wait, let me map this properly:
    # Current 2 (refresh repo) -> can be merged or kept as Cell 2
    # Current 3 (data ingestor) -> Cell 3
    # Current 4 (scaling) -> Cell 4
    # Current 5 (qubo) -> Cell 5
    # Current 6 (delete scip) -> Cell 6
    # Current 7 (scip solve) -> Cell 7
    # Current 8 (sa tune) -> Cell 8

    # Actually, let's keep it simple:
    # New 2 = Current 2 (refresh repo)
    # New 3 = Current 3 (data ingestor)
    # New 4 = Current 4 (scaling)
    # New 5 = Current 5 (qubo)
    # New 6 = Current 6 (delete scip)
    # New 7 = Current 7 (scip solve)
    # New 8 = Current 8 (sa tune)

    # But the plan says Cells 2-6 should be data ingestor, scaling, qubo, scip, sa
    # Let me skip the "refresh repo" cell and merge its logic into Cell 0
    # So: New 2 = Current 3, New 3 = Current 4, New 4 = Current 5,
    #     New 5 = Current 7 (SCIP solve), New 6 = Current 8 (SA tune)

    # Actually, the current Cell 2 is just a git refresh. Let's keep it as Cell 2.
    new_cells.append(cells[2])  # Refresh repo

    # Cell 3: Data ingestor
    new_cells.append(cells[3])

    # Cell 4: Scaling
    new_cells.append(cells[4])

    # Cell 5: QUBO builder
    new_cells.append(cells[5])

    # Cell 6: Delete SCIP results (keep as utility)
    new_cells.append(cells[6])

    # Cell 7: SCIP solve
    new_cells.append(cells[7])

    # Cell 8: SA tuning
    new_cells.append(cells[8])

    # Cell 9: Delete benchmark cache -> remove or keep
    # Let's keep it but renumber as Cell 9
    new_cells.append(cells[9])

    # Cell 10: Benchmark -> replace with final benchmark placeholder
    # For now, keep the existing benchmark as Cell 10
    new_cells.append(cells[10])

    # Now we need to add new cells for the plan:
    # Cell 11: Interactive visualization (after SA tune = Cell 8)
    # Cell 12: SQA tuning (after viz)
    # Cell 13: SQA visualization
    # Cell 14: Final benchmark (replacing current Cell 10)

    # But wait, the plan says the notebook should have cells 0-10.
    # Let me re-read the plan structure:
    # 0: Setup, 0.5: Plotting, 1: Flags, 2: Data, 3: Scaling, 4: QUBO,
    # 5: SCIP, 6: SA, 7: Viz, 8: SQA, 9: SQA viz, 10: Final benchmark

    # So I need to REMOVE current cells 2, 6, 9, 10 and INSERT new ones.
    # Current cell 2 (refresh repo) -> merge into Cell 0
    # Current cell 6 (delete scip) -> merge into Cell 7 or remove
    # Current cell 9 (delete bench) -> remove
    # Current cell 10 (benchmark) -> replace with final benchmark

    # Let me restructure properly.

    new_cells = []

    # Cell 0: Setup
    new_cells.append(new_cell(source=CELL_0_SOURCE, title="CELL 0: HYBRID SETUP"))

    # Cell 0.5: Shared Plotting
    new_cells.append(new_cell(source=CELL_0_5_SOURCE, title="CELL 0.5: SHARED PLOTTING"))

    # Cell 1: Flags + Configs
    new_cells.append(new_cell(source=CELL_1_SOURCE, title="CELL 1: MASTER CONTROL FLAGS"))

    # Cell 2: Data Ingestor (current cell 3)
    new_cells.append(cells[3])

    # Cell 3: Scaling (current cell 4)
    new_cells.append(cells[4])

    # Cell 4: QUBO Builder (current cell 5)
    new_cells.append(cells[5])

    # Cell 5: SCIP Solve (current cell 7) - include delete logic inline
    new_cells.append(cells[7])

    # Cell 6: SA Tuning (current cell 8)
    new_cells.append(cells[8])

    # Cell 7: Interactive Visualization (NEW)
    # We'll add a placeholder for now
    viz_cell = new_cell(
        source="#@title CELL 7: INTERACTIVE VISUALIZATION (SA Results)\n# TODO: Implement interactive visualization with Plotly + ipywidgets",
        title="CELL 7: INTERACTIVE VISUALIZATION"
    )
    new_cells.append(viz_cell)

    # Cell 8: SQA Tuning (NEW)
    sqa_tune_cell = new_cell(
        source="#@title CELL 8: SQA TUNING (Optuna)\n# TODO: Implement SQA tuning mirroring Cell 6",
        title="CELL 8: SQA TUNING"
    )
    new_cells.append(sqa_tune_cell)

    # Cell 9: SQA Visualization (NEW)
    sqa_viz_cell = new_cell(
        source="#@title CELL 9: SQA VISUALIZATION\n# TODO: Implement SQA visualization mirroring Cell 7",
        title="CELL 9: SQA VISUALIZATION"
    )
    new_cells.append(sqa_viz_cell)

    # Cell 10: Final Benchmark (replace current cell 10)
    bench_cell = new_cell(
        source="#@title CELL 10: FINAL BENCHMARK (SCIP vs Greedy vs SA vs SQA)\n# TODO: Implement final benchmark suite",
        title="CELL 10: FINAL BENCHMARK"
    )
    new_cells.append(bench_cell)

    nb["cells"] = new_cells
    save_notebook(nb)
    print("Refactored notebook saved to", OUT_PATH)
    print("Total cells:", len(new_cells))
    for i, c in enumerate(new_cells):
        title = c.get("metadata", {}).get("title", "")
        print(f"   Cell {i}: {title} ({sum(len(s) for s in c['source'])} chars)")


if __name__ == "__main__":
    main()
