#@title 🎯 CELL 5: MIQP SOLVE WITH SCIP (Resumable, Config‑Aware)
"""
================================================================================
MIQP SOLVE WITH SCIP – REAL DATA (MULTI-TIER) – RESUMABLE
================================================================================
- Uses FORCE_RECOMPUTE_SCIP flag from Cell 1.
- Result filenames include config_hash (based on MASTER_QUBO_CONFIG).
- Loads hashed instance files (fallback to unhashed, auto‑migrate).
- Loads hashed master file (fallback to unhashed).
- No greedy fallback – SCIP must find a solution.
- Always generates deployment and matrix plots.
================================================================================
"""

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
import hashlib
import json

# MASTER_QUBO_CONFIG is already defined in Cell 2.
# If not, we compute it (fallback).
try:
    MASTER_QUBO_CONFIG
except NameError:
    print("⚠️ MASTER_QUBO_CONFIG not found; using defaults.")
    MASTER_QUBO_CONFIG = {
        "L_c": 7500.0,
        "L_w": 1000.0,
        "Beta": 1.0,
        "Delta": 1.0,
        "Current_vector": (1.0, 0.0),
        "K": 5,
        "D_max_buffer": 1.15,
    }
config_str = json.dumps(MASTER_QUBO_CONFIG, sort_keys=True)
config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
print(f"🔑 SCIP config hash: {config_hash}")

# Force recompute flag (defined in Cell 1)
try:
    FORCE_RECOMPUTE_SCIP
except NameError:
    FORCE_RECOMPUTE_SCIP = False
    print("⚠️ FORCE_RECOMPUTE_SCIP not defined; defaulting to False.")

# Paths
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
GDRIVE_BASE.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import sys
import time
import pickle
import re
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import jijmodeling as jm
import shutil

# OMMX SCIP adapter
try:
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("⚠️ SCIP not available. Cell 5 will fail.")

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (hashed first, fallback to unhashed)
# -----------------------------------------------------------------------------
master_path_hashed_local = LOCAL_CACHE / f"master_real_{config_hash}.pkl"
master_path_hashed_drive = GDRIVE_BASE / f"master_real_{config_hash}.pkl"
master_path_unhashed_local = LOCAL_CACHE / "master_real.pkl"
master_path_unhashed_drive = GDRIVE_BASE / "master_real.pkl"

master_data = None

# 1. Try hashed local
if master_path_hashed_local.exists():
    with open(master_path_hashed_local, "rb") as f:
        master_data = pickle.load(f)
    print("✅ Loaded hashed master from local.")
elif master_path_hashed_drive.exists():
    print("📂 Loading hashed master from Drive (copying to local)...")
    shutil.copy(master_path_hashed_drive, master_path_hashed_local)
    with open(master_path_hashed_local, "rb") as f:
        master_data = pickle.load(f)
    print("✅ Loaded hashed master from Drive.")
# 2. If not found, try unhashed (backward compatibility)
elif master_path_unhashed_local.exists():
    print("📂 Loading unhashed master from local (migrating to hashed)...")
    with open(master_path_unhashed_local, "rb") as f:
        master_data = pickle.load(f)
    # Save as hashed
    with open(master_path_hashed_local, "wb") as f:
        pickle.dump(master_data, f)
    with open(master_path_hashed_drive, "wb") as f:
        pickle.dump(master_data, f)
    print("✅ Migrated master to hashed version.")
elif master_path_unhashed_drive.exists():
    print("📂 Loading unhashed master from Drive (migrating to hashed)...")
    shutil.copy(master_path_unhashed_drive, master_path_unhashed_local)
    with open(master_path_unhashed_local, "rb") as f:
        master_data = pickle.load(f)
    with open(master_path_hashed_local, "wb") as f:
        pickle.dump(master_data, f)
    with open(master_path_hashed_drive, "wb") as f:
        pickle.dump(master_data, f)
    print("✅ Migrated master to hashed version.")

if master_data is None:
    raise FileNotFoundError("Master data not found. Run Cell 2 first.")

a_master = master_data["a"]
Q_master_edges = master_data["Q_edges"]
U_master = master_data["U"]
coords_master = master_data["coords"]
M_indices_master = master_data["M_indices"]
print(f"✅ Loaded Master Data: N={len(a_master)}, Q_edges={len(Q_master_edges)}")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (sparse energy, feasibility)
# -----------------------------------------------------------------------------
def compute_energy_sparse(x: np.ndarray, a: np.ndarray, Q_edges: list) -> float:
    x = np.asarray(x, dtype=bool)
    energy = np.dot(a, x.astype(float))
    for i, j, val in Q_edges:
        if x[i] and x[j]:
            energy += val
    return float(energy)

def check_feasibility_sparse(x, neighbors, K, fixed_neighbors=None):
    x = np.asarray(x, dtype=bool)
    selected = np.where(x)[0]
    num_selected = len(selected)
    budget_ok = (num_selected == K)

    connectivity_ok = True
    isolated_indices = []
    for i in selected:
        has_free = any(x[j] for j in neighbors[i])
        has_fixed = False
        if not has_free and fixed_neighbors is not None:
            if isinstance(fixed_neighbors, (list, np.ndarray)):
                has_fixed = bool(fixed_neighbors[i])
            elif isinstance(fixed_neighbors, dict):
                has_fixed = len(fixed_neighbors.get(i, [])) > 0
            else:
                has_fixed = False
        if not (has_free or has_fixed):
            connectivity_ok = False
            isolated_indices.append(i)

    feasible = budget_ok and connectivity_ok
    return {
        "feasible": feasible,
        "budget_ok": budget_ok,
        "connectivity_ok": connectivity_ok,
        "num_selected": num_selected,
        "isolated_indices": isolated_indices,
    }

# -----------------------------------------------------------------------------
# MIQP PROBLEM BUILDER (same as Cell 4)
# -----------------------------------------------------------------------------
def build_miqp_problem_sparse(N: int, max_degree: int, num_edges: int) -> jm.Problem:
    problem = jm.Problem("WQM_MIQP_sparse", sense=jm.ProblemSense.MINIMIZE)
    a = problem.Placeholder("a", shape=(N,), dtype=jm.DataType.FLOAT)
    neighbor_indices = problem.Placeholder("neighbor_indices", shape=(N, max_degree), dtype=jm.DataType.NATURAL)
    neighbor_mask = problem.Placeholder("neighbor_mask", shape=(N, max_degree), dtype=jm.DataType.BINARY)
    fixed_neighbors = problem.Placeholder("fixed_neighbors", shape=(N,), dtype=jm.DataType.FLOAT)
    K_total = problem.Placeholder("K_total", ndim=0, dtype=jm.DataType.INTEGER)
    x = problem.BinaryVar("x", shape=(N,))
    linear = jm.sum(jm.product(N), lambda i: a[i[0]] * x[i[0]])
    problem += linear
    if num_edges > 0:
        edges = problem.Placeholder("edges", shape=(num_edges, 2), dtype=jm.DataType.NATURAL)
        Q_vals = problem.Placeholder("Q_vals", shape=(num_edges,), dtype=jm.DataType.FLOAT)
        quad = jm.sum(jm.product(num_edges), lambda e: Q_vals[e[0]] * x[edges[e[0], 0]] * x[edges[e[0], 1]])
        problem += quad
    problem += problem.Constraint("budget", jm.sum(jm.product(N), lambda i: x[i[0]]) == K_total)
    problem += problem.Constraint(
        "connectivity",
        lambda i: x[i] <= fixed_neighbors[i] + jm.sum(
            jm.product(max_degree),
            lambda k: neighbor_mask[i, k[0]] * x[neighbor_indices[i, k[0]]]
        ),
        domain=N
    )
    return problem

# -----------------------------------------------------------------------------
# SCIP SOLVER WRAPPER (no fallback)
# -----------------------------------------------------------------------------
def solve_scip_sparse(instance_data, verbose=False):
    if not SCIP_AVAILABLE:
        raise RuntimeError("SCIP not available.")

    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"])
    Q_edges = instance_data["Q_edges"]
    neighbors = instance_data["neighbors"]
    fixed_neighbors = instance_data.get("fixed_neighbors", None)

    miqp_data = instance_data["miqp_data"]
    miqp_params = instance_data["miqp_params"]
    max_degree = miqp_params["max_degree"]
    num_edges = miqp_params["num_edges"]

    problem = build_miqp_problem_sparse(N, max_degree, num_edges)
    miqp_instance = problem.eval(miqp_data)

    start = time.perf_counter()
    try:
        adapter = OMMXPySCIPOptAdapter(miqp_instance)
        model = adapter.solver_input
        model.setParam('limits/gap', 0.0)
        model.setParam('limits/absgap', 0.0)
        model.setParam('display/verblevel', 4 if verbose else 0)
        try:
            import pyscipopt
            model.setEmphasis(pyscipopt.SCIP_PARAMEMPHASIS.OPTIMALITY, quiet=True)
        except Exception:
            pass
        model.optimize()
        scip_status = model.getStatus()
        solution = adapter.decode(model)
    except AttributeError:
        solution = OMMXPySCIPOptAdapter.solve(miqp_instance)
        scip_status = "solved"
    except Exception as e:
        raise RuntimeError(f"SCIP solver error: {e}")

    runtime = time.perf_counter() - start

    if solution is None:
        raise RuntimeError("SCIP returned no solution.")

    # Round solution
    x_sol = np.zeros(N, dtype=int)
    if hasattr(solution, "decision_variables_df"):
        df = solution.decision_variables_df
        x_df = df[df["name"] == "x"]
        for _, row in x_df.iterrows():
            if row["value"] > 0.5:
                subs = row["subscripts"]
                idx = subs[0] if isinstance(subs, (tuple, list)) and len(subs) > 0 else int(subs)
                if 0 <= idx < N:
                    x_sol[idx] = 1
    else:
        if hasattr(solution, "state") and hasattr(solution.state, "entries"):
            for var_id, value in solution.state.entries.items():
                if isinstance(var_id, tuple) and var_id[0] == "x":
                    idx = var_id[1]
                    if 0 <= idx < N:
                        x_sol[idx] = int(round(value))

    tier_energy = compute_energy_sparse(x_sol, a, Q_edges)
    feas_detail = check_feasibility_sparse(x_sol, neighbors, K, fixed_neighbors)
    feasible = feas_detail["feasible"]
    violation_rate = 0.0 if feasible else 1.0

    return {
        "solution": x_sol,
        "energy": tier_energy,
        "runtime": runtime,
        "status": scip_status,
        "feasible": feasible,
        "violation_rate": violation_rate,
        "budget_ok": feas_detail["budget_ok"],
        "connectivity_ok": feas_detail["connectivity_ok"],
    }

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (unchanged)
# -----------------------------------------------------------------------------
def plot_deployment(instance_data, result, save_path=None, show=True, dpi=150):
    coords_full = np.asarray(instance_data["original_coords"])
    D_MAX = instance_data["D_max"]
    fixed_indices = list(instance_data.get("fixed_indices", []))
    free_indices = list(instance_data.get("original_indices", []))
    snapped_indices = instance_data.get("snapped_indices", None)

    N_free = len(free_indices)
    N_total = len(coords_full)

    x_sol = result.get("solution")
    if x_sol is None:
        x_sol = np.zeros(N_free, dtype=int)
    else:
        x_sol = np.asarray(x_sol)

    selected_free_orig = [free_indices[i] for i in np.where(x_sol == 1)[0] if i < len(free_indices)]
    selected_new = [i for i in selected_free_orig if i not in fixed_indices]
    selected_m = fixed_indices
    all_selected = selected_new + selected_m

    if len(coords_master) > 0:
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

    if len(coords_master) > 0 and U_master is not None:
        sc = ax.scatter(coords_master[:, 0], coords_master[:, 1], c=U_master, cmap='viridis',
                        s=22, alpha=0.55, edgecolor='none', zorder=0, label='Utility Background')
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Utility $U_i$', fontsize=11)
    else:
        ax.scatter(coords_full[:, 0], coords_full[:, 1], c="#cbd5e1", s=22, alpha=0.6, edgecolor='none', zorder=0, label='_nolegend_')

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
    ax.set_title(f'Deployment (SCIP) – N_total={N_total}, N_free={N_free}', fontsize=13, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, markeredgecolor='black', label=f'New Stations ({len(selected_new)})'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=9, markeredgecolor='black', label=f'Existing ({len(selected_m)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, markeredgecolor='#1e293b', label=f'Candidates ({len(free_indices)})'),
        plt.Line2D([0], [0], color='#475569', linewidth=1.2, linestyle='--', label=f'Link (≤ {D_MAX:.1f}m)')
    ]
    ax_info.legend(handles=handles, loc='upper left', frameon=True, fontsize=10, title="Legend", title_fontsize=11)

    selected_new_str = ', '.join(map(str, selected_new[:10])) + ('...' if len(selected_new) > 10 else '') if selected_new else 'None'
    selected_m_str = ', '.join(map(str, selected_m[:10])) + ('...' if len(selected_m) > 10 else '') if selected_m else 'None'

    info_text = (
        f"SOLVER SUMMARY\n"
        f"---------------------------\n"
        f"Status: {result.get('status', 'N/A')}\n"
        f"Feasible: {result.get('feasible', False)}\n"
        f"Tier Energy: {result.get('energy', np.nan):.4f}\n"
        f"Master Energy: {result.get('master_energy', np.nan):.4f}\n"
        f"Runtime: {result.get('runtime', 0.0):.2f}s\n\n"
        f"PROBLEM SPECS\n"
        f"---------------------------\n"
        f"Total Stations: {N_total}\n"
        f"Free Variables (N): {N_free}\n"
        f"Target New (K): {instance_data.get('K', 0)}\n"
        f"Max Dist (D_max): {D_MAX:.1f}m\n\n"
        f"DEPLOYMENT INDICES\n"
        f"---------------------------\n"
        f"New: [{selected_new_str}]\n"
        f"Existing: [{selected_m_str}]"
    )
    ax_info.text(0.0, 0.62, info_text, transform=ax_info.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.95, edgecolor='#ced4da'))

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
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# DISCOVER INSTANCE DATA FILES (hashed first, fallback to unhashed)
# -----------------------------------------------------------------------------
pattern_hashed = re.compile(r"instance_data_N(\d+)_([a-f0-9]{8})\.pkl")
pattern_unhashed = re.compile(r"instance_data_N(\d+)\.pkl")

instance_files = []
# 1. Hashed files
for p in GDRIVE_BASE.glob("instance_data_N*_*.pkl"):
    m = pattern_hashed.match(p.name)
    if m and m.group(2) == config_hash:
        instance_files.append(p)
# 2. If none, try unhashed (fallback)
if not instance_files:
    for p in GDRIVE_BASE.glob("instance_data_N*.pkl"):
        if pattern_unhashed.match(p.name):
            instance_files.append(p)
# Also check local cache
if not instance_files:
    for p in LOCAL_CACHE.glob("instance_data_N*_*.pkl"):
        m = pattern_hashed.match(p.name)
        if m and m.group(2) == config_hash:
            instance_files.append(p)
    if not instance_files:
        for p in LOCAL_CACHE.glob("instance_data_N*.pkl"):
            if pattern_unhashed.match(p.name):
                instance_files.append(p)

if not instance_files:
    raise FileNotFoundError("No instance_data_N*.pkl files found in local or Drive.")

# Optionally filter by TARGET_N
try:
    TARGET_N = CONFIG_SA.get("TARGET_N", None)
except NameError:
    TARGET_N = None
if TARGET_N is not None:
    def extract_N(p):
        m = pattern_hashed.match(p.name) or pattern_unhashed.match(p.name)
        if m:
            return int(m.group(1))
        return None
    instance_files = [p for p in instance_files if extract_N(p) == TARGET_N]

# Sort by N
def extract_N_from_path(p):
    m = pattern_hashed.match(p.name) or pattern_unhashed.match(p.name)
    return int(m.group(1)) if m else 0

instance_files.sort(key=extract_N_from_path)
print(f"Found {len(instance_files)} instance data files:")
for p in instance_files:
    print(f"  {p.name}")

# -----------------------------------------------------------------------------
# SOLVE EACH TIER
# -----------------------------------------------------------------------------
all_results = []

for instance_path in instance_files:
    # Extract N from filename
    m = pattern_hashed.match(instance_path.name) or pattern_unhashed.match(instance_path.name)
    if not m:
        print(f"⚠️ Skipping unrecognized file: {instance_path.name}")
        continue
    N_true = int(m.group(1))

    print("\n" + "=" * 70)
    print(f"🔬 Solving tier N={N_true}")
    print("=" * 70)

    # Load instance data
    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)

    # If unhashed, save as hashed for future
    if pattern_unhashed.match(instance_path.name):
        hashed_name = f"instance_data_N{N_true}_{config_hash}.pkl"
        hashed_local = LOCAL_CACHE / hashed_name
        hashed_drive = GDRIVE_BASE / hashed_name
        with open(hashed_local, "wb") as f:
            pickle.dump(instance_data, f)
        with open(hashed_drive, "wb") as f:
            pickle.dump(instance_data, f)
        print(f"✅ Migrated instance data to hashed: {hashed_local}")
        # Use hashed path for result saving
        instance_path = hashed_local

    N_free = instance_data["N"]
    K = instance_data["K"]
    dmax = instance_data.get("D_max", 0.0)
    fixed_count = len(instance_data["fixed_indices"])
    snapped_indices = instance_data.get("snapped_indices", None)

    # Result filename with config hash
    scip_filename = f"scip_result_N{N_true}_{config_hash}.pkl"
    scip_local = LOCAL_CACHE / scip_filename
    scip_gdrive = GDRIVE_BASE / scip_filename

    # If force recompute, delete existing files
    if FORCE_RECOMPUTE_SCIP:
        for f in [scip_local, scip_gdrive]:
            if f.exists():
                f.unlink()
                print(f"🗑️  Removed {f}")

    # Load cached result if exists
    scip_result = None
    if scip_local.exists():
        print(f"📂 Loading existing SCIP result from local cache: {scip_local}")
        with open(scip_local, "rb") as f:
            scip_result = pickle.load(f)
    elif scip_gdrive.exists():
        print(f"📂 Copying SCIP result from Drive to local: {scip_gdrive} → {scip_local}")
        shutil.copy(scip_gdrive, scip_local)
        with open(scip_local, "rb") as f:
            scip_result = pickle.load(f)

    if scip_result is None:
        print("🔍 Solving with SCIP from scratch...")
        try:
            scip_result = solve_scip_sparse(instance_data, verbose=True)
            scip_result["status"] = "solved"
        except Exception as e:
            print(f"❌ SCIP solver failed: {e}")
            raise  # No fallback

        # Save result
        with open(scip_local, "wb") as f:
            pickle.dump(scip_result, f)
        with open(scip_gdrive, "wb") as f:
            pickle.dump(scip_result, f)
        print(f"✅ SCIP result saved for N={N_true}.")

    # Compute master energy
    master_energy_val = np.nan
    if scip_result.get("solution") is not None and snapped_indices is not None:
        x_sol = np.asarray(scip_result["solution"], dtype=int)
        free_indices = instance_data["original_indices"]
        fixed_indices = instance_data["fixed_indices"]

        x_master = np.zeros(len(a_master), dtype=int)
        N_free_local = len(free_indices)

        for idx, val in enumerate(x_sol):
            if val == 1:
                master_idx = snapped_indices[idx]
                if 0 <= master_idx < len(a_master):
                    x_master[master_idx] = 1

        for k, f_idx in enumerate(fixed_indices):
            master_idx = snapped_indices[N_free_local + k]
            if 0 <= master_idx < len(a_master):
                x_master[master_idx] = 1

        master_energy_val = compute_energy_sparse(x_master, a_master, Q_master_edges)
    scip_result["master_energy"] = master_energy_val

    energy = scip_result["energy"]
    feasible = scip_result["feasible"]
    runtime = scip_result["runtime"]
    status = scip_result.get("status", "unknown")

    print(f"  Status: {status}")
    print(f"  Tier Energy: {energy:.6f}")
    print(f"  Master Energy: {master_energy_val:.6f}")
    print(f"  Feasible: {feasible}")
    print(f"  Runtime: {runtime:.4f}s")

    # ---- PLOTTING ----
    deploy_save_path = LOCAL_CACHE / f"deployment_N{N_true}_{config_hash}.png"
    plot_deployment(instance_data, scip_result, save_path=deploy_save_path, show=True, dpi=150)

    matrix_save_path = LOCAL_CACHE / f"miqp_matrix_N{N_true}_{config_hash}.png"
    plot_miqp_matrix_reduced(instance_data, save_path=matrix_save_path, show=True, dpi=150,
                              title_prefix=f"$N_{{total}}$={len(instance_data['original_coords'])} | ")

    all_results.append({
        "N_true": N_true,
        "N_free": N_free,
        "K": K,
        "fixed_count": fixed_count,
        "D_max": dmax,
        "energy": energy,
        "master_energy": master_energy_val,
        "runtime": runtime,
        "feasible": feasible,
        "status": status,
        "instance_data": instance_data,
        "scip_result": scip_result,
    })

# -----------------------------------------------------------------------------
# SUMMARY TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 105)
print("📊 SUMMARY OF ALL SOLVED TIERS (WITH MASTER GRID ENERGY)")
print("=" * 105)
print(f"{'N_true':<8} | {'N_free':<8} | {'K':<4} | {'D_max (m)':<10} | {'Tier Energy':<13} | {'Master Energy':<14} | {'Runtime(s)':<11} | {'Feasible':<9} | {'Status':<12}")
print("-" * 105)
for res in all_results:
    m_energy_str = f"{res['master_energy']:.6f}" if not np.isnan(res['master_energy']) else "N/A"
    print(f"{res['N_true']:<8} | {res['N_free']:<8} | {res['K']:<4} | {res['D_max']:<10.1f} | {res['energy']:<13.6f} | {m_energy_str:<14} | {res['runtime']:<11.4f} | {str(res['feasible']):<9} | {res['status']:<12}")
print("=" * 105)

print("\n✅ All SCIP solves complete.")