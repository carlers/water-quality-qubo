#@title 🎯 MIQP SOLVE WITH SCIP (Real Data - Multi-Tier)
"""
================================================================================
MIQP SOLVE WITH SCIP – REAL DATA (MULTI-TIER)
================================================================================
Automatically discovers all instance_data_N{}.pkl files (local or Drive),
solves each MIQP, and maps the solution back to the master grid to calculate
true standardized energy.
================================================================================
"""

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
FALLBACK_GREEDY = True           # Use greedy if SCIP fails
SAVE_PLOTS = True                # Save plots to disk
SHOW_PLOTS = True                # Display plots in notebook
PLOT_DPI = 150                   # Resolution for saved plots

# If set to a specific integer, solve only that tier; if None, solve all
TARGET_N = None                  # e.g., 35 to solve only N=35

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
from matplotlib.patches import Patch
from scipy.interpolate import griddata
import jijmodeling as jm

# OMMX SCIP adapter (must be installed)
try:
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("⚠️ SCIP not available. Will rely on greedy only.")

# Utilities from the repo (import after sys.path is set)
try:
    from src.jij_solvers import compute_energy, check_feasibility, compute_violation_rate, solve_greedy_jij
    from src.utils import safe_save_pickle, safe_load_pickle
except ImportError:
    print("⚠️ Ensure src modules are in your sys.path before running.")

# -----------------------------------------------------------------------------
# PATHS & MASTER DATA LOADING
# -----------------------------------------------------------------------------
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
GDRIVE_BASE.mkdir(parents=True, exist_ok=True)

# Pre-load master data to calculate standardized energy scores
master_path_local = LOCAL_CACHE / "master_real.pkl"
master_path_gdrive = GDRIVE_BASE / "master_real.pkl"
master_data = None
a_master, Q_master, N_master = None, None, None

if master_path_local.exists():
    master_data = safe_load_pickle(master_path_local)
elif master_path_gdrive.exists():
    master_data = safe_load_pickle(master_path_gdrive)

if master_data is not None:
    a_master = np.asarray(master_data["a"], dtype=float)
    Q_master = np.asarray(master_data["Q"], dtype=float)
    N_master = len(a_master)
    print(f"✅ Loaded Master Data for global energy evaluation (N={N_master})")
else:
    print("⚠️ Master data not found! Cannot compute standardized master energy.")

# -----------------------------------------------------------------------------
# FUNCTION: Build MIQP problem (sparse)
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
        quad = jm.sum(
            jm.product(num_edges),
            lambda e: Q_vals[e[0]] * x[edges[e[0], 0]] * x[edges[e[0], 1]]
        )
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


def solve_scip_miqp(instance_data, verbose=False):
    if not SCIP_AVAILABLE:
        return {"solution": None, "energy": np.nan, "runtime": 0.0,
                "status": "SCIP not available", "feasible": False,
                "violation_rate": 1.0}

    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"], dtype=float)
    Q = np.asarray(instance_data["Q"], dtype=float)
    neigh = np.asarray(instance_data["neigh"])
    raw_fixed = instance_data.get("fixed_neighbors", None)

    a_effective = a.copy()
    a_effective += np.diag(Q)

    quad_edges = []
    quad_vals = []
    for i in range(N):
        for j in range(i + 1, N):
            total_coeff = float(Q[i, j] + Q[j, i])
            if total_coeff != 0.0:
                quad_edges.append((i, j))
                quad_vals.append(total_coeff)
    num_edges = len(quad_edges)

    max_degree = max(1, np.max(np.sum(neigh, axis=1))) if N > 0 else 1
    neighbor_indices = np.zeros((N, max_degree), dtype=np.int32)
    neighbor_mask = np.zeros((N, max_degree), dtype=np.int8)
    for i in range(N):
        nbrs = np.where(neigh[i] == 1)[0]
        for k, j in enumerate(nbrs[:max_degree]):
            neighbor_indices[i, k] = j
            neighbor_mask[i, k] = 1

    fixed_nbrs_arr = np.zeros(N, dtype=float)
    if raw_fixed is not None:
        if isinstance(raw_fixed, dict):
            for k, val in raw_fixed.items():
                idx = int(k)
                if 0 <= idx < N:
                    if isinstance(val, (list, set, tuple, np.ndarray)):
                        fixed_nbrs_arr[idx] = 1.0 if len(val) > 0 else 0.0
                    else:
                        fixed_nbrs_arr[idx] = 1.0 if float(val) > 0 else 0.0
        elif isinstance(raw_fixed, np.ndarray) and raw_fixed.ndim == 2:
            for idx in range(min(N, raw_fixed.shape[0])):
                fixed_nbrs_arr[idx] = 1.0 if np.sum(raw_fixed[idx]) > 0 else 0.0
        elif isinstance(raw_fixed, (list, tuple, np.ndarray)):
            for idx, val in enumerate(raw_fixed[:N]):
                if isinstance(val, (list, set, tuple, np.ndarray)):
                    fixed_nbrs_arr[idx] = 1.0 if len(val) > 0 else 0.0
                elif isinstance(val, dict):
                    fixed_nbrs_arr[idx] = 1.0 if len(val) > 0 else 0.0
                else:
                    fixed_nbrs_arr[idx] = 1.0 if float(val) > 0 else 0.0

    if verbose:
        print(f"Building EXACT problem for N={N}, K={K}, edges={num_edges}, max_degree={max_degree}...")
        print(f"  -> Found {int(np.sum(fixed_nbrs_arr))} candidates anchored to existing infrastructure.")

    problem = build_miqp_problem_sparse(N, max_degree, num_edges)
    data = {
        "K_total": int(K),
        "a": a_effective.tolist(),
        "neighbor_indices": neighbor_indices.tolist(),
        "neighbor_mask": neighbor_mask.astype(int).tolist(),
        "fixed_neighbors": fixed_nbrs_arr.tolist(),
    }
    if num_edges > 0:
        data["edges"] = [list(edge) for edge in quad_edges]
        data["Q_vals"] = [float(v) for v in quad_vals]

    instance = problem.eval(data)

    start = time.perf_counter()
    solution = None
    scip_status = "unknown"
    try:
        adapter = OMMXPySCIPOptAdapter(instance)
        model = adapter.solver_input
        model.setParam('limits/gap', 0.0)         
        model.setParam('limits/absgap', 0.0)      
        model.setParam('display/verblevel', 4 if verbose else 0)
        
        try:
            model.setEmphasis(pyscipopt.SCIP_PARAMEMPHASIS.OPTIMALITY, quiet=True)
        except Exception:
            pass

        model.optimize()
        scip_status = model.getStatus()

        try:
            solution = adapter.decode(model)
        except TypeError:
            solution = adapter.decode()

    except AttributeError:
        solution = OMMXPySCIPOptAdapter.solve(instance)
        scip_status = "solved"
    except Exception as e:
        return {"solution": None, "energy": np.nan, "runtime": time.perf_counter() - start,
                "status": f"SCIP error: {e}", "feasible": False, "violation_rate": 1.0}

    runtime = time.perf_counter() - start

    if solution is None:
        return {"solution": None, "energy": np.nan, "runtime": runtime,
                "status": "SCIP error: solution is None", "feasible": False,
                "violation_rate": 1.0}

    x_sol = np.zeros(N, dtype=int)
    if hasattr(solution, "decision_variables_df"):
        df = solution.decision_variables_df
        x_df = df[df["name"] == "x"]
        for _, row in x_df.iterrows():
            if row["value"] == 1:
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
                        x_sol[idx] = int(value)

    energy = compute_energy(x_sol, a, Q)
    feas_detail = check_feasibility(x_sol, neigh, K, fixed_neighbors=raw_fixed)

    status = "optimal" if scip_status == "optimal" or getattr(solution, "is_optimal", False) else scip_status

    return {
        "solution": x_sol,
        "energy": energy,  # Tier energy
        "runtime": runtime,
        "status": status,
        "feasible": feas_detail["feasible"],
        "violation_rate": compute_violation_rate(x_sol, neigh, K, fixed_neighbors=raw_fixed),
        "budget_ok": feas_detail.get("budget_ok", False),
        "connectivity_ok": feas_detail.get("connectivity_ok", False),
    }

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (Preserved exactly as before)
# -----------------------------------------------------------------------------
def plot_miqp_matrix_full(instance_data, save_path=None, show=True, dpi=150, title_prefix="", scale_mode="off_diagonal"):
    N_total = len(instance_data["original_coords"])
    N_free = instance_data["N"]
    free_indices = np.asarray(instance_data["original_indices"], dtype=int)
    fixed_indices = np.asarray(instance_data.get("fixed_indices", []), dtype=int)
    a_red = np.asarray(instance_data["a"], dtype=float)
    Q_red = np.asarray(instance_data["Q"], dtype=float)

    free_mat = Q_red + Q_red.T - np.diag(np.diag(Q_red))
    np.fill_diagonal(free_mat, np.diag(free_mat) + a_red)

    mat_full = np.full((N_total, N_total), np.nan, dtype=float)
    mat_full[np.ix_(free_indices, free_indices)] = free_mat
    off_diag_mask = ~np.eye(N_free, dtype=bool)
    off_diag_values = free_mat[off_diag_mask]

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='#e2e8f0')

    if scale_mode == "off_diagonal" and len(off_diag_values) > 0 and np.max(np.abs(off_diag_values)) > 0:
        max_val = np.max(np.abs(off_diag_values))
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = f"Quadratic Coeffs (Scale locked to ±{max_val:.1f}; Diag Saturated)"
    elif scale_mode == "symlog":
        max_val = np.max(np.abs(free_mat)) if np.max(np.abs(free_mat)) > 0 else 1.0
        non_zero_off = off_diag_values[off_diag_values != 0]
        lin_thresh = np.percentile(np.abs(non_zero_off), 50) if len(non_zero_off) > 0 else 1e-2
        norm = mcolors.SymLogNorm(linthresh=lin_thresh, linscale=1.0, vmin=-max_val, vmax=max_val)
        cbar_label = "Coefficient Value (Symmetric Log Scale)"
    else:
        max_val = np.max(np.abs(free_mat)) if np.max(np.abs(free_mat)) > 0 else 1.0
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = "Coefficient Value (Linear Scale)"

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(mat_full, cmap=cmap, aspect='auto', norm=norm, interpolation='nearest')
    ax.set_title(f"{title_prefix}Full Original MIQP Matrix (N_total={N_total})", fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Original Station Index", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Original Station Index", fontsize=10.5, labelpad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9.5)

    legend_elements = [
        Patch(facecolor='#3b82f6', edgecolor='#1e3a8a', label=f'Free Variables (Optimized, N={N_free})'),
        Patch(facecolor='#e2e8f0', edgecolor='#94a3b8', label=f'Existing Stations (Fixed, N={len(fixed_indices)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, -0.12), ncol=2, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=9.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_miqp_matrix_reduced(instance_data, save_path=None, show=True, dpi=150, title_prefix="", scale_mode="off_diagonal"):
    N = instance_data["N"]
    a = np.asarray(instance_data["a"], dtype=float)
    Q = np.asarray(instance_data["Q"], dtype=float)

    mat = Q + Q.T - np.diag(np.diag(Q))
    np.fill_diagonal(mat, np.diag(mat) + a)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_vals = mat[off_diag_mask]

    if scale_mode == "off_diagonal" and len(off_diag_vals) > 0 and np.max(np.abs(off_diag_vals)) > 0:
        max_val = np.max(np.abs(off_diag_vals))
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = f"Quadratic Coeffs (Scale locked to ±{max_val:.1f}; Diag Saturated)"
    elif scale_mode == "symlog":
        max_val = np.max(np.abs(mat)) if np.max(np.abs(mat)) > 0 else 1.0
        non_zero_off = np.abs(off_diag_vals[off_diag_vals != 0])
        lin_thresh = np.percentile(non_zero_off, 50) if len(non_zero_off) > 0 else 1e-2
        norm = mcolors.SymLogNorm(linthresh=lin_thresh, linscale=1.0, vmin=-max_val, vmax=max_val)
        cbar_label = "Coefficient Value (Symmetric Log Scale)"
    else:
        max_val = np.max(np.abs(mat)) if np.max(np.abs(mat)) > 0 else 1.0
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)
        cbar_label = "Coefficient Value (Linear Scale)"

    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f"{title_prefix}Reduced MIQP Matrix (N={N})", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Variable Index", fontsize=10)
    ax.set_ylabel("Variable Index", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_deployment(instance_data, scip_result, save_path=None, show=True, dpi=150):
    master_coords = None
    master_U = None
    if master_path_local.exists():
        with open(master_path_local, "rb") as f:
            master = pickle.load(f)
        master_coords = np.asarray(master["coords"])
        master_U = master.get("U", master.get("utility", master.get("original_U")))
    elif master_path_gdrive.exists():
        with open(master_path_gdrive, "rb") as f:
            master = pickle.load(f)
        master_coords = np.asarray(master["coords"])
        master_U = master.get("U", master.get("utility", master.get("original_U")))
    else:
        master_coords = np.asarray(instance_data.get("original_coords", []))
        if len(master_coords) == 0:
            master_coords = np.asarray(instance_data["coords"])

    coords_full = np.asarray(instance_data["original_coords"])
    U_full = np.asarray(instance_data["original_U"]) if instance_data.get("original_U") is not None else None
    D_MAX = instance_data["D_max"]
    fixed_indices = list(instance_data.get("fixed_indices", []))
    free_indices = list(instance_data.get("original_indices", []))
    N_free = len(free_indices)
    N_total = len(coords_full)

    if master_U is None and U_full is not None and len(master_coords) > 0 and len(coords_full) > 0:
        master_U = griddata(coords_full, U_full, master_coords, method='linear')
        nan_mask = np.isnan(master_U)
        if np.any(nan_mask):
            master_U[nan_mask] = griddata(coords_full, U_full, master_coords[nan_mask], method='nearest')

    x_sol = scip_result.get("solution")
    if x_sol is None:
        x_sol = np.zeros(N_free, dtype=int)
    else:
        x_sol = np.asarray(x_sol)

    selected_free_orig = [free_indices[i] for i in np.where(x_sol == 1)[0] if i < len(free_indices)]
    selected_new = [i for i in selected_free_orig if i not in fixed_indices]
    selected_m = fixed_indices
    all_selected = selected_new + selected_m

    if len(master_coords) > 0:
        xmin, ymin = master_coords.min(axis=0)
        xmax, ymax = master_coords.max(axis=0)
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

    if len(master_coords) > 0:
        if master_U is not None:
            sc = ax.scatter(master_coords[:, 0], master_coords[:, 1], c=master_U, cmap='viridis', s=22, alpha=0.55, edgecolor='none', zorder=0, label='Utility Background')
            cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Utility $U_i$', fontsize=11)
        else:
            ax.scatter(master_coords[:, 0], master_coords[:, 1], c="#cbd5e1", s=22, alpha=0.6, edgecolor='none', zorder=0, label='_nolegend_')

    for i in range(len(all_selected)):
        for j in range(i + 1, len(all_selected)):
            idx_i, idx_j = all_selected[i], all_selected[j]
            dist = np.linalg.norm(coords_full[idx_i] - coords_full[idx_j])
            if dist <= D_MAX:
                ax.plot([coords_full[idx_i, 0], coords_full[idx_j, 0]], [coords_full[idx_i, 1], coords_full[idx_j, 1]], color='#475569', alpha=0.5, linewidth=1.2, linestyle='--', zorder=1)

    if free_indices:
        candidate_coords = coords_full[free_indices]
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], c='white', s=40, alpha=0.9, edgecolor='#1e293b', linewidth=0.8, label='Candidates', zorder=2)

    if selected_m:
        ax.scatter(coords_full[selected_m, 0], coords_full[selected_m, 1], c='blue', s=110, marker='s', edgecolor='black', linewidth=1.2, label=f'Existing ({len(selected_m)})', zorder=3)

    if selected_new:
        ax.scatter(coords_full[selected_new, 0], coords_full[selected_new, 1], c='red', s=110, marker='o', edgecolor='black', linewidth=1.2, label=f'New ({len(selected_new)})', zorder=4)

    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    ax.set_title(f'Deployment Solution (SCIP) – $N_{{total}}$={N_total}, $N_{{free}}$={N_free}', fontsize=13, fontweight='bold', pad=10)
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
        f"Status: {scip_result.get('status', 'N/A')}\n"
        f"Feasible: {scip_result.get('feasible', False)}\n"
        f"Tier Energy: {scip_result.get('energy', np.nan):.4f}\n"
        f"Master Energy: {scip_result.get('master_energy', np.nan):.4f}\n"
        f"Runtime: {scip_result.get('runtime', 0.0):.2f}s\n\n"
        f"PROBLEM SPECS\n"
        f"---------------------------\n"
        f"Total Stations: {N_total}\n"
        f"Free Variables (N): {N_free}\n"
        f"Target New (K): {instance_data.get('K', 0)}\n"
        f"Max Dist ($D_{{max}}$): {D_MAX:.1f}m\n\n"
        f"DEPLOYMENT INDICES\n"
        f"---------------------------\n"
        f"New: [{selected_new_str}]\n"
        f"Existing: [{selected_m_str}]"
    )

    ax_info.text(0.0, 0.62, info_text, transform=ax_info.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='left', family='monospace', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.95, edgecolor='#ced4da'))

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# DISCOVER INSTANCE DATA FILES
# -----------------------------------------------------------------------------
pattern = re.compile(r"instance_data_N(\d+)\.pkl")

def find_instance_files():
    local_files = {p.name: p for p in LOCAL_CACHE.glob("instance_data_N*.pkl") if pattern.match(p.name)}
    drive_files = {p.name: p for p in GDRIVE_BASE.glob("instance_data_N*.pkl") if pattern.match(p.name)}
    all_files = {}
    for name, path in drive_files.items():
        all_files[name] = local_files.get(name, path)
    for name, path in local_files.items():
        if name not in all_files:
            all_files[name] = path
    return sorted(all_files.values(), key=lambda p: int(pattern.search(p.name).group(1)))

instance_files = find_instance_files()
if not instance_files:
    raise FileNotFoundError("No instance_data_N*.pkl files found in local or Drive.")

if TARGET_N is not None:
    instance_files = [p for p in instance_files if int(pattern.search(p.name).group(1)) == TARGET_N]
    if not instance_files:
        raise ValueError(f"No instance data found for N={TARGET_N}")

print(f"Found {len(instance_files)} instance data files: {[p.name for p in instance_files]}")

# -----------------------------------------------------------------------------
# SOLVE EACH TIER
# -----------------------------------------------------------------------------
all_results = []

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print("\n" + "=" * 70)
    print(f"🔬 Solving tier N={N_true}")
    print("=" * 70)

    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)

    N_free = instance_data["N"]
    K = instance_data["K"]
    dmax = instance_data.get("D_max", 0.0)
    fixed_count = len(instance_data["fixed_indices"])

    scip_filename = f"scip_result_N{N_true}_K{K}_Dmax{int(dmax)}_M{fixed_count}.pkl"
    scip_local = LOCAL_CACHE / scip_filename
    scip_gdrive = GDRIVE_BASE / scip_filename

    scip_result = None
    if scip_local.exists():
        print(f"📂 Loading existing SCIP result from local cache: {scip_local}")
        scip_result = safe_load_pickle(scip_local)
    elif scip_gdrive.exists():
        print(f"📂 Loading existing SCIP result from GDrive: {scip_gdrive}")
        scip_result = safe_load_pickle(scip_gdrive)
        safe_save_pickle(scip_local, scip_result, verbose=False)

    if scip_result is None:
        print("🔍 Solving from scratch...")
        if not SCIP_AVAILABLE and not FALLBACK_GREEDY:
            print("❌ SCIP not available and FALLBACK_GREEDY is False. Skipping tier.")
            continue

        try:
            scip_result = solve_scip_miqp(instance_data, verbose=True)
            if scip_result["solution"] is None and FALLBACK_GREEDY:
                print("  ⚠️ SCIP returned no solution. Falling back to greedy.")
                greedy = solve_greedy_jij(instance_data, verbose=False)
                scip_result = {
                    "solution": greedy["solution"],
                    "energy": greedy["energy"],
                    "runtime": greedy["runtime"],
                    "feasible": greedy["feasible"],
                    "violation_rate": greedy["violation_rate"],
                    "budget_ok": greedy["budget_ok"],
                    "connectivity_ok": greedy["connectivity_ok"],
                    "status": "greedy_fallback"
                }
            elif scip_result["solution"] is not None:
                scip_result["status"] = "solved"
            else:
                scip_result["status"] = "failed"
        except Exception as e:
            print(f"  ❌ SCIP solver threw exception: {e}")
            if FALLBACK_GREEDY:
                print("  🔄 Falling back to greedy solver...")
                greedy = solve_greedy_jij(instance_data, verbose=False)
                scip_result = {
                    "solution": greedy["solution"],
                    "energy": greedy["energy"],
                    "runtime": greedy["runtime"],
                    "feasible": greedy["feasible"],
                    "violation_rate": greedy["violation_rate"],
                    "budget_ok": greedy["budget_ok"],
                    "connectivity_ok": greedy["connectivity_ok"],
                    "status": "greedy_fallback"
                }
            else:
                raise
        
        safe_save_pickle(scip_local, scip_result, verbose=True)
        safe_save_pickle(scip_gdrive, scip_result, verbose=True)
        print(f"✅ SCIP result saved for N={N_true}.")
        
    # --- MASTER GRID ENERGY EVALUATION ---
    master_energy_val = np.nan
    if a_master is not None and Q_master is not None and scip_result.get("solution") is not None:
        x_master = np.zeros(N_master, dtype=int)
        x_tier = np.asarray(scip_result["solution"], dtype=int)
        free_indices = instance_data.get("original_indices", [])
        fixed_indices = instance_data.get("fixed_indices", [])
        
        # 1. Map chosen free variables back to master space
        for idx, val in enumerate(x_tier):
            if val == 1 and idx < len(free_indices):
                x_master[free_indices[idx]] = 1
                
        # 2. Map fixed (existing) stations back to master space
        for f_idx in fixed_indices:
            x_master[f_idx] = 1
            
        # 3. Evaluate exact energy against master a and Q matrices
        master_energy_val = compute_energy(x_master, a_master, Q_master)
    
    # Store standard master energy back into the result payload
    scip_result["master_energy"] = master_energy_val

    energy = scip_result["energy"]
    feasible = scip_result["feasible"]
    violation = scip_result["violation_rate"]
    runtime = scip_result["runtime"]
    status = scip_result.get("status", "unknown")

    print(f"  Status: {status}")
    print(f"  Tier Energy: {energy:.6f}")
    print(f"  Master Energy: {master_energy_val:.6f}")
    print(f"  Feasible: {feasible}")
    print(f"  Runtime: {runtime:.4f}s")

    if SAVE_PLOTS or SHOW_PLOTS:
        deploy_save_path = LOCAL_CACHE / f"deployment_N{N_true}_K{K}_Dmax{int(dmax)}_M{fixed_count}.png" if SAVE_PLOTS else None
        plot_deployment(instance_data, scip_result, save_path=deploy_save_path, show=SHOW_PLOTS, dpi=PLOT_DPI)

        matrix_save_path = LOCAL_CACHE / f"miqp_matrix_N{N_true}_K{K}_Dmax{int(dmax)}_M{fixed_count}.png" if SAVE_PLOTS else None
        plot_miqp_matrix_reduced(instance_data, save_path=matrix_save_path, show=SHOW_PLOTS, dpi=PLOT_DPI,
                                 title_prefix=f"$N_{{total}}$={len(instance_data['original_coords'])} | ")

        plot_miqp_matrix_full(instance_data, save_path=matrix_save_path, show=SHOW_PLOTS, dpi=PLOT_DPI,
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
        "violation_rate": violation,
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

print("\n✅ All SCIP cells complete.")