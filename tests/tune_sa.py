#@title 🔬 CELL 6: SA TUNE v4.1 (No External Solver, Fully Inlined)
"""
================================================================================
SA TUNING WITH OPTUNA v4.1 – REAL DATA (MULTI-TIER, FULLY INLINED)
================================================================================
- All solver logic is inside the objective function.
- The best solution vector (x_best) is stored as a trial attribute.
- No separate solve_sa_jij function – simplifies and avoids re‑runs.
- Includes fixed_neighbors fix, timing breakdown, master energy, selected indices.
- Generates deployment and QUBO matrix plots for the winner.
- Saves tuned parameters (lambda, master energy, selected indices, etc.) per tier.
================================================================================
"""

import sys, os, math, time, pickle, re, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import optuna
import wandb
import openjij as oj
import jijmodeling as jm
from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_SA = {
    "TARGET_N": None,                     # None = tune all discovered tiers
    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    
    # Mathematical Convergence Config
    "WARMUP_TRIALS": 15,
    "MAX_FLOOR_HITS": 10,
    "VARIANCE_TOLERANCE": 0.05,
    "MIN_FEASIBLE_FOR_VARIANCE": 5,

    # Rank-Constrained Single Objective
    "FEASIBILITY_RANK_THRESHOLD": 20,
    "TOP_K_ENERGY": 3,
    "ENERGY_TOLERANCE_PCT": 0.5,

    "LAMBDA_LOWER": 0.0001,
    
    "USE_WANDB": True,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,
    "RUN_ID": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "FORCE_RETUNE": True,
}

TARGET_N = CONFIG_SA["TARGET_N"]
NUM_SWEEPS = CONFIG_SA["NUM_SWEEPS"]
NUM_READS = CONFIG_SA["NUM_READS"]
WARMUP_TRIALS = CONFIG_SA["WARMUP_TRIALS"]
MAX_FLOOR_HITS = CONFIG_SA["MAX_FLOOR_HITS"]
VARIANCE_TOLERANCE = CONFIG_SA["VARIANCE_TOLERANCE"]
MIN_FEASIBLE_FOR_VARIANCE = CONFIG_SA["MIN_FEASIBLE_FOR_VARIANCE"]
FEASIBILITY_RANK_THRESHOLD = CONFIG_SA["FEASIBILITY_RANK_THRESHOLD"]
TOP_K_ENERGY = CONFIG_SA["TOP_K_ENERGY"]
ENERGY_TOLERANCE_PCT = CONFIG_SA["ENERGY_TOLERANCE_PCT"]
LAMBDA_LOWER = CONFIG_SA["LAMBDA_LOWER"]
USE_WANDB = CONFIG_SA["USE_WANDB"]
WANDB_PROJECT = CONFIG_SA["WANDB_PROJECT"]
LOG_WANDB_TABLE = CONFIG_SA["LOG_WANDB_TABLE"]
RUN_ID = CONFIG_SA["RUN_ID"]
FORCE_RETUNE = CONFIG_SA["FORCE_RETUNE"]

LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
RUN_DIR = GDRIVE_BASE / f"run_{RUN_ID}"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD MASTER DATA (for master energy computation)
# -----------------------------------------------------------------------------
master_path = GDRIVE_BASE / "master_real.pkl"
if not master_path.exists():
    master_path = LOCAL_CACHE / "master_real.pkl"
with open(master_path, "rb") as f:
    master_data = pickle.load(f)

a_master = master_data["a"]                     # length 5417
Q_master_edges = master_data["Q_edges"]         # list of (i,j,val)
coords_master = master_data["coords"]
U_master = master_data["U"]
M_indices_master = master_data["M_indices"]
print(f"✅ Master data loaded: N_master={len(a_master)}, Q_edges={len(Q_master_edges)}")

def compute_energy_sparse(x: np.ndarray, a: np.ndarray, Q_edges: list) -> float:
    x = np.asarray(x, dtype=bool)
    energy = np.dot(a, x.astype(float))
    for i, j, val in Q_edges:
        if x[i] and x[j]:
            energy += val
    return float(energy)

def compute_master_energy_from_solution(x_sol, snapped_indices, fixed_indices):
    """Map local free solution to master grid and compute master energy."""
    x_master = np.zeros(len(a_master), dtype=int)
    for idx, val in enumerate(x_sol):
        if val == 1:
            master_idx = snapped_indices[idx]
            if 0 <= master_idx < len(a_master):
                x_master[master_idx] = 1
    for f_idx in fixed_indices:
        if 0 <= f_idx < len(a_master):
            x_master[f_idx] = 1
    energy = compute_energy_sparse(x_master, a_master, Q_master_edges)
    return energy

# -----------------------------------------------------------------------------
# VECTORIZED HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def compute_energy_vectorized(x: np.ndarray, a: np.ndarray, Q: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    linear = np.dot(a, x)
    quad = 0.5 * x @ Q @ x
    return float(linear + quad)

def check_feasibility_vectorized(x: np.ndarray, neigh: np.ndarray, K: int, fixed_neighbors: np.ndarray = None) -> dict:
    x_bool = np.asarray(x, dtype=bool)
    num_selected = np.sum(x_bool)
    budget_ok = (num_selected == K)

    neighbor_counts = neigh @ x_bool.astype(float)
    if fixed_neighbors is not None:
        fixed_nbrs = np.asarray(fixed_neighbors, dtype=float)
        has_connection = (neighbor_counts > 0) | (fixed_nbrs > 0)
    else:
        has_connection = (neighbor_counts > 0)
    selected_mask = x_bool
    isolated = selected_mask & ~has_connection
    connectivity_ok = not np.any(isolated)

    feasible = budget_ok and connectivity_ok
    isolated_indices = np.where(isolated)[0].tolist()

    return {
        "feasible": feasible,
        "budget_ok": budget_ok,
        "connectivity_ok": connectivity_ok,
        "num_selected": int(num_selected),
        "isolated_indices": isolated_indices,
    }

def decode_solution(solution_obj, N: int) -> np.ndarray:
    x_sol = np.zeros(N, dtype=int)
    if hasattr(solution_obj, "decision_variables_df"):
        df = solution_obj.decision_variables_df
        x_df = df[df["name"] == "x"]
        for _, row in x_df.iterrows():
            if row["value"] > 0.5:
                subs = row["subscripts"]
                idx = subs[0] if isinstance(subs, (tuple, list)) and len(subs) > 0 else int(subs)
                if 0 <= idx < N:
                    x_sol[idx] = 1
        return x_sol
    if hasattr(solution_obj, "first"):
        best_sample = solution_obj.first.sample
        for idx, val in best_sample.items():
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(round(val))
        return x_sol
    if isinstance(solution_obj, dict):
        for idx, val in solution_obj.items():
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(round(val))
        return x_sol
    raise TypeError(f"Unsupported solution type: {type(solution_obj)}")

# -----------------------------------------------------------------------------
# ADAPTER: Sparse → Dense (Q and neigh)
# -----------------------------------------------------------------------------
def adapt_sparse_to_dense(instance_data: dict) -> dict:
    N = instance_data["N"]
    Q = np.zeros((N, N), dtype=float)
    for i, j, val in instance_data.get("Q_edges", []):
        Q[i, j] = val
        Q[j, i] = val
    neigh = np.zeros((N, N), dtype=int)
    for i, nbrs in enumerate(instance_data.get("neighbors", [])):
        for j in nbrs:
            if j < N:
                neigh[i, j] = 1
    fixed_neighbors_raw = instance_data.get("fixed_neighbors", None)
    if fixed_neighbors_raw is not None:
        if isinstance(fixed_neighbors_raw, (list, np.ndarray)):
            fixed_neighbors = np.asarray(fixed_neighbors_raw, dtype=float)
        elif isinstance(fixed_neighbors_raw, dict):
            fixed_neighbors = np.zeros(N, dtype=float)
            for k, v in fixed_neighbors_raw.items():
                if int(k) < N and (isinstance(v, (list, tuple, dict)) and len(v) > 0) or (isinstance(v, (int, float)) and v > 0):
                    fixed_neighbors[int(k)] = 1.0
        else:
            fixed_neighbors = None
    else:
        fixed_neighbors = None
    instance_data["Q"] = Q
    instance_data["neigh"] = neigh
    instance_data["fixed_neighbors"] = fixed_neighbors
    return instance_data

# -----------------------------------------------------------------------------
# MATHEMATICAL CONVERGENCE ENGINE (Unchanged)
# -----------------------------------------------------------------------------
class MathematicalConvergenceEngine:
    def __init__(self, warmup, max_floor_hits, var_tol, min_feas_var, wandb_run=None):
        self.warmup = warmup
        self.max_floor_hits = max_floor_hits
        self.var_tol = var_tol
        self.min_feas_var = min_feas_var
        self.wandb_run = wandb_run
        self.global_best_feasible_energy = float('inf')
        self.best_penalty_sum = float('inf')
        self.floor_hits = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        current_e = trial.user_attrs.get("top_k_avg_energy", float('inf'))
        current_p = trial.user_attrs.get("penalty_sum", float('inf'))
        lb = trial.user_attrs.get("lambda_budget", 0.0)
        lc = trial.user_attrs.get("lambda_conn", 0.0)
        m_budget = trial.user_attrs.get("Mth_budget_dev", 1e9)
        m_isolated = trial.user_attrs.get("Mth_isolated_count", 1e9)
        is_feas = (m_budget == 0 and m_isolated == 0)
        feas_rate = trial.user_attrs.get("feasibility_rate", 0.0)
        runtime_sec = trial.user_attrs.get("runtime_sec", 0.0)
        std_dev = trial.user_attrs.get("top_k_energy_std_dev", 0.0)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        feasible_trials = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

        sigma_j = float('inf')
        sigma_str = "N/A"
        if len(feasible_trials) >= self.min_feas_var:
            feasible_trials.sort(key=lambda t: (t.user_attrs["top_k_avg_energy"], t.user_attrs["penalty_sum"]))
            top_m = feasible_trials[:self.min_feas_var]
            log_b = [math.log10(max(t.user_attrs["lambda_budget"], 1e-12)) for t in top_m]
            log_c = [math.log10(max(t.user_attrs["lambda_conn"], 1e-12)) for t in top_m]
            var_b = np.var(log_b, ddof=1) if len(log_b) > 1 else 0.0
            var_c = np.var(log_c, ddof=1) if len(log_c) > 1 else 0.0
            sigma_j = math.sqrt(var_b + var_c)
            sigma_str = f"{sigma_j:.3f}"

        status_str = "❌ INFEASIBLE                        "
        if is_feas:
            if self.global_best_feasible_energy == float('inf'):
                self.global_best_feasible_energy = current_e
                self.best_penalty_sum = current_p
                self.floor_hits = 1
                status_str = "🌟 NEW BEST FEASIBLE!             "
            else:
                rel_tol = max(1e-6, 1e-6 * abs(self.global_best_feasible_energy))
                if current_e < self.global_best_feasible_energy - rel_tol:
                    self.global_best_feasible_energy = current_e
                    self.best_penalty_sum = current_p
                    self.floor_hits = 1
                    status_str = "🌟 NEW BEST FEASIBLE!             "
                elif abs(current_e - self.global_best_feasible_energy) <= rel_tol:
                    if current_p < self.best_penalty_sum - 1e-4:
                        self.best_penalty_sum = current_p
                        self.floor_hits = 1
                        status_str = "🎯 FEASIBLE MATCH (LOWER PENALTY!)"
                    else:
                        self.floor_hits += 1
                        status_str = "✅ FEASIBLE MATCH                 "
                else:
                    status_str = "✅ FEASIBLE                       "

        trial.set_user_attr("global_best_feasible_energy", self.global_best_feasible_energy if self.global_best_feasible_energy != float('inf') else None)

        floor_str = f"{self.floor_hits:2d}/{self.max_floor_hits:<2d}"
        pen_breakdown_str = f"{current_p:7.2f} (λb={lb:.2f}, λc={lc:.2f})"
        qb_time = trial.user_attrs.get("qubo_build_time", 0.0)
        sa_time = trial.user_attrs.get("sa_sampling_time", 0.0)
        pp_time = trial.user_attrs.get("postprocess_time", 0.0)
        total_time = trial.user_attrs.get("runtime_sec", 0.0)
        sel_indices = trial.user_attrs.get("selected_free_indices", "")

        if is_feas:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | Top-3 Avg: {current_e:10.4f} (std:{std_dev:6.4f}) | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | "
                  f"Timing: QUBO={qb_time:5.2f}s SA={sa_time:5.2f}s Post={pp_time:5.2f}s Total={total_time:5.2f}s | "
                  f"Sel: {sel_indices}")
        else:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | M-th Viol: Budg={m_budget:2.0f}, Isol={m_isolated:2.0f} | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | "
                  f"Timing: QUBO={qb_time:5.2f}s SA={sa_time:5.2f}s Post={pp_time:5.2f}s Total={total_time:5.2f}s | "
                  f"Sel: {sel_indices}")

        if self.wandb_run:
            self.wandb_run.log({
                "trial": trial.number,
                "best_single_energy": current_e if current_e != float('inf') else None,
                "global_best_feasible_energy": self.global_best_feasible_energy if self.global_best_feasible_energy != float('inf') else None,
                "feasibility_rate": feas_rate,
                "lambda_budget": lb,
                "lambda_conn": lc,
                "penalty_sum": current_p if current_p != float('inf') else None,
                "floor_hits": self.floor_hits,
                "sigma_j": sigma_j if sigma_j != float('inf') else None,
                "runtime_sec": total_time,
                "qubo_build_time": qb_time,
                "sa_sampling_time": sa_time,
                "postprocess_time": pp_time,
                "Mth_budget_dev": m_budget,
                "Mth_isolated_count": m_isolated,
                "master_energy": trial.user_attrs.get("master_energy", np.nan),
            })

        if trial.number >= self.warmup:
            if self.floor_hits >= self.max_floor_hits:
                print(f"\n🛑 [Early Stopping: Strategy B] Internal Top-K energy floor ({self.global_best_feasible_energy:.4f}) hit {self.max_floor_hits} times. (Best Pen: {self.best_penalty_sum:.2f})")
                study.stop()
            elif sigma_j <= self.var_tol:
                print(f"\n🛑 [Early Stopping: Strategy A] Parameter variance collapsed (σ = {sigma_j:.4f} ≤ {self.var_tol}). Search exhausted.")
                study.stop()

# -----------------------------------------------------------------------------
# OBJECTIVE FUNCTION (fully inlined, stores solution)
# -----------------------------------------------------------------------------
def make_objective(precompiled_instance, N, K, a, Q, neigh, fixed_neighbors, snapped_indices, fixed_indices_orig, LAMBDA_LOWER, LAMBDA_UPPER):
    def objective(trial):
        start_time = time.perf_counter()
        lambda_budget = trial.suggest_float("lambda_budget", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        penalty_sum = lambda_budget + lambda_conn
        trial.set_user_attr("lambda_budget", lambda_budget)
        trial.set_user_attr("lambda_conn", lambda_conn)
        trial.set_user_attr("penalty_sum", penalty_sum)

        # 1. QUBO build time
        t0 = time.perf_counter()
        penalty_weights = get_penalty_weights(precompiled_instance, lambda_budget, lambda_conn)
        qubo_dict, _ = precompiled_instance.to_qubo(penalty_weights=penalty_weights)
        t1 = time.perf_counter()
        qubo_build_time = t1 - t0

        # 2. SA sampling time
        sampler = oj.SASampler()
        response = sampler.sample_qubo(
            qubo_dict,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            sparse=True,
        )
        t2 = time.perf_counter()
        sa_sampling_time = t2 - t1

        # 3. Post-processing (decode, compute energy, violation, master energy)
        all_samples = []
        for idx in range(response.record.shape[0]):
            sample_arr = response.record['sample'][idx]
            x_sample = np.zeros(N, dtype=int)
            for var_idx, val in zip(response.indices, sample_arr):
                if var_idx < N:
                    x_sample[var_idx] = int(round(val))
            e_sample = compute_energy_vectorized(x_sample, a, Q)
            f_sample = check_feasibility_vectorized(x_sample, neigh, K, fixed_neighbors)
            v_sample = float(not f_sample["budget_ok"]) * 0.5 + float(not f_sample["connectivity_ok"]) * 0.5
            all_samples.append({
                "solution": x_sample.copy(),
                "energy": e_sample,
                "violation_rate": v_sample,
                "feasible": f_sample["feasible"],
                "budget_ok": f_sample["budget_ok"],
                "connectivity_ok": f_sample["connectivity_ok"],
                "num_selected": f_sample["num_selected"],
            })

        # Rank by violation then energy
        all_samples.sort(key=lambda s: (s["violation_rate"], s["energy"]))
        m_idx = min(FEASIBILITY_RANK_THRESHOLD - 1, len(all_samples) - 1)
        m_sample = all_samples[m_idx]
        m_budget = float(not m_sample["budget_ok"])
        m_isolated = float(not m_sample["connectivity_ok"])
        trial.set_user_attr("Mth_budget_dev", m_budget)
        trial.set_user_attr("Mth_isolated_count", m_isolated)

        feasible_samples = [s for s in all_samples if s["violation_rate"] == 0]
        feas_rate = len(feasible_samples) / len(all_samples) if all_samples else 0.0
        trial.set_user_attr("feasibility_rate", feas_rate)

        # Top-K energy
        if feasible_samples:
            feasible_samples.sort(key=lambda s: s["energy"])
            top_k_samples = feasible_samples[:min(TOP_K_ENERGY, len(feasible_samples))]
        else:
            top_k_samples = all_samples[:min(TOP_K_ENERGY, len(all_samples))]
        top_k_energies = [s["energy"] for s in top_k_samples if not np.isnan(s["energy"])]
        if top_k_energies:
            top_k_avg = float(np.mean(top_k_energies))
            top_k_std = float(np.std(top_k_energies)) if len(top_k_energies) > 1 else 0.0
        else:
            top_k_avg = 1e9
            top_k_std = 0.0
        trial.set_user_attr("top_k_avg_energy", top_k_avg)
        trial.set_user_attr("top_k_energy_std_dev", top_k_std)

        # Best solution (rank 0) – store for later plotting
        best_sample = all_samples[0]
        x_best = best_sample["solution"]
        trial.set_user_attr("winner_solution", x_best.tolist())   # store solution vector

        # Master energy
        master_energy = compute_master_energy_from_solution(x_best, snapped_indices, fixed_indices_orig)
        trial.set_user_attr("master_energy", master_energy)

        # Selected indices (free)
        sel_free = np.where(x_best == 1)[0].tolist()
        trial.set_user_attr("selected_free_indices", str(sel_free))
        sel_master = [snapped_indices[i] for i in sel_free if i < len(snapped_indices)]
        trial.set_user_attr("selected_master_indices", str(sel_master))

        # Timing breakdown
        trial.set_user_attr("qubo_build_time", qubo_build_time)
        trial.set_user_attr("sa_sampling_time", sa_sampling_time)
        trial.set_user_attr("postprocess_time", time.perf_counter() - t2)
        trial.set_user_attr("runtime_sec", time.perf_counter() - start_time)

        return top_k_avg
    return objective

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (Deployment and QUBO Matrix)
# -----------------------------------------------------------------------------
def plot_deployment(instance_data, solution, save_path=None, show=True, dpi=150):
    coords_full = np.asarray(instance_data["original_coords"])
    D_MAX = instance_data["D_max"]
    fixed_indices = list(instance_data.get("fixed_indices", []))
    free_indices = list(instance_data.get("original_indices", []))
    N_free = len(free_indices)
    N_total = len(coords_full)

    x_sol = np.asarray(solution, dtype=int)
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
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                   c='white', s=40, alpha=0.9, edgecolor='#1e293b', linewidth=0.8,
                   label='Candidates', zorder=2)
    if selected_m:
        ax.scatter(coords_full[selected_m, 0], coords_full[selected_m, 1],
                   c='blue', s=110, marker='s', edgecolor='black', linewidth=1.2, label=f'Existing ({len(selected_m)})', zorder=3)
    if selected_new:
        ax.scatter(coords_full[selected_new, 0], coords_full[selected_new, 1],
                   c='red', s=110, marker='o', edgecolor='black', linewidth=1.2, label=f'New ({len(selected_new)})', zorder=4)

    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    ax.set_title(f'Deployment (SA Tuned) – N_total={N_total}, N_free={N_free}', fontsize=13, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, markeredgecolor='black', label=f'New ({len(selected_new)})'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=9, markeredgecolor='black', label=f'Existing ({len(selected_m)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=8, markeredgecolor='#1e293b', label=f'Candidates ({len(free_indices)})'),
        plt.Line2D([0], [0], color='#475569', linewidth=1.2, linestyle='--', label=f'Link (≤ {D_MAX:.1f}m)')
    ]
    ax_info.legend(handles=handles, loc='upper left', frameon=True, fontsize=10, title="Legend", title_fontsize=11)

    selected_new_str = ', '.join(map(str, selected_new[:10])) + ('...' if len(selected_new) > 10 else '') if selected_new else 'None'
    selected_m_str = ', '.join(map(str, selected_m[:10])) + ('...' if len(selected_m) > 10 else '') if selected_m else 'None'
    info_text = (
        f"SA TUNED SOLUTION\n"
        f"---------------------------\n"
        f"Feasible: {check_feasibility_vectorized(x_sol, instance_data['neigh'], instance_data['K'], instance_data['fixed_neighbors'])['feasible']}\n"
        f"Energy: {compute_energy_vectorized(x_sol, instance_data['a'], instance_data['Q']):.4f}\n"
        f"Master Energy: {compute_master_energy_from_solution(x_sol, instance_data['snapped_indices'], instance_data['fixed_indices']):.4f}\n"
        f"New: {len(selected_new)}, Existing: {len(selected_m)}\n"
        f"Selected new indices: {selected_new_str}\n"
        f"Selected existing: {selected_m_str}"
    )
    ax_info.text(0.0, 0.62, info_text, transform=ax_info.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.95, edgecolor='#ced4da'))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_qubo_matrix(precompiled_instance, penalty_weights, N, save_path=None, show=True, dpi=150, title_prefix=""):
    qubo_dict, offset = precompiled_instance.to_qubo(penalty_weights=penalty_weights)
    Q_mat = np.zeros((N, N), dtype=float)
    for (i, j), val in qubo_dict.items():
        if isinstance(i, tuple): i = i[0]
        if isinstance(j, tuple): j = j[0]
        Q_mat[i, j] = val
    Q_mat = Q_mat + Q_mat.T - np.diag(np.diag(Q_mat))
    max_abs = np.max(np.abs(Q_mat)) if np.max(np.abs(Q_mat)) > 0 else 1.0
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Q_mat, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(f"{title_prefix}QUBO Matrix (with penalties) N={N}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Variable Index")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient Value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# MAIN LOOP OVER ALL TIERS
# -----------------------------------------------------------------------------
pattern = re.compile(r"instance_data_N(\d+)\.pkl")
instance_files = [p for p in GDRIVE_BASE.glob("instance_data_N*.pkl") if pattern.match(p.name)]
if not instance_files:
    instance_files = [p for p in LOCAL_CACHE.glob("instance_data_N*.pkl") if pattern.match(p.name)]

if TARGET_N is not None:
    instance_files = [p for p in instance_files if int(pattern.search(p.name).group(1)) == TARGET_N]

if not instance_files:
    raise FileNotFoundError("No instance_data_N*.pkl files found.")

instance_files.sort(key=lambda p: int(pattern.search(p.name).group(1)))

tuned_params_all = {}

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print(f"\n{'='*115}\n🔬 Tuning SA for tier N={N_true} (All Tiers, Vectorized)\n{'='*115}")

    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)
    instance_data = adapt_sparse_to_dense(instance_data)

    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"], dtype=float)
    Q = np.asarray(instance_data["Q"], dtype=float)
    neigh = np.asarray(instance_data["neigh"], dtype=int)
    fixed_neighbors = instance_data.get("fixed_neighbors", None)
    snapped_indices = instance_data.get("snapped_indices", None)
    fixed_indices_orig = instance_data.get("fixed_indices", [])

    # Compute qsum
    qsum = np.sum(np.abs(a)) + np.sum(np.abs(np.triu(Q, 1)))
    print(f"📏 qsum for N={N_true}: {qsum:.4f}")

    # Precompile instance WITH fixed_neighbors
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh", "fixed_neighbors"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    precompiled_instance = compile_instance(model, filtered_data)

    LAMBDA_UPPER = qsum  # as requested, no multiplier

    study_name = f"sa_tuning_N{N_true}_{RUN_ID}"
    local_db = LOCAL_CACHE / f"{study_name}.db"

    if FORCE_RETUNE:
        try:
            optuna.delete_study(study_name=study_name, storage=f"sqlite:///{local_db}")
        except KeyError:
            pass

    wandb_run = None
    if USE_WANDB:
        wandb_run = wandb.init(project=WANDB_PROJECT, config=CONFIG_SA, name=f"{study_name}", reinit=True)

    convergence_cb = MathematicalConvergenceEngine(
        warmup=WARMUP_TRIALS,
        max_floor_hits=MAX_FLOOR_HITS,
        var_tol=VARIANCE_TOLERANCE,
        min_feas_var=MIN_FEASIBLE_FOR_VARIANCE,
        wandb_run=wandb_run,
    )

    def constraint_func(trial):
        return [trial.user_attrs.get("Mth_budget_dev", 1e9), trial.user_attrs.get("Mth_isolated_count", 1e9)]

    objective = make_objective(
        precompiled_instance=precompiled_instance,
        N=N,
        K=K,
        a=a,
        Q=Q,
        neigh=neigh,
        fixed_neighbors=fixed_neighbors,
        snapped_indices=snapped_indices,
        fixed_indices_orig=fixed_indices_orig,
        LAMBDA_LOWER=LAMBDA_LOWER,
        LAMBDA_UPPER=LAMBDA_UPPER,
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=WARMUP_TRIALS, constraints_func=constraint_func)
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{local_db}", sampler=sampler, direction="minimize")

    study.optimize(objective, callbacks=[convergence_cb])

    # Post-study: select winner (best feasible trial)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

    if not feasible:
        print(f"⚠️ No feasible trials found for N={N_true}. Skipping this tier.")
        if wandb_run:
            wandb_run.finish()
        continue

    feasible.sort(key=lambda t: t.user_attrs["top_k_avg_energy"])
    best_e = feasible[0].user_attrs["top_k_avg_energy"]
    cutoff_e = best_e + abs(best_e) * (ENERGY_TOLERANCE_PCT / 100.0) if best_e < 0 else best_e * (1.0 + ENERGY_TOLERANCE_PCT / 100.0)
    near_best = [t for t in feasible if t.user_attrs["top_k_avg_energy"] <= cutoff_e]
    near_best.sort(key=lambda t: (t.user_attrs["top_k_avg_energy"], t.user_attrs["penalty_sum"]))
    winner = near_best[0]

    # Extract winner data
    winner_lb = winner.user_attrs["lambda_budget"]
    winner_lc = winner.user_attrs["lambda_conn"]
    winner_penalty_sum = winner.user_attrs["penalty_sum"]
    winner_energy = winner.user_attrs["top_k_avg_energy"]
    winner_master_energy = winner.user_attrs.get("master_energy", np.nan)
    winner_feas_rate = winner.user_attrs["feasibility_rate"]
    winner_sel_free = winner.user_attrs.get("selected_free_indices", "")
    winner_sel_master = winner.user_attrs.get("selected_master_indices", "")
    winner_solution = winner.user_attrs.get("winner_solution", None)

    # Print tie-break table
    print(f"\n{'='*115}")
    print(f"🏆 TIE-BREAK BREAKDOWN (Feasible Trials within {ENERGY_TOLERANCE_PCT}% of Best Energy: {best_e:.4f})")
    print(f"{'='*115}")
    print(f"{'Trial':<7} | {'Top-3 Energy':<14} | {'Std Dev':<8} | {'Penalty Sum':<12} | {'Feas %':<8} | {'λ_budget':<10} | {'λ_conn':<10} | {'Master Energy':<14} | {'Sel Free Idx'}")
    print(f"{'-'*115}")
    for t in near_best:
        num = t.number
        e_val = t.user_attrs["top_k_avg_energy"]
        std_v = t.user_attrs["top_k_energy_std_dev"]
        p_sum = t.user_attrs["penalty_sum"]
        f_pct = t.user_attrs["feasibility_rate"] * 100.0
        lb_v = t.user_attrs["lambda_budget"]
        lc_v = t.user_attrs["lambda_conn"]
        m_e = t.user_attrs.get("master_energy", np.nan)
        sel_free = t.user_attrs.get("selected_free_indices", "")
        notes = "🥇 WINNER" if t.number == winner.number else ""
        print(f"#{num:<6} | {e_val:<14.4f} | {std_v:<8.4f} | {p_sum:<12.2f} | {f_pct:<7.2f}% | {lb_v:<10.2f} | {lc_v:<10.2f} | {m_e:<14.4f} | {sel_free:<20} | {notes}")
    print(f"{'='*115}\n")

    # W&B table logging
    if LOG_WANDB_TABLE and wandb_run:
        wb_table = wandb.Table(columns=["Trial", "Top3_Energy", "Std_Dev", "Penalty_Sum", "Feas_Pct", "Lambda_Budget", "Lambda_Conn", "Master_Energy", "Selected_Free_Idx", "Notes"])
        for t in near_best:
            num = t.number
            e_val = t.user_attrs["top_k_avg_energy"]
            std_v = t.user_attrs["top_k_energy_std_dev"]
            p_sum = t.user_attrs["penalty_sum"]
            f_pct = t.user_attrs["feasibility_rate"] * 100.0
            lb_v = t.user_attrs["lambda_budget"]
            lc_v = t.user_attrs["lambda_conn"]
            m_e = t.user_attrs.get("master_energy", np.nan)
            sel_free = t.user_attrs.get("selected_free_indices", "")
            notes = "🥇 WINNER" if t.number == winner.number else ""
            wb_table.add_data(f"#{num}", e_val, std_v, p_sum, f_pct, lb_v, lc_v, m_e, sel_free, notes)
        wandb_run.log({"tie_break_candidates": wb_table})
        wandb_run.summary["global_best_feasible_energy"] = best_e
        wandb_run.summary["winning_penalty_sum"] = winner_penalty_sum

    # Generate deployment and QUBO matrix plots for the winner
    if winner_solution is not None:
        x_winner = np.asarray(winner_solution, dtype=int)
        # Deployment plot
        deploy_save_path = RUN_DIR / f"deployment_SA_N{N_true}_K{K}.png"
        plot_deployment(instance_data, x_winner, save_path=deploy_save_path, show=False, dpi=150)

        # QUBO matrix plot (use winner's lambdas)
        winner_penalty_weights = get_penalty_weights(precompiled_instance, winner_lb, winner_lc)
        qubo_save_path = RUN_DIR / f"qubo_matrix_SA_N{N_true}_K{K}.png"
        plot_qubo_matrix(precompiled_instance, winner_penalty_weights, N, save_path=qubo_save_path, show=False, dpi=150, title_prefix=f"SA Tuned N={N_true} | ")
    else:
        print(f"⚠️ No solution vector stored for winner of N={N_true}. Skipping plots.")

    # Save tuned parameters to JSON
    tuned_params = {
        "N": N_true,
        "lambda_budget": float(winner_lb),
        "lambda_conn": float(winner_lc),
        "best_energy": float(winner_energy),
        "best_master_energy": float(winner_master_energy),
        "feasibility_rate": float(winner_feas_rate),
        "penalty_sum": float(winner_penalty_sum),
        "selected_free_indices": winner_sel_free,
        "selected_master_indices": winner_sel_master,
        "qsum": float(qsum),
        "winner_trial_number": int(winner.number),
        "num_trials": len(completed),
        "num_feasible_trials": len(feasible),
        "early_stopped": study.should_stop,
    }
    tuned_params_all[N_true] = tuned_params

    json_path = GDRIVE_BASE / f"tuned_sa_N{N_true}.json"
    with open(json_path, "w") as f:
        json.dump(tuned_params, f, indent=2)
    (LOCAL_CACHE / f"tuned_sa_N{N_true}.json").write_text(json.dumps(tuned_params, indent=2))
    print(f"✅ Tuned parameters saved to {json_path}")

    if wandb_run:
        wandb_run.finish()

# -----------------------------------------------------------------------------
# SUMMARY: All tuned parameters
# -----------------------------------------------------------------------------
print("\n" + "=" * 115)
print("📊 TUNING SUMMARY – ALL TIERS")
print("=" * 115)
if tuned_params_all:
    print(f"{'N':<8} | {'λ_budget':<12} | {'λ_conn':<12} | {'Best Energy':<14} | {'Master Energy':<14} | {'Feas %':<8} | {'qsum':<12} | {'Feasible Trials'}")
    print("-" * 115)
    for N, p in sorted(tuned_params_all.items()):
        print(f"{N:<8} | {p['lambda_budget']:<12.4f} | {p['lambda_conn']:<12.4f} | {p['best_energy']:<14.4f} | {p['best_master_energy']:<14.4f} | {p['feasibility_rate']*100:<7.2f}% | {p['qsum']:<12.4f} | {p['num_feasible_trials']}/{p['num_trials']}")
else:
    print("⚠️ No tuned parameters saved (no feasible trials for any tier).")
print("=" * 115)
print("✅ SA tuning complete for all tiers.")