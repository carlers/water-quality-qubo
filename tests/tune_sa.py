#@title 🔬 CELL 6: SA TUNE v3.0 (Fully Inlined, Vectorized, All Tiers)
"""
================================================================================
SA TUNING WITH OPTUNA v3.0 – REAL DATA (MULTI-TIER, VECTORIZED)
================================================================================
- Fully self-contained: all solver logic inlined (no external jij_solvers import).
- Sparse → dense adapter (Q_edges → Q, neighbors → neigh) for vectorized math.
- Pre-compiled JijModeling instance per tier (no rebuild per trial).
- Vectorized compute_energy and check_feasibility (no Python loops).
- Tunes ALL discovered N tiers (20, 50, 100, 200, 500, 1000).
- Preserves MathematicalConvergenceEngine (dual‑trigger) & W&B logging.
- Saves tuned parameters (lambda_budget, lambda_conn) as JSON per tier.
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
# SECTION 1: VECTORIZED SOLVER HELPERS (Inlined)
# -----------------------------------------------------------------------------

def compute_energy_vectorized(x: np.ndarray, a: np.ndarray, Q: np.ndarray) -> float:
    """
    Vectorized MIQP energy: a·x + 0.5 * xᵀ Q x
    Assumes Q is symmetric (we will symmetrize during adapter).
    """
    x = np.asarray(x, dtype=float)
    linear = np.dot(a, x)
    quad = 0.5 * x @ Q @ x
    return float(linear + quad)


def check_feasibility_vectorized(x: np.ndarray, neigh: np.ndarray, K: int, fixed_neighbors: np.ndarray = None) -> dict:
    """
    Vectorized feasibility check.
    - Budget: sum(x) == K
    - Connectivity: for every selected i, (neigh[i] @ x > 0) OR fixed_neighbors[i] == 1
    """
    x_bool = np.asarray(x, dtype=bool)
    num_selected = np.sum(x_bool)
    budget_ok = (num_selected == K)

    # Free-neighbor connectivity: neigh @ x gives count of selected neighbors per node
    neighbor_counts = neigh @ x_bool.astype(float)

    # If fixed_neighbors is provided, treat them as already connected
    if fixed_neighbors is not None:
        fixed_nbrs = np.asarray(fixed_neighbors, dtype=float)
        has_connection = (neighbor_counts > 0) | (fixed_nbrs > 0)
    else:
        has_connection = (neighbor_counts > 0)

    # Check only selected nodes
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
    """
    Robust binary decoder for OMMX / OpenJij outputs.
    """
    x_sol = np.zeros(N, dtype=int)

    # OMMX Solution with DataFrame
    if hasattr(solution_obj, "decision_variables_df"):
        df = solution_obj.decision_variables_df
        x_df = df[df["name"] == "x"]
        for _, row in x_df.iterrows():
            if row["value"] > 0.5:  # catch floating point
                subs = row["subscripts"]
                idx = subs[0] if isinstance(subs, (tuple, list)) and len(subs) > 0 else int(subs)
                if 0 <= idx < N:
                    x_sol[idx] = 1
        return x_sol

    # OpenJij Response with .first.sample
    if hasattr(solution_obj, "first"):
        best_sample = solution_obj.first.sample
        for idx, val in best_sample.items():
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(round(val))
        return x_sol

    # Direct dict fallback
    if isinstance(solution_obj, dict):
        for idx, val in solution_obj.items():
            if isinstance(idx, tuple):
                idx = idx[0]
            if 0 <= idx < N:
                x_sol[idx] = int(round(val))
        return x_sol

    raise TypeError(f"Unsupported solution type: {type(solution_obj)}")


# -----------------------------------------------------------------------------
# SECTION 2: INLINED SA & SQA SOLVERS (Vectorized, Pre-compiled Instance)
# -----------------------------------------------------------------------------

def solve_sa_jij_inlined(
    precompiled_instance,   # JijModeling Instance (already built)
    penalty_weights: dict,
    N: int,
    a: np.ndarray,
    Q: np.ndarray,
    neigh: np.ndarray,
    fixed_neighbors: np.ndarray = None,
    num_reads: int = 1024,
    num_sweeps: int = 1000,
    return_all: bool = False,
    verbose: bool = False,
) -> dict:
    """SA solver using precompiled instance – vectorized energy & feasibility."""
    start = time.perf_counter()
    try:
        # Build QUBO from precompiled instance
        qubo_dict, _ = precompiled_instance.to_qubo(penalty_weights=penalty_weights)
        sampler = oj.SASampler()
        response = sampler.sample_qubo(
            qubo_dict,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            sparse=True,
        )
        runtime = time.perf_counter() - start

        # Decode best solution
        x_sol = decode_solution(response, N)

        # Vectorized energy & feasibility
        energy = compute_energy_vectorized(x_sol, a, Q)
        feas_detail = check_feasibility_vectorized(x_sol, neigh, K=instance_data["K"], fixed_neighbors=fixed_neighbors)
        feasible = feas_detail["feasible"]
        violation_rate = float(not feas_detail["budget_ok"]) * 0.5 + float(not feas_detail["connectivity_ok"]) * 0.5

        # All samples if requested
        all_samples = []
        if return_all and hasattr(response, "record"):
            for idx in range(response.record.shape[0]):
                sample_arr = response.record['sample'][idx]
                x_sample = np.zeros(N, dtype=int)
                for var_idx, val in zip(response.indices, sample_arr):
                    if var_idx < N:
                        x_sample[var_idx] = int(round(val))
                e_sample = compute_energy_vectorized(x_sample, a, Q)
                f_sample = check_feasibility_vectorized(x_sample, neigh, K=instance_data["K"], fixed_neighbors=fixed_neighbors)
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

        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "violation_rate": violation_rate,
            "status": "feasible" if feasible else "infeasible",
            "all_samples": all_samples if return_all else None,
            "budget_ok": feas_detail["budget_ok"],
            "connectivity_ok": feas_detail["connectivity_ok"],
            "num_selected": feas_detail["num_selected"],
            "isolated_indices": feas_detail["isolated_indices"],
        }
    except Exception as e:
        runtime = time.perf_counter() - start
        if verbose:
            print(f"    SA error: {e}")
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": runtime,
            "feasible": False,
            "violation_rate": 1.0,
            "status": f"error: {e}",
            "all_samples": None,
            "budget_ok": False,
            "connectivity_ok": False,
            "num_selected": 0,
            "isolated_indices": [],
        }


def solve_sqa_jij_inlined(
    precompiled_instance,
    penalty_weights: dict,
    N: int,
    a: np.ndarray,
    Q: np.ndarray,
    neigh: np.ndarray,
    fixed_neighbors: np.ndarray = None,
    num_reads: int = 1024,
    num_sweeps: int = 1000,
    trotter: int = 16,
    return_all: bool = False,
    verbose: bool = False,
) -> dict:
    """SQA solver using precompiled instance – vectorized energy & feasibility."""
    start = time.perf_counter()
    try:
        qubo_dict, _ = precompiled_instance.to_qubo(penalty_weights=penalty_weights)
        sampler = oj.SQASampler()
        response = sampler.sample_qubo(
            qubo_dict,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            trotter=trotter,
            sparse=True,
        )
        runtime = time.perf_counter() - start

        x_sol = decode_solution(response, N)
        energy = compute_energy_vectorized(x_sol, a, Q)
        feas_detail = check_feasibility_vectorized(x_sol, neigh, K=instance_data["K"], fixed_neighbors=fixed_neighbors)
        feasible = feas_detail["feasible"]
        violation_rate = float(not feas_detail["budget_ok"]) * 0.5 + float(not feas_detail["connectivity_ok"]) * 0.5

        all_samples = []
        if return_all and hasattr(response, "record"):
            for idx in range(response.record.shape[0]):
                sample_arr = response.record['sample'][idx]
                x_sample = np.zeros(N, dtype=int)
                for var_idx, val in zip(response.indices, sample_arr):
                    if var_idx < N:
                        x_sample[var_idx] = int(round(val))
                e_sample = compute_energy_vectorized(x_sample, a, Q)
                f_sample = check_feasibility_vectorized(x_sample, neigh, K=instance_data["K"], fixed_neighbors=fixed_neighbors)
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

        return {
            "solution": x_sol,
            "energy": energy,
            "runtime": runtime,
            "feasible": feasible,
            "violation_rate": violation_rate,
            "status": "feasible" if feasible else "infeasible",
            "all_samples": all_samples if return_all else None,
            "budget_ok": feas_detail["budget_ok"],
            "connectivity_ok": feas_detail["connectivity_ok"],
            "num_selected": feas_detail["num_selected"],
            "isolated_indices": feas_detail["isolated_indices"],
            "trotter_used": trotter,
        }
    except Exception as e:
        runtime = time.perf_counter() - start
        if verbose:
            print(f"    SQA error: {e}")
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": runtime,
            "feasible": False,
            "violation_rate": 1.0,
            "status": f"error: {e}",
            "all_samples": None,
            "budget_ok": False,
            "connectivity_ok": False,
            "num_selected": 0,
            "isolated_indices": [],
            "trotter_used": trotter,
        }


# -----------------------------------------------------------------------------
# SECTION 3: SPARSE → DENSE ADAPTER (Cell 4 format → vectorized format)
# -----------------------------------------------------------------------------

def adapt_sparse_to_dense(instance_data: dict) -> dict:
    """
    Convert Cell 4's sparse Q_edges + neighbors list-of-lists
    into dense Q (N×N) and neigh (N×N) for vectorized operations.
    Also builds fixed_neighbors as a 1D boolean array.
    """
    N = instance_data["N"]

    # Dense Q (symmetric)
    Q = np.zeros((N, N), dtype=float)
    for i, j, val in instance_data.get("Q_edges", []):
        Q[i, j] = val
        Q[j, i] = val  # Symmetrize

    # Dense neigh (binary adjacency)
    neigh = np.zeros((N, N), dtype=int)
    for i, nbrs in enumerate(instance_data.get("neighbors", [])):
        for j in nbrs:
            if j < N:
                neigh[i, j] = 1

    # fixed_neighbors as 1D boolean array
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

    # Store dense arrays back into instance_data
    instance_data["Q"] = Q
    instance_data["neigh"] = neigh
    instance_data["fixed_neighbors"] = fixed_neighbors
    return instance_data


# -----------------------------------------------------------------------------
# SECTION 4: MATHEMATICAL CONVERGENCE ENGINE (Unchanged)
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

        # Parameter space variance
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

        # Floor Saturation & Status
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

        # Terminal HUD
        floor_str = f"{self.floor_hits:2d}/{self.max_floor_hits:<2d}"
        pen_breakdown_str = f"{current_p:7.2f} (λb={lb:.2f}, λc={lc:.2f})"
        if is_feas:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | Top-3 Avg: {current_e:10.4f} (std:{std_dev:6.4f}) | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | {runtime_sec:5.2f}s")
        else:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | M-th Viol: Budg={m_budget:2.0f}, Isol={m_isolated:2.0f} | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | {runtime_sec:5.2f}s")

        # W&B logging
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
                "runtime_sec": runtime_sec,
                "Mth_budget_dev": m_budget,
                "Mth_isolated_count": m_isolated,
            })

        # Early stopping triggers
        if trial.number >= self.warmup:
            if self.floor_hits >= self.max_floor_hits:
                print(f"\n🛑 [Early Stopping: Strategy B] Internal Top-K energy floor ({self.global_best_feasible_energy:.4f}) hit {self.max_floor_hits} times. (Best Pen: {self.best_penalty_sum:.2f})")
                study.stop()
            elif sigma_j <= self.var_tol:
                print(f"\n🛑 [Early Stopping: Strategy A] Parameter variance collapsed (σ = {sigma_j:.4f} ≤ {self.var_tol}). Search exhausted.")
                study.stop()


# -----------------------------------------------------------------------------
# SECTION 5: OBJECTIVE FACTORY (Uses Pre-compiled Instance)
# -----------------------------------------------------------------------------

def make_objective(precompiled_instance, N, K, a, Q, neigh, fixed_neighbors, LAMBDA_LOWER, LAMBDA_UPPER):
    """Objective for Optuna – uses precompiled instance and vectorized helpers."""
    def objective(trial):
        start_time = time.time()
        lambda_budget = trial.suggest_float("lambda_budget", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        penalty_sum = lambda_budget + lambda_conn
        trial.set_user_attr("lambda_budget", lambda_budget)
        trial.set_user_attr("lambda_conn", lambda_conn)
        trial.set_user_attr("penalty_sum", penalty_sum)

        penalty_weights = get_penalty_weights(precompiled_instance, lambda_budget, lambda_conn)
        result = solve_sa_jij_inlined(
            precompiled_instance=precompiled_instance,
            penalty_weights=penalty_weights,
            N=N,
            a=a,
            Q=Q,
            neigh=neigh,
            fixed_neighbors=fixed_neighbors,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            return_all=True,
            verbose=False,
        )
        trial.set_user_attr("runtime_sec", time.time() - start_time)

        all_samples = result.get("all_samples", [])
        if not all_samples:
            trial.set_user_attr("Mth_budget_dev", 1e9)
            trial.set_user_attr("Mth_isolated_count", 1e9)
            trial.set_user_attr("feasibility_rate", 0.0)
            return 1e9

        # Compute M-th violation (ranked by violation then energy)
        all_samples.sort(key=lambda s: (s["violation_rate"], s["energy"]))
        m_idx = min(FEASIBILITY_RANK_THRESHOLD - 1, len(all_samples) - 1)
        m_sample = all_samples[m_idx]
        m_budget = float(not m_sample["budget_ok"])  # 0 or 1
        m_isolated = float(not m_sample["connectivity_ok"])
        trial.set_user_attr("Mth_budget_dev", m_budget)
        trial.set_user_attr("Mth_isolated_count", m_isolated)

        # Feasibility rate
        feasible_samples = [s for s in all_samples if s["violation_rate"] == 0]
        feas_rate = len(feasible_samples) / len(all_samples) if all_samples else 0.0
        trial.set_user_attr("feasibility_rate", feas_rate)

        # Top-K energy (only feasible, else all samples)
        if feasible_samples:
            feasible_samples.sort(key=lambda s: s["energy"])
            top_k_samples = feasible_samples[:min(TOP_K_ENERGY, len(feasible_samples))]
        else:
            all_samples.sort(key=lambda s: s["energy"])
            top_k_samples = all_samples[:min(TOP_K_ENERGY, len(all_samples))]

        top_k_energies = [s["energy"] for s in top_k_samples if not np.isnan(s["energy"])]
        if not top_k_energies:
            top_k_avg = 1e9
            top_k_std = 0.0
        else:
            top_k_avg = float(np.mean(top_k_energies))
            top_k_std = float(np.std(top_k_energies)) if len(top_k_energies) > 1 else 0.0

        trial.set_user_attr("top_k_avg_energy", top_k_avg)
        trial.set_user_attr("top_k_energy_std_dev", top_k_std)

        return top_k_avg

    return objective


# -----------------------------------------------------------------------------
# SECTION 6: MAIN LOOP OVER ALL TIERS
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

# We'll store tuned params for all tiers in a dict to export later
tuned_params_all = {}

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print(f"\n{'='*115}\n🔬 Tuning SA for tier N={N_true} (All Tiers, Vectorized)\n{'='*115}")

    # 1. Load and adapt to dense
    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)
    instance_data = adapt_sparse_to_dense(instance_data)

    N = instance_data["N"]
    K = instance_data["K"]
    a = np.asarray(instance_data["a"], dtype=float)
    Q = np.asarray(instance_data["Q"], dtype=float)
    neigh = np.asarray(instance_data["neigh"], dtype=int)
    fixed_neighbors = instance_data.get("fixed_neighbors", None)

    # 2. Pre-compile JijModeling Instance ONCE per tier
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    precompiled_instance = compile_instance(model, filtered_data)

    # 3. Compute LAMBDA_UPPER from qsum
    qsum = max(np.sum(np.abs(a)) + np.sum(np.abs(np.triu(Q, 1))), 1.0)
    LAMBDA_UPPER = qsum

    # 4. Optuna study setup
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

    # 5. Convergence Engine
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
        LAMBDA_LOWER=LAMBDA_LOWER,
        LAMBDA_UPPER=LAMBDA_UPPER,
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=WARMUP_TRIALS, constraints_func=constraint_func)
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{local_db}", sampler=sampler, direction="minimize")

    # 6. Run optimization
    study.optimize(objective, callbacks=[convergence_cb])

    # 7. Post-study tie-break breakdown
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

    winner = None
    if feasible:
        feasible.sort(key=lambda t: t.user_attrs["top_k_avg_energy"])
        best_e = feasible[0].user_attrs["top_k_avg_energy"]
        cutoff_e = best_e + abs(best_e) * (ENERGY_TOLERANCE_PCT / 100.0) if best_e < 0 else best_e * (1.0 + ENERGY_TOLERANCE_PCT / 100.0)
        near_best = [t for t in feasible if t.user_attrs["top_k_avg_energy"] <= cutoff_e]
        near_best.sort(key=lambda t: (t.user_attrs["top_k_avg_energy"], t.user_attrs["penalty_sum"]))
        winner = near_best[0]

        print(f"\n{'='*115}")
        print(f"🏆 TIE-BREAK BREAKDOWN (Feasible Trials within {ENERGY_TOLERANCE_PCT}% of Best Energy: {best_e:.4f})")
        print(f"{'='*115}")
        print(f"{'Trial':<7} | {'Top-3 Energy':<14} | {'Std Dev':<8} | {'Penalty Sum':<12} | {'Feas %':<8} | {'λ_budget':<10} | {'λ_conn':<10} | Notes")
        print(f"{'-'*115}")

        if LOG_WANDB_TABLE and wandb_run:
            wb_table = wandb.Table(columns=["Trial", "Top3_Energy", "Std_Dev", "Penalty_Sum", "Feas_Pct", "Lambda_Budget", "Lambda_Conn", "Notes"])

        for t in near_best:
            num = t.number
            e_val = t.user_attrs["top_k_avg_energy"]
            std_v = t.user_attrs["top_k_energy_std_dev"]
            p_sum = t.user_attrs["penalty_sum"]
            f_pct = t.user_attrs["feasibility_rate"] * 100.0
            lb_v = t.user_attrs["lambda_budget"]
            lc_v = t.user_attrs["lambda_conn"]
            notes = "🥇 WINNER" if t.number == winner.number else ""
            print(f"#{num:<6} | {e_val:<14.4f} | {std_v:<8.4f} | {p_sum:<12.2f} | {f_pct:<7.2f}% | {lb_v:<10.2f} | {lc_v:<10.2f} | {notes}")
            if LOG_WANDB_TABLE and wandb_run:
                wb_table.add_data(f"#{num}", e_val, std_v, p_sum, f_pct, lb_v, lc_v, notes)

        print(f"{'='*115}\n")
        if LOG_WANDB_TABLE and wandb_run:
            wandb_run.log({"tie_break_candidates": wb_table})
            wandb_run.summary["global_best_feasible_energy"] = best_e
            wandb_run.summary["winning_penalty_sum"] = winner.user_attrs["penalty_sum"]

    else:
        print("\n⚠️ No feasible trials found. Using trial with lowest violation.")
        # Fallback: pick trial with minimal violation
        completed.sort(key=lambda t: (t.user_attrs.get("Mth_budget_dev", 1e9) + t.user_attrs.get("Mth_isolated_count", 1e9), t.user_attrs["top_k_avg_energy"]))
        winner = completed[0] if completed else None

    # 8. Save tuned parameters for Cell 8
    if winner is not None:
        tuned_params = {
            "N": N_true,
            "lambda_budget": float(winner.user_attrs["lambda_budget"]),
            "lambda_conn": float(winner.user_attrs["lambda_conn"]),
            "best_energy": float(winner.user_attrs["top_k_avg_energy"]),
            "penalty_sum": float(winner.user_attrs["penalty_sum"]),
            "feasibility_rate": float(winner.user_attrs["feasibility_rate"]),
        }
        tuned_params_all[N_true] = tuned_params

        # Save to GDrive and local
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
    print(f"{'N':<8} | {'λ_budget':<12} | {'λ_conn':<12} | {'Best Energy':<14} | {'Feas %':<8}")
    print("-" * 115)
    for N, p in sorted(tuned_params_all.items()):
        print(f"{N:<8} | {p['lambda_budget']:<12.4f} | {p['lambda_conn']:<12.4f} | {p['best_energy']:<14.4f} | {p['feasibility_rate']*100:<7.2f}%")
else:
    print("⚠️ No tuned parameters were saved.")
print("=" * 115)
print("✅ SA tuning complete for all tiers.")