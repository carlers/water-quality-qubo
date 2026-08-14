#@title 🔬 SA TUNE v2.13 (Dual-Trigger Mathematical Convergence + Full Rich Logging)
"""
================================================================================
SA TUNING WITH OPTUNA v2.13 – REAL DATA (MULTI-TIER)
================================================================================
- Single-Objective Lexicographic: Minimize Energy -> Minimize Penalty Sum.
- Constraint: M-th sample (e.g., 20th) must have 0 violations.
- Strategy A (Variance Collapse): Stops when Top-5 feasible log-variance σ_logλ ≤ 0.05.
- Strategy B (Floor Saturation): Stops when Exact Energy Floor hit 10x with no pen drop.
- Real-time terminal HUD tracks Floor Hits, Space Variance, λb, λc, and Global Best.
- Restores full Post-Study Tie-Break Breakdown Table & W&B Table artifact syncing.
================================================================================
"""

import sys, os, math, time, pickle, re, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import optuna
import wandb

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_SA = {
    "TARGET_N": 35,                     # None = tune all; or int (e.g., 35)

    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    
    # ---- Mathematical Convergence Config ----
    "WARMUP_TRIALS": 15,                # Minimum trials before ANY early stopping
    "MAX_FLOOR_HITS": 10,               # Stop if exact ground-state floor hit 10x without penalty improvement
    "VARIANCE_TOLERANCE": 0.05,         # Stop if joint log10 standard deviation of top parameters drops below this
    "MIN_FEASIBLE_FOR_VARIANCE": 5,     # Require at least 5 feasible trials to calculate variance

    # ---- Rank-Constrained Single Objective Config ----
    "FEASIBILITY_RANK_THRESHOLD": 20,   # M-th sample used for constraint boundary
    "TOP_K_ENERGY": 3,                  # Average top K energies for the objective
    "ENERGY_TOLERANCE_PCT": 0.5,        # 0.5% window for tie-break table candidates

    "LAMBDA_LOWER": 0.0001,
    
    "USE_WANDB": True,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,
    "RUN_ID": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "FORCE_RETUNE": True,
}

# Extract config
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

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
RUN_DIR = GDRIVE_BASE / f"run_{RUN_ID}"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

try:
    from src.jij_solvers import solve_sa_jij
    from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
    from src.utils import NumpyEncoder
except ImportError:
    print("⚠️ Ensure src modules are in your sys.path before running.")

# -----------------------------------------------------------------------------
# DUAL-TRIGGER CONVERGENCE CALLBACK & REAL-TIME LOGGER
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

        # 1. Retrieve current trial attributes
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

        # 2. Collect all completed feasible trials for parameter variance calculation
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        feasible_trials = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

        # 3. Strategy A: Joint Parameter Space Variance (σ_logλ)
        sigma_j = float('inf')
        sigma_str = "N/A"
        if len(feasible_trials) >= self.min_feas_var:
            feasible_trials.sort(key=lambda t: (t.user_attrs["top_k_avg_energy"], t.user_attrs["penalty_sum"]))
            top_m = feasible_trials[:self.min_feas_var]
            
            log_b = [math.log10(t.user_attrs["lambda_budget"]) for t in top_m]
            log_c = [math.log10(t.user_attrs["lambda_conn"]) for t in top_m]
            
            var_b = np.var(log_b, ddof=1)
            var_c = np.var(log_c, ddof=1)
            sigma_j = math.sqrt(var_b + var_c)
            sigma_str = f"{sigma_j:.3f}"

        # 4. Strategy B: Floor Saturation Tracker & Status Selection
        status_str = "❌ INFEASIBLE                        "
        if is_feas:
            rel_tol = max(1e-6, 1e-6 * abs(self.global_best_feasible_energy))
            
            # Found a strictly lower energy basin
            if current_e < self.global_best_feasible_energy - rel_tol:
                self.global_best_feasible_energy = current_e
                self.best_penalty_sum = current_p
                self.floor_hits = 1
                status_str = "🌟 NEW BEST FEASIBLE!             "
                
            # Hit the existing global ground-state energy floor
            elif abs(current_e - self.global_best_feasible_energy) <= rel_tol:
                if current_p < self.best_penalty_sum - 1e-4:
                    self.best_penalty_sum = current_p
                    self.floor_hits = 1  # Reset floor counter because penalty boundary was lowered
                    status_str = "🎯 FEASIBLE MATCH (LOWER PENALTY!)"
                else:
                    self.floor_hits += 1
                    status_str = "✅ FEASIBLE MATCH                 "
            
            # Feasible, but worse energy than current best
            else:
                status_str = "✅ FEASIBLE                       "

        # Record tracked global best feasible energy to trial attributes
        trial.set_user_attr("global_best_feasible_energy", self.global_best_feasible_energy if self.global_best_feasible_energy != float('inf') else None)

        # 5. Rich Live Terminal HUD Formatting
        floor_str = f"{self.floor_hits:2d}/{self.max_floor_hits:<2d}"
        pen_breakdown_str = f"{current_p:7.2f} (λb={lb:.2f}, λc={lc:.2f})"
        
        if is_feas:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | Top-3 Avg: {current_e:10.4f} (std:{std_dev:6.4f}) | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | {runtime_sec:5.2f}s")
        else:
            print(f"[Trial {trial.number:3d}] {status_str:<32} | M-th Viol: Budg={m_budget:2.0f}, Isol={m_isolated:2.0f} | "
                  f"Feas: {feas_rate:6.1%} | Pen Sum: {pen_breakdown_str} | Floor: {floor_str} | Var σ: {sigma_str:<5} | {runtime_sec:5.2f}s")

        # 6. Weights & Biases Step Logging
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

        # 7. Evaluate Mathematical Termination Triggers (After Warmup)
        if trial.number >= self.warmup:
            if self.floor_hits >= self.max_floor_hits:
                print(f"\n🛑 [Early Stopping: Strategy B] Exact energy floor ({self.global_best_feasible_energy:.4f}) hit {self.max_floor_hits} times with no penalty improvement. (Best Pen: {self.best_penalty_sum:.2f})")
                study.stop()
            elif sigma_j <= self.var_tol:
                print(f"\n🛑 [Early Stopping: Strategy A] Parameter space variance collapsed (σ = {sigma_j:.4f} ≤ {self.var_tol}). Search space is exhausted.")
                study.stop()


# -----------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# -----------------------------------------------------------------------------
def make_objective(instance_data, neigh, K, fixed_neighbors, LAMBDA_LOWER, LAMBDA_UPPER):
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    def objective(trial):
        start_time = time.time()
        lambda_budget = trial.suggest_float("lambda_budget", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        penalty_sum = lambda_budget + lambda_conn

        trial.set_user_attr("lambda_budget", lambda_budget)
        trial.set_user_attr("lambda_conn", lambda_conn)
        trial.set_user_attr("penalty_sum", penalty_sum)

        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)
        result = solve_sa_jij(instance_data, penalty_weights, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, return_all=True)
        all_samples = result.get("all_samples", [])
        
        trial.set_user_attr("runtime_sec", time.time() - start_time)

        if not all_samples:
            trial.set_user_attr("Mth_budget_dev", 1e9)
            trial.set_user_attr("Mth_isolated_count", 1e9)
            trial.set_user_attr("feasibility_rate", 0.0)
            return 1e9

        b_devs, i_cnts = [], []
        for s in all_samples:
            num_selected = np.sum(s["solution"])
            b_dev = float(abs(num_selected - K))
            i_cnt = 0.0
            selected = np.where(s["solution"] == 1)[0]
            for i in selected:
                has_free = np.sum(neigh[i] * s["solution"]) > 0
                has_fixed = False
                if not has_free and fixed_neighbors is not None:
                    if len(fixed_neighbors.get(i, [])) > 0: has_fixed = True
                if not (has_free or has_fixed): i_cnt += 1.0
            
            s["budget_dev"] = b_dev
            s["isolated_count"] = i_cnt
            s["total_violation"] = b_dev + i_cnt
            b_devs.append(b_dev)
            i_cnts.append(i_cnt)

        trial.set_user_attr("avg_budget_dev_all", float(np.mean(b_devs)))
        trial.set_user_attr("avg_isolated_count_all", float(np.mean(i_cnts)))

        all_samples.sort(key=lambda x: (x["total_violation"], x["energy"]))

        m_idx = min(FEASIBILITY_RANK_THRESHOLD - 1, len(all_samples) - 1)
        m_sample = all_samples[m_idx]
        m_budget = m_sample["budget_dev"]
        m_isolated = m_sample["isolated_count"]
        trial.set_user_attr("Mth_budget_dev", m_budget)
        trial.set_user_attr("Mth_isolated_count", m_isolated)

        feasible_samples = [s for s in all_samples if s["total_violation"] == 0]
        feas_rate = len(feasible_samples) / len(all_samples)
        trial.set_user_attr("feasibility_rate", feas_rate)

        if m_budget == 0 and m_isolated == 0:
            feasible_samples.sort(key=lambda x: x["energy"])
            top_k_samples = feasible_samples[:min(TOP_K_ENERGY, len(feasible_samples))]
        else:
            top_k_samples = all_samples[:min(TOP_K_ENERGY, len(all_samples))]

        top_k_energies = [s["energy"] for s in top_k_samples]
        top_k_avg_energy = float(np.mean(top_k_energies))
        top_k_std_dev = float(np.std(top_k_energies)) if len(top_k_energies) > 1 else 0.0

        trial.set_user_attr("top_k_avg_energy", top_k_avg_energy)
        trial.set_user_attr("top_k_energy_std_dev", top_k_std_dev)

        return top_k_avg_energy

    return objective

# -----------------------------------------------------------------------------
# MAIN LOOP & POST-STUDY TIE-BREAK BREAKDOWN
# -----------------------------------------------------------------------------
pattern = re.compile(r"instance_data_N(\d+)\.pkl")
instance_files = [p for p in GDRIVE_BASE.glob("instance_data_N*.pkl") if pattern.match(p.name)]
if not instance_files: instance_files = [p for p in LOCAL_CACHE.glob("instance_data_N*.pkl") if pattern.match(p.name)]

if TARGET_N: instance_files = [p for p in instance_files if int(pattern.search(p.name).group(1)) == TARGET_N]
instance_files.sort(key=lambda p: int(pattern.search(p.name).group(1)))

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print(f"\n{'='*115}\n🔬 Tuning SA for tier N={N_true} (Mathematical Convergence Engine)\n{'='*115}")

    with open(instance_path, "rb") as f: instance_data = pickle.load(f)
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    qsum = max(np.sum(np.abs(a)) + np.sum(np.abs(np.triu(Q, 1))), 1.0)
    LAMBDA_UPPER = qsum

    study_name = f"sa_tuning_N{N_true}_{RUN_ID}"
    local_db = LOCAL_CACHE / f"{study_name}.db"

    if FORCE_RETUNE:
        try: optuna.delete_study(study_name=study_name, storage=f"sqlite:///{local_db}")
        except KeyError: pass

    wandb_run = None
    if USE_WANDB:
        wandb_run = wandb.init(project=WANDB_PROJECT, config=CONFIG_SA, name=f"{study_name}")
    
    # Initialize Engine
    convergence_cb = MathematicalConvergenceEngine(
        warmup=WARMUP_TRIALS,
        max_floor_hits=MAX_FLOOR_HITS,
        var_tol=VARIANCE_TOLERANCE,
        min_feas_var=MIN_FEASIBLE_FOR_VARIANCE,
        wandb_run=wandb_run
    )

    def constraint_func(trial):
        return [trial.user_attrs.get("Mth_budget_dev", 1e9), trial.user_attrs.get("Mth_isolated_count", 1e9)]

    objective = make_objective(instance_data, instance_data["neigh"], instance_data["K"], instance_data.get("fixed_neighbors"), LAMBDA_LOWER, LAMBDA_UPPER)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=WARMUP_TRIALS, constraints_func=constraint_func)
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{local_db}", sampler=sampler, direction="minimize")

    study.optimize(objective, callbacks=[convergence_cb])

    # -------------------------------------------------------------------------
    # POST-STUDY TIE-BREAK BREAKDOWN TABLE & W&B SYNC
    # -------------------------------------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

    if feasible:
        feasible.sort(key=lambda t: t.user_attrs["top_k_avg_energy"])
        best_e = feasible[0].user_attrs["top_k_avg_energy"]
        
        # Filter candidates within ENERGY_TOLERANCE_PCT (0.5%) of global minimum feasible energy
        cutoff_e = best_e + abs(best_e) * (ENERGY_TOLERANCE_PCT / 100.0) if best_e < 0 else best_e * (1.0 + ENERGY_TOLERANCE_PCT / 100.0)
        near_best = [t for t in feasible if t.user_attrs["top_k_avg_energy"] <= cutoff_e]
        
        # Sort candidates primarily by minimum energy, then by lowest penalty sum
        near_best.sort(key=lambda t: (t.user_attrs["top_k_avg_energy"], t.user_attrs["penalty_sum"]))

        print(f"\n{'='*115}")
        print(f"🏆 TIE-BREAK BREAKDOWN (Feasible Trials within {ENERGY_TOLERANCE_PCT}% of Best Energy: {best_e:.4f})")
        print(f"{'='*115}")
        print(f"{'Trial':<7} | {'Top-3 Energy':<14} | {'Std Dev':<8} | {'Penalty Sum':<12} | {'Feas %':<8} | {'λ_budget':<10} | {'λ_conn':<10} | Notes")
        print(f"{'-'*115}")

        if LOG_WANDB_TABLE and wandb_run:
            wb_table = wandb.Table(columns=["Trial", "Top3_Energy", "Std_Dev", "Penalty_Sum", "Feas_Pct", "Lambda_Budget", "Lambda_Conn", "Notes"])

        primary_winner = near_best[0]

        for i, t in enumerate(near_best):
            num = t.number
            e_val = t.user_attrs["top_k_avg_energy"]
            std_v = t.user_attrs["top_k_energy_std_dev"]
            p_sum = t.user_attrs["penalty_sum"]
            f_pct = t.user_attrs["feasibility_rate"] * 100.0
            lb_v = t.user_attrs["lambda_budget"]
            lc_v = t.user_attrs["lambda_conn"]

            notes = ""
            if t.number == primary_winner.number:
                notes = "🥇 SA/SQA WINNER | 🥈 QAOA CANDIDATE (SAME)"

            print(f"#{num:<6} | {e_val:<14.4f} | {std_v:<8.4f} | {p_sum:<12.2f} | {f_pct:<7.2f}% | {lb_v:<10.2f} | {lc_v:<10.2f} | {notes}")

            if LOG_WANDB_TABLE and wandb_run:
                wb_table.add_data(f"#{num}", e_val, std_v, p_sum, f_pct, lb_v, lc_v, notes)

        print(f"{'='*115}\n")

        if LOG_WANDB_TABLE and wandb_run:
            wandb_run.log({"tie_break_candidates": wb_table})
            wandb_run.summary["global_best_feasible_energy"] = best_e
            wandb_run.summary["winning_penalty_sum"] = primary_winner.user_attrs["penalty_sum"]

    else:
        print("\n⚠️ No feasible trials were found during this optimization run.")

    if wandb_run:
        wandb_run.finish()