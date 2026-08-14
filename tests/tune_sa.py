#@title 🔬 SA TUNE (Single-Objective, Rank-Constrained, Rich Logging)
"""
================================================================================
SA TUNING WITH OPTUNA – REAL DATA (MULTI-TIER)
================================================================================
- Single-Objective: Minimize Top-K Average Energy.
- Constraint: M-th sample (e.g., 20th) must have 0 violations.
- Dual Winner Selection: Primary (absolute best energy) & QAOA Candidate (lowest penalties within 0.5% gap).
- Rich terminal logging and W&B integration.
================================================================================
"""

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG_SA = {
    "TARGET_N": 35,              # None = tune all; or int (e.g., 35)

    "NUM_SWEEPS": 1000,
    "NUM_READS": 1024,
    "PATIENCE": 50,                # Stop after N trials without objective improvement
    "WARMUP_TRIALS": 20,

    # ---- Rank-Constrained Single Objective Config ----
    "FEASIBILITY_RANK_THRESHOLD": 20, # M-th sample used for constraint boundary
    "TOP_K_ENERGY": 3,                # Average top K energies for the objective
    "ENERGY_TOLERANCE_PCT": 0.5,      # 0.5% tolerance for finding QAOA Honorable Mention

    "LAMBDA_LOWER": 0.0001,
    
    "USE_WANDB": True,
    "WANDB_PROJECT": "wqm-placement-optimization",
    "LOG_WANDB_TABLE": True,
    "RUN_ID": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "FORCE_RETUNE": True,
    "BACKUP_FREQ": 10,
    "SAVE_PLOTS": True,
    "SHOW_PLOTS": True,
}

# Extract config
TARGET_N = CONFIG_SA["TARGET_N"]
NUM_SWEEPS = CONFIG_SA["NUM_SWEEPS"]
NUM_READS = CONFIG_SA["NUM_READS"]
PATIENCE = CONFIG_SA["PATIENCE"]
WARMUP_TRIALS = CONFIG_SA["WARMUP_TRIALS"]
FEASIBILITY_RANK_THRESHOLD = CONFIG_SA["FEASIBILITY_RANK_THRESHOLD"]
TOP_K_ENERGY = CONFIG_SA["TOP_K_ENERGY"]
ENERGY_TOLERANCE_PCT = CONFIG_SA["ENERGY_TOLERANCE_PCT"]
LAMBDA_LOWER = CONFIG_SA["LAMBDA_LOWER"]
USE_WANDB = CONFIG_SA["USE_WANDB"]
WANDB_PROJECT = CONFIG_SA["WANDB_PROJECT"]
LOG_WANDB_TABLE = CONFIG_SA["LOG_WANDB_TABLE"]
RUN_ID = CONFIG_SA["RUN_ID"]
FORCE_RETUNE = CONFIG_SA["FORCE_RETUNE"]
BACKUP_FREQ = CONFIG_SA["BACKUP_FREQ"]
SAVE_PLOTS = CONFIG_SA["SAVE_PLOTS"]
SHOW_PLOTS = CONFIG_SA["SHOW_PLOTS"]

# -----------------------------------------------------------------------------
# IMPORTS & PATHS
# -----------------------------------------------------------------------------
import sys, os, json, time, pickle, re, shutil, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import optuna
import wandb

try:
    from src.jij_solvers import solve_sa_jij
    from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
    from src.utils import safe_save_pickle, safe_load_pickle, NumpyEncoder
except ImportError:
    print("⚠️ Ensure src modules are in your sys.path before running.")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
RUN_DIR = GDRIVE_BASE / f"run_{RUN_ID}"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def compute_qsum(instance_data):
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    qsum = np.sum(np.abs(a)) + np.sum(np.abs(np.triu(Q, 1)))
    return max(qsum, 1.0)

def compute_decoupled_violations(x, neigh, K, fixed_neighbors=None):
    num_selected = np.sum(x)
    budget_dev = float(abs(num_selected - K))
    isolated_count = 0
    selected = np.where(x == 1)[0]
    for i in selected:
        has_free = np.sum(neigh[i] * x) > 0
        has_fixed = False
        if not has_free and fixed_neighbors is not None:
            if len(fixed_neighbors.get(i, [])) > 0:
                has_fixed = True
        if not (has_free or has_fixed):
            isolated_count += 1
    return budget_dev, float(isolated_count)

# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------
class ConstrainedSingleObjectivePatience:
    def __init__(self, patience: int, warmup: int):
        self.patience = patience
        self.warmup = warmup
        self.best_feasible_energy = float('inf')
        self.no_improvement_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # Check if trial is strictly feasible according to constraints
        if trial.user_attrs.get("Mth_budget_dev", 1) > 0 or trial.user_attrs.get("Mth_isolated_count", 1) > 0:
            return # Ignore infeasible trials for patience counting

        current_val = trial.value
        if current_val < self.best_feasible_energy:
            self.best_feasible_energy = current_val
            self.no_improvement_count = 0
        else:
            if trial.number >= self.warmup:
                self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            print(f"\n🛑 [Early Stopping] No feasible energy improvement for {self.patience} trials.")
            study.stop()

# -----------------------------------------------------------------------------
# OBJECTIVE
# -----------------------------------------------------------------------------
def make_objective(instance_data, neigh, K, fixed_neighbors, LAMBDA_LOWER, LAMBDA_UPPER, wandb_run=None, wandb_table=None):
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)

    def objective(trial):
        lambda_budget = trial.suggest_float("lambda_budget", LAMBDA_LOWER, LAMBDA_UPPER, log=True)
        lambda_conn = trial.suggest_float("lambda_conn", LAMBDA_LOWER, LAMBDA_UPPER, log=True)

        penalty_weights = get_penalty_weights(instance, lambda_budget, lambda_conn)

        result = solve_sa_jij(instance_data, penalty_weights, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, return_all=True)
        all_samples = result.get("all_samples", [])

        if not all_samples:
            trial.set_user_attr("Mth_budget_dev", 1e9)
            trial.set_user_attr("Mth_isolated_count", 1e9)
            return 1e9

        # Compute violations
        for s in all_samples:
            b_dev, i_cnt = compute_decoupled_violations(s["solution"], neigh, K, fixed_neighbors)
            s["budget_dev"] = b_dev
            s["isolated_count"] = i_cnt
            s["total_violation"] = b_dev + i_cnt

        # Sort by total violation, then energy
        all_samples.sort(key=lambda x: (x["total_violation"], x["energy"]))

        # M-th Sample logic
        m_idx = min(FEASIBILITY_RANK_THRESHOLD - 1, len(all_samples) - 1)
        m_sample = all_samples[m_idx]
        m_budget = m_sample["budget_dev"]
        m_isolated = m_sample["isolated_count"]
        trial.set_user_attr("Mth_budget_dev", m_budget)
        trial.set_user_attr("Mth_isolated_count", m_isolated)

        # Track total feasibility rate
        feasible_samples = [s for s in all_samples if s["total_violation"] == 0]
        feas_rate = len(feasible_samples) / len(all_samples)
        trial.set_user_attr("feasibility_rate", feas_rate)

        # Calculate Objective: Top-K Average Energy
        if m_budget == 0 and m_isolated == 0:
            # We have >= M perfectly feasible samples. Average top K feasible.
            feasible_samples.sort(key=lambda x: x["energy"])
            top_k_samples = feasible_samples[:min(TOP_K_ENERGY, len(feasible_samples))]
        else:
            # Not enough feasible samples. Average top K closest to feasible.
            top_k_samples = all_samples[:min(TOP_K_ENERGY, len(all_samples))]

        top_k_avg_energy = float(np.mean([s["energy"] for s in top_k_samples]))
        trial.set_user_attr("top_k_avg_energy", top_k_avg_energy)
        trial.set_user_attr("penalty_sum", lambda_budget + lambda_conn)

        if wandb_run:
            log_dict = {
                "trial": trial.number,
                "top_k_avg_energy": top_k_avg_energy,
                "feasibility_rate": feas_rate,
                "Mth_budget_dev": m_budget,
                "Mth_isolated_count": m_isolated,
                "lambda_budget": lambda_budget,
                "lambda_conn": lambda_conn,
                "penalty_sum": lambda_budget + lambda_conn
            }
            wandb_run.log(log_dict)
            if wandb_table:
                wandb_table.add_data(trial.number, top_k_avg_energy, feas_rate, m_budget, m_isolated, lambda_budget, lambda_conn, lambda_budget + lambda_conn)

        return top_k_avg_energy

    return objective

def constraint_func(trial):
    return [trial.user_attrs.get("Mth_budget_dev", 1e9), trial.user_attrs.get("Mth_isolated_count", 1e9)]

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
pattern = re.compile(r"instance_data_N(\d+)\.pkl")
instance_files = [p for p in LOCAL_CACHE.glob("instance_data_N*.pkl") if pattern.match(p.name)]
if TARGET_N:
    instance_files = [p for p in instance_files if int(pattern.search(p.name).group(1)) == TARGET_N]
instance_files.sort(key=lambda p: int(pattern.search(p.name).group(1)))

for instance_path in instance_files:
    N_true = int(pattern.search(instance_path.name).group(1))
    print(f"\n{'='*80}\n🔬 Tuning SA for tier N={N_true}\n{'='*80}")

    with open(instance_path, "rb") as f:
        instance_data = pickle.load(f)
    
    qsum = compute_qsum(instance_data)
    LAMBDA_UPPER = qsum
    print(f"  qsum = {qsum:.4f}  →  Lambda upper bound = {LAMBDA_UPPER:.4f}")

    study_name = f"sa_tuning_N{N_true}_{RUN_ID}"
    local_db = LOCAL_CACHE / f"{study_name}.db"

    if FORCE_RETUNE:
        try: optuna.delete_study(study_name=study_name, storage=f"sqlite:///{local_db}")
        except KeyError: pass

    wandb_run, wandb_table = None, None
    if USE_WANDB:
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            config=CONFIG_SA,
            name=f"{study_name}_{datetime.now().strftime('%H%M%S')}"
        )
        if LOG_WANDB_TABLE:
            wandb_table = wandb.Table(columns=["trial", "top_k_avg_energy", "feas_rate", "Mth_budg", "Mth_isol", "lam_b", "lam_c", "pen_sum"])

    objective = make_objective(instance_data, instance_data["neigh"], instance_data["K"], instance_data.get("fixed_neighbors"), LAMBDA_LOWER, LAMBDA_UPPER, wandb_run, wandb_table)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=WARMUP_TRIALS, constraints_func=constraint_func)
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{local_db}", sampler=sampler, direction="minimize")

    study.optimize(objective, callbacks=[ConstrainedSingleObjectivePatience(PATIENCE, WARMUP_TRIALS)])

    # --- RESULTS EXTRACTION ---
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in completed if t.user_attrs.get("Mth_budget_dev", 1) == 0 and t.user_attrs.get("Mth_isolated_count", 1) == 0]

    if not feasible:
        print(f"\n❌ No trials achieved the {FEASIBILITY_RANK_THRESHOLD}-sample feasibility constraint.")
        if wandb_run: wandb_run.finish()
        continue

    # Sort feasible by objective (Top-3 Avg Energy)
    feasible.sort(key=lambda t: t.value)
    primary_winner = feasible[0]
    best_energy = primary_winner.value

    # Find QAOA Candidate (Lowest penalty within X% gap)
    tolerance_margin = abs(best_energy) * (ENERGY_TOLERANCE_PCT / 100.0)
    cutoff = best_energy + tolerance_margin # Assuming energies are negative/minimization
    
    bracket = [t for t in feasible if t.value <= cutoff]
    bracket.sort(key=lambda t: t.user_attrs["penalty_sum"])
    qaoa_candidate = bracket[0]

    # --- TERMINAL REPORT ---
    print(f"\n{'='*95}")
    print(f"🏆 TIE-BREAK BREAKDOWN (Top Feasible Trials within {ENERGY_TOLERANCE_PCT}% of Best Energy)")
    print(f"{'='*95}")
    print(f"{'Trial':<6} | {'Top-3 Energy':<14} | {'Penalty Sum':<12} | {'Feas %':<8} | {'λ_budget':<10} | {'λ_conn':<10} | {'Notes'}")
    print("-" * 95)
    
    for t in sorted(bracket, key=lambda x: x.value):
        note = ""
        if t.number == primary_winner.number: note += "🥇 SA WINNER "
        if t.number == qaoa_candidate.number and qaoa_candidate.number != primary_winner.number: note += "🥈 QAOA CANDIDATE "
        
        print(f"#{t.number:<5} | {t.value:<14.4f} | {t.user_attrs['penalty_sum']:<12.2f} | "
              f"{t.user_attrs['feasibility_rate']:<8.2%} | {t.user_attrs['lambda_budget']:<10.2f} | "
              f"{t.user_attrs['lambda_conn']:<10.2f} | {note}")
    print(f"{'='*95}\n")

    # Save Best Params (Defaults to Primary Winner for SA benchmarking)
    best_params = primary_winner.params.copy()
    with open(RUN_DIR / f"best_sa_N{N_true}.json", "w") as f:
        json.dump(best_params, f, indent=2, cls=NumpyEncoder)
    
    if wandb_run:
        wandb_run.summary["primary_winner_trial"] = primary_winner.number
        wandb_run.summary["qaoa_candidate_trial"] = qaoa_candidate.number
        wandb_run.finish()