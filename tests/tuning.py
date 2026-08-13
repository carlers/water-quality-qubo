#@title TUNING SCRIPT – COLAB VERSION (with SCIP instead of Gurobi)
# =============================================================================
# 1. Edit the configuration below to match your needs.
# 2. Run the cell – it will install dependencies, clone the repo (if needed),
#    and execute the tuning pipeline.
# =============================================================================

# --- USER CONFIGURATION (edit these) -----------------------------------------
MODE = "full"                # "test" for quick run (10 trials) or "full" (50)
DO_TUNE_SA = True               # whether to tune Simulated Annealing
DO_TUNE_SQA = True              # whether to tune Simulated Quantum Annealing
FORCE_RETUNE = True         # ignore existing Optuna studies and retune
USE_WANDB = True            # log to Weights & Biases (requires wandb account)
SEED = 42
K = 5
N = 20
DOMAIN_SIZE = 100


import sys, os, subprocess, warnings, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import wandb


# Install required packages for SCIP (if missing)
try:
    import ommx_pyscipopt_adapter
except ImportError:
    !pip install -q ommx ommx-pyscipopt-adapter

# Now import your local modules
from data.synthetic_data import load_master_data, generate_and_save_all
from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
from src.jij_solvers import solve_sa_jij, solve_sqa_jij, solve_greedy_jij, compute_energy
from src.jij_optuna import tune_sa, tune_sqa, compute_qsum
from src.utils import (
    compute_violation_rate, compute_matrix_differences, build_full_qubo_matrix,
    select_best_from_pareto, print_multiobjective_tuning_summary,
    safe_save_pickle, safe_load_pickle, NumpyEncoder,
    cleanup_tqdm, suppress_optuna_trial_logs,
)
from src.plotting import (
    plot_jij_deployment, plot_qubo_matrix_direct,
    plot_objective_vs_reads, plot_violation_vs_reads,
    save_and_log_optuna_plots, plot_scaling_benchmark,
)
from src.model import build_miqp, build_qubo, compute_pairwise_terms

# SCIP / OMMX imports
try:
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter
    import jijmodeling as jm
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("⚠️ SCIP / OMMX not available. Will fall back to greedy for baseline.")

warnings.filterwarnings('ignore')

# W&B availability
try:
    import wandb
    print("✅ Weights & Biases available.")
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# ============================================================================
# SCIP MIQP solver (sparse, using JijModeling + OMMX adapter)
# – Now WITHOUT fixed_mask constraint (fixed stations are removed from decision variables)
# ============================================================================
def build_miqp_problem_sparse(N: int, max_degree: int, num_edges: int) -> jm.Problem:
    problem = jm.Problem("WQM_MIQP_sparse", sense=jm.ProblemSense.MINIMIZE)

    a = problem.Placeholder("a", shape=(N,), dtype=jm.DataType.FLOAT)
    neighbor_indices = problem.Placeholder("neighbor_indices", shape=(N, max_degree), dtype=jm.DataType.NATURAL)
    neighbor_mask = problem.Placeholder("neighbor_mask", shape=(N, max_degree), dtype=jm.DataType.BINARY)
    # fixed_mask still exists but will be set to all zeros (so it does nothing)
    fixed_mask = problem.Placeholder("fixed_mask", shape=(N,), dtype=jm.DataType.BINARY)
    K_total = problem.Placeholder("K_total", ndim=0, dtype=jm.DataType.INTEGER)

    x = problem.BinaryVar("x", shape=(N,))

    # Linear term
    linear = jm.sum(jm.product(N), lambda i: a[i[0]] * x[i[0]])
    problem += linear

    # Quadratic term
    if num_edges > 0:
        edges = problem.Placeholder("edges", shape=(num_edges, 2), dtype=jm.DataType.NATURAL)
        Q_vals = problem.Placeholder("Q_vals", shape=(num_edges,), dtype=jm.DataType.FLOAT)
        quad = jm.sum(
            jm.product(num_edges),
            lambda e: Q_vals[e[0]] * x[edges[e[0], 0]] * x[edges[e[0], 1]]
        )
        problem += quad

    # Budget constraint
    problem += problem.Constraint("budget", jm.sum(jm.product(N), lambda i: x[i[0]]) == K_total)

    # Connectivity constraint (only among free stations – but fixed stations are not variables, 
    # so this only enforces connectivity via free stations; we will check feasibility separately)
    problem += problem.Constraint(
        "connectivity",
        lambda i: x[i] <= jm.sum(
            jm.product(max_degree),
            lambda k: neighbor_mask[i, k[0]] * x[neighbor_indices[i, k[0]]]
        ),
        domain=N
    )

    # Fixed stations constraint – now always satisfied because fixed_mask is all zeros
    problem += problem.Constraint(
        "fixed",
        jm.sum(jm.product(N), lambda i: fixed_mask[i[0]] * (1 - x[i[0]])) == 0
    )

    return problem

def solve_scip_miqp(instance_data, verbose=False):
    if not SCIP_AVAILABLE:
        return {"solution": None, "energy": np.nan, "runtime": 0.0,
                "status": "SCIP not available", "feasible": False,
                "violation_rate": 1.0}

    N = instance_data["N"]
    K = instance_data["K"]   # number of NEW stations to place (fixed stations already removed)
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    coords = np.asarray(instance_data["coords"])
    D_MAX = instance_data.get("D_max", 8.0)
    # has_fixed_neighbor indicates which free stations can connect to an existing station
    has_fixed_neighbor = instance_data.get("has_fixed_neighbor", np.zeros(N, dtype=int))

    # Build sparse edges from dense Q (upper triangular, non-zero)
    quad_edges = []
    quad_vals = []
    for i in range(N):
        for j in range(i+1, N):
            if Q[i, j] != 0:
                quad_edges.append((i, j))
                quad_vals.append(float(Q[i, j]))
    num_edges = len(quad_edges)

    # Build neighbor graph among free stations (already in instance_data["neigh"])
    neigh = np.asarray(instance_data["neigh"])  # N x N, binary
    max_degree = max(1, np.max(np.sum(neigh, axis=1))) if N > 0 else 1

    neighbor_indices = np.zeros((N, max_degree), dtype=np.int32)
    neighbor_mask = np.zeros((N, max_degree), dtype=np.int8)
    for i in range(N):
        nbrs = np.where(neigh[i] == 1)[0]
        for k, j in enumerate(nbrs[:max_degree]):
            neighbor_indices[i, k] = j
            neighbor_mask[i, k] = 1

    # Build JijModeling problem
    problem = build_miqp_problem_sparse(N, max_degree, num_edges)

    # Prepare data – fixed_mask is all zeros (no forced selections)
    data = {
        "K_total": int(K),
        "a": a.astype(float).tolist(),
        "neighbor_indices": neighbor_indices.tolist(),
        "neighbor_mask": neighbor_mask.astype(int).tolist(),
        "fixed_mask": [0] * N,   # dummy, no fixed stations
    }
    if num_edges > 0:
        data["edges"] = [list(edge) for edge in quad_edges]
        data["Q_vals"] = [float(v) for v in quad_vals]

    instance = problem.eval(data)

    start = time.perf_counter()
    try:
        solution = OMMXPySCIPOptAdapter.solve(instance)
    except Exception as e:
        runtime = time.perf_counter() - start
        return {"solution": None, "energy": np.nan, "runtime": runtime,
                "status": f"SCIP error: {e}", "feasible": False,
                "violation_rate": 1.0}

    runtime = time.perf_counter() - start

    # Decode solution
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

    selected = np.where(x_sol == 1)[0]
    print(f"  [SCIP] Selected free indices: {selected.tolist()} (total {len(selected)})")

    # Compute energy using new a, Q
    energy = compute_energy(x_sol, a, Q)

    # Feasibility check with budget and connectivity (including fixed neighbors)
    from src.jij_solvers import check_feasibility
    # Pass fixed_neighbors if available
    fixed_neighbors = instance_data.get("fixed_neighbors", None)
    feas_detail = check_feasibility(x_sol, neigh, K, fixed_neighbors=fixed_neighbors)
    feasible = feas_detail["feasible"]
    violation_rate = compute_violation_rate(x_sol, neigh, K)

    status = "optimal" if getattr(solution, "is_optimal", False) else "feasible"

    return {
        "solution": x_sol,
        "energy": energy,
        "runtime": runtime,
        "status": status,
        "feasible": feasible,
        "violation_rate": violation_rate,
        "budget_ok": feas_detail.get("budget_ok", False),
        "connectivity_ok": feas_detail.get("connectivity_ok", False),
    }

# ============================================================================
# Helper functions (copied from original script)
# ============================================================================
def extract_miqp_matrix(instance_data):
    N = instance_data["N"]
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    mat = np.zeros((N, N))
    for i in range(N):
        mat[i, i] += a[i]
    for i in range(N):
        for j in range(i+1, N):
            if Q[i, j] != 0:
                mat[i, j] += Q[i, j]
                mat[j, i] += Q[i, j]
    return mat

def build_qubo_matrix_from_instance(instance_data, penalty_weights):
    from src.jij_model import build_augmented_model, compile_instance
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)
    qubo_dict, _ = instance.to_qubo(penalty_weights=penalty_weights)
    N = instance_data["N"]
    Q_mat = np.zeros((N, N))
    for (i, j), coeff in qubo_dict.items():
        if i < N and j < N:
            if i == j:
                Q_mat[i, i] += coeff
            else:
                Q_mat[i, j] += coeff
                Q_mat[j, i] += coeff
    return Q_mat

def compute_esr_mcr(miqp_mat, qubo_mat):
    norm_obj = np.linalg.norm(miqp_mat, 'fro')
    norm_pen = np.linalg.norm(qubo_mat - miqp_mat, 'fro')
    esr = norm_obj / norm_pen if norm_pen > 1e-12 else float('inf')
    max_obj = np.max(np.abs(miqp_mat))
    max_pen = np.max(np.abs(qubo_mat - miqp_mat))
    mcr = max_pen / max_obj if max_obj > 1e-12 else float('inf')
    return {"ESR": esr, "MCR": mcr}

# ============================================================================
# Main tuning function
# ============================================================================
def run_tuning(mode="test", do_tune_sa=True, do_tune_sqa=True, force_retune=False,
               use_wandb=False, seed=42, K=5, N=20):
    """
    Execute the tuning pipeline with the given parameters.
    """
    # Set output directories
    SAVE_DIR = Path(f"results_tuning_{mode}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR = SAVE_DIR / "optuna_studies"
    OPTUNA_DIR.mkdir(exist_ok=True)
    PLOTS_DIR = SAVE_DIR / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("🔬 JijModeling Tuning Orchestration (Colab) – using SCIP as exact solver")
    print("=" * 80)
    print(f"  Mode: {mode.upper()}")
    print(f"  Tune SA: {tune_sa}")
    print(f"  Tune SQA: {tune_sqa}")
    print(f"  Force retune: {force_retune}")
    print(f"  W&B: {use_wandb}")
    print(f"  Seed: {seed}")
    print(f"  K: {K}")
    print(f"  N: {N}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Data generation / loading
    # -------------------------------------------------------------------------
    print("\n📦 Loading/Generating synthetic data...")
    data_dir = Path(f"data_seefdd{seed}")
    if not data_dir.exists():
        generate_and_save_all(seed=seed, output_dir=str(data_dir), n_master=500, domain_size=DOMAIN_SIZE, n_existing=3, min_existing_distance=15, subset_sizes=[20, 50, 100, 200], D_max=100)
    coords_master, factors_master, U_master, subsets, meta = load_master_data(str(data_dir))
    if N not in subsets:
        raise ValueError(f"N={N} not in subsets. Available: {list(subsets.keys())}")
    subset_data = subsets[N]
    coords = subset_data["coords"]
    U = subset_data["U"]
    M_indices_original = meta["existing_indices"]
    orig_indices = subset_data["indices"]
    idx_map = {orig: new for new, orig in enumerate(orig_indices)}
    M_indices = [idx_map[m] for m in M_indices_original if m in idx_map]
    # Build pairwise
    L_c = 5.0; L_w = 1.0; beta = 1.0; delta = 1.0; connectivity_range = 100.0
    current_vector = (1.0, 0.0)
    pairwise = compute_pairwise_terms(
        coords=coords, U=U, M_indices=M_indices,
        L_c=L_c, L_w=L_w, current_vector=current_vector,
        beta=beta, delta=delta, connectivity_range=connectivity_range,
        verbose=False,
    )

    # ===== NEW: Remove fixed stations from decision space =====
    free_indices = [i for i in range(N) if i not in M_indices]
    M_set = set(M_indices)

    # 1. New linear coefficients (a_free)
    a_new = np.zeros(len(free_indices))
    for new_i, orig_i in enumerate(free_indices):
        a_new[new_i] = pairwise["linear"].get(orig_i, 0.0)
        # Add interactions with fixed stations
        for m in M_set:
            if (orig_i, m) in pairwise["quad"]:
                a_new[new_i] += pairwise["quad"][(orig_i, m)]
            elif (m, orig_i) in pairwise["quad"]:
                a_new[new_i] += pairwise["quad"][(m, orig_i)]

    # 2. New quadratic coefficients (Q_new)
    Q_new = np.zeros((len(free_indices), len(free_indices)))
    for a_idx, orig_a in enumerate(free_indices):
        for b_idx, orig_b in enumerate(free_indices):
            if a_idx < b_idx:
                val = pairwise["quad"].get((orig_a, orig_b), 0.0)
                if val != 0:
                    Q_new[a_idx, b_idx] = val
                    Q_new[b_idx, a_idx] = val

    # 3. New neighbor graph (only among free stations)
    neigh_new = np.zeros((len(free_indices), len(free_indices)), dtype=int)
    for i, orig_i in enumerate(free_indices):
        for j, orig_j in enumerate(free_indices):
            if i != j and pairwise["neighbors"][orig_i].count(orig_j) > 0:
                neigh_new[i, j] = 1

    # 4. For connectivity, store which free stations can connect to existing stations
    fixed_neighbors = {
        i: [m for m in M_set if pairwise["neighbors"][orig_i].count(m) > 0]
        for i, orig_i in enumerate(free_indices)
    }
    has_fixed_neighbor = np.array([len(fixed_neighbors[i]) > 0 for i in range(len(free_indices))], dtype=int)

    # 5. Build reduced instance_data
    instance_data = {
        "N": len(free_indices),
        "K": K,
        "a": a_new,
        "Q": Q_new,
        "neigh": neigh_new,
        "coords": coords[free_indices],          # reduced coords
        "U": U[free_indices],
        "M_indices": [],                         # no fixed indices in decision space
        "D_max": connectivity_range,
        "original_indices": free_indices,        # mapping from new index to original
        "fixed_indices": M_indices,              # original fixed indices
        "fixed_neighbors": fixed_neighbors,
        "has_fixed_neighbor": has_fixed_neighbor,
        "L_c": L_c, "L_w": L_w,
        # --- Add these for plotting ---
        "original_coords": coords,               # full coordinates (all stations)
        "original_U": U,                         # full utilities
        "original_all_indices": list(range(N)),  # optional
    }

    print(f"  Loaded subset N={N} with |M|={len(M_indices)}")
    print(f"  Reduced decision space: {len(free_indices)} free stations, K={K}")
    print(f"  Linear terms: {len(pairwise['linear'])}, Quadratic terms: {len(pairwise['quad'])}")

    qsum = compute_qsum(instance_data)
    print(f"  Qsum = {qsum:.4f}")

    # -------------------------------------------------------------------------
    # 2. SCIP baseline (exact MIQP)
    # -------------------------------------------------------------------------
    print("\n🎯 Solving MIQP exactly with SCIP...")
    scip_result = solve_scip_miqp(instance_data, verbose=False)
    if scip_result["solution"] is not None and scip_result["feasible"]:
        print(f"  SCIP optimal energy: {scip_result['energy']:.8f}")
        print(f"  SCIP feasible: {scip_result['feasible']}, violation: {scip_result['violation_rate']}")
        optimal_energy = scip_result["energy"]
        if scip_result["solution"] is not None and scip_result["feasible"]:
            selected = np.where(scip_result["solution"] == 1)[0]
        print(f"  SCIP selected indices: {selected.tolist()} (total {len(selected)})")
        optimal_energy = scip_result["energy"]
    else:
        print(f"  ⚠️ SCIP failed: {scip_result['status']}. Using greedy fallback.")
        greedy = solve_greedy_jij(instance_data, verbose=False)
        optimal_energy = greedy["energy"]
        print(f"  Greedy energy: {optimal_energy:.8f}")
    miqp_energy = optimal_energy

    # Plot SCIP deployment
    if scip_result["solution"] is not None:
        # Note: plot_jij_deployment expects M_indices in instance_data; we now have M_indices = [],
        # but we stored the original fixed_indices separately. We'll pass a temporary dict.
        plot_jij_deployment(
            instance_data, scip_result["solution"], {},
            save_path=PLOTS_DIR / "scip_deployment.png",
            show_fig=True, dpi=150, domain_size=DOMAIN_SIZE
        )

    # -------------------------------------------------------------------------
    # 3. W&B
    # -------------------------------------------------------------------------
    import os
    import wandb
    
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        try:
            # 1. Force the API key directly into the Python environment
            os.environ["WANDB_API_KEY"] = "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"
            
            # 2. Log in programmatically using the key
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
            
            # 3. Initialize the run
            wandb_run = wandb.init(
                project="wqm-placement-optimization",
                config={
                    "mode": mode,
                    "seed": seed,
                    "K": K,
                    "N": N,
                    "tune_sa": do_tune_sa,      
                    "tune_sqa": do_tune_sqa,    
                    "qsum": qsum
                },
                name=f"tuning_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            print("✅ W&B logging enabled and authenticated.")
        except Exception as e:
            print(f"  ⚠️ W&B init failed: {e}")
            wandb_run = None

    optuna.logging.set_verbosity(optuna.logging.INFO)


    # -------------------------------------------------------------------------
    # 4. SA Tuning
    # -------------------------------------------------------------------------
    sa_params = None
    sa_study = None
    if tune_sa:
        print("\n" + "=" * 80)
        print("🔬 SA Tuning")
        print("=" * 80)
        patience = 10 if mode == "test" else 50
        sa_params, sa_study = tune_sa(
            instance_data=instance_data,
            qsum=qsum,
            num_sweeps=10000,
            num_reads=1024,
            patience=patience,
            study_name="sa_tuning",
            storage_dir=OPTUNA_DIR,
            force_retune=force_retune,
            wandb_run=wandb_run,
            verbose=True,
        )
        with open(SAVE_DIR / "best_sa_params.json", "w") as f:
            json.dump(sa_params, f, indent=2, cls=NumpyEncoder)

    # -------------------------------------------------------------------------
    # 5. SQA Tuning
    # -------------------------------------------------------------------------
    sqa_params = None
    sqa_study = None
    if tune_sqa:
        print("\n" + "=" * 80)
        print("🔬 SQA Tuning")
        print("=" * 80)
        patience = 10 if mode == "test" else 50
        sqa_params, sqa_study = tune_sqa(
            instance_data=instance_data,
            qsum=qsum,
            num_sweeps=10000,
            num_reads=1024,
            patience=patience,
            study_name="sqa_tuning",
            storage_dir=OPTUNA_DIR,
            force_retune=force_retune,
            wandb_run=wandb_run,
            verbose=True,
        )
        with open(SAVE_DIR / "best_sqa_params.json", "w") as f:
            json.dump(sqa_params, f, indent=2, cls=NumpyEncoder)

    # -------------------------------------------------------------------------
    # 6. Final evaluation with return_all=True
    # -------------------------------------------------------------------------
    def evaluate_solver(solver_name, params, solver_func, extra_kwargs=None):
        print(f"\n📊 Evaluating {solver_name} with best parameters...")
        if extra_kwargs is None:
            extra_kwargs = {}
        model = build_augmented_model()
        model_keys = {"N", "K", "a", "Q", "neigh"}
        filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
        instance = compile_instance(model, filtered_data)
        penalty_weights = get_penalty_weights(instance, params["lambda_budget"], params["lambda_conn"])
        result = solver_func(
            instance_data=instance_data,
            penalty_weights=penalty_weights,
            num_reads=params.get("num_reads", 1024),
            num_sweeps=params.get("num_sweeps", 10000),
            return_all=True,
            verbose=False,
            **extra_kwargs,
        )
        all_samples = result.get("all_samples", [])
        if all_samples:
            read_indices = list(range(len(all_samples)))
            objectives = [s["energy"] for s in all_samples]
            violation_rates = [s["violation_rate"] for s in all_samples]
        else:
            read_indices = [0]
            objectives = [result["energy"]]
            violation_rates = [result["violation_rate"]]
        sqr = result["energy"] / miqp_energy if miqp_energy != 0 else np.nan
        print(f"  {solver_name} best energy: {result['energy']:.8f}")
        print(f"  {solver_name} SQR: {sqr:.4f}")
        print(f"  {solver_name} feasible: {result['feasible']}, violation_rate: {result['violation_rate']}")
        print(f"  {solver_name} runtime: {result['runtime']:.4f}s")
        qubo_mat = build_qubo_matrix_from_instance(instance_data, penalty_weights)
        miqp_mat = extract_miqp_matrix(instance_data)
        esr_mcr = compute_esr_mcr(miqp_mat, qubo_mat)
        print(f"  {solver_name} ESR: {esr_mcr['ESR']:.4f}, MCR: {esr_mcr['MCR']:.4f}")
        diff = compute_matrix_differences(miqp_mat, qubo_mat)
        print(f"  {solver_name} Matrix diff: MSE={diff['MSE']:.6f}, RMSE={diff['RMSE']:.6f}, Frobenius={diff['Frobenius']:.6f}")
        result["sqr"] = sqr
        result["read_indices"] = read_indices
        result["objectives"] = objectives
        result["violation_rates"] = violation_rates
        result["qubo_matrix"] = qubo_mat
        result["miqp_matrix"] = miqp_mat
        result["esr_mcr"] = esr_mcr
        result["matrix_diff"] = diff
        return result

    sa_eval = None
    sqa_eval = None
    if sa_params is not None:
        sa_eval = evaluate_solver("SA", sa_params, solve_sa_jij)
    if sqa_params is not None:
        trotter = sqa_params.get("trotter", 16)
        sqa_eval = evaluate_solver("SQA", sqa_params, solve_sqa_jij, extra_kwargs={"trotter": trotter})

    # -------------------------------------------------------------------------
    # 7. Generate plots
    # -------------------------------------------------------------------------
    print("\n📈 Generating plots...")
    if sa_eval and sa_eval["read_indices"]:
        fig, ax = plt.subplots(figsize=(8,5))
        plot_objective_vs_reads(ax, sa_eval["read_indices"], sa_eval["objectives"], label="SA", color="blue")
        if sqa_eval and sqa_eval["read_indices"]:
            plot_objective_vs_reads(ax, sqa_eval["read_indices"], sqa_eval["objectives"], label="SQA", color="red")
        ax.axhline(y=miqp_energy, color='green', linestyle='--', label='SCIP Optimum')
        ax.set_title("Objective Value vs Read Index")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "objective_vs_reads.png", dpi=150)
        plt.show()
        plt.close(fig)
        if wandb_run:
            wandb_run.log({"objective_vs_reads": wandb.Image(str(PLOTS_DIR / "objective_vs_reads.png"))})

    if sa_eval and sa_eval["read_indices"]:
        fig, ax = plt.subplots(figsize=(8,5))
        plot_violation_vs_reads(ax, sa_eval["read_indices"], sa_eval["violation_rates"], label="SA", color="blue")
        if sqa_eval and sqa_eval["read_indices"]:
            plot_violation_vs_reads(ax, sqa_eval["read_indices"], sqa_eval["violation_rates"], label="SQA", color="red")
        ax.set_title("Violation Rate vs Read Index")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "violation_vs_reads.png", dpi=150)
        plt.show()
        plt.close(fig)
        if wandb_run:
            wandb_run.log({"violation_vs_reads": wandb.Image(str(PLOTS_DIR / "violation_vs_reads.png"))})

    if sa_eval and sa_eval["solution"] is not None:
        plot_jij_deployment(
            instance_data, sa_eval["solution"],
            get_penalty_weights(instance, sa_params["lambda_budget"], sa_params["lambda_conn"]),
            save_path=PLOTS_DIR / "sa_deployment.png", show_fig=True, dpi=150,
        )
        if wandb_run:
            wandb_run.log({"sa_deployment": wandb.Image(str(PLOTS_DIR / "sa_deployment.png"))})
    if sqa_eval and sqa_eval["solution"] is not None:
        plot_jij_deployment(
            instance_data, sqa_eval["solution"],
            get_penalty_weights(instance, sqa_params["lambda_budget"], sqa_params["lambda_conn"]),
            save_path=PLOTS_DIR / "sqa_deployment.png", show_fig=True, dpi=150,
        )
        if wandb_run:
            wandb_run.log({"sqa_deployment": wandb.Image(str(PLOTS_DIR / "sqa_deployment.png"))})

    if sa_eval:
        plot_qubo_matrix_direct(
            sa_eval["qubo_matrix"],
            save_path=PLOTS_DIR / "qubo_matrix_sa.png",
            title=f"SA QUBO Matrix (λ₁={sa_params['lambda_budget']:.4f}, λ₂={sa_params['lambda_conn']:.4f})",
            show_fig=True, dpi=150,
        )
        if wandb_run:
            wandb_run.log({"qubo_matrix_sa": wandb.Image(str(PLOTS_DIR / "qubo_matrix_sa.png"))})
    if sqa_eval:
        plot_qubo_matrix_direct(
            sqa_eval["qubo_matrix"],
            save_path=PLOTS_DIR / "qubo_matrix_sqa.png",
            title=f"SQA QUBO Matrix (λ₁={sqa_params['lambda_budget']:.4f}, λ₂={sqa_params['lambda_conn']:.4f})",
            show_fig=True, dpi=150,
        )
        if wandb_run:
            wandb_run.log({"qubo_matrix_sqa": wandb.Image(str(PLOTS_DIR / "qubo_matrix_sqa.png"))})

    if sa_study is not None:
        sa_optuna_dir = PLOTS_DIR / "optuna_sa"
        sa_optuna_dir.mkdir(exist_ok=True)
        save_and_log_optuna_plots(sa_study, sa_optuna_dir, wandb_run, show_fig=True)
    if sqa_study is not None:
        sqa_optuna_dir = PLOTS_DIR / "optuna_sqa"
        sqa_optuna_dir.mkdir(exist_ok=True)
        save_and_log_optuna_plots(sqa_study, sqa_optuna_dir, wandb_run, show_fig=True)

    # -------------------------------------------------------------------------
    # 8. Save results
    # -------------------------------------------------------------------------
    results = {
        "config": {"mode": mode, "seed": seed, "K": K, "N": N, "qsum": qsum, "scip_miqp": miqp_energy},
        "scip": scip_result,
        "sa": {"params": sa_params, "eval": sa_eval} if sa_eval else None,
        "sqa": {"params": sqa_params, "eval": sqa_eval} if sqa_eval else None,
    }
    safe_save_pickle(SAVE_DIR / "tuning_results.pkl", results, verbose=True)
    with open(SAVE_DIR / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # -------------------------------------------------------------------------
    # 9. Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ TUNING COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {SAVE_DIR.resolve()}")
    print("  - best_sa_params.json, best_sqa_params.json")
    print("  - tuning_results.pkl, tuning_results.json")
    print("  - plots/ (deployment, QUBO matrices, objective/violation curves, Optuna plots)")
    print("  - optuna_studies/ (SQLite databases)")
    print("\n📊 Summary:")
    print(f"  SCIP MIQP: {miqp_energy:.8f}")
    if sa_params:
        print(f"  SA best λ₁={sa_params['lambda_budget']:.4f}, λ₂={sa_params['lambda_conn']:.4f}")
        if sa_eval:
            print(f"    SA SQR={sa_eval['sqr']:.4f}, violation={sa_eval['violation_rate']:.1f}")
    if sqa_params:
        print(f"  SQA best λ₁={sqa_params['lambda_budget']:.4f}, λ₂={sqa_params['lambda_conn']:.4f}, trotter={sqa_params.get('trotter', 'N/A')}")
        if sqa_eval:
            print(f"    SQA SQR={sqa_eval['sqr']:.4f}, violation={sqa_eval['violation_rate']:.1f}")
    print("=" * 80)
    if wandb_run:
        wandb_run.finish()
    cleanup_tqdm()
    print("🧹 Cleanup complete.")


# ============================================================================
#  RUN THE TUNING – uses the global flags defined at the top
# ============================================================================
if __name__ == "__main__":
    run_tuning(
        mode=MODE,
        do_tune_sa=DO_TUNE_SA,
        do_tune_sqa=DO_TUNE_SQA,
        force_retune=FORCE_RETUNE,
        use_wandb=USE_WANDB,
        seed=SEED,
        K=K,
        N=N
    )