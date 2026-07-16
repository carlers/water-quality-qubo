#!/usr/bin/env python3
# =============================================================================
# tuning.py – JijModeling Tuning Orchestration
# =============================================================================
# This script performs hyperparameter tuning for SA and/or SQA on a fixed
# instance (N=20, K=5) using the JijModeling pipeline. It:
#   1. Generates/loads synthetic data subset N=20 with fixed M_indices.
#   2. Solves Gurobi baseline (exact MIQP) for reference.
#   3. Tunes SA (λ₁, λ₂) and/or SQA (λ₁, λ₂, trotter).
#   4. Evaluates the best hyperparameters with return_all=True to get per-read
#      objective and violation rate curves.
#   5. Generates deployment maps, QUBO matrix heatmaps, and Optuna plots.
#   6. Computes matrix differences (MIQP vs QUBO) and side metrics (ESR, MCR).
#   7. Logs all metrics, plots, and parameters to W&B (optional).
#   8. Saves results (JSON, pickles, .npy) to disk.
#
# Usage:
#   python tuning.py --mode test                    # Quick run (5 trials)
#   python tuning.py --mode full --tune_sa          # Full SA tuning
#   python tuning.py --mode full --tune_sqa --wandb # Full SQA with W&B
#   python tuning.py --mode test --tune_sa --tune_sqa --force_retune
# =============================================================================

import sys
import os
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from data.synthetic_data import load_master_data, generate_and_save_all
from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
from src.jij_solvers import (
    solve_sa_jij,
    solve_sqa_jij,
    solve_greedy_jij,
    compute_energy,
)
from src.jij_optuna import tune_sa, tune_sqa, compute_qsum
from src.utils import (
    compute_violation_rate,
    compute_matrix_differences,
    build_full_qubo_matrix,
    select_best_from_pareto,
    print_multiobjective_tuning_summary,
    safe_save_pickle,
    safe_load_pickle,
    NumpyEncoder,
    cleanup_tqdm,
    suppress_optuna_trial_logs,
)
from src.plotting import (
    plot_jij_deployment,
    plot_qubo_matrix_direct,
    plot_objective_vs_reads,
    plot_violation_vs_reads,
    save_and_log_optuna_plots,
    plot_scaling_benchmark,
)
from src.model import build_miqp, build_qubo

warnings.filterwarnings('ignore')

# Optional: W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Optional: Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None


# ============================================================================
# Helper: Solve Gurobi MIQP (exact baseline)
# ============================================================================
def solve_gurobi_miqp(
    instance_data: Dict,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> Dict:
    """
    Solve the MIQP exactly using Gurobi.
    Returns dict with 'solution', 'energy', 'runtime', 'status'.
    """
    if not GUROBI_AVAILABLE:
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": 0.0,
            "status": "Gurobi not installed",
            "feasible": False,
            "violation_rate": 1.0,
        }
    
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    neigh = np.asarray(instance_data["neigh"])
    K = instance_data["K"]
    N = instance_data["N"]
    
    start = time.perf_counter()
    try:
        model = gp.Model("WQM_Aug")
        model.setParam("OutputFlag", 1 if verbose else 0)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPGap", 1e-6)
        model.setParam("NonConvex", 2)
        
        x = model.addVars(N, vtype=GRB.BINARY, name="x")
        obj = gp.LinExpr()
        for i in range(N):
            obj += a[i] * x[i]
        qobj = gp.QuadExpr()
        for i in range(N):
            for j in range(i + 1, N):
                if Q[i, j] != 0:
                    qobj += Q[i, j] * x[i] * x[j]
        model.setObjective(obj + qobj, GRB.MINIMIZE)
        model.addConstr(gp.quicksum(x[i] for i in range(N)) == K, name="budget")
        for i in range(N):
            model.addConstr(x[i] <= gp.quicksum(neigh[i, j] * x[j] for j in range(N)), name=f"conn_{i}")
        model.optimize()
        runtime = time.perf_counter() - start
        
        if model.Status == GRB.OPTIMAL:
            x_sol = np.array([x[i].X for i in range(N)])
            energy = compute_energy(x_sol, a, Q)
            # Check feasibility (should be true)
            from src.jij_solvers import check_feasibility
            feas_detail = check_feasibility(x_sol, neigh, K)
            feasible = feas_detail["feasible"]
            violation_rate = compute_violation_rate(x_sol, neigh, K)
            return {
                "solution": x_sol,
                "energy": energy,
                "runtime": runtime,
                "status": "optimal",
                "feasible": feasible,
                "violation_rate": violation_rate,
                "budget_ok": feas_detail["budget_ok"],
                "connectivity_ok": feas_detail["connectivity_ok"],
            }
        else:
            return {
                "solution": None,
                "energy": np.nan,
                "runtime": runtime,
                "status": f"not optimal ({model.Status})",
                "feasible": False,
                "violation_rate": 1.0,
            }
    except Exception as e:
        runtime = time.perf_counter() - start
        return {
            "solution": None,
            "energy": np.nan,
            "runtime": runtime,
            "status": f"error: {e}",
            "feasible": False,
            "violation_rate": 1.0,
        }


# ============================================================================
# Helper: Extract MIQP matrix from instance_data (pairwise_raw)
# ============================================================================
def extract_miqp_matrix(instance_data: Dict) -> np.ndarray:
    """
    Extract the MIQP objective matrix (without penalties) from instance_data.
    Assumes instance_data contains 'a' and 'Q' arrays.
    Returns N×N symmetric matrix.
    """
    N = instance_data["N"]
    a = np.asarray(instance_data["a"])
    Q = np.asarray(instance_data["Q"])
    mat = np.zeros((N, N))
    # Linear terms on diagonal
    for i in range(N):
        mat[i, i] += a[i]
    # Quadratic off-diagonal (symmetric)
    for i in range(N):
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                mat[i, j] += Q[i, j]
                mat[j, i] += Q[i, j]  # symmetric
    return mat


# ============================================================================
# Helper: Build QUBO matrix from instance and penalty weights
# ============================================================================
def build_qubo_matrix_from_instance(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
) -> np.ndarray:
    """
    Build full QUBO matrix (symmetric) from a compiled instance and penalty weights.
    """
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


# ============================================================================
# Helper: Compute ESR and MCR
# ============================================================================
def compute_esr_mcr(miqp_mat: np.ndarray, qubo_mat: np.ndarray) -> Dict[str, float]:
    """
    Compute Energy Scale Ratio (ESR) and Max Coefficient Ratio (MCR).
    """
    # Remove diagonal? Usually we compare off-diagonal or full.
    # Here we use full Frobenius norms.
    norm_obj = np.linalg.norm(miqp_mat, 'fro')
    norm_pen = np.linalg.norm(qubo_mat - miqp_mat, 'fro')
    esr = norm_obj / norm_pen if norm_pen > 1e-12 else float('inf')
    max_obj = np.max(np.abs(miqp_mat))
    max_pen = np.max(np.abs(qubo_mat - miqp_mat))
    mcr = max_pen / max_obj if max_obj > 1e-12 else float('inf')
    return {"ESR": esr, "MCR": mcr}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="JijModeling Tuning Orchestration"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' for quick run (5 trials), 'full' for full tuning (patience=50)"
    )
    parser.add_argument(
        "--tune_sa",
        action="store_true",
        help="Enable SA tuning"
    )
    parser.add_argument(
        "--tune_sqa",
        action="store_true",
        help="Enable SQA tuning"
    )
    parser.add_argument(
        "--force_retune",
        action="store_true",
        help="Ignore existing Optuna studies and retune from scratch"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        help="Number of new stations (K)"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=20,
        help="Number of candidate sites (N) – must be in subsets"
    )
    args = parser.parse_args()

    # If neither SA nor SQA selected, default to both
    if not args.tune_sa and not args.tune_sqa:
        args.tune_sa = True
        args.tune_sqa = True

    # Output directory
    SAVE_DIR = Path(f"results_tuning_{args.mode}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR = SAVE_DIR / "optuna_studies"
    OPTUNA_DIR.mkdir(exist_ok=True)
    PLOTS_DIR = SAVE_DIR / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("🔬 JijModeling Tuning Orchestration")
    print("=" * 80)
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Tune SA: {args.tune_sa}")
    print(f"  Tune SQA: {args.tune_sqa}")
    print(f"  Force retune: {args.force_retune}")
    print(f"  W&B: {args.wandb}")
    print(f"  Seed: {args.seed}")
    print(f"  K: {args.K}")
    print(f"  N: {args.N}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Data generation / loading
    # -------------------------------------------------------------------------
    print("\n📦 Loading/Generating synthetic data...")
    # Ensure master data exists
    data_dir = Path(f"data_seed{args.seed}")
    if not data_dir.exists():
        generate_and_save_all(seed=args.seed, output_dir=str(data_dir))
    
    coords_master, factors_master, U_master, subsets, meta = load_master_data(str(data_dir))
    if args.N not in subsets:
        raise ValueError(f"N={args.N} not in subsets. Available: {list(subsets.keys())}")
    subset_data = subsets[args.N]
    coords = subset_data["coords"]
    U = subset_data["U"]
    M_indices_original = meta["existing_indices"]
    # Map M_indices to subset indices
    orig_indices = subset_data["indices"]
    idx_map = {orig: new for new, orig in enumerate(orig_indices)}
    M_indices = [idx_map[m] for m in M_indices_original if m in idx_map]
    N = args.N
    K = args.K
    # Build instance_data (with a, Q, neigh)
    # We need to compute pairwise terms for this subset. We'll use compute_pairwise_terms from model.
    from src.model import compute_pairwise_terms
    L_c = 5.0
    L_w = 1.0
    beta = 1.0
    delta = 1.0
    connectivity_range = 12.0
    current_vector = (1.0, 0.0)
    pairwise = compute_pairwise_terms(
        coords=coords,
        U=U,
        M_indices=M_indices,
        L_c=L_c,
        L_w=L_w,
        current_vector=current_vector,
        beta=beta,
        delta=delta,
        connectivity_range=connectivity_range,
        verbose=False,
    )
    # Build a, Q arrays from pairwise
    N_total = pairwise["N_total"]
    a = np.zeros(N_total)
    for i, coeff in pairwise["linear"].items():
        a[i] = coeff
    Q = np.zeros((N_total, N_total))
    for (i, j), coeff in pairwise["quad"].items():
        Q[i, j] = coeff
        Q[j, i] = coeff
    neigh = np.zeros((N_total, N_total), dtype=int)
    for i, nbrs in pairwise["neighbors"].items():
        for j in nbrs:
            neigh[i, j] = 1
    instance_data = {
        "N": N,
        "K": K,
        "a": a.tolist(),
        "Q": Q.tolist(),
        "neigh": neigh.tolist(),
        "coords": coords.tolist(),
        "U": U.tolist(),
        "M_indices": M_indices,
        "D_max": connectivity_range,
        "L_c": L_c,
        "L_w": L_w,
    }
    print(f"  Loaded subset N={N} with |M|={len(M_indices)}")
    print(f"  Linear terms: {len(pairwise['linear'])}, Quadratic terms: {len(pairwise['quad'])}")

    # Compute Qsum for penalty scaling
    qsum = compute_qsum(instance_data)
    print(f"  Qsum = {qsum:.4f}")

    # -------------------------------------------------------------------------
    # 2. Gurobi baseline (exact MIQP)
    # -------------------------------------------------------------------------
    print("\n🎯 Solving Gurobi baseline...")
    gurobi_result = solve_gurobi_miqp(instance_data, time_limit=60.0, verbose=False)
    if gurobi_result["solution"] is not None:
        print(f"  Gurobi optimal energy: {gurobi_result['energy']:.8f}")
        print(f"  Gurobi feasible: {gurobi_result['feasible']}, violation: {gurobi_result['violation_rate']}")
        gurobi_energy = gurobi_result["energy"]
    else:
        print(f"  ⚠️ Gurobi failed: {gurobi_result['status']}. Using greedy fallback.")
        greedy = solve_greedy_jij(instance_data, verbose=False)
        gurobi_energy = greedy["energy"]
        print(f"  Greedy energy: {gurobi_energy:.8f}")
    gurobi_miqp = gurobi_energy

    # Plot Gurobi deployment
    if gurobi_result["solution"] is not None:
        plot_jij_deployment(
            instance_data,
            gurobi_result["solution"],
            {},  # no penalties for MIQP
            save_path=PLOTS_DIR / "gurobi_deployment.png",
            show_fig=True,
            dpi=150,
        )
    else:
        print("  ⚠️ No Gurobi solution to plot.")

    # -------------------------------------------------------------------------
    # 3. W&B initialization
    # -------------------------------------------------------------------------
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project="wqm-placement-optimization",
                config={
                    "mode": args.mode,
                    "seed": args.seed,
                    "K": K,
                    "N": N,
                    "tune_sa": args.tune_sa,
                    "tune_sqa": args.tune_sqa,
                    "qsum": qsum,
                },
                name=f"tuning_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            wandb_run = wandb
            print("✅ W&B logging enabled.")
        except Exception as e:
            print(f"  ⚠️ W&B init failed: {e}")

    # Suppress Optuna trial logs
    suppress_optuna_trial_logs()

    # -------------------------------------------------------------------------
    # 4. SA Tuning
    # -------------------------------------------------------------------------
    sa_params = None
    sa_study = None
    if args.tune_sa:
        print("\n" + "=" * 80)
        print("🔬 SA Tuning")
        print("=" * 80)
        patience = 10 if args.mode == "test" else 50
        sa_params, sa_study = tune_sa(
            instance_data=instance_data,
            qsum=qsum,
            num_sweeps=10000,
            num_reads=1024,
            patience=patience,
            study_name="sa_tuning",
            storage_dir=OPTUNA_DIR,
            force_retune=args.force_retune,
            wandb_run=wandb_run,
            verbose=True,
        )
        # Save best params
        with open(SAVE_DIR / "best_sa_params.json", "w") as f:
            json.dump(sa_params, f, indent=2, cls=NumpyEncoder)

    # -------------------------------------------------------------------------
    # 5. SQA Tuning
    # -------------------------------------------------------------------------
    sqa_params = None
    sqa_study = None
    if args.tune_sqa:
        print("\n" + "=" * 80)
        print("🔬 SQA Tuning")
        print("=" * 80)
        patience = 10 if args.mode == "test" else 50
        sqa_params, sqa_study = tune_sqa(
            instance_data=instance_data,
            qsum=qsum,
            num_sweeps=10000,
            num_reads=1024,
            patience=patience,
            study_name="sqa_tuning",
            storage_dir=OPTUNA_DIR,
            force_retune=args.force_retune,
            wandb_run=wandb_run,
            verbose=True,
        )
        with open(SAVE_DIR / "best_sqa_params.json", "w") as f:
            json.dump(sqa_params, f, indent=2, cls=NumpyEncoder)

    # -------------------------------------------------------------------------
    # 6. Final Evaluation with return_all=True
    # -------------------------------------------------------------------------
    # Helper to evaluate a solver with its best params
    def evaluate_solver(solver_name, params, solver_func, extra_kwargs=None):
        print(f"\n📊 Evaluating {solver_name} with best parameters...")
        if extra_kwargs is None:
            extra_kwargs = {}
        # Build penalty weights
        model = build_augmented_model()
        model_keys = {"N", "K", "a", "Q", "neigh"}
        filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
        instance = compile_instance(model, filtered_data)
        penalty_weights = get_penalty_weights(instance, params["lambda_budget"], params["lambda_conn"])
        
        # Run solver with return_all=True
        result = solver_func(
            instance_data=instance_data,
            penalty_weights=penalty_weights,
            num_reads=params.get("num_reads", 1024),
            num_sweeps=params.get("num_sweeps", 10000),
            return_all=True,
            verbose=False,
            **extra_kwargs,
        )
        # Extract per-read data
        all_samples = result.get("all_samples", [])
        if all_samples:
            read_indices = list(range(len(all_samples)))
            objectives = [s["energy"] for s in all_samples]
            violation_rates = [s["violation_rate"] for s in all_samples]
        else:
            read_indices = [0]
            objectives = [result["energy"]]
            violation_rates = [result["violation_rate"]]
        
        # Compute SQR
        sqr = result["energy"] / gurobi_miqp if gurobi_miqp != 0 else np.nan
        
        # Print summary
        print(f"  {solver_name} best energy: {result['energy']:.8f}")
        print(f"  {solver_name} SQR: {sqr:.4f}")
        print(f"  {solver_name} feasible: {result['feasible']}, violation_rate: {result['violation_rate']}")
        print(f"  {solver_name} runtime: {result['runtime']:.4f}s")
        
        # Compute ESR/MCR for this solver's QUBO matrix
        qubo_mat = build_qubo_matrix_from_instance(instance_data, penalty_weights)
        miqp_mat = extract_miqp_matrix(instance_data)
        esr_mcr = compute_esr_mcr(miqp_mat, qubo_mat)
        print(f"  {solver_name} ESR: {esr_mcr['ESR']:.4f}, MCR: {esr_mcr['MCR']:.4f}")
        
        # Matrix differences
        diff = compute_matrix_differences(miqp_mat, qubo_mat)
        print(f"  {solver_name} Matrix diff: MSE={diff['MSE']:.6f}, RMSE={diff['RMSE']:.6f}, Frobenius={diff['Frobenius']:.6f}")
        
        # Store for later
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

    # 7a. Objective vs reads
    if sa_eval and sa_eval["read_indices"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_objective_vs_reads(
            ax,
            sa_eval["read_indices"],
            sa_eval["objectives"],
            label="SA",
            color="blue",
        )
        if sqa_eval and sqa_eval["read_indices"]:
            plot_objective_vs_reads(
                ax,
                sqa_eval["read_indices"],
                sqa_eval["objectives"],
                label="SQA",
                color="red",
            )
        ax.axhline(y=gurobi_miqp, color='green', linestyle='--', label='Gurobi Optimum')
        ax.set_title("Objective Value vs Read Index")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "objective_vs_reads.png", dpi=150)
        plt.show()
        plt.close(fig)
        if wandb_run:
            wandb_run.log({"objective_vs_reads": wandb.Image(str(PLOTS_DIR / "objective_vs_reads.png"))})

    # 7b. Violation rate vs reads
    if sa_eval and sa_eval["read_indices"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_violation_vs_reads(
            ax,
            sa_eval["read_indices"],
            sa_eval["violation_rates"],
            label="SA",
            color="blue",
        )
        if sqa_eval and sqa_eval["read_indices"]:
            plot_violation_vs_reads(
                ax,
                sqa_eval["read_indices"],
                sqa_eval["violation_rates"],
                label="SQA",
                color="red",
            )
        ax.set_title("Violation Rate vs Read Index")
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "violation_vs_reads.png", dpi=150)
        plt.show()
        plt.close(fig)
        if wandb_run:
            wandb_run.log({"violation_vs_reads": wandb.Image(str(PLOTS_DIR / "violation_vs_reads.png"))})

    # 7c. Deployment for SA and SQA
    if sa_eval and sa_eval["solution"] is not None:
        plot_jij_deployment(
            instance_data,
            sa_eval["solution"],
            get_penalty_weights(instance, sa_params["lambda_budget"], sa_params["lambda_conn"]),
            save_path=PLOTS_DIR / "sa_deployment.png",
            show_fig=True,
            dpi=150,
        )
        if wandb_run:
            wandb_run.log({"sa_deployment": wandb.Image(str(PLOTS_DIR / "sa_deployment.png"))})
    
    if sqa_eval and sqa_eval["solution"] is not None:
        plot_jij_deployment(
            instance_data,
            sqa_eval["solution"],
            get_penalty_weights(instance, sqa_params["lambda_budget"], sqa_params["lambda_conn"]),
            save_path=PLOTS_DIR / "sqa_deployment.png",
            show_fig=True,
            dpi=150,
        )
        if wandb_run:
            wandb_run.log({"sqa_deployment": wandb.Image(str(PLOTS_DIR / "sqa_deployment.png"))})

    # 7d. QUBO matrix heatmaps
    if sa_eval:
        plot_qubo_matrix_direct(
            sa_eval["qubo_matrix"],
            save_path=PLOTS_DIR / "qubo_matrix_sa.png",
            title=f"SA QUBO Matrix (λ₁={sa_params['lambda_budget']:.4f}, λ₂={sa_params['lambda_conn']:.4f})",
            show_fig=True,
            dpi=150,
        )
        # Also save .npy already done inside plot_qubo_matrix_direct
        if wandb_run:
            wandb_run.log({"qubo_matrix_sa": wandb.Image(str(PLOTS_DIR / "qubo_matrix_sa.png"))})
    
    if sqa_eval:
        plot_qubo_matrix_direct(
            sqa_eval["qubo_matrix"],
            save_path=PLOTS_DIR / "qubo_matrix_sqa.png",
            title=f"SQA QUBO Matrix (λ₁={sqa_params['lambda_budget']:.4f}, λ₂={sqa_params['lambda_conn']:.4f})",
            show_fig=True,
            dpi=150,
        )
        if wandb_run:
            wandb_run.log({"qubo_matrix_sqa": wandb.Image(str(PLOTS_DIR / "qubo_matrix_sqa.png"))})

    # 7e. Optuna plots (if studies exist)
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
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "K": K,
            "N": N,
            "qsum": qsum,
            "gurobi_miqp": gurobi_miqp,
        },
        "gurobi": gurobi_result,
        "sa": {
            "params": sa_params,
            "eval": sa_eval,
        } if sa_eval else None,
        "sqa": {
            "params": sqa_params,
            "eval": sqa_eval,
        } if sqa_eval else None,
    }
    # Save as pickle and JSON
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
    print(f"  Gurobi MIQP: {gurobi_miqp:.8f}")
    if sa_params:
        print(f"  SA best λ₁={sa_params['lambda_budget']:.4f}, λ₂={sa_params['lambda_conn']:.4f}")
        if sa_eval:
            print(f"    SA SQR={sa_eval['sqr']:.4f}, violation={sa_eval['violation_rate']:.1f}")
    if sqa_params:
        print(f"  SQA best λ₁={sqa_params['lambda_budget']:.4f}, λ₂={sqa_params['lambda_conn']:.4f}, trotter={sqa_params.get('trotter', 'N/A')}")
        if sqa_eval:
            print(f"    SQA SQR={sqa_eval['sqr']:.4f}, violation={sqa_eval['violation_rate']:.1f}")
    print("=" * 80)

    # Close W&B
    if wandb_run:
        wandb_run.finish()

    cleanup_tqdm()
    print("🧹 Cleanup complete.")


if __name__ == "__main__":
    main()