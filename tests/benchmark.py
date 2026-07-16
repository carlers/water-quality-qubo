#!/usr/bin/env python3
# =============================================================================
# benchmark.py – Scaling Benchmark for JijModeling Solvers
# =============================================================================
# This script loads the best hyperparameters from tuning.py and runs a scaling
# benchmark across N = [50, 100, 200, 300, 500] (or [50, 100] in test mode).
#
# For each N, it solves:
#   - Gurobi (exact MIQP)
#   - Greedy (utility-based, no penalties)
#   - SA (with tuned λ₁, λ₂)
#   - SQA (with tuned λ₁, λ₂, trotter)
#
# Metrics recorded: SQR, runtime, best objective, violation rate, feasibility rate.
# Per‑read curves (objective and violation vs reads) are generated for SA and SQA.
# Summary plots: Runtime vs N, SQR vs N, SQR vs Runtime.
# Also runs oj.solver_benchmark for TTS curves on the first N.
#
# All plots are displayed inline and logged to W&B if enabled.
#
# Usage:
#   python benchmark.py --mode test                    # Quick run (N=50,100)
#   python benchmark.py --mode full --wandb            # Full run with W&B
#   python benchmark.py --mode full --tune_dir ./results_tuning_full
#   python benchmark.py --mode full --no-resume        # Disable crash recovery
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
from tqdm.notebook import tqdm

# Local imports
from data.synthetic_data import load_master_data, generate_and_save_all
from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
from src.jij_solvers import (
    solve_sa_jij,
    solve_sqa_jij,
    solve_greedy_jij,
    compute_energy,
)
from src.utils import (
    compute_violation_rate,
    print_benchmark_summary,
    safe_save_pickle,
    safe_load_pickle,
    NumpyEncoder,
    cleanup_tqdm,
)
from src.plotting import (
    plot_jij_deployment,
    plot_objective_vs_reads,
    plot_violation_vs_reads,
    plot_scaling_benchmark,
    plot_jij_benchmark_summary,
)
from src.model import compute_pairwise_terms

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
# Helper: Gurobi MIQP solver (same as in tuning.py)
# ============================================================================
def solve_gurobi_miqp(
    instance_data: Dict,
    time_limit: float = 60.0,
    verbose: bool = False,
) -> Dict:
    """Solve MIQP exactly with Gurobi."""
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
# Helper: load/generate instance for a given N
# ============================================================================
def load_instance_for_N(
    N: int,
    seed: int = 42,
    K: int = 5,
    L_c: float = 5.0,
    L_w: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    connectivity_range: float = 8.0,
    current_vector: Tuple[float, float] = (1.0, 0.0),
) -> Dict:
    """
    Load or generate a subset of size N with fixed M_indices.
    Returns instance_data dict with a, Q, neigh, coords, U, M_indices, etc.
    """
    data_dir = Path(f"data_seed{seed}")
    if not data_dir.exists():
        generate_and_save_all(seed=seed, output_dir=str(data_dir))

    coords_master, factors_master, U_master, subsets, meta = load_master_data(str(data_dir))
    if N not in subsets:
        raise ValueError(f"N={N} not in subsets. Available: {list(subsets.keys())}")
    subset = subsets[N]
    coords = subset["coords"]
    U = subset["U"]
    M_indices_original = meta["existing_indices"]
    orig_indices = subset["indices"]
    idx_map = {orig: new for new, orig in enumerate(orig_indices)}
    M_indices = [idx_map[m] for m in M_indices_original if m in idx_map]

    # Compute pairwise terms
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

    return {
        "N": N_total,
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


# ============================================================================
# Helper: solver_benchmark wrapper
# ============================================================================
def run_solver_benchmark(
    instance_data: Dict,
    penalty_weights: Dict[int, float],
    solver_name: str,
    num_reads: int = 1024,
    num_sweeps: int = 10000,
    trotter: Optional[int] = None,
    time_list: List[int] = [200, 500, 1000, 2000, 5000],
) -> Dict:
    """
    Run oj.solver_benchmark for SA or SQA.
    Returns the benchmark results dict.
    """
    try:
        import openjij as oj
    except ImportError:
        return {"error": "openjij not installed"}

    # Build QUBO
    from src.jij_model import build_augmented_model, compile_instance
    model = build_augmented_model()
    model_keys = {"N", "K", "a", "Q", "neigh"}
    filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
    instance = compile_instance(model, filtered_data)
    qubo_dict, _ = instance.to_qubo(penalty_weights=penalty_weights)

    # Define sampler
    if solver_name.lower() == "sa":
        def sampler(time_param, **args):
            nr = args.get('num_reads', num_reads)
            return oj.SASampler().sample_qubo(qubo_dict, num_reads=nr, num_sweeps=time_param, sparse=True)
    elif solver_name.lower() == "sqa":
        if trotter is None:
            trotter = 16
        def sampler(time_param, **args):
            nr = args.get('num_reads', num_reads)
            return oj.SQASampler().sample_qubo(qubo_dict, num_reads=nr, num_sweeps=time_param, trotter=trotter, sparse=True)
    else:
        return {"error": f"Unknown solver: {solver_name}"}

    # Get Gurobi optimum for success probability
    gurobi = solve_gurobi_miqp(instance_data, time_limit=30.0)
    if gurobi["solution"] is None:
        return {"error": "Gurobi optimum not available"}
    optimal_solution = gurobi["solution"].tolist()

    result = oj.solver_benchmark(
        solver=sampler,
        time_list=time_list,
        solutions=[optimal_solution],
        p_r=0.99,
        args={'num_reads': num_reads},
    )
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scaling Benchmark for JijModeling Solvers"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="test",
        help="'test' for N=[50,100], 'full' for N=[50,100,200,300,500]"
    )
    parser.add_argument(
        "--tune_dir",
        type=str,
        default=None,
        help="Directory containing best_sa_params.json and best_sqa_params.json from tuning"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable crash recovery (start fresh)"
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
    args = parser.parse_args()

    # Determine tune_dir
    if args.tune_dir is None:
        # Look for tuning results in default location
        possible_dir = Path(f"results_tuning_{args.mode}")
        if possible_dir.exists():
            args.tune_dir = str(possible_dir)
        else:
            raise ValueError(f"No tuning results found. Please specify --tune_dir.")
    tune_dir = Path(args.tune_dir)
    if not tune_dir.exists():
        raise ValueError(f"Tune directory not found: {tune_dir}")

    # Load best parameters
    sa_params_path = tune_dir / "best_sa_params.json"
    sqa_params_path = tune_dir / "best_sqa_params.json"
    if sa_params_path.exists():
        with open(sa_params_path, "r") as f:
            sa_params = json.load(f)
        print(f"✅ Loaded SA params from {sa_params_path}")
    else:
        sa_params = None
        print("⚠️ No SA params found; SA will be skipped.")
    if sqa_params_path.exists():
        with open(sqa_params_path, "r") as f:
            sqa_params = json.load(f)
        print(f"✅ Loaded SQA params from {sqa_params_path}")
    else:
        sqa_params = None
        print("⚠️ No SQA params found; SQA will be skipped.")

    # Define N list
    if args.mode == "test":
        N_list = [50, 100]
    else:
        N_list = [50, 100, 200, 300, 500]

    # Output directory
    SAVE_DIR = Path(f"results_benchmark_{args.mode}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = SAVE_DIR / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    READ_CURVES_DIR = PLOTS_DIR / "read_curves"
    READ_CURVES_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("🚀 JijModeling Scaling Benchmark")
    print("=" * 80)
    print(f"  Mode: {args.mode.upper()}")
    print(f"  N list: {N_list}")
    print(f"  K: {args.K}")
    print(f"  Seed: {args.seed}")
    print(f"  Tune dir: {tune_dir}")
    print(f"  W&B: {args.wandb}")
    print(f"  Resume: {not args.no_resume}")
    print("=" * 80)

    # W&B init
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project="wqm-placement-optimization",
                config={
                    "mode": args.mode,
                    "seed": args.seed,
                    "K": args.K,
                    "N_list": N_list,
                    "sa_params": sa_params,
                    "sqa_params": sqa_params,
                },
                name=f"benchmark_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            wandb_run = wandb
            print("✅ W&B logging enabled.")
        except Exception as e:
            print(f"  ⚠️ W&B init failed: {e}")

    # -------------------------------------------------------------------------
    # Progress management
    # -------------------------------------------------------------------------
    progress_file = SAVE_DIR / "completed_N.pkl"
    if args.no_resume:
        completed = set()
        all_results = []
    else:
        completed = safe_load_pickle(progress_file, set())
        all_results = safe_load_pickle(SAVE_DIR / "benchmark_results.pkl", [])
        print(f"Loaded {len(all_results)} existing results, {len(completed)} completed N.")

    # Global progress bar
    pbar = tqdm(total=len(N_list), desc="Benchmark progress", position=0, leave=True)

    # Store per‑N summaries for summary plots
    per_N_summary = []

    # -------------------------------------------------------------------------
    # Main loop over N
    # -------------------------------------------------------------------------
    for N in N_list:
        if N in completed and not args.no_resume:
            print(f"\n⏩ N={N} already completed. Skipping.")
            pbar.update(1)
            continue

        print("\n" + "=" * 80)
        print(f"📊 Benchmarking N={N}")
        print("=" * 80)

        # 1. Load instance
        instance_data = load_instance_for_N(N, seed=args.seed, K=args.K)

        # 2. Gurobi baseline
        print("  Gurobi baseline...")
        gurobi_result = solve_gurobi_miqp(instance_data, time_limit=60.0)
        if gurobi_result["solution"] is None:
            print(f"    ⚠️ Gurobi failed: {gurobi_result['status']}. Skipping N={N}.")
            completed.add(N)
            safe_save_pickle(progress_file, completed, verbose=False)
            pbar.update(1)
            continue
        print(f"    Gurobi energy: {gurobi_result['energy']:.8f}, feasible={gurobi_result['feasible']}")
        gurobi_energy = gurobi_result["energy"]

        # 3. Greedy baseline
        print("  Greedy baseline...")
        greedy_result = solve_greedy_jij(instance_data, verbose=False)
        greedy_sqr = greedy_result["energy"] / gurobi_energy if gurobi_energy != 0 else np.nan
        print(f"    Greedy energy: {greedy_result['energy']:.8f}, SQR={greedy_sqr:.4f}")

        # 4. SA solver (if params available)
        sa_result = None
        if sa_params is not None:
            print("  SA solver...")
            model = build_augmented_model()
            model_keys = {"N", "K", "a", "Q", "neigh"}
            filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
            instance = compile_instance(model, filtered_data)
            penalty_weights = get_penalty_weights(instance, sa_params["lambda_budget"], sa_params["lambda_conn"])
            sa_result = solve_sa_jij(
                instance_data,
                penalty_weights,
                num_reads=sa_params.get("num_reads", 1024),
                num_sweeps=sa_params.get("num_sweeps", 10000),
                return_all=True,
                verbose=False,
            )
            sa_sqr = sa_result["energy"] / gurobi_energy if gurobi_energy != 0 else np.nan
            print(f"    SA energy: {sa_result['energy']:.8f}, SQR={sa_sqr:.4f}, feasible={sa_result['feasible']}")
        else:
            sa_sqr = np.nan

        # 5. SQA solver (if params available)
        sqa_result = None
        if sqa_params is not None:
            print("  SQA solver...")
            model = build_augmented_model()
            model_keys = {"N", "K", "a", "Q", "neigh"}
            filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
            instance = compile_instance(model, filtered_data)
            penalty_weights = get_penalty_weights(instance, sqa_params["lambda_budget"], sqa_params["lambda_conn"])
            trotter = sqa_params.get("trotter", 16)
            sqa_result = solve_sqa_jij(
                instance_data,
                penalty_weights,
                num_reads=sqa_params.get("num_reads", 1024),
                num_sweeps=sqa_params.get("num_sweeps", 10000),
                trotter=trotter,
                return_all=True,
                verbose=False,
            )
            sqa_sqr = sqa_result["energy"] / gurobi_energy if gurobi_energy != 0 else np.nan
            print(f"    SQA energy: {sqa_result['energy']:.8f}, SQR={sqa_sqr:.4f}, feasible={sqa_result['feasible']}")
        else:
            sqa_sqr = np.nan

        # 6. Per‑N read curves (SA and SQA)
        fig_obj, ax_obj = plt.subplots(figsize=(10, 6))
        fig_viol, ax_viol = plt.subplots(figsize=(10, 6))

        if sa_result and sa_result.get("all_samples"):
            samples = sa_result["all_samples"]
            read_idx = list(range(len(samples)))
            obj_vals = [s["energy"] for s in samples]
            viol_vals = [s["violation_rate"] for s in samples]
            plot_objective_vs_reads(ax_obj, read_idx, obj_vals, label="SA", color="blue")
            plot_violation_vs_reads(ax_viol, read_idx, viol_vals, label="SA", color="blue")

        if sqa_result and sqa_result.get("all_samples"):
            samples = sqa_result["all_samples"]
            read_idx = list(range(len(samples)))
            obj_vals = [s["energy"] for s in samples]
            viol_vals = [s["violation_rate"] for s in samples]
            plot_objective_vs_reads(ax_obj, read_idx, obj_vals, label="SQA", color="red")
            plot_violation_vs_reads(ax_viol, read_idx, viol_vals, label="SQA", color="red")

        # Add Gurobi baseline to objective plot
        if gurobi_result["solution"] is not None:
            ax_obj.axhline(y=gurobi_energy, color='green', linestyle='--', label='Gurobi Optimum')
        ax_obj.set_title(f"Objective vs Read Index (N={N})")
        ax_obj.legend()
        fig_obj.tight_layout()
        fig_obj.savefig(READ_CURVES_DIR / f"objective_vs_reads_N{N}.png", dpi=150)
        plt.show(fig_obj)
        plt.close(fig_obj)

        ax_viol.set_title(f"Violation Rate vs Read Index (N={N})")
        ax_viol.legend()
        fig_viol.tight_layout()
        fig_viol.savefig(READ_CURVES_DIR / f"violation_vs_reads_N{N}.png", dpi=150)
        plt.show(fig_viol)
        plt.close(fig_viol)

        # Log read curves to W&B
        if wandb_run:
            wandb_run.log({
                f"objective_vs_reads_N{N}": wandb.Image(str(READ_CURVES_DIR / f"objective_vs_reads_N{N}.png")),
                f"violation_vs_reads_N{N}": wandb.Image(str(READ_CURVES_DIR / f"violation_vs_reads_N{N}.png")),
            })

        # 7. Store per‑N summary
        entry = {
            "N": N,
            "gurobi": {
                "energy": gurobi_result["energy"],
                "runtime": gurobi_result["runtime"],
                "feasible": gurobi_result["feasible"],
                "violation_rate": gurobi_result["violation_rate"],
            },
            "greedy": {
                "energy": greedy_result["energy"],
                "runtime": greedy_result["runtime"],
                "feasible": greedy_result["feasible"],
                "violation_rate": greedy_result["violation_rate"],
                "sqr": greedy_sqr,
            },
        }
        if sa_result:
            entry["sa"] = {
                "energy": sa_result["energy"],
                "runtime": sa_result["runtime"],
                "feasible": sa_result["feasible"],
                "violation_rate": sa_result["violation_rate"],
                "sqr": sa_sqr,
                "num_samples": len(sa_result.get("all_samples", [])),
            }
        if sqa_result:
            entry["sqa"] = {
                "energy": sqa_result["energy"],
                "runtime": sqa_result["runtime"],
                "feasible": sqa_result["feasible"],
                "violation_rate": sqa_result["violation_rate"],
                "sqr": sqa_sqr,
                "num_samples": len(sqa_result.get("all_samples", [])),
                "trotter_used": trotter if sqa_params else None,
            }
        per_N_summary.append(entry)
        all_results.append(entry)

        # 8. Save progress
        completed.add(N)
        safe_save_pickle(progress_file, completed, verbose=False)
        safe_save_pickle(SAVE_DIR / "benchmark_results.pkl", all_results, verbose=False)

        # 9. Run solver_benchmark on this N (only for N=50 to save time)
        if N == 50 and sa_params is not None:
            print("  Running solver_benchmark for SA...")
            model = build_augmented_model()
            model_keys = {"N", "K", "a", "Q", "neigh"}
            filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
            instance = compile_instance(model, filtered_data)
            penalty_weights = get_penalty_weights(instance, sa_params["lambda_budget"], sa_params["lambda_conn"])
            bench_sa = run_solver_benchmark(
                instance_data,
                penalty_weights,
                "SA",
                num_reads=sa_params.get("num_reads", 1024),
                num_sweeps=sa_params.get("num_sweeps", 10000),
            )
            if "error" not in bench_sa:
                with open(SAVE_DIR / "solver_benchmark_sa.json", "w") as f:
                    json.dump(bench_sa, f, indent=2, cls=NumpyEncoder)
                # Plot TTS curves (we'll use a separate helper in plotting)
                # For now, just log the JSON
                if wandb_run:
                    wandb_run.log({"solver_benchmark_sa": wandb.Table(dataframe=pd.DataFrame(bench_sa))})

        if N == 50 and sqa_params is not None:
            print("  Running solver_benchmark for SQA...")
            model = build_augmented_model()
            model_keys = {"N", "K", "a", "Q", "neigh"}
            filtered_data = {k: v for k, v in instance_data.items() if k in model_keys}
            instance = compile_instance(model, filtered_data)
            penalty_weights = get_penalty_weights(instance, sqa_params["lambda_budget"], sqa_params["lambda_conn"])
            trotter = sqa_params.get("trotter", 16)
            bench_sqa = run_solver_benchmark(
                instance_data,
                penalty_weights,
                "SQA",
                num_reads=sqa_params.get("num_reads", 1024),
                num_sweeps=sqa_params.get("num_sweeps", 10000),
                trotter=trotter,
            )
            if "error" not in bench_sqa:
                with open(SAVE_DIR / "solver_benchmark_sqa.json", "w") as f:
                    json.dump(bench_sqa, f, indent=2, cls=NumpyEncoder)
                if wandb_run:
                    wandb_run.log({"solver_benchmark_sqa": wandb.Table(dataframe=pd.DataFrame(bench_sqa))})

        pbar.update(1)

    pbar.close()

    # -------------------------------------------------------------------------
    # Generate summary plots
    # -------------------------------------------------------------------------
    if per_N_summary:
        # Build DataFrame for summary plots
        rows = []
        for entry in per_N_summary:
            N = entry["N"]
            for solver in ["gurobi", "greedy", "sa", "sqa"]:
                if solver in entry:
                    data = entry[solver]
                    rows.append({
                        "N": N,
                        "solver": solver.capitalize(),
                        "sqr": data.get("sqr", np.nan),
                        "runtime": data.get("runtime", np.nan),
                        "feasible": data.get("feasible", False),
                        "violation_rate": data.get("violation_rate", 1.0),
                        "energy": data.get("energy", np.nan),
                    })
        df = pd.DataFrame(rows)

        # Summary plots using existing functions
        plot_scaling_benchmark(df, save_dir=PLOTS_DIR, show_fig=True)
        plot_jij_benchmark_summary(df, save_dir=PLOTS_DIR, show_fig=True)

        # Also create a custom SQR vs Runtime plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for solver in df["solver"].unique():
            sub = df[df["solver"] == solver]
            ax.scatter(sub["runtime"], sub["sqr"], label=solver, s=80, alpha=0.7)
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("SQR")
        ax.set_title("SQR vs Runtime (All N)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "sqr_vs_runtime_summary.png", dpi=150)
        plt.show(fig)
        plt.close(fig)

        # Log summary plots to W&B
        if wandb_run:
            for fname in ["sqr_vs_N.png", "runtime_vs_N.png", "feasibility_vs_N.png", "sqr_vs_runtime_summary.png"]:
                p = PLOTS_DIR / fname
                if p.exists():
                    wandb_run.log({f"summary_{fname}": wandb.Image(str(p))})

        # Print final benchmark summary
        print_benchmark_summary({entry["N"]: entry for entry in per_N_summary}, title="Final Benchmark Summary")

        # Save full DataFrame
        df.to_csv(SAVE_DIR / "benchmark_full_results.csv", index=False)

    # -------------------------------------------------------------------------
    # Finalize
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {SAVE_DIR.resolve()}")
    print("  - benchmark_results.pkl, benchmark_results.csv")
    print("  - plots/ (summary plots, read curves)")
    print("  - solver_benchmark_*.json (if run)")
    print("=" * 80)

    if wandb_run:
        wandb_run.finish()

    cleanup_tqdm()
    print("🧹 Cleanup complete.")


if __name__ == "__main__":
    main()