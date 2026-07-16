#!/usr/bin/env python3
# =============================================================================
# benchmark.py – Main Driver for JijModeling Pipeline
# =============================================================================
# This script orchestrates the entire JijModeling-based benchmarking pipeline:
#   1. Generate synthetic instances
#   2. Tune SA and SQA hyperparameters (Optuna) on a fixed instance
#   3. Visualise deployment and QUBO matrix for tuned models
#   4. Run scaling benchmark (N=20,30,50; K=5,10; 3 seeds)
#   5. Generate TTS/success probability/residual energy curves (solver_benchmark)
#   6. Save results and plots
#
# Usage: python benchmark.py [--mode test|full] [--wandb]
# =============================================================================

import sys
import os
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our JijModeling modules
from src.jij_model import build_augmented_model, compile_instance, get_penalty_weights
from src.jij_solvers import solve_sa_jij, solve_sqa_jij, decode_solution, compute_energy
from src.jij_optuna import tune_sa, tune_sqa, compute_qsum
from src.utils import (
    print_jij_tuning_summary,
    print_jij_benchmark_summary,
    print_jij_single_run,
    save_jij_results,
    load_jij_results,
    safe_save_pickle,
    safe_load_pickle,
    NumpyEncoder,
    cleanup_tqdm,
)
from src.plotting import (
    plot_jij_deployment,
    plot_jij_qubo_matrix,
    plot_jij_scaling_benchmark,
    plot_jij_benchmark_summary,
)

warnings.filterwarnings('ignore')

# Optional W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Optional Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("⚠️ Gurobi not available; exact baseline will be skipped.")

# -----------------------------------------------------------------------------
# 1. Data generation and helper functions
# -----------------------------------------------------------------------------
def generate_instance(N: int, K: int, seed: int, D_max: float = 8.0) -> Dict:
    """Generate synthetic instance with coords, utility, pairwise, and neighbor matrix."""
    np.random.seed(seed)
    coords = np.random.rand(N, 2) * 50.0
    U = np.random.rand(N)
    L_c = 5.0
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    R = np.maximum(0, 1 - dist / L_c)
    Q = R
    a = -U
    neigh = (dist <= D_max).astype(int)
    np.fill_diagonal(neigh, 0)
    return {
        "a": a,
        "Q": Q,
        "coords": coords,
        "U": U,
        "N": N,
        "K": K,
        "seed": seed,
        "neigh": neigh,
        "D_max": D_max,
    }


def solve_gurobi_exact(instance_data: Dict, time_limit: float = 30.0) -> Dict:
    """Solve the MIQP exactly using Gurobi."""
    if not GUROBI_AVAILABLE:
        return {"solution": None, "energy": np.nan, "runtime": np.nan, "status": "Gurobi not available"}
    a, Q, neigh, K, N = (
        instance_data["a"],
        instance_data["Q"],
        instance_data["neigh"],
        instance_data["K"],
        instance_data["N"],
    )
    start = time.perf_counter()
    try:
        model = gp.Model("WQM_Aug")
        model.setParam("OutputFlag", 0)
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
        model.addConstr(gp.quicksum(x[i] for i in range(N)) == K, "budget")
        for i in range(N):
            model.addConstr(x[i] <= gp.quicksum(neigh[i, j] * x[j] for j in range(N)), f"conn_{i}")
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            x_sol = np.array([x[i].X for i in range(N)])
            energy = compute_energy(x_sol, a, Q)
            runtime = time.perf_counter() - start
            return {"solution": x_sol, "energy": energy, "runtime": runtime, "status": "optimal"}
        else:
            runtime = time.perf_counter() - start
            return {"solution": None, "energy": np.nan, "runtime": runtime, "status": f"not optimal ({model.Status})"}
    except Exception as e:
        runtime = time.perf_counter() - start
        return {"solution": None, "energy": np.nan, "runtime": runtime, "status": f"error: {e}"}


def solve_greedy_utility(instance_data: Dict, verbose: bool = False) -> Dict:
    """Greedy: select K sites with highest utility (ignores connectivity)."""
    a = instance_data["a"]
    Q = instance_data["Q"]
    neigh = instance_data["neigh"]
    K = instance_data["K"]
    N = instance_data["N"]
    start = time.perf_counter()
    indices = np.argsort(a)[:K]  # smallest a = highest utility
    x_sol = np.zeros(N)
    x_sol[indices] = 1
    energy = compute_energy(x_sol, a, Q)
    runtime = time.perf_counter() - start
    # Check feasibility (for reporting)
    selected = indices
    budget_ok = (len(selected) == K)
    conn_ok = all(np.any(neigh[i, selected] == 1) for i in selected) if len(selected) > 0 else True
    feasible = budget_ok and conn_ok
    if verbose:
        print(f"    Greedy: energy={energy:.6f}, runtime={runtime:.4f}s, feasible={feasible}")
    return {"solution": x_sol, "energy": energy, "runtime": runtime, "status": "feasible" if feasible else "infeasible", "feasible": feasible}


def run_tuning(tune_instance: Dict, n_trials_sa: int = 20, n_trials_sqa: int = 15, timeout: int = 300) -> Tuple[Dict, Dict]:
    """
    Run Optuna tuning for SA and SQA on the given instance.
    Returns (sa_params, sqa_params).
    """
    print("\n" + "=" * 80)
    print("🔬 Tuning SA hyperparameters...")
    print("=" * 80)
    sa_params = tune_sa(tune_instance, n_trials=n_trials_sa, timeout=timeout, verbose=True)

    print("\n" + "=" * 80)
    print("🔬 Tuning SQA hyperparameters...")
    print("=" * 80)
    sqa_params = tune_sqa(tune_instance, n_trials=n_trials_sqa, timeout=timeout, verbose=True)

    return sa_params, sqa_params


def run_benchmark(
    N_list: List[int],
    K_list: List[int],
    seeds: List[int],
    sa_params: Dict,
    sqa_params: Dict,
    save_dir: Path,
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run scaling benchmark with crash recovery.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load progress if resuming
    if resume:
        results, completed = load_jij_results("benchmark", save_dir)
        print(f"Loaded {len(results)} existing results.")
    else:
        results = []
        completed = set()

    # Define all configs
    configs = []
    for N in N_list:
        for K in K_list:
            for seed in seeds:
                configs.append((N, K, seed))

    total = len(configs)
    done = 0
    for N, K, seed in configs:
        key = (N, K, seed)
        if key in completed:
            done += 1
            continue

        print(f"\nProgress: {done+1}/{total}  (N={N}, K={K}, seed={seed})")
        inst = generate_instance(N, K, seed)

        # Gurobi baseline
        gurobi = solve_gurobi_exact(inst)
        if np.isnan(gurobi["energy"]):
            print(f"  ⚠️ Gurobi failed: {gurobi['status']}. Skipping instance.")
            # Still mark as completed to avoid infinite retries
            completed.add(key)
            save_jij_results(results, completed, "benchmark", save_dir)
            done += 1
            continue
        print(f"  Gurobi: energy={gurobi['energy']:.6f}, runtime={gurobi['runtime']:.4f}s")
        e_gurobi = gurobi["energy"]

        # Greedy
        greedy = solve_greedy_utility(inst, verbose=verbose)

        # SA
        model = build_augmented_model()
        instance = compile_instance(model, inst)
        penalty_ids = {c.name: c.id for c in instance.constraints}
        sa_penalty_weights = {
            penalty_ids["budget"]: sa_params["lambda_budget"],
            penalty_ids["connectivity"]: sa_params["lambda_conn"],
        }
        sa = solve_sa_jij(
            inst,
            sa_penalty_weights,
            num_reads=sa_params.get("num_reads", 100),
            num_sweeps=sa_params.get("num_sweeps", 1000),
            verbose=verbose,
        )

        # SQA
        sqa_penalty_weights = {
            penalty_ids["budget"]: sqa_params["lambda_budget"],
            penalty_ids["connectivity"]: sqa_params["lambda_conn"],
        }
        sqa = solve_sqa_jij(
            inst,
            sqa_penalty_weights,
            num_reads=sqa_params.get("num_reads", 100),
            num_sweeps=sqa_params.get("num_sweeps", 1000),
            trotter=sqa_params.get("trotter", 16),
            verbose=verbose,
        )

        # Collect results for all solvers
        for solver_name, res in [
            ("Gurobi", gurobi),
            ("Greedy", greedy),
            ("SA", sa),
            ("SQA", sqa),
        ]:
            if res["solution"] is None:
                print(f"  {solver_name}: no solution")
                continue
            sqr = res["energy"] / e_gurobi if e_gurobi != 0 else np.nan
            results.append({
                "seed": seed,
                "N": N,
                "K": K,
                "solver": solver_name,
                "energy": res["energy"],
                "runtime": res["runtime"],
                "feasible": res.get("feasible", False),
                "sqr": sqr,
                "status": res.get("status", ""),
            })

        # Mark completed and save progress
        completed.add(key)
        save_jij_results(results, completed, "benchmark", save_dir)
        done += 1

    df = pd.DataFrame(results)
    df.to_csv(save_dir / "benchmark_results.csv", index=False)
    return df


def run_solver_benchmark(
    instance_data: Dict,
    sa_params: Dict,
    sqa_params: Dict,
    time_list: List[int] = [200, 500, 1000, 2000, 5000],
    save_dir: Path = None,
) -> None:
    """
    Run oj.solver_benchmark for SA and SQA with tuned parameters.
    Saves JSON and plots.
    """
    import openjij as oj

    if save_dir is None:
        save_dir = Path(".")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    N = instance_data["N"]
    K = instance_data["K"]
    a = instance_data["a"]
    Q = instance_data["Q"]
    neigh = instance_data["neigh"]
    model = build_augmented_model()
    data = {"N": N, "K": K, "a": a.tolist(), "Q": Q.tolist(), "neigh": neigh.tolist()}
    instance = compile_instance(model, data)

    # Get Gurobi optimum for success probability
    gurobi = solve_gurobi_exact(instance_data)
    if np.isnan(gurobi["energy"]):
        print("⚠️ Cannot run solver_benchmark without Gurobi optimal solution.")
        return
    optimal_solution = gurobi["solution"]

    # Penalty dicts
    penalty_ids = {c.name: c.id for c in instance.constraints}
    sa_penalty_weights = {
        penalty_ids["budget"]: sa_params["lambda_budget"],
        penalty_ids["connectivity"]: sa_params["lambda_conn"],
    }
    sqa_penalty_weights = {
        penalty_ids["budget"]: sqa_params["lambda_budget"],
        penalty_ids["connectivity"]: sqa_params["lambda_conn"],
    }

    # Build QUBO for SA and SQA (separately because penalties differ)
    qubo_sa, _ = instance.to_qubo(penalty_weights=sa_penalty_weights)
    qubo_sqa, _ = instance.to_qubo(penalty_weights=sqa_penalty_weights)

    num_reads_sa = sa_params.get("num_reads", 100)
    num_reads_sqa = sqa_params.get("num_reads", 100)
    trotter = sqa_params.get("trotter", 16)

    def sa_sampler(time_param, **args):
        nr = args.get('num_reads', num_reads_sa)
        return oj.SASampler().sample_qubo(qubo_sa, num_reads=nr, num_sweeps=time_param, sparse=True)

    def sqa_sampler(time_param, **args):
        nr = args.get('num_reads', num_reads_sqa)
        return oj.SQASampler().sample_qubo(qubo_sqa, num_reads=nr, num_sweeps=time_param, trotter=trotter, sparse=True)

    correct_state = optimal_solution.tolist()

    print("\n🔬 Running solver_benchmark for SA...")
    result_sa = oj.solver_benchmark(
        solver=sa_sampler,
        time_list=time_list,
        solutions=[correct_state],
        p_r=0.99,
        args={'num_reads': num_reads_sa},
    )
    print("🔬 Running solver_benchmark for SQA...")
    result_sqa = oj.solver_benchmark(
        solver=sqa_sampler,
        time_list=time_list,
        solutions=[correct_state],
        p_r=0.99,
        args={'num_reads': num_reads_sqa},
    )

    bench_results = {"SA": result_sa, "SQA": result_sqa}

    # Save JSON
    with open(save_dir / "solver_benchmark_results.json", "w") as f:
        json.dump(bench_results, f, indent=2, cls=NumpyEncoder)

    # Plot
    fig, (axL, axC, axR) = plt.subplots(ncols=3, figsize=(15, 3))
    plt.subplots_adjust(wspace=0.4)
    fontsize = 10
    colors = {'SA': 'blue', 'SQA': 'red'}

    for solver, res in bench_results.items():
        color = colors[solver]
        time_vals = np.array(res['time'])
        tts = np.array(res['tts'])
        mask = np.isfinite(tts)
        if not np.any(mask):
            print(f"⚠️ No finite TTS for {solver}; skipping TTS plot.")
            continue
        time_f = time_vals[mask]
        tts_f = tts[mask]

        axL.plot(time_f, tts_f, '-o', color=color, label=solver)
        if 'se_lower_tts' in res and 'se_upper_tts' in res:
            lower = np.array(res['se_lower_tts'])[mask]
            upper = np.array(res['se_upper_tts'])[mask]
            axL.errorbar(time_f, tts_f, yerr=(lower, upper),
                         capsize=5, fmt='o', markersize=5, ecolor='black',
                         markeredgecolor="black", color='w', alpha=0.5)

        sp = np.array(res['success_prob'])[mask]
        axC.plot(time_f, sp, '-o', color=color)
        if 'se_success_prob' in res:
            se_sp = np.array(res['se_success_prob'])[mask]
            axC.errorbar(time_f, sp, yerr=se_sp, capsize=5, fmt='o',
                         markersize=5, ecolor='black', markeredgecolor="black",
                         color='w', alpha=0.5)

        re = np.array(res['residual_energy'])[mask]
        axR.plot(time_f, re, '-o', color=color)
        if 'se_residual_energy' in res:
            se_re = np.array(res['se_residual_energy'])[mask]
            axR.errorbar(time_f, re, yerr=se_re, capsize=5, fmt='o',
                         markersize=5, ecolor='black', markeredgecolor="black",
                         color='w', alpha=0.5)

    axL.set_xlabel('Annealing time (s)', fontsize=fontsize)
    axL.set_ylabel('TTS (s)', fontsize=fontsize)
    axL.set_yscale("log")
    axL.legend()

    axC.set_xlabel('Annealing time (s)', fontsize=fontsize)
    axC.set_ylabel('Success probability', fontsize=fontsize)

    axR.set_xlabel('Annealing time (s)', fontsize=fontsize)
    axR.set_ylabel('Residual energy', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(save_dir / "solver_benchmark_curves.png", dpi=150)
    plt.show()
    plt.close(fig)

    print("📁 solver_benchmark results saved.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="JijModeling Benchmark Pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="'test' for quick run, 'full' for full grid")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no-resume", action="store_true", help="Disable crash recovery")
    args = parser.parse_args()

    # Output directory
    SAVE_DIR = Path(f"results_jij_{args.mode}")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="wqm-placement-optimization", config={"mode": args.mode})
        print("✅ W&B logging enabled.")

    # -------------------------------------------------------------------------
    # 1. Tune on fixed instance (N=20, K=5, seed=42)
    # -------------------------------------------------------------------------
    tune_inst = generate_instance(20, 5, 42)
    qsum = compute_qsum(tune_inst)
    print(f"Qsum = {qsum:.4f} (used for penalty scaling)")

    sa_params, sqa_params = run_tuning(
        tune_inst,
        n_trials_sa=20 if args.mode == "full" else 8,
        n_trials_sqa=15 if args.mode == "full" else 6,
        timeout=600 if args.mode == "full" else 120,
    )

    # Print tuning summaries
    # We need the best values (energies) – we don't have them directly, so we'll re-evaluate.
    # For simplicity, we just print the parameters.
    print("\n" + "=" * 80)
    print("🏆 TUNING COMPLETE")
    print("=" * 80)
    print("SA parameters:")
    for k, v in sa_params.items():
        print(f"  {k}: {v}")
    print("\nSQA parameters:")
    for k, v in sqa_params.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # 2. Visualise deployment and QUBO matrix for tuned models
    # -------------------------------------------------------------------------
    print("\n📊 Visualising deployment and QUBO matrix...")
    # Build penalty dicts for visualisation
    model = build_augmented_model()
    instance = compile_instance(model, tune_inst)
    penalty_ids = {c.name: c.id for c in instance.constraints}

    sa_penalty_weights = {
        penalty_ids["budget"]: sa_params["lambda_budget"],
        penalty_ids["connectivity"]: sa_params["lambda_conn"],
    }
    sqa_penalty_weights = {
        penalty_ids["budget"]: sqa_params["lambda_budget"],
        penalty_ids["connectivity"]: sqa_params["lambda_conn"],
    }

    # Solve with SA and SQA to get solutions for visualisation
    sa_res = solve_sa_jij(
        tune_inst,
        sa_penalty_weights,
        num_reads=sa_params.get("num_reads", 100),
        num_sweeps=sa_params.get("num_sweeps", 1000),
        verbose=True,
    )
    sqa_res = solve_sqa_jij(
        tune_inst,
        sqa_penalty_weights,
        num_reads=sqa_params.get("num_reads", 100),
        num_sweeps=sqa_params.get("num_sweeps", 1000),
        trotter=sqa_params.get("trotter", 16),
        verbose=True,
    )

    # Deployment plots
    plot_jij_deployment(tune_inst, sa_res["solution"], sa_penalty_weights,
                        save_path=SAVE_DIR / "deployment_sa.png", show_fig=True)
    plot_jij_deployment(tune_inst, sqa_res["solution"], sqa_penalty_weights,
                        save_path=SAVE_DIR / "deployment_sqa.png", show_fig=True)

    # QUBO matrix plots
    plot_jij_qubo_matrix(tune_inst, sa_penalty_weights,
                         save_path=SAVE_DIR / "qubo_matrix_sa.png", show_fig=True)
    plot_jij_qubo_matrix(tune_inst, sqa_penalty_weights,
                         save_path=SAVE_DIR / "qubo_matrix_sqa.png", show_fig=True)

    # -------------------------------------------------------------------------
    # 3. Run scaling benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🚀 Running scaling benchmark...")
    print("=" * 80)

    if args.mode == "test":
        N_list = [20]
        K_list = [5]
        seeds = [42, 43]
    else:
        N_list = [20, 30, 50]
        K_list = [5, 10]
        seeds = [42, 43, 44]

    df = run_benchmark(
        N_list=N_list,
        K_list=K_list,
        seeds=seeds,
        sa_params=sa_params,
        sqa_params=sqa_params,
        save_dir=SAVE_DIR,
        resume=not args.no_resume,
        verbose=True,
    )

    if df.empty:
        print("⚠️ No benchmark data collected. Exiting.")
        return

    print_jij_benchmark_summary(df, title="Scaling Benchmark")

    # Generate scaling plots
    plot_jij_scaling_benchmark(df, save_dir=SAVE_DIR / "scaling_plots", show_fig=True)
    plot_jij_benchmark_summary(df, save_dir=SAVE_DIR / "summary_plots", show_fig=True)

    # -------------------------------------------------------------------------
    # 4. Run solver_benchmark on a representative instance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔬 Running solver_benchmark for TTS curves...")
    print("=" * 80)
    run_solver_benchmark(
        tune_inst,
        sa_params,
        sqa_params,
        time_list=[200, 500, 1000, 2000, 5000],
        save_dir=SAVE_DIR,
    )

    # -------------------------------------------------------------------------
    # 5. W&B final logging
    # -------------------------------------------------------------------------
    if args.wandb and WANDB_AVAILABLE:
        wandb.log({
            "sa_params": sa_params,
            "sqa_params": sqa_params,
            "benchmark_results": wandb.Table(dataframe=df),
        })
        # Log plots
        for fname in SAVE_DIR.glob("*.png"):
            wandb.log({fname.stem: wandb.Image(str(fname))})
        wandb.finish()

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"📁 Results saved to: {SAVE_DIR.resolve()}")
    print("  - benchmark_results.csv")
    print("  - deployment_sa.png, deployment_sqa.png")
    print("  - qubo_matrix_sa.png, qubo_matrix_sqa.png")
    print("  - scaling_plots/ (SQR, runtime, feasibility vs N)")
    print("  - summary_plots/ (boxplots, scatter, bar)")
    print("  - solver_benchmark_results.json, solver_benchmark_curves.png")
    print("=" * 80)

    cleanup_tqdm()


if __name__ == "__main__":
    main()