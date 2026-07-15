"""
src/experiment.py

Level 2 orchestration layer for the water quality monitoring QUBO project.

This module provides reusable functions for:
    1. evaluate_qubo      – Builds QUBO, computes ESR/MCR, runs SA, returns raw SQR.
    2. make_objective     – Returns an Optuna objective callable.
    3. run_optuna_study   – Creates/loads an Optuna study with RDBStorage.
    4. validate_study     – Validates top trials, computes Spearman correlation.
    5. run_ablation_experiment – Full tuning + validation pipeline for experiments.

All functions import only from Level 1 (model, solvers, utils, environment, plotting).

CRITICAL: SQR is now computed using the RAW MIQP energy (pairwise_raw), not the
normalized energy. This ensures a fair comparison to the Gurobi baseline.
UPDATED: nan-safe aggregations, load_if_exists=True, division-by-zero guard.
"""

import time
import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import scipy.stats as stats

# Level 1 imports
from src.model import build_qubo
from src.solvers import solve_sa, build_schedule
from src.utils import (
    safe_save_pickle,
    safe_load_pickle,
    NumpyEncoder,
    print_tuning_summary,
    print_validation_summary,
    print_global_summary,
)
from src.environment import get_environment
from src.plotting import (
    save_optuna_plots,
    plot_optuna_learning_curve,
    plot_optuna_learning_curve_enhanced,
    plot_performance_scatter,
    plot_qubo_matrix_heatmap,
    plot_deployment_with_qubo,
    plot_tuning_vs_validation_deployment,
)

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# Helper: Build full Q matrix from h, J
# -----------------------------------------------------------------------------

def _build_full_Q(h: Dict[int, float], J: Dict[Tuple[int, int], float], N: int) -> np.ndarray:
    """Build symmetric full QUBO matrix from linear (h) and quadratic (J) coefficients."""
    Q = np.zeros((N, N))
    for i, val in h.items():
        if i < N:
            Q[i, i] += val
    for (i, j), val in J.items():
        if i >= N or j >= N:
            continue
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += val / 2.0
            Q[j, i] += val / 2.0
    return Q


# -----------------------------------------------------------------------------
# Helper: Compute raw MIQP energy from a solution
# -----------------------------------------------------------------------------

def _compute_raw_energy(
    solution: np.ndarray,
    pairwise_raw: Dict,
) -> float:
    """
    Compute the raw MIQP energy (no penalties) for a given solution.
    Uses the original (un‑normalized) linear and quadratic coefficients.
    """
    energy = 0.0
    # Linear terms
    for i, coeff in pairwise_raw['linear'].items():
        if i < len(solution):
            energy += coeff * solution[i]
    # Quadratic terms (symmetric, i < j)
    for (i, j), coeff in pairwise_raw['quad'].items():
        if i < len(solution) and j < len(solution):
            energy += coeff * solution[i] * solution[j]
    return energy


# -----------------------------------------------------------------------------
# 1. evaluate_qubo – Core evaluation function (UPDATED: division-by-zero guard)
# -----------------------------------------------------------------------------

def evaluate_qubo(
    env: Dict,
    lam1: float,
    lam2: float,
    num_sweeps: int,
    schedule_type: str,  # 'old' or 'new'
    num_reads: int,
    seed: int,
    # Old schedule parameters
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    cooling_power: Optional[float] = None,
    num_steps: Optional[int] = None,
    # New schedule parameters
    beta_min_mult: Optional[float] = None,
    beta_max_mult: Optional[float] = None,
    return_all: bool = False,
    use_seed_none: bool = True,
    compute_esr_mcr: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Builds QUBO from env['pairwise_norm'] and given lambdas, runs SA,
    and returns a result dict. SQR is computed using the RAW MIQP energy.

    Returns:
        dict with keys:
            - best_sqr: raw SQR of best feasible solution
            - best_raw_miqp: raw MIQP energy of best feasible solution
            - avg_sqr: mean raw SQR of all feasible solutions
            - p10_sqr: 10th percentile raw SQR
            - feas_rate: fraction of feasible samples
            - best_solution: binary vector of the best solution
            - ESR, MCR (if compute_esr_mcr=True)
            - all_samples (if return_all=True)
    """
    pairwise_norm = env['pairwise_norm']
    pairwise_raw = env['pairwise_raw']
    K_new = env['config']['K_new']
    free_indices = pairwise_norm['free_indices']
    M_indices = env['M_indices']
    N_total = len(env['coords'])
    gurobi_miqp = env['gurobi_miqp']
    Q_obj = env['Q_obj']  # from normalized objective

    # 1. Build QUBO (normalized)
    qubo = build_qubo(pairwise_norm, K_new, lambda1=lam1, lambda2=lam2,
                      use_jijmodeling=False, verbose=False)
    h = qubo['h']
    J = qubo['J']
    constant = qubo['constant']

    # 2. Compute ESR/MCR (from normalized QUBO, for diagnostics)
    esr = None
    mcr = None
    if compute_esr_mcr:
        Q_total = _build_full_Q(h, J, N_total)
        Q_pen = Q_total - Q_obj
        norm_obj = np.linalg.norm(Q_obj, 'fro')
        norm_pen = np.linalg.norm(Q_pen, 'fro')
        esr = norm_obj / norm_pen if norm_pen > 1e-12 else float('inf')
        max_obj = np.max(np.abs(Q_obj))
        max_pen = np.max(np.abs(Q_pen))
        mcr = max_pen / max_obj if max_obj > 1e-12 else float('inf')

    # 3. Build schedule or beta_range
    if schedule_type == 'old':
        if None in (beta_min, beta_max, cooling_power, num_steps):
            raise ValueError("Old schedule requires beta_min, beta_max, cooling_power, num_steps")
        schedule = build_schedule(beta_min, beta_max, num_sweeps, num_steps, cooling_power)
        result = solve_sa(
            h=h, J=J, constant=constant,
            K_new=K_new,
            free_indices=free_indices,
            M_indices=M_indices,
            N_total=N_total,
            pairwise_data=pairwise_norm,  # used only for normalized energy (optional)
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            schedule=schedule,
            seed=seed,
            return_all=return_all,
            use_seed_none=use_seed_none,
            verbose=verbose,
        )
    elif schedule_type == 'new':
        if None in (beta_min_mult, beta_max_mult):
            raise ValueError("New schedule requires beta_min_mult, beta_max_mult")
        # Compute physics-informed baseline from the QUBO itself
        abs_h = [abs(v) for v in h.values() if abs(v) > 1e-12]
        abs_J = [abs(v) for v in J.values() if abs(v) > 1e-12]
        if not abs_h and not abs_J:
            base_beta_min, base_beta_max = 0.01, 100.0
        else:
            # Sparsity-aware normalization: scale = mean(|J|)
            mean_J = np.mean(abs_J) if abs_J else 1.0
            scale = max(mean_J, 1e-12)
            norm_h = [v / scale for v in abs_h]
            norm_J = [v / scale for v in abs_J]
            all_norm_terms = norm_h + norm_J
            E_max_barrier = sum(all_norm_terms)
            E_granularity = sum(all_norm_terms) / len(all_norm_terms) if all_norm_terms else 1.0
            base_beta_min = 0.5 / max(E_max_barrier, 1e-4)
            base_beta_max = 10.0 / max(E_granularity, 1e-4)
            if base_beta_max <= base_beta_min:
                base_beta_max = base_beta_min * 100.0

        beta_min_abs = max(base_beta_min * beta_min_mult, 1e-8)
        beta_max_abs = max(base_beta_max * beta_max_mult, 1e-8)
        if beta_max_abs <= beta_min_abs:
            beta_max_abs = beta_min_abs * 100.0

        result = solve_sa(
            h=h, J=J, constant=constant,
            K_new=K_new,
            free_indices=free_indices,
            M_indices=M_indices,
            N_total=N_total,
            pairwise_data=pairwise_norm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            beta_range=[beta_min_abs, beta_max_abs],
            seed=seed,
            return_all=return_all,
            use_seed_none=use_seed_none,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    all_samples = result.get('all_samples', [])
    best_solution = result.get('solution')
    feas_rate = result.get('violations', {}).get('feasible', False)
    norm_best_miqp = result.get('miqp_energy')  # normalized MIQP (for reference)

    # If return_all=True but no samples, fallback to single solution
    if return_all and not all_samples:
        # Use the best solution (if feasible) and compute raw energy
        if best_solution is not None and feas_rate:
            raw_miqp = _compute_raw_energy(best_solution, pairwise_raw)
            # --- division-by-zero guard ---
            if gurobi_miqp == 0:
                best_sqr = 0.0
            else:
                best_sqr = raw_miqp / gurobi_miqp
            return {
                'best_miqp': raw_miqp,
                'best_sqr': best_sqr,
                'avg_sqr': best_sqr,
                'p10_sqr': best_sqr,
                'best_solution': best_solution,
                'feas_rate': 1.0,
                'feasible_miqps': [raw_miqp],
                'ESR': esr,
                'MCR': mcr,
                'all_samples': all_samples,
            }
        else:
            return {
                'best_miqp': float('inf'),
                'best_sqr': 0.0,
                'avg_sqr': 0.0,
                'p10_sqr': 0.0,
                'best_solution': None,
                'feas_rate': 0.0,
                'feasible_miqps': [],
                'ESR': esr,
                'MCR': mcr,
                'all_samples': all_samples,
            }

    # Process all_samples to compute raw energies for feasible solutions
    feasible_raw_miqps = []
    best_raw_miqp = float('inf')
    best_sol = None
    for sample in all_samples:
        if sample.get('violations', {}).get('feasible', False):
            sol = sample.get('solution')
            if sol is not None:
                raw_e = _compute_raw_energy(sol, pairwise_raw)
                feasible_raw_miqps.append(raw_e)
                if raw_e < best_raw_miqp:
                    best_raw_miqp = raw_e
                    best_sol = sol

    feas_rate = len(feasible_raw_miqps) / len(all_samples) if all_samples else 0.0

    if feasible_raw_miqps and np.isfinite(best_raw_miqp) and best_raw_miqp != float('inf'):
        # --- division-by-zero guard ---
        if gurobi_miqp == 0:
            best_sqr = 0.0
        else:
            best_sqr = best_raw_miqp / gurobi_miqp
        avg_sqr = np.mean([e / gurobi_miqp for e in feasible_raw_miqps]) if gurobi_miqp != 0 else 0.0
        p10_sqr = np.percentile([e / gurobi_miqp for e in feasible_raw_miqps], 10) if gurobi_miqp != 0 else 0.0
        best_solution = best_sol if best_sol is not None else best_solution
    else:
        best_raw_miqp = float('inf')
        best_sqr = 0.0
        avg_sqr = 0.0
        p10_sqr = 0.0
        best_solution = None

    return {
        'best_miqp': best_raw_miqp,          # raw MIQP energy
        'best_sqr': best_sqr,                # raw SQR
        'avg_sqr': avg_sqr,
        'p10_sqr': p10_sqr,
        'best_solution': best_solution,
        'feas_rate': feas_rate,
        'feasible_miqps': feasible_raw_miqps,
        'ESR': esr,
        'MCR': mcr,
        'all_samples': all_samples if return_all else None,
        'norm_best_miqp': norm_best_miqp,    # for debugging
    }


# -----------------------------------------------------------------------------
# 2. make_objective – Optuna objective factory (UPDATED: stores best_solution)
# -----------------------------------------------------------------------------

def make_objective(
    env: Dict,
    schedule_type: str,
    objective_type: str,
    tuning_reads: int,
    tuning_seed: int,
    use_seed_none: bool = True,
    compute_esr_mcr: bool = True,
) -> Callable:
    """
    Returns a callable objective function for Optuna.
    The objective uses raw SQR from evaluate_qubo.
    Stores best_solution and hyperparameters as user attrs.
    """
    Q_sum = env['Q_sum']

    def objective(trial):
        # Suggest hyperparameters
        lam1 = trial.suggest_float('lam1', 1e-4, Q_sum, log=True)
        lam2 = trial.suggest_float('lam2', 1e-4, Q_sum, log=True)
        num_sweeps = trial.suggest_int('num_sweeps', 5000, 30000, step=500)

        trial.set_user_attr('lam1', lam1)
        trial.set_user_attr('lam2', lam2)
        trial.set_user_attr('num_sweeps_actual', num_sweeps)

        if schedule_type == 'old':
            beta_min = trial.suggest_float('beta_min', 0.001, 0.1, log=True)
            beta_max = trial.suggest_float('beta_max', 10.0, 80.0, log=True)
            cooling_power = trial.suggest_float('cooling_power', 0.5, 3.0)
            num_steps = max(10, num_sweeps // 150)
            trial.set_user_attr('beta_min', beta_min)
            trial.set_user_attr('beta_max', beta_max)
            trial.set_user_attr('cooling_power', cooling_power)
            trial.set_user_attr('num_steps_actual', num_steps)
            extra_kwargs = {
                'beta_min': beta_min,
                'beta_max': beta_max,
                'cooling_power': cooling_power,
                'num_steps': num_steps,
            }
        else:  # 'new'
            beta_min_mult = trial.suggest_float('beta_min_mult', 0.1, 10.0, log=True)
            beta_max_mult = trial.suggest_float('beta_max_mult', 0.1, 10.0, log=True)
            trial.set_user_attr('beta_min_mult', beta_min_mult)
            trial.set_user_attr('beta_max_mult', beta_max_mult)
            extra_kwargs = {
                'beta_min_mult': beta_min_mult,
                'beta_max_mult': beta_max_mult,
            }

        # Run evaluation (returns raw SQR)
        try:
            result = evaluate_qubo(
                env=env,
                lam1=lam1,
                lam2=lam2,
                num_sweeps=num_sweeps,
                schedule_type=schedule_type,
                num_reads=tuning_reads,
                seed=tuning_seed,
                return_all=True,
                use_seed_none=use_seed_none,
                compute_esr_mcr=compute_esr_mcr,
                verbose=False,
                **extra_kwargs,
            )
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed: {e}")
            trial.set_user_attr('feas_rate', 0.0)
            trial.set_user_attr('best_sqr', 0.0)
            trial.set_user_attr('avg_sqr', 0.0)
            trial.set_user_attr('p10_sqr', 0.0)
            trial.set_user_attr('ESR', np.nan)
            trial.set_user_attr('MCR', np.nan)
            trial.set_user_attr('best_solution', None)
            return 1.0 if objective_type != 'Multi' else [1.0, 1.0]

        best_sqr = result.get('best_sqr', 0.0)
        feas_rate = result.get('feas_rate', 0.0)
        avg_sqr = result.get('avg_sqr', 0.0)
        p10_sqr = result.get('p10_sqr', 0.0)
        esr = result.get('ESR', np.nan)
        mcr = result.get('MCR', np.nan)
        best_solution = result.get('best_solution')

        trial.set_user_attr('feas_rate', feas_rate)
        trial.set_user_attr('best_sqr', best_sqr)
        trial.set_user_attr('avg_sqr', avg_sqr)
        trial.set_user_attr('p10_sqr', p10_sqr)
        trial.set_user_attr('ESR', esr)
        trial.set_user_attr('MCR', mcr)
        if best_solution is not None:
            trial.set_user_attr('best_solution', best_solution.tolist())
        else:
            trial.set_user_attr('best_solution', None)

        # Compute objective value (now using raw SQR)
        if objective_type == 'BestOnly':
            return 1.0 - best_sqr
        elif objective_type == 'Lin':
            return 1.0 - (best_sqr * feas_rate)
        elif objective_type == 'Sq':
            return 1.0 - (best_sqr * (feas_rate ** 2))
        elif objective_type == 'Cube':
            return 1.0 - (best_sqr * (feas_rate ** 3))
        elif objective_type == 'Avg':
            return 1.0 - avg_sqr
        elif objective_type == 'Pctl10':
            return 1.0 - p10_sqr
        elif objective_type == 'Penalty-0.1':
            score = best_sqr - 0.1 * (1.0 - feas_rate)
            return 1.0 - max(0.0, min(1.0, score))
        elif objective_type == 'Penalty-0.5':
            score = best_sqr - 0.5 * (1.0 - feas_rate)
            return 1.0 - max(0.0, min(1.0, score))
        elif objective_type == 'Multi':
            return [1.0 - best_sqr, 1.0 - feas_rate]
        else:
            raise ValueError(f"Unknown objective_type: {objective_type}")

    return objective


# -----------------------------------------------------------------------------
# 3. run_optuna_study – Create/load and run study (UPDATED: load_if_exists=True)
# -----------------------------------------------------------------------------

# In src/experiment.py

def run_optuna_study(
    experiment_name: str,
    objective: Callable,
    n_trials: int,
    storage_dir: Union[str, Path],
    directions: Optional[List[str]] = None,
    sampler_type: str = 'TPE',
    seed: int = 42,
    load_if_exists: bool = True,  # <-- already True
    use_wandb_callback: bool = False,  # NEW
    verbose: bool = True,
) -> 'optuna.Study':
    """Create or load an Optuna study with RDBStorage."""
    import optuna
    from optuna.storages import RDBStorage

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"{experiment_name}.db"

    if directions is None:
        directions = ['minimize']

    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={'connect_args': {'timeout': 60, 'check_same_thread': False}}
    )

    if sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    elif sampler_type == 'NSGAII':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

    study = optuna.create_study(
        study_name=experiment_name,
        storage=storage,
        sampler=sampler,
        directions=directions,
        load_if_exists=load_if_exists,
    )

    existing_trials = len(study.trials)
    if existing_trials < n_trials:
        if verbose:
            print(f"  Optimizing {n_trials - existing_trials} more trials...")

        # --- W&B Callback ---
        callbacks = []
        if use_wandb_callback:
            try:
                # Only if wandb is active
                import wandb
                if wandb.run is not None:
                    from optuna.integration.wandb import WeightsAndBiasesCallback
                    wandb_callback = WeightsAndBiasesCallback(
                        wandb_kwargs={'reinit': False, 'resume': 'allow'},  # reuse existing run
                        as_multirun=False,  # log to the same run, not separate runs
                        metric_name="objective_value",
                    )
                    callbacks.append(wandb_callback)
                    print("  📊 Optuna trials will be logged to W&B.")
                else:
                    print("  ⚠️ W&B run not active; skipping callback.")
            except ImportError as e:
                print(f"  ⚠️ Could not import W&B callback: {e}")
            except Exception as e:
                print(f"  ⚠️ W&B callback setup failed: {e}")

        try:
            study.optimize(
                objective,
                n_trials=n_trials - existing_trials,
                show_progress_bar=True,
                callbacks=callbacks,
            )
        except Exception as e:
            print(f"  ⚠️ Optimization failed: {e}")
    else:
        if verbose:
            print(f"  ✅ Study already has {existing_trials} trials. Skipping optimization.")

    return study


# -----------------------------------------------------------------------------
# 4. validate_study – Validate top trials (UPDATED: nan-safe aggregation)
# -----------------------------------------------------------------------------

def validate_study(
    study: 'optuna.Study',
    env: Dict,
    schedule_type: str,
    val_reads: int,
    val_seed: int,
    top_k: int = 15,
    use_seed_none: bool = True,
    compute_esr_mcr: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Extract top_k trials and re-evaluate with more reads.
    Returns validation results with raw SQR.
    Uses np.nanmax and np.nanmean for safe aggregation.
    """
    directions = study.directions
    if len(directions) == 1:
        valid_trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
        if valid_trials:
            sorted_trials = sorted(valid_trials, key=lambda t: t.value)
        else:
            valid_trials = [t for t in study.trials if t.user_attrs.get('best_sqr', 0.0) > 0]
            sorted_trials = sorted(valid_trials, key=lambda t: t.user_attrs.get('best_sqr', 0.0), reverse=True)
        top_trials = sorted_trials[:top_k]
    else:
        pareto_trials = study.best_trials
        feasible_pareto = [t for t in pareto_trials if t.user_attrs.get('feas_rate', 0.0) > 0.01]
        if feasible_pareto:
            top_trials = sorted(feasible_pareto, key=lambda x: x.user_attrs.get('best_sqr', 0.0), reverse=True)[:top_k]
        else:
            top_trials = sorted(pareto_trials, key=lambda x: x.user_attrs.get('feas_rate', 0.0), reverse=True)[:top_k]

    if not top_trials:
        if verbose:
            print("  ⚠️ No trials found for validation.")
        return {
            'best_sqr': np.nan,
            'avg_top5_sqr': np.nan,
            'feas_rate': np.nan,
            'spearman_rho': np.nan,
            'spearman_p': np.nan,
            'n_validated': 0,
            'trials': [],
        }

    val_results = []
    tuning_scores = []
    validation_sqrs = []

    if verbose:
        print(f"  Validating {len(top_trials)} trials with {val_reads} reads...")

    for trial in top_trials:
        lam1 = trial.params.get('lam1', 1.0)
        lam2 = trial.params.get('lam2', 1.0)
        num_sweeps = trial.params.get('num_sweeps', 10000)

        if schedule_type == 'old':
            beta_min = trial.params.get('beta_min', 0.01)
            beta_max = trial.params.get('beta_max', 40.0)
            cooling_power = trial.params.get('cooling_power', 1.8)
            num_steps = max(10, num_sweeps // 150)
            extra_kwargs = {
                'beta_min': beta_min,
                'beta_max': beta_max,
                'cooling_power': cooling_power,
                'num_steps': num_steps,
            }
        else:
            beta_min_mult = trial.params.get('beta_min_mult', 1.0)
            beta_max_mult = trial.params.get('beta_max_mult', 1.0)
            extra_kwargs = {
                'beta_min_mult': beta_min_mult,
                'beta_max_mult': beta_max_mult,
            }

        if len(directions) == 1:
            tuning_score = trial.value if trial.value is not None else 1.0
        else:
            tuning_score = 1.0 - trial.user_attrs.get('best_sqr', 0.0)

        try:
            result = evaluate_qubo(
                env=env,
                lam1=lam1,
                lam2=lam2,
                num_sweeps=num_sweeps,
                schedule_type=schedule_type,
                num_reads=val_reads,
                seed=val_seed,
                return_all=True,
                use_seed_none=use_seed_none,
                compute_esr_mcr=compute_esr_mcr,
                verbose=False,
                **extra_kwargs,
            )
        except Exception as e:
            if verbose:
                print(f"    ⚠️ Trial {trial.number} validation failed: {e}")
            continue

        best_sqr = result.get('best_sqr', np.nan)
        feas_rate = result.get('feas_rate', 0.0)
        esr = result.get('ESR', np.nan)
        mcr = result.get('MCR', np.nan)

        val_results.append({
            'trial_number': trial.number,
            'lam1': lam1,
            'lam2': lam2,
            'num_sweeps': num_sweeps,
            'best_sqr': best_sqr,
            'feas_rate': feas_rate,
            'ESR': esr,
            'MCR': mcr,
            'solution': result.get('best_solution'),
        })
        tuning_scores.append(tuning_score)
        validation_sqrs.append(best_sqr)

    # --- Summary stats with nan-safe aggregation ---
    # Filter out NaN values
    valid_best_sqrs = [r['best_sqr'] for r in val_results if np.isfinite(r['best_sqr']) and r['best_sqr'] > 0]
    if valid_best_sqrs:
        best_sqr = np.nanmax(valid_best_sqrs)  # <-- nan-safe max
        # Find corresponding trial
        best_trial = next((r for r in val_results if r['best_sqr'] == best_sqr), None)
        feas_rate = best_trial['feas_rate'] if best_trial else np.nan
        top5_sqrs = sorted(valid_best_sqrs, reverse=True)[:5]
        avg_top5 = np.nanmean(top5_sqrs) if top5_sqrs else np.nan
    else:
        best_sqr = np.nan
        avg_top5 = np.nan
        feas_rate = np.nan

    # Spearman correlation: filter pairs where both are finite
    paired = [(ts, vs) for ts, vs in zip(tuning_scores, validation_sqrs)
              if np.isfinite(vs) and vs > 0 and np.isfinite(ts)]
    if len(paired) >= 3:
        ts_vals, vs_vals = zip(*paired)
        rho, pval = stats.spearmanr(ts_vals, vs_vals, nan_policy='omit')
    else:
        rho, pval = np.nan, np.nan

    if verbose:
        print(f"  ✅ Validation complete:")
        print(f"     Best SQR = {best_sqr:.4f}, Feas = {feas_rate:.2f}")
        print(f"     Spearman ρ = {rho:.4f} (p={pval:.4f}, n={len(paired)})")

    return {
        'best_sqr': best_sqr,
        'avg_top5_sqr': avg_top5,
        'feas_rate': feas_rate,
        'spearman_rho': rho,
        'spearman_p': pval,
        'n_validated': len(val_results),
        'trials': val_results,
    }


# -----------------------------------------------------------------------------
# 5. run_ablation_experiment – Full ablation/comparison orchestration
# -----------------------------------------------------------------------------

def run_ablation_experiment(
    experiment_name: str,
    env: Dict,
    schedule_type: str,
    objective_type: str,
    n_trials: int,
    tuning_reads: int,
    val_reads: int,
    val_seed: int,
    storage_dir: Union[str, Path],
    val_top_k: int = 15,
    tuning_seed: int = 42,
    use_seed_none: bool = True,
    compute_esr_mcr: bool = True,
    force_retune: bool = False,
    show_plots: bool = True,
    verbose: bool = True,
) -> Tuple[Dict, 'optuna.Study']:
    """
    Full pipeline for a single ablation/comparison experiment.
    Now includes enhanced summaries, enhanced learning curve, and new plots.
    """
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # --- Build objective ---
    objective = make_objective(
        env=env,
        schedule_type=schedule_type,
        objective_type=objective_type,
        tuning_reads=tuning_reads,
        tuning_seed=tuning_seed,
        use_seed_none=use_seed_none,
        compute_esr_mcr=compute_esr_mcr,
    )

    directions = ['minimize'] if objective_type != 'Multi' else ['minimize', 'minimize']
    sampler_type = 'TPE' if objective_type != 'Multi' else 'NSGAII'

    # --- Run tuning ---
    study = run_optuna_study(
        experiment_name=experiment_name,
        objective=objective,
        n_trials=n_trials,
        storage_dir=storage_dir,
        directions=directions,
        sampler_type=sampler_type,
        seed=tuning_seed,
        load_if_exists=not force_retune,  # <-- if force_retune, don't load existing
        verbose=verbose,
    )

    # --- Print tuning summary ---
    print_tuning_summary(study, experiment_name)

    # --- Generate enhanced learning curve ---
    plot_dir = storage_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_optuna_learning_curve_enhanced(
        study,
        plot_dir / "learning_curve_enhanced.png",
        show_fig=show_plots,
    )

    # --- Validate ---
    val_results = validate_study(
        study=study,
        env=env,
        schedule_type=schedule_type,
        val_reads=val_reads,
        val_seed=val_seed,
        top_k=val_top_k,
        use_seed_none=use_seed_none,
        compute_esr_mcr=compute_esr_mcr,
        verbose=verbose,
    )

    # --- Save validation results ---
    val_path = storage_dir / f"{experiment_name}_val.pkl"
    safe_save_pickle(val_path, val_results, verbose=verbose)

    # --- JSON summary (with JSON import now fixed) ---
    json_path = storage_dir / f"{experiment_name}_val.json"
    json_data = {
        'experiment_name': experiment_name,
        'schedule_type': schedule_type,
        'objective_type': objective_type,
        'n_trials': n_trials,
        'val_results': {
            'best_sqr': val_results['best_sqr'],
            'avg_top5_sqr': val_results['avg_top5_sqr'],
            'feas_rate': val_results['feas_rate'],
            'spearman_rho': val_results['spearman_rho'],
            'spearman_p': val_results['spearman_p'],
            'n_validated': val_results['n_validated'],
        },
        'config': env['config'],
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)

    # --- Print validation summary ---
    print_validation_summary(val_results, experiment_name)

    # --- Performance scatter (SQR vs ESR, Feas vs MCR) ---
    if val_results['trials']:
        scatter_path = plot_dir / "performance_scatter.png"
        plot_performance_scatter(val_results, scatter_path, show_fig=show_plots)

    # --- Extract best tuning trial and solution ---
    best_tuning_trial = study.best_trial
    tuning_solution = best_tuning_trial.user_attrs.get('best_solution')
    if tuning_solution is not None:
        tuning_solution = np.array(tuning_solution)
    tuning_trial_info = {
        'trial_number': best_tuning_trial.number,
        'lam1': best_tuning_trial.params.get('lam1'),
        'lam2': best_tuning_trial.params.get('lam2'),
        'best_sqr': best_tuning_trial.user_attrs.get('best_sqr', np.nan),
        'solution': tuning_solution,
    }

    # --- Extract best validation trial and solution ---
    if val_results['trials']:
        best_val_trial = val_results['trials'][0]  # already sorted by best_sqr
        val_trial_info = {
            'trial_number': best_val_trial.get('trial_number'),
            'lam1': best_val_trial.get('lam1'),
            'lam2': best_val_trial.get('lam2'),
            'best_sqr': best_val_trial.get('best_sqr'),
            'solution': best_val_trial.get('solution'),
        }
    else:
        val_trial_info = None

    # --- Tuning vs Validation deployment comparison (2x2) ---
    if tuning_solution is not None and val_trial_info is not None and val_trial_info['solution'] is not None:
        plot_tuning_vs_validation_deployment(
            env=env,
            tuning_trial=tuning_trial_info,
            val_trial=val_trial_info,
            coords=env['coords'],
            U=env['U'],
            M_indices=env['M_indices'],
            DOMAIN_SIZE=50.0,
            current_vector=(1.0, 0.0),
            CONNECTIVITY_RANGE=env['connectivity_range'],
            experiment_name=experiment_name,
            save_path=plot_dir / "tuning_vs_validation_deployment.png",
            dpi=150,
            show_fig=show_plots,
        )

    # --- Deployment + QUBO for validation champion ---
    if val_trial_info is not None and val_trial_info['solution'] is not None:
        solution = val_trial_info['solution']
        selected_new = [i for i in range(len(solution)) if solution[i] == 1 and i not in env['M_indices']]
        lam1 = val_trial_info['lam1']
        lam2 = val_trial_info['lam2']
        title = f"Validation Champion: Trial #{val_trial_info['trial_number']} | SQR={val_trial_info['best_sqr']:.4f}"
        plot_deployment_with_qubo(
            env=env,
            lam1=lam1,
            lam2=lam2,
            solution=solution,
            M_indices=env['M_indices'],
            selected_new=selected_new,
            U=env['U'],
            coords=env['coords'],
            DOMAIN_SIZE=50.0,
            current_vector=(1.0, 0.0),
            CONNECTIVITY_RANGE=env['connectivity_range'],
            title=title,
            save_path=plot_dir / "validation_champion_deployment_qubo.png",
            dpi=150,
            show_fig=show_plots,
        )

    # --- Optuna posterior plots (standard) ---
    save_optuna_plots(study, plot_dir, show_fig=show_plots)

    # --- QUBO matrix heatmap for best validation trial (legacy) ---
    if val_trial_info is not None and val_trial_info['lam1'] is not None:
        plot_qubo_matrix_heatmap(
            env=env,
            lam1=val_trial_info['lam1'],
            lam2=val_trial_info['lam2'],
            K_new=env['config']['K_new'],
            save_path=plot_dir / f"qubo_heatmap_lam1_{val_trial_info['lam1']:.4f}_lam2_{val_trial_info['lam2']:.4f}.png",
            show_fig=show_plots,
        )

    # --- Prepare global summary data ---
    summary_results = {
        'phase_times': {'Tuning': 0, 'Validation': 0},
        'total_time': 0,
        'champion': {
            'best_sqr': best_tuning_trial.user_attrs.get('best_sqr', np.nan) if best_tuning_trial else np.nan,
        },
        'validation_results': val_results,
        'best_sharpen': {},
        'spearman': {
            'rho': val_results.get('spearman_rho', np.nan),
            'available': val_results.get('spearman_rho') is not None,
        },
    }
    print_global_summary(summary_results, experiment_name)

    if verbose:
        print(f"  ✅ Experiment '{experiment_name}' complete.")
        print(f"     Results saved to {storage_dir}")

    return val_results, study


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    'evaluate_qubo',
    'make_objective',
    'run_optuna_study',
    'validate_study',
    'run_ablation_experiment',
]