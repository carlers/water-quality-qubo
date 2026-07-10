"""
src/utils.py

Utility functions for the water quality monitoring SA pipeline.

This module provides:
    1. Phase execution context manager (timing + logging)
    2. Safe I/O operations (pickle + JSON)
    3. NumpyEncoder for JSON serialization
    4. Config builder for multi-seed runs
    5. Spearman correlation computation
    6. Champion extraction
    7. Plotting functions (validation grid, convergence, deployment)

All functions preserve verbose logging and error handling.
"""

import json
import pickle
import time
import gc
import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import matplotlib.image as mpimg
from scipy.interpolate import griddata
import scipy.stats as stats

# Global phase timing dictionary (used by execute_phase context manager)
PHASE_TIMES = {}


# ============================================================================
# 1. PHASE EXECUTION CONTEXT MANAGER
# ============================================================================

@contextlib.contextmanager
def execute_phase(phase_name):
    """
    Context manager for timing and logging phases.
    
    Usage:
        with execute_phase("Phase 1: Tuning"):
            # ... code ...
    
    This automatically:
        - Prints start/end messages
        - Times execution
        - Stores time in PHASE_TIMES dict
        - Handles exceptions gracefully
    """
    print("\n" + "=" * 70)
    print(f"🚀 [{phase_name}] START")
    print("=" * 70)
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        PHASE_TIMES[phase_name] = elapsed
        print(f"\n✅ [{phase_name}] COMPLETE: {elapsed/60:.2f} minutes")
        print("=" * 70)


# ============================================================================
# 2. SAFE I/O HELPERS
# ============================================================================

def safe_load_pickle(filepath, default=None):
    """
    Safely load a pickle file, returning default if not found or corrupted.
    
    Args:
        filepath: Path to pickle file
        default: Value to return if file not found or corrupted
    
    Returns:
        Loaded data or default
    """
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.PickleError, AttributeError):
        return default


def safe_save_pickle(filepath, data, verbose=True):
    """
    Safely save data to a pickle file, creating parent directories.
    
    Args:
        filepath: Path to save to
        data: Data to save
        verbose: Print success message
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"  ✓ Saved to {filepath}")
        return True
    except Exception as e:
        print(f"  ⚠️ Failed to save to {filepath}: {e}")
        return False


def safe_load_json(filepath, default=None):
    """
    Safely load a JSON file, returning default if not found or corrupted.
    
    Args:
        filepath: Path to JSON file
        default: Value to return if file not found or corrupted
    
    Returns:
        Loaded data or default
    """
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ============================================================================
# 3. CUSTOM JSON ENCODER
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy types and NaN values.
    
    Usage:
        json.dump(data, f, cls=NumpyEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# 4. CONFIG BUILDER
# ============================================================================

def build_config(seed, test_mode, **kwargs):
    """
    Build configuration dictionary dynamically for each seed.
    
    Args:
        seed: Dataset seed
        test_mode: Boolean indicating test mode
        **kwargs: Additional config parameters
    
    Returns:
        Dictionary with full configuration
    """
    config = {
        'test_mode': test_mode,
        'seed': seed,
        'n_master': kwargs.get('n_master', 100),
        'domain_size_km': kwargs.get('domain_size', 50.0),
        'L_c_km': kwargs.get('L_c', 5.0),
        'L_w_km': kwargs.get('L_w', 1.0),
        'connectivity_range_km': kwargs.get('connectivity_range', 6.0),
        'beta': kwargs.get('beta', 1.0),
        'delta': kwargs.get('delta', 1.0),
        'K_new': kwargs.get('K_new', 5),
        'tuning_trials': kwargs.get('tuning_trials', 1000),
        'tuning_reads': kwargs.get('tuning_reads', 128),
        'tuning_seed': kwargs.get('tuning_seed', 42),
        'val_reads': kwargs.get('val_reads', 512),
        'val_seed': kwargs.get('val_seed', 43),
        'val_top_k': kwargs.get('val_top_k', 20),
        'sharpen_reads': kwargs.get('sharpen_reads', 2048),
        'sharpen_repeats': kwargs.get('sharpen_repeats', 3),
        'penalty_regime': 'upper_bound_mentor',
        'Q_sum': kwargs.get('Q_sum', None),
    }
    return config


# ============================================================================
# 5. SPEARMAN CORRELATION COMPUTATION
# ============================================================================

def compute_spearman_correlation(study, validation_results, pairwise, K_new, 
                                  M_indices, free_indices, neighbors, N_total,
                                  GUROBI_MIQP, VAL_READS, VAL_SEED,
                                  verbose=True):
    """
    Compute stratified Spearman correlation between tuning and validation.
    
    Args:
        study: Optuna study object
        validation_results: List of validation results
        pairwise: Pairwise data from compute_pairwise_terms()
        K_new: Number of new stations
        M_indices: Existing station indices
        free_indices: Free variable indices
        neighbors: Neighbor dict from pairwise
        N_total: Total number of candidates
        GUROBI_MIQP: Gurobi baseline MIQP value
        VAL_READS: Number of reads per validation trial
        VAL_SEED: Seed for validation
        verbose: Print progress
    
    Returns:
        dict with 'rho', 'p', 'n', 'significant' or None if insufficient data
    """
    from src.model import build_qubo, compute_energy
    from src.solvers import compute_violations
    
    trials_df = study.trials_dataframe()
    trials_df['trial_number'] = trials_df.index
    
    can_run = (
        'user_attrs_score' in trials_df.columns and
        not trials_df['user_attrs_score'].isna().all() and
        trials_df['user_attrs_score'].max() > 0.0 and
        len(trials_df) > 5
    )
    
    if not can_run:
        if verbose:
            print("  ⚠️ Skipping Spearman correlation (insufficient feasible trials).")
        return None
    
    all_trials_df = trials_df[trials_df['user_attrs_score'].notna()]
    all_trials_df = all_trials_df.sort_values('user_attrs_score', ascending=False)
    all_trials_df['trial_number'] = all_trials_df.index
    n_total = len(all_trials_df)
    
    if verbose:
        print(f"  Total trials with score: {n_total}")
        print(f"  Stratified sample: Top 10, Middle 10, Bottom 10")
    
    # Stratified sampling
    SPEARMAN_TOP_N = 10
    SPEARMAN_MID_N = 10
    SPEARMAN_BOT_N = 10
    
    top_sample = all_trials_df.head(SPEARMAN_TOP_N)
    mid_start = max(0, n_total // 2 - SPEARMAN_MID_N // 2)
    mid_end = min(n_total, mid_start + SPEARMAN_MID_N)
    mid_sample = all_trials_df.iloc[mid_start:mid_end]
    bot_sample = all_trials_df.tail(SPEARMAN_BOT_N)
    stratified_df = pd.concat([top_sample, mid_sample, bot_sample])
    
    def validate_trial(trial_num):
        """Run validation for a single trial."""
        row = all_trials_df[all_trials_df['trial_number'] == trial_num].iloc[0]
        lam1 = row['params_lam1']
        lam2 = row['params_lam2']
        beta_min = row['params_beta_min']
        beta_max = row['params_beta_max']
        sweeps = int(row['user_attrs_num_sweeps_actual'])
        steps = int(row['user_attrs_num_steps_actual'])
        power = row['params_cooling_power']
        
        qubo = build_qubo(pairwise, K_new, lam1, lam2, use_jijmodeling=False, verbose=False)
        h, J, constant = qubo['h'], qubo['J'], qubo['constant']
        Q = {(i, i): coeff for i, coeff in h.items()}
        for (i, j), coeff in J.items():
            Q[(i, j)] = coeff
        
        progress = np.linspace(0.0, 1.0, steps) ** power
        betas = beta_min + (beta_max - beta_min) * progress
        sweeps_per_step = sweeps // steps
        schedule = [[float(b), sweeps_per_step] for b in betas]
        remaining = sweeps - (steps * sweeps_per_step)
        if remaining > 0:
            schedule[-1][1] += remaining
        
        try:
            import openjij as oj
            sampler = oj.SASampler()
            response = sampler.sample_qubo(Q, num_reads=VAL_READS, schedule=schedule, seed=VAL_SEED)
        except:
            return np.nan
        
        miqp_list = []
        for i in range(response.record.shape[0]):
            sample_arr = response.record['sample'][i]
            x_full = np.zeros(N_total, dtype=int)
            for m in M_indices:
                x_full[m] = 1
            for var_idx, val in zip(response.indices, sample_arr):
                if var_idx < N_total:
                    x_full[var_idx] = int(round(val))
            viol = compute_violations(x_full, free_indices, M_indices, neighbors, K_new)
            if viol['feasible']:
                miqp = compute_energy(x_full, pairwise, K_new, lambda1=None, lambda2=None)
                miqp_list.append(miqp)
        
        if not miqp_list:
            return np.nan
        return np.mean(miqp_list) / GUROBI_MIQP
    
    # Compute validation SQRs for stratified trials
    stratified_sqrs = []
    stratified_scores = []
    
    for _, row in stratified_df.iterrows():
        trial_num = int(row['trial_number'])
        tuning_score = row['user_attrs_score']
        existing = next((r for r in validation_results if r['trial'] == trial_num), None)
        if existing:
            val_sqr = existing['best_sqr']
        else:
            val_sqr = validate_trial(trial_num)
        if not np.isnan(val_sqr):
            stratified_sqrs.append(val_sqr)
            stratified_scores.append(tuning_score)
    
    if len(stratified_sqrs) <= 2:
        if verbose:
            print(f"  ⚠️ Not enough valid results (need > 2, got {len(stratified_sqrs)})")
        return None
    
    # Compute Spearman correlation
    spearman_rho, spearman_p = stats.spearmanr(stratified_scores, stratified_sqrs)
    
    if verbose:
        print(f"\n  Spearman ρ (tuning best vs validation best): {spearman_rho:.4f}")
        print(f"  P-value:                                    {spearman_p:.4f}")
        print(f"  N:                                          {len(stratified_sqrs)}")
        
        if spearman_p < 0.05:
            print(f"  ✓ Statistically significant (p < 0.05)")
        else:
            print(f"  ⚠️ Not statistically significant (p >= 0.05)")
        
        if spearman_rho > 0.7:
            print(f"  ✅ Strong positive correlation")
        elif spearman_rho > 0.3:
            print(f"  ℹ️  Moderate positive correlation")
        elif spearman_rho > -0.3:
            print(f"  ℹ️  Weak/No correlation")
        else:
            print(f"  ⚠️ Negative correlation")
    
    return {
        'rho': float(spearman_rho),
        'p': float(spearman_p),
        'n': int(len(stratified_sqrs)),
        'significant': bool(spearman_p < 0.05)
    }


# ============================================================================
# 6. CHAMPION EXTRACTION
# ============================================================================

def extract_top3_champions(validation_results, champion, sharpen_top_k=3):
    """
    Extract top K unique champions from validation results.
    
    Args:
        validation_results: List of validation result dicts
        champion: Best champion dict
        sharpen_top_k: Number of top champions to extract
    
    Returns:
        List of top K champion dicts (unique by trial number)
    """
    unique_top3 = []
    seen = set()
    
    # Try to get unique top trials
    for t in validation_results[:sharpen_top_k * 2]:
        if t['trial'] not in seen and t['feas_rate'] > 0 and not np.isnan(t['best_sqr']):
            unique_top3.append(t)
            seen.add(t['trial'])
            if len(unique_top3) >= sharpen_top_k:
                break
    
    # Ensure we always have at least sharpen_top_k
    while len(unique_top3) < sharpen_top_k:
        if champion['trial'] not in seen:
            unique_top3.append(champion)
            seen.add(champion['trial'])
        else:
            fallback_copy = champion.copy()
            fallback_copy['trial'] = -1
            unique_top3.append(fallback_copy)
            break
    
    # Remove fallback entries if we have enough real ones
    unique_top3 = [t for t in unique_top3 if t['trial'] != -1]
    while len(unique_top3) < sharpen_top_k:
        unique_top3.append(champion)
    
    return unique_top3


# ============================================================================
# 7. GET SOLUTION FOR PARAMS
# ============================================================================

def get_solution_for_params(params, pairwise, K_new, M_indices, N_total, seed=42):
    """
    Run SA for a single set of parameters and return the best solution.
    
    Args:
        params: Dict with lam1, lam2, beta_min, beta_max, num_sweeps, num_steps, cooling_power
        pairwise: Pairwise data from compute_pairwise_terms()
        K_new: Number of new stations
        M_indices: Existing station indices
        N_total: Total number of candidates
        seed: Random seed
    
    Returns:
        Binary solution vector of length N_total
    """
    from src.model import build_qubo
    
    lam1 = params['lam1']
    lam2 = params['lam2']
    beta_min = params['beta_min']
    beta_max = params['beta_max']
    sweeps = params['num_sweeps']
    steps = params['num_steps']
    power = params['cooling_power']
    
    qubo = build_qubo(pairwise, K_new, lam1, lam2, use_jijmodeling=False, verbose=False)
    h, J, constant = qubo['h'], qubo['J'], qubo['constant']
    Q = {(i, i): coeff for i, coeff in h.items()}
    for (i, j), coeff in J.items():
        Q[(i, j)] = coeff
    
    progress = np.linspace(0.0, 1.0, steps) ** power
    betas = beta_min + (beta_max - beta_min) * progress
    sweeps_per_step = sweeps // steps
    schedule = [[float(b), sweeps_per_step] for b in betas]
    remaining = sweeps - (steps * sweeps_per_step)
    if remaining > 0:
        schedule[-1][1] += remaining
    
    try:
        import openjij as oj
        sampler = oj.SASampler()
        response = sampler.sample_qubo(Q, num_reads=1, schedule=schedule, seed=seed)
        best = response.first
        x_sample = best.sample
        x_full = np.zeros(N_total, dtype=int)
        for m in M_indices:
            x_full[m] = 1
        for idx_val, val in x_sample.items():
            if idx_val < N_total:
                x_full[idx_val] = int(round(val))
        return x_full
    except:
        x_full = np.zeros(N_total, dtype=int)
        for m in M_indices:
            x_full[m] = 1
        return x_full


def get_solutions_for_params(trials, pairwise, K_new, M_indices, N_total, seed=42):
    """Get solutions for a list of trials."""
    solutions = []
    for trial in trials:
        params = {
            'lam1': trial['lam1'],
            'lam2': trial['lam2'],
            'beta_min': trial['beta_min'],
            'beta_max': trial['beta_max'],
            'num_sweeps': trial['num_sweeps'],
            'num_steps': trial['num_steps'],
            'cooling_power': trial['cooling_power'],
        }
        sol = get_solution_for_params(params, pairwise, K_new, M_indices, N_total, seed)
        solutions.append(sol)
    return solutions


# ============================================================================
# 8. PLOTTING FUNCTIONS
# ============================================================================

def plot_validation_grid(coords, U, gurobi_solution, top3_solutions,
                         top3_trials, M_indices, DOMAIN_SIZE,
                         save_path, dpi=100):
    """
    Generate 2x2 validation grid comparing Gurobi vs Top 3.
    
    Args:
        coords: (N, 2) array of coordinates
        U: (N,) array of utility scores
        gurobi_solution: Binary vector from Gurobi
        top3_solutions: List of 3 binary solution vectors
        top3_trials: List of 3 trial dicts with 'trial' and 'best_sqr'
        M_indices: Existing station indices
        DOMAIN_SIZE: Domain size in km
        save_path: Path to save the plot
        dpi: Resolution
    """
    N_total = len(coords)
    grid_x = np.linspace(0, DOMAIN_SIZE, 100)
    grid_y = np.linspace(0, DOMAIN_SIZE, 100)
    grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = ["Gurobi (Exact)"] + [f"Trial {t['trial']} (Rank {i+1})" for i, t in enumerate(top3_trials[:3])]
    solutions = [gurobi_solution] + top3_solutions
    sqrs = ["1.0000"] + [f"{t['best_sqr']:.4f}" for t in top3_trials[:3]]
    
    for ax, title, x, sqr in zip(axes.flat, titles, solutions, sqrs):
        ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.6,
                   edgecolor='gray', linewidth=0.3)
        
        selected = [i for i in range(N_total) if x[i] == 1]
        selected_new = [i for i in selected if i not in M_indices]
        selected_m = [i for i in selected if i in M_indices]
        
        if selected_m:
            ax.scatter(coords[selected_m, 0], coords[selected_m, 1],
                       c='blue', s=120, marker='s', edgecolor='black', label='Existing (M)')
        if selected_new:
            ax.scatter(coords[selected_new, 0], coords[selected_new, 1],
                       c='red', s=150, marker='o', edgecolor='black', label='New')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_aspect('equal')
        ax.set_xlim(-2, DOMAIN_SIZE + 2)
        ax.set_ylim(-2, DOMAIN_SIZE + 2)
        ax.text(0.02, 0.98, f'SQR = {sqr}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    handles = [Patch(facecolor='red', edgecolor='black', label='New'),
               Patch(facecolor='blue', edgecolor='black', label='Existing (M)'),
               Patch(facecolor='lightgray', edgecolor='gray', label='Candidates')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_convergence_profile(best_sharpen, convergence_data, GUROBI_MIQP,
                             save_path, dpi=100):
    """
    Generate convergence profile plot.
    
    Args:
        best_sharpen: Best sharpening result dict
        convergence_data: Dict of convergence data from sharpening
        GUROBI_MIQP: Gurobi baseline MIQP value
        save_path: Path to save the plot
        dpi: Resolution
    """
    conv_key = f"trial_{best_sharpen['trial']}_run_{best_sharpen['run']}"
    conv_data = convergence_data.get(conv_key)
    
    if conv_data is None:
        print("  ⚠️ No convergence data available")
        return
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    cumulative = conv_data['cumulative_best']
    seed = conv_data['seed']
    
    cumulative_sqr = [x / GUROBI_MIQP for x in cumulative if x != float('inf')]
    cumulative_miqp = [x for x in cumulative if x != float('inf')]
    read_indices = range(len(cumulative_sqr))
    
    ax1.plot(read_indices, cumulative_sqr, 'b-', linewidth=2, label='SQR')
    ax1.set_xlabel('Read Index', fontsize=12)
    ax1.set_ylabel('SQR (Best so far)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Gurobi Optimum')
    
    ax2 = ax1.twinx()
    ax2.plot(read_indices, cumulative_miqp, 'r-', linewidth=2, label='MIQP')
    ax2.set_ylabel('MIQP (Best so far)', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.set_title(f'Convergence Profile (Trial {best_sharpen["trial"]}, Run {best_sharpen["run"]}, Seed: {seed})', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.12, 1.0))
    
    final_sqr = cumulative_sqr[-1] if cumulative_sqr else 0
    ax1.text(0.02, 0.98, f'Final SQR = {final_sqr:.4f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_final_deployment(coords, U, best_solution, M_indices, selected_new,
                          DOMAIN_SIZE, current_vector, CONNECTIVITY_RANGE,
                          champion, best_sharpen, sharpening_means,
                          save_path, dpi=100):
    """
    Generate final deployment plot with utility landscape.
    
    Args:
        coords: (N, 2) array of coordinates
        U: (N,) array of utility scores
        best_solution: Binary solution vector
        M_indices: Existing station indices
        selected_new: List of new station indices
        DOMAIN_SIZE: Domain size in km
        current_vector: (dx, dy) current direction
        CONNECTIVITY_RANGE: D_max in km
        champion: Champion dict
        best_sharpen: Best sharpening result dict
        sharpening_means: Sharpening means dict
        save_path: Path to save the plot
        dpi: Resolution
    """
    if best_solution is None:
        print("  ⚠️ No solution available for final deployment plot")
        return
    
    grid_x = np.linspace(0, DOMAIN_SIZE, 100)
    grid_y = np.linspace(0, DOMAIN_SIZE, 100)
    grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')
    
    selected_m = [i for i in range(len(best_solution)) if best_solution[i] == 1 and i in M_indices]
    all_selected = selected_m + selected_new
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    plt.subplots_adjust(right=0.72)
    
    cf = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Utility $U_i$ (Full Domain)', fontsize=12)
    
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.5,
               edgecolor='gray', linewidth=0.2)
    
    if selected_m:
        ax.scatter(coords[selected_m, 0], coords[selected_m, 1],
                   c='blue', s=150, marker='s', edgecolor='black', linewidth=2, label='Existing (M)')
    
    if selected_new:
        sizes = 100 + 200 * (U[selected_new] - U[selected_new].min()) / (U[selected_new].max() - U[selected_new].min() + 1e-6)
        ax.scatter(coords[selected_new, 0], coords[selected_new, 1],
                   c='red', s=sizes, edgecolor='black', linewidth=2, zorder=3,
                   label=f'New Stations ({len(selected_new)})')
    
    # Annotate selected stations
    for i in all_selected:
        label = f"{i}\nU={U[i]:.3f}"
        color = 'blue' if i in selected_m else 'red'
        ax.annotate(label, (coords[i, 0] + 0.5, coords[i, 1] + 0.5),
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Draw connectivity links
    D_max = CONNECTIVITY_RANGE
    for i, idx_i in enumerate(all_selected):
        for j, idx_j in enumerate(all_selected):
            if i < j:
                dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                if dist <= D_max:
                    ax.plot([coords[idx_i, 0], coords[idx_j, 0]],
                            [coords[idx_i, 1], coords[idx_j, 1]],
                            color='gray', alpha=0.4, linewidth=1.5, linestyle='--')
    
    # Current direction
    ax.quiver(0, 0, current_vector[0], current_vector[1],
              angles='xy', scale_units='xy', scale=10,
              color='blue', width=0.02, label='Current', zorder=6)
    
    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    ax.set_title("Deployment: Best Solution After Sharpening", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2, DOMAIN_SIZE + 2)
    ax.set_ylim(-2, DOMAIN_SIZE + 2)
    
    handles = [Patch(facecolor='red', edgecolor='black', label='New'),
               Patch(facecolor='blue', edgecolor='black', label='Existing (M)'),
               Patch(facecolor='lightgray', edgecolor='gray', label='Candidates')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    
    # Info text
    info_text = (
        f"Best Deployment Solution\n"
        f"{'─' * 30}\n"
        f"Trial: #{best_sharpen['trial']} (Run {best_sharpen['run']})\n"
        f"λ₁ = {best_sharpen['lam1']:.4f}\n"
        f"λ₂ = {best_sharpen['lam2']:.4f}\n"
        f"Sweeps: {champion['num_sweeps']}\n"
        f"Steps: {champion['num_steps']}\n"
        f"Power: {champion['cooling_power']:.4f}\n"
        f"Seed: {best_sharpen['seed']}\n"
        f"Feasibility: {best_sharpen['feas_rate']*100:.1f}%\n"
        f"{'─' * 30}\n"
        f"BEST SQR: {best_sharpen['best_sqr']:.4f}\n"
        f"Mean Best SQR: {sharpening_means['best_sqr']:.4f}\n"
        f"Selected: {len(selected_new)} stations"
    )
    plt.figtext(0.601, 0.15, info_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 0.68, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def display_saved_plots(seed_dir):
    """Display all three saved plots from a completed seed."""
    print("\n" + "-" * 70)
    print(f"📊 DISPLAYING SAVED PLOTS (mode: {seed_dir.parent.name}/{seed_dir.name})")
    print("-" * 70)
    
    plots = [
        ("phase2_validation_2x2.png", "2×2 Validation Plot (Gurobi + Top 3)"),
        ("phase3_convergence.png", "Convergence Profile (Best Sharpening Run)"),
        ("phase3_deployment_solution.png", "Final Deployment Solution"),
    ]
    
    for filename, title in plots:
        filepath = seed_dir / filename
        if filepath.exists():
            try:
                img = mpimg.imread(filepath)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(title, fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                plt.close()
            except Exception as e:
                print(f"  ⚠️ Failed to display {filename}: {e}")
        else:
            print(f"  ⚠️ Plot not found: {filename}")


# ============================================================================
# 9. LOADED SEED SUMMARY
# ============================================================================

def print_loaded_seed_summary(results, seed, mode):
    """
    Print a rich summary table for a loaded seed.
    
    Args:
        results: Dictionary loaded from deployment_results.json
        seed: Seed number
        mode: 'test' or 'full'
    """
    print(f"\n📊 COMPLETED SEED {seed} SUMMARY ({mode.upper()} MODE)")
    print("─" * 70)
    
    best_sharpen = results.get('best_sharpen', {})
    champion = results.get('champion', {})
    spearman = results.get('spearman', {})
    
    # Best SQR
    best_sqr = best_sharpen.get('best_sqr', 'N/A')
    if isinstance(best_sqr, float):
        print(f"  Best SQR:              {best_sqr:.4f}")
    else:
        print(f"  Best SQR:              {best_sqr}")
    
    # Champion info
    print(f"  Champion Trial:        {champion.get('trial', 'N/A')}")
    
    # Hyperparameters
    lam1 = champion.get('lam1', 'N/A')
    lam2 = champion.get('lam2', 'N/A')
    if isinstance(lam1, float) and isinstance(lam2, float):
        print(f"  λ₁:                    {lam1:.4f},  λ₂: {lam2:.4f}")
    else:
        print(f"  λ₁:                    {lam1},  λ₂: {lam2}")
    
    beta_min = champion.get('beta_min', 'N/A')
    beta_max = champion.get('beta_max', 'N/A')
    if isinstance(beta_min, float) and isinstance(beta_max, float):
        print(f"  β_min:                 {beta_min:.4f},  β_max: {beta_max:.2f}")
    else:
        print(f"  β_min:                 {beta_min},  β_max: {beta_max}")
    
    sweeps = champion.get('num_sweeps', 'N/A')
    steps = champion.get('num_steps', 'N/A')
    power = champion.get('cooling_power', 'N/A')
    if isinstance(power, float):
        print(f"  Sweeps:                {sweeps},  Steps: {steps},  Power: {power:.4f}")
    else:
        print(f"  Sweeps:                {sweeps},  Steps: {steps},  Power: {power}")
    
    # Selected stations
    selected = champion.get('selected_new', 'N/A')
    if isinstance(selected, list):
        if len(selected) <= 15:
            print(f"  Selected Stations:     {selected}")
        else:
            print(f"  Selected Stations:     {selected[:5]}... (total {len(selected)})")
    else:
        print(f"  Selected Stations:     {selected}")
    
    # Feasibility
    feas_rate = champion.get('feas_rate', 'N/A')
    if isinstance(feas_rate, float):
        print(f"  Feasibility:           {feas_rate*100:.1f}%")
    else:
        print(f"  Feasibility:           {feas_rate}")
    
    # Spearman correlation - safe handling
    if spearman and spearman.get('rho') is not None:
        print(f"  Spearman ρ:            {spearman['rho']:.4f}")
    else:
        print(f"  Spearman ρ:            N/A")
    
    # Total time
    total_time = results.get('total_time', 'N/A')
    if isinstance(total_time, (int, float)):
        print(f"  Total Time:            {total_time/60:.2f} min")
    else:
        print(f"  Total Time:            {total_time}")
    
    print("─" * 70)


# ============================================================================
# 10. DYNAMIC SUMMARY N
# ============================================================================

def get_summary_n(n_total, max_n=10):
    """
    Dynamically determine how many rows to show in each section.
    
    Args:
        n_total: Total number of feasible trials
        max_n: Maximum rows to show (default 10)
    
    Returns:
        int: Number of rows for each section (Top, Middle, Bottom)
    """
    if n_total <= 0:
        return 0
    elif n_total <= 10:
        return n_total
    elif n_total <= 20:
        return max(3, n_total // 3)
    elif n_total <= 50:
        return max(5, n_total // 5)
    else:
        return max_n


# ============================================================================
# 11. MODULE EXPORTS
# ============================================================================

__all__ = [
    # Context manager
    'execute_phase',
    'PHASE_TIMES',
    
    # I/O helpers
    'safe_load_pickle',
    'safe_save_pickle',
    'safe_load_json',
    'NumpyEncoder',
    
    # Config
    'build_config',
    
    # Computation
    'compute_spearman_correlation',
    'extract_top3_champions',
    'get_solution_for_params',
    'get_solutions_for_params',
    
    # Plotting
    'plot_validation_grid',
    'plot_convergence_profile',
    'plot_final_deployment',
    'display_saved_plots',
    
    # Summary
    'print_loaded_seed_summary',
    'get_summary_n',
]