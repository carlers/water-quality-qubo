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
    7. Loaded seed summary
    8. Dynamic summary N (get_summary_n)
    9. tqdm cleanup

All plotting functions have been moved to src/plotting.py.
The obsolete calibrate_qubo_parameters function has been removed.
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
    """
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.PickleError, AttributeError):
        return default


def safe_save_pickle(filepath, data, verbose=True):
    """
    Safely save data to a pickle file, creating parent directories.
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
        'version': kwargs.get('version', '4.21'),
    }
    return config


# ============================================================================
# 5. SPEARMAN CORRELATION COMPUTATION
# ============================================================================

def compute_spearman_correlation(study, validation_results, verbose=True):
    """
    Compute Spearman correlation between tuning scores and validation SQRs.
    """
    trials_df = study.trials_dataframe()
    trials_df['trial_number'] = trials_df.index

    if 'user_attrs_score' not in trials_df.columns or trials_df['user_attrs_score'].isna().all():
        if verbose:
            print("  ⚠️ Skipping Spearman correlation (no tuning scores found).")
        return {
            'rho': None,
            'p': None,
            'n': 0,
            'significant': False,
            'available': False
        }

    tuning_scores = {}
    for _, row in trials_df.iterrows():
        tn = row['trial_number']
        if not pd.isna(row.get('user_attrs_score', np.nan)):
            tuning_scores[tn] = row['user_attrs_score']

    validation_sqrs = {}
    for val in validation_results:
        trial_num = val.get('trial')
        best_sqr = val.get('best_sqr')
        if trial_num is not None and trial_num >= 0 and best_sqr is not None and not np.isnan(best_sqr):
            validation_sqrs[trial_num] = best_sqr

    common_trials = set(tuning_scores.keys()) & set(validation_sqrs.keys())
    if len(common_trials) < 3:
        if verbose:
            print(f"  ⚠️ Insufficient common trials (need > 2, got {len(common_trials)}). Skipping.")
        return {
            'rho': None,
            'p': None,
            'n': int(len(common_trials)),
            'significant': False,
            'available': False
        }

    if verbose:
        print(f"  Computing Spearman on {len(common_trials)} validated trials...")

    scores_list = [tuning_scores[t] for t in common_trials]
    sqrs_list = [validation_sqrs[t] for t in common_trials]
    spearman_rho, spearman_p = stats.spearmanr(scores_list, sqrs_list)

    if verbose:
        print(f"\n  Spearman ρ (tuning score vs validation SQR): {spearman_rho:.4f}")
        print(f"  P-value:                                      {spearman_p:.4f}")
        print(f"  N:                                            {len(common_trials)}")
        if spearman_p < 0.05:
            print(f"  ✓ Statistically significant (p < 0.05)")
        else:
            print(f"  ⚠️ Not statistically significant (p >= 0.05)")

    return {
        'rho': float(spearman_rho),
        'p': float(spearman_p),
        'n': int(len(common_trials)),
        'significant': bool(spearman_p < 0.05),
        'available': True
    }


# ============================================================================
# 6. CHAMPION EXTRACTION
# ============================================================================

def extract_top3_champions(validation_results, champion, sharpen_top_k=3):
    """
    Extract top K unique champions from validation results.
    """
    unique_top3 = []
    seen = set()
    for t in validation_results[:sharpen_top_k * 2]:
        if t['trial'] not in seen and t.get('feas_rate', 0) > 0 and not np.isnan(t.get('best_sqr', np.nan)):
            unique_top3.append(t)
            seen.add(t['trial'])
            if len(unique_top3) >= sharpen_top_k:
                break

    while len(unique_top3) < sharpen_top_k:
        if champion['trial'] not in seen:
            unique_top3.append(champion)
            seen.add(champion['trial'])
        else:
            fallback_copy = champion.copy()
            fallback_copy['trial'] = -1
            unique_top3.append(fallback_copy)
            break

    unique_top3 = [t for t in unique_top3 if t['trial'] != -1]
    while len(unique_top3) < sharpen_top_k:
        unique_top3.append(champion)

    return unique_top3


# ============================================================================
# 7. DYNAMIC SUMMARY N
# ============================================================================

def get_summary_n(n_total, max_n=10):
    """
    Dynamically determine how many rows to show in each section.
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


def cleanup_tqdm():
    """Clean up tqdm instances to prevent display clutter."""
    try:
        from tqdm import tqdm
        tqdm._instances.clear()
    except:
        pass


# ============================================================================
# 8. PRINT LOADED SEED SUMMARY
# ============================================================================

def print_loaded_seed_summary(results, seed, mode):
    """
    Print a rich summary table for a loaded seed.
    """
    print(f"\n📊 COMPLETED SEED {seed} SUMMARY ({mode.upper()} MODE)")
    print("─" * 70)

    best_sharpen = results.get('best_sharpen', {})
    champion = results.get('champion', {})
    spearman = results.get('spearman', {})
    version = results.get('version', 'unknown')

    best_sqr = best_sharpen.get('best_sqr', 'N/A')
    if isinstance(best_sqr, float):
        print(f"  Best SQR:              {best_sqr:.4f}")
    else:
        print(f"  Best SQR:              {best_sqr}")

    print(f"  Champion Trial:        {champion.get('trial', 'N/A')}")
    print(f"  Version:               {version}")

    lam1 = champion.get('lam1', 'N/A')
    lam2 = champion.get('lam2', 'N/A')
    if isinstance(lam1, float) and isinstance(lam2, float):
        print(f"  λ₁:                    {lam1:.4f},  λ₂: {lam2:.4f}")
    else:
        print(f"  λ₁:                    {lam1},  λ₂: {lam2}")

    beta_min_mult = champion.get('beta_min_mult', None)
    beta_max_mult = champion.get('beta_max_mult', None)
    if beta_min_mult is not None and beta_max_mult is not None:
        if isinstance(beta_min_mult, float) and isinstance(beta_max_mult, float):
            print(f"  β_min_mult:            {beta_min_mult:.4f},  β_max_mult: {beta_max_mult:.4f}")
        else:
            print(f"  β_min_mult:            {beta_min_mult},  β_max_mult: {beta_max_mult}")
        beta_min_abs = champion.get('beta_min_abs', None)
        beta_max_abs = champion.get('beta_max_abs', None)
        if beta_min_abs is not None and beta_max_abs is not None:
            if isinstance(beta_min_abs, float) and isinstance(beta_max_abs, float):
                print(f"  β_min_abs:             {beta_min_abs:.4f},  β_max_abs: {beta_max_abs:.4f}")
    else:
        beta_min = champion.get('beta_min', 'N/A')
        beta_max = champion.get('beta_max', 'N/A')
        if isinstance(beta_min, float) and isinstance(beta_max, float):
            print(f"  β_min:                 {beta_min:.4f},  β_max: {beta_max:.4f}")
        else:
            print(f"  β_min:                 {beta_min},  β_max: {beta_max}")

    sweeps = champion.get('num_sweeps', 'N/A')
    if isinstance(sweeps, (int, float)):
        print(f"  Sweeps:                {sweeps}")
    else:
        print(f"  Sweeps:                {sweeps}")

    selected = champion.get('selected_new', 'N/A')
    if isinstance(selected, list):
        if len(selected) <= 15:
            print(f"  Selected Stations:     {selected}")
        else:
            print(f"  Selected Stations:     {selected[:5]}... (total {len(selected)})")
    else:
        print(f"  Selected Stations:     {selected}")

    feas_rate = champion.get('feas_rate', 'N/A')
    if isinstance(feas_rate, float):
        print(f"  Feasibility:           {feas_rate*100:.1f}%")
    else:
        print(f"  Feasibility:           {feas_rate}")

    if spearman and spearman.get('available', False):
        print(f"  Spearman ρ:            {spearman['rho']:.4f}")
    else:
        print(f"  Spearman ρ:            N/A")

    total_time = results.get('total_time', 'N/A')
    if isinstance(total_time, (int, float)):
        print(f"  Total Time:            {total_time/60:.2f} min")
    else:
        print(f"  Total Time:            {total_time}")

    print("─" * 70)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    'execute_phase',
    'PHASE_TIMES',
    'safe_load_pickle',
    'safe_save_pickle',
    'safe_load_json',
    'NumpyEncoder',
    'build_config',
    'compute_spearman_correlation',
    'extract_top3_champions',
    'get_summary_n',
    'cleanup_tqdm',
    'print_loaded_seed_summary',
]