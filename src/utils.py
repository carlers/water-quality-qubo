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
    10. Enhanced summary printers for tuning, validation, sharpening, and global results.
    11. Single-run summary printer
    12. Cross-strategy summary printer (dynamic ranked table)
    13. Optuna trial log suppressor
    14. Config header printer
    15. Cross-strategy metrics (correlations, efficiency, CV, rankings)
    16. Horizontal formatting helpers
    17. JijModeling tuning summary
    18. JijModeling benchmark summary
    19. JijModeling single run summary
    20. JijModeling result saving/loading (crash recovery)
    21. NEW: compute_violation_rate
    22. NEW: compute_matrix_differences
    23. NEW: print_multiobjective_tuning_summary
    24. NEW: print_benchmark_summary
    25. NEW: select_best_from_pareto
    26. NEW: build_full_qubo_matrix

All plotting functions have been moved to src/plotting.py.
"""

import json
import pickle
import time
import gc
import contextlib
import logging
import math
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
# 8. PRINT LOADED SEED SUMMARY (kept for compatibility)
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
    if isinstance(best_sqr, (int, float)) and np.isfinite(best_sqr):
        print(f"  Best SQR:              {best_sqr:.4f}")
    else:
        print(f"  Best SQR:              {best_sqr}")

    print(f"  Champion Trial:        {champion.get('trial', 'N/A')}")
    print(f"  Version:               {version}")

    lam1 = champion.get('lam1', 'N/A')
    lam2 = champion.get('lam2', 'N/A')
    if isinstance(lam1, (int, float)) and isinstance(lam2, (int, float)):
        print(f"  λ₁:                    {lam1:.4f},  λ₂: {lam2:.4f}")
    else:
        print(f"  λ₁:                    {lam1},  λ₂: {lam2}")

    beta_min_mult = champion.get('beta_min_mult', None)
    beta_max_mult = champion.get('beta_max_mult', None)
    if beta_min_mult is not None and beta_max_mult is not None:
        if isinstance(beta_min_mult, (int, float)) and isinstance(beta_max_mult, (int, float)):
            print(f"  β_min_mult:            {beta_min_mult:.4f},  β_max_mult: {beta_max_mult:.4f}")
        else:
            print(f"  β_min_mult:            {beta_min_mult},  β_max_mult: {beta_max_mult}")
        beta_min_abs = champion.get('beta_min_abs', None)
        beta_max_abs = champion.get('beta_max_abs', None)
        if beta_min_abs is not None and beta_max_abs is not None:
            if isinstance(beta_min_abs, (int, float)) and isinstance(beta_max_abs, (int, float)):
                print(f"  β_min_abs:             {beta_min_abs:.4f},  β_max_abs: {beta_max_abs:.4f}")
    else:
        beta_min = champion.get('beta_min', 'N/A')
        beta_max = champion.get('beta_max', 'N/A')
        if isinstance(beta_min, (int, float)) and isinstance(beta_max, (int, float)):
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
    if isinstance(feas_rate, (int, float)):
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
# 9. ENHANCED SUMMARY PRINTERS (v7)
# ============================================================================

def _safe_format(val, fmt=".4f"):
    """Safely format a value, returning 'N/A' if not numeric."""
    if isinstance(val, (int, float)) and np.isfinite(val):
        return f"{val:{fmt}}"
    return "N/A"


# -----------------------------------------------------------------------------
# 9a. Horizontal formatting helpers
# -----------------------------------------------------------------------------

def _format_horizontal(items, width=200, sep="  "):
    """
    Format a list of (key, value) pairs into a single line with columns.
    """
    # Determine max key length
    max_key_len = max(len(str(k)) for k, _ in items) if items else 0
    # Try to fit as many as possible in one line
    lines = []
    current_line = []
    current_len = 0
    for key, val in items:
        entry = f"{key}={val}"
        entry_len = len(entry) + len(sep)
        if current_len + entry_len > width and current_line:
            lines.append(sep.join(current_line))
            current_line = [entry]
            current_len = entry_len
        else:
            current_line.append(entry)
            current_len += entry_len
    if current_line:
        lines.append(sep.join(current_line))
    return "\n".join(lines)


def _format_dict_horizontal(d, width=200, sep="  "):
    """Format a dictionary into horizontal lines."""
    items = [(str(k), str(v)) for k, v in d.items()]
    return _format_horizontal(items, width, sep)


# -----------------------------------------------------------------------------
# 9b. Updated summary printers with horizontal option
# -----------------------------------------------------------------------------

def print_tuning_summary(study, experiment_name, compact=True, width=200):
    """
    Print a detailed summary of the best tuning trial from an Optuna study.
    If compact=True, uses horizontal layout to save vertical space.
    """
    if study is None or len(study.trials) == 0:
        print(f"⚠️ No trials found for experiment '{experiment_name}'.")
        return

    best = study.best_trial
    print("\n" + "=" * 80)
    print(f"📊 TUNING SUMMARY: {experiment_name}")
    print("=" * 80)
    print(f"  Best Trial #: {best.number}")
    print(f"  Best Objective Value: {_safe_format(best.value, '.6f')}")

    if not compact:
        print("\n  Hyperparameters:")
        for key, val in best.params.items():
            if isinstance(val, float):
                print(f"    {key:20s}: {_safe_format(val, '.6f')}")
            else:
                print(f"    {key:20s}: {val}")

        print("\n  User Attributes (metrics):")
        attrs = best.user_attrs
        solution = attrs.get('best_solution', None)
        for key, val in attrs.items():
            if key == 'best_solution':
                continue
            if isinstance(val, float):
                print(f"    {key:20s}: {_safe_format(val, '.6f')}")
            else:
                print(f"    {key:20s}: {val}")
    else:
        # Horizontal format
        # Combine params and user attrs into a single dict
        combined = {**best.params, **{k: v for k, v in best.user_attrs.items() if k != 'best_solution'}}
        # Format as horizontal lines
        items = [(k, _safe_format(v, '.6f') if isinstance(v, float) else v) for k, v in combined.items()]
        print("\n  " + _format_horizontal(items, width=width))

        # Show solution preview
        solution = best.user_attrs.get('best_solution')
        if solution is not None:
            selected = [i for i, v in enumerate(solution) if v == 1]
            if len(selected) <= 15:
                idx_str = str(selected)
            else:
                idx_str = str(selected[:10]) + f" ... (total {len(selected)})"
            print(f"  best_solution={idx_str}")

    print("=" * 80)


def print_validation_summary(val_results, experiment_name, compact=True, width=200):
    """
    Print a detailed summary of the validation champion.
    If compact=True, uses horizontal layout to save vertical space.
    """
    if val_results is None or not val_results.get('trials'):
        print(f"⚠️ No validation results for experiment '{experiment_name}'.")
        return

    # The first trial in 'trials' is the best (sorted by best_sqr descending)
    best_trial = val_results['trials'][0]

    print("\n" + "=" * 80)
    print(f"📊 VALIDATION SUMMARY: {experiment_name}")
    print("=" * 80)

    if not compact:
        print(f"  Best Validation SQR:   {_safe_format(best_trial.get('best_sqr', 'N/A'), '.6f')}")
        print(f"  Feasibility Rate:       {_safe_format(best_trial.get('feas_rate', 'N/A'), '.6f')}")
        print(f"  Trial #:                {best_trial.get('trial_number', 'N/A')}")
        print(f"  λ₁:                     {_safe_format(best_trial.get('lam1', 'N/A'), '.6f')}")
        print(f"  λ₂:                     {_safe_format(best_trial.get('lam2', 'N/A'), '.6f')}")
        print(f"  num_sweeps:             {best_trial.get('num_sweeps', 'N/A')}")
        if 'ESR' in best_trial:
            print(f"  ESR:                    {_safe_format(best_trial['ESR'], '.4f')}")
        if 'MCR' in best_trial:
            print(f"  MCR:                    {_safe_format(best_trial['MCR'], '.4f')}")
        if best_trial.get('solution') is not None:
            sol = best_trial['solution']
            selected = [i for i, v in enumerate(sol) if v == 1]
            if len(selected) <= 15:
                idx_str = str(selected)
            else:
                idx_str = str(selected[:10]) + f" ... (total {len(selected)})"
            print(f"  Selected stations:      {idx_str}")
        else:
            print("  Selected stations:      Not available")
        print(f"  Spearman ρ:             {_safe_format(val_results.get('spearman_rho', 'N/A'), '.4f')}")
        print(f"  Spearman p-value:       {_safe_format(val_results.get('spearman_p', 'N/A'), '.4f')}")
        print(f"  Validated trials:       {val_results.get('n_validated', 0)}")
    else:
        # Horizontal format
        items = [
            ("best_sqr", _safe_format(best_trial.get('best_sqr'), '.6f')),
            ("feas", _safe_format(best_trial.get('feas_rate'), '.4f')),
            ("trial", best_trial.get('trial_number', 'N/A')),
            ("lam1", _safe_format(best_trial.get('lam1'), '.6f')),
            ("lam2", _safe_format(best_trial.get('lam2'), '.6f')),
            ("sweeps", best_trial.get('num_sweeps', 'N/A')),
        ]
        if 'ESR' in best_trial:
            items.append(("ESR", _safe_format(best_trial['ESR'], '.4f')))
        if 'MCR' in best_trial:
            items.append(("MCR", _safe_format(best_trial['MCR'], '.4f')))
        # Show selected stations preview
        if best_trial.get('solution') is not None:
            sol = best_trial['solution']
            selected = [i for i, v in enumerate(sol) if v == 1]
            if len(selected) <= 15:
                idx_str = str(selected)
            else:
                idx_str = str(selected[:10]) + f" ... ({len(selected)} total)"
            items.append(("selected", idx_str))
        items.append(("rho", _safe_format(val_results.get('spearman_rho'), '.4f')))
        items.append(("p", _safe_format(val_results.get('spearman_p'), '.4f')))
        items.append(("n_val", val_results.get('n_validated', 0)))

        print("\n  " + _format_horizontal(items, width=width))

    print("=" * 80)


def print_sharpening_summary(best_sharpen, experiment_name):
    """
    Print a detailed summary of the best sharpening result.
    """
    if best_sharpen is None:
        print(f"⚠️ No sharpening results for experiment '{experiment_name}'.")
        return

    print("\n" + "=" * 80)
    print(f"📊 SHARPENING SUMMARY: {experiment_name}")
    print("=" * 80)
    print(f"  Best Sharpening SQR:    {_safe_format(best_sharpen.get('best_sqr', 'N/A'), '.6f')}")
    print(f"  Feasibility Rate:       {_safe_format(best_sharpen.get('feas_rate', 'N/A'), '.6f')}")
    print(f"  Trial #:                {best_sharpen.get('trial', 'N/A')}")
    print(f"  Run #:                  {best_sharpen.get('run', 'N/A')}")
    print(f"  λ₁:                     {_safe_format(best_sharpen.get('lam1', 'N/A'), '.6f')}")
    print(f"  λ₂:                     {_safe_format(best_sharpen.get('lam2', 'N/A'), '.6f')}")
    print(f"  num_sweeps:             {best_sharpen.get('num_sweeps', 'N/A')}")
    if 'ESR' in best_sharpen:
        print(f"  ESR:                    {_safe_format(best_sharpen['ESR'], '.4f')}")
    if 'MCR' in best_sharpen:
        print(f"  MCR:                    {_safe_format(best_sharpen['MCR'], '.4f')}")
    # Selected stations
    if best_sharpen.get('solution') is not None:
        sol = best_sharpen['solution']
        selected = [i for i, v in enumerate(sol) if v == 1]
        if len(selected) <= 15:
            idx_str = str(selected)
        else:
            idx_str = str(selected[:10]) + f" ... (total {len(selected)})"
        print(f"  Selected stations:      {idx_str}")
    else:
        print("  Selected stations:      Not available")
    print("=" * 80)


def print_global_summary(results, experiment_name):
    """
    Print a final global summary table, including phase runtimes and key metrics.
    Expects a results dictionary containing phase_times, total_time, champion, best_sharpen, validation_results, spearman.
    """
    print("\n" + "=" * 80)
    print(f"🌍 GLOBAL SUMMARY: {experiment_name}")
    print("=" * 80)

    # Phase timings
    phase_times = results.get('phase_times', {})
    total_time = results.get('total_time', 0)
    print("\n⏱️  PHASE RUNTIMES")
    print("-" * 40)
    if phase_times:
        for phase, t in phase_times.items():
            print(f"  {phase:20s}: {_safe_format(t/60, '.2f')} minutes")
    else:
        print("  (No phase timings recorded)")
    if isinstance(total_time, (int, float)) and np.isfinite(total_time):
        print(f"  {'Total':20s}: {_safe_format(total_time/60, '.2f')} minutes")
    else:
        print(f"  {'Total':20s}: N/A")

    # Key metrics
    champion = results.get('champion', {})
    best_sharpen = results.get('best_sharpen', {})
    val_results = results.get('validation_results', {})
    spearman = results.get('spearman', {})

    print("\n📈 KEY METRICS")
    print("-" * 40)
    print(f"  Best Tuning SQR:       {_safe_format(champion.get('best_sqr', 'N/A'), '.6f')}")
    print(f"  Best Validation SQR:   {_safe_format(val_results.get('best_sqr', 'N/A'), '.6f')}")
    print(f"  Best Sharpening SQR:   {_safe_format(best_sharpen.get('best_sqr', 'N/A'), '.6f')}")
    print(f"  Validation Feasibility: {_safe_format(val_results.get('feas_rate', 'N/A'), '.4f')}")
    if spearman.get('available', False):
        print(f"  Spearman ρ:             {_safe_format(spearman.get('rho', 'N/A'), '.4f')}")
    else:
        print(f"  Spearman ρ:             N/A")

    # Final hyperparameters
    print("\n🔧 FINAL HYPERPARAMETERS (Sharpening Champion)")
    print("-" * 40)
    print(f"  λ₁:                     {_safe_format(best_sharpen.get('lam1', 'N/A'), '.6f')}")
    print(f"  λ₂:                     {_safe_format(best_sharpen.get('lam2', 'N/A'), '.6f')}")
    print(f"  num_sweeps:             {best_sharpen.get('num_sweeps', 'N/A')}")
    if 'beta_min_mult' in best_sharpen:
        print(f"  β_min_mult:             {_safe_format(best_sharpen.get('beta_min_mult', 'N/A'), '.4f')}")
        print(f"  β_max_mult:             {_safe_format(best_sharpen.get('beta_max_mult', 'N/A'), '.4f')}")
    elif 'beta_min' in best_sharpen:
        print(f"  β_min:                  {_safe_format(best_sharpen.get('beta_min', 'N/A'), '.4f')}")
        print(f"  β_max:                  {_safe_format(best_sharpen.get('beta_max', 'N/A'), '.4f')}")
    if 'cooling_power' in best_sharpen:
        print(f"  cooling_power:          {_safe_format(best_sharpen.get('cooling_power', 'N/A'), '.4f')}")

    # Final selected stations
    if best_sharpen.get('solution') is not None:
        sol = best_sharpen['solution']
        selected = [i for i, v in enumerate(sol) if v == 1]
        if len(selected) <= 15:
            idx_str = str(selected)
        else:
            idx_str = str(selected[:10]) + f" ... (total {len(selected)})"
        print(f"\n📍 FINAL SELECTED STATIONS: {idx_str}")
    else:
        print("\n📍 FINAL SELECTED STATIONS: Not available")

    print("=" * 80)


# ============================================================================
# 10. SINGLE-RUN SUMMARY (for strategy comparison)
# ============================================================================

def print_single_run_summary(config: Dict, result: Dict, elapsed: float):
    """
    Print a concise yet rich summary for a single completed run.
    Now uses effective SQR = max(tuning_sqr, val_sqr) and also MIQP energy.
    """
    print("\n" + "=" * 80)
    print("📌 SINGLE RUN COMPLETE")
    print("=" * 80)
    print(f"  Seed:            {config['seed']}")
    print(f"  Objective:       {config['objective']}")
    print(f"  Tuning Trials:   {config['tuning_trials']}")
    print(f"  Tuning Reads:    {config['tuning_reads']}")
    print(f"  Validation TopK: {config['val_top_k']}")
    print(f"  Validation Reads:{config['val_reads']}")
    print(f"  N (candidates):  {config['N']}")
    print("-" * 40)

    # Effective SQR (max of tuning and validation)
    tuning_sqr = result.get('tuning_best_sqr', np.nan)
    val_sqr = result.get('best_sqr', np.nan)
    effective_sqr = max(tuning_sqr, val_sqr) if not (np.isnan(tuning_sqr) or np.isnan(val_sqr)) else (tuning_sqr if not np.isnan(tuning_sqr) else val_sqr)
    sqr_source = "tuning" if effective_sqr == tuning_sqr and not np.isnan(tuning_sqr) else "validation"

    print(f"  Best SQR:        {_safe_format(effective_sqr, '.6f')} (from {sqr_source})")
    print(f"  Tuning SQR:      {_safe_format(tuning_sqr, '.6f')}")
    print(f"  Validation SQR:  {_safe_format(val_sqr, '.6f')}")
    print(f"  Feasibility:     {_safe_format(result.get('feas_rate', np.nan), '.4f')}")
    if 'spearman_rho' in result and result['spearman_rho'] is not None and not np.isnan(result['spearman_rho']):
        print(f"  Spearman ρ:      {_safe_format(result['spearman_rho'], '.4f')}")
    else:
        print("  Spearman ρ:      N/A")
    print(f"  MIQP Energy:     {_safe_format(result.get('miqp_energy', np.nan), '.6f')}")
    print(f"  Total Samples:   {result.get('total_samples', 'N/A')}")
    print(f"  Runtime:         {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print("=" * 80)


# ============================================================================
# 11. CROSS-STRATEGY SUMMARY (dynamic ranked table) - with nan handling
# ============================================================================

def print_cross_strategy_summary(results_df: pd.DataFrame, title: str = "Cross-Strategy Summary", top_n: int = 10):
    """
    Print a dynamic ranked table of all completed configurations so far.
    Uses nan-safe aggregation.
    """
    if results_df.empty:
        print("\n⚠️ No completed configurations yet.")
        return

    # Ensure we have a 'time_seconds' column; if not, compute from runtime
    if 'time_seconds' not in results_df.columns and 'runtime' in results_df.columns:
        results_df['time_seconds'] = results_df['runtime']
    elif 'time_seconds' not in results_df.columns:
        results_df['time_seconds'] = np.nan

    # Create a configuration label
    results_df['config'] = (
        results_df['objective'] + " T" + results_df['tuning_trials'].astype(str) +
        " R" + results_df['tuning_reads'].astype(str) +
        " K" + results_df['val_top_k'].astype(str) +
        " V" + results_df['val_reads'].astype(str)
    )

    # Group by config (excluding seed) and aggregate using nan-safe functions
    agg = results_df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
        'best_sqr': ['mean', 'std', 'count'],
        'feas_rate': ['mean', 'std'],
        'spearman_rho': ['mean', 'std'],
        'total_samples': ['mean'],
        'time_seconds': ['mean', 'std']
    }).reset_index()

    # Flatten columns
    agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                   'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                   'feas_mean', 'feas_std',
                   'rho_mean', 'rho_std',
                   'samples_mean',
                   'time_mean', 'time_std']

    # Replace NaN std with 0 for single seed
    agg['best_sqr_std'] = agg['best_sqr_std'].fillna(0)
    agg['time_std'] = agg['time_std'].fillna(0)

    # Sort by best_sqr_mean descending
    agg_sorted = agg.sort_values('best_sqr_mean', ascending=False)

    # Display top_n rows
    print("\n" + "=" * 80)
    print(f"📊 {title} (completed configs)")
    print("=" * 80)
    print(f"Total completed runs: {len(results_df)}")
    print(f"Unique configurations: {len(agg_sorted)}")
    print("-" * 80)

    # Select columns to show
    display_cols = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads',
                    'N', 'best_sqr_mean', 'best_sqr_std', 'feas_mean', 'rho_mean', 'samples_mean', 'time_mean']
    # Format float columns
    formatted = agg_sorted[display_cols].head(top_n).copy()
    for col in ['best_sqr_mean', 'best_sqr_std', 'feas_mean', 'rho_mean', 'samples_mean', 'time_mean']:
        formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    # Format time as minutes
    formatted['time_mean'] = formatted['time_mean'].apply(
        lambda x: f"{x/60:.2f} min" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
    )

    print(formatted.to_string(index=False))
    if len(agg_sorted) > top_n:
        print(f"\n... and {len(agg_sorted)-top_n} more configurations (see CSV).")
    print("=" * 80)


# ============================================================================
# 12. SUPPRESS OPTUNA TRIAL LOGS
# ============================================================================

def suppress_optuna_trial_logs():
    """
    Suppress Optuna's per-trial logging output (e.g., "Trial 0 finished with value: ...")
    but keep the progress bar visible.
    """
    optuna_logger = logging.getLogger('optuna')
    optuna_logger.setLevel(logging.WARNING)


# ============================================================================
# 13. NEW: CONFIG HEADER PRINTER
# ============================================================================

def print_config_header(config: Dict):
    """
    Print a one-line header before running a configuration.
    """
    print("\n" + "─" * 80)
    print(f"▶ Config: Seed={config['seed']}, Obj={config['objective']}, "
          f"T={config['tuning_trials']}, R={config['tuning_reads']}, "
          f"K={config['val_top_k']}, V={config['val_reads']}, N={config['N']}")
    print("─" * 80)


# ============================================================================
# 14. NEW: CROSS-STRATEGY METRICS (correlations, efficiency, CV, rankings)
# ============================================================================

def print_cross_strategy_metrics(results_df: pd.DataFrame, title: str = "Cross-Strategy Metrics"):
    """
    Compute and print:
    - Spearman correlation: SQR vs Runtime, SQR vs Samples
    - Efficiency: SQR / Runtime, SQR / Samples
    - Coefficient of Variation (CV) for each config
    - Rankings by Efficiency, SQR, and CV
    """
    if results_df.empty:
        print("\n⚠️ No data for cross-strategy metrics.")
        return

    # Ensure we have runtime and samples
    if 'time_seconds' not in results_df.columns and 'runtime' in results_df.columns:
        results_df['time_seconds'] = results_df['runtime']
    if 'time_seconds' not in results_df.columns:
        results_df['time_seconds'] = np.nan

    # Group by config (excluding seed) and aggregate with nan-safe functions
    agg = results_df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
        'best_sqr': ['mean', 'std', 'count'],
        'feas_rate': ['mean', 'std'],
        'spearman_rho': ['mean', 'std'],
        'total_samples': ['mean'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                   'sqr_mean', 'sqr_std', 'n_seeds',
                   'feas_mean', 'feas_std',
                   'rho_mean', 'rho_std',
                   'samples_mean',
                   'time_mean', 'time_std']

    # Drop rows where sqr_mean is NaN
    agg = agg[agg['sqr_mean'].notna()]
    if agg.empty:
        print("⚠️ No valid configurations with SQR.")
        return

    # --- 1. Compute Spearman correlations (across configurations, using mean SQR) ---
    valid_time = agg[agg['time_mean'].notna() & (agg['time_mean'] > 0)]
    valid_samples = agg[agg['samples_mean'].notna() & (agg['samples_mean'] > 0)]

    rho_sqr_time = np.nan
    p_sqr_time = np.nan
    if len(valid_time) >= 3:
        rho_sqr_time, p_sqr_time = stats.spearmanr(valid_time['sqr_mean'], valid_time['time_mean'], nan_policy='omit')

    rho_sqr_samples = np.nan
    p_sqr_samples = np.nan
    if len(valid_samples) >= 3:
        rho_sqr_samples, p_sqr_samples = stats.spearmanr(valid_samples['sqr_mean'], valid_samples['samples_mean'], nan_policy='omit')

    # --- 2. Efficiency (SQR per runtime and per sample) ---
    agg['efficiency_time'] = agg['sqr_mean'] / agg['time_mean']
    agg['efficiency_samples'] = agg['sqr_mean'] / agg['samples_mean']

    # --- 3. Coefficient of Variation (CV) ---
    agg['cv'] = agg['sqr_std'] / agg['sqr_mean']

    # --- 4. Rankings ---
    rank_by_sqr = agg.sort_values('sqr_mean', ascending=False).reset_index(drop=True)
    rank_by_eff_time = agg.sort_values('efficiency_time', ascending=False).reset_index(drop=True)
    rank_by_cv = agg.sort_values('cv', ascending=True).reset_index(drop=True)  # lower CV is better

    # --- Print ---
    print("\n" + "=" * 80)
    print(f"📈 {title}")
    print("=" * 80)

    print("\n🔗 Correlations (across configurations):")
    if not np.isnan(rho_sqr_time):
        print(f"  Spearman ρ(SQR, Runtime): {rho_sqr_time:.4f}  (p={p_sqr_time:.4f})")
    else:
        print("  Spearman ρ(SQR, Runtime): N/A (insufficient data)")
    if not np.isnan(rho_sqr_samples):
        print(f"  Spearman ρ(SQR, Samples): {rho_sqr_samples:.4f}  (p={p_sqr_samples:.4f})")
    else:
        print("  Spearman ρ(SQR, Samples): N/A (insufficient data)")

    print("\n🏆 Rank by Efficiency (SQR / Runtime):")
    print(rank_by_eff_time[['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                            'sqr_mean', 'time_mean', 'efficiency_time']].head(5).to_string(index=False, float_format="%.4f"))

    print("\n🏆 Rank by Best SQR:")
    print(rank_by_sqr[['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                       'sqr_mean', 'sqr_std', 'feas_mean']].head(5).to_string(index=False, float_format="%.4f"))

    print("\n🏆 Rank by Robustness (lowest CV):")
    print(rank_by_cv[['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                      'sqr_mean', 'sqr_std', 'cv']].head(5).to_string(index=False, float_format="%.4f"))

    print("=" * 80)


# =============================================================================
# ADDITIONS TO src/utils.py (JijModeling Pipeline)
# =============================================================================

# -----------------------------------------------------------------------------
# JijModeling tuning summary
# -----------------------------------------------------------------------------
def print_jij_tuning_summary(
    best_params: Dict,
    best_value: float,
    solver_name: str = "SA",
    title: str = "Tuning Summary",
) -> None:
    """
    Print a concise summary of Optuna tuning results.
    """
    print("\n" + "=" * 80)
    print(f"📊 {title} – {solver_name} Tuning")
    print("=" * 80)
    print(f"  Best objective (energy): {_safe_format(best_value, '.6f')}")
    print("  Best hyperparameters:")
    for key, val in best_params.items():
        if isinstance(val, float):
            print(f"    {key:20s}: {_safe_format(val, '.6f')}")
        else:
            print(f"    {key:20s}: {val}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# JijModeling benchmark summary
# -----------------------------------------------------------------------------
def print_jij_benchmark_summary(
    df: pd.DataFrame,
    title: str = "Benchmark Summary",
) -> None:
    """
    Print a formatted summary table for benchmark results.
    Expects columns: solver, N, sqr, runtime, feasible, energy, status.
    """
    if df.empty:
        print("⚠️ No data to summarise.")
        return

    # Group by solver and N
    agg = df.groupby(["solver", "N"]).agg({
        "sqr": ["mean", "std"],
        "runtime": ["mean", "std"],
        "feasible": "mean",
        "energy": ["mean", "std"],
    }).reset_index()
    agg.columns = [
        "solver", "N",
        "sqr_mean", "sqr_std",
        "runtime_mean", "runtime_std",
        "feasibility",
        "energy_mean", "energy_std"
    ]

    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    print(f"Total runs: {len(df)}")
    print(f"Unique configs: {len(agg)}")
    print("-" * 80)

    # Select columns to display
    display_cols = ["solver", "N", "sqr_mean", "sqr_std", "runtime_mean", "runtime_std", "feasibility"]
    formatted = agg[display_cols].copy()
    formatted["sqr_mean"] = formatted["sqr_mean"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    formatted["sqr_std"] = formatted["sqr_std"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    formatted["runtime_mean"] = formatted["runtime_mean"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    formatted["runtime_std"] = formatted["runtime_std"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    formatted["feasibility"] = formatted["feasibility"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(formatted.to_string(index=False))
    print("=" * 80)


# -----------------------------------------------------------------------------
# JijModeling single run summary
# -----------------------------------------------------------------------------
def print_jij_single_run(
    config: Dict,
    result: Dict,
    elapsed: float,
) -> None:
    """
    Print a concise summary for a single solver run.
    config: dict with keys like 'solver', 'N', 'K', 'seed', 'lambda_budget', 'lambda_conn', ...
    result: dict with keys 'energy', 'feasible', 'status'
    """
    solver = config.get("solver", "Unknown")
    N = config.get("N", "?")
    K = config.get("K", "?")
    seed = config.get("seed", "?")
    energy = result.get("energy", np.nan)
    feasible = result.get("feasible", False)
    status = result.get("status", "")

    print(f"\n  {solver:6s} | N={N:3d} K={K:2d} seed={seed:2d} | "
          f"energy={_safe_format(energy, '.6f')} | feasible={feasible} | "
          f"runtime={elapsed:.2f}s | status={status}")


# -----------------------------------------------------------------------------
# JijModeling result saving/loading (crash recovery)
# -----------------------------------------------------------------------------
def save_jij_results(
    results: List[Dict],
    completed: set,
    stage: Union[int, str],
    save_dir: Union[str, Path],
) -> None:
    """
    Save results and completed set for a given stage.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_file = save_dir / f"jij_stage{stage}_results.pkl"
    completed_file = save_dir / f"jij_stage{stage}_completed.pkl"
    csv_file = save_dir / f"jij_stage{stage}_results.csv"

    safe_save_pickle(results_file, results, verbose=False)
    safe_save_pickle(completed_file, completed, verbose=False)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)


def load_jij_results(
    stage: Union[int, str],
    save_dir: Union[str, Path],
) -> tuple[List[Dict], set]:
    """
    Load results and completed set for a given stage.
    Returns (results_list, completed_set).
    """
    save_dir = Path(save_dir)
    results_file = save_dir / f"jij_stage{stage}_results.pkl"
    completed_file = save_dir / f"jij_stage{stage}_completed.pkl"

    results = safe_load_pickle(results_file, [])
    completed = safe_load_pickle(completed_file, set())
    return results, completed


# =============================================================================
# NEW FUNCTIONS FOR MULTI-OBJECTIVE TUNING
# =============================================================================

def compute_violation_rate(
    x_sol: np.ndarray,
    neigh: np.ndarray,
    K: int
) -> float:
    """
    Compute violation rate for a solution.
    
    Returns:
        0.0  -> both budget and connectivity constraints satisfied
        0.5  -> exactly one constraint violated
        1.0  -> both constraints violated
    """
    if x_sol is None or neigh is None:
        return 1.0
    
    x_sol = np.asarray(x_sol)
    neigh = np.asarray(neigh)
    selected = np.where(x_sol == 1)[0]
    
    # Budget constraint
    budget_ok = (len(selected) == K)
    
    # Connectivity constraint
    if len(selected) == 0:
        conn_ok = False
    else:
        conn_ok = all(np.any(neigh[i, selected] == 1) for i in selected)
    
    if budget_ok and conn_ok:
        return 0.0
    elif budget_ok or conn_ok:
        return 0.5
    else:
        return 1.0


def compute_matrix_differences(
    mat1: np.ndarray,
    mat2: np.ndarray
) -> Dict[str, float]:
    """
    Compute difference metrics between two matrices.
    
    Args:
        mat1: First matrix (e.g., MIQP objective matrix)
        mat2: Second matrix (e.g., QUBO matrix with penalties)
    
    Returns:
        dict with 'MSE', 'RMSE', 'Frobenius' keys
    """
    if mat1.shape != mat2.shape:
        raise ValueError(f"Matrix shapes must match: {mat1.shape} vs {mat2.shape}")
    
    diff = mat1 - mat2
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    frob = np.linalg.norm(diff, 'fro')
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'Frobenius': float(frob)
    }


def build_full_qubo_matrix(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    N: int
) -> np.ndarray:
    """
    Build a symmetric full N×N QUBO matrix from h (linear) and J (quadratic) dicts.
    
    Args:
        h: dict {i: coeff} for linear terms
        J: dict {(i,j): coeff} for quadratic terms (i < j)
        N: total number of variables
    
    Returns:
        N×N symmetric matrix
    """
    Q = np.zeros((N, N))
    
    # Linear terms (on diagonal)
    for i, coeff in h.items():
        if i < N:
            Q[i, i] += coeff
    
    # Quadratic terms (off-diagonal, symmetric)
    for (i, j), coeff in J.items():
        if i < N and j < N:
            if i == j:
                Q[i, i] += coeff
            else:
                Q[i, j] += coeff
                Q[j, i] += coeff
    
    return Q


def select_best_from_pareto(
    study: 'optuna.Study'
) -> Optional['optuna.trial.FrozenTrial']:
    """
    Select the best trial from the Pareto front.
    
    Selection criteria:
        1. Prefer trials with violation_rate = 0.0
        2. Among those, select the one with lowest MIQP energy
        3. If no zero-violation trials, select the one with lowest violation_rate
        4. Among equal violation_rate, select lowest MIQP energy
    
    Returns:
        The selected FrozenTrial, or None if no trials exist
    """
    pareto_trials = study.best_trials
    
    if not pareto_trials:
        return None
    
    # Filter by violation_rate (stored in user_attrs)
    zero_violation = []
    for t in pareto_trials:
        vio = t.user_attrs.get('violation_rate', 1.0)
        if isinstance(vio, (int, float)) and vio == 0.0:
            zero_violation.append(t)
    
    if zero_violation:
        # Pick the one with lowest MIQP energy (first objective)
        best = min(zero_violation, key=lambda t: t.values[0] if t.values else float('inf'))
    else:
        # Pick the one with lowest violation_rate, then lowest MIQP
        best = min(pareto_trials, key=lambda t: (
            t.user_attrs.get('violation_rate', 1.0),
            t.values[0] if t.values else float('inf')
        ))
    
    return best


def print_multiobjective_tuning_summary(
    study: 'optuna.Study',
    title: str = "Multi-Objective Tuning Summary",
    verbose: bool = True
) -> None:
    """
    Print a comprehensive summary for a multi-objective Optuna study.
    
    Displays:
        - Number of trials
        - Number of Pareto-optimal trials
        - Best trial selected by our heuristic
        - Hyperparameters of the best trial
        - Objective values of the best trial
        - User attributes (violation_rate, ESR, MCR, etc.)
    """
    if study is None or len(study.trials) == 0:
        print(f"⚠️ No trials found for {title}.")
        return
    
    n_trials = len(study.trials)
    n_pareto = len(study.best_trials)
    
    best_trial = select_best_from_pareto(study)
    
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    print(f"  Total trials:            {n_trials}")
    print(f"  Pareto-optimal trials:   {n_pareto}")
    print("-" * 40)
    
    if best_trial is None:
        print("  No valid trials found.")
        print("=" * 80)
        return
    
    print("  Best Trial (selected):")
    print(f"    Trial #:               {best_trial.number}")
    
    if best_trial.values:
        print(f"    MIQP Energy:           {_safe_format(best_trial.values[0], '.6f')}")
        if len(best_trial.values) > 1:
            print(f"    Violation Rate:         {_safe_format(best_trial.values[1], '.2f')}")
    
    print("    Hyperparameters:")
    for key, val in best_trial.params.items():
        if isinstance(val, float):
            print(f"      {key:20s}: {_safe_format(val, '.6f')}")
        else:
            print(f"      {key:20s}: {val}")
    
    # User attributes
    if best_trial.user_attrs:
        print("    User Attributes:")
        attrs = best_trial.user_attrs
        for key, val in attrs.items():
            if key == 'best_solution' and val is not None:
                # Summarise solution
                if isinstance(val, list):
                    selected = [i for i, v in enumerate(val) if v == 1]
                    if len(selected) <= 15:
                        val_str = str(selected)
                    else:
                        val_str = str(selected[:10]) + f" ... (total {len(selected)})"
                    print(f"      {key:20s}: {val_str}")
                else:
                    print(f"      {key:20s}: {val}")
            elif isinstance(val, float):
                print(f"      {key:20s}: {_safe_format(val, '.6f')}")
            else:
                print(f"      {key:20s}: {val}")
    
    print("=" * 80)


def print_benchmark_summary(
    results_by_N: Dict[int, Dict],
    title: str = "Benchmark Summary"
) -> None:
    """
    Print a summary table for benchmark results across different N.
    
    Args:
        results_by_N: dict mapping N -> dict with 'gurobi', 'greedy', 'sa', 'sqa' results
        title: Title for the summary
    """
    if not results_by_N:
        print("⚠️ No benchmark results to summarise.")
        return
    
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    # Collect rows
    rows = []
    for N, results in sorted(results_by_N.items()):
        for solver_name, res in results.items():
            if res is None:
                continue
            rows.append({
                'N': N,
                'Solver': solver_name,
                'SQR': res.get('sqr', np.nan),
                'Runtime (s)': res.get('runtime', np.nan),
                'Violation Rate': res.get('violation_rate', np.nan),
                'Feasible': res.get('feasible', False),
                'Energy': res.get('energy', np.nan),
            })
    
    if not rows:
        print("  No data available.")
        print("=" * 80)
        return
    
    df = pd.DataFrame(rows)
    
    # Pivot for cleaner display
    print("\n  SQR by N and Solver:")
    pivot_sqr = df.pivot(index='N', columns='Solver', values='SQR')
    print(pivot_sqr.round(4).to_string())
    
    print("\n  Runtime (seconds) by N and Solver:")
    pivot_time = df.pivot(index='N', columns='Solver', values='Runtime (s)')
    print(pivot_time.round(4).to_string())
    
    print("\n  Violation Rate by N and Solver:")
    pivot_viol = df.pivot(index='N', columns='Solver', values='Violation Rate')
    print(pivot_viol.round(4).to_string())
    
    # Additional stats
    print("\n  Feasibility Summary:")
    feas_summary = df.groupby(['N', 'Solver'])['Feasible'].mean().unstack()
    print(feas_summary.round(4).to_string())
    
    print("=" * 80)


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
    'print_tuning_summary',
    'print_validation_summary',
    'print_sharpening_summary',
    'print_global_summary',
    'print_single_run_summary',
    'print_cross_strategy_summary',
    'suppress_optuna_trial_logs',
    'print_config_header',
    'print_cross_strategy_metrics',
    'print_jij_tuning_summary',
    'print_jij_benchmark_summary',
    'print_jij_single_run',
    'save_jij_results',
    'load_jij_results',
    # NEW
    'compute_violation_rate',
    'compute_matrix_differences',
    'build_full_qubo_matrix',
    'select_best_from_pareto',
    'print_multiobjective_tuning_summary',
    'print_benchmark_summary',
    '_safe_format',
]