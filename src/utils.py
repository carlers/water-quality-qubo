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
   11. NEW: Single-run summary printer
   12. NEW: Cross-strategy summary printer (dynamic ranked table)
   13. NEW: Optuna trial log suppressor

All plotting functions have been moved to src/plotting.py.
"""

import json
import pickle
import time
import gc
import contextlib
import logging
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


def print_tuning_summary(study, experiment_name):
    """
    Print a detailed summary of the best tuning trial from an Optuna study.
    Displays hyperparameters, user attributes, and a preview of the selected stations.
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
    print("\n  Hyperparameters:")
    for key, val in best.params.items():
        if isinstance(val, float):
            print(f"    {key:20s}: {_safe_format(val, '.6f')}")
        else:
            print(f"    {key:20s}: {val}")

    # User attributes (excluding solutions to keep it clean)
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

    # Print selected stations preview
    if solution is not None:
        selected = [i for i, v in enumerate(solution) if v == 1]
        if len(selected) <= 15:
            idx_str = str(selected)
        else:
            idx_str = str(selected[:10]) + f" ... (total {len(selected)})"
        print(f"    {'best_solution':20s}: {idx_str}")
    else:
        print(f"    {'best_solution':20s}: Not stored")
    print("=" * 80)


def print_validation_summary(val_results, experiment_name):
    """
    Print a detailed summary of the validation champion.
    Expects val_results from validate_study().
    """
    if val_results is None or not val_results.get('trials'):
        print(f"⚠️ No validation results for experiment '{experiment_name}'.")
        return

    # The first trial in 'trials' is the best (sorted by best_sqr descending)
    best_trial = val_results['trials'][0]

    print("\n" + "=" * 80)
    print(f"📊 VALIDATION SUMMARY: {experiment_name}")
    print("=" * 80)
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
    # Selected stations
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
# 10. NEW: SINGLE-RUN SUMMARY (for strategy comparison)
# ============================================================================

def print_single_run_summary(config: Dict, result: Dict, elapsed: float):
    """
    Print a concise yet rich summary for a single completed run.
    
    Args:
        config: dict with keys like 'seed', 'objective', 'tuning_trials', 'tuning_reads',
                'val_top_k', 'val_reads', 'N'
        result: dict from run_one_config() containing 'best_sqr', 'feas_rate', 'spearman_rho', etc.
        elapsed: runtime in seconds
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
    print(f"  Best SQR:        {_safe_format(result.get('best_sqr', np.nan), '.6f')}")
    print(f"  Feasibility:     {_safe_format(result.get('feas_rate', np.nan), '.4f')}")
    if 'spearman_rho' in result and result['spearman_rho'] is not None:
        print(f"  Spearman ρ:      {_safe_format(result['spearman_rho'], '.4f')}")
    else:
        print("  Spearman ρ:      N/A")
    print(f"  Total Samples:   {result.get('total_samples', 'N/A')}")
    print(f"  Runtime:         {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print("=" * 80)


# ============================================================================
# 11. NEW: CROSS-STRATEGY SUMMARY (dynamic ranked table)
# ============================================================================

def print_cross_strategy_summary(results_df: pd.DataFrame, title: str = "Cross-Strategy Summary"):
    """
    Print a dynamic ranked table of all completed configurations so far.
    
    Args:
        results_df: DataFrame with columns: seed, objective, tuning_trials, tuning_reads,
                    val_top_k, val_reads, N, best_sqr, feas_rate, spearman_rho, total_samples, time_seconds
        title: optional title for the summary
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

    # Group by config and aggregate across seeds (if multiple seeds present)
    agg = results_df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
        'best_sqr': ['mean', 'std', 'count'],
        'feas_rate': ['mean', 'std'],
        'spearman_rho': ['mean', 'std'],
        'total_samples': ['mean'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                   'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                   'feas_mean', 'feas_std',
                   'rho_mean', 'rho_std',
                   'samples_mean',
                   'time_mean', 'time_std']

    # Sort by best_sqr_mean descending
    agg_sorted = agg.sort_values('best_sqr_mean', ascending=False)

    # Display top rows (up to 20)
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
    formatted = agg_sorted[display_cols].copy()
    for col in ['best_sqr_mean', 'best_sqr_std', 'feas_mean', 'rho_mean', 'samples_mean', 'time_mean']:
        formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
    # Format time as minutes
    formatted['time_mean'] = formatted['time_mean'].apply(
        lambda x: f"{x/60:.2f} min" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
    )

    # Print top rows (show all if <= 20, else top 10 and note)
    if len(formatted) <= 20:
        print(formatted.to_string(index=False))
    else:
        print(formatted.head(10).to_string(index=False))
        print(f"\n... and {len(formatted)-10} more configurations.")
        # Also show the best and worst for context
        print("\n🏆 Best overall:")
        print(formatted.iloc[0].to_string())
        print("\n📉 Worst overall:")
        print(formatted.iloc[-1].to_string())

    print("=" * 80)


# ============================================================================
# 12. NEW: SUPPRESS OPTUNA TRIAL LOGS
# ============================================================================

def suppress_optuna_trial_logs():
    """
    Suppress Optuna's per-trial logging output (e.g., "Trial 0 finished with value: ...")
    but keep the progress bar visible.
    """
    # Set Optuna's logger to WARNING level to suppress INFO messages
    optuna_logger = logging.getLogger('optuna')
    optuna_logger.setLevel(logging.WARNING)
    # Also suppress the root logger if it's propagating
    # We can also adjust the logging level for the 'optuna' namespace
    # The progress bar is handled separately by show_progress_bar=True in study.optimize()


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
    'print_single_run_summary',        # NEW
    'print_cross_strategy_summary',    # NEW
    'suppress_optuna_trial_logs',      # NEW
]