"""
src/plotting.py

All visualisation functions for the water quality monitoring QUBO project.

This module provides:
    1. Validation grid (2x2 comparison: Gurobi + Top 3)
    2. Convergence profile (sharpening cumulative best)
    3. Final deployment map
    4. Optuna posterior plots (importances, parallel, slice, learning curve)
    5. Sensitivity analysis plot
    6. Display saved plots (inline in Colab)
    7. Performance scatter (SQR vs ESR/MCR, feasibility vs MCR)
    8. QUBO matrix heatmap (for champion hyperparameters)
    9. Experiment comparison table (bar chart comparing multiple experiments)

All functions accept a save_path and a show_fig flag (default True in Colab).
"""

import json
import pickle
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import scipy.stats as stats

# Optional: optuna for posterior plots
try:
    import optuna
    from optuna.visualization.matplotlib import (
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ============================================================================
# 1. VALIDATION GRID (2x2)
# ============================================================================

def plot_validation_grid(
    coords: np.ndarray,
    U: np.ndarray,
    gurobi_solution: np.ndarray,
    top3_solutions: List[np.ndarray],
    top3_trials: List[Dict],
    M_indices: List[int],
    DOMAIN_SIZE: float,
    save_path: Union[str, Path],
    dpi: int = 100,
    show_fig: bool = True,
) -> None:
    """
    Generate 2x2 validation grid comparing Gurobi vs Top 3.
    Saves to disk and optionally displays inline.
    """
    N_total = len(coords)
    grid_x = np.linspace(0, DOMAIN_SIZE, 100)
    grid_y = np.linspace(0, DOMAIN_SIZE, 100)
    grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(bottom=0.15)

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
        ax.text(0.98, 0.98, f'SQR = {sqr}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Info text box below plots
    info_lines = []
    info_lines.append("Gurobi (Exact) — SQR = 1.0000")
    for i, t in enumerate(top3_trials[:3]):
        trial_num = t.get('trial', '?')
        sqr = t.get('best_sqr', None)
        feas = t.get('feas_rate', None)
        lam1 = t.get('lam1', None)
        lam2 = t.get('lam2', None)
        beta_min_mult = t.get('beta_min_mult', None)
        beta_max_mult = t.get('beta_max_mult', None)
        num_sweeps = t.get('num_sweeps', None)

        sqr_str = f"{sqr:.4f}" if isinstance(sqr, float) else "N/A"
        feas_str = f"{feas*100:.1f}%" if isinstance(feas, float) else "N/A"
        lam1_str = f"{lam1:.4f}" if isinstance(lam1, float) else "N/A"
        lam2_str = f"{lam2:.4f}" if isinstance(lam2, float) else "N/A"
        sweeps_str = str(num_sweeps) if num_sweeps is not None else "N/A"
        beta_display = f"β_mult=[{beta_min_mult:.4f}, {beta_max_mult:.4f}]" if beta_min_mult is not None else "β=[N/A]"
        line = (f"Trial {trial_num:4d}  SQR={sqr_str}  Feas={feas_str}  "
                f"λ₁={lam1_str}  λ₂={lam2_str}  {beta_display}  Sweeps={sweeps_str}")
        info_lines.append(line)

    info_text = "\n".join(info_lines)
    fig.text(0.5, 0.04, info_text, ha='center', va='bottom', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
             linespacing=1.3)

    handles = [Patch(facecolor='red', edgecolor='black', label='New'),
               Patch(facecolor='blue', edgecolor='black', label='Existing (M)'),
               Patch(facecolor='lightgray', edgecolor='gray', label='Candidates')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 2. CONVERGENCE PROFILE (Sharpening)
# ============================================================================

def plot_convergence_profile(
    best_sharpen: Dict,
    convergence_data: Dict,
    GUROBI_MIQP: float,
    save_path: Union[str, Path],
    dpi: int = 100,
    show_fig: bool = True,
) -> None:
    """
    Generate convergence profile plot for the best sharpening run.
    """
    conv_key = f"trial_{best_sharpen['trial']}_run_{best_sharpen['run']}"
    conv_data = convergence_data.get(conv_key)
    if conv_data is None:
        print(f"  ⚠️ No convergence data for key: {conv_key}")
        return

    cumulative = conv_data['cumulative_best']
    cumulative_sqr = [x / GUROBI_MIQP for x in cumulative if x != float('inf')]
    cumulative_miqp = [x for x in cumulative if x != float('inf')]
    read_indices = range(len(cumulative_sqr))

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
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

    seed = conv_data.get('seed', '?')
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
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 3. FINAL DEPLOYMENT PLOT
# ============================================================================

def plot_final_deployment(
    coords: np.ndarray,
    U: np.ndarray,
    best_solution: np.ndarray,
    M_indices: List[int],
    selected_new: List[int],
    DOMAIN_SIZE: float,
    current_vector: Tuple[float, float],
    CONNECTIVITY_RANGE: float,
    champion: Dict,
    best_sharpen: Dict,
    sharpening_means: Dict,
    save_path: Union[str, Path],
    dpi: int = 100,
    show_fig: bool = True,
) -> None:
    """
    Generate final deployment plot with utility landscape.
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

    # Info text box
    beta_min_mult = best_sharpen.get('beta_min_mult', None)
    beta_max_mult = best_sharpen.get('beta_max_mult', None)
    if beta_min_mult is None:
        beta_display = f"β=[{best_sharpen.get('beta_min', 'N/A')}, {best_sharpen.get('beta_max', 'N/A')}] (legacy)"
    else:
        beta_display = f"β_mult=[{beta_min_mult:.4f}, {beta_max_mult:.4f}] (v4.21)"

    info_text = (
        f"Best Deployment Solution (v4.21)\n"
        f"{'─' * 30}\n"
        f"Trial: #{best_sharpen['trial']} (Run {best_sharpen['run']})\n"
        f"λ₁ = {best_sharpen['lam1']:.4f}\n"
        f"λ₂ = {best_sharpen['lam2']:.4f}\n"
        f"{beta_display}\n"
        f"Sweeps: {best_sharpen.get('num_sweeps', 'N/A')}\n"
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
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 4. OPTUNA POSTERIOR PLOTS (Save only by default, but can show)
# ============================================================================

def save_optuna_plots(
    study: 'optuna.Study',
    save_dir: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = False,
) -> None:
    """
    Generate and save Optuna posterior plots (importances, parallel, slice, learning curve).
    Optionally display inline.
    """
    if not OPTUNA_AVAILABLE:
        print("  ⚠️ Optuna not available; skipping plots.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(plot_func: Callable, filename: str, **kwargs) -> None:
        try:
            fig = plot_func(study, **kwargs)
            # fig may be a Figure or an Axes or array of Axes
            if hasattr(fig, 'savefig'):
                pass  # it's a Figure
            elif hasattr(fig, 'figure'):
                fig = fig.figure
            elif isinstance(fig, np.ndarray) and len(fig) > 0 and hasattr(fig[0], 'figure'):
                fig = fig[0].figure
            else:
                raise ValueError(f"Could not extract figure from {type(fig)}")
            fig.savefig(save_dir / filename, dpi=dpi, bbox_inches='tight')
            if show_fig:
                plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"  ⚠️ Failed to generate {filename}: {e}")

    _save_plot(plot_param_importances, "optuna_param_importance.png")
    _save_plot(plot_parallel_coordinate, "optuna_parallel_coordinate.png")
    _save_plot(plot_slice, "optuna_slice.png")

    # Learning curve (custom)
    best_values = [t.value for t in study.trials if t.value is not None]
    if best_values:
        cumulative_best = np.minimum.accumulate(best_values)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(cumulative_best)+1), cumulative_best, 'b-', linewidth=2)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Best Objective (1 - score)')
        ax.set_title(f'Optuna Learning Curve')
        ax.grid(True, alpha=0.3)
        fig.savefig(save_dir / "optuna_learning_curve.png", dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)


# ============================================================================
# 5. SENSITIVITY ANALYSIS PLOT
# ============================================================================

def save_sensitivity_plot(
    results_df: pd.DataFrame,
    baseline_sqr: float,
    save_dir: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = True,
) -> None:
    """
    Generate and save hyperparameter sensitivity analysis plot.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if results_df is None or len(results_df) == 0:
        print("  ⚠️ No sensitivity results to plot.")
        return

    params = ['lam1', 'lam2', 'num_sweeps', 'beta_min_mult', 'beta_max_mult']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for i, param in enumerate(params):
        ax = axes_flat[i]
        subset = results_df[results_df['param'] == param]
        if len(subset) > 0:
            ax.plot(subset['fraction'], subset['best_sqr'], 'o-', linewidth=2, color='blue', markersize=8)
            for _, row in subset.iterrows():
                ax.annotate(f"{row['feas_rate']*100:.0f}%",
                            (row['fraction'], row['best_sqr']),
                            textcoords="offset points", xytext=(0, 8),
                            ha='center', fontsize=8, color='gray')
            ax.axhline(y=baseline_sqr, color='red', linestyle='--', alpha=0.7, label='Champion')
            ax.set_xlabel('Fraction of champion value')
            ax.set_ylabel('Best SQR')
            ax.set_title(param)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.75, 1.25)
            if i == 0:
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(param)

    if len(params) < 6:
        axes_flat[len(params)].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / "sensitivity_analysis.png", dpi=dpi, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 6. DISPLAY SAVED PLOTS (inline)
# ============================================================================

def display_saved_plots(
    plot_dir: Union[str, Path],
    file_list: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Display saved plots from a directory inline (for Colab).
    If file_list is None, uses a default list of known plot files.
    """
    plot_dir = Path(plot_dir)
    print("\n" + "-" * 70)
    print(f"📊 DISPLAYING SAVED PLOTS (from {plot_dir})")
    print("-" * 70)

    if file_list is None:
        file_list = [
            ("optuna_param_importance.png", "Optuna: Parameter Importance"),
            ("optuna_parallel_coordinate.png", "Optuna: Parallel Coordinate"),
            ("optuna_slice.png", "Optuna: Slice Plots"),
            ("optuna_learning_curve.png", "Optuna: Learning Curve"),
            ("phase2_validation_2x2.png", "2×2 Validation Grid (Gurobi + Top 3)"),
            ("phase3_convergence.png", "Convergence Profile (Best Sharpening Run)"),
            ("phase3_deployment_solution.png", "Final Deployment Solution"),
            ("sensitivity_analysis.png", "Hyperparameter Sensitivity Analysis"),
        ]

    for filename, title in file_list:
        filepath = plot_dir / filename
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
# 7. NEW PLOTS FOR REFACTORED EXPERIMENTS
# ============================================================================

def plot_performance_scatter(
    validation_results: Dict,
    save_path: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Create scatter plots of validation SQR vs ESR, and Feasibility vs MCR.
    """
    trials = validation_results.get('trials', [])
    if not trials:
        print("  ⚠️ No trials to plot in performance scatter.")
        return

    # Extract data
    sqrs = []
    esrs = []
    feases = []
    mcrs = []
    for t in trials:
        sqr = t.get('best_sqr')
        esr = t.get('ESR')
        feas = t.get('feas_rate')
        mcr = t.get('MCR')
        if sqr is not None and np.isfinite(sqr) and sqr > 0:
            sqrs.append(sqr)
            esrs.append(esr if esr is not None and np.isfinite(esr) else np.nan)
            feases.append(feas if feas is not None and np.isfinite(feas) else np.nan)
            mcrs.append(mcr if mcr is not None and np.isfinite(mcr) else np.nan)

    if len(sqrs) < 3:
        print("  ⚠️ Insufficient valid trials for scatter plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SQR vs ESR
    ax = axes[0]
    valid = [(s, e) for s, e in zip(sqrs, esrs) if not np.isnan(e)]
    if valid:
        s_vals, e_vals = zip(*valid)
        ax.scatter(e_vals, s_vals, alpha=0.7)
        ax.set_xlabel('ESR (Energy Scale Ratio)')
        ax.set_ylabel('Validation SQR')
        ax.set_title('SQR vs ESR')
        ax.grid(True, alpha=0.3)
        # Add vertical line at ESR=1
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ESR=1')
        ax.legend()

    # Feasibility vs MCR
    ax = axes[1]
    valid = [(f, m) for f, m in zip(feases, mcrs) if not np.isnan(m)]
    if valid:
        f_vals, m_vals = zip(*valid)
        ax.scatter(m_vals, f_vals, alpha=0.7)
        ax.set_xlabel('MCR (Max Coefficient Ratio)')
        ax.set_ylabel('Feasibility Rate')
        ax.set_title('Feasibility vs MCR')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='MCR=1')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


def plot_qubo_matrix_heatmap(
    env: Dict,
    lam1: float,
    lam2: float,
    K_new: int,
    save_path: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Generate a heatmap of the full QUBO matrix for given hyperparameters.
    Assumes env contains 'pairwise_norm' and 'coords' with spatial ordering.
    """
    from src.model import build_qubo
    from src.utils import safe_load_pickle  # avoid circular import? Actually we can use env['pairwise_norm']

    pairwise_norm = env['pairwise_norm']
    coords = env['coords']

    # Build QUBO
    qubo = build_qubo(pairwise_norm, K_new, lambda1=lam1, lambda2=lam2,
                      use_jijmodeling=False, verbose=False)
    h = qubo['h']
    J = qubo['J']

    # Build full matrix (symmetric)
    N_total = len(coords)
    Q = np.zeros((N_total, N_total))
    for i, val in h.items():
        Q[i, i] += val
    for (i, j), val in J.items():
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += val / 2.0
            Q[j, i] += val / 2.0

    # Reorder by spatial x-coordinate
    spatial_order = np.argsort(coords[:, 0])
    Q_reordered = Q[spatial_order, :][:, spatial_order]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(Q_reordered, cmap='coolwarm', aspect='auto')
    ax.set_title(f'QUBO Matrix Heatmap (λ₁={lam1:.4f}, λ₂={lam2:.4f}, K={K_new})')
    ax.set_xlabel('Site index (spatial order)')
    ax.set_ylabel('Site index (spatial order)')
    plt.colorbar(cax, label='Energy coefficient')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


def plot_experiment_comparison_table(
    results_dict: Dict[str, Dict],
    save_path: Union[str, Path],
    metrics: Optional[List[str]] = None,
    show_fig: bool = True,
) -> None:
    """
    Generate a bar chart comparing multiple experiments (e.g., schedules or objectives).
    results_dict: {exp_name: validation_results}
    metrics: list of keys to compare, e.g., ['best_sqr', 'spearman_rho', 'feas_rate'].
    """
    if metrics is None:
        metrics = ['best_sqr', 'spearman_rho', 'feas_rate']

    # Build DataFrame
    rows = []
    for name, res in results_dict.items():
        row = {'Experiment': name}
        for m in metrics:
            val = res.get(m, np.nan)
            if isinstance(val, (int, float)) and np.isfinite(val):
                row[m] = val
            else:
                row[m] = 0.0  # or np.nan, but for plotting we can use 0
        rows.append(row)
    df = pd.DataFrame(rows)

    if df.empty:
        print("  ⚠️ No data to compare.")
        return

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(df)*0.8), 6))
    x = np.arange(len(df))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric], width, label=metric, color=colors[i % len(colors)])

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Experiment Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df['Experiment'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, row in df.iterrows():
        for j, metric in enumerate(metrics):
            val = row[metric]
            if val != 0.0:
                ax.text(i + j*width, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 7. OPTUNA PLOTS WRAPPERS (for individual use)
# ============================================================================

def plot_optuna_learning_curve(
    study: 'optuna.Study',
    save_path: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Generate and save Optuna learning curve (cumulative best).
    """
    best_values = [t.value for t in study.trials if t.value is not None]
    if not best_values:
        print("  ⚠️ No values to plot learning curve.")
        return
    cumulative_best = np.minimum.accumulate(best_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_best)+1), cumulative_best, 'b-', linewidth=2)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Best Objective')
    ax.set_title('Optuna Learning Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Existing
    'plot_validation_grid',
    'plot_convergence_profile',
    'plot_final_deployment',
    'save_optuna_plots',
    'save_sensitivity_plot',
    'display_saved_plots',
    # New
    'plot_performance_scatter',
    'plot_qubo_matrix_heatmap',
    'plot_experiment_comparison_table',
    'plot_optuna_learning_curve',
]