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
   10. Deployment + QUBO matrix side-by-side (with text outside)
   11. Tuning vs Validation deployment comparison (2x2)
   12. Enhanced Optuna learning curve (best, median, rolling mean, scatter, NO worst)
   13. Gurobi baseline plot
   14. SQR vs Runtime scatter plot
   15. Cross‑strategy progress plots (with legends outside)
   (NEW) 16. Objective vs reads curve
   (NEW) 17. Violation rate vs reads curve
   (NEW) 18. Optuna comprehensive plots (8 types via optuna.visualization.matplotlib)
   (NEW) 19. Direct QUBO matrix heatmap with .npy saving
   (NEW) 20. Optuna plot wrapper with W&B logging

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
import seaborn as sns
from src.jij_solvers import compute_energy

# Optional: optuna for posterior plots
try:
    import optuna
    from optuna.visualization.matplotlib import (
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_pareto_front,
        plot_timeline,
        plot_intermediate_values,
        plot_hypervolume_history,
        plot_edf,
        plot_optimization_history,
        plot_rank,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Optional: W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Local import for building QUBO matrices
from src.model import build_qubo


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
    try:
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

        # Info text box below plots (already outside)
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

        # Figure-level legend (already outside)
        handles = [Patch(facecolor='red', edgecolor='black', label='New'),
                   Patch(facecolor='blue', edgecolor='black', label='Existing (M)'),
                   Patch(facecolor='lightgray', edgecolor='gray', label='Candidates')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to generate validation grid: {e}")


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
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate convergence profile: {e}")


# ============================================================================
# 3. FINAL DEPLOYMENT PLOT (Legacy, kept for compatibility)
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
    Legend placed outside to the right.
    """
    try:
        if best_solution is None:
            print("  ⚠️ No solution available for final deployment plot")
            return

        grid_x = np.linspace(0, DOMAIN_SIZE, 100)
        grid_y = np.linspace(0, DOMAIN_SIZE, 100)
        grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')

        selected_m = [i for i in range(len(best_solution)) if best_solution[i] == 1 and i in M_indices]
        all_selected = selected_m + selected_new

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        plt.subplots_adjust(right=0.72)  # make room for legend + text

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

        # Connectivity links
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

        # Legend placed outside to the right
        handles = [Patch(facecolor='red', edgecolor='black', label='New'),
                   Patch(facecolor='blue', edgecolor='black', label='Existing (M)'),
                   Patch(facecolor='lightgray', edgecolor='gray', label='Candidates')]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Info text box (already outside)
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate final deployment plot: {e}")


# ============================================================================
# 4. OPTUNA POSTERIOR PLOTS (with multi-objective handling)
# ============================================================================

def save_optuna_plots(
    study: 'optuna.Study',
    save_dir: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = False,
) -> None:
    """Unchanged: no legends."""
    if not OPTUNA_AVAILABLE:
        print("  ⚠️ Optuna not available; skipping plots.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(plot_func: Callable, filename: str, **kwargs) -> None:
        try:
            fig = plot_func(study, **kwargs)
            if hasattr(fig, 'savefig'):
                pass
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

    try:
        _save_plot(plot_param_importances, "optuna_param_importance.png")
    except Exception as e:
        print(f"  ⚠️ Could not generate parameter importance: {e}")

    try:
        if len(study.directions) > 1:
            def target_func(t):
                return t.values[0] if t.values is not None else None
            _save_plot(plot_parallel_coordinate, "optuna_parallel_coordinate.png", target=target_func)
        else:
            _save_plot(plot_parallel_coordinate, "optuna_parallel_coordinate.png")
    except Exception as e:
        print(f"  ⚠️ Failed to generate parallel coordinate plot: {e}")

    try:
        if len(study.directions) > 1:
            def target_func(t):
                return t.values[0] if t.values is not None else None
            _save_plot(plot_slice, "optuna_slice.png", target=target_func)
        else:
            _save_plot(plot_slice, "optuna_slice.png")
    except Exception as e:
        print(f"  ⚠️ Failed to generate slice plot: {e}")

    # Custom learning curve (no legend)
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate learning curve: {e}")


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
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate sensitivity plot: {e}")


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
# 7. PERFORMANCE SCATTER
# ============================================================================

def plot_performance_scatter(
    validation_results: Dict,
    save_path: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Create scatter plots of validation SQR vs ESR, and Feasibility vs MCR.
    """
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate performance scatter: {e}")


# ============================================================================
# 8. QUBO MATRIX HEATMAP
# ============================================================================

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
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate QUBO matrix heatmap: {e}")


# ============================================================================
# 9. EXPERIMENT COMPARISON TABLE (Bar chart)
# ============================================================================

def plot_experiment_comparison_table(
    results_dict: Dict[str, Dict],
    save_path: Union[str, Path],
    metrics: Optional[List[str]] = None,
    show_fig: bool = True,
) -> None:
    """
    Generate a bar chart comparing multiple experiments (e.g., schedules or objectives).
    """
    try:
        if metrics is None:
            metrics = ['best_sqr', 'spearman_rho', 'feas_rate']

        rows = []
        for name, res in results_dict.items():
            row = {'Experiment': name}
            for m in metrics:
                val = res.get(m, np.nan)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    row[m] = val
                else:
                    row[m] = 0.0
            rows.append(row)
        df = pd.DataFrame(rows)

        if df.empty:
            print("  ⚠️ No data to compare.")
            return

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
    except Exception as e:
        print(f"  ⚠️ Failed to generate experiment comparison table: {e}")


# ============================================================================
# 10. OPTUNA LEARNING CURVE (Legacy)
# ============================================================================

def plot_optuna_learning_curve(
    study: 'optuna.Study',
    save_path: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Generate and save Optuna learning curve (cumulative best).
    """
    try:
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
    except Exception as e:
        print(f"  ⚠️ Failed to generate learning curve: {e}")


# ============================================================================
# 11. ENHANCED OPTUNA LEARNING CURVE (NO worst, with scatter)
# ============================================================================

def plot_optuna_learning_curve_enhanced(
    study: 'optuna.Study',
    save_path: Union[str, Path],
    window: int = 10,
    show_fig: bool = True,
) -> None:
    """
    Enhanced learning curve: cumulative best, median, rolling mean, and scatter.
    No worst line (since it's often 1.0).
    """
    try:
        values = [t.value for t in study.trials if t.value is not None]
        if not values:
            print("  ⚠️ No values to plot enhanced learning curve.")
            return

        n_trials = len(values)
        cumulative_best = np.minimum.accumulate(values)
        cumulative_median = [np.median(values[:i+1]) for i in range(n_trials)]

        window = min(window, n_trials)
        rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter of all trials
        ax.scatter(range(n_trials), values, alpha=0.3, s=10, color='gray', label='All Trials')

        # Best and Median (no worst)
        ax.plot(cumulative_best, 'b-', linewidth=2, label='Best so far')
        ax.plot(cumulative_median, 'g--', linewidth=1.5, label='Median so far')

        if len(rolling_mean) > 0:
            ax.plot(range(window-1, n_trials), rolling_mean, 'm-.', linewidth=1.5, label=f'Rolling Mean (w={window})')

        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value (1 - Score)')
        study_name = getattr(study, 'study_name', 'Optuna Study')
        ax.set_title(f'Enhanced Learning Curve: {study_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to generate enhanced learning curve: {e}")


# ============================================================================
# 12. DEPLOYMENT + QUBO MATRIX SIDE-BY-SIDE
# ============================================================================

def plot_deployment_with_qubo(
    env: Dict,
    lam1: float,
    lam2: float,
    solution: np.ndarray,
    M_indices: List[int],
    selected_new: List[int],
    U: np.ndarray,
    coords: np.ndarray,
    DOMAIN_SIZE: float,
    current_vector: Tuple[float, float],
    CONNECTIVITY_RANGE: float,
    title: str,
    save_path: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = True,
) -> None:
    """
    Side-by-side: Deployment map (left) + QUBO matrix heatmap (right).
    Legend placed outside to the right.
    """
    try:
        if solution is None:
            print("  ⚠️ No solution provided for deployment+QUBO plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.subplots_adjust(bottom=0.15, top=0.85)

        # ---------- LEFT: Deployment Map ----------
        ax1 = axes[0]
        grid_x = np.linspace(0, DOMAIN_SIZE, 100)
        grid_y = np.linspace(0, DOMAIN_SIZE, 100)
        grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')

        ax1.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
        ax1.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.5,
                    edgecolor='gray', linewidth=0.2)

        selected_m = [i for i in range(len(solution)) if solution[i] == 1 and i in M_indices]
        if selected_m:
            ax1.scatter(coords[selected_m, 0], coords[selected_m, 1],
                        c='blue', s=120, marker='s', edgecolor='black', label='Existing (M)')
        if selected_new:
            ax1.scatter(coords[selected_new, 0], coords[selected_new, 1],
                        c='red', s=150, marker='o', edgecolor='black', label='New')

        # Connectivity links
        all_selected = selected_m + selected_new
        for i, idx_i in enumerate(all_selected):
            for j, idx_j in enumerate(all_selected):
                if i < j:
                    dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                    if dist <= CONNECTIVITY_RANGE:
                        ax1.plot([coords[idx_i, 0], coords[idx_j, 0]],
                                 [coords[idx_i, 1], coords[idx_j, 1]],
                                 color='gray', alpha=0.4, linewidth=1.5, linestyle='--')

        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_aspect('equal')
        ax1.set_xlim(-2, DOMAIN_SIZE + 2)
        ax1.set_ylim(-2, DOMAIN_SIZE + 2)
        ax1.set_title(f'{title}\nDeployment Map', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # ---------- RIGHT: QUBO Matrix ----------
        ax2 = axes[1]
        pairwise_norm = env['pairwise_norm']
        K_new = env['config']['K_new']
        qubo = build_qubo(pairwise_norm, K_new, lambda1=lam1, lambda2=lam2,
                          use_jijmodeling=False, verbose=False)
        h = qubo['h']
        J = qubo['J']
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

        spatial_order = np.argsort(coords[:, 0])
        Q_reordered = Q[spatial_order, :][:, spatial_order]

        cax = ax2.imshow(Q_reordered, cmap='coolwarm', aspect='auto')
        ax2.set_xlabel('Site index (spatial order)')
        ax2.set_ylabel('Site index (spatial order)')
        ax2.set_title(f'{title}\nQUBO Matrix (λ₁={lam1:.4f}, λ₂={lam2:.4f})')
        plt.colorbar(cax, ax=ax2, label='Energy coefficient')

        # Info text box outside
        info_text = (
            f"Seed: {env['config'].get('seed', 'N/A')} | "
            f"N: {N_total} | K_new: {K_new} | "
            f"Selected: {len(selected_new)} new, {len(selected_m)} existing"
        )
        fig.text(0.5, 0.03, info_text, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.06, 0.85, 0.95])
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to generate deployment+QUBO plot: {e}")


# ============================================================================
# 13. TUNING VS VALIDATION DEPLOYMENT COMPARISON (2x2)
# ============================================================================

def plot_tuning_vs_validation_deployment(
    env: Dict,
    tuning_trial: Dict,
    val_trial: Dict,
    coords: np.ndarray,
    U: np.ndarray,
    M_indices: List[int],
    DOMAIN_SIZE: float,
    current_vector: Tuple[float, float],
    CONNECTIVITY_RANGE: float,
    experiment_name: str,
    save_path: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = True,
) -> None:
    """
    2x2 grid comparing tuning deployment+QUBO (top row) vs validation deployment+QUBO (bottom row).
    Legends placed outside.
    """
    try:
        # Extract data from tuning trial
        tune_lam1 = tuning_trial.get('lam1')
        tune_lam2 = tuning_trial.get('lam2')
        tune_solution = tuning_trial.get('solution')
        tune_sqr = tuning_trial.get('best_sqr', 'N/A')
        tune_trial_num = tuning_trial.get('trial_number', 'N/A')

        # Extract from validation trial
        val_lam1 = val_trial.get('lam1')
        val_lam2 = val_trial.get('lam2')
        val_solution = val_trial.get('solution')
        val_sqr = val_trial.get('best_sqr', 'N/A')
        val_trial_num = val_trial.get('trial_number', 'N/A')

        # Helper: build QUBO matrix
        def build_q(lam1, lam2):
            pairwise_norm = env['pairwise_norm']
            K_new = env['config']['K_new']
            qubo = build_qubo(pairwise_norm, K_new, lambda1=lam1, lambda2=lam2,
                              use_jijmodeling=False, verbose=False)
            h = qubo['h']
            J = qubo['J']
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
            spatial_order = np.argsort(coords[:, 0])
            return Q[spatial_order, :][:, spatial_order]

        def draw_deployment(ax, solution, title_sub):
            grid_x = np.linspace(0, DOMAIN_SIZE, 100)
            grid_y = np.linspace(0, DOMAIN_SIZE, 100)
            grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')
            ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
            ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.5,
                       edgecolor='gray', linewidth=0.2)
            selected_m = [i for i in range(len(solution)) if solution[i] == 1 and i in M_indices]
            selected_new = [i for i in range(len(solution)) if solution[i] == 1 and i not in M_indices]
            if selected_m:
                ax.scatter(coords[selected_m, 0], coords[selected_m, 1],
                           c='blue', s=120, marker='s', edgecolor='black', label='Existing (M)')
            if selected_new:
                ax.scatter(coords[selected_new, 0], coords[selected_new, 1],
                           c='red', s=150, marker='o', edgecolor='black', label='New')
            all_selected = selected_m + selected_new
            for i, idx_i in enumerate(all_selected):
                for j, idx_j in enumerate(all_selected):
                    if i < j:
                        dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                        if dist <= CONNECTIVITY_RANGE:
                            ax.plot([coords[idx_i, 0], coords[idx_j, 0]],
                                    [coords[idx_i, 1], coords[idx_j, 1]],
                                    color='gray', alpha=0.4, linewidth=1.5, linestyle='--')
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_aspect('equal')
            ax.set_xlim(-2, DOMAIN_SIZE + 2)
            ax.set_ylim(-2, DOMAIN_SIZE + 2)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title(title_sub, fontsize=11)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Top row: Tuning
        draw_deployment(axes[0, 0], tune_solution, f'Tuning Trial #{tune_trial_num} | SQR={tune_sqr:.4f}')
        if tune_lam1 is not None and tune_lam2 is not None:
            axes[0, 1].imshow(build_q(tune_lam1, tune_lam2), cmap='coolwarm', aspect='auto')
            axes[0, 1].set_title(f'Tuning QUBO (λ₁={tune_lam1:.4f}, λ₂={tune_lam2:.4f})')
            axes[0, 1].set_xlabel('Site index (spatial order)')
            axes[0, 1].set_ylabel('Site index (spatial order)')
        else:
            axes[0, 1].text(0.5, 0.5, 'QUBO not available', ha='center', va='center', transform=axes[0, 1].transAxes)

        # Bottom row: Validation
        draw_deployment(axes[1, 0], val_solution, f'Validation Trial #{val_trial_num} | SQR={val_sqr:.4f}')
        if val_lam1 is not None and val_lam2 is not None:
            axes[1, 1].imshow(build_q(val_lam1, val_lam2), cmap='coolwarm', aspect='auto')
            axes[1, 1].set_title(f'Validation QUBO (λ₁={val_lam1:.4f}, λ₂={val_lam2:.4f})')
            axes[1, 1].set_xlabel('Site index (spatial order)')
            axes[1, 1].set_ylabel('Site index (spatial order)')
        else:
            axes[1, 1].text(0.5, 0.5, 'QUBO not available', ha='center', va='center', transform=axes[1, 1].transAxes)

        fig.suptitle(f'{experiment_name}: Tuning vs Validation Deployment', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # make room for legends
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to generate tuning vs validation deployment plot: {e}")


# ============================================================================
# 14. GUROBI BASELINE PLOT
# ============================================================================

def plot_gurobi_baseline(
    env: Dict,
    save_path: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = True,
) -> None:
    """
    Plot the Gurobi (or greedy fallback) baseline solution as a deployment map.
    Legend placed outside.
    """
    try:
        solution = env.get('gurobi_solution')
        coords = env['coords']
        U = env['U']
        M_indices = env['M_indices']
        DOMAIN_SIZE = 50.0
        CONNECTIVITY_RANGE = env.get('connectivity_range', 8.0)

        if solution is None:
            print("  ⚠️ No Gurobi solution available.")
            return

        selected_m = [i for i in range(len(solution)) if solution[i] == 1 and i in M_indices]
        selected_new = [i for i in range(len(solution)) if solution[i] == 1 and i not in M_indices]
        all_selected = selected_m + selected_new

        fig, ax = plt.subplots(figsize=(10, 8))
        grid_x = np.linspace(0, DOMAIN_SIZE, 100)
        grid_y = np.linspace(0, DOMAIN_SIZE, 100)
        grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')

        ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.5,
                   edgecolor='gray', linewidth=0.2)

        if selected_m:
            ax.scatter(coords[selected_m, 0], coords[selected_m, 1],
                       c='blue', s=120, marker='s', edgecolor='black', label='Existing (M)')
        if selected_new:
            ax.scatter(coords[selected_new, 0], coords[selected_new, 1],
                       c='red', s=150, marker='o', edgecolor='black', label='New')

        # Connectivity links
        for i, idx_i in enumerate(all_selected):
            for j, idx_j in enumerate(all_selected):
                if i < j:
                    dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                    if dist <= CONNECTIVITY_RANGE:
                        ax.plot([coords[idx_i, 0], coords[idx_j, 0]],
                                [coords[idx_i, 1], coords[idx_j, 1]],
                                color='gray', alpha=0.4, linewidth=1.5, linestyle='--')

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_aspect('equal')
        ax.set_xlim(-2, DOMAIN_SIZE + 2)
        ax.set_ylim(-2, DOMAIN_SIZE + 2)
        ax.set_title('Gurobi Exact Optimum (SQR=1.0000)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Info text outside
        info_text = (
            f"Seed: {env['config'].get('seed', 'N/A')} | "
            f"N: {len(coords)} | K_new: {env['config']['K_new']} | "
            f"Gurobi MIQP: {env['gurobi_miqp']:.6f} (SQR=1.0000) | "
            f"Selected: {len(selected_new)} new, {len(selected_m)} existing"
        )
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        plt.tight_layout(rect=[0, 0.06, 0.85, 1])
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to generate Gurobi baseline plot: {e}")


# ============================================================================
# 15. SQR vs RUNTIME SCATTER PLOT
# ============================================================================

def plot_sqr_vs_runtime(
    results_df: pd.DataFrame,
    save_path: Union[str, Path],
    dpi: int = 150,
    show_fig: bool = True,
) -> None:
    """
    Scatter plot of SQR vs Runtime, colour-coded by objective.
    Legend placed outside.
    """
    if results_df.empty:
        print("  ⚠️ No data for SQR vs Runtime plot.")
        return

    agg = results_df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
        'best_sqr': ['mean', 'std', 'count'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                   'sqr_mean', 'sqr_std', 'n_seeds',
                   'time_mean', 'time_std']

    agg = agg[agg['sqr_mean'].notna() & agg['time_mean'].notna() & (agg['time_mean'] > 0)]
    if agg.empty:
        print("  ⚠️ No valid data for SQR vs Runtime plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for obj in agg['objective'].unique():
        sub = agg[agg['objective'] == obj]
        ax.scatter(sub['time_mean'], sub['sqr_mean'], label=obj, s=80, alpha=0.7)

    # Pareto frontier (approx)
    sorted_agg = agg.sort_values('time_mean')
    pareto = []
    best_sqr = -np.inf
    for _, row in sorted_agg.iterrows():
        if row['sqr_mean'] > best_sqr:
            pareto.append(row)
            best_sqr = row['sqr_mean']

    if pareto:
        pareto_df = pd.DataFrame(pareto)
        ax.plot(pareto_df['time_mean'], pareto_df['sqr_mean'], 'k--', linewidth=2, label='Pareto Frontier')

    ax.set_xlabel('Mean Runtime (seconds)')
    ax.set_ylabel('Mean SQR')
    ax.set_title('SQR vs Runtime')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# ============================================================================
# 16. CROSS-STRATEGY PROGRESS (updated with legend outside)
# ============================================================================

def plot_cross_strategy_progress(
    results_df: pd.DataFrame,
    save_dir: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Generate updated cross-strategy progress plots from the accumulated DataFrame.
    All legends placed outside.
    """
    if results_df.empty:
        print("  ⚠️ No data to plot cross-strategy progress.")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    agg = results_df.groupby(['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N']).agg({
        'best_sqr': ['mean', 'std', 'count'],
        'feas_rate': ['mean'],
        'spearman_rho': ['mean'],
        'total_samples': ['mean'],
        'time_seconds': ['mean', 'std']
    }).reset_index()
    agg.columns = ['objective', 'tuning_trials', 'tuning_reads', 'val_top_k', 'val_reads', 'N',
                   'best_sqr_mean', 'best_sqr_std', 'n_seeds',
                   'feas_mean', 'rho_mean', 'samples_mean',
                   'time_mean', 'time_std']

    # Drop rows with NaN SQR
    agg = agg[agg['best_sqr_mean'].notna()]

    if agg.empty:
        print("  ⚠️ No valid aggregated data.")
        return

    # ---- Heatmap (for N=100) ----
    stage1_df = agg[agg['N'] == 100]
    if not stage1_df.empty:
        for obj in stage1_df['objective'].unique():
            sub = stage1_df[stage1_df['objective'] == obj]
            sub['row_label'] = sub['tuning_trials'].astype(str) + "T," + sub['tuning_reads'].astype(str) + "R"
            sub['col_label'] = "K" + sub['val_top_k'].astype(str) + ",V" + sub['val_reads'].astype(str)
            pivot = sub.pivot(index='row_label', columns='col_label', values='best_sqr_mean')
            if pivot.empty:
                continue
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis',
                        cbar_kws={'label': 'Mean SQR'})
            plt.title(f'Mean SQR Heatmap (Objective: {obj}, N=100)')
            plt.xlabel('Validation Strategy')
            plt.ylabel('Tuning Strategy')
            plt.tight_layout()
            plt.savefig(save_dir / f"heatmap_{obj}_N100.png", dpi=150)
            if show_fig:
                plt.show()
            plt.close()

    # ---- Pareto Front: SQR vs Samples ----
    plt.figure(figsize=(10, 6))
    for obj in agg['objective'].unique():
        sub = agg[agg['objective'] == obj]
        plt.scatter(sub['samples_mean'], sub['best_sqr_mean'],
                    label=obj, s=60, alpha=0.7)
    plt.xlabel('Total Samples (Tuning + Validation)')
    plt.ylabel('Mean Validation SQR')
    plt.title('Pareto Front: SQR vs Samples')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_dir / "pareto_front_samples.png", dpi=150)
    if show_fig:
        plt.show()
    plt.close()

    # ---- SQR vs Feasibility ----
    plt.figure(figsize=(10, 6))
    for obj in agg['objective'].unique():
        sub = agg[agg['objective'] == obj]
        plt.scatter(sub['feas_mean'], sub['best_sqr_mean'],
                    label=obj, s=60, alpha=0.7)
    plt.xlabel('Mean Feasibility Rate')
    plt.ylabel('Mean Validation SQR')
    plt.title('SQR vs Feasibility')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_dir / "sqr_vs_feas.png", dpi=150)
    if show_fig:
        plt.show()
    plt.close()

    # ---- Spearman ρ Heatmap ----
    if not stage1_df.empty:
        for obj in stage1_df['objective'].unique():
            sub = stage1_df[stage1_df['objective'] == obj]
            sub['row_label'] = sub['tuning_trials'].astype(str) + "T," + sub['tuning_reads'].astype(str) + "R"
            sub['col_label'] = "K" + sub['val_top_k'].astype(str) + ",V" + sub['val_reads'].astype(str)
            pivot = sub.pivot(index='row_label', columns='col_label', values='rho_mean')
            if pivot.empty or pivot.isnull().all().all():
                continue
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm',
                        cbar_kws={'label': 'Spearman ρ'})
            plt.title(f'Spearman ρ Heatmap (Objective: {obj}, N=100)')
            plt.xlabel('Validation Strategy')
            plt.ylabel('Tuning Strategy')
            plt.tight_layout()
            plt.savefig(save_dir / f"spearman_heatmap_{obj}_N100.png", dpi=150)
            if show_fig:
                plt.show()
            plt.close()

    # ---- SQR vs Runtime (new) ----
    plot_sqr_vs_runtime(results_df, save_dir / "sqr_vs_runtime.png", show_fig=show_fig)


# ============================================================================
# 17. REGENERATE PLOTS (calls all of the above)
# ============================================================================

def regenerate_plots(
    results_df: pd.DataFrame,
    save_dir: Union[str, Path],
    show_fig: bool = True,
) -> None:
    """
    Regenerate all cross-strategy progress plots from a saved DataFrame.
    Called on startup to restore plots after a crash or resume.
    """
    if results_df.empty:
        print("  ⚠️ No data to regenerate plots.")
        return
    print("  🔄 Regenerating cross-strategy progress plots from saved data...")
    plot_cross_strategy_progress(results_df, save_dir, show_fig=show_fig)
    print("  ✅ Plots regenerated.")

# =============================================================================
# ADDITIONS TO src/plotting.py (JijModeling Pipeline)
# =============================================================================

# -----------------------------------------------------------------------------
# Helper: extract QUBO matrix from instance
# -----------------------------------------------------------------------------
def _extract_qubo_matrix(instance, N: int) -> np.ndarray:
    """
    Extract full N×N QUBO matrix from an OMMX instance using to_qubo.
    Assumes instance is already compiled.
    """
    qubo_dict, _ = instance.to_qubo(uniform_penalty_weight=1.0)  # weight doesn't matter for matrix extraction
    Q = np.zeros((N, N))
    for (i, j), coeff in qubo_dict.items():
        if i == j:
            Q[i, i] += coeff
        else:
            Q[i, j] += coeff
            Q[j, i] += coeff
    return Q


# -----------------------------------------------------------------------------
# Plot deployment map
# -----------------------------------------------------------------------------
def plot_jij_deployment(
    instance_data: Dict,
    solution: np.ndarray,
    penalty_weights: Dict[int, float],
    save_path: Optional[Union[str, Path]] = None,
    show_fig: bool = True,
    dpi: int = 150,
    domain_size: int = 50,
) -> None:
    """
    Plot the deployment map from a JijModeling solution.
    Uses coordinates and utility from instance_data.
    """
    N = instance_data["N"]
    coords = instance_data["coords"]
    U = instance_data["U"]
    M_indices = instance_data.get("M_indices", [])
    selected = np.where(solution == 1)[0]
    selected_new = [i for i in selected if i not in M_indices]
    selected_m = [i for i in selected if i in M_indices]

    DOMAIN_SIZE = domain_size  # default
    CONNECTIVITY_RANGE = 8.0  # default; could be stored in instance_data
    # If we have connectivity range in instance_data, use it
    if "D_max" in instance_data:
        CONNECTIVITY_RANGE = instance_data["D_max"]

    # Create grid for utility contour
    grid_x = np.linspace(0, DOMAIN_SIZE, 100)
    grid_y = np.linspace(0, DOMAIN_SIZE, 100)
    grid_z = griddata(coords, U, (grid_x[None, :], grid_y[:, None]), method='cubic')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background contour
    cf = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', alpha=0.3)
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Utility $U_i$', fontsize=12)

    # All candidates
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=30, alpha=0.5,
               edgecolor='gray', linewidth=0.2, label='Candidates')

    # Existing stations (if any)
    if selected_m:
        ax.scatter(coords[selected_m, 0], coords[selected_m, 1],
                   c='blue', s=120, marker='s', edgecolor='black', label='Existing (M)')

    # New selected stations
    if selected_new:
        ax.scatter(coords[selected_new, 0], coords[selected_new, 1],
                   c='red', s=120, edgecolor='black', linewidth=1, zorder=3,
                   label=f'New Stations ({len(selected_new)})')

    # Connectivity links (for selected stations within D_max)
    all_selected = selected_m + selected_new
    for i, idx_i in enumerate(all_selected):
        for j, idx_j in enumerate(all_selected):
            if i < j:
                dist = np.linalg.norm(coords[idx_i] - coords[idx_j])
                if dist <= CONNECTIVITY_RANGE:
                    ax.plot([coords[idx_i, 0], coords[idx_j, 0]],
                            [coords[idx_i, 1], coords[idx_j, 1]],
                            color='gray', alpha=0.4, linewidth=1, linestyle='--')

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title('Deployment Solution (SCIP)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-2, DOMAIN_SIZE + 2)
    ax.set_ylim(-2, DOMAIN_SIZE + 2)

    # Legend outside
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='New'),
               plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Existing (M)'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Candidates')]
    ax.legend(handles=handles, bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=10)

    # Format selected indices as comma-separated strings
    selected_new_str = ', '.join(map(str, selected_new)) if selected_new else 'None'
    selected_m_str = ', '.join(map(str, selected_m)) if selected_m else 'None'
    
    # Info text outside with indices included
    info_text = (
        f"N: {N} | K: {instance_data['K']}\n"
        f"Selected: {len(selected_new)} new, {len(selected_m)} existing\n"
        f"New indices: [{selected_new_str}]\n"
        f"Existing indices: [{selected_m_str}]\n"
        f"Energy: {compute_energy(solution, instance_data['a'], instance_data['Q']):.6f}"
    )
    
    plt.figtext(0.55, 0.08, info_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0, 0.68, 1])
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plot QUBO matrix heatmap (direct matrix input)
# -----------------------------------------------------------------------------
def plot_qubo_matrix_direct(
    Q_mat: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "QUBO Matrix",
    show_fig: bool = True,
    dpi: int = 150,
    cmap: str = 'coolwarm',
) -> None:
    """
    Plot QUBO matrix heatmap directly from a numpy array and save .npy.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(Q_mat, ax=ax, cmap=cmap, square=False, cbar_kws={'label': 'Energy coefficient'})
        ax.set_title(title)
        ax.set_xlabel('Site index')
        ax.set_ylabel('Site index')
        plt.tight_layout()
        if save_path:
            # Save .npy alongside the figure
            npy_path = Path(save_path).with_suffix('.npy')
            np.save(npy_path, Q_mat)
            # Save figure
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️ Failed to plot QUBO matrix: {e}")


# -----------------------------------------------------------------------------
# QUBO matrix heatmap with spatial ordering
# -----------------------------------------------------------------------------
def plot_qubo_matrix_heatmap_with_order(
    Q_mat: np.ndarray,
    coords: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "QUBO Matrix (spatial order)",
    show_fig: bool = True,
    dpi: int = 150,
) -> None:
    """
    Plot QUBO matrix heatmap reordered by spatial x-coordinate.
    """
    try:
        spatial_order = np.argsort(coords[:, 0])
        Q_reordered = Q_mat[spatial_order, :][:, spatial_order]
        plot_qubo_matrix_direct(Q_reordered, save_path, title, show_fig, dpi)
    except Exception as e:
        print(f"  ⚠️ Failed to plot ordered QUBO matrix: {e}")


# -----------------------------------------------------------------------------
# Objective vs reads curve
# -----------------------------------------------------------------------------
def plot_objective_vs_reads(
    ax: plt.Axes,
    read_indices: np.ndarray,
    objective_values: np.ndarray,
    label: str,
    color: str = 'blue',
    marker: str = 'o',
    linestyle: str = '-',
) -> None:
    """
    Plot objective value (e.g., MIQP energy) vs read index.
    """
    ax.plot(read_indices, objective_values, linestyle=linestyle, marker=marker,
            color=color, linewidth=1.5, markersize=4, label=label)
    ax.set_xlabel('Read Index')
    ax.set_ylabel('Objective Value (MIQP Energy)')
    ax.grid(True, alpha=0.3)
    ax.legend()


# -----------------------------------------------------------------------------
# Violation rate vs reads curve
# -----------------------------------------------------------------------------
def plot_violation_vs_reads(
    ax: plt.Axes,
    read_indices: np.ndarray,
    violation_rates: np.ndarray,
    label: str,
    color: str = 'red',
    marker: str = 's',
    linestyle: str = '-',
) -> None:
    """
    Plot violation rate vs read index (step-like).
    """
    # Use step plot for discrete values
    ax.step(read_indices, violation_rates, where='post', linestyle=linestyle,
            color=color, linewidth=1.5, label=label)
    ax.set_xlabel('Read Index')
    ax.set_ylabel('Violation Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()


# -----------------------------------------------------------------------------
# Optuna comprehensive plots (8 types + parallel_coordinate)
# -----------------------------------------------------------------------------
def save_and_log_optuna_plots(
    study: 'optuna.Study',
    save_dir: Union[str, Path],
    wandb_run=None,
    show_fig: bool = True,
    dpi: int = 150,
) -> Dict[str, plt.Figure]:
    """
    Generate all 8 Optuna visualisation plots (plus parallel_coordinate) and
    optionally log them to W&B.

    Plots generated:
        - timeline
        - pareto_front
        - intermediate_values
        - hypervolume_history
        - edf
        - optimization_history
        - param_importances
        - rank
        - parallel_coordinate (extra)

    Args:
        study: Optuna study object.
        save_dir: Directory to save images.
        wandb_run: Optional W&B run object to log images.
        show_fig: Whether to display figures inline.
        dpi: Resolution for saved images.

    Returns:
        Dict mapping plot name to matplotlib Figure object.
    """
    if not OPTUNA_AVAILABLE:
        print("  ⚠️ Optuna not available; skipping plot generation.")
        return {}

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define plot functions and their filenames
    plot_specs = [
        ('timeline', plot_timeline, 'optuna_timeline.png'),
        ('pareto_front', plot_pareto_front, 'optuna_pareto_front.png'),
        ('intermediate_values', plot_intermediate_values, 'optuna_intermediate_values.png'),
        ('hypervolume_history', plot_hypervolume_history, 'optuna_hypervolume_history.png'),
        ('edf', plot_edf, 'optuna_edf.png'),
        ('optimization_history', plot_optimization_history, 'optuna_optimization_history.png'),
        ('param_importances', plot_param_importances, 'optuna_param_importances.png'),
        ('rank', plot_rank, 'optuna_rank.png'),
        ('parallel_coordinate', plot_parallel_coordinate, 'optuna_parallel_coordinate.png'),
    ]

    figures = {}
    for name, plot_func, filename in plot_specs:
        try:
            # For multi-objective studies, some plots require target function
            # For rank, we can use default; for pareto_front, it works automatically
            fig = plot_func(study)
            # Some functions return a Figure object; others return an array of axes
            if isinstance(fig, plt.Figure):
                fig = fig
            elif isinstance(fig, np.ndarray) and len(fig) > 0:
                # Sometimes plot_* returns an array of axes; we need to get the figure
                fig = fig[0].figure
            else:
                # Try to get the current figure
                fig = plt.gcf()
            # Save
            save_path = save_dir / filename
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            figures[name] = fig
            if show_fig:
                plt.show()
            # Log to W&B
            if wandb_run is not None and WANDB_AVAILABLE:
                wandb_run.log({name: wandb.Image(str(save_path))})
            plt.close(fig)
        except Exception as e:
            print(f"  ⚠️ Failed to generate {name} plot: {e}")

    return figures


# ============================================================================
# Scaling plots
# ============================================================================
def plot_scaling_benchmark(
    df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    show_fig: bool = True,
    dpi: int = 150,
) -> None:
    """
    Generate scaling plots: SQR vs N, Runtime vs N, Feasibility vs N.
    """
    if df.empty:
        print("⚠️ No data for scaling plots.")
        return

    save_dir = Path(save_dir) if save_dir else Path("scaling_plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by solver, N
    agg = df.groupby(["solver", "N"]).agg({
        "sqr": ["mean", "std"],
        "runtime": ["mean", "std"],
        "feasible": "mean"
    }).reset_index()
    agg.columns = ["solver", "N", "sqr_mean", "sqr_std", "runtime_mean", "runtime_std", "feasibility"]

    # 1. SQR vs N
    plt.figure(figsize=(10, 6))
    for solver in agg["solver"].unique():
        sub = agg[agg["solver"] == solver]
        plt.errorbar(sub["N"], sub["sqr_mean"], yerr=sub["sqr_std"], marker='o', label=solver, capsize=5)
    plt.xlabel('Number of candidate sites (N)', fontsize=12)
    plt.ylabel('Mean SQR', fontsize=12)
    plt.title('Scaling: SQR vs N')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "sqr_vs_N.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()

    # 2. Runtime vs N
    plt.figure(figsize=(10, 6))
    for solver in agg["solver"].unique():
        sub = agg[agg["solver"] == solver]
        plt.errorbar(sub["N"], sub["runtime_mean"], yerr=sub["runtime_std"], marker='o', label=solver, capsize=5)
    plt.xlabel('Number of candidate sites (N)', fontsize=12)
    plt.ylabel('Mean Runtime (s)', fontsize=12)
    plt.title('Scaling: Runtime vs N')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "runtime_vs_N.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()

    # 3. Feasibility vs N
    plt.figure(figsize=(10, 6))
    for solver in agg["solver"].unique():
        sub = agg[agg["solver"] == solver]
        plt.plot(sub["N"], sub["feasibility"], marker='o', label=solver)
    plt.xlabel('Number of candidate sites (N)', fontsize=12)
    plt.ylabel('Mean Feasibility Rate', fontsize=12)
    plt.title('Scaling: Feasibility vs N')
    plt.ylim(-0.05, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "feasibility_vs_N.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Benchmark summary plots (boxplots, scatter, etc.)
# -----------------------------------------------------------------------------
def plot_jij_benchmark_summary(
    df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    show_fig: bool = True,
    dpi: int = 150,
) -> None:
    """
    Generate summary plots: SQR boxplots, Runtime vs SQR scatter, Feasibility bar.
    """
    if df.empty:
        print("⚠️ No data for summary plots.")
        return

    save_dir = Path(save_dir) if save_dir else Path("benchmark_plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. SQR boxplots by solver and N
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="solver", y="sqr", hue="N")
    plt.title("SQR by Solver and N")
    plt.ylabel("SQR")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(title="N")
    plt.tight_layout()
    plt.savefig(save_dir / "sqr_boxplot.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()

    # 2. Runtime vs SQR scatter (Pareto frontier)
    plt.figure(figsize=(10, 6))
    for solver in df["solver"].unique():
        sub = df[df["solver"] == solver]
        plt.scatter(sub["runtime"], sub["sqr"], label=solver, alpha=0.6, s=60)
    plt.xlabel("Runtime (s)")
    plt.ylabel("SQR")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "runtime_vs_sqr.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()

    # 3. Feasibility rate bar chart
    feasible_rate = df.groupby(["solver", "N"])["feasible"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feasible_rate, x="solver", y="feasible", hue="N")
    plt.title("Feasibility Rate")
    plt.ylabel("Proportion Feasible")
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title="N")
    plt.tight_layout()
    plt.savefig(save_dir / "feasibility_bar.png", dpi=dpi)
    if show_fig:
        plt.show()
    plt.close()


# ============================================================================
# Helper: compute energy (for info text)
# ============================================================================
def _compute_energy_from_solution(x: np.ndarray, a: np.ndarray, Q: np.ndarray) -> float:
    return np.dot(a, x) + 0.5 * np.dot(x, np.dot(Q, x))


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    'plot_validation_grid',
    'plot_convergence_profile',
    'plot_final_deployment',
    'save_optuna_plots',
    'save_sensitivity_plot',
    'display_saved_plots',
    'plot_performance_scatter',
    'plot_qubo_matrix_heatmap',
    'plot_experiment_comparison_table',
    'plot_optuna_learning_curve',
    'plot_optuna_learning_curve_enhanced',
    'plot_deployment_with_qubo',
    'plot_tuning_vs_validation_deployment',
    'plot_gurobi_baseline',
    'plot_sqr_vs_runtime',
    'plot_cross_strategy_progress',
    'regenerate_plots',
    'plot_jij_deployment',
    'plot_qubo_matrix_direct',
    'plot_qubo_matrix_heatmap_with_order',
    'plot_objective_vs_reads',
    'plot_violation_vs_reads',
    'save_and_log_optuna_plots',
    'plot_scaling_benchmark',
    'plot_jij_benchmark_summary',
]