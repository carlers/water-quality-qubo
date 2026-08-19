#!/usr/bin/env python3
"""
Implement Cell 7: Interactive Visualization of SA Results.
Writes the cell source to tools/cell_07_viz.py
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "tools" / "cell_07_viz.py"

CELL_7_SOURCE = '''#@title CELL 7: INTERACTIVE VISUALIZATION (SA Results)
"""
===============================================================================
INTERACTIVE VISUALIZATION OF SA TUNING RESULTS
- Loads sa_samples_N{N}_{config_hash}.pkl and tuned_sa_N{N}_{config_hash}.json
- Interactive dashboard with ipywidgets Dropdown for tier selection
- Tabs: Deployment map, Convergence plot, QUBO matrix, Parameter space
===============================================================================
"""

import os
import sys
import pickle
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Check for optional plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

warnings.filterwarnings('ignore')

# === PATH FALLBACK (safety net) ===
try:
    CACHE_DIR
except NameError:
    import importlib.util
    try:
        IN_COLAB = importlib.util.find_spec("google.colab") is not None
    except Exception:
        IN_COLAB = False
    if IN_COLAB:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        CACHE_DIR = Path("/content/wqm_data")
        RESULTS_DIR = Path("/content/drive/MyDrive/wqm_data")
        DATA_DIR = RESULTS_DIR
    else:
        REPO_PATH = Path.cwd().resolve()
        for _ in range(6):
            if (REPO_PATH / "notebooks").exists() or (REPO_PATH / "src").exists() or (REPO_PATH / ".git").exists():
                break
            REPO_PATH = REPO_PATH.parent
        DATA_DIR = REPO_PATH / "data"
        CACHE_DIR = DATA_DIR / "cache"
        RESULTS_DIR = REPO_PATH / "results"
    LOCAL_CACHE = CACHE_DIR
    GDRIVE_BASE = RESULTS_DIR
    OUTPUT_DIR = RESULTS_DIR
    DRIVE_CACHE = RESULTS_DIR
    GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"

# === LOAD CONFIG HASH ===
try:
    config_hash
except NameError:
    import hashlib
    default_config = {
        "L_c": 7500.0, "L_w": 1000.0, "Beta": 1.0, "Delta": 1.0,
        "Current_vector": (1.0, 0.0), "K": 5, "D_max_buffer": 1.15,
    }
    config_str = json.dumps(default_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

RUN_DIR = RESULTS_DIR / f"run_{config_hash}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# === DISCOVER TIERS ===
pattern = re.compile(r"instance_data_N(\\d+)_([a-f0-9]{8})\\.pkl")
instance_files = []
for p in GDRIVE_BASE.glob("instance_data_N*_*.pkl"):
    m = pattern.match(p.name)
    if m and m.group(2) == config_hash:
        instance_files.append((int(m.group(1)), p))
if not instance_files:
    for p in LOCAL_CACHE.glob("instance_data_N*_*.pkl"):
        m = pattern.match(p.name)
        if m and m.group(2) == config_hash:
            instance_files.append((int(m.group(1)), p))
instance_files.sort()
available_tiers = [n for n, _ in instance_files]

if not available_tiers:
    raise FileNotFoundError("No instance files found. Run Cells 2-4 first.")

# === LOAD HELPERS ===
try:
    coords_master
    U_master
    M_indices_master
except NameError:
    master_path = LOCAL_CACHE / f"master_real_{config_hash}.pkl"
    if not master_path.exists():
        master_path = GDRIVE_BASE / f"master_real_{config_hash}.pkl"
    if master_path.exists():
        with open(master_path, "rb") as f:
            master_data = pickle.load(f)
        coords_master = master_data["coords"]
        U_master = master_data["U"]
        M_indices_master = master_data["M_indices"]
    else:
        coords_master = None
        U_master = None
        M_indices_master = []

# Shared plotting functions (fallback if Cell 0.5 wasn't run)
try:
    plot_deployment
except NameError:
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    import numpy as _np

    def plot_deployment(instance_data, result_or_solution, save_path=None, show=True, dpi=150):
        coords_full = _np.asarray(instance_data["original_coords"])
        D_MAX = instance_data["D_max"]
        fixed_indices = list(instance_data.get("fixed_indices", []))
        free_indices = list(instance_data.get("original_indices", []))
        x_sol = _np.asarray(result_or_solution, dtype=int) if result_or_solution is not None else _np.zeros(len(free_indices), dtype=int)
        selected_free_orig = [free_indices[i] for i in _np.where(x_sol == 1)[0] if i < len(free_indices)]
        selected_new = [i for i in selected_free_orig if i not in fixed_indices]
        selected_m = fixed_indices
        all_selected = selected_new + selected_m
        if coords_master is not None and len(coords_master) > 0:
            xmin, ymin = coords_master.min(axis=0)
            xmax, ymax = coords_master.max(axis=0)
        else:
            xmin, ymin = coords_full.min(axis=0)
            xmax, ymax = coords_full.max(axis=0)
        pad_x = max(0.05 * (xmax - xmin), 1.0)
        pad_y = max(0.05 * (ymax - ymin), 1.0)
        fig = plt.figure(figsize=(12, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.3], wspace=0.15)
        ax = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
        ax_info.axis('off')
        if coords_master is not None and U_master is not None:
            sc = ax.scatter(coords_master[:, 0], coords_master[:, 1], c=U_master, cmap='viridis', s=22, alpha=0.55, edgecolor='none', zorder=0)
            plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04).set_label('Utility $U_i$', fontsize=11)
        if free_indices:
            candidate_coords = coords_full[free_indices]
            ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1], c='white', s=40, alpha=0.9, edgecolor='#1e293b', linewidth=0.8, label='Candidates', zorder=2)
        if selected_m:
            ax.scatter(coords_full[selected_m, 0], coords_full[selected_m, 1], c='blue', s=110, marker='s', edgecolor='black', linewidth=1.2, label=f'Existing ({len(selected_m)})', zorder=3)
        if selected_new:
            ax.scatter(coords_full[selected_new, 0], coords_full[selected_new, 1], c='red', s=110, marker='o', edgecolor='black', linewidth=1.2, label=f'New ({len(selected_new)})', zorder=4)
        ax.set_xlabel('Easting (m)', fontsize=11)
        ax.set_ylabel('Northing (m)', fontsize=11)
        ax.set_title('Deployment (SA Tuned)', fontsize=13, fontweight='bold', pad=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

# === DATA LOADERS ===
def load_instance(N):
    inst_path = LOCAL_CACHE / f"instance_data_N{N}_{config_hash}.pkl"
    if not inst_path.exists():
        inst_path = GDRIVE_BASE / f"instance_data_N{N}_{config_hash}.pkl"
    if not inst_path.exists():
        raise FileNotFoundError(f"Instance not found: instance_data_N{N}_{config_hash}.pkl")
    with open(inst_path, "rb") as f:
        return pickle.load(f)

def load_tuned_params(N):
    json_path = GDRIVE_BASE / f"tuned_sa_N{N}_{config_hash}.json"
    if not json_path.exists():
        json_path = LOCAL_CACHE / f"tuned_sa_N{N}_{config_hash}.json"
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)

def load_sa_samples(N):
    samples_path = LOCAL_CACHE / f"sa_samples_N{N}_{config_hash}.pkl"
    if not samples_path.exists():
        samples_path = GDRIVE_BASE / f"sa_samples_N{N}_{config_hash}.pkl"
    if not samples_path.exists():
        return None
    with open(samples_path, "rb") as f:
        return pickle.load(f)

# === MAIN INTERACTIVE DASHBOARD ===
def build_dashboard(N):
    print(f"\\n🔬 Building interactive dashboard for N={N}...")

    instance_data = load_instance(N)
    tuned = load_tuned_params(N)
    samples = load_sa_samples(N)

    if tuned is None:
        print(f"⚠️ No tuned parameters found for N={N}. Run Cell 6 first.")
        return

    if samples is None:
        print(f"⚠️ No SA samples found for N={N}.")
        return

    # Build static plots with matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Deployment
    ax = axes[0, 0]
    x_sol = np.zeros(instance_data["N"], dtype=int)
    if tuned.get("selected_free_indices"):
        for idx in eval(tuned["selected_free_indices"]):
            if idx < instance_data["N"]:
                x_sol[idx] = 1
    plot_deployment(instance_data, x_sol, save_path=RUN_DIR / f"deployment_SA_N{N}_interactive.png", show=False, dpi=150)
    img = plt.imread(RUN_DIR / f"deployment_SA_N{N}_interactive.png")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Deployment (N={N})", fontweight="bold")

    # Plot 2: Convergence
    ax = axes[0, 1]
    trial_nums = sorted(samples.keys())
    best_energies = []
    for t in trial_nums:
        trial_samples = samples[t]
        feas = [s for s in trial_samples if s.get("feasible", False)]
        if feas:
            best_energies.append(min(s["energy"] for s in feas))
        else:
            best_energies.append(None)
    ax.plot(trial_nums, best_energies, marker='o', linestyle='-', color='steelblue')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best Feasible Energy")
    ax.set_title("Convergence Plot", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 3: QUBO Matrix
    ax = axes[1, 0]
    N_vars = instance_data["N"]
    Q = np.zeros((N_vars, N_vars))
    for i, j, val in instance_data.get("Q_edges", []):
        Q[i, j] = val
        Q[j, i] = val
    np.fill_diagonal(Q, instance_data["a"])
    im = ax.imshow(Q, cmap='RdBu_r', aspect='auto',
                   norm=plt.Normalize(vmin=-np.max(np.abs(Q)), vmax=np.max(np.abs(Q))))
    ax.set_title(f"QUBO Matrix (N={N_vars})", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 4: Parameter Space
    ax = axes[1, 1]
    param_data = []
    for t in trial_nums:
        trial_samples = samples[t]
        if trial_samples:
            s = trial_samples[0]
            param_data.append({
                "trial": t,
                "lambda_budget": tuned.get("lambda_budget", 0),
                "lambda_conn": tuned.get("lambda_conn", 0),
                "energy": s.get("energy", np.nan),
            })
    if param_data:
        pdf = pd.DataFrame(param_data)
        sc = ax.scatter(pdf["lambda_budget"], pdf["lambda_conn"], c=pdf["energy"], cmap="viridis", s=50, alpha=0.8)
        ax.set_xlabel("lambda_budget")
        ax.set_ylabel("lambda_conn")
        ax.set_title("Parameter Space", fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.colorbar(sc, ax=ax, label="Energy")

    plt.suptitle(f"SA Results Dashboard – N={N}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RUN_DIR / f"sa_dashboard_N{N}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"✅ Dashboard saved to {RUN_DIR / f'sa_dashboard_N{N}.png'}")

# === WIDGETS ===
if WIDGETS_AVAILABLE:
    tier_dropdown = widgets.Dropdown(
        options=available_tiers,
        value=available_tiers[0] if available_tiers else None,
        description="Tier N:",
        style={"description_width": "initial"}
    )

    def on_tier_change(change):
        if change["new"] is not None:
            build_dashboard(change["new"])

    tier_dropdown.observe(on_tier_change, names="value")
    display(tier_dropdown)

    # Auto-build for default tier
    if available_tiers:
        build_dashboard(available_tiers[0])
else:
    print("⚠️ ipywidgets not available. Building static dashboard for all tiers...")
    for N in available_tiers:
        try:
            build_dashboard(N)
        except Exception as e:
            print(f"⚠️ Failed for N={N}: {e}")
            continue

print("\\n✅ Cell 7 complete.")
'''


OUT_PATH = REPO / "tools" / "cell_07_viz.py"

def main():
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(CELL_7_SOURCE)
    print("Cell 7 source written to", OUT_PATH)


if __name__ == "__main__":
    main()
