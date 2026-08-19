# Water Quality Monitoring Placement Optimization (Laguna Lake)

Optimize the placement of water quality monitoring stations in **Laguna Lake** using Mixed-Integer Quadratic Programming (MIQP) and quantum-inspired metaheuristics. This project compares exact solvers against Simulated Annealing (SA) and Simulated Quantum Annealing (SQA) as the number of candidate sites scales.

## Project Goal

Select an optimal subset of monitoring station locations that:
- Maximizes total utility (weighted spatial factors)
- Maintains network connectivity (no isolated stations)
- Respects a fixed budget (exactly K new stations)
- Accounts for hydrodynamic constraints (current alignment, proximity decay)

## Dataset

`data/LDB_centroids_clean.geojson` contains candidate site centroids with 8 spatially-weighted factors:

| Factor | Description |
|--------|-------------|
| `a1_river_proximity` | Proximity to river inflows |
| `a2_runoff_proximity` | Proximity to runoff zones |
| `b1_fishpen_density` | Aquaculture/fishpen density |
| `b2_hypoxic_proximity` | Proximity to hypoxic zones |
| `c1_bathymetric_depressions` | Bathymetric depression presence |
| `c2_outlet_proximity` | Proximity to lake outlet |
| `d1_boatramp_proximity` | Proximity to boat ramps |
| `d2_road_proximity` | Proximity to road access |

Utility score `U_i` is computed as a weighted average of valid factors, clipped to [0, 1].

## Methodology

### 1. MIQP Formulation

The optimization problem is formulated as a sparse Mixed-Integer Quadratic Program:

```
minimize:  Σ -U_i * x_i + Σ β * R_ij * x_i * x_j + Σ δ * W_ij * x_i * x_j
subject to:
  Σ x_i = K                                    (budget)
  x_i ≤ fixed_neighbors_i + Σ neighbor_ij * x_j (connectivity)
  x_i ∈ {0, 1}                                 (binary)
```

Where:
- `R_ij` = spatial proximity decay (`max(0, 1 - d/L_c)`)
- `W_ij` = current-aligned interaction weight (`exp(-d/L_w) * cos(θ)` if `cos(θ) > 0.7071`)
- `β` = proximity weight, `δ` = current weight
- `L_c` = connectivity range, `L_w` = current decay length
- `K` = number of new stations to deploy

### 2. Nested Resolution Scaling

Candidate sites are subsampled using **Farthest Point Sampling (FPS)** seeded with existing stations, generating nested tiers: N = 20, 50, 100, 200, 500, 1000. This enables scalability analysis without re-running the full pipeline.

### 3. Exact Solver: SCIP

Each tier is solved exactly using SCIP via `ommx-pyscipopt-adapter`. The sparse MIQP is encoded as a JijModeling problem with budget and connectivity constraints. Optimal solutions provide ground-truth energy values for benchmarking.

### 4. Metaheuristics: SA and SQA

**Simulated Annealing (SA)** uses OpenJij's `SASampler` with JijModeling QUBO compilation. Penalty weights for budget and connectivity constraints are tuned via Optuna TPE sampler with a custom convergence engine:
- **Early stopping A**: Parameter space variance collapse (σ_j ≤ tolerance)
- **Early stopping B**: Internal Top-K energy floor hit N times

**Simulated Quantum Annealing (SQA)** uses OpenJij's `SQASampler` with transverse-field schedules. The same tuning framework is applied with SQA-specific parameters (trotter fragments, schedule type).

### 5. Benchmarking: JijModeling vs Custom QUBO Builder

A dedicated benchmark compares QUBO construction time, SA sampling time, and number of QUBO terms between:
- **JijModeling**: Declarative model → `to_qubo()` compilation
- **Custom Sparse Builder**: Direct dictionary construction with penalty expansion

Correctness is verified by evaluating random binary vectors against a manual energy function.

## Tech Stack

| Component | Library |
|-----------|---------|
| QUBO/MIQP modeling | JijModeling ≥ 1.0 |
| SA/SQA sampling | OpenJij ≥ 0.12 |
| Exact solving | SCIP (PySCIPOpt + ommx-pyscipopt-adapter) |
| Hyperparameter tuning | Optuna ≥ 3.0 |
| Experiment tracking | Weights & Biases (optional) |
| Numerical computing | NumPy ≥ 2.0, SciPy ≥ 1.9 |
| Geospatial | Shapely ≥ 2.0 |
| Visualization | Matplotlib ≥ 3.5, Plotly (Cell 7), ipywidgets |
| Data persistence | Pickle, SQLite (Optuna studies) |

## Repository Structure

```
water-quality-qubo/
├── notebooks/
│   └── experiment.ipynb   # Self-contained workflow (11 cells)
├── data/
│   └── LDB_centroids_clean.geojson
├── results/               # Persistent outputs (plots, tuned params)
├── .venv/                 # Local virtual environment
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
- Python 3.10+ (local) or Google Colab
- Git (for cloning)

### Local Setup (Windows / Fedora KDE)

```bash
# Windows (PowerShell)
C:\Users\PC00\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Fedora KDE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

1. Upload `notebooks/experiment.ipynb` to Colab
2. Run Cell 0 (setup) — it auto-detects Colab, mounts Drive, and installs dependencies
3. Execute cells sequentially

### Execution Order

| Cell | Purpose |
|------|---------|
| 0 | Hybrid setup: env detection, package install, path resolution |
| 1 | Master control flags (recompute, retune, wandb) |
| 2 | Real data ingestion: GeoJSON → utility scores → master MIQP |
| 3 | Nested resolution scaling (FPS) |
| 4 | Sparse QUBO + MIQP instance generation per tier |
| 5 | SCIP exact solving (resumable) |
| 6 | SA hyperparameter tuning with Optuna |
| 7 | Interactive visualization of SA results |
| 8 | SQA hyperparameter tuning |
| 9 | Interactive visualization of SQA results |
| 10 | Final benchmark: SCIP vs Greedy vs SA vs SQA |

## Output Artifacts

All outputs are cached to disk with config-hash-aware filenames:

| Artifact | Format | Description |
|----------|--------|-------------|
| `master_real_{hash}.pkl` | Pickle | Master data (coords, utility, QUBO edges) |
| `scaling_results.pkl` | Pickle | Nested tier coordinates and utilities |
| `instance_data_N{N}_{hash}.pkl` | Pickle | Per-tier sparse QUBO instances |
| `scip_result_N{N}_{hash}.pkl` | Pickle | SCIP solutions per tier |
| `tuned_sa_N{N}_{hash}.json` | JSON | Best SA hyperparameters per tier |
| `sa_samples_N{N}_{hash}.pkl` | Pickle | All SA samples for visualization |
| `deployment_*.png` | PNG | Station deployment maps |
| `qubo_matrix_*.png` | PNG | QUBO coefficient matrices |
| `benchmark_results.pkl` | Pickle | JM vs Custom builder benchmark data |

## Key Design Decisions

- **Notebook-centric**: All logic lives in a single `experiment.ipynb` for easy customization and linear workflow.
- **Dual storage**: Fast local cache + persistent Google Drive backup (Colab).
- **Config-hash filenames**: Changing QUBO parameters automatically invalidates stale caches.
- **Resumable**: Every cell checks for existing artifacts before recomputing.
- **Single numpy version**: `numpy>=2.0` everywhere with graceful fallback to `<2.0` if needed.

## Troubleshooting

- **Import errors after install**: Re-run Cell 0 to refresh `site.main()` + `importlib.invalidate_caches()`.
- **SCIP out of memory**: Reduce `target_sizes` in Cell 3 or increase `D_max_buffer` in `MASTER_QUBO_CONFIG`.
- **SA infeasible trials**: Increase `LAMBDA_LOWER`/`LAMBDA_UPPER` range or increase `NUM_READS`/`NUM_SWEEPS`.
- **Colab runtime disconnect**: Results are auto-saved to Drive; re-mount and resume from cached pickle files.

## License

Academic project. Contact maintainers for reuse permissions.
