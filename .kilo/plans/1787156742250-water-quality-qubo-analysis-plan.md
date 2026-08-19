# Water Quality QUBO: Implementation Plan

## 1. Context & Documentation

**Project**: Optimize water quality monitoring station placement in Laguna Lake using MIQP/QUBO.
**Solvers**: SCIP (exact), SA (OpenJij + JijModeling), SQA (planned).
**Benchmark**: JijModeling QUBO builder vs custom sparse QUBO builder.
**Data**: `data/LDB_centroids_clean.geojson` (8 spatially-weighted factors, utility scoring).

### README.md
Generate covering: project goal, dataset, MIQP formulation, scaling (FPS), solver comparison, tuning (Optuna + custom convergence), tech stack, how to run (local venv + Colab), cell execution order, output artifacts.

## 2. Environment Compatibility

### Platforms
- **Local**: Fedora KDE (Python 3.10+) and Windows 11 (Python 3.10+ via venv)
- **Colab**: Python 3.12.13, default numpy >= 2.0

### Critical Issues & Fixes
1. **numpy split-brain**: `requirements.txt` says `>=2.0`, Colab cell installs `<2.0`.
   - **Fix**: Standardize on `numpy>=2.0` everywhere. Setup cell installs numpy 2.x first, then requirements. If any install fails, fall back to `numpy<2.0` with warning.
2. **Non-existent package**: `ommx-openjij-adapter` does not exist on PyPI.
   - **Fix**: Remove from Cell 0 PACKAGES list.
3. **Broken venv**: `.venv` has only pip, no packages.
   - **Fix**: Recreate venv with `--pip` flag using platform-appropriate Python.
4. **SSL cert errors**: pip subprocesses fail when building cmake for numpy.
   - **Fix**: Use `--only-binary=:all:` for numpy, upgrade pip/certifi first.

### Revised requirements.txt
```
numpy>=2.0
scipy>=1.9
pandas>=2.0,<3.0
matplotlib>=3.5
shapely>=2.0
tqdm>=4.60
ipywidgets>=8.0
jijmodeling>=2.7
openjij>=0.12
ommx-pyscipopt-adapter>=2.0
optuna>=3.0
wandb>=0.15
```

### Setup Cell (Cell 0) Logic
1. Detect `IN_COLAB`
2. Upgrade pip + certifi
3. Install `numpy>=2.0` with `--only-binary=:all:`
4. Install core packages
5. On failure: fall back to `numpy<2.0`, retry install
6. Call `site.main()` + `importlib.invalidate_caches()`

### Cross-Platform venv Commands
| Platform | Create | Activate |
|---|---|---|
| Windows (PowerShell) | `C:\Users\PC00\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv --pip` | `.\.venv\Scripts\Activate.ps1` |
| Fedora KDE | `python3 -m venv .venv --pip` | `source .venv/bin/activate` |
| Colab | N/A | N/A |

## 3. Refactoring: Notebook-Centric (No src/ extraction)

### Guiding Principle
All logic stays in `notebooks/experiment.ipynb`. The notebook is the sole deliverable.

### Target Structure
```
notebooks/experiment.ipynb
├── Cell 0:  HYBRID SETUP (env detection, installs, paths, aliases)
├── Cell 0.5: SHARED PLOTTING (fallback-enabled functions)
├── Cell 1:  MASTER CONTROL FLAGS + CONFIG_SA + CONFIG_SQA + CONFIG_BENCH
├── Cell 2:  REAL DATA INGESTOR + MASTER MIQP
├── Cell 3:  NESTED RESOLUTION SCALING (FPS)
├── Cell 4:  SPARSE QUBO + MIQP DATA
├── Cell 5:  MIQP SOLVE WITH SCIP
├── Cell 6:  SA TUNING (Optuna)
├── Cell 7:  INTERACTIVE VISUALIZATION (SA Results)
├── Cell 8:  SQA TUNING (Optuna)
├── Cell 9:  SQA VISUALIZATION
└── Cell 10: FINAL BENCHMARK (SCIP vs Greedy vs SA vs SQA)
```

### Current State
| Cell | Status |
|------|--------|
| Cell 0 | Needs refactor: centralize paths, fix numpy logic, remove `ommx-openjij-adapter` |
| Cell 0.5 | Needs insertion: shared plotting with fallbacks |
| Cell 1 | Needs expansion: add `CONFIG_SA`, `CONFIG_SQA`, `CONFIG_BENCH` |
| Cell 2 | Exists, needs PATH_FALLBACK removal |
| Cell 3 | Exists, needs PATH_FALLBACK removal |
| Cell 4 | Exists, needs PATH_FALLBACK removal |
| Cell 5 | Exists, needs PATH_FALLBACK removal |
| Cell 6 | Exists, needs PATH_FALLBACK removal |
| Cell 7 | Needs implementation |
| Cell 8 | Needs implementation |
| Cell 9 | Needs implementation |
| Cell 10 | Needs implementation |

### Refactoring Actions

| Phase | Action | Benefit |
|---|---|---|
| **A** | **Cell 0**: Centralize ALL path resolution, env detection, package checks. All subsequent cells reference `CACHE_DIR`, `RESULTS_DIR`, `DATA_DIR`, `LOCAL_CACHE`, `GDRIVE_BASE`, `OUTPUT_DIR`, `DRIVE_CACHE`, `GEOJSON_PATH` from Cell 0. | Eliminates PATH_FALLBACK duplication in Cells 1–10. |
| **B** | **Cell 0.5**: Define shared plotting functions (`plot_deployment`, `plot_matrix`, `plot_connectivity`) once. Every consuming cell has a fallback copy guarded by `try/except NameError`. | Reduces duplication while keeping cells self-contained. |
| **C** | **Cell 1**: Expand `CONFIG_SA`, `CONFIG_SQA`, `CONFIG_BENCH` dictionaries. All tunable parameters live here. | Single knob for simulation flags. |
| **D** | **Cell 6, 8, 10**: Break into sub-functions within the same cell (e.g., `def _run_tuning():`, `def _plot_results():`). Use Jupyter code folding. | Improves readability. |
| **E** | **Testing**: Add hidden Cell -1 with pytest-style assertions for energy computation, feasibility checks, QUBO builder correctness. | Catches regressions. |

### Key Decisions
- **Cell 0 centralization**: Yes. All paths defined in Cell 0. Subsequent cells use `try: CACHE_DIR except NameError: PATH_FALLBACK` as safety net.
- **Plotting**: Define in Cell 0.5, but every cell that uses plots has a fallback copy. This preserves self-containment.
- **Testing**: Hidden Cell -1 with assertions.

## 4. Feature Implementation

### Cell 7: Interactive Visualization
- Load `sa_samples_N{N}_{config_hash}.pkl` and `tuned_sa_N{N}_{config_hash}.json`
- Interactive dashboard with ipywidgets Dropdown for tier selection
- Tabs: deployment map (Plotly), convergence plot, QUBO matrix heatmap, parameter space explorer, SA vs SCIP overlay

### Cells 8–9: SQA
- Port SA logic to SQA using OpenJij `SQASampler`
- Same `build_augmented_model`, `compile_instance`, `get_penalty_weights` as Cell 6
- Optuna search space: `num_sweeps`, `num_reads`, `trotter`, schedule type
- Save as `sqa_samples_N{N}_{config_hash}.pkl` and `tuned_sqa_N{N}_{config_hash}.json`
- Convergence engine: same criteria (variance collapse, floor hits)

### Cell 10: Final Benchmark
- Solvers: SCIP, Greedy, SA (tuned), SQA (tuned)
- Metrics: runtime, master energy, feasibility rate, optimality gap, scalability
- Outputs: `benchmark_summary.csv`, publication-quality plots (matplotlib + seaborn)

## 5. Completed Work

1. **Environment fixed**: Recreated `.venv` with Python 3.10, installed all requirements (`numpy 2.2.6`, `jijmodeling 1.14.2`, `openjij 0.12.2`, `optuna 4.9.0`, `scipy 1.15.3`, `pandas 2.3.3`, `matplotlib 3.10.9`, `shapely 2.1.2`, `PySCIPOpt 6.2.1`, `ommx-pyscipopt-adapter 2.6.2`, `wandb 0.28.2`).
2. **README.md generated**: `README.md` created with project goal, methodology, tech stack, setup instructions, cell order, and output artifacts.
3. **Partial notebook refactor**: Created `notebooks/experiment_refactored.ipynb` with new Cell 0, Cell 0.5, and Cell 1 placeholders. Existing cells 2–8 were carried over. Cells 9–11 remain placeholders.
4. **Plan updated**: This file now reflects the notebook-centric structure and remaining work.

## 6. Remaining Implementation Tasks

1. **Cell 0 refactor**: Replace with centralized setup cell that handles env detection, numpy>=2.0 install with fallback, path resolution, and module purge.
2. **Cell 0.5 insertion**: Add shared plotting functions with fallbacks.
3. **Cell 1 expansion**: Add `CONFIG_SA`, `CONFIG_SQA`, `CONFIG_BENCH`.
4. **Cells 2–6 cleanup**: Remove PATH_FALLBACK blocks, rely on Cell 0 namespace.
5. **Cell 7 implementation**: Interactive visualization of SA results.
6. **Cells 8–9 implementation**: SQA tuning and visualization.
7. **Cell 10 implementation**: Final benchmark suite.
8. **Validation**: Syntax check, path audit, artifact verification.

## 7. Validation
- Notebook syntax check: `python -c "import json; nb=json.load(open('notebooks/experiment.ipynb', encoding='utf-8')); [compile(''.join(c['source']), f'c{i}', 'exec') for i,c in enumerate(nb['cells']) if c['cell_type']=='code']"`
- Path audit: grep for hardcoded `/content` or `MyDrive` in cell sources
- After each cell: verify pickle/JSON artifacts exist and are loadable
- Cross-check: SA/SQA best energy within 5–10% of SCIP optimal for N≤100
