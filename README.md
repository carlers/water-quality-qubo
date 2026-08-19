# water-quality-qubo

QUBO/MIQP experiments for water-quality monitoring buoy placement in Laguna de Bay
(EPSG:32651), tuned with SA (OpenJij), solved exactly with SCIP, and optimized
with Optuna.

## Local run (VS Code + venv kernel)

The notebook `notebooks/experiment.ipynb` is designed to run **locally** on the
`.venv` Python kernel (it no longer mounts Google Drive or writes to `/content`).

1. Create / activate the venv and install dependencies:

   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Open `notebooks/experiment.ipynb` in VS Code and select the `.venv` kernel
   (it is already stored in the notebook metadata).

3. Run the cells **in order**:

   - **Cell 1 – LOCAL ENVIRONMENT CHECK**: verifies the venv and installs any
     missing packages.
   - **Cell 1.5 – MASTER CONTROL FLAGS**: recompute flags (set once).
   - **Cell [RUN] – LOCAL PATHS & MODULE RELOAD**: defines the local paths used
     by every other cell:
     - `DATA_DIR` → `data/` (raw inputs, incl. `LDB_centroids_clean.geojson`)
     - `CACHE_DIR` → `data/cache/` (intermediate pkl / sqlite / plots)
     - `RESULTS_DIR` → `results/` (persistent outputs, tuned JSON, samples)
   - Cells 2–6 + benchmark: unchanged logic, but all reads/writes go to the
     local directories above instead of Google Drive.

> Each compute cell keeps a small path fallback, so it still works if run
> without the "LOCAL PATHS" cell first.

## Path migration notes

- The old Colab setup cell installed packages, mounted Drive and cloned the repo
  from git. That is replaced by the environment-check cell + `requirements.txt`.
- The repo's `.git` is used only for version control locally; the notebook no
  longer runs `git reset --hard` (that would wipe local changes).
- `GDRIVE_BASE` / `OUTPUT_DIR` / `DRIVE_CACHE` now point to `results/`
  (`LOCAL_CACHE` → `data/cache/`), so the existing dual-storage code simply
  keeps a working cache plus a persistent results copy.
- `ommx-openjij-adapter` (in the old setup list) does not exist on PyPI and has
  been dropped. SCIP support uses `ommx-pyscipopt-adapter`.

## Tooling

- `tools/refactor_local_paths.py` – (re)applies the Colab → local path refactor
  to the notebook (idempotent; clears stale outputs).
- `tools/patch_benchmark_cell.py` – patches the benchmark cell from
  `tools/benchmark_cell_src.py` (path-local aware).
