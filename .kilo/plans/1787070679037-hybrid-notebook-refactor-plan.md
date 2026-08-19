# Hybrid Notebook Refactor Plan (Direct Edit Approach)

## Goal

Transform `notebooks/experiment.ipynb` from Colab-only to a **hybrid** notebook:
- **Local (venv)**: saves to `REPO_PATH/data/cache/` (fast) + `REPO_PATH/results/` (persistent)
- **Colab**: auto-detects via `google.colab`, mounts Drive, saves to `/content/wqm_data` (fast) + `/content/drive/MyDrive/wqm_data` (persistent)

Single environment flag (`IN_COLAB`) gates all path resolution. No hard-coded
`/content` or `MyDrive` strings remain in any cell source.

---

## Notebook State (11 cells)

| Cell | id | Current state | Action |
|---|---|---|---|
| 0 | `c0727250` | Colab SETUP (mount + clone + install + numpy downgrade) | **Replace source** (hybrid) |
| 1 | `ca003397` | Master control flags | No change |
| 2 | `ad5de2f5` | `/content` git refresh | **Replace source** (hybrid paths) |
| 3 | `e16da7f9` | Data ingestor (hard-coded GDRIVE_BASE/LOCAL_CACHE) | **Replace path block** |
| 4 | `6a321279` | Scaling (uses LOCAL_CACHE alias) | **Insert PATH_FALLBACK** |
| 5 | `35d53b65` | QUBO builder (uses LOCAL_CACHE alias) | **Insert PATH_FALLBACK** |
| 6 | `df76202f` | Delete SCIP results (hard-coded paths) | **Replace path block** |
| 7 | `5db92c6a` | SCIP solve (uses GDRIVE_BASE alias) | **Insert PATH_FALLBACK** |
| 8 | `eb81ee52` | SA tune (hard-coded paths, USE_WANDB=True) | **Replace path block + flags** |
| 9 | `1edc5a91` | Delete benchmark cache (hard-coded path) | **Replace source** |
| 10 | `948ccc34` | Benchmark (patched, local-only PATH_FALLBACK) | **Replace PATH_FALLBACK** |

---

## Direct-Edit Tasks (in order)

### Task 1: Cell 0 (c0727250) — Hybrid SETUP

Replace the entire cell source with a hybrid version:
1. Env detection: `IN_COLAB = importlib.util.find_spec("google.colab") is not None`
2. Colab branch: mount Drive → git clone/checkout `/content/water-quality-qubo`
   → install packages (existing list) → numpy<2.0 downgrade
3. Local branch: print venv info → install missing packages via
   `importlib.util.find_spec` → no numpy downgrade
4. `site.main()` + `importlib.invalidate_caches()` in both branches
5. Print `IN_COLAB` status

### Task 2: Cell 2 (ad5de2f5) — Hybrid Paths & Module Reload

Replace the entire cell source:
1. Re-detect `IN_COLAB` (in case Cell 0 wasn't run)
2. If Colab: `REPO_PATH = Path("/content/water-quality-qubo")`, git pull, chdir
3. If local: walk up from `Path.cwd()` to find repo root (notebooks/src/.git),
   no git operations
4. Set unified paths:
   ```python
   if IN_COLAB:
       CACHE_DIR = Path("/content/wqm_data")
       RESULTS_DIR = Path("/content/drive/MyDrive/wqm_data")
       DATA_DIR = RESULTS_DIR
   else:
       DATA_DIR = REPO_PATH / "data"
       CACHE_DIR = DATA_DIR / "cache"
       RESULTS_DIR = REPO_PATH / "results"
   ```
5. Legacy aliases: `LOCAL_CACHE = CACHE_DIR`, `GDRIVE_BASE = RESULTS_DIR`,
   `OUTPUT_DIR = RESULTS_DIR`, `DRIVE_CACHE = RESULTS_DIR`,
   `GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"`
6. mkdir + validate GeoJSON + sys.path + module purge (same module list)

### Task 3: Cells 3, 6, 8 — Replace hard-coded path blocks

In each of these cells, find the block that defines `LOCAL_CACHE = Path("/content/...")`
etc. and replace it with the standard **HYBRID PATH_FALLBACK** snippet below.

### Task 4: Cell 3 (e16da7f9) — Data Ingestor path replacement

Replace lines:
```python
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE = Path("/content/wqm_data")
GEOJSON_PATH = os.path.join(GDRIVE_BASE, "LDB_centroids_clean.geojson")
OUTPUT_DIR = GDRIVE_BASE
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GDRIVE_BASE.mkdir(parents=True, exist_ok=True)
```
with `PATH_FALLBACK` + `GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"`.

### Task 5: Cell 6 (df76202f) — Delete SCIP results path replacement

Replace:
```python
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
```
with `PATH_FALLBACK`. Keep delete function logic unchanged.

### Task 6: Cell 8 (eb81ee52) — SA TUNE path + flag replacement

Replace:
```python
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
RUN_DIR = GDRIVE_BASE / f"run_{config_hash}"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
```
with `PATH_FALLBACK` + `RUN_DIR = RESULTS_DIR / f"run_{config_hash}"`.

Also change `"USE_WANDB": True` → `"USE_WANDB": False` with comment
`# local default; set True (after `wandb login`) to enable`.

### Task 7: Cells 4, 5, 7 — Insert PATH_FALLBACK for self-contained safety

After the import block (before CONFIG or first path usage), insert the
HYBRID PATH_FALLBACK snippet. These cells already use legacy aliases
(`LOCAL_CACHE`, `OUTPUT_DIR`, `GDRIVE_BASE`) that will resolve correctly
from Cell 2's definitions; the fallback ensures they work standalone.

For Cell 7, also update the FileNotFoundError message:
`"No instance_data_N*.pkl files found in local or Drive."` →
`"No instance_data_N*.pkl files found in cache or results."`

### Task 8: Cell 9 (1edc5a91) — Delete Benchmark Cache

Replace the entire cell source with:
```python
#@title 🗔️ Delete Benchmark Cache
"""Delete cached benchmark result files (config-hash aware + legacy)."""
from pathlib import Path

# --- PATH FALLBACK (hybrid: Colab + local) ---
try:
    CACHE_DIR
except NameError:
    # [HYBRID PATH_FALLBACK — same snippet]
    ...

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config-hash-aware caches
for cache_file in CACHE_DIR.glob("benchmark_results_*.pkl"):
    cache_file.unlink()
    print(f"✅ Deleted {cache_file}")

# Legacy unhashed cache
for cache_file in RESULTS_DIR.glob("benchmark_results_*.pkl"):
    cache_file.unlink()
    print(f"✅ Deleted {cache_file}")

legacy = CACHE_DIR / "benchmark_results.pkl"
if legacy.exists():
    legacy.unlink()
    print(f"✅ Deleted {legacy}")
legacy_r = RESULTS_DIR / "benchmark_results.pkl"
if legacy_r.exists():
    legacy_r.unlink()
    print(f"✅ Deleted {legacy_r}")

if not any(...):
    print("ℹ️ No benchmark cache files found (already clean).")
```

### Task 9: Cell 10 (948ccc34) — Benchmark PATH_FALLBACK upgrade

The cell source was patched from `benchmark_cell_src.py`. Update its
PATH_FALLBACK (currently local-only) to the hybrid version. Update
`LOCAL_CACHE` / `DRIVE_CACHE` usage in `load_instance` and `run_benchmark`
to use `CACHE_DIR` / `RESULTS_DIR` (the fallback now sets these aliases).

### Task 10: Clear all outputs + reset execution_count

After all source edits, set `"outputs": []` and `"execution_count": null`
on every code cell.

---

## HYBRID PATH_FALLBACK (canonical snippet)

Insert this verbatim in Cells 3, 4, 5, 6, 7, 8, 9, 10 (after imports,
before first path usage):

```python
# === PATH FALLBACK (hybrid: Colab + local) ===
try:
    CACHE_DIR
except NameError:
    import importlib.util
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
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
    # Legacy aliases (used by existing cell code)
    LOCAL_CACHE = CACHE_DIR
    GDRIVE_BASE = RESULTS_DIR
    OUTPUT_DIR = RESULTS_DIR
    DRIVE_CACHE = RESULTS_DIR
    GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Design Rationale

- **Legacy aliases preserved**: Cells 3-10 already use `LOCAL_CACHE`,
  `GDRIVE_BASE`, `OUTPUT_DIR`, `DRIVE_CACHE` — these alias to the correct
  backend per environment. No per-cell I/O logic change needed; only path
  *definitions* change.
- **Dual-storage preserved**: Fast cache (local tmp / `data/cache/`) +
  persistent store (Drive / `results/`). The `shutil.copy` pattern in
  existing cells copies persistent→fast on load.
- **Detection is safe**: `importlib.util.find_spec("google.colab")` does not
  import or mount — it just checks if the module is installed.
- **Cell 0 only sets `IN_COLAB` + mounts Drive** — path setup lives in Cell 2
  for separation of concerns.
- **GEOJSON_PATH**: `DATA_DIR / "LDB_centroids_clean.geojson"` works in both
  modes (Colab: `DATA_DIR == RESULTS_DIR == /content/drive/MyDrive/wqm_data`;
  Local: `DATA_DIR == REPO_PATH/data`).

---

## Validation Plan

1. **Syntax**: `python3 -c "import json; nb=json.load(open('notebooks/experiment.ipynb')); [compile(''.join(c['source']), f'c{i}', 'exec') for i,c in enumerate(nb['cells']) if c['cell_type']=='code']"`

2. **Path audit**: Confirm no `/content` or `MyDrive` string literals remain in
   any cell `source` (grep the notebook JSON for these in source lines only).

3. **Alias consistency**: Every cell that uses `LOCAL_CACHE`, `GDRIVE_BASE`,
   `OUTPUT_DIR`, `DRIVE_CACHE`, `CACHE_DIR`, `RESULTS_DIR`, `DATA_DIR`, or
   `GEOJSON_PATH` either has PATH_FALLBACK or follows Cell 2 which defines them.

4. **Hybrid fallback test**: Extract the PATH_FALLBACK code, run it in a
   subprocess, verify it resolves `CACHE_DIR` and `RESULTS_DIR` correctly
   without attempting a Drive mount.

5. **Benchmark validation**: Copy `benchmark_cell_src.py` → `/tmp/bench_cell.py`,
   run `python3 tools/validate_benchmark_cell.py`, confirm
   `canonical_qubo_terms` returns 3-tuple `(linear, quad, offset)`.

6. **Idempotency check**: Run the notebook's Cell 0 + Cell 2 in sequence
   (locally, no Colab) — Cell 2's PATH_FALLBACK should be a no-op (aliases
   already defined).

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `!git` magic doesn't work in local VS Code kernel | Cell 2 uses `subprocess.run(["git", ...])` only inside `if IN_COLAB:` branch; local branch skips git entirely. |
| Drive mount prompt blocks local runs | `find_spec("google.colab")` → `None` locally; mount code never executes. |
| `shutil.copy` persistent→fast fails if source missing | Existing cells already guard with `if src.exists()` in most places; PATH_FALLBACK only `mkdir`s, doesn't copy. |
| GeoJSON path inconsistency (`tests/data_ingestor.py` uses `/water_quality_results/`) | Use `DATA_DIR / "LDB_centroids_clean.geojson"` consistently in notebook; tests/ files are out of scope. |
| numpy downgrade harmful in local Python 3.14 | Gated behind `if IN_COLAB:` — never runs locally. |
| `tqdm.notebook` import in local kernel | Works in VS Code Jupyter extension; acceptable. |
