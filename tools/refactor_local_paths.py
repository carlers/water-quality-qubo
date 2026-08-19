"""
Refactor notebooks/experiment.ipynb from a Colab / Google Drive notebook to a
local venv (VS Code Python kernel) notebook.

What changes:
  - Cell 0 (SETUP):  the Colab "install + mount Drive + clone repo" cell is
                     replaced with a LOCAL environment check + dependency install.
  - Cell 2 (refresh):the "/content git fetch/reset" cell is replaced with local
                     path resolution (repo root, data/, cache/, results/) plus
                     sys.path setup.
  - Cells 3-10:      every "/content/wqm_data" and "/content/drive/MyDrive/wqm_data"
                     reference is migrated to:
                       LOCAL_CACHE  -> CACHE_DIR   (data/cache  - intermediate pkl/sqlite)
                       GDRIVE_BASE / OUTPUT_DIR / DRIVE_CACHE -> RESULTS_DIR (results/)
  - All stored outputs are cleared (they contain stale Colab paths) and every
    execution_count is reset so the notebook starts from a clean slate.

The refactor keeps the notebook's self-contained style: each cell that reads or
writes paths carries a small fallback that resolves the repo root on its own if
the "LOCAL PATHS & MODULE RELOAD" cell (cell 2) has not been run yet.

Usage:
    .venv/bin/python tools/refactor_local_paths.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "experiment.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_source_list(text: str):
    """nbformat-style source: a list of lines, each with a trailing newline
    except the very last line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


# Small standalone snippet that reproduces the central path logic so every
# self-contained cell still works if the "LOCAL PATHS" cell hasn't run yet.
PATH_FALLBACK = '''\
try:
    CACHE_DIR
except NameError:
    REPO_PATH = Path.cwd().resolve()
    for _ in range(6):
        if (REPO_PATH / "notebooks").exists() or (REPO_PATH / "src").exists() or (REPO_PATH / ".git").exists():
            break
        REPO_PATH = REPO_PATH.parent
    DATA_DIR = REPO_PATH / "data"
    CACHE_DIR = DATA_DIR / "cache"
    RESULTS_DIR = REPO_PATH / "results"
'''


# ---------------------------------------------------------------------------
# New Cell 0 - LOCAL ENVIRONMENT CHECK (replaces the Colab SETUP cell)
# ---------------------------------------------------------------------------
NEW_CELL0 = '''#@title CELL 1: LOCAL ENVIRONMENT CHECK (Venv / Local Kernel)
"""
================================================================================
LOCAL SETUP v1.0: Verify the active venv & install missing packages.
================================================================================
- Runs inside the LOCAL .venv Python kernel (NOT Colab).
- No Google Drive mount, no repo clone, no numpy downgrade.
  (Python 3.14 needs numpy>=2.x, so the old Colab "numpy<2.0" step is gone.)
- Installs ONLY the missing packages into the ACTIVE venv.
- The same list lives in requirements.txt at the repo root.
================================================================================
"""
import sys
import subprocess
import time
import importlib
import importlib.util
import shlex
import warnings
import site

warnings.filterwarnings("ignore")

print("=" * 70)
print("🚀 LOCAL SETUP: Environment Check & Dependency Install")
print("=" * 70)
print(f"  Python: {sys.version.split()[0]}  |  Executable: {sys.executable}")

# (distribution name, import name) actually imported by the notebook cells.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("shapely", "shapely"),
    ("tqdm", "tqdm"),
    ("ipywidgets", "ipywidgets"),
    ("jijmodeling", "jijmodeling"),
    ("openjij", "openjij"),
    ("ommx-pyscipopt-adapter", "ommx_pyscipopt_adapter"),
    ("optuna", "optuna"),
    ("wandb", "wandb"),
]


def run_command(cmd, description=None, max_retries=2, wait=2, timeout=900):
    if description:
        print(f"  ▶ {description}...")
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"    Retry {attempt}/{max_retries}...")
            time.sleep(wait)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, check=False,
            )
            if result.returncode == 0:
                out = result.stdout.strip()
                if len(out) > 300:
                    out = out[:300] + "..."
                if out:
                    print(f"    {out}")
                return True
            err = result.stderr.strip()
            if len(err) > 200:
                err = err[-200:]
            print(f"    Attempt {attempt} failed: {err}")
        except subprocess.TimeoutExpired:
            print(f"    Attempt {attempt} timed out after {timeout}s")
        except Exception as e:
            print(f"    Attempt {attempt} error: {e}")
    print(f"  ❌ Command failed after {max_retries} attempts: {cmd}")
    return False


def is_importable(import_name):
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


# ---- Check / install ----
missing = [dist for dist, mod in REQUIRED_PACKAGES if not is_importable(mod)]
if missing:
    print(f"\\n[1] Missing {len(missing)} package(s): {', '.join(missing)}")
    print("    Installing into the active venv...")
    ok = run_command(
        f"{shlex.quote(sys.executable)} -m pip install {' '.join(missing)} --quiet",
        "pip install missing packages",
    )
    if not ok:
        print("  ❌ Installation failed. You can also install manually with:\\n")
        print("      .venv/bin/pip install -r requirements.txt")
else:
    print("\\n[1] All required packages are already installed. ✅")

# ---- Refresh import paths ----
site.main()
importlib.invalidate_caches()

print("\\n" + "=" * 70)
print("✅ LOCAL SETUP COMPLETE")
print("=" * 70)
'''


# ---------------------------------------------------------------------------
# New Cell 2 - LOCAL PATHS & MODULE RELOAD (replaces the "/content git" cell)
# ---------------------------------------------------------------------------
NEW_CELL2 = '''#@title [RUN] LOCAL PATHS & MODULE RELOAD
"""
Defines every path used by the rest of the notebook:
  REPO_PATH    - root of this repository
  DATA_DIR     - data/           (raw inputs, e.g. the GeoJSON)
  CACHE_DIR    - data/cache/     (intermediate artifacts: pkl, sqlite, plots)
  RESULTS_DIR  - results/        (persistent outputs, tuned json, samples)
  GEOJSON_PATH - data/LDB_centroids_clean.geojson

Legacy aliases (LOCAL_CACHE, GDRIVE_BASE, OUTPUT_DIR, DRIVE_CACHE) are kept so
any remaining references behave correctly after the Colab -> local migration.
"""
import os
import sys
from pathlib import Path

# === REPO PATH ===
# Walk up from the kernel's cwd until we find the folder containing `notebooks/`.
REPO_PATH = Path.cwd().resolve()
for _ in range(6):
    if (REPO_PATH / "notebooks").exists() or (REPO_PATH / "src").exists() or (REPO_PATH / ".git").exists():
        break
    REPO_PATH = REPO_PATH.parent
os.chdir(REPO_PATH)
print(f"📂 Repo root: {REPO_PATH}")

# === LOCAL PATHS (single source of truth) ===
DATA_DIR = REPO_PATH / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = REPO_PATH / "results"
GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Legacy aliases (used by older cells during the Colab -> local migration)
LOCAL_CACHE = CACHE_DIR
GDRIVE_BASE = RESULTS_DIR
OUTPUT_DIR = RESULTS_DIR
DRIVE_CACHE = RESULTS_DIR

print(f"   DATA_DIR    : {DATA_DIR}")
print(f"   CACHE_DIR   : {CACHE_DIR}")
print(f"   RESULTS_DIR : {RESULTS_DIR}")
print(f"   GEOJSON     : {GEOJSON_PATH}  (exists={GEOJSON_PATH.exists()})")

# === PYTHON PATH ===
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(REPO_PATH / "src"))

# === PURGE CACHED MODULES ===
modules_to_reload = [
    'src.model', 'src.solvers', 'src.utils', 'src.plotting',
    'src.environment', 'src.experiment', 'src.jij_model',
    'src.jij_solvers', 'src.jij_optuna',
    'data.synthetic_data', 'tests.benchmark',
]
for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"🔄 Reloaded: {mod}")

print("\\n✅ Paths configured. Ready to run.")
'''


# ---------------------------------------------------------------------------
# Block replacements for the cells that defined their own paths
# ---------------------------------------------------------------------------
OLD_CELL3_PATHS = '''# -----------------------------------------------------------------------------
# USER CONFIGURATION (unchanged)
# -----------------------------------------------------------------------------
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
LOCAL_CACHE = Path("/content/wqm_data")

GEOJSON_PATH = os.path.join(GDRIVE_BASE, "LDB_centroids_clean.geojson")
OUTPUT_DIR = GDRIVE_BASE
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GDRIVE_BASE.mkdir(parents=True, exist_ok=True)

if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"❌ GeoJSON not found at: {GEOJSON_PATH}")'''

NEW_CELL3_PATHS = '''# -----------------------------------------------------------------------------
# PATHS (single source of truth is the "LOCAL PATHS & MODULE RELOAD" cell)
# -----------------------------------------------------------------------------
''' + PATH_FALLBACK + '''
GEOJSON_PATH = DATA_DIR / "LDB_centroids_clean.geojson"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"❌ GeoJSON not found at: {GEOJSON_PATH}")'''

OLD_CELL6_PATHS = '''# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")'''

NEW_CELL6_PATHS = '''# -----------------------------------------------------------------------------
# PATHS (fallback if the "LOCAL PATHS & MODULE RELOAD" cell hasn't run yet)
# -----------------------------------------------------------------------------
''' + PATH_FALLBACK

OLD_CELL8_PATHS = '''LOCAL_CACHE = Path("/content/wqm_data")
GDRIVE_BASE = Path("/content/drive/MyDrive/wqm_data")
# Use config_hash in the run directory to separate experiments
RUN_DIR = GDRIVE_BASE / f"run_{config_hash}"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)'''

NEW_CELL8_PATHS = '''# -----------------------------------------------------------------------------
# PATHS (fallback if the "LOCAL PATHS & MODULE RELOAD" cell hasn't run yet)
# -----------------------------------------------------------------------------
''' + PATH_FALLBACK + '''
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Use config_hash in the run directory to separate experiments
RUN_DIR = RESULTS_DIR / f"run_{config_hash}"
RUN_DIR.mkdir(parents=True, exist_ok=True)'''

OLD_CELL9_PATHS = 'cache_dir = Path("/content/wqm_data")'
NEW_CELL9_PATHS = '''# Fallback path resolution (same as the "LOCAL PATHS & MODULE RELOAD" cell)
try:
    CACHE_DIR
except NameError:
    REPO_PATH = Path.cwd().resolve()
    for _ in range(6):
        if (REPO_PATH / "notebooks").exists() or (REPO_PATH / "src").exists() or (REPO_PATH / ".git").exists():
            break
        REPO_PATH = REPO_PATH.parent
    CACHE_DIR = REPO_PATH / "data" / "cache"
cache_dir = CACHE_DIR'''

OLD_CELL10_PATHS = '''LOCAL_CACHE = Path('/content/wqm_data')
DRIVE_CACHE = Path('/content/drive/MyDrive/wqm_data')
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)'''

NEW_CELL10_PATHS = '''# -----------------------------------------------------------------------------
# PATHS (fallback if the "LOCAL PATHS & MODULE RELOAD" cell hasn't run yet)
# -----------------------------------------------------------------------------
''' + PATH_FALLBACK + '''
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)'''

OLD_CACHE_FILE = """    'CACHE_FILE': f'/content/wqm_data/benchmark_results_{config_hash}.pkl',"""
NEW_CACHE_FILE = """    'CACHE_FILE': str(CACHE_DIR / f'benchmark_results_{config_hash}.pkl'),"""


# ---------------------------------------------------------------------------
# Per-cell transforms: each entry is a list of (old, new) exact replacements,
# applied in order. Global renames come after the block replacements.
# ---------------------------------------------------------------------------
TRANSFORMS = {
    "e16da7f9": [  # CELL 2: REAL DATA INGESTOR + MASTER MIQP
        (OLD_CELL3_PATHS, NEW_CELL3_PATHS),
        ("GDRIVE_BASE", "RESULTS_DIR"),
        ("OUTPUT_DIR", "RESULTS_DIR"),
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("# - Dual storage: local and Drive.", "# - Dual storage: cache and results."),
        ("Loading hashed master data from Drive (copying to local)",
         "Loading hashed master data from results (copying to cache)"),
        ("Loading unhashed master data from Drive (migrating to hashed)",
         "Loading unhashed master data from results (migrating to hashed)"),
    ],
    "6a321279": [  # CELL 3: NESTED RESOLUTION SCALING
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("OUTPUT_DIR", "RESULTS_DIR"),
        ("- Dual storage: local and Drive.", "- Dual storage: cache and results."),
        ("Loading scaling results from Drive (copying to local)",
         "Loading scaling results from results (copying to cache)"),
    ],
    "35d53b65": [  # CELL 4: GENERATE SPARSE QUBO + MIQP DATA
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("OUTPUT_DIR", "RESULTS_DIR"),
        ("- Dual storage: local and Drive.", "- Dual storage: cache and results."),
    ],
    "df76202f": [  # DELETE SCIP RESULTS
        (OLD_CELL6_PATHS, NEW_CELL6_PATHS),
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("GDRIVE_BASE", "RESULTS_DIR"),
        ("from local and Drive.", "from cache and results."),
    ],
    "5db92c6a": [  # CELL 5: MIQP SOLVE WITH SCIP
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("GDRIVE_BASE", "RESULTS_DIR"),
        ("Loading hashed master from Drive (copying to local)",
         "Loading hashed master from results (copying to cache)"),
        ("Loaded hashed master from Drive.", "Loaded hashed master from results."),
        ("Loading unhashed master from Drive (migrating to hashed)",
         "Loading unhashed master from results (migrating to hashed)"),
        ("Copying SCIP result from Drive to local:",
         "Copying SCIP result from results to cache:"),
        ("No instance_data_N*.pkl files found in local or Drive.",
         "No instance_data_N*.pkl files found in cache or results."),
    ],
    "eb81ee52": [  # CELL 6: SA TUNE
        (OLD_CELL8_PATHS, NEW_CELL8_PATHS),
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("GDRIVE_BASE", "RESULTS_DIR"),
        ("Dual SQLite storage: local (fast) + Drive (persistent), periodic sync.",
         "Dual SQLite storage: cache (fast) + results (persistent), periodic sync."),
        ("Loaded existing SA samples from Drive (copied to local)",
         "Loaded existing SA samples from results (copied to cache)"),
        ("Copying study DB from Drive to local:",
         "Copying study DB from results to cache:"),
        ('    "USE_WANDB": True,',
         '    "USE_WANDB": False,  # local default; set True (after `wandb login`) to enable'),
    ],
    "1edc5a91": [  # DELETE BENCHMARK CACHE
        (OLD_CELL9_PATHS, NEW_CELL9_PATHS),
    ],
    "948ccc34": [  # BENCHMARK
        (OLD_CELL10_PATHS, NEW_CELL10_PATHS),
        ("LOCAL_CACHE", "CACHE_DIR"),
        ("DRIVE_CACHE", "RESULTS_DIR"),
        (OLD_CACHE_FILE, NEW_CACHE_FILE),
    ],
}

# Cells that get their ENTIRE source replaced
FULL_REPLACE = {
    "c0727250": NEW_CELL0,  # SETUP -> LOCAL ENVIRONMENT CHECK
    "ad5de2f5": NEW_CELL2,  # /content git -> LOCAL PATHS & MODULE RELOAD
}


def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    seen = set()
    # These tokens indicate a cell has NOT been refactored yet. If none are
    # present, the cell is already local-only and we skip it (idempotent re-runs).
    OLD_INDICATORS = (
        "/content", "MyDrive", "LOCAL_CACHE", "OUTPUT_DIR",
        "GDRIVE_BASE", "DRIVE_CACHE", '"USE_WANDB": True',
    )
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid in FULL_REPLACE:
            cell["source"] = to_source_list(FULL_REPLACE[cid])
            seen.add(cid)
            continue
        if cid in TRANSFORMS:
            src = "".join(cell.get("source", []))
            if not any(tok in src for tok in OLD_INDICATORS):
                print(f"  · ({cid}) already refactored - skipping")
                seen.add(cid)
                continue
            for old, new in TRANSFORMS[cid]:
                count = src.count(old)
                # Pure identifier renames legitimately hit 0 after the path-block
                # replacement already consumed them - don't warn for those.
                if count == 0 and old not in ("GDRIVE_BASE", "OUTPUT_DIR", "LOCAL_CACHE", "DRIVE_CACHE"):
                    print(f"  ⚠️ ({cid}) pattern NOT FOUND: {old[:60]!r}")
                src = src.replace(old, new)
            cell["source"] = to_source_list(src)
            seen.add(cid)

    missing = (set(FULL_REPLACE.keys()) | set(TRANSFORMS.keys())) - seen
    if missing:
        print(f"❌ Some cells were not found in the notebook: {sorted(missing)}")
        raise SystemExit(1)

    # Clear stale outputs (they contain old Colab paths) and reset execution counts
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    leftover = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        # Real Colab references (the legacy alias identifiers above are expected)
        if "/content" in src or "MyDrive" in src:
            leftover.append(cell.get("id"))
    if leftover:
        print(f"⚠️  Remaining colab/Drive path references in cells: {leftover}")
    else:
        print("   No /content or MyDrive references remain.")

    print(f"✅ Refactored notebook -> {NB_PATH}")
    print(f"   Processed {len(seen)} cells; cleared outputs on all code cells.")


if __name__ == "__main__":
    main()