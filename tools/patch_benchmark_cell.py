"""
Patch notebooks/experiment.ipynb:
  - Replace the benchmark cell (id 948ccc34) with tools/benchmark_cell_src.py content
  - Update the delete-cache cell (id 1edc5a91) to clear hashed cache files
  - Clear stale outputs / execution_count on both cells
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "experiment.ipynb"
SRC_PATH = ROOT / "tools" / "benchmark_cell_src.py"

DEL_CACHE_SRC = '''#@title 🗑️ Delete Benchmark Cache
"""
Delete all cached benchmark result files (config-hash aware + legacy).
Run this before re-benchmarking after any code fix or config change.
"""
from pathlib import Path

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

if not any(CACHE_DIR.glob("benchmark_results_*.pkl")) and not any(RESULTS_DIR.glob("benchmark_results_*.pkl")) and not legacy.exists() and not legacy_r.exists():
    print("ℹ️ No benchmark cache files found (already clean).")
'''


def to_source_list(text: str):
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main():
    bench_src = SRC_PATH.read_text(encoding="utf-8")
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    found_bench = found_del = False
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid == "948ccc34":  # benchmark cell
            cell["source"] = to_source_list(bench_src)
            cell["outputs"] = []
            cell["execution_count"] = None
            found_bench = True
        elif cid == "1edc5a91":  # delete cache cell
            cell["source"] = to_source_list(DEL_CACHE_SRC)
            cell["outputs"] = []
            cell["execution_count"] = None
            found_del = True

    if not (found_bench and found_del):
        raise RuntimeError(f"Cell lookup failed: bench={found_bench} del={found_del}")

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print("✅ Patched notebook: benchmark cell + delete-cache cell.")


if __name__ == "__main__":
    main()
