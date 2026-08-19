#@title 📊 BENCHMARK: JijModeling vs Custom Sparse Builder
"""
Benchmark the QUBO build path used during SA tuning.

This cell compares two builders on the same compressed instance data:
- JijModeling `to_qubo` on the compiled instance
- a custom sparse builder that emits the same canonical QUBO terms directly

The benchmark separates builder time from sampler time and uses a single
canonical sparse edge convention for correctness checks.

FIXES (v2):
1. build_penalty_weights now maps constraint ids (same as Cell 6's
   get_penalty_weights) instead of the fragile penalty_method().parameters API.
2. canonical_qubo_terms now also returns the constant offset
   (lambda_budget * K**2 + lambda_conn * sum_i b_i**2), and the energy
   comparison includes it, so max_abs_jm_vs_custom is a real per-term check.
3. canonical_to_sampler_qubo emits only upper-triangular (i<j) off-diagonals
   (OpenJij dict convention) instead of duplicating (i,j) and (j,i), which
   would double-count quadratic energies in the sampler.
4. Cache file is config-hash aware; load_instance falls back to unhashed files.
"""

import json
import hashlib
import pickle
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jijmodeling as jm
import openjij as oj
import warnings

warnings.filterwarnings('ignore')

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


def get_config_hash() -> str:
    try:
        if 'config_hash' in globals() and isinstance(config_hash, str) and len(config_hash) == 8:
            return config_hash
    except Exception:
        pass
    default_config = {
        'L_c': 7500.0,
        'L_w': 1000.0,
        'Beta': 1.0,
        'Delta': 1.0,
        'Current_vector': (1.0, 0.0),
        'K': 5,
        'D_max_buffer': 1.15,
    }
    return hashlib.md5(json.dumps(default_config, sort_keys=True).encode()).hexdigest()[:8]


config_hash = get_config_hash()
print(f'Benchmark config hash: {config_hash}')

CONFIG_BENCH = {
    'N_TESTS': [200, 500, 1000],
    'PENALTY_PAIRS': [
        (0.1, 0.1),
        (1.0, 1.0),
        (10.0, 10.0),
    ],
    'NUM_READS': 256,
    'NUM_SWEEPS': 200,
    'REPEAT_TIMES': 3,
    'RANDOM_SOLUTIONS': 5,
    'CACHE_FILE': str(CACHE_DIR / f'benchmark_results_{config_hash}.pkl'),
}


def build_augmented_model() -> jm.Problem:
    problem = jm.Problem('WQM_Augmented', sense=jm.ProblemSense.MINIMIZE)
    n_ph = problem.Length('N')
    k_ph = problem.Length('K')
    a_ph = problem.Float('a', shape=(n_ph,))
    q_ph = problem.Float('Q', shape=(n_ph, n_ph))
    neigh_ph = problem.Binary('neigh', shape=(n_ph, n_ph))
    fixed_neighbors_ph = problem.Float('fixed_neighbors', shape=(n_ph,))
    x = problem.BinaryVar('x', shape=(n_ph,))

    problem += jm.sum(n_ph, lambda i: a_ph[i] * x[i])
    problem += jm.sum(jm.product(n_ph, n_ph).filter(lambda i, j: i < j), lambda i, j: q_ph[i, j] * x[i] * x[j])
    problem += problem.Constraint('budget', jm.sum(n_ph, lambda i: x[i]) == k_ph)
    problem += problem.Constraint(
        'connectivity',
        lambda i: x[i] <= fixed_neighbors_ph[i] + jm.sum(n_ph, lambda j: neigh_ph[i, j] * x[j]),
        domain=n_ph,
    )
    return problem


def compile_instance(problem: jm.Problem, instance_data: dict):
    placeholder_names = {p.name for p in problem.placeholders.values()}
    filtered = {k: v for k, v in instance_data.items() if k in placeholder_names}
    return problem.eval(filtered)


def load_instance(N: int) -> dict:
    # Hashed first (config-hash aware), then unhashed (backward compatible)
    candidates = []
    hashed_name = f'instance_data_N{N}_{config_hash}.pkl'
    unhashed_name = f'instance_data_N{N}.pkl'
    for base in (CACHE_DIR, RESULTS_DIR):
        candidates.append(base / hashed_name)
    for base in (CACHE_DIR, RESULTS_DIR):
        candidates.append(base / unhashed_name)

    for candidate in candidates:
        if candidate.exists():
            if candidate.parent == RESULTS_DIR:
                shutil.copy(candidate, CACHE_DIR / candidate.name)
                candidate = CACHE_DIR / candidate.name
            with open(candidate, 'rb') as f:
                instance = pickle.load(f)
            print(f'📂 Loaded benchmark instance from {candidate}')
            n_free = int(instance['N'])
            for i, j, _ in instance.get('Q_edges', []):
                if i >= n_free or j >= n_free:
                    raise ValueError(f'Q_edges contains out-of-range index ({i}, {j}) for N={n_free}')
            return instance
    raise FileNotFoundError(f'Instance file not found for N={N} (hashed={hashed_name})')


def adapt_sparse_to_dense(instance_data: dict) -> dict:
    dense = dict(instance_data)
    n = int(dense['N'])
    q = np.zeros((n, n), dtype=float)
    for i, j, val in dense.get('Q_edges', []):
        q[i, j] = float(val)
        q[j, i] = float(val)
    neigh = np.zeros((n, n), dtype=int)
    for i, nbrs in enumerate(dense.get('neighbors', [])):
        for j in nbrs:
            if 0 <= j < n:
                neigh[i, j] = 1
    fixed_neighbors_raw = dense.get('fixed_neighbors', None)
    if fixed_neighbors_raw is None:
        fixed_neighbors = np.zeros(n, dtype=float)
    else:
        fixed_neighbors = np.asarray(fixed_neighbors_raw, dtype=float)
        if fixed_neighbors.shape[0] != n:
            fixed_neighbors = np.resize(fixed_neighbors, n).astype(float)
    dense['Q'] = q
    dense['neigh'] = neigh
    dense['fixed_neighbors'] = fixed_neighbors
    return dense


def _key_to_pair(key):
    ints = []

    def collect(value):
        if isinstance(value, (int, np.integer)):
            ints.append(int(value))
        elif isinstance(value, tuple):
            for elem in value:
                collect(elem)

    collect(key)
    if not ints:
        raise ValueError(f'Could not parse QUBO key: {key!r}')
    if len(ints) == 1:
        return ints[0], ints[0]
    return ints[0], ints[1]


def qubo_energy(qubo_dict, offset, x):
    x = np.asarray(x, dtype=int)
    energy = float(offset)
    seen = set()
    for key, val in qubo_dict.items():
        i, j = _key_to_pair(key)
        pair = (i, j) if i <= j else (j, i)
        if pair in seen:
            continue
        seen.add(pair)
        if i == j:
            energy += float(val) * int(x[i])
        else:
            energy += float(val) * int(x[i]) * int(x[j])
    return float(energy)


def canonical_energy(linear_dict, quad_dict, x, offset=0.0):
    x = np.asarray(x, dtype=int)
    energy = float(offset)
    for i, val in linear_dict.items():
        energy += float(val) * int(x[i])
    for (i, j), val in quad_dict.items():
        energy += float(val) * int(x[i]) * int(x[j])
    return float(energy)



def canonical_qubo_terms(instance: dict, lambda_budget: float, lambda_conn: float):
    """
    Build the penalised QUBO directly from the sparse instance data.

    Returns (linear_dict, quad_dict, offset):
      - linear_dict: {i: coeff}
      - quad_dict:   {(i, j): coeff} with i < j
      - offset:      constant penalty terms:
                       lambda_budget * K**2              (from (sum x - K)^2)
                     + lambda_conn * sum_i b_i**2        (from (x_i - b_i - sum N x)^2)

    These offsets exactly match the constant JijModeling folds into the
    `const` return value of `to_qubo`, so energies are directly comparable.
    """
    a = np.asarray(instance['a'], dtype=float)
    q_edges = instance.get('Q_edges', [])
    neighbors = instance.get('neighbors', [])
    fixed_neighbors = np.asarray(instance.get('fixed_neighbors', np.zeros(len(a))), dtype=float)
    n = int(instance['N'])
    k = int(instance['K'])

    linear = np.array(a, dtype=float)
    quad = {}
    offset = 0.0

    for i, j, val in q_edges:
        if i == j:
            linear[i] += float(val)
        else:
            key = (i, j) if i < j else (j, i)
            quad[key] = quad.get(key, 0.0) + float(val)

    if lambda_budget != 0:
        # (sum_i x_i - K)^2 = (1-2K) sum_i x_i + 2 sum_{i<j} x_i x_j + K^2
        linear += float(lambda_budget) * (1.0 - 2.0 * k)
        for i in range(n):
            for j in range(i + 1, n):
                quad[(i, j)] = quad.get((i, j), 0.0) + 2.0 * float(lambda_budget)
        offset += float(lambda_budget) * (k * k)

    if lambda_conn != 0:
        # (x_i - b_i - sum_j N_ij x_j)^2 expansion:
        #   linear x_i:    (1 - 2 b_i)
        #   linear x_j:    (1 + 2 b_i) per neighbour j
        #   quad (i,j):    -2 per neighbour j
        #   quad (j,k):    +2 per neighbour pair (j,k)
        #   constant:      b_i^2
        for i, nbrs in enumerate(neighbors):
            b_i = float(fixed_neighbors[i])
            linear[i] += float(lambda_conn) * (1.0 - 2.0 * b_i)
            for j in nbrs:
                linear[j] += float(lambda_conn) * (1.0 + 2.0 * b_i)
                key = (i, j) if i < j else (j, i)
                quad[key] = quad.get(key, 0.0) - 2.0 * float(lambda_conn)
            for idx_j in range(len(nbrs)):
                j = nbrs[idx_j]
                for idx_k in range(idx_j + 1, len(nbrs)):
                    k_nbr = nbrs[idx_k]
                    key = (j, k_nbr) if j < k_nbr else (k_nbr, j)
                    quad[key] = quad.get(key, 0.0) + 2.0 * float(lambda_conn)
            offset += float(lambda_conn) * (b_i * b_i)

    eps = 1e-12
    linear_dict = {i: float(v) for i, v in enumerate(linear) if abs(v) > eps}
    quad_dict = {k: float(v) for k, v in quad.items() if abs(v) > eps}
    return linear_dict, quad_dict, offset


def canonical_to_sampler_qubo(linear_dict, quad_dict):
    """
    Convert canonical (linear_dict, quad_dict) to an OpenJij-style QUBO dict.

    OpenJij dict convention: {(i, i): h_i, (i, j): J_ij} with i < j for the
    off-diagonal terms. Providing both (i,j) and (j,i) would make the sampler
    count the quadratic term twice.
    """
    qubo = {(i, i): float(val) for i, val in linear_dict.items()}
    for (i, j), val in quad_dict.items():
        i, j = int(i), int(j)
        if i == j:
            qubo[(i, i)] = qubo.get((i, i), 0.0) + float(val)
        else:
            qubo[(min(i, j), max(i, j))] = float(val)
    return qubo



def build_penalty_weights(compiled, lambda_budget, lambda_conn):
    """
    Map penalty weights to constraint ids (same approach as Cell 6's
    get_penalty_weights). JijModeling's `to_qubo(penalty_weights=...)`
    expects weights keyed by constraint id.
    """
    weights = {}
    constraint_names = []
    for c in compiled.constraints:
        constraint_names.append(c.name)
        if c.name == 'budget':
            weights[c.id] = float(lambda_budget)
        elif c.name == 'connectivity':
            weights[c.id] = float(lambda_conn)
    if len(weights) < 2:
        raise ValueError(
            f'Expected at least two penalty parameters, found {len(weights)} '
            f'(constraints seen: {constraint_names})'
        )
    return weights


def benchmark_instance(N, penalty_pairs, num_reads, num_sweeps, repeat=3, random_sols=5):
    instance = load_instance(N)
    dense_instance = adapt_sparse_to_dense(instance)
    model = build_augmented_model()
    compiled = compile_instance(
        model,
        {k: v for k, v in dense_instance.items() if k in {'N', 'K', 'a', 'Q', 'neigh', 'fixed_neighbors'}},
    )
    sampler = oj.SASampler()
    results = []

    for lb, lc in penalty_pairs:
        penalty_weights = build_penalty_weights(compiled, lb, lc)

        jm_build_times = []
        jm_qubo = None
        jm_offset = 0.0
        for _ in range(repeat):
            t0 = time.perf_counter()
            jm_qubo, jm_offset = compiled.to_qubo(penalty_weights=penalty_weights)
            jm_build_times.append(time.perf_counter() - t0)
        jm_build_s = float(np.median(jm_build_times))

        jm_sample_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            sampler.sample_qubo(jm_qubo, num_reads=num_reads, num_sweeps=num_sweeps, sparse=True)
            jm_sample_times.append(time.perf_counter() - t0)
        jm_sample_s = float(np.median(jm_sample_times))

        custom_build_times = []
        custom_linear = None
        custom_quad = None
        custom_offset = 0.0
        custom_qubo = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            custom_linear, custom_quad, custom_offset = canonical_qubo_terms(instance, lb, lc)
            custom_qubo = canonical_to_sampler_qubo(custom_linear, custom_quad)
            custom_build_times.append(time.perf_counter() - t0)
        custom_build_s = float(np.median(custom_build_times))

        custom_sample_times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            sampler.sample_qubo(custom_qubo, num_reads=num_reads, num_sweeps=num_sweeps, sparse=True)
            custom_sample_times.append(time.perf_counter() - t0)
        custom_sample_s = float(np.median(custom_sample_times))

        rng = np.random.default_rng(42)
        max_abs_jm_vs_custom = 0.0
        max_abs_custom_internal = 0.0
        for _ in range(random_sols):
            x = rng.integers(0, 2, size=int(instance['N']), dtype=int)
            jm_energy = qubo_energy(jm_qubo, jm_offset, x)
            custom_energy = qubo_energy(custom_qubo, custom_offset, x)
            canonical_energy_value = canonical_energy(custom_linear, custom_quad, x, offset=custom_offset)
            max_abs_jm_vs_custom = max(max_abs_jm_vs_custom, abs(jm_energy - custom_energy))
            max_abs_custom_internal = max(max_abs_custom_internal, abs(custom_energy - canonical_energy_value))

        results.append({
            'N': N,
            'lambda_budget': float(lb),
            'lambda_conn': float(lc),
            'jm_build_s': jm_build_s,
            'jm_sample_s': jm_sample_s,
            'custom_build_s': custom_build_s,
            'custom_sample_s': custom_sample_s,
            'jm_terms': int(len(jm_qubo)),
            'custom_terms': int(len(custom_qubo)),
            'jm_offset': float(jm_offset),
            'custom_offset': float(custom_offset),
            'max_abs_jm_vs_custom': float(max_abs_jm_vs_custom),
            'max_abs_custom_internal': float(max_abs_custom_internal),
            'speedup_build': float(jm_build_s / custom_build_s) if custom_build_s > 0 else np.inf,
        })

        print(
            f'N={N}, lb={lb:.2f}, lc={lc:.2f}: '
            f'build JM {jm_build_s:.4f}s vs custom {custom_build_s:.4f}s | '
            f'sample JM {jm_sample_s:.4f}s vs custom {custom_sample_s:.4f}s | '
            f'terms JM {len(jm_qubo)} vs custom {len(custom_qubo)} | '
            f'|JM-custom|={max_abs_jm_vs_custom:.2e} '
            f'(offset diff |{jm_offset:.2f} - {custom_offset:.2f}| = {abs(jm_offset - custom_offset):.2e})'
        )

    return results



def run_benchmark():
    cache_file = Path(CONFIG_BENCH['CACHE_FILE'])
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        print(f'Loaded cached benchmark results from {cache_file}')
        return df

    all_results = []
    for N in CONFIG_BENCH['N_TESTS']:
        print('\n' + '=' * 70)
        print(f'Benchmarking N={N}')
        print('=' * 70)
        all_results.extend(
            benchmark_instance(
                N=N,
                penalty_pairs=CONFIG_BENCH['PENALTY_PAIRS'],
                num_reads=CONFIG_BENCH['NUM_READS'],
                num_sweeps=CONFIG_BENCH['NUM_SWEEPS'],
                repeat=CONFIG_BENCH['REPEAT_TIMES'],
                random_sols=CONFIG_BENCH['RANDOM_SOLUTIONS'],
            )
        )

    df = pd.DataFrame(all_results)
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✅ Cached benchmark results to {cache_file}')
    return df


df = run_benchmark()

if df.empty:
    print('No benchmark results to plot.')
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    grouped = df.groupby('N').mean(numeric_only=True)

    axes[0, 0].plot(grouped.index, grouped['jm_build_s'], marker='o', label='JijModeling')
    axes[0, 0].plot(grouped.index, grouped['custom_build_s'], marker='s', label='Custom')
    axes[0, 0].set_xlabel('N')
    axes[0, 0].set_ylabel('Build time (s)')
    axes[0, 0].set_title('QUBO build time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(grouped.index, grouped['jm_sample_s'], marker='o', label='JijModeling')
    axes[0, 1].plot(grouped.index, grouped['custom_sample_s'], marker='s', label='Custom')
    axes[0, 1].set_xlabel('N')
    axes[0, 1].set_ylabel('Sampling time (s)')
    axes[0, 1].set_title(f'SA sampling time ({CONFIG_BENCH["NUM_READS"]} reads, {CONFIG_BENCH["NUM_SWEEPS"]} sweeps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(grouped.index, grouped['jm_terms'], marker='o', label='JijModeling')
    axes[1, 0].plot(grouped.index, grouped['custom_terms'], marker='s', label='Custom')
    axes[1, 0].set_xlabel('N')
    axes[1, 0].set_ylabel('QUBO terms')
    axes[1, 0].set_title('QUBO size')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].bar(grouped.index.astype(str), grouped['speedup_build'])
    axes[1, 1].set_xlabel('N')
    axes[1, 1].set_ylabel('JM / Custom')
    axes[1, 1].set_title('Average build speedup')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    summary = df.groupby('N').agg({
        'jm_build_s': 'mean',
        'custom_build_s': 'mean',
        'jm_sample_s': 'mean',
        'custom_sample_s': 'mean',
        'jm_terms': 'mean',
        'custom_terms': 'mean',
        'speedup_build': 'mean',
        'max_abs_jm_vs_custom': 'max',
        'max_abs_custom_internal': 'max',
    }).reset_index()
    print('\nBenchmark summary (averaged by N):')
    print(summary.to_string(index=False))

    print('\nMax correctness diagnostics:')
    print(df[['max_abs_jm_vs_custom', 'max_abs_custom_internal']].max().to_string())

print('\nBenchmark complete.')

