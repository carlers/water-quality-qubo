"""
Standalone validation of the benchmark cell's custom QUBO builder math.

Runs WITHOUT jijmodeling/openjij:
  - builds a small synthetic instance (free variables N, budget K)
  - brute-forces the penalised objective over all 2^N assignments
  - compares against qubo_energy(custom_qubo, custom_offset, x)
  - checks upper-triangular convention of the sampler dict
"""
import itertools
import sys
import numpy as np

sys.path.insert(0, "/tmp")
from bench_cell import canonical_qubo_terms, canonical_to_sampler_qubo, qubo_energy


def build_synthetic_instance(n=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 0.0, n)  # negative = higher utility is better
    # random sparse Q-edges (upper triangular only)
    q_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.25:
                q_edges.append((i, j, float(rng.uniform(-0.5, 0.5))))
    # random symmetric neighbor graph (free-free only)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.3:
                neighbors[i].append(j)
                neighbors[j].append(i)
    for i in range(n):
        neighbors[i] = sorted(set(neighbors[i]))
    # random 0/1 flag: node i has a fixed/existing neighbour
    fixed_neighbors = rng.integers(0, 2, size=n).astype(float)
    instance = {
        "N": n,
        "K": k,
        "a": a,
        "Q_edges": q_edges,
        "neighbors": neighbors,
        "fixed_neighbors": fixed_neighbors,
    }
    return instance


def brute_force_energy(x, instance, lb, lc):
    """Direct evaluation of the penalised objective (the ground truth)."""
    n = int(instance["N"])
    k = int(instance["K"])
    a = np.asarray(instance["a"], dtype=float)
    q_edges = instance["Q_edges"]
    neighbors = instance["neighbors"]
    fixed = np.asarray(instance["fixed_neighbors"], dtype=float)
    x = np.asarray(x, dtype=float)

    energy = float(np.dot(a, x))
    for i, j, val in q_edges:
        energy += float(val) * x[i] * x[j]

    # budget penalty: (sum x_i - K)^2
    if lb != 0:
        energy += lb * (np.sum(x) - k) ** 2

    # connectivity penalty: sum_i (x_i - b_i - sum_j N_ij x_j)^2
    if lc != 0:
        for i in range(n):
            s = fixed[i]
            for j in neighbors[i]:
                s += x[j]
            energy += lc * (x[i] - s) ** 2
    return float(energy)


def main():
    for seed in range(5):
        inst = build_synthetic_instance(n=8, k=3, seed=seed)
        for lb, lc in [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0), (1.0, 0.0), (0.0, 1.0)]:
            linear_dict, quad_dict, offset = canonical_qubo_terms(inst, lb, lc)
            qubo = canonical_to_sampler_qubo(linear_dict, quad_dict)

            # 1) upper-triangular convention check
            for (i, j) in qubo:
                assert i <= j, f"lower-tri key found: {(i, j)}"
            diag_keys = [(i, i) for i in range(8) if (i, i) in qubo]

            # 2) energy equivalence over all 2^N assignments
            for bits in itertools.product([0, 1], repeat=8):
                x = np.array(bits, dtype=int)
                e_brute = brute_force_energy(x, inst, lb, lc)
                e_qubo = qubo_energy(qubo, offset, x)
                assert abs(e_brute - e_qubo) < 1e-9, (seed, lb, lc, bits, e_brute, e_qubo)
        print(f"seed={seed}: OK")

    print("ALL SYNTHETIC CHECKS PASSED")


if __name__ == "__main__":
    main()
