import numpy as np
import cvxpy as cp
import openjij as oj

# No top-level Qiskit imports to avoid errors.
# All Qiskit-related imports are inside solve_qaoa().

def solve_exact_qubo(qubo, N, K):
    """
    Exact QUBO solver using dimod.ExactSolver.
    Returns best state (as list of ints) and minimum energy.
    """
    import dimod
    # qubo is a dict {(i,j): coeff} with i<=j (diagonal for linear terms)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampler = dimod.ExactSolver()
    response = sampler.sample(bqm)
    best = response.first
    # best.sample is a dict {var: value}
    x = np.array([best.sample[i] for i in range(N)])
    return x.astype(int), best.energy

def solve_miqp_cvxpy(a_i, b_ij, K, N):
    """Exact MIQP solver using CVXPY."""
    x = cp.Variable(N, boolean=True)
    # Build symmetric matrix B_full such that x^T B_full x = sum_{i<j} b_ij x_i x_j
    B_full = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                B_full[i, j] = b_ij[i, j] / 2.0
    objective = cp.Minimize(a_i.T @ x + cp.quad_form(x, B_full))
    constraints = [cp.sum(x) == K]
    prob = cp.Problem(objective, constraints)
    
    # Use Gurobi as the exact baseline solver
    try:
        prob.solve(solver=cp.GUROBI, enforce_dcp=False)
    except cp.SolverError:
        # Fallback to letting CVXPY find any available mixed-integer solver
        prob.solve()
        
    return x.value.astype(int), prob.value

def solve_sa(qubo, num_reads=30, sweeps=1000):
    """Simulated Annealing using OpenJij."""
    sampler = oj.SASampler()   # no arguments
    response = sampler.sample_qubo(qubo)
    # The best (lowest energy) solution is the first record
    best_state = response.record[0][0]
    energy = response.record[0][1]
    return best_state, energy

def solve_sqa(qubo, num_reads=30, sweeps=1000, trotter=32):
    """Simulated Quantum Annealing using OpenJij."""
    sampler = oj.SQASampler()   # no arguments
    response = sampler.sample_qubo(qubo)
    best_state = response.record[0][0]
    energy = response.record[0][1]
    return best_state, energy

def solve_qaoa(qubo, N, p=1, shots=1024, max_iter=100):
    """
    QAOA solver using Qiskit.
    This function imports Qiskit packages only when called.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.providers.basic_provider import BasicProvider
        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.converters import QuadraticProgramToQubo
    except ImportError as e:
        raise ImportError("Qiskit or qiskit-optimization not installed. Please run: !pip install qiskit qiskit-optimization") from e

    # Build QuadraticProgram from QUBO dict
    qp = QuadraticProgram()
    qp.binary_var_list(range(N), name='x')
    for (i, j), coeff in qubo.items():
        if i == j:
            qp.objective.linear[i] += coeff
        else:
            qp.objective.quadratic[i, j] += coeff
    # Convert to QUBO (though it's already a QUBO)
    qubo_converter = QuadraticProgramToQubo()
    qubo_problem = qubo_converter.convert(qp)

    # Use BasicProvider (no Aer needed) for simulation
    backend = BasicProvider().get_backend('statevector_simulator')
    optimizer = COBYLA(maxiter=max_iter)
    qaoa = QAOA(optimizer=optimizer, reps=p, quantum_instance=backend)
    eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = eigen_optimizer.solve(qubo_problem)

    x_sol = np.array([result.x[i] for i in range(N)])
    energy = result.fval
    return x_sol, energy