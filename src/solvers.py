import numpy as np
import cvxpy as cp
import openjij as oj
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

def solve_miqp_cvxpy(a_i, b_ij, K, N):
    """Exact MIQP solver using CVXPY with SCIP (or Gurobi if licensed)."""
    x = cp.Variable(N, boolean=True)
    # b_ij is symmetric; cvxpy.quad_form uses 0.5 * x^T B x, so we need to multiply by 2?
    # Actually quad_form(x, B) = x^T B x. Since our b_ij is symmetric and we want sum_{i<j} b_ij x_i x_j,
    # the full matrix B_full should have zeros on diagonal and b_ij on off-diagonals, but quad_form sums all pairs including i>j.
    # To avoid double counting, we can define B_full = b_ij (symmetric) and divide by 2? Let's be careful.
    # Easier: build the full symmetric matrix B_full where B_full[i,j] = b_ij/2 for i!=j, diagonal=0.
    # Then x^T B_full x = sum_{i<j} b_ij x_i x_j.
    B_full = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                B_full[i, j] = b_ij[i, j] / 2.0  # because quad_form sums i and j both ways
    objective = cp.Minimize(a_i.T @ x + cp.quad_form(x, B_full))
    constraints = [cp.sum(x) == K]
    prob = cp.Problem(objective, constraints)
    # Use SCIP if available, else fallback to ECOS_BB (but that may not handle integer well)
    try:
        prob.solve(solver=cp.SCIP)
    except:
        prob.solve(solver=cp.ECOS_BB)
    return x.value.astype(int), prob.value

def solve_sa(qubo, num_reads=30, sweeps=1000):
    """Simulated Annealing using OpenJij."""
    sampler = oj.SASampler(num_reads=num_reads, sweeps=sweeps)
    response = sampler.sample_qubo(qubo)
    best_state = response.record[0][0]  # first solution
    energy = response.record[0][1]
    return best_state, energy

def solve_sqa(qubo, num_reads=30, sweeps=1000, trotter=32):
    """Simulated Quantum Annealing using OpenJij."""
    sampler = oj.SQASampler(num_reads=num_reads, sweeps=sweeps, trotter=trotter)
    response = sampler.sample_qubo(qubo)
    best_state = response.record[0][0]
    energy = response.record[0][1]
    return best_state, energy

def solve_qaoa(qubo, N, p=1, shots=1024, max_iter=100):
    """
    Solve QUBO using QAOA (classical simulation via Qiskit).
    Returns best state and energy.
    """
    # Convert qubo dict to QuadraticProgram
    qp = QuadraticProgram()
    qp.binary_var_list(range(N), name='x')
    # Linear terms
    for (i,j), coeff in qubo.items():
        if i == j:
            qp.objective.linear[i] += coeff
        else:
            qp.objective.quadratic[i, j] += coeff
    # Convert to QUBO (penalty already included, but Qiskit expects minimization)
    qubo_converter = QuadraticProgramToQubo()
    qubo_problem = qubo_converter.convert(qp)
    
    # Use QAOA with COBYLA
    backend = AerSimulator()
    optimizer = COBYLA(maxiter=max_iter)
    qaoa = QAOA(optimizer=optimizer, reps=p, quantum_instance=backend)
    eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = eigen_optimizer.solve(qubo_problem)
    # Extract solution
    x_sol = np.array([result.x[i] for i in range(N)])
    energy = result.fval
    return x_sol, energy