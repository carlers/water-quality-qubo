import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
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

def solve_miqp_gurobi(a_i, b_ij, K, N):
    """Exact MIQP solver using Gurobi directly."""

    # Create a Gurobi model
    model = gp.Model("MIQP")

    # Add binary variables (x_i in {0, 1})
    x = model.addVars(N, vtype=GRB.BINARY, name="x")

    # Build the objective: a_i^T x + sum_{i<j} b_ij x_i x_j
    objective = gp.QuadExpr()

    # Linear term: a_i^T x
    for i in range(N):
        objective += a_i[i] * x[i]

    # Quadratic term: sum_{i<j} b_ij x_i x_j
    for i in range(N):
        for j in range(i + 1, N):
            objective += b_ij[i, j] * x[i] * x[j]

    model.setObjective(objective, GRB.MINIMIZE)

    # Add constraint: sum(x) == K
    model.addConstr(gp.quicksum(x[i] for i in range(N)) == K, "cardinality_constraint")

    # Solve the model
    model.optimize()

    # Extract the solution
    x_solution = np.array([x[i].X for i in range(N)], dtype=int)
    optimal_value = model.objVal

    return x_solution, optimal_value

def solve_sa(qubo, N, num_reads=30, sweeps=1000):
    """Simulated Annealing."""
    sampler = oj.SASampler()
    response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=sweeps)
    best = response.first
    # best.sample is dict {var: value}, variables are 0..N-1
    x = np.array([best.sample[i] for i in range(N)])
    return x.astype(int), best.energy

def solve_sqa(qubo, N, num_reads=30, sweeps=1000, trotter=32):
    """Simulated Quantum Annealing."""
    sampler = oj.SQASampler()
    response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=sweeps, trotter=trotter)
    best = response.first
    x = np.array([best.sample[i] for i in range(N)])
    return x.astype(int), best.energy

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