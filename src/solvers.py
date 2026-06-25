import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import openjij as oj

# No top-level Qiskit imports to avoid errors.
# All Qiskit-related imports are inside solve_qaoa().

def solve_miqp_gurobi(a_i, b_ij, K, N):
    """Exact MIQP solver using Gurobi directly."""

    # Create a Gurobi model
    model = gp.Model("MIQP")

    # Suppress all Gurobi output
    model.setParam('OutputFlag', 0)

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

def solve_sa(qubo_dict, N, linear_coeffs=None, num_reads=30, sweeps=1000):
    """
    Simulated Annealing using OpenJij.
    qubo_dict: JijModeling QUBO dict (may include self-terms)
    linear_coeffs: if provided, these are the linear coefficients from the QUBO
    """
    sampler = oj.SASampler()
    
    # Build complete QUBO dictionary with self-terms for linear coefficients
    qubo_full = {}
    
    # If we have linear coefficients, add them as self-terms (i, i)
    if linear_coeffs is not None:
        for i, coeff in enumerate(linear_coeffs):
            if coeff != 0:  # Only add non-zero terms
                qubo_full[(i, i)] = coeff
    
    # Add quadratic terms
    for (i, j), coeff in qubo_dict.items():
        if i != j:
            # OpenJij expects i < j for quadratic terms
            if i < j:
                qubo_full[(i, j)] = coeff
            else:
                qubo_full[(j, i)] = qubo_full.get((j, i), 0) + coeff
        else:
            # This is already a self-term (linear)
            qubo_full[(i, i)] = qubo_full.get((i, i), 0) + coeff
    
    # Sample with the complete QUBO
    response = sampler.sample_qubo(
        qubo_full,
        num_reads=num_reads,
        num_sweeps=sweeps
    )
    
    if num_reads == 1:
        print(f"SA response type: {type(response)}")
        print(f"SA response.record shape: {response.record.shape}")
        print(f"SA first sample: {response.first}")
        best_state = response.first[0]
        energy = response.first[1]
        x = np.array([best_state[i] for i in range(N)])
        return x.astype(int), energy
    
    best_state = response.record[0][0]
    energy = response.record[0][1]
    return best_state, energy


def solve_sqa(qubo_dict, N, linear_coeffs=None, num_reads=30, sweeps=1000, trotter=32):
    """
    Simulated Quantum Annealing using OpenJij.
    """
    sampler = oj.SQASampler()
    
    # Build complete QUBO dictionary with self-terms for linear coefficients
    qubo_full = {}
    
    # If we have linear coefficients, add them as self-terms (i, i)
    if linear_coeffs is not None:
        for i, coeff in enumerate(linear_coeffs):
            if coeff != 0:  # Only add non-zero terms
                qubo_full[(i, i)] = coeff
    
    # Add quadratic terms
    for (i, j), coeff in qubo_dict.items():
        if i != j:
            # OpenJij expects i < j for quadratic terms
            if i < j:
                qubo_full[(i, j)] = coeff
            else:
                qubo_full[(j, i)] = qubo_full.get((j, i), 0) + coeff
        else:
            # This is already a self-term (linear)
            qubo_full[(i, i)] = qubo_full.get((i, i), 0) + coeff
    
    # Sample with the complete QUBO
    response = sampler.sample_qubo(
        qubo_full,
        num_reads=num_reads,
        num_sweeps=sweeps,
        trotter=trotter
    )
    
    if num_reads == 1:
        print(f"SQA response type: {type(response)}")
        print(f"SQA response.record shape: {response.record.shape}")
        print(f"SQA first sample: {response.first}")
        best_state = response.first[0]
        energy = response.first[1]
        x = np.array([best_state[i] for i in range(N)])
        return x.astype(int), energy
    
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