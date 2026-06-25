import jijmodeling as jm
from jijmodeling_transpiler.core import compile_model
from jijmodeling_transpiler.core.pubo import transpile_to_pubo
import numpy as np

def build_water_quality_problem(N, K_target, a_i, b_ij, lambda_penalty):
    """
    Returns a JijModeling Problem object with the QUBO:
      E = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j + lambda*(sum x_i - K)^2
    """
    prob = jm.Problem('WaterQualityPlacement')
    x = jm.BinaryVar('x', shape=(N,))
    i = jm.Element('i', belong_to=(0, N))
    j = jm.Element('j', belong_to=(0, N))
    
    a = jm.Placeholder('a', shape=(N,))
    b = jm.Placeholder('b', shape=(N, N))
    lam = jm.Placeholder('lambda')
    K = jm.Placeholder('K')
    
    linear = jm.sum(i, a[i] * x[i])
    quad = 0.5 * jm.sum([i, j], b[i][j] * x[i] * x[j])
    penalty = lam * (jm.sum(i, x[i]) - K) ** 2
    
    prob += linear + quad + penalty
    return prob

def transpile_qubo(prob, instance_data):
    """Compile model and convert to QUBO dict for OpenJij."""
    compiled = compile_model(prob, instance_data)
    pubo_builder = transpile_to_pubo(compiled)
    qubo, consts = pubo_builder.get_qubo_dict()
    return qubo, consts

def qubo_to_openjij_format(qubo_dict, N):
    """
    Convert JijModeling QUBO dict to OpenJij format.
    - Diagonal terms (i,i) become linear coefficients
    - Off-diagonal terms (i,j) with i<j remain as quadratic
    """
    linear = np.zeros(N)
    quadratic = {}
    
    for (i, j), coeff in qubo_dict.items():
        if i == j:
            linear[i] += coeff
        elif i < j:
            quadratic[(i, j)] = coeff
        else:
            # If i > j, swap to i < j and add
            quadratic[(j, i)] = quadratic.get((j, i), 0) + coeff
    
    return linear, quadratic