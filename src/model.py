import jijmodeling as jm
import jijmodeling_transpiler as jmt
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
    quad = jm.sum([i, j], b[i][j] * x[i] * x[j], i < j)
    penalty = lam * (jm.sum(i, x[i]) - K) ** 2
    
    prob += linear + quad + penalty
    return prob

def transpile_qubo(prob, instance_data):
    """Compile model and convert to QUBO dict for OpenJij."""
    compiled = jmt.compile_model(prob, instance_data)
    qubo, consts = jmt.to_qubo(compiled)
    # qubo is a dict {(i,j): coeff} for i<j
    return qubo, consts