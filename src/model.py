import jijmodeling as jm
# Import the correct v1 transpilation components from their internal modules
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
    
    # Linear objective term
    linear = jm.sum(i, a[i] * x[i])
    
    # Quadratic objective term:
    # Since b_ij is symmetric and diagonal is 0, summing over all [i, j] 
    # double-counts every pair. Multiplying by 0.5 yields exactly the sum over i < j.
    quad = 0.5 * jm.sum([i, j], b[i][j] * x[i] * x[j])
    
    # Penalty term for choosing exactly K stations
    penalty = lam * (jm.sum(i, x[i]) - K) ** 2
    
    prob += linear + quad + penalty
    return prob

def transpile_qubo(prob, instance_data):
    """Compile model and convert to QUBO dict for OpenJij."""
    # 1. Compile the abstract mathematical structure with the specific data instance
    compiled = compile_model(prob, instance_data)
    
    # 2. Convert the compiled mathematical representation into a PUBO/QUBO builder object
    pubo_builder = transpile_to_pubo(compiled)
    
    # 3. Extract the standard python dictionary structure {(i, j): coefficient} and the offset constant
    qubo, consts = pubo_builder.get_qubo_dict()
    
    return qubo, consts