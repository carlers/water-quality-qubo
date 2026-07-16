# =============================================================================
# src/jij_model.py – JijModeling Problem Definition (Budget + Connectivity)
# =============================================================================

import jijmodeling as jm
from typing import Dict

__all__ = [
    "build_augmented_model",
    "get_penalty_weights",
    "compile_instance",
]


def build_augmented_model() -> jm.Problem:
    """Build the JijModeling Problem with budget + connectivity constraints."""
    problem = jm.Problem("WQM_Augmented", sense=jm.ProblemSense.MINIMIZE)

    N_ph = problem.Length("N")
    K_ph = problem.Length("K")
    a_ph = problem.Float("a", shape=(N_ph,))
    Q_ph = problem.Float("Q", shape=(N_ph, N_ph))
    neigh_ph = problem.Binary("neigh", shape=(N_ph, N_ph))
    x = problem.BinaryVar("x", shape=(N_ph,))

    obj_linear = jm.sum(N_ph, lambda i: a_ph[i] * x[i])
    obj_quad = jm.sum(
        jm.product(N_ph, N_ph).filter(lambda i, j: i < j),
        lambda i, j: Q_ph[i, j] * x[i] * x[j]
    )
    problem += obj_linear + obj_quad

    problem += problem.Constraint("budget", jm.sum(N_ph, lambda i: x[i]) == K_ph)
    problem += problem.Constraint(
        "connectivity",
        lambda i: x[i] <= jm.sum(N_ph, lambda j: neigh_ph[i, j] * x[j]),
        domain=N_ph
    )
    return problem


def get_penalty_weights(instance, lambda_budget: float, lambda_conn: float) -> Dict[int, float]:
    """Map OMMX constraint IDs to penalty weights."""
    penalty_weights = {}
    for c in instance.constraints:
        if c.name == "budget":
            penalty_weights[c.id] = lambda_budget
        elif c.name == "connectivity":
            penalty_weights[c.id] = lambda_conn
    return penalty_weights


def compile_instance(problem: jm.Problem, instance_data: Dict):
    # Keep only keys that are placeholders in the model
    placeholder_names = {p.name for p in problem.placeholders.values()}
    filtered = {k: v for k, v in instance_data.items() if k in placeholder_names}
    return problem.eval(filtered)