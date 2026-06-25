import numpy as np

CONFIG = {
    # Domain and lattice
    'domain_size': 50.0,       # km
    'L_c': 5.0,                # spatial correlation length (km)
    'L_w': 1.0,                # wake persistence length (km)
    'R_comm': 3.0,             # wireless communication range (km)
    'd_nn': 2.5,               # nearest-neighbor distance = L_c/2
    'v': np.array([1.0, 0.0]), # uniform current (eastward)

    # Weights
    'beta': 1.0,               # redundancy penalty weight
    'gamma': 1.0,              # communication reward weight
    'delta': 1.0,              # wake penalty weight

    # AHP weights (baseline)
    'ahp_weights': np.array([0.35, 0.20, 0.12, 0.12, 0.14, 0.07]),

    # Environmental scenarios (for sensitivity)
    'env_weights': {
        'environmentalist': np.array([0.20, 0.40, 0.20, 0.05, 0.10, 0.05]),
        'logistician': np.array([0.25, 0.10, 0.10, 0.35, 0.10, 0.10])
    },

    # Number of trials per solver
    'num_trials': 30,
    # SA/SQA parameters
    'sa_sweeps': 1000,
    'sqa_trotter': 32,
    'sqa_sweeps': 1000,
    'num_reads': 30,            # number of runs per solver call
}