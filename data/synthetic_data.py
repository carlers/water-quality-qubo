import numpy as np
from scipy.spatial import distance_matrix

def generate_hexagonal_lattice(d_min, domain_size):
    """Generate pointy-top hexagonal lattice sites."""
    s = d_min / np.sqrt(3)          # side length
    rows = int(domain_size / (np.sqrt(3) * s)) + 2
    cols = int(domain_size / (1.5 * s)) + 2
    sites = []
    for r in range(rows):
        for c in range(cols):
            x = c * 1.5 * s
            y = r * np.sqrt(3) * s
            if c % 2 == 1:
                y += np.sqrt(3) * s / 2
            if x <= domain_size and y <= domain_size:
                sites.append([x, y])
    return np.array(sites)

def generate_utility_scores(sites, weights):
    """
    Create 6 synthetic criteria:
      - Pollution load (Gaussian peaks)
      - Ecological sensitivity (Gaussian reserve)
      - Hydrodynamic variability (near river mouth)
      - Accessibility (distance to origin)
      - Data scarcity (north-east)
      - Socio-economic exposure (central area)
    Returns normalized utility U_i = sum(weights * normalized criteria)
    """
    N = len(sites)
    # Simulate criteria
    pollution = np.exp(-((sites[:,0]-20)**2 + (sites[:,1]-25)**2) / 50)
    ecology = np.exp(-((sites[:,0]-30)**2 + (sites[:,1]-20)**2) / 30)
    hydro = np.exp(-((sites[:,0]-5)**2 + (sites[:,1]-10)**2) / 20)
    access = 1 / (1 + np.linalg.norm(sites, axis=1) / 30)
    data_scarcity = (sites[:,0] + sites[:,1]) / 100
    socio = np.exp(-((sites[:,0]-25)**2 + (sites[:,1]-25)**2) / 100)

    criteria = np.array([pollution, ecology, hydro, access, data_scarcity, socio]).T
    # Min-max normalize
    criteria = (criteria - criteria.min(axis=0)) / (criteria.max(axis=0) - criteria.min(axis=0) + 1e-8)
    U = criteria @ weights
    return U

def build_pairwise_terms(sites, config):
    """Compute symmetric QUBO coupling b_ij = beta*R - gamma*C + delta*(W_ij+W_ji)."""
    N = len(sites)
    d_mat = distance_matrix(sites, sites)
    # Redundancy
    R = np.maximum(0, 1 - d_mat / config['L_c'])
    # Communication reward
    C = np.maximum(0, 1 - d_mat / config['R_comm'])
    # Wake interference (directional)
    W = np.zeros_like(d_mat)
    v = config['v']
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d_vec = sites[j] - sites[i]
            d_norm = np.linalg.norm(d_vec)
            if d_norm == 0:
                continue
            cos_theta = np.dot(v, d_vec) / (np.linalg.norm(v) * d_norm + 1e-8)
            if cos_theta > np.cos(np.radians(45)):
                W[i, j] = np.exp(-d_norm / config['L_w']) * cos_theta
    b_ij = config['beta']*R - config['gamma']*C + config['delta']*(W + W.T)
    np.fill_diagonal(b_ij, 0)
    return b_ij

def select_random_sites(sites, N_candidates):
    """Select a subset of N_candidates sites randomly (or first N)."""
    if len(sites) >= N_candidates:
        # For reproducibility, we can sort by x and y then pick evenly spaced.
        # But for simplicity, pick first N after shuffling.
        np.random.seed(42)
        idx = np.random.choice(len(sites), N_candidates, replace=False)
        return sites[idx]
    else:
        return sites