import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, voronoi_plot_2d
from config import CONFIG as config

def compute_sqr(energy_method, energy_exact):
    """
    SQR = energy_method / energy_exact, since both are minimization energies.
    Because E = -U, and U_exact is max utility, energy_exact is most negative.
    Thus SQR = E_method / E_exact (both negative) gives a value ≤ 1.
    """
    return energy_method / energy_exact

def compute_coverage_metrics(sites, selected_indices, domain_size):
    """
    Compute:
      - HCR (Hexagonal Coverage Ratio): fraction of domain covered by disks of radius L_c/2
      - Voronoi STD: standard deviation of Voronoi cell areas
      - NND_mean: mean nearest-neighbor distance among selected sites
    Returns dict.
    """
    selected = sites[selected_indices]
    N = len(selected)
    if N == 0:
        return {'HCR': 0.0, 'voronoi_std': np.nan, 'nnd_mean': np.nan}
    
    # Nearest neighbor distances
    from scipy.spatial import distance_matrix
    d_mat = distance_matrix(selected, selected)
    np.fill_diagonal(d_mat, np.inf)
    nnd = d_mat.min(axis=1)
    nnd_mean = np.mean(nnd)
    
    # Voronoi area (if N >= 4)
    try:
        vor = Voronoi(selected)
        # Compute areas of finite regions
        areas = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                polygon = vor.vertices[region]
                # area
                area = 0.5 * abs(np.sum(np.cross(polygon, np.roll(polygon, 1, axis=0))))
                areas.append(area)
        if len(areas) > 0:
            vor_std = np.std(areas)
        else:
            vor_std = np.nan
    except:
        vor_std = np.nan
    
    # Simple HCR: approximate as fraction of area within L_c/2 of selected sites
    # We'll use a coarse grid approximation
    grid_res = 0.5  # km
    x = np.arange(0, domain_size, grid_res)
    y = np.arange(0, domain_size, grid_res)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    from scipy.spatial import KDTree
    tree = KDTree(selected)
    dists, _ = tree.query(pts)
    covered = (dists <= config['L_c']/2).sum()
    total = len(pts)
    hcr = covered / total
    return {'HCR': hcr, 'voronoi_std': vor_std, 'nnd_mean': nnd_mean}

def plot_layout(sites, selected_indices, fixed_indices, U, title, save_path=None):
    """Plot sites colored by utility, selected in black, fixed in red."""
    plt.figure(figsize=(8,6))
    sc = plt.scatter(sites[:,0], sites[:,1], c=U, cmap='viridis', s=50, edgecolors='k', alpha=0.7)
    plt.colorbar(sc, label='Utility U_i')
    # Selected
    if len(selected_indices) > 0:
        plt.scatter(sites[selected_indices,0], sites[selected_indices,1], 
                    facecolors='none', edgecolors='black', s=120, linewidths=2, label='Selected')
    # Fixed (if any)
    if fixed_indices is not None and len(fixed_indices) > 0:
        plt.scatter(sites[fixed_indices,0], sites[fixed_indices,1],
                    facecolors='red', edgecolors='red', s=80, marker='s', label='Fixed')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()