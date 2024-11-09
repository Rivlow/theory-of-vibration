import numpy as np
from scipy.optimize import fsolve


# Utility functions
def normalize_eigenvectors(eigen_vectors, M):
    """Normalize eigenvectors."""
    norms = np.linalg.norm(eigen_vectors, axis=0)
    return eigen_vectors / norms[np.newaxis, :]


# Damping-related functions
def dampingCoef(f1, f2, mode1_ratio, mode2_ratio ):
    """Compute damping coefficients."""
    def equations(xy):
        x, y = xy
        return (0.5 * (f1 * x + y / f1) - mode1_ratio, 0.5 * (f2 * x + y / f2) - mode2_ratio)
    
    return fsolve(equations, (1, 1))

def dampingMatrix(K, M, f1, f2, mode1_ratio, mode2_ratio):
    """Construct damping matrix."""
    a, b = dampingCoef(f1, f2, mode1_ratio, mode2_ratio)
    return a, b, a * K + b * M

def dampingRatios(a, b, eigen_vals):
    """Compute damping ratios."""
    return 0.5 * (a * eigen_vals + b / eigen_vals)