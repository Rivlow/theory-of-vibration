import numpy as np
from scipy.sparse.linalg import inv
import scipy as sp
pi = np.pi

def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF


def computeForce(params, nodes_clamped, eigen_vectors, nb_modes, t_span):

    h = params['h']
    m_tot = params['m_tot']
    delta_t = params['delta_t']
    freq = params['freq']
    g = params['g']

    v = np.sqrt(2 * g * h)
    momentum = m_tot * v
    A = 0.8 * (momentum / delta_t)
    w = 2 * pi * freq
    
    # Extract DOFs
    DOF_1 = extractDOF(23, nodes_clamped)
    DOF_2 = extractDOF(24, nodes_clamped)
    DOF_force = [DOF_1+2, DOF_2+2]
    
    # Compute F for all times and modes
    sin_wt = np.sin(w * t_span)
    F = np.zeros((len(eigen_vectors), len(t_span)))
    F[DOF_force] = -(A/2) * sin_wt
 
    return F


def computeH(w_d, w_r, eps_r, t_span):
    """computation of h for all modes."""
    t_matrix = t_span[np.newaxis, :]
    return (1 / w_d[:, np.newaxis]) * np.exp(-eps_r[:, np.newaxis] * w_r[:, np.newaxis] * t_matrix) * np.sin(w_d[:, np.newaxis] * t_matrix)

# Main computation functions
def etaPhiMu(w_all, x, eps, M, F, t_span, nb_modes):
    """Compute eta and phi using vectorized operations and convolution."""
    
    w_d = w_all[:nb_modes] * np.sqrt(1 - eps[:nb_modes]**2)
    mu = np.sum(x[:, :nb_modes] * (M @ x[:, :nb_modes]), axis=0)
    
    F_proj = np.sum(x[:, :nb_modes].T[:, :, np.newaxis] * F, axis=1) / mu[:, np.newaxis]
    
    h = computeH(w_d, w_all[:nb_modes], eps[:nb_modes], t_span)
    phi = F_proj
    
    dt = t_span[1] - t_span[0]
    eta = np.array([sp.signal.convolve(F_proj[r], h[r], mode='full')[:len(t_span)] * dt 
                    for r in range(nb_modes)])
    
    return eta, phi, mu


def modeDisplacementMethod(eta, x, t_span, nb_modes):
    """Compute mode displacement method."""
    return np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes])

def modeAccelerationMethod(eta, w_all, x, t_span, K, phi, F, nb_modes):
    """Compute mode acceleration method."""
   
    q = np.zeros((x.shape[0], len(t_span)))
    
    K_inv = inv(K.tocsc())
    w_sq_inv = 1 / (w_all[:nb_modes]**2)

    KF = K_inv @ F
        
    q = np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes]) + KF
    q -= np.einsum('rm,r,dr->dm', phi[:nb_modes], w_sq_inv, x[:, :nb_modes])
 
    return q



def mNorm(eta, mu, nb_modes):
    return np.sum(np.power(eta[:nb_modes], 2) * mu[:nb_modes, np.newaxis], axis=0)