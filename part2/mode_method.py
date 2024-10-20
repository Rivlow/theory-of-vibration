import numpy as np
from scipy.sparse.linalg import inv
import scipy as sp
import matplotlib.pyplot as plt
pi = np.pi

from Tools_part2 import *


def computeForce(params, 
                 nodes_clamped, nodes_force, xyz,
                 modes, 
                 t_span):
    """ Compute induced force by the supporters on excitation nodes [23, 24]"""
    h = params['h']
    m_tot = params['m_tot']
    delta_t = params['dt']
    freq = params['freq']
    g = params['g']

    v = np.sqrt(2 * g * h)
    momentum = m_tot * v
    A = 0.8 * (momentum / delta_t)
    w = 2 * pi * freq

    sin_wt = np.sin(w * t_span)
    F = np.zeros((len(modes), len(t_span)))
    
    # Distribute force F on each nodes_force
    for idx in nodes_force:
        DOF = extractDOF(idx, nodes_clamped)
        F[DOF + xyz] = -(A/len(nodes_force)) * sin_wt


    return F


def computeH(w_d, w_r, eps_r, t_span):
    """computation of impulse rsponse h for all modes."""
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


def modeDisplacementMethod(eta, x, nb_modes):
    """Compute mode displacement method."""

    q = np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes])

   
    return q



def modeAccelerationMethod(eta, w_all, x, t_span, K, phi, F, nb_modes):
    """Compute mode acceleration method."""
   
    q_acc = np.zeros((x.shape[0], len(t_span)))
    
    K_inv = inv(K.tocsc())
    w_sq_inv = 1 / (w_all[:nb_modes]**2)

    KF = K_inv @ F
        
    q_acc = np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes]) + KF
    q_acc -= np.einsum('rm,r,dr->dm', phi[:nb_modes], w_sq_inv, x[:, :nb_modes])

   
    return q_acc

def mNorm(eta, mu, nb_modes):
    return np.sum(np.power(eta[:nb_modes], 2) * mu[:nb_modes, np.newaxis], axis=0)

def convergence(eta, modes, frequencies, K, phi, F, t_span, nb_modes, z_dir):
    q_full, q_acc_full = [], []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, span_mode in enumerate(range(nb_modes)):
        q_full.append(modeDisplacementMethod(eta, modes, span_mode))
        q_acc_full.append(modeAccelerationMethod(eta, 2*np.pi*frequencies, modes, t_span, K, phi, F, span_mode))
        
        ax1.plot(t_span, q_full[i][z_dir,:], label=f'Mode {span_mode}')
        ax2.plot(t_span, q_acc_full[i][z_dir,:], label=f'Mode {span_mode}')
    
    ax1.set_title('Mode Displacement Method')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Déplacement')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Mode Acceleration Method')
    ax2.set_xlabel('Temps')
    ax2.set_ylabel('Déplacement')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return q_full, q_acc_full

