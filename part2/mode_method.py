import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse.linalg import inv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import part2.Tools_part2 as Tools

SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def isLatex(latex):
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')     


def computeForce(sim_data, 
                 geom_data, xyz,
                 modes, 
                 t_span):
    """ Compute induced force by the supporters on excitation nodes """

    h = sim_data['h']
    m_tot = sim_data['m_tot']
    delta_t = sim_data['dt']
    freq = sim_data['freq']
    g = sim_data['g']

    v = np.sqrt(2 * g * h)
    momentum = m_tot * v
    A = 0.8 * (momentum / delta_t)
    w = 2 * np.pi * freq

    sin_wt = np.sin(w * t_span)
    F = np.zeros((len(modes), len(t_span)))
    Ampl = -A/len(sim_data["nodes_force"])    

    # Distribute force F on each nodes_force
    for idx in sim_data["nodes_force"]:
        DOF = Tools.extractDOF(idx, geom_data["nodes_clamped"])
        F[DOF + xyz, :] = Ampl * sin_wt[:]

    return F

def computeH(w_d, w_r, eps_r, t_span):
    """computation of impulse rsponse h for all modes."""
    t_matrix = t_span[np.newaxis, :]
    return (1 / w_d[:, np.newaxis]) * np.exp(-eps_r[:, np.newaxis] * w_r[:, np.newaxis] * t_matrix) * np.sin(w_d[:, np.newaxis] * t_matrix)

# Main computation functions
def etaPhiMu(w_all, x, eps, M, F, t_span, n_modes):
    """Compute eta and phi using vectorized operations and convolution."""
    
    w_d = w_all[:n_modes] * np.sqrt(1 - eps[:n_modes]**2)
    mu = np.sum(x[:, :n_modes]* (M @ x[:, :n_modes]), axis=0)

    F_proj = np.sum(x[:, :n_modes].T[:, :, np.newaxis] * F, axis=1) / mu[:, np.newaxis]
    
    h = computeH(w_d, w_all[:n_modes], eps[:n_modes], t_span)
    phi = F_proj
    
    dt = t_span[1] - t_span[0]
    eta = np.array([sp.signal.convolve(F_proj[r], h[r], mode='full')[:len(t_span)] * dt 
                    for r in range(n_modes)])
    
    return eta, phi, mu


def modeDisplacementMethod(eta, x, n_modes):
    """Compute mode displacement method."""
    return np.einsum('rm,dr->dm', eta[:n_modes], x[:, :n_modes])


def modeAccelerationMethod(t_span, eta, w_all, x, K, phi, F, n_modes):
    """Compute mode acceleration method."""
   
    q_acc = np.zeros((x.shape[0], len(t_span)))
    
    K_inv = inv(K.tocsc())
    w_sq_inv = 1 / (w_all[:n_modes]**2)

    KF = K_inv @ F
        
    q_acc = np.einsum('rm,dr->dm', eta[:n_modes], x[:, :n_modes]) + KF
    q_acc -= np.einsum('rm,r,dr->dm', phi[:n_modes], w_sq_inv, x[:, :n_modes])

    return q_acc


