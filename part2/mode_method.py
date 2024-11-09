import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse.linalg import inv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from .Tools_part2 import extractDOF


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
        DOF = extractDOF(idx, geom_data["nodes_clamped"])
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

def mNorm(eta, mu, n_modes):
    return np.sum(np.power(eta[:n_modes], 2) * mu[:n_modes, np.newaxis], axis=0)
    

def convergence(eta, modes, frequencies, K, phi, F, t_span, n_modes, z_dir, save, github):
    q_full, q_acc_full = [], []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, span_mode in enumerate(range(n_modes)):
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
    
    if github:
        for ax in [ax1, ax2]:
            ax.set_facecolor('none')
            ax.tick_params(axis='x', colors='cyan')
            ax.tick_params(axis='y', colors='cyan')
            ax.xaxis.label.set_color('cyan')
            ax.yaxis.label.set_color('cyan')
            ax.title.set_color('cyan')
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_color("cyan")
        fig.patch.set_alpha(0)

    plt.tight_layout()
    
    if save:
        if github:
            plt.savefig('part2/Pictures/convergence.png', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('part2/Pictures/convergence.pdf')
    
    plt.show()
    
    return q_full, q_acc_full

