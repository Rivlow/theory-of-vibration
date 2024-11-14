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
    
def convergenceMax(eta, modes, frequencies, K, phi, F, t_span, n_modes, z_dir, save, latex):

    q_full, q_acc_full = [], []
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    isLatex(latex)
    
    for i, span_mode in enumerate(range(1, n_modes)):
        q = modeDisplacementMethod(eta, modes, span_mode)[z_dir,:]
        q_acc = modeAccelerationMethod(t_span, eta, 2*np.pi*frequencies, modes, K, phi, F, span_mode)[z_dir,:]
        q_full.append(np.max(q))
        q_acc_full.append(np.max(q_acc))
    
  
    q_ref = q_full[-1]  
    q_acc_ref = q_acc_full[-1] 
    
    err_cumul_q = [np.abs(q - q_ref)/np.abs(q_ref) * 100 for q in q_full]
    err_cumul_acc = [np.abs(q - q_acc_ref)/np.abs(q_acc_ref) * 100 for q in q_acc_full]

   
    ax.scatter(range(1, len(err_cumul_q)-1), err_cumul_q[:-2], label="displacement method")
    ax.scatter(range(1, len(err_cumul_acc)-1), err_cumul_acc[:-2], label="acceleration method")
    ax.plot(range(1, len(err_cumul_q)-1), err_cumul_q[:-2], ls="--")
    ax.plot(range(1, len(err_cumul_acc)-1), err_cumul_acc[:-2], ls="--")
    
    
    #ax.set_yscale('log') 
    ax.set_xticks(range(0, n_modes))
    ax.set_xlabel('Number of modes n[-]')
    ax.set_ylabel('Relative error [%]')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend()
    

    
    plt.tight_layout()
    
    if save:
        plt.savefig('part2/Pictures/convergence.pdf')
    
    plt.show()

def convergenceShape(eta, modes, frequencies, K, phi, F, t_span, n_modes, z_dir, save, latex):
    q_full, q_acc_full = [], []
    
    fig, ax  = plt.subplots(1, 1, figsize=(15, 6))
    isLatex(latex)
    nb_modes = [1, 5, 8]
    
    for i, span_mode in enumerate(range(1, n_modes)):

        q = modeDisplacementMethod(eta, modes, span_mode)[z_dir,:]
        q_acc = modeAccelerationMethod(t_span, eta, 2*np.pi*frequencies, modes, K, phi, F, span_mode)[z_dir,:]
        q_full.append(np.max(q))
        q_acc_full.append(np.max(q_acc))

   

    var_q = [np.abs(q_full[i+1]-q_full[i])/q_full[i] for i in range(len(q_full)-1)]
    var_q_acc = [np.abs(q_acc_full[i+1]-q_acc_full[i])/q_acc_full[i] for i in range(len(q_acc_full)-1)]
        
    ax.scatter(range(1, len(q_full)), var_q, label="displacement method")
    ax.scatter(range(1, len(q_acc_full)), var_q_acc, label="acceleration method")
    ax.plot(range(1, len(q_full)), var_q, ls="--")
    ax.plot(range(1, len(q_acc_full)), var_q_acc, ls="--")
    ax.set_xticks(range(1, len(q_full)))
    ax.set_xlabel('Number of modes n[-]')
    ax.set_ylabel(r'$\frac{max(q_{n+1})-max(q_n)}{max(q_n)}$[%]')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    if save:
        plt.savefig('part2/Pictures/convergence.pdf')
    
    plt.show()
    

