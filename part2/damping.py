import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy.optimize import fsolve
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.signal import hilbert


pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp


# Utility functions
def normalize_eigenvectors(eigen_vectors, M):
    """Normalize eigenvectors."""
    norms = np.linalg.norm(eigen_vectors, axis=0)
    return eigen_vectors / norms[np.newaxis, :]

def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF

def extract_envelope(q, t_span):
    """
    Extraire l'enveloppe du signal q en utilisant la transformée de Hilbert.
    """
    analytic_signal = hilbert(q)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def plot_signal_with_envelope(t_span, q, envelope):
    plt.figure(figsize=(12, 6))
    plt.plot(t_span, q, label='Signal')
    plt.plot(t_span, envelope, 'r', label='Enveloppe')
    plt.plot(t_span, -envelope, 'r')
    plt.xlabel('Temps [s]')
    plt.ylabel('Amplitude')
    plt.title('Signal avec son enveloppe')
    plt.legend()
    plt.grid(True)
    plt.show()

# Force and response functions
def forceParams(m_tot, h, g, dt, freq, nodes_clamped, mode):
    """Calculate force parameters."""
    v = np.sqrt(2 * g * h)
    momentum = m_tot * v
    A = 0.8 * (momentum / dt)
    w = 2 * pi * freq
    DOF_1 = extractDOF(23, nodes_clamped)
    DOF_2 = extractDOF(24, nodes_clamped)
    return A/2, w, [DOF_1+2, DOF_2+2]

def computeF(amplitude, w, x, DOF_force, t_span):
    """computation of F for all times and modes."""
    sin_wt = np.sin(w * t_span)
    F = np.zeros((len(x), len(t_span)))
    F[DOF_force] = -amplitude * sin_wt
    return F

def computeH(w_d, w_r, eps_r, t_span):
    """computation of h for all modes."""
    t_matrix = t_span[np.newaxis, :]
    return (1 / w_d[:, np.newaxis]) * np.exp(-eps_r[:, np.newaxis] * w_r[:, np.newaxis] * t_matrix) * np.sin(w_d[:, np.newaxis] * t_matrix)

# Main computation functions
def etaPhi(w_all, x, eps, M, nodes_clamped, params, t_span, nb_modes):
    """Compute eta and phi using vectorized operations and convolution."""
    m_tot, h_, g, dt, freq = params["m_tot"], params["h"], params["g"], params["dt"], params["freq"]
    
    w_d = w_all[:nb_modes] * np.sqrt(1 - eps[:nb_modes]**2)
    mu_r = np.sum(x[:, :nb_modes] * (M @ x[:, :nb_modes]), axis=0)
    
    amplitude, w, DOF_force = forceParams(m_tot, h_, g, dt, freq, nodes_clamped, x[:, 0])
    
    F = computeF(amplitude, w, x[:, :nb_modes], DOF_force, t_span)
    F_proj = np.sum(x[:, :nb_modes].T[:, :, np.newaxis] * F, axis=1) / mu_r[:, np.newaxis]
    
    h = computeH(w_d, w_all[:nb_modes], eps[:nb_modes], t_span)
    phi = F_proj

    plt.figure()
    plt.plot(t_span, F_proj[0,:], c="r")
    plt.figure()
    plt.plot(t_span, h[0,:], c="b")
    plt.plot()

    
    dt = t_span[1] - t_span[0]
    eta = np.array([sp.signal.convolve(F_proj[r], h[r], mode='full')[:len(t_span)] * dt 
                    for r in range(nb_modes)])
    
    return eta, phi

def modeDisplacementMethod(eta, x, t_span, nb_modes):
    """Compute mode displacement method."""
    return np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes])

def modeAccelerationMethod(eta, w_all, x, t_span, K, phi, params, nodes_clamped, nb_modes):
    """Compute mode acceleration method."""
    m_tot, h, g, dt, freq = params["m_tot"], params["h"], params["g"], params["dt"], params["freq"]
    
    amplitude, w, DOF_force = forceParams(m_tot, h, g, dt, freq, nodes_clamped, x[:, 0])
    q = np.zeros((x.shape[0], len(t_span)))
    
    K_inv = inv(K.tocsc())
    w_sq_inv = 1 / (w_all[:nb_modes]**2)

    F = computeF(amplitude, w, x[:, :nb_modes], DOF_force, t_span)
    KF = K_inv @ F
        
    q = np.einsum('rm,dr->dm', eta[:nb_modes], x[:, :nb_modes]) + KF
    q -= np.einsum('rm,r,dr->dm', phi[:nb_modes], w_sq_inv, x[:, :nb_modes])
 
    return q

def newmark_integration(M, C, K, x0, v0, t_span, m_tot, h, g, freq, nodes_clamped, mode, gamma=0.5, beta=0.25):
    """
    Perform time integration using the Newmark algorithm with custom force parameters.
    
    Parameters:
    - M, C, K: Mass, damping, and stiffness matrices
    - x0, v0: Initial displacement and velocity vectors
    - t_span: Array of time points for integration
    - m_tot: Total mass for force calculation
    - h: Height for force calculation
    - g: Gravity constant
    - freq: Frequency for force calculation
    - nodes_clamped: Clamped nodes information
    - mode: Mode shape for force application
    - gamma, beta: Newmark method parameters (default: average acceleration method)
    
    Returns:
    - x, v, a: Displacement, velocity, and acceleration time histories
    """
    n = len(x0)
    nt = len(t_span)
    dt = t_span[1] - t_span[0]  # Assuming uniform time step
    
    # Calculate force parameters
    A, w, DOF = forceParams(m_tot, h, g, dt, freq, nodes_clamped, mode)
    
    # Initialize arrays
    x = np.zeros((n, nt))
    v = np.zeros((n, nt))
    a = np.zeros((n, nt))
    
    # Set initial conditions
    x[:, 0] = x0
    v[:, 0] = v0
    F0 = computeF(A, w, mode, DOF, 0)
    a[:, 0] = spsolve(M, F0 - C @ v0 - K @ x0)
    
    # Compute constant matrices
    A_mat = (M + gamma * dt * C + beta * dt**2 * K).tocsc()
    A_inv = sp.sparse.linalg.inv(A_mat)
    
    for i in range(1, nt):
        # Predict
        x_pred = x[:, i-1] + dt * v[:, i-1] + 0.5 * dt**2 * ((1 - 2*beta) * a[:, i-1])
        v_pred = v[:, i-1] + (1 - gamma) * dt * a[:, i-1]
        
        # Compute force at current time
        F_current = computeF(A, w, mode, DOF, t_span[i])
        
        # Compute effective force
        F_eff = F_current - C @ v_pred - K @ x_pred
        
        # Solve for acceleration
        da = A_inv @ F_eff
        
        # Correct
        a[:, i] = a[:, i-1] + da
        v[:, i] = v_pred + gamma * dt * da
        x[:, i] = x_pred + beta * dt**2 * da
    
    return x, v, a


# Damping-related functions
def dampingCoef(f1, f2):
    """Compute damping coefficients."""
    def equations(xy):
        x, y = xy
        return (0.5 * (f1 * x + y / f1) - 0.005, 0.5 * (f2 * x + y / f2) - 0.005)
    
    return fsolve(equations, (1, 1))

def dampingMatrix(K, M, f1, f2):
    """Construct damping matrix."""
    a, b = dampingCoef(f1, f2)
    return a, b, a * K + b * M

def dampingRatios(a, b, eigen_vals):
    """Compute damping ratios."""
    return 0.5 * (a * eigen_vals + b / eigen_vals)