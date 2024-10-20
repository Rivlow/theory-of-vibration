import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.signal import hilbert
import matplotlib.pyplot as plt

def newmark_integration(M, C, K, x0, v0, t_span, F, gamma, beta):
    n = len(x0)
    nt = len(t_span)
    h = t_span[1] - t_span[0]

    x = np.zeros((n, nt))
    v = np.zeros((n, nt))
    a = np.zeros((n, nt))

    x[:, 0] = x0
    v[:, 0] = v0
    a[:, 0] = spsolve(M, F[:,0] - C @ v0 - K @ x0)

    gamma_h = gamma * h
    beta_h2 = beta * h**2
    one_minus_gamma_h = (1 - gamma) * h
    half_minus_beta_h2 = (0.5 - beta) * h**2

    S = (M + gamma_h*C + beta_h2*K).tocsc()
    S_inv = inv(S)


    for i in range(1, nt):
        v_pred = v[:, i-1] + one_minus_gamma_h * a[:, i-1]
        x_pred = x[:, i-1] + h * v[:, i-1] + half_minus_beta_h2 * a[:, i-1]

        da = S_inv @ (F[:,i] - C @ v_pred - K @ x_pred)

        a[:, i] = da
        v[:, i] = v_pred + gamma_h * da
        x[:, i] = x_pred + beta_h2 * da

    return x, v, a

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




