import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq

def NewmarkIntegration(M, C, K, x0, v0, t_span, F, gamma, beta):
    
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
        a_pred = S_inv @ (F[:,i] - C @ v_pred - K @ x_pred)

        a[:, i] = a_pred
        v[:, i] = v_pred + gamma_h * a_pred
        x[:, i] = x_pred + beta_h2 * a_pred

    return x, v, a


def analysisTransient(q, t_span):

    analytic_signal = hilbert(q)
    amplitude_envelope = np.abs(analytic_signal)
    transition_index = np.argmin(np.abs(np.gradient(amplitude_envelope, t_span)))
    transition_time = t_span[transition_index]
    print(f"State transition (transient -> steady) around t = {transition_time:.2f} s")

def FFTNewmark(x, t_span):
    N = len(t_span)
    T = t_span[1] - t_span[0]  
    yf = fft(x)
    xf = fftfreq(N, T)[:N//2]  # positive freq
    amplitude = 2.0/N * np.abs(yf[0:N//2])
   
    return xf, amplitude

