import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq

def NewmarkIntegration(M, C, K, x0, v0, t_span, F,
                                 gamma, beta):
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

def extractEnvelope(q, t_span):
    """
    Extraire l'enveloppe du signal q en utilisant la transformée de Hilbert.
    """
    analytic_signal = hilbert(q)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def analysisFFT(x, t_span):
    # Calcul de la FFT
    N = len(t_span)
    T = t_span[1] - t_span[0]  # période d'échantillonnage
    yf = fft(x)
    xf = fftfreq(N, T)[:N//2]  # fréquences positives uniquement
    amplitude = 2.0/N * np.abs(yf[0:N//2])

    plt.figure()
    plt.plot(xf, amplitude)
    plt.show()

    """
    # Calcul de l'amplitude du spectre
    amplitude = 2.0/N * np.abs(yf[0:N//2])

    # Création du graphique 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Création d'une grille pour le graphique 3D
    X, Y = np.meshgrid(t_span, xf)
    Z = np.zeros_like(X)

    Z = amplitude[:, np.newaxis] * np.sin(2 * np.pi * xf[:, np.newaxis] * t_span)

    # Tracé de la surface 3D
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)

    # Configuration des axes et du titre
    ax.set_xlabel('Temps')
    ax.set_ylabel('Fréquence (Hz)')
    ax.set_zlabel('Amplitude')
    ax.set_title('Analyse FFT - Représentation 3D')

    # Ajout d'une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

        """
    # Retour des résultats de la FFT pour une utilisation ultérieure si nécessaire
    return xf, amplitude



