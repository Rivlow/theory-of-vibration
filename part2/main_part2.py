import numpy as np
import os
import scipy
import sys

from part1 import FEM, set_parameters
from damping import *
from NewMark import *
from mode_method import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



def main():

    #===============================================#
    # From part 1, compute K, M, eigen values/modes #
    #===============================================#
    geom_data, phys_data = set_parameters.setParams()
    elem_per_beam = 1

    nodes_list_init, nodes_pairs_init = FEM.initializeGeometry(geom_data, phys_data)
    nodes_list, nodes_pairs = FEM.addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = FEM.createElements(nodes_pairs, nodes_list, geom_data, phys_data)

    solver = FEM.Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])

    K, M = solver.extractMatrices()
    eigen_vals, eigen_vectors = solver.solve()
    eigen_vectors = normalize_eigenvectors(eigen_vectors, M) # sparse.linalg.eig does not normalize
    eigen_vals = 2*pi*eigen_vals

    #=============================#
    # Transient response (part 2) #
    #=============================#
    m_tot = 9*80 # [kg]
    h = 0.2 # [m]
    g = 9.81 # [m/s^2]
    freq = 2 # [Hz]
    delta_t = 0.1 # [s]
    params = {"m_tot":m_tot, "h":h, "g":g, "dt":delta_t, "freq":freq}
    nb_modes = 4

    node_force = [23, 24]
    node_observation = [10, 11]
    nodes_clamped = [0, 1, 6, 7, 12, 13]

    a, b, C = dampingMatrix(K, M, eigen_vals[0], eigen_vals[1])
    epsilon = dampingRatios(a, b, eigen_vals)

    period = 1/2 # f = 2 [hz] <-> T = 1/2 [s]
    nb_step = 1000
    n = 100
    t_span = np.linspace(0.1, n*period, n*nb_step)

    F = computeForce(params, nodes_clamped, eigen_vectors, nb_modes, t_span)

    eta, phi, mu  = etaPhiMu(eigen_vals, eigen_vectors, epsilon, M, F, t_span, nb_modes)

    #---------------------------------------#
    # Mode displacement/acceleration method #
    #---------------------------------------#
    DOF_1 = extractDOF(23, nodes_clamped)
    DOF_2 = extractDOF(11, nodes_clamped)
    z_dir = DOF_1+2

    """
    
    q = modeDisplacementMethod(eta, eigen_vectors, t_span, nb_modes)[z_dir,:]
    q_acc = modeAccelerationMethod(eta, eigen_vals, eigen_vectors, t_span, K, phi, params, nodes_clamped, nb_modes)

    q_fft = np.fft.fft(q)
    freq = np.fft.fftfreq(len(t_span), t_span[1]-t_span[0]) #frequency samples

    
    # On ne garde que la partie positive du spectre
    positive_freq = freq[:len(freq)//2]
    magnitudes = np.abs(q_fft)[:len(freq)//2]

    # Détection des pics (fréquences d'excitation)
    peaks, _ = find_peaks(magnitudes)  # Ajustez le seuil si nécessaire
    excitation_freqs = positive_freq[peaks]

 

    M_norm = mNorm(eta, mu, nb_modes)

    # Transient analysis
    envelope = extract_envelope(q, t_span)
    #plot_signal_with_envelope(t_span, q, envelope)
    transition_index = np.argmin(np.abs(np.gradient(envelope, t_span)))
    transition_time = t_span[transition_index]
    print(f"State transition (transient -> steady) around t = {transition_time:.2f} s")


    """
    
    # NewMark integration algorithm
    x0 = v0 = np.zeros_like(eigen_vectors[:,0])
    x, v, a = newmark_integration(M, C, K, x0, v0, t_span,
                                  F, gamma=0.5, beta=0.25)

    plt.figure()
    plt.plot(t_span, x[z_dir,:])
    plt.show()


    


   
    

if __name__  == "__main__":
    main()