import numpy as np
import os
import scipy
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1 import FEM, set_parameters
from damping import *
from NewMark import *
from mode_method import *
from Tools_part2 import *




def main():

    #################################################
    # Part 1 : compute K, M, eigen values/modes #
    #################################################
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
    frequencies, modes = solver.solve()
    modes = normalize_eigenvectors(modes, M) # sparse.linalg.eig does not normalize

    ###############################
    # Part 2 : transient response #
    ###############################
    params = {"m_tot":9*80, "h":0.2, "g":9.81, "dt":0.1, "freq":2}
    nb_modes = 6

    nodes_force = [23, 24]
    nodes_observation = [10, 11]
    nodes_clamped = [0, 1, 6, 7, 12, 13]

    a, b, C = dampingMatrix(K, M, 2*pi*frequencies[0], 2*pi*frequencies[1])
    epsilon = dampingRatios(a, b, 2*pi*frequencies)

    period = 1/2 # f = 2 [hz] <-> T = 1/2 [s]
    nb_timestep = 1000
    n = 1
    t_span = np.linspace(0.1, n*period, n*nb_timestep)

    F = computeForce(params, 
                     nodes_clamped, nodes_force, 2,
                     modes, 
                     t_span)

    #---------------------------------------#
    # Mode displacement/acceleration method #
    #---------------------------------------#
    DOF_1 = extractDOF(23, nodes_clamped)
    DOF_2 = extractDOF(11, nodes_clamped)
    z_dir = DOF_1+2

    """
    eta, phi, mu  = etaPhiMu(2*pi*frequencies, modes, epsilon, M, F, t_span, nb_modes)
    q = modeDisplacementMethod(eta, modes, nb_modes)
    q_acc = modeAccelerationMethod(eta, 2*pi*frequencies, modes, t_span, K, phi, F, nb_modes)
    M_norm = mNorm(eta, mu, nb_modes)
    """

    #convergence(eta, modes, 2*pi*frequencies, K, phi, F, t_span, nb_modes, z_dir)
    

    #-------------------------------#
    # NewMark integration algorithm #
    #-------------------------------#
    x0 = v0 = np.zeros_like(modes[:,0]) # initial conditions
    q_nm, q_dot_nm, q_dot_dot_nm = NewmarkIntegration(M, C, K, 
                                                      x0, v0, t_span, F,
                                                      0.5, 0.25)
    
    #plotAll(t_span, q[z_dir,:], q_nm[z_dir,:], separate=False)

    
    """
    # Transient analysis
    envelope = extractEnvelope(q, t_span)
    transition_index = np.argmin(np.abs(np.gradient(envelope, t_span)))
    transition_time = t_span[transition_index]
    print(f"State transition (transient -> steady) around t = {transition_time:.2f} s")
    """
    
    # FFt analysis
    analysisFFT(q_nm[z_dir,:], t_span)




    


   
    

if __name__  == "__main__":
    main()