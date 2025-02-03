import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from part1.FEM import Solver, initializeGeometry, addMoreNodes, createElements

from part2.Tools_part2  import *
from part2.get_params_part2 import setParams as setParams2
from part2.damping import *
from part2.mode_method import *
from part2.Newmark import *


def main():

    #############################################
    # Part 1 : compute K, M, eigen values/modes #
    #############################################
    elem_per_beam = 1
    n_modes = 10

    geom_data, phys_data, sim_data = setParams2()
    nodes_list_init, nodes_pairs_init = initializeGeometry(geom_data, phys_data)
    nodes_list, nodes_pairs = addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = createElements(nodes_pairs, nodes_list, geom_data, phys_data)

    solver = Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])

    K, M = solver.extractMatrices()
    frequencies, modes = solver.solve(n_modes)

    ###############################
    # Part 2 : transient response #
    ###############################
    a, b, C = dampingMatrix(K, M, 2*np.pi*frequencies[0], 2*np.pi*frequencies[1], 
                            sim_data["damping"]["mode1_ratio"], sim_data["damping"]["mode2_ratio"])
    
    epsilon = dampingRatios(a, b, 2*np.pi*frequencies)
    t_span = sim_data["t_span"]

    F = computeForce(sim_data, geom_data, 2, modes, t_span)

    #---------------------------------------#
    # Mode displacement/acceleration method #
    #---------------------------------------#
    eta, phi, mu  = etaPhiMu(2*np.pi*frequencies, modes, epsilon, M, F, t_span, n_modes)
    #q = modeDisplacementMethod(eta, modes, n_modes)
    #q_acc = modeAccelerationMethod(t_span, eta, 2*np.pi*frequencies, modes, K, phi, F, n_modes)
    #M_norm = mNorm(eta, mu, n_modes)
    
    #-------------------------------#
    # NewMark integration algorithm #
    #-------------------------------#
    x0 = v0 = np.zeros_like(modes[:,0]) # initial conditions
    #q_nm, q_dot_nm, q_dot_dot_nm = NewmarkIntegration(M, C, K, x0, v0, t_span, F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])

    #----------------#
    # Plot variables #
    #----------------#
    DOF_1 = extractDOF(sim_data["nodes_obs"][0], geom_data["nodes_clamped"])
    DOF_2 = extractDOF(sim_data["nodes_obs"][1], geom_data["nodes_clamped"])
    z_dir = DOF_2 + 2

    displayObsLoadNodes(nodes_list, 
                        obs_nodes=sim_data["nodes_obs"], 
                        load_nodes_1=[23, 24], 
                        load_nodes_2=[10, 11], 
                        elems_list=elems_list,
                        save=True,
                        latex=True)

    #convergenceAmplitude(eta, modes, 2*np.pi*frequencies, K, phi, F, t_span, n_modes, z_dir, name="conv_amplitude_lc_2_23", save=False, latex=False)
    #analysisTransient(q_nm[z_dir,:], t_span) # find time at which transient -> steady state
    #convergence(eta, modes, 2*pi*frequencies, K, phi, F, t_span, n_modes, z_dir)
    #plotSingle(t_span, relative_error, "Time [s]", "Displacement [m]", save=False, name="nm_vs_disp_10_first_load", latex=False)

    #compareNm(t_span, q_nm, q, z_dir, latex=True, save=True, name='compare_nm_disp_10_first_load')
    #analyze_error(q_nm, q, z_dir, latex=True, save=True, name='stat_nm_disp_10_first_load')


    
    '''
    xf, amplitude = FFTNewmark(q_nm[z_dir,:], t_span) 
    plotFFT(xf, amplitude, save=True, name_save="fft_nm_10_second_load", latex=True, 
            height_ratio=0.000001,     # Plus petit = plus de pics détectés
            prominence_ratio=0.0000001, # Plus petit = plus de pics détectés
            distance=3)     
    #plotSingle(t_span, q_nm[z_dir,:], "Time [s]", "Displacement [m]", save=True, name="nm_23_second_load", latex=True)
    var_to_plot = [q[z_dir, :], q_acc[z_dir,:]]
    var_name = ['displacement method', 'acceleration method']
    var_ls = ['-', ':']
    var_color = ['blue', 'red']
    xlabel = "time t [s]"
    ylabel = "generalized displacement [m]"
    
  
    plotAll(t_span, var_to_plot, var_name, var_color, var_ls, xlabel, ylabel, save=False, latex=False, name_save="acc_node_23_long")
    '''

if __name__  == "__main__":
    main()