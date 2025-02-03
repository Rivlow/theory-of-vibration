import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from part1.FEM import *

from part2.Newmark import *
from part2.Tools_part2 import *
from part2.mode_method import *
from part2.damping import *

from part3.reduction_method import *
from part3.Tools_part3 import *
from part3.get_params_part3 import setParams as setParams3


def main():

    #########################
    # Part 1 : compute K, M #
    #########################
    elem_per_beam = 5
    n_modes =  14

    geom_data, phys_data, sim_data, reduction_data = setParams3()
    nodes_list_init, nodes_pairs_init = initializeGeometry(geom_data, phys_data)
    nodes_list, nodes_pairs = addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = createElements(nodes_pairs, nodes_list, geom_data, phys_data)

    solver = Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])
    freq_init, modes_init = solver.solve(n_modes)

    K, M = solver.extractMatrices()

    ######################
    # Part 2 : compute C #
    ######################
    C = dampingMatrix(K, M, 2*np.pi*freq_init[0], 2*np.pi*freq_init[1], 
                      sim_data["damping"]["mode1_ratio"], sim_data["damping"]["mode2_ratio"])[2]

    #################################
    # Part 3 : Reduction algorithms #
    #################################
    #displayRetained(nodes_list, reduction_data["nodes_retained"], elems_list, geom_data, save=True, latex=True)

    
    # Define input variables
    t_span = sim_data["t_span"]
    F = computeForce(sim_data, geom_data, 2, modes_init, t_span)
    x0 = v0 = np.zeros_like(modes_init[:,0]) # initial conditions

    # Separate K,M and C, F, x0, v0 w.r.t. condensed/retained dofs
    retained_dofs = retainedDOF(reduction_data["nodes_retained"], reduction_data["node_dof_config"], geom_data["nodes_clamped"])
    K_parts, M_parts, C_parts, condensed_dofs, retained_dofs = partition_matrices(K, M, C, retained_dofs)

    # Apply reduction method
    freq_gi, modes_gi, K_gi, M_gi, C_gi, R_gi, F_gi, x0_gi, v0_gi = GuyanIronsReduction(K_parts, M_parts, C_parts, retained_dofs, F, x0, v0, n_modes)
    freq_cb, modes_cb, K_cb, M_cb, C_cb, R_cb, F_cb, x0_cb, v0_cb = CraigBamptonReduction(K_parts, M_parts, C_parts, retained_dofs, condensed_dofs, 
                                                                                          F, x0, v0, n_interface_modes=5, n_eigen=n_modes)
    
    #plotFrequenciesComparison(freq_init, freq_cb, freq_gi, save=True, latex=True)
    #compareFullGIFreq(freq_init, freq_gi)
    #compareFullCBFreq(freq_init, freq_cb)
    #convergenceCB(freq_init, K_parts, M_parts, C_parts, condensed_dofs, range(0, 20), save=True, latex=True)
    #compareTimeReductionMethod(t_span, sim_data, K, M, C, K_parts, M_parts, C_parts, condensed_dofs, retained_dofs,  F, x0, v0,
    #                           6, geom_data, phys_data, np.arange(1, 15, 1), save=False, latex=False)


    # Newmark on reduced model
    DOF_1 = extractDOF(sim_data["nodes_obs"][0], geom_data["nodes_clamped"])
    DOF_2 = extractDOF(sim_data["nodes_obs"][1], geom_data["nodes_clamped"])
    z_dir = DOF_2 + 2

    DOF_1_red = findReducedDOF(sim_data["nodes_obs"][0], reduction_data["nodes_retained"], reduction_data["node_dof_config"])
    DOF_2_red = findReducedDOF(sim_data["nodes_obs"][1], reduction_data["nodes_retained"], reduction_data["node_dof_config"])
    z_dir_red = DOF_2_red + 2
    

    
    x_init, v_init, a_init = NewmarkIntegration(M, C, K, x0, v0, t_span, F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
    x_gi, v_gi, a_gi = NewmarkIntegration(M_gi, C_gi, K_gi, x0_gi, v0_gi, t_span, F_gi, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
    x_cb, v_cb, a_cb = NewmarkIntegration(M_cb, C_cb, K_cb, x0_cb, v0_cb, t_span, F_cb, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])

    var_to_plot = [x_init[z_dir, :], x_gi[z_dir_red,:]]
    var_name = ['Full system', "Guyan-Irons"]
    var_ls = ['-', "--"]
    var_color = ['blue', 'green']
    xlabel = "time t [s]"
    ylabel = "generalized displacement [m]"
    plotAll3(t_span, var_to_plot, var_name, var_color, var_ls, xlabel, ylabel, save=True, latex=True, name_save="guyan_vs_newmark_first_load_23")
    


    # Compute MAC matrices
    #compute_MAC(modes_init, modes_gi, R_gi, retained_dofs, condensed_dofs, name="gi", save=True, latex=True)
    #compute_MAC(modes_init, modes_cb, R_cb, retained_dofs, condensed_dofs, name="cb", save=True, latex=True)




    

if __name__  == "__main__":
    main()