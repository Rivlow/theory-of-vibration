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
    elem_per_beam = 1
    n_modes = 6

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

    # Separate K,M and C w.r.t. condensed/retained dofs
    retained_dofs = retainedDOF(reduction_data["nodes_retained"], reduction_data["node_dof_config"], geom_data["nodes_clamped"])
    K_parts, M_parts, C_parts, condensed_dofs, retained_dofs = partition_matrices(K, M, C, retained_dofs)

    # Apply reduction method
    freq_gi, modes_gi, K_gi, M_gi, C_gi, R_gi = GuyanIronsReduction(K_parts, M_parts, C_parts, retained_dofs)
    freq_cb, modes_cb, K_cb, M_cb, C_cb, R_cb = CraigBamptonReduction(K_parts, M_parts, C_parts, condensed_dofs, n_modes)
    
    #print(f"\nfrequencies by Craig Bampton (rel error in %): {abs(freq_cb-freq_init)/freq_init*100}")

    # Newmark on reduced model
    t_span = sim_data["t_span"]

    F = computeForce(sim_data, geom_data, 2, modes_init, t_span)
    DOF_1 = extractDOF(sim_data["nodes_obs"][0], geom_data["nodes_clamped"])
    DOF_2 = extractDOF(sim_data["nodes_obs"][1], geom_data["nodes_clamped"])
    DOF_1_reduced = findReducedDOF(sim_data["nodes_obs"][0], 
                                   reduction_data["nodes_retained"], 
                                   reduction_data["node_dof_config"])
    
    DOF_2_reduced = findReducedDOF(sim_data["nodes_obs"][1], 
                                   reduction_data["nodes_retained"], 
                                   reduction_data["node_dof_config"])
    
    z_dir_reduced = DOF_2_reduced + 2 
    z_dir = DOF_1 + 2
    print(z_dir_reduced)
    print(z_dir)

    # initial conditions
    x0 = v0 = np.zeros_like(modes_init[:,0]) 

    x_init, v_init, a_init = NewmarkIntegration(M, C, K, x0, v0, t_span, F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
    x_gi, v_gi, a_gi = NewmarkIntegration(M_gi, C_gi, K_gi, R_gi.T@x0, R_gi.T@v0, t_span, R_gi.T@F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
    x_cb, v_cb, a_cb = NewmarkIntegration(M_cb, C_cb, K_cb, R_cb.T@x0, R_cb.T@v0, t_span, R_cb.T@F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])

    print(np.shape(x_cb))

    var_to_plot = [x_init[z_dir, :], x_cb[4,:]]
    var_name = ['Full system', "Craig-Bampton"]
    var_ls = ['-', "--"]
    var_color = ['blue', 'green']
    xlabel = "time t [s]"
    ylabel = "generalized displacement [m]"
    plotAll(t_span, var_to_plot, var_name, var_color, var_ls, xlabel, ylabel, save=False, name="disp_vs_acc")
    

if __name__  == "__main__":
    main()