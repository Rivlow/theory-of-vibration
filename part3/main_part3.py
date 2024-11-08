import numpy as np
import os
import scipy
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1 import FEM, set_parameters
from part2 import Newmark, Tools_part2
from reduction_method import *
from Tools_part3 import *


def main():

    #########################
    # Part 1 : compute K, M #
    #########################
    geom_data, phys_data = set_parameters.setParams()
    elem_per_beam = 1

    nodes_list_init, nodes_pairs_init = FEM.initializeGeometry(geom_data, phys_data)
    nodes_list, nodes_pairs = FEM.addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = FEM.createElements(nodes_pairs, nodes_list, geom_data, phys_data)

    solver = FEM.Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])
    freq_init, modes_init = solver.solve()

    K, M = solver.extractMatrices()

    #################################
    # Part 3 : Reduction algorithms #
    #################################

    nodes_force = [23, 24]
    nodes_observation = [10, 11]
    nodes_clamped = [0, 1, 6, 7, 12, 13]
    nodes_retained = [10, 23, 24, 29, 30]
    node_dof_config = {nodes_retained[0]: ["u", "v", "w"],
                       nodes_retained[1]: ["u", "v", "w"],
                       nodes_retained[2]: ["u", "v", "w"],
                       nodes_retained[3]: ["u", "v", "w"],
                       nodes_retained[4]: ["u", "v", "w"]}
    


    # Separate K,M w.r.t. condensed/retained dofs
    retained_dofs = retainedDOF(nodes_retained, node_dof_config, nodes_clamped)
    print(retained_dofs)

    K_parts, M_parts, condensed_dofs, retained_dofs = partition_matrices(K, M, retained_dofs)

    freq_gi, modes_gi = GuyanIronsReduction(K_parts, M_parts, condensed_dofs, retained_dofs)
    freq_cb, modes_cb = CraigBamptonReduction(K_parts, M_parts, condensed_dofs,n_modes=4)
    
    print("True frequencies:")
    print(freq_init)

    print("\nfrequencies by Guyan Irons:")
    print(freq_gi)

    print("\nfrequencies by Craig Bampton:")
    print(freq_cb)
    
    


   

if __name__  == "__main__":
    main()