import numpy as np
import matplotlib.pyplot as plt

from part1.FEM import *
from part1.Tools_part1 import *
from part1.get_params_part1 import setParams as setParams1

def main():

    geom_data, phys_data = setParams1()
    elem_per_beam = 1
    n_modes = 6

    # Define initial geometry
    nodes_list_init, nodes_pairs_init = initializeGeometry(geom_data, phys_data)

    # Add nodes if needed
    nodes_list, nodes_pairs = addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = createElements(nodes_pairs, nodes_list, geom_data, phys_data)
    printData(nodes_list, elems_list, phys_data, geom_data)
    
    solver = Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    print(solver.computeMass())
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])

    K, M = solver.extractMatrices()
    frequencies, modes = solver.solve(n_modes)

    print(frequencies)

    # Display
    fig = plt.figure(figsize=(10, 8), facecolor='none', edgecolor='none')
    ax = fig.add_subplot(projection='3d')
    display(fig, ax, nodes_list, elems_list, geom_data, save=True, github=False, latex=True)
    #plotModes(nodes_list, n_modes, modes, elems_list, geom_data["nodes_clamped"], save=True, github=False, latex=True)
    plt.show()

    #convergence(geom_data, phys_data, n_modes, max_nb_elem=8, plot=True, github=False, latex=True)

if __name__  == "__main__":
    main()