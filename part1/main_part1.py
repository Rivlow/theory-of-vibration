import numpy as np
import matplotlib.pyplot as plt
import sys
import os
pi = np.pi

from FEM import *
from Tools_part1 import *
from set_parameters import *
from convergence_part1 import *

def main():

    # Dictionnary to monitor outputs of the code
    activation = {"print":True, "plot":True}

    geom_data, phys_data = setParams()
    elem_per_beam = 1

    # Define initial geometry
    nodes_list_init, nodes_pairs_init = initializeGeometry(geom_data, phys_data)

    # Add nodes if needed
    nodes_list, nodes_pairs = addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
    elems_list = createElements(nodes_pairs, nodes_list, geom_data, phys_data)
    printData(nodes_list, elems_list, phys_data, geom_data)
    
    solver = Solver()
    solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
    solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
    solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])

    K, M = solver.extractMatrices()
    eigen_vals, eigen_vectors = solver.solve()

    # Display
    fig = plt.figure(figsize=(10, 8), facecolor='none', edgecolor='none')
    ax = fig.add_subplot(projection='3d')
    #display(fig, ax, nodes_list, elems_list, geom_data, save=True, github=True)
    #plotModes(fig, ax, nodes_list, eigen_vectors[:,1], elems_list, geom_data["nodes_clamped"], save=True, github=True)
    plt.show()

    convergence(geom_data, phys_data, max_nb_elem=8, plot=True, github=True)

if __name__  == "__main__":
    main()