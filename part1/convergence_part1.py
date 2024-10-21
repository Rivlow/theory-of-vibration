import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
pi = np.pi

from FEM import *
from set_parameters import *

def convergence(geom_data, phys_data, max_nb_elem, plot, github):
    elem_per_beam_list = np.arange(1, max_nb_elem+1, 1)
    eigen_freq_matrix = []

    for elem_per_beam in elem_per_beam_list:
        # Define initial geometry
        nodes_list_init, nodes_pairs_init = initializeGeometry(geom_data, phys_data)

        # Add nodes if needed
        nodes_list, nodes_pairs = addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
        elems_list = createElements(nodes_pairs, nodes_list, geom_data, phys_data)
        
        solver = Solver()
        solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
        solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
        solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])
        
        eigen_vals, eigen_vectors = solver.solve()
        eigen_freq_matrix.append(eigen_vals)

    eigen_freq_matrix = np.array(eigen_freq_matrix)
    
    relative_errors = np.zeros((len(elem_per_beam_list)-1, eigen_freq_matrix.shape[1]))

    for i in range(len(elem_per_beam_list)-1):
        relative_errors[i] = abs((eigen_freq_matrix[i+1] - eigen_freq_matrix[i]) / eigen_freq_matrix[i] * 100)

    plt.figure(figsize=(10, 8))

    for i in range(relative_errors.shape[1]):
        plt.plot(elem_per_beam_list[1:], relative_errors[:, i], marker='o', label=f'Mode {i+1}')

    plt.xlabel('Number of elements per beam [-]', color='white' if github else 'black')
    plt.ylabel('Relative error [%]', color='white' if github else 'black')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if github:
        plt.gca().set_facecolor('none')
        plt.gcf().patch.set_alpha(0)
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        
        # Change legend text color to white
        legend = plt.gca().get_legend()
        for text in legend.get_texts():
            text.set_color("white")

    if plot:
        if github:
            plt.savefig('part1/Pictures/convergence_part1.png', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('part1/Pictures/convergence_part1.PDF')
        
        plt.show()
   