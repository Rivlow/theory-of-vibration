import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def display(fig, ax, activation, nodes, elems):


    """
    if activation["print"]:

        print(f"Total number of nodes: {np.shape(nodes)[0]}")
        print(f"Total number of DOF's: {np.shape(DOF)[0]*2}")
        print(f"Total number of beams: {np.shape(elems)[0]}")
    """
    if activation["plot"]:
        # Nodes display
        x_node = [node.pos[0] for node in nodes]
        y_node = [node.pos[1] for node in nodes]
        z_node = [node.pos[2] for node in nodes]

        # Scatter plot for nodes
        ax.scatter(x_node, y_node, z_node, c="blue", s=20, depthshade=True)

        # Annotate nodes with their index
        for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
            ax.text(x_val, y_val, z_val, f'{i}', fontsize=9, ha='center', color='black')

        # Beams (elements) display
        for elem in elems:
            idx_in = nodes[elem.nodes[0]].idx
            idx_out = nodes[elem.nodes[1]].idx

            x = [nodes[idx_in].pos[0], nodes[idx_out].pos[0]]
            y = [nodes[idx_in].pos[1], nodes[idx_out].pos[1]]
            z = [nodes[idx_in].pos[2], nodes[idx_out].pos[2]]

            ax.plot(x, y, z, color="blue", linewidth=1.5, alpha=0.5)

        # Set plot labels and limits
        ax.set_xlabel("X-axis", fontsize=12)
        ax.set_ylabel("Y-axis", fontsize=12)
        ax.set_zlabel("Z-axis", fontsize=12)

        # Automatically adjust the limits based on the node coordinates
        ax.set_xlim([min(x_node) - 1, max(x_node) + 1])
        ax.set_ylim([min(y_node) - 1, max(y_node) + 1])
        ax.set_zlim([min(z_node) - 1, max(z_node) + 1])

        # Add grid and title
        ax.grid(True)




def printData(nodes_list, elems_list, phys_data, geom_data):

    print("#----------------------#")
    print("#       FEM data       #")
    print("#----------------------#")

    print(f"Number of nodes : {len(nodes_list)}")
    print(f"Number of (unclamped) DOF's : {6*len(nodes_list) - 36}")
    print(f"Number of elements : {len(elems_list)} \n")

    print(f"A : {round(geom_data["A"],6)} [m^2]")
    print(f"Iz : {round(geom_data["Iz"],6)} [m^4]")
    print(f"Iy : {round(geom_data["Iy"],6)} [m^4]")
    print(f"Jx : {round(geom_data["Jx"],6)} [m^4]")
    print(f"M_lumped : {round(phys_data["M_lumped"],6)} [kg]")
    print("\n")



def plotModes(fig, ax, nodes_list, displacements, elems_list, coef, clamped_nodes):


    # Remove clamped index nodes 
    unclamped_nodes_list = [x for x in range(len(nodes_list)) if x not in clamped_nodes]

    # Retained only three first DOF's of each node (u, v, w displacements)
    new_idx = [6*i + j for i in unclamped_nodes_list for j in range(3)]
    new_idx_i = np.arange(0, len(new_idx)+3, 3)

    print(f"new_idx {new_idx}")
    print("\n")
    print(f"new_idx_i {new_idx_i}")

    nodes = nodes_list.copy()

    for i in unclamped_nodes_list:
        for j in new_idx_i:
            nodes[i].pos[:] += coef*displacements[j:j+3]

    # nodes display
    x_node = [xyz.pos[0] for xyz in nodes]
    y_node = [xyz.pos[1] for xyz in nodes]
    z_node = [xyz.pos[2] for xyz in nodes]


    #ax.scatter(x_node, y_node, z_node, color = "red", alpha=0.5)
    
    
    # Beams display
    for i in range(np.shape(elems_list)[0]):

        idx_in = nodes[elems_list[i].nodes[0]].idx # elems[i].nodes[0] = 14
        idx_out = nodes[elems_list[i].nodes[1]].idx

        x = [nodes[idx_in].pos[0], nodes[idx_out].pos[0]]
        y = [nodes[idx_in].pos[1], nodes[idx_out].pos[1]]
        z = [nodes[idx_in].pos[2], nodes[idx_out].pos[2]]

        ax.plot(x, y, z, color="red", linestyle = "--")



    

  
    



