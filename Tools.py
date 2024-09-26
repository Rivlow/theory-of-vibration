import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def display(fig, ax, activation, nodes, elems):

    """
    if activation["print"]:

        print(f"Total number of nodes: {np.shape(nodes)[0]}")
        print(f"Total number of DOF's: {np.shape(DOF)[0]*2}")
        print(f"Total number of beams: {np.shape(elems)[0]}")
    """

    if activation["plot"]:

        # nodes display
        x_node = [xyz.pos[0] for xyz in nodes]
        y_node = [xyz.pos[1] for xyz in nodes]
        z_node = [xyz.pos[2] for xyz in nodes]

        ax.scatter(x_node, y_node, z_node)
        for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
            ax.text(x_val, y_val, z_val, str(i), fontsize=9, ha='center')

        
        # Beams display
        for i in range(np.shape(elems)[0]):

            idx_in = nodes[elems[i].nodes[0]].idx # elems[i].nodes[0] = 14
            idx_out = nodes[elems[i].nodes[1]].idx

            x = [nodes[idx_in].pos[0], nodes[idx_out].pos[0]]
            y = [nodes[idx_in].pos[1], nodes[idx_out].pos[1]]
            z = [nodes[idx_in].pos[2], nodes[idx_out].pos[2]]

            ax.plot(x, y, z, color="blue")
        
        ax.set_zlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")


def printData(nodes_list, elems_list):

    print("#----------------------#")
    print("#       FEM data       #")
    print("#----------------------#")

    print(f"Number of nodes : {len(nodes_list)}")
    print(f"Number of (unclamped) DOF's : {6*len(nodes_list) - 36}")
    print(f"Number of elements : {len(elems_list)} \n")

    elem_0 = elems_list[0]

    print(f"A : {elem_0.A} [m^2]")
    print(f"Iz : {elem_0.Iz} [m^4]")
    print(f"Iy : {elem_0.Iy} [m^4]")
    print(f"Jx : {elem_0.Jx} [m^4]")
    print(f"M_lumped : {elem_0.M_lumped} [kg]")
    print("\n")



def plotModes(fig, ax, nodes_list, displacements, elems_list, coef, clamped_nodes):


    # Remove clamped index nodes 
    unclamped_nodes_list = [x for x in range(len(nodes_list)) if x not in clamped_nodes]

    # Retained only three first DOF's of each node (u, v, w displacements)
    new_idx = [6*i + j for i in unclamped_nodes_list for j in range(3)]
    new_idx_i = np.arange(0, len(new_idx)+3, 3)

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
  
    



