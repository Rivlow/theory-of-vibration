import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def printData(nodes_list, elems_list, phys_data, geom_data):

    print("#----------------------#")
    print("#       FEM data       #")
    print("#----------------------#")

    print(f"Total number of nodes : {len(nodes_list)}, ({len(geom_data["nodes_clamped"])} clamped)")
    print(f"Number of lumped nodes : {len(geom_data["nodes_lumped"])}")
    print(f"Number of (unclamped) DOF's : {6*len(nodes_list) - 36}, ({6*len(nodes_list)} - 36)")
    print(f"Number of elements : {len(elems_list)} \n")

    print(f"A : {round(geom_data["A"],6)} [m^2]")
    print(f"Iz : {round(geom_data["Iz"],6)} [m^4]")
    print(f"Iy : {round(geom_data["Iy"],6)} [m^4]")
    print(f"Jx : {round(geom_data["Jx"],6)} [m^4]")
    print(f"M_lumped : {round(phys_data["M_lumped"],6)} [kg]")
    print("\n")


def display(fig, ax, activation, nodes_list, elems_list, geom_data):

    if activation["plot"]:
        # Nodes display
        x_node = [node.pos[0] for node in nodes_list]
        y_node = [node.pos[1] for node in nodes_list]
        z_node = [node.pos[2] for node in nodes_list]

        # Flags to avoid multiple labels
        label_clamped_shown = False
        label_lumped_shown = False

        for node in nodes_list:
            if node.idx in geom_data["nodes_clamped"]:
                if not label_clamped_shown:
                    ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="green", s=20, label="clamped", depthshade=True)
                    label_clamped_shown = True
                else:
                    ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="green", s=20, depthshade=True)

            elif hasattr(node, 'M_lumped'):
                if not label_lumped_shown:
                    ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="red", s=20, label="lumped", depthshade=True)
                    label_lumped_shown = True
                else:
                    ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="red", s=20, depthshade=True)

            else:
                ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="blue", s=20, depthshade=True)

        # Annotate nodes with their index
        for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
            ax.text(x_val, y_val, z_val, f'{i}', fontsize=9, ha='center', color='black')

        # Beams (elements) display
        for elem in elems_list:
            idx_in = nodes_list[elem.nodes[0]].idx
            idx_out = nodes_list[elem.nodes[1]].idx

            x = [nodes_list[idx_in].pos[0], nodes_list[idx_out].pos[0]]
            y = [nodes_list[idx_in].pos[1], nodes_list[idx_out].pos[1]]
            z = [nodes_list[idx_in].pos[2], nodes_list[idx_out].pos[2]]

            ax.plot(x, y, z, color="blue", linewidth=1.5, alpha=0.5)

        # Set plot labels 
        ax.set_xlabel("X-axis", fontsize=12)
        ax.set_ylabel("Y-axis", fontsize=12)
        ax.set_zlabel("Z-axis", fontsize=12)

        # Adjust the limits based on the node coordinates
        ax.set_xlim([min(x_node) - 1, max(x_node) + 1])
        ax.set_ylim([min(y_node) - 1, max(y_node) + 1])
        ax.set_zlim([min(z_node) - 1, max(z_node) + 1])

        ax.legend(loc="best")
        ax.grid(True)


def plotModes(fig, ax, nodes_list, displacements, elems_list, coef, nodes_clamped):


    # Remove clamped index nodes 
    unclamped_nodes_list = [x for x in range(len(nodes_list)) if x not in nodes_clamped]

    # Retained only three first DOF's of each node (u, v, w displacements)
    new_idx = [6*i + j for i in unclamped_nodes_list for j in range(3)]
    new_idx_i = np.arange(0, len(new_idx)+3, 3)

    nodes = nodes_list.copy()

    for i in unclamped_nodes_list:
        for j in new_idx_i:
            nodes[i].pos[:] += coef*displacements[j:j+3]

    for elem in elems_list:

        idx_in = nodes[elem.nodes[0]].idx 
        idx_out = nodes[elem.nodes[1]].idx

        x = [nodes[idx_in].pos[0], nodes[idx_out].pos[0]]
        y = [nodes[idx_in].pos[1], nodes[idx_out].pos[1]]
        z = [nodes[idx_in].pos[2], nodes[idx_out].pos[2]]

        ax.plot(x, y, z, color="red")

def computeMass(M):

    # Rigid-body mode in translation
    u = np.array([1 if (i == 0 or i%6 == 0) else 0 for i in range(M.shape[0])])

    return u.T @ M @ u



    

  
    



