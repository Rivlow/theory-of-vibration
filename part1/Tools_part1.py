import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


def display(fig, ax, nodes_list, elems_list, geom_data, save, github):
    # Nodes display
    x_node = [node.pos[0] for node in nodes_list.values()]
    y_node = [node.pos[1] for node in nodes_list.values()]
    z_node = [node.pos[2] for node in nodes_list.values()]

    # Flags to avoid multiple labels
    label_clamped_shown = False
    label_lumped_shown = False

    for node in nodes_list.values():
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
    text_color = 'white' if github else 'black'
    for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
        ax.text(x_val, y_val, z_val, f'{i}', fontsize=9, ha='center', color=text_color)

    # Beams (elements) display
    for elem in elems_list:
        idx_in = nodes_list[elem.nodes[0]].idx
        idx_out = nodes_list[elem.nodes[1]].idx
        x = [nodes_list[idx_in].pos[0], nodes_list[idx_out].pos[0]]
        y = [nodes_list[idx_in].pos[1], nodes_list[idx_out].pos[1]]
        z = [nodes_list[idx_in].pos[2], nodes_list[idx_out].pos[2]]
        ax.plot(x, y, z, color="blue", linewidth=1.5, alpha=0.5)

    # Set plot labels and adjust for GitHub if needed
    label_color = 'white' if github else 'black'
    ax.set_xlabel("X-axis", fontsize=12, color=label_color)
    ax.set_ylabel("Y-axis", fontsize=12, color=label_color)
    ax.set_zlabel("Z-axis", fontsize=12, color=label_color)

    # Adjust the limits based on the node coordinates
    ax.set_xlim([min(x_node) - 1, max(x_node) + 1])
    ax.set_ylim([min(y_node) - 1, max(y_node) + 1])
    ax.set_zlim([min(z_node) - 1, max(z_node) + 1])

    # Set transparent background only if github is True
    if github:
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
    
    ax.legend(loc="best")
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Adjust tick colors for GitHub
    if github:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        # Change legend text color to white
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_color("white")

    if save:
        if github:
            plt.savefig('part1/Pictures/structure.png', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('part1/Pictures/structure.PDF')


def plotModes(fig, ax, nodes_list, displacements, elems_list, nodes_clamped, save, github):
    # Remove clamped index nodes
    unclamped_nodes_list = [x for x in nodes_list.keys() if x not in nodes_clamped]
    mask = np.arange(0, 6*len(unclamped_nodes_list), 1)
    mask = mask[::6]
    
    coef = 2000 * np.max(np.abs(displacements))
    
    # Calculate displacement magnitudes
    disp_magnitudes = {}
    for idx, i in enumerate(unclamped_nodes_list):
        disp = coef * displacements[mask[idx]:mask[idx]+3]
        nodes_list[i].pos += disp
        disp_magnitudes[i] = np.linalg.norm(disp)
    
    # Create color map
    color_map = cm.get_cmap('inferno')
    min_disp = min(disp_magnitudes.values())
    max_disp = max(disp_magnitudes.values())
    
    # Plot elements with color gradient
    for elem in elems_list:
        n1 = nodes_list[elem.nodes[0]].idx
        n2 = nodes_list[elem.nodes[1]].idx
        x = [nodes_list[n1].pos[0], nodes_list[n2].pos[0]]
        y = [nodes_list[n1].pos[1], nodes_list[n2].pos[1]]
        z = [nodes_list[n1].pos[2], nodes_list[n2].pos[2]]
        
        # Calculate average displacement for the element
        avg_disp = (disp_magnitudes.get(n1, 0) + disp_magnitudes.get(n2, 0)) / 2
        
        # Normalize displacement and get color
        if max_disp > min_disp:
            norm_disp = (avg_disp - min_disp) / (max_disp - min_disp)
        else:
            norm_disp = 0
        color = color_map(norm_disp)
        
        ax.plot(x, y, z, color=color, linewidth=2)
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=min_disp/coef, vmax=max_disp/coef))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Displacement [m]')
    
    # Set transparent background only if github is True
    if github:
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
    
    # Adjust grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set labels, ticks, and colorbar to white only if github is True
    if github:
        ax.set_xlabel('X [m]', color='white')
        ax.set_ylabel('Y [m]', color='white')
        ax.set_zlabel('Z [m]', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Displacement [m]', color='white')
    else:
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

    if save:
        if github:
            plt.savefig('part1/Pictures/mode_shape.png', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('part1/Pictures/mode_shape.PDF')






    

  
    



