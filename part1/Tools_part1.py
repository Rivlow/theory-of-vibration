import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def isLatex(latex):
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')      



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


def display(fig, ax, nodes_list, elems_list, geom_data, save, github, latex):
   
    isLatex(latex)
    
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
            ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="black", s=20, depthshade=True)

    # Annotate nodes with their index
    text_color = 'cyan' if github else 'black'
    for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
        ax.text(x_val, y_val, z_val, f'{i}', fontsize=9, ha='center', color=text_color)

    # Beams (elements) display
    for elem in elems_list:
        idx_in = nodes_list[elem.nodes[0]].idx
        idx_out = nodes_list[elem.nodes[1]].idx
        x = [nodes_list[idx_in].pos[0], nodes_list[idx_out].pos[0]]
        y = [nodes_list[idx_in].pos[1], nodes_list[idx_out].pos[1]]
        z = [nodes_list[idx_in].pos[2], nodes_list[idx_out].pos[2]]
        ax.plot(x, y, z, color="black", linewidth=1.5, alpha=0.5)

    # Set plot labels and adjust for GitHub if needed
    label_color = 'cyan' if github else 'black'
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
        ax.tick_params(axis='x', colors='cyan')
        ax.tick_params(axis='y', colors='cyan')
        ax.tick_params(axis='z', colors='cyan')
        
        # Change legend text color to cyan
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_color("cyan")

    if save:
        if github:
            plt.savefig('part1/Pictures/structure.png', transparent=True, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig('part1/Pictures/structure.PDF')


def plotModes(nodes_list, nb_modes, all_modes, elems_list, nodes_clamped, save, github, latex):
   
   isLatex(latex)

   unclamped_nodes_list = [x for x in nodes_list.keys() if x not in nodes_clamped]
   mask = np.arange(0, 6*len(unclamped_nodes_list), 1)
   mask = mask[::6]
   
   for idx_mode in range(6):
       fig = plt.figure(figsize=(10, 8), facecolor='none', edgecolor='none')
       ax = fig.add_subplot(111, projection='3d')
       
       current_nodes = {}
       for k, n in nodes_list.items():
           current_nodes[k] = type('Node', (), {
               'idx': n.idx,
               'pos': n.pos.copy()
           })
       
       modes = all_modes[:,idx_mode]
       coef = 2000 * np.max(np.abs(modes))
       
       disp_magnitudes = {}
       for idx, i in enumerate(unclamped_nodes_list):
           disp = coef * modes[mask[idx]:mask[idx]+3]
           current_nodes[i].pos += disp
           disp_magnitudes[i] = np.linalg.norm(disp)
       
       color_map = cm.get_cmap('jet')
       min_disp = min(disp_magnitudes.values())
       max_disp = max(disp_magnitudes.values())
       
       for elem in elems_list:
           n1 = current_nodes[elem.nodes[0]].idx
           n2 = current_nodes[elem.nodes[1]].idx
           x = [current_nodes[n1].pos[0], current_nodes[n2].pos[0]]
           y = [current_nodes[n1].pos[1], current_nodes[n2].pos[1]]
           z = [current_nodes[n1].pos[2], current_nodes[n2].pos[2]]
           
           avg_disp = (disp_magnitudes.get(n1, 0) + disp_magnitudes.get(n2, 0)) / 2
           
           if max_disp > min_disp:
               norm_disp = (avg_disp - min_disp) / (max_disp - min_disp)
           else:
               norm_disp = 0
           color = color_map(norm_disp)
           
           ax.plot(x, y, z, color=color, linewidth=2)
              
       if github:
           ax.set_facecolor('none')
       
       ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
       
       if github:
           ax.set_xlabel('X [m]', color='cyan')
           ax.set_ylabel('Y [m]', color='cyan')
           ax.set_zlabel('Z [m]', color='cyan')
           ax.tick_params(axis='x', colors='cyan')
           ax.tick_params(axis='y', colors='cyan')
           ax.tick_params(axis='z', colors='cyan')
           ax.title.set_color('cyan')
       else:
           ax.set_xlabel('X [m]')
           ax.set_ylabel('Y [m]')
           ax.set_zlabel('Z [m]')
       
       if github:
           fig.patch.set_alpha(0)
       
       if save:
           if github:
               plt.savefig(f'part1/Pictures/mode_shape_{idx_mode + 1}.png', transparent=True, bbox_inches='tight', pad_inches=0)
           else:
               plt.savefig(f'part1/Pictures/mode_shape_{idx_mode + 1}.pdf', bbox_inches='tight', pad_inches=0)
       
       plt.close(fig)





    

  
    



