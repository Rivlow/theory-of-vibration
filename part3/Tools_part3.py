import os
import sys
import matplotlib.pyplot as plt
import time

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part2.Tools_part2 import extractDOF
from part2.Newmark import *
from part3.reduction_method import *
from part1 import FEM


SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def isLatex(latex):
    if latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')    


dof_map = {"u": 0,
           "v": 1,
           "w": 2,
           "phi_u": 3,
           "phi_v": 4,
           "phi_w": 5}


def retainedDOF(nodes_retained, node_dof_config, nodes_clamped):

    retained_dofs = []

    for node in nodes_retained:
        DOF_init = extractDOF(node, nodes_clamped)
        for dof in node_dof_config[node]:
            retained_dofs.append(DOF_init + dof_map[dof])

    return retained_dofs

def findReducedDOF(node_idx, nodes_retained, node_dof_config):

    if node_idx not in nodes_retained:
        return None
        
    position = 0
    for node in nodes_retained:
        if node == node_idx:
            return position
        position += len(node_dof_config[node])
    return None


def displayRetained(nodes_list, retained_nodes, elems_list, geom_data, save, latex):
    fig = plt.figure(figsize=(10, 8), facecolor='none', edgecolor='none')
    ax = fig.add_subplot(projection='3d')
   
    isLatex(latex)
    
    # Nodes display
    x_node = [node.pos[0] for node in nodes_list.values()]
    y_node = [node.pos[1] for node in nodes_list.values()]
    z_node = [node.pos[2] for node in nodes_list.values()]

    # Flags to avoid multiple labels
    label_retained_shown = False

    # Display nodes
    for node in nodes_list.values():
        if node.idx in retained_nodes:
            if not label_retained_shown:
                ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="red", s=70, label="retained", depthshade=True)
                label_retained_shown = True
            else:
                ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="red", s=70, depthshade=True)
        else:
            ax.scatter(node.pos[0], node.pos[1], node.pos[2], c="black", s=20, depthshade=True)

    # Annotate nodes with their index
    text_color = 'black'
    for i, (x_val, y_val, z_val) in enumerate(zip(x_node, y_node, z_node)):
        ax.text(x_val, y_val, 1.03*z_val, f'{i}', fontsize=14, ha='center', color=text_color)

    # Beams (elements) display
    for elem in elems_list:
        idx_in = nodes_list[elem.nodes[0]].idx
        idx_out = nodes_list[elem.nodes[1]].idx
        x = [nodes_list[idx_in].pos[0], nodes_list[idx_out].pos[0]]
        y = [nodes_list[idx_in].pos[1], nodes_list[idx_out].pos[1]]
        z = [nodes_list[idx_in].pos[2], nodes_list[idx_out].pos[2]]
        ax.plot(x, y, z, color="black", linewidth=1.5, alpha=0.5)

    # Set plot labels and adjust for GitHub if needed
    label_color = 'black'

    # Adjust the limits based on the node coordinates
    ax.set_xlim([min(x_node) - 1, max(x_node) + 1])
    ax.set_ylim([min(y_node) - 1, max(y_node) + 1])
    ax.set_zlim([min(z_node) - 1, max(z_node) + 1])    
    
    ax.legend(loc="best")
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)   

    if save:
        plt.savefig('part3/Pictures/structure_retained.PDF')
    
    plt.show()

def compareFullGIFreq(freq_init, freq_gi):

    rel_errors = abs((freq_gi - freq_init) / freq_init * 100)
    
    print("\n{:<5} {:<15} {:<15} {:<15}".format("Mode", "Full (Hz)", "Guyan-Iron (Hz)", "Error (%)"))
    print("-" * 50)
    
    for i in range(len(freq_init)):
        print("{:<5d} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            i+1,
            freq_init[i],
            freq_gi[i],
            rel_errors[i]
        ))

def compareFullCBFreq(freq_init, freq_cb):

    rel_errors = abs((freq_cb - freq_init) / freq_init * 100)
    
    print("\n{:<5} {:<15} {:<15} {:<15}".format("Mode", "Full (Hz)", "Craig-Bampton (Hz)", "Error (%)"))
    print("-" * 50)
    
    for i in range(len(freq_init)):
        print("{:<5d} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            i+1,
            freq_init[i],
            freq_cb[i],
            rel_errors[i]
        ))

def convergenceCB(freq_ref, K_parts, M_parts, C_parts, condensed_dofs, n_interface_modes_range, save, latex):

    n_eigen = len(freq_ref)
    errors_all = []
    
    for n_interface in n_interface_modes_range:
        eigen_values = CraigBamptonReduction(K_parts, M_parts, C_parts, condensed_dofs,
                                           n_interface_modes=n_interface, n_eigen=n_eigen)[0]
        
        errors = np.abs(freq_ref - eigen_values) / freq_ref * 100
        errors_all.append(errors)
    
    plt.figure(figsize=(10, 6))
    isLatex(latex)
    
    for i in range(len(freq_ref)):
        plt.plot(n_interface_modes_range,
                [errors[i] for errors in errors_all],
                '-o',
                label=f'Mode {i+1}')
    
    plt.axhline(y=1, color='gray', linestyle='--', label=r'1[$\%$] threshold')
    
    convergence_point = None
    for n_modes, errors in enumerate(errors_all):
        if all(error < 1 for error in errors):
            convergence_point = n_modes
            break
    
    if convergence_point is not None:
        plt.axvline(x=convergence_point, color='gray', linestyle=':')
        plt.annotate('',
                    xy=(convergence_point, 1),
                    xytext=(convergence_point+1, 2),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.grid(True)
    plt.xlabel('Number of interface modes [-]')
    plt.ylabel(r'Relative error [$\%$]')
    plt.xlim(0, len(errors_all)-1)
    plt.xticks(n_interface_modes_range)
    plt.legend()
    
    if save:
        plt.savefig('part3/Pictures/convergence_CB.PDF')
    plt.show()

def plotFrequenciesComparison(freq_init, freq_cb, freq_gi, save, latex):

    modes = np.arange(1, len(freq_init) + 1)
    isLatex(latex)
    
    plt.figure(figsize=(10, 8))
    plt.plot(modes, freq_init, 's', label='EXACT', color='b', markersize=8, 
             markerfacecolor='none') 
    plt.plot(modes, freq_cb, 'rx', label='CRAIG-BAMPTON', markersize=8)
    plt.plot(modes, freq_gi, 'ko', label='GUYAN', markersize=8,
             markerfacecolor='none') 
    
    plt.xlabel('Mode number [-]')
    plt.ylabel('Frequency [Hz]')
    plt.grid(True)
    
    plt.xlim(0, max(modes) + 1)
    plt.ylim(0, max(max(freq_init), max(freq_cb), max(freq_gi)) * 1.1)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save:
        plt.savefig(f'part3/Pictures/comparison_freq_full_gi_cb.PDF')
    plt.show()



def compareTimeReductionMethod(t_span, sim_data, K, M, C, K_parts, M_parts, C_parts, condensed_dofs, 
                               retained_dofs,  F, x0, v0, n_modes, 
                               geom_data, phys_data, elem_per_beam_range, save, latex):
    isLatex(latex)

    time_full = []
    time_gi = []
    time_cb = []
    for elem_per_beam in elem_per_beam_range:

        # Full system
        nodes_list_init, nodes_pairs_init = FEM.initializeGeometry(geom_data, phys_data)
        nodes_list, nodes_pairs = FEM.addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam-1)
        elems_list = FEM.createElements(nodes_pairs, nodes_list, geom_data, phys_data)
        solver = FEM.Solver()
        solver.assembly(elems_list, nodes_list, geom_data["nodes_clamped"])
        solver.addLumpedMass(nodes_list, geom_data["nodes_lumped"])
        solver.removeClampedNodes(nodes_list, geom_data["nodes_clamped"])
        t_full_init = time.time()
        freq_init, modes_init = solver.solve(n_modes)
        x_init, v_init, a_init = NewmarkIntegration(M, C, K, x0, v0, t_span, F, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
        t_full_end = time.time()
        time_full.append(t_full_end-t_full_init)

        # Reduction method
        freq_gi, modes_gi, K_gi, M_gi, C_gi, R_gi, F_gi, x0_gi, v0_gi = GuyanIronsReduction(K_parts, M_parts, C_parts, retained_dofs, F, x0, v0, n_modes)
        t_gi_init = time.time()
        x_gi, v_gi, a_gi = NewmarkIntegration(M_gi, C_gi, K_gi, x0_gi, v0_gi, t_span, F_gi, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
        t_gi_end = time.time()
        time_gi.append(t_gi_end-t_gi_init)

        
        freq_cb, modes_cb, K_cb, M_cb, C_cb, R_cb, F_cb, x0_cb, v0_cb = CraigBamptonReduction(K_parts, M_parts, C_parts, retained_dofs, condensed_dofs, 
                                                                                              F, x0, v0, n_interface_modes=5, n_eigen=n_modes)
        t_cb_init = time.time()
        x_cb, v_cb, a_cb = NewmarkIntegration(M_cb, C_cb, K_cb, x0_cb, v0_cb, t_span, F_cb, sim_data["newmark"]["gamma"], sim_data["newmark"]["beta"])
        t_cb_end = time.time()
        time_cb.append(t_cb_end-t_cb_init)



    plt.plot(elem_per_beam_range, time_full, 's', label='EXACT', color='b', markersize=8, 
             markerfacecolor='none')
    plt.plot(elem_per_beam_range, time_gi, 'rx', label='CRAIG-BAMPTON', markersize=8)
    plt.plot(elem_per_beam_range, time_cb, 'ko', label='GUYAN', markersize=8,
             markerfacecolor='none') 
    plt.xlabel("Number of elements per beam [-]")
    plt.ylabel("Total computation time [s]")
    plt.legend(loc='best')


    if save:
        plt.savefig(f'part3/Pictures/comparison_time_full_gi_cb.PDF')
    plt.show()

def plotAll3(time, variables, names, colors, line_styles, xlabel, ylabel, save, latex, name_save):

    isLatex(latex)
    fig, ax = plt.subplots(1, 1, figsize=(6.77, 4))
    
    if not (len(variables) == len(names) == len(colors) == len(line_styles)):
        raise ValueError("All entry must have same size")
    
    for var, name, color, ls in zip(variables, names, colors, line_styles):
        plt.plot(time, var, 
                 linestyle=ls, 
                 color=color, 
                 label=name,
                 linewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    
    plt.tight_layout()
    if save:
        plt.savefig(f"part3/Pictures/{name_save}.PDF")

    plt.show()
    
    return plt.gcf()

def compute_MAC(modes_full, modes_reduced, R, retained_dofs, condensed_dofs, name, save, latex):
    modes_reduced_full = R @ modes_reduced    
    
    modes_full_reordered = np.vstack([
        modes_full[retained_dofs, :],  
        modes_full[condensed_dofs, :]
    ])
    
    n_modes_full = modes_full.shape[1]
    n_modes_reduced = modes_reduced.shape[1]
    MAC = np.zeros((n_modes_reduced, n_modes_full))
    
    # Compute MAC values
    for i in range(n_modes_reduced):
        for j in range(n_modes_full):
            numerator = np.abs(modes_full_reordered[:, j].T @ modes_reduced_full[:, i]) ** 2
            denominator = (modes_full_reordered[:, j].T @ modes_full_reordered[:, j]) * \
                         (modes_reduced_full[:, i].T @ modes_reduced_full[:, i])
            MAC[i, j] = numerator / denominator
    
    plt.figure(figsize=(8, 6))
    isLatex(latex)
    plt.imshow(MAC, cmap='Greys', interpolation='none', aspect='equal', origin='lower')  # origin='lower' pour inverser l'axe y (utile eheh)
    plt.colorbar(label="Correlation")
    plt.ylabel("Approximated mode")
    plt.xlabel("Reference mode (full system)")
    plt.tight_layout()

    if save:
        plt.savefig(f'part3/Pictures/MAC_{name}.PDF')
    plt.show()
    
  