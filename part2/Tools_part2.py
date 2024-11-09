import matplotlib.pyplot as plt

def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF


def plotAll(time, variables, names, colors, line_styles, xlabel, ylabel, save, name):

    plt.figure(figsize=(10, 6))
    
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
        plt.savefig(f"part2/Pictures/{name}.PDF")

    plt.show()
    
    return plt.gcf()