import matplotlib.pyplot as plt

def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF


def plotAll(t_span, *args, separate):
    if len(args) == 0:
        raise ValueError("At least 1 param needed to plot.")
    
    param_names = ['displacement method', 'acceleration method', 'Newmark']
    params = dict(zip(param_names[:len(args)], args))
    
    styles = [
        {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
        {'color': 'red', 'linestyle': '--', 'linewidth': 2.5},
        {'color': 'green', 'linestyle': ':', 'linewidth': 3}
    ]
    
    if separate:
        fig, axs = plt.subplots(len(params), 1, figsize=(12, 5 * len(params)))
        
        if len(params) == 1:
            axs = [axs]
        
        for i, (name, values) in enumerate(params.items()):
            axs[i].plot(t_span, values, **styles[i])
            axs[i].set_title(name)
            axs[i].set_xlabel('Time t [s]')
            axs[i].set_ylabel('displacement [m]')
        
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    else:
        plt.figure(figsize=(12, 6))
        for (name, values), style in zip(params.items(), styles):
            plt.plot(t_span, values, label=name, **style)
        plt.xlabel('Time t [s]')
        plt.ylabel('displacement [m]')
        plt.legend(loc="best")
        plt.tight_layout()
    
    plt.subplots_adjust(hspace=0.4)
    plt.show()