import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from scipy.signal import find_peaks
import numpy as np

import part2.mode_method as mode_method

SMALL_SIZE = 8
MEDIUM_SIZE = 14
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



def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF

def plotSingle(time, q, xlabel, ylabel, save, name, latex):

    isLatex(latex)
    
    fig = plt.figure(figsize=(6.77, 4), facecolor='white')
    ax = fig.add_subplot(111)

    ax.plot(time, q, 'navy', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=MEDIUM_SIZE)
    ax.set_ylabel(ylabel, fontsize=MEDIUM_SIZE)  
    
    ax.tick_params(axis='both', labelsize=MEDIUM_SIZE)

    axins = inset_axes(ax,
                      "85%", "80%",
                      bbox_to_anchor=(0.45, 0.35, 0.5, 0.5),
                      bbox_transform=ax.transAxes,
                      loc='center')

    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(1.0)
    axins.patch.set_zorder(3)
    
    for spine in axins.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('white')
        spine.set_zorder(4)
    
    axins.plot(time, q, 'navy', linewidth=1.5, zorder=5)
    axins.grid(True, linestyle='--', alpha=0.4, zorder=4)
    
    x1, x2 = 0, 1
    y1, y2 = 1.1*min(q), 1.1*max(q)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, zorder=6)
    
    axins.tick_params(axis='both', colors='navy', labelsize=MEDIUM_SIZE)
    for label in axins.get_xticklabels() + axins.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Ajouter une petite marge autour du zoom
    plt.subplots_adjust(right=0.95, top=0.95, bottom=0.1, left=0.1)

    if save:
        plt.savefig(f"part2/Pictures/{name}.PDF", bbox_inches='tight', dpi=300)
    
    plt.show()

def plotFFT(xf, amplitude, save, name_save, latex, height_ratio=0.00001, prominence_ratio=0.000001, distance=5):

    isLatex(latex)
    
    fig = plt.figure(figsize=(6.77, 4), facecolor='white')
    ax = fig.add_subplot(111)
    
    ax.plot(xf, amplitude, 'navy', linewidth=1)
    
    mask_20hz = xf <= 20
    xf_masked = xf[mask_20hz]
    amplitude_masked = amplitude[mask_20hz]
    
    window = len(amplitude_masked) // 10
    avg_amplitude = np.convolve(amplitude_masked, np.ones(window)/window, mode='same')
    threshold = avg_amplitude * 2
    
    peaks, _ = find_peaks(amplitude_masked, 
                         height=max(amplitude_masked) * height_ratio, 
                         distance=distance,                          
                         prominence=max(amplitude_masked) * prominence_ratio) 
    
    peak_freqs = xf_masked[peaks]
    peak_amps = amplitude_masked[peaks]
    
    ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5)
    
    for freq, amp in zip(peak_freqs, peak_amps):
        if amp > max(amplitude_masked) * height_ratio:
            ax.annotate(f'{freq:.2f} Hz',
                       xy=(freq, amp),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency [Hz]', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Fourier transform magnitude [m]', fontsize=MEDIUM_SIZE)
    
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=MEDIUM_SIZE)
    ax.set_xlim(0, 20)
    ax.set_ylim(min(amplitude)/2, max(amplitude)*2)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"part2/Pictures/{name_save}.PDF", bbox_inches='tight')
    
    plt.show()

def plotAll(time, variables, names, colors, line_styles, xlabel, ylabel, save, name_save):

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
        plt.savefig(f"part2/Pictures/{name_save}.PDF")

    plt.show()
    
    return plt.gcf()

def mNorm(eta, mu, n_modes):
    return np.sum(np.power(eta[:n_modes], 2) * mu[:n_modes, np.newaxis], axis=0)
    
def convergenceAmplitude(eta, modes, frequencies, K, phi, F, t_span, n_modes, z_dir, name, save, latex):

    q_full, q_acc_full = [], []
    fig, ax = plt.subplots(1, 1, figsize=(6.77, 4))
    isLatex(latex)
    
    for i, span_mode in enumerate(range(1, n_modes)):
        q = mode_method.modeDisplacementMethod(eta, modes, span_mode)[z_dir,:]
        q_acc = mode_method.modeAccelerationMethod(t_span, eta, 2*np.pi*frequencies, modes, K, phi, F, span_mode)[z_dir,:]
        q_full.append(np.max(q))
        q_acc_full.append(np.max(q_acc))
    
  
    q_ref = q_full[-1]  
    q_acc_ref = q_acc_full[-1] 
    
    err_cumul_q = [np.abs(q - q_ref)/np.abs(q_ref) * 100 for q in q_full]
    err_cumul_acc = [np.abs(q - q_acc_ref)/np.abs(q_acc_ref) * 100 for q in q_acc_full]

   
    ax.scatter(range(1, len(err_cumul_q)-1), err_cumul_q[:-2], label="displacement method")
    ax.scatter(range(1, len(err_cumul_acc)-1), err_cumul_acc[:-2], label="acceleration method")
    ax.plot(range(1, len(err_cumul_q)-1), err_cumul_q[:-2], ls="--")
    ax.plot(range(1, len(err_cumul_acc)-1), err_cumul_acc[:-2], ls="--")
    
    
    #ax.set_yscale('log') 
    ax.set_xticks(range(0, n_modes))
    ax.set_xlabel('Number of modes n[-]')
    ax.set_ylabel('Relative error [%]')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.legend()
    

    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'part2/Pictures/{name}.pdf')
    
    plt.show()

def convergenceShape(eta, modes, frequencies, K, phi, F, t_span, n_modes, z_dir, save, latex):
    q_full, q_acc_full = [], []
    
    fig, ax  = plt.subplots(1, 1, figsize=(15, 6))
    isLatex(latex)
    nb_modes = [1, 5, 8]
    
    for i, span_mode in enumerate(range(1, n_modes)):

        q = mode_method.modeDisplacementMethod(eta, modes, span_mode)[z_dir,:]
        q_acc = mode_method.modeAccelerationMethod(t_span, eta, 2*np.pi*frequencies, modes, K, phi, F, span_mode)[z_dir,:]
        q_full.append(np.max(q))
        q_acc_full.append(np.max(q_acc))

   

    var_q = [np.abs(q_full[i+1]-q_full[i])/q_full[i] for i in range(len(q_full)-1)]
    var_q_acc = [np.abs(q_acc_full[i+1]-q_acc_full[i])/q_acc_full[i] for i in range(len(q_acc_full)-1)]
        
    ax.scatter(range(1, len(q_full)), var_q, label="displacement method")
    ax.scatter(range(1, len(q_acc_full)), var_q_acc, label="acceleration method")
    ax.plot(range(1, len(q_full)), var_q, ls="--")
    ax.plot(range(1, len(q_acc_full)), var_q_acc, ls="--")
    ax.set_xticks(range(1, len(q_full)))
    ax.set_xlabel('Number of modes n[-]')
    ax.set_ylabel(r'$\frac{max(q_{n+1})-max(q_n)}{max(q_n)}$[%]')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    if save:
        plt.savefig('part2/Pictures/convergence.pdf')
    
    plt.show()
    
