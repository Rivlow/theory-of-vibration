import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from scipy.signal import find_peaks
import numpy as np

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



def extractDOF(node_index, nodes_clamped):
    """Find DOF considering clamped nodes (not present in eigen_mode)."""
    clamped_before = sum(1 for n in nodes_clamped if n < node_index)
    adjusted_index = node_index - clamped_before
    return 6 * adjusted_index  # first DOF

def plotSingle(time, q, xlabel, ylabel, save, name, latex):
    # Configurer le style avant de créer la figure
    isLatex(latex)
    
    # Créer une figure avec un fond blanc
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111)

    # Plot principal
    ax.plot(time, q, 'navy', linewidth=1.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=MEDIUM_SIZE)  # Utiliser la taille définie
    ax.set_ylabel(ylabel, fontsize=MEDIUM_SIZE)  # Utiliser la taille définie
    
    # Configurer les ticks du plot principal
    ax.tick_params(axis='both', labelsize=MEDIUM_SIZE)

    # Créer le zoom inset avec une taille plus grande
    axins = inset_axes(ax,
                      "85%", "80%",
                      bbox_to_anchor=(0.45, 0.35, 0.5, 0.5),
                      bbox_transform=ax.transAxes,
                      loc='center')

    # Configurer le fond blanc avec une bordure grise
    axins.patch.set_facecolor('white')
    axins.patch.set_alpha(1.0)
    axins.patch.set_zorder(3)
    
    # Ajouter une bordure blanche plus large autour du zoom
    for spine in axins.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('white')
        spine.set_zorder(4)
    
    # Tracer le zoom
    axins.plot(time, q, 'navy', linewidth=1.5, zorder=5)
    
    # Configurer la grille du zoom
    axins.grid(True, linestyle='--', alpha=0.4, zorder=4)
    
    # Définir la région du zoom
    x1, x2 = 0, 1
    y1, y2 = 1.1*min(q), 1.1*max(q)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # Ajouter les lignes de connexion avec un style plus fin
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1, zorder=6)
    
    # Configurer les ticks du zoom avec la même taille que le plot principal
    axins.tick_params(axis='both', colors='navy', labelsize=MEDIUM_SIZE)
    for label in axins.get_xticklabels() + axins.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Ajouter une petite marge autour du zoom
    plt.subplots_adjust(right=0.95, top=0.95, bottom=0.1, left=0.1)

    if save:
        plt.savefig(f"part2/Pictures/{name}.PDF", bbox_inches='tight', dpi=300)
    
    plt.show()

def plotFFT(xf, amplitude, save, name_save, latex, height_ratio=0.00001, prominence_ratio=0.000001, distance=5):

    # Configurer le style
    isLatex(latex)
    
    # Créer la figure
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Plot principal de la FFT avec ligne continue
    ax.plot(xf, amplitude, 'navy', linewidth=1)
    
    # Trouver les pics significatifs jusqu'à 20 Hz
    mask_20hz = xf <= 20
    xf_masked = xf[mask_20hz]
    amplitude_masked = amplitude[mask_20hz]
    
    # Calculer le seuil dynamique basé sur la moyenne locale
    window = len(amplitude_masked) // 10
    avg_amplitude = np.convolve(amplitude_masked, np.ones(window)/window, mode='same')
    threshold = avg_amplitude * 2
    
    peaks, _ = find_peaks(amplitude_masked, 
                         height=max(amplitude_masked) * height_ratio,  # Seuil paramétrable
                         distance=distance,                           # Distance paramétrable
                         prominence=max(amplitude_masked) * prominence_ratio)  # Prominence paramétrable
    
    peak_freqs = xf_masked[peaks]
    peak_amps = amplitude_masked[peaks]
    
    # Ajouter les points rouges sur les pics
    ax.scatter(peak_freqs, peak_amps, color='red', s=25, zorder=5)
    
    # Annoter uniquement les pics les plus importants
    for freq, amp in zip(peak_freqs, peak_amps):
        if amp > max(amplitude_masked) * height_ratio:
            ax.annotate(f'{freq:.2f} Hz',
                       xy=(freq, amp),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=SMALL_SIZE)
    
    # Configuration des axes
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency [Hz]', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Fourier transform magnitude [mm]', fontsize=MEDIUM_SIZE)
    
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=MEDIUM_SIZE)
    ax.set_xlim(0, 20)
    ax.set_ylim(min(amplitude)/2, max(amplitude)*2)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"part2/Pictures/{name_save}.PDF", bbox_inches='tight')
    
    plt.show()

def plotAll(time, variables, names, colors, line_styles, xlabel, ylabel, save, name_save):

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
        plt.savefig(f"part2/Pictures/{name_save}.PDF")

    plt.show()
    
    return plt.gcf()