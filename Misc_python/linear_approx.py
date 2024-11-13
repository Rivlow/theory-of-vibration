import numpy as np
import matplotlib.pyplot as plt


def isLatex(latex):
    if latex:
        
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
        plt.rc('text', usetex=True)
        plt.rc('font', family='lmodern')     


def linear_approximation(f, x_min, x_max, n_points):
   
    x_approx = np.linspace(x_min, x_max, n_points)
    y_approx = f(x_approx)
    
    x_fine = np.linspace(x_min, x_max, 1000)
    y_fine = np.zeros_like(x_fine)
    
    for i in range(len(x_fine)):

        idx = np.searchsorted(x_approx, x_fine[i]) - 1
        idx = max(0, min(idx, len(x_approx)-2))
        
        x1, x2 = x_approx[idx], x_approx[idx+1]
        y1, y2 = y_approx[idx], y_approx[idx+1]
        y_fine[i] = y1 + (y2-y1)*(x_fine[i]-x1)/(x2-x1)
    
    return x_fine, y_fine, x_approx, y_approx

isLatex(True)

f = lambda x: np.tanh(x)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
axes = [ax1, ax2, ax3, ax4]
n_points_list = [2, 4, 8, 16]

x_exact = np.linspace(-3, 3, 1000)
y_exact = f(x_exact)

for ax, n_points in zip(axes, n_points_list):

    x_fine, y_fine, x_approx, y_approx = linear_approximation(f, -3, 3, n_points)
    
    ax.plot(x_exact, y_exact, '--', color='gray', label='Exact')
    ax.plot(x_fine, y_fine, '-', color='blue', label='Approximation')
    ax.plot(x_approx, y_approx, 'o', color='red', label='Points')
    
    ax.set_title(f'n = {n_points} points')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig("Misc_python/Pictures/linear_approx.PDF")
plt.show()