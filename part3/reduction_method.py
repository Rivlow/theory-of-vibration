import numpy as np
from scipy.sparse import linalg, csr_matrix, hstack, vstack, eye
from scipy.sparse.linalg import eigsh, spsolve, inv
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

SMALL_SIZE = 12
MEDIUM_SIZE = 18
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



def partition_matrices(K, M, C, retained_dofs):
    

    n = K.shape[0]
    condensed_dofs = np.array(sorted(list(set(range(n)) - set(retained_dofs))))
    retained_dofs = np.array(sorted(retained_dofs))
    
    K_CC = K[condensed_dofs][:, condensed_dofs]
    K_CR = K[condensed_dofs][:, retained_dofs]
    K_RC = K[retained_dofs][:, condensed_dofs]
    K_RR = K[retained_dofs][:, retained_dofs]

    M_CC = M[condensed_dofs][:, condensed_dofs]
    M_CR = M[condensed_dofs][:, retained_dofs]
    M_RC = M[retained_dofs][:, condensed_dofs]
    M_RR = M[retained_dofs][:, retained_dofs]

    C_CC = C[condensed_dofs][:, condensed_dofs]
    C_CR = C[condensed_dofs][:, retained_dofs]
    C_RC = C[retained_dofs][:, condensed_dofs]
    C_RR = C[retained_dofs][:, retained_dofs]
  
    return (K_CC, K_CR, K_RC, K_RR), (M_CC, M_CR, M_RC, M_RR), (C_CC, C_CR, C_RC, C_RR), condensed_dofs, retained_dofs

def GuyanIronsReduction(K_parts, M_parts, C_parts, retained_dofs, F, x0, v0, n_modes):

    K_CC, K_CR, K_RC, K_RR = K_parts
    M_CC, M_CR, M_RC, M_RR = M_parts
    C_CC, C_CR, C_RC, C_RR = C_parts
    
    # Rearrange matrices
    K_tild = vstack([hstack([K_RR, K_RC]), hstack([K_CR, K_CC])])
    M_tild = vstack([hstack([M_RR, M_RC]), hstack([M_CR, M_CC])])
    C_tild = vstack([hstack([C_RR, C_RC]), hstack([C_CR, C_CC])])
    
    # Compute transformation matrix
    phi_s = -inv(K_CC.tocsc()) @ K_CR
    R = vstack([eye(len(retained_dofs), format='csr'), phi_s])
    
    # Reduce system matrices
    K_red = R.T @ K_tild @ R
    M_red = R.T @ M_tild @ R
    C_red = R.T @ C_tild @ R
    
    # Rearrange F, x0, v0
    F_red = F[retained_dofs, :]
    x0_red = x0[retained_dofs]
    v0_red = v0[retained_dofs]
    
    # Compute eigenvalues
    eigen_values, eigen_vectors = eigsh(K_red, k=n_modes, M=M_red, sigma=0, which='LM')
    frequencies = np.sqrt(np.abs(eigen_values)) / (2 * np.pi)
    
    return frequencies, eigen_vectors, K_red, M_red, C_red, R, F_red, x0_red, v0_red

def CraigBamptonReduction(K_parts, M_parts, C_parts, retained_dofs, condensed_dofs, F, x0, v0, n_interface_modes, n_eigen):

    K_CC, K_CR, K_RC, K_RR = K_parts
    M_CC, M_CR, M_RC, M_RR = M_parts
    C_CC, C_CR, C_RC, C_RR = C_parts
    
    # Rearrange matrices
    n_retained = len(retained_dofs)
    n_interface = n_interface_modes
    K_tild = vstack([hstack([K_RR, K_RC]), hstack([K_CR, K_CC])])
    M_tild = vstack([hstack([M_RR, M_RC]), hstack([M_CR, M_CC])])
    C_tild = vstack([hstack([C_RR, C_RC]), hstack([C_CR, C_CC])])
    
    # Compute modes and transformation matrix
    phi_m = csr_matrix((K_CC.shape[0], 0)) if n_interface == 0 else \
            linalg.eigsh(K_CC.tocsc(), k=n_interface, M=M_CC.tocsc(), sigma=0, which='LM')[1]
    phi_s = -inv(K_CC.tocsc()) @ K_CR
    R = vstack([hstack([eye(n_retained, format='csr'), csr_matrix((n_retained, n_interface))]),
                hstack([phi_s, phi_m])])
    
    # Reduce system matrices
    K_red = R.T @ K_tild @ R
    M_red = R.T @ M_tild @ R
    C_red = R.T @ C_tild @ R

    # Rearrange F, x0 and v0
    F_red = np.vstack([F[retained_dofs, :], phi_m.T @ F[condensed_dofs, :] if n_interface > 0 else np.array([]).reshape(0, F.shape[1])])
    x0_red = np.concatenate([x0[retained_dofs], phi_m.T @ x0[condensed_dofs] if n_interface > 0 else []])
    v0_red = np.concatenate([v0[retained_dofs], phi_m.T @ v0[condensed_dofs] if n_interface > 0 else []])
    
    eigen_values, eigen_vectors = linalg.eigsh(K_red, k=min(n_eigen, K_red.shape[0]), M=M_red, sigma=0, which='LM')
    frequencies = np.sqrt(np.sort(eigen_values.real)) / (2 * np.pi) 
    
    return frequencies, eigen_vectors, K_red, M_red, C_red, R, F_red, x0_red, v0_red


    





    

