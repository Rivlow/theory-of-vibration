import numpy as np
from scipy.sparse import linalg, csr_matrix, hstack, vstack, eye
from scipy.sparse.linalg import eigsh, spsolve, inv
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


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

def GuyanIronsReduction(K_parts, M_parts, C_parts, retained_dofs):

    K_CC, K_CR, K_RC, K_RR = K_parts
    M_CC, M_CR, M_RC, M_RR = M_parts
    C_CC, C_CR, C_RC, C_RR = C_parts
    
    # Rearrange K, M and C
    K_top = hstack([K_RR, K_RC]) 
    K_bot = hstack([K_CR, K_CC])
    K_tild = vstack([K_top, K_bot])

    M_top = hstack([M_RR, M_RC]) 
    M_bot = hstack([M_CR, M_CC])
    M_tild = vstack([M_top, M_bot])

    C_top = hstack([C_RR, C_RC]) 
    C_bot = hstack([C_CR, C_CC])
    C_tild = vstack([C_top, C_bot])

    # Compute transformation matrix R
    K_CC_inv = inv(K_CC.tocsc())    
    phi_s = -K_CC_inv @ K_CR
    n_r = len(retained_dofs)
    I = eye(np.shape(phi_s)[1], format='csr')
    R = vstack([I, phi_s])
    
    # Reduce K and M
    K_red = R.T @ K_tild @ R
    M_red = R.T @ M_tild @ R
    C_red = R.T @ C_tild @ R
    
    # Compute reduced eigen_values
    eigen_values, eigen_vectors = eigsh(K_red, k=6, M=M_red, sigma=0, which='LM')
    frequencies = np.sqrt(np.abs(eigen_values)) / (2 * np.pi)
    
    return frequencies, eigen_vectors, K_red, M_red, C_red, R

def CraigBamptonReduction(K_parts, M_parts, C_parts, condensed_dofs, n_modes):

    K_CC, K_CR, K_RC, K_RR = K_parts
    M_CC, M_CR, M_RC, M_RR = M_parts
    C_CC, C_CR, C_RC, C_RR = C_parts

    # Rearrange K and M 
    K_top = hstack([K_RR, K_RC]) 
    K_bot = hstack([K_CR, K_CC])
    K_tild = vstack([K_top, K_bot])

    M_top = hstack([M_RR, M_RC]) 
    M_bot = hstack([M_CR, M_CC])
    M_tild = vstack([M_top, M_bot])

    C_top = hstack([C_RR, C_RC]) 
    C_bot = hstack([C_CR, C_CC])
    C_tild = vstack([C_top, C_bot])

    
    static_eigvals, static_eigvect = linalg.eigsh(K_CC.tocsc(), k=n_modes, M=M_CC.tocsc(), sigma=0, which='LM')
    
    # Compute transformation matrix R
    n_c = len(condensed_dofs)
    n_r = K_RR.shape[0]
    K_CC_inv = inv(K_CC.tocsc())    
    phi_s = -K_CC_inv @ K_CR

    I = eye(n_r, format='csr')
    O = csr_matrix((n_r, n_modes))
    R_top = hstack([I, O])
    R_bot = hstack([phi_s, static_eigvect[:, :n_modes]])
    R = vstack([R_top, R_bot])

    K_red = R.T @ K_tild @ R
    M_red = R.T @ M_tild@ R
    C_red = R.T @ C_tild @ R
    
    eigen_values, eigen_vectors = linalg.eigsh(K_red, k=n_modes, M=M_red, sigma=0, which='LM')
    eigen_values = np.sqrt(np.sort(eigen_values.real)) / (2 * np.pi)
    
    return eigen_values, eigen_vectors, K_red, M_red, C_red, R
    

