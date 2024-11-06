import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import os
import sys
from scipy.sparse import linalg as splinalg
import scipy as sp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix, hstack, vstack, eye
from scipy.sparse.linalg import spsolve


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1 import FEM, set_parameters
from Tools_part3 import *

def partition_matrices(K, M, retained_dofs):

    n = K.shape[0]
    condensed_dofs = np.array(sorted(list(set(range(n)) - set(retained_dofs))))
    retained_dofs = np.array(sorted(retained_dofs))
    
    K_CC = K[np.ix_(condensed_dofs, condensed_dofs)] 
    K_CR = K[np.ix_(condensed_dofs, retained_dofs)]   
    K_RC = K[np.ix_(retained_dofs, condensed_dofs)]   
    K_RR = K[np.ix_(retained_dofs, retained_dofs)]   
    
    M_CC = M[np.ix_(condensed_dofs, condensed_dofs)]
    M_CR = M[np.ix_(condensed_dofs, retained_dofs)]
    M_RC = M[np.ix_(retained_dofs, condensed_dofs)]
    M_RR = M[np.ix_(retained_dofs, retained_dofs)]
    
    return (K_CC, K_CR, K_RC, K_RR), (M_CC, M_CR, M_RC, M_RR), condensed_dofs, retained_dofs

def GuyanIronsReduction(K, M, retained_dofs):

    K_parts, M_parts, condensed_dofs, retained_dofs = partition_matrices(K, M, retained_dofs)
    K_CC, K_CR, K_RC, K_RR = K_parts
    M_CC, M_CR, M_RC, M_RR = M_parts

    K_CC_inv = sp.sparse.linalg.inv(K_CC.tocsc())    
    phi_s = -K_CC_inv @ K_CR
    
    n_r = len(retained_dofs)
    I = eye(n_r, format='csr')
    R = vstack([I, csr_matrix(phi_s)])
    
    K_red = (R.T @ K @ R).tocsr()
    M_red = (R.T @ M @ R).tocsr()
    
    eigen_values, eigen_vectors = eigsh(K_red, k=6, M=M_red, sigma=0, which='LM')
    frequencies = np.sqrt(np.abs(eigen_values)) / (2 * np.pi)
    idx = np.argsort(frequencies)
    frequencies = frequencies[idx]
    eigen_vectors = eigen_vectors[:, idx]
    
    return frequencies, eigen_vectors


def CraigBamptonReduction(K, M, retained_dofs, n_modes):

    n = K.shape[0]
    all_dofs = np.arange(n)
    condensed_dofs = np.setdiff1d(all_dofs, retained_dofs)
    
    perm = np.concatenate((retained_dofs, condensed_dofs))
    Kpp = K[np.ix_(perm, perm)]
    Mpp = M[np.ix_(perm, perm)]
    
    nr = len(retained_dofs)
    Krr = Kpp[:nr, :nr]
    Krc = Kpp[:nr, nr:]
    Kcr = Kpp[nr:, :nr]
    Kcc = Kpp[nr:, nr:]
    
    Mrr = Mpp[:nr, :nr]
    Mrc = Mpp[:nr, nr:]
    Mcr = Mpp[nr:, :nr]
    Mcc = Mpp[nr:, nr:]
    
    A = -linalg.spsolve(Kcc.tocsc(), Kcr)
    I = sp.eye(nr)
    
    # Calculer les modes normaux à interface fixe (Craig-Bampton)
    eigen_values, eigen_vectors = linalg.eigsh(Kcc.tocsc(), k=n_modes, M=Mcc.tocsc(), sigma=0, which='LM')
    
    # Normaliser les vecteurs propres
    for i in range(n_modes):
        eigen_vectors[:, i] /= np.sqrt(eigen_vectors[:, i].T @ Mcc @ eigen_vectors[:, i])
    
    # Construire la matrice de réduction R
    R = sp.bmat([[I, sp.csr_matrix((nr, n_modes))],
                 [A, eigen_vectors]])
    
    # Réduire les matrices
    K_reduced = R.T @ Kpp @ R
    M_reduced = R.T @ Mpp @ R
    
    # Résoudre le problème aux valeurs propres réduit
    eigen_values, eigen_vectors = linalg.eigsh(K_reduced, k=nr, M=M_reduced, sigma=0, which='LM')
    
    # Tri et conversion en Hz
    eigen_values = np.sqrt(np.sort(eigen_values.real)) / (2 * np.pi)
    order = np.argsort(eigen_values)
    sorted_eigen_vectors = eigen_vectors.real[:, order]
    
    return eigen_values, sorted_eigen_vectors

