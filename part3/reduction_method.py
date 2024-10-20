import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import os
import sys
from scipy.sparse import linalg as splinalg
import scipy as sp
from scipy.sparse import linalg


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part1 import FEM, set_parameters
from Tools_part3 import *


def GuyanIronsReduction(K, M, retained_dofs):
    n = K.shape[0]
    all_dofs = np.arange(n)
    condensed_dofs = np.setdiff1d(all_dofs, retained_dofs)
    
    # Réorganiser les matrices
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

    A = -sp.sparse.linalg.inv(Kcc.tocsc()) @ Kcr
    I = sparse.eye(len(retained_dofs))
    R = sparse.bmat([[I], [A]])

    K_reduced = R.T @ K @ R
    M_reduced = R.T @ M @ R

    eigen_values, eigen_vectors = sp.sparse.linalg.eigsh(K_reduced, k=6, M=M_reduced, sigma=0, which='LM')

    # Sorting (by increasing values)
    eigen_values = np.sqrt(np.sort(eigen_values.real)) / (2 * np.pi)
    order = np.argsort(eigen_values)
    sorted_eigen_vectors = eigen_vectors.real[:, order]

    return eigen_values, sorted_eigen_vectors

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

