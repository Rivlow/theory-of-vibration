import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
pi = np.pi

from FEM import *
from Tools import *


def setParams():

    # Geometrical values
    z_min, z_mid, z_max = 1, 3, 5 # [m]
    x_min, x_mid, x_max = 0, 4, 8 # [m]
    y_min, y_max = 0, 2 # [m]
    D = 150*(10**-3) # [m]
    e = 5*(10**-3) # [m]
    d = D - 2*e
    A = pi*(D**2 - d**2)/4 # [m^2]
    Iz = Iy = pi*(D**4 - d**4)/64 # [m^4]
    Jx = 2*Iy # [m^4]

    # Node indices on which supporters are sitting
    nodes_lumped = [4, 5, 21, 22, 23, 24,
                    25, 26, 10, 11, 27, 28, 
                    29, 30, 31, 32, 14, 15]

    # Node indices attached on the ground
    nodes_clamped = [0, 1, 6, 7, 12, 13]

    # Physical values
    M_lumped = 51*80/18 # [kg]
    rho = 7800 # [kg/m^3]
    nu = 0.3 # [-]
    E = 210*(10**9) #  [Pa]
    G = E/(2*(1+nu)) # [Pa]

    phys_data = {"rho":rho, "nu":nu, "E":E, "G":G, "M_lumped":M_lumped}
    geom_data = {"z_min":z_min, "z_mid":z_mid, "z_max":z_max,
                    "x_min":x_min, "x_mid":x_mid, "x_max":x_max,
                    "y_min":y_min, "y_max":y_max,
                    "D":D, "e":e, "A":A, 
                    "Iz":Iz, "Iy":Iy, "Jx":Jx,
                    "nodes_lumped":nodes_lumped,
                    "nodes_clamped":nodes_clamped}

    return geom_data, phys_data
