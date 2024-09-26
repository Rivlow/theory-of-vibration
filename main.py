import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
pi = np.pi

from FEM import *
from Tools import *

def main():

    # Dictionnary to monitor outputs of the code
    activation = {"print":True, "plot":True}

    #------------------#
    #   Parameters     #
    #------------------#

    # Geometrical values
    z_min, z_mid, z_max = 1, 3, 5 # [m]
    x_min, x_mid, x_max = 0, 4, 8 # [m]
    y_min, y_max = 0, 2 # [m]
    D = 150e-3 # [m]
    e = 51e-3 # [m]
    A = pi*(D**2 - (D - e)**2)/4 # [m^2]
    Iz = Iy = pi*(D**4 - (D - e)**4)/64 # [m^4]
    Jx= 2*Iy # [m^4]

    nodes_lumped = [4, 5, 21, 22, 23, 24,
                    25, 26, 10, 11, 27, 28, 
                    29, 30, 31, 32, 14, 15]
    
    clamped_nodes = [0, 1, 6, 7, 12, 13]
    
    # Physical values
    M_lumped = 51*80/18 # [kg]
    rho = 7800 #  [kg/m^3]
    nu = 0.3 # [-]
    E = 2101e6 #  [Pa]
    G = E /(2*(1+nu)) # [Pa]

    phys_data = {"rho":rho, "nu":nu, "E":E, "G":G, "M_lumped":M_lumped}
    geom_data = {"z_min":z_min, "z_mid":z_mid, "z_max":z_max,
                 "x_min":x_min, "x_mid":x_mid, "x_max":x_max,
                 "y_min":y_min, "y_max":y_max,
                 "D":D, "e":e, "A":A, 
                 "Iz":Iz, "Iy":Iy, "Jx":Jx,
                 "nodes_lumped":nodes_lumped}

    #---------------#
    #   Assembly    #
    #---------------#

    nodes_list = createNodes(geom_data, phys_data)
    elems_list = createElements(nodes_list, geom_data, phys_data)
    printData(nodes_list, elems_list)
    
    solver = Solver()
    K, M = solver.assembly(elems_list, nodes_list, clamped_nodes)

    #---------------#
    #  Mode shapes  #
    #---------------#

    eigen_values, eigen_vectors = solver.solve(K, M)
    eigen_values = np.sqrt(eigen_values)/(2*pi)
    eigen_vectors = np.squeeze(eigen_vectors, axis=-1) # shape (162, 162, 1) -> (162, 162)

    #---------------#
    #    Display    #
    #---------------#

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    display(fig, ax, activation, nodes_list, elems_list)
    plotModes(fig, ax, nodes_list, eigen_vectors[0], elems_list, 4.5, clamped_nodes)
    plt.show()




if __name__  == "__main__":
    main()