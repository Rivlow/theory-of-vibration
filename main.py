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

    # Nodes on which supporters are sitting
    nodes_lumped = [4, 5, 21, 22, 23, 24,
                    25, 26, 10, 11, 27, 28, 
                    29, 30, 31, 32, 14, 15]
    
    # Nodes attached on the ground
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

    #----------------------#
    #   Assembly process   #
    #----------------------#

    nodes_list = createNodes(geom_data, phys_data)
    elems_list = createElements(nodes_list, nodes_lumped, geom_data, phys_data)
    printData(nodes_list, elems_list, phys_data, geom_data)

    
    solver = Solver()

    solver.assembly(elems_list, nodes_list, clamped_nodes)
    solver.addLumpedMass(nodes_list, nodes_lumped)
    solver.applyClampedNodes(nodes_list, clamped_nodes)

    K, M = solver.extractMatrices()


    #-----------------------#
    #  Compute mode shapes  #
    #-----------------------#

    eigen_values, eigen_vectors = solver.solve()

    # Extract 6 first frequencies/mode shape
    eig_vals = eigen_values[:6]
    eig_vects = eigen_vectors[:6]
    print(f"eigen values : {eig_vals} [Hz] \n")


    #---------------------------#
    #    Display deformations   #
    #---------------------------#

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    display(fig, ax, activation, nodes_list, elems_list)
    #plotModes(fig, ax, nodes_list, eigen_vectors[0], elems_list, 4.5, clamped_nodes)
    plt.show()

    # TEMPORARY REF VALUES !!!!!
    mu = np.mean(eig_vals)
    std = np.std(eig_vals)
    temp_eig_refs = np.random.normal(loc=mu, scale=std, size=6)
    #convergence(eig_vals, temp_eig_refs)




if __name__  == "__main__":
    main()