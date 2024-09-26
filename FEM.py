import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


class Solver:

    def __init__(self):
            self.ne = 0

    def assembly(self, elems_list, nodes_list, clamped_nodes):
         
        M = np.zeros((6*len(nodes_list), 6*len(nodes_list)))
        K = np.zeros((6*len(nodes_list), 6*len(nodes_list)))

        for i in range(len(elems_list)):
             
            elem = elems_list[i]
   
            ddls_down = elem.locel[0]
            ddls_up = elem.locel[1]

            for k in range(6):
                for h in range(6):
                    M[ddls_down[k] - 1, ddls_down[h] - 1] += elem.M_eS[k, h]
                    M[ddls_up[k] - 1, ddls_up[h] - 1] += elem.M_eS[6 + k, 6 + h]
                    M[ddls_down[k] - 1, ddls_up[h] - 1] += elem.M_eS[k, 6 + h]
                    M[ddls_up[k] - 1, ddls_down[h] - 1] += elem.M_eS[6 + k, h]
                    
                    K[ddls_down[k] - 1, ddls_down[h] - 1] += elem.K_eS[k, h]
                    K[ddls_up[k] - 1, ddls_up[h] - 1] += elem.K_eS[6 + k, 6 + h]
                    K[ddls_down[k] - 1, ddls_up[h] - 1] += elem.K_eS[k, 6 + h]
                    K[ddls_up[k] - 1, ddls_down[h] - 1] += elem.K_eS[6 + k, h]

        # Apply clamped nodes condition on K, M
        clamped_dofs = []

        for node in clamped_nodes:
            clamped_dofs.extend(nodes_list[node].DOF)

        M = np.delete(M, clamped_dofs, axis=0)  
        M = np.delete(M, clamped_dofs, axis=1)  
        K = np.delete(K, clamped_dofs, axis=0)  
        K = np.delete(K, clamped_dofs, axis=1)  

        return K, M
            

    def solve(self, K, M):

        # Solve system
        eigen_values, eigen_vectors = sp.linalg.eig(K, M) 

        # Sort (increasing value)
        eigen_vectors = np.matrix(eigen_vectors)
        eigen_values=eigen_values.real
        order = np.argsort(eigen_values) 
        eigen_values.sort()

        sorted_eigen_vectors = []
        eigen_vectors=eigen_vectors.real
        for j in order:
            sorted_eigen_vectors.append(eigen_vectors[:,j]) 

        return eigen_values, sorted_eigen_vectors



class Node:
    
    def __init(self):
        self.idx = 0
        self.M_lumped = 0
        self.pos = None
        self.DOF = None
        

class Element:
    
    def __init(self):
        self.nodes = None
        self.locel = None
        self.rho = 0
        self.A = 0
        self.l = 0
        self.E = 0
        self.Iz = 0
        self.Iy = 0
        self.Jy = 0
        self.G = 0
        self.Jx = 0
        self.r = 0
        self.M_lumped = 0
        self.T = None
        self.K_el = None
        self.K_eS = None
        self.M_el = None
        self.M_eS = None

    def getT(self, nodeList):

        pos_1, pos_2 = [], []
        for i in range(3):
            pos_1.append(nodeList[self.nodes[0]].pos[i])
            pos_2.append(nodeList[self.nodes[1]].pos[i])
        
        pos_3 = [2. + pos_2[0], 3.487 + pos_2[1], -4.562 + pos_2[2]] # third not-aligned point
        d_3, d_2 = [], []
        for i in range(3):
            d_2.append(pos_2[i] - pos_1[i])
            d_3.append(pos_3[i] - pos_1[i])


        # orthogonal base (global)
        I = np.eye(3, dtype=float)
        eX, eY, eZ = I[0], I[1], I[2]

        # orthogonal base (local)
        ex = [(pos_2[i] - pos_1[i])/self.l for i in range(3)]
        ey = np.cross(d_3,d_2)/(np.linalg.norm(np.cross(d_3,d_2))) 
        ez = np.cross(ex,ey) 

        R = np.block([[np.dot(eX,ex),np.dot(eY,ex),np.dot(eZ,ex)],
                      [np.dot(eX,ey),np.dot(eY,ey),np.dot(eZ,ey)],
                      [np.dot(eX,ez),np.dot(eY,ez),np.dot(eZ,ez)]])
        
        
                    
        O = np.zeros((3,3)) # 3 ddls per node
        self.T = np.block([[R, O, O, O],
                           [O, R, O, O],
                           [O, O, R, O],
                           [O, O, O, R]])

    def getK(self):

        K_el = np.array([[self.E*self.A/self.l,0,0,0,0,0,0,0,0,0,0,0],
                         [0,12*self.E*self.Iz/(self.l**3),0,0,0,0,0,0,0,0,0,0],
                         [0,0,12*self.E*self.Iy/(self.l**3),0,0,0,0,0,0,0,0,0],
                         [0,0,0,self.G*self.Jx/self.l,0,0,0,0,0,0,0,0],
                         [0,0,-6*self.E*self.Iy/(self.l**2),0,4*self.E*self.Iy/self.l,0,0,0,0,0,0,0],
                         [0,6*self.E*self.Iz/(self.l**2),0,0,0,4*self.E*self.Iz/self.l,0,0,0,0,0,0],
                         [-self.E*self.A/self.l,0,0,0,0,0,self.E*self.A/self.l,0,0,0,0,0],
                         [0,-12*self.E*self.Iz/(self.l**3),0,0,0,-6*self.E*self.Iz/(self.l**2),0,12*self.E*self.Iz/(self.l**3),0,0,0,0],
                         [0,0,-12*self.E*self.Iy/(self.l**3),0,6*self.E*self.Iy/(self.l**2),0,0,0,12*self.E*self.Iy/(self.l**3),0,0,0],
                         [0,0,0,-self.G*self.Jx/self.l,0,0,0,0,0,self.G*self.Jx/self.l,0,0],
                         [0,0,-6*self.E*self.Iy/(self.l**2),0,2*self.E*self.Iy/self.l,0,0,0,6*self.E*self.Iy/(self.l**2),0,4*self.E*self.Iy/self.l,0],
                         [0,6*self.E*self.Iz/(self.l**2),0,0,0,2*self.E*self.Iy/self.l,0,-6*self.E*self.Iz/(self.l**2),0,0,0,4*self.E*self.Iz/self.l]])

        self.K_el = np.tril(K_el) + np.tril(K_el, -1).T 
        self.K_eS = self.T.T @ self.K_el @ self.T

    def getM(self, lumped_nodes):

        coef = self.rho*self.A*self.l
        n1, n2 = self.nodes[0], self.nodes[1]
        M_lumped_n1, M_lumped_n2 = 0, 0

        if n1 in lumped_nodes:
            M_lumped_n1 = self.M_lumped
            M_lumped_n1 /= coef

        if n2 in lumped_nodes:
            M_lumped_n2 = self.M_lumped
            M_lumped_n2 /= coef

        M_el = coef*np.array([[1/3 + M_lumped_n1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 13/35 + M_lumped_n1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 13/35 + M_lumped_n1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, self.r**2/3, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -11*self.l/210, 0, self.l**2/105, 0, 0, 0, 0, 0, 0, 0],
                              [0, 11*self.l/210, 0, 0, 0, self.l**2/105, 0, 0, 0, 0, 0, 0],
                              [1/6 ,0 ,0 ,0 , 0, 0, 1/3 + M_lumped_n2, 0, 0, 0, 0 ,0],
                              [0, 9/70, 0, 0, 0, 13*self.l/420, 0, 13/35 + M_lumped_n2, 0, 0, 0, 0],
                              [0, 0, 9/70, 0, -13*self.l/420, 0, 0, 0, 13/35 + M_lumped_n2, 0, 0, 0],
                              [0, 0, 0, self.r**2/6, 0, 0, 0, 0, 0, self.r**2/3, 0, 0],
                              [0 ,0 , 13*self.l/420, 0, -self.l**2/140, 0, 0, 0, 11*self.l/210, 0, self.l**2/105, 0],
                              [0 , -13*self.l/420, 0, 0, 0, -self.l**2/140, 0, -11*self.l/210, 0, 0, 0, self.l**2/105]])
        
        self.M_el = np.tril(M_el) + np.tril(M_el, -1).T
        self.M_eS = self.T.T @ self.M_el @ self.T


def createNodes(geom_data, phys_data):


    z_min, z_mid, z_max = geom_data["z_min"], geom_data["z_mid"], geom_data["z_max"]
    # z init (lateral bars)
    z_0 = np.linspace(0, z_max, 3)
    z_4 = np.linspace(0, z_mid, 3)
    z_8 = np.linspace(0, z_min, 2)
    z_tab = [z_0, z_4, z_8]

    # z init (oblique bars)
    z_oblique = np.linspace(z_max, z_min, 9)
    
    # z init (lateral x-crossing bars)
    z_supp_0 = np.arange(z_max/4, z_max, z_max/2)
    z_supp_4 = np.arange(z_mid/4, z_mid, z_mid/2)
    z_supp_8 = [z_min/2]
    z_tab_supp = [z_supp_0, z_supp_4, z_supp_8]

    # x init
    x_min, x_mid, x_max = geom_data["x_min"], geom_data["x_mid"], geom_data["x_max"]
    x_tab = np.arange(0, 12, 4)
    x_oblique = np.linspace(x_min, x_max, 9)

    # y init
    y_min, y_max = geom_data["y_min"], geom_data["y_max"]
    y_tab = np.array([y_min, y_max])

    #================#
    # Nodes creation #
    #================#

    nodes = []
    incr = 1
    idx = 0
    nodes_lumped, M_lumped = geom_data["nodes_lumped"], phys_data["M_lumped"]

    # Vertical points (on edges)
    for i, x_val in enumerate(x_tab):
        for z_val in z_tab[i]:
            for y_val in y_tab:
                node = Node()
                node.idx = idx
                if idx in nodes_lumped:
                    node.M_lumped = M_lumped
                node.pos = [x_val, y_val, z_val]
                node.DOF = list(range(incr, incr+6))
                nodes.append(node)
                incr += 6
                idx += 1
    
    # Vertical points (for crossing lines)
    for i, x_val in enumerate(x_tab):
        for z_val in z_tab_supp[i]:
            node = Node()
            node.idx = idx
            if idx in nodes_lumped:
                node.M_lumped = M_lumped
            node.pos = [x_val, 1.0, z_val]
            node.DOF = list(range(incr, incr+6))
            nodes.append(node)
            incr += 6
            idx += 1
    
    # Points on descending slope
    for x_val, z_val in zip(x_oblique, z_oblique):
            if z_val in [z_max, z_mid, z_min]:
                    continue
            else:
                for y_val in y_tab:
                        node = Node()
                        node.idx = idx
                        if idx in nodes_lumped:
                            node.M_lumped = M_lumped
                        node.pos = [x_val, y_val, z_val]
                        node.DOF = list(range(incr, incr+6))
                        nodes.append(node)
                        incr += 6
                        idx += 1

    return nodes

def createElements(nodes_list, geom_data, phys_data):

    # oh no please...
    nodes_pairs = [[0, 16], [0, 2], [1, 16], [1, 3], [2, 16], [3, 16],
                   [2, 3], [2, 4], [3, 5], [2, 17], [3, 17], [4, 17], [5, 17], [4, 5],
                   [4, 21], [5, 22], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                   [31, 32], [21, 23], [22, 24], [23, 25], [24, 26], [25, 10], [26, 11],
                   [10, 27], [11, 28], [6, 8], [6, 18], [7, 18], [7, 9], [8, 18], [8, 10],
                   [9, 11], [9, 18], [8, 19], [9, 19], [10, 19], [11, 19], [8, 9], [10, 11],
                   [27, 29], [28, 30], [29, 31], [30, 32], [31, 14], [32, 15], [12, 20],
                   [13, 20], [14, 20], [15, 20], [12, 14], [13, 15], [14, 15]]
    
    elems = []

    for pair in nodes_pairs:

        elem = Element()
        elem.nodes = pair
        elem.locel = [nodes_list[pair[0]].DOF, nodes_list[pair[1]].DOF]

        x = nodes_list[pair[0]].pos[0] - nodes_list[pair[1]].pos[0]
        y = nodes_list[pair[0]].pos[1] - nodes_list[pair[1]].pos[1]
        z = nodes_list[pair[0]].pos[2] - nodes_list[pair[1]].pos[2]

        A, Iz, Iy, Jx = geom_data["A"], geom_data["Iz"], geom_data["Iy"], geom_data["Jx"]
        nodes_lumped, M_lumped = geom_data["nodes_lumped"], phys_data["M_lumped"]
        E, G, rho = phys_data["E"], phys_data["G"], phys_data["rho"]

        elem.l = np.sqrt(x*x + y*y + z*z)
        elem.E = E
        elem.Iz = Iz
        elem.Iy = Iy
        elem.Jx = Jx
        elem.A = A
        elem.r = np.sqrt(Jx/A)
        elem.G = G
        elem.rho = rho
        elem.M_lumped = M_lumped

        elem.getT(nodes_list)
        elem.getK()
        elem.getM(nodes_lumped)
    
        elems.append(elem)

    return elems





