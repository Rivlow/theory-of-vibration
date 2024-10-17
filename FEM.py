import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
pi = np.pi


class Solver:

    def __init__(self):
        self.M = None
        self.K = None

    def assembly(self, elems_list, nodes_list, clamped_nodes):
        n = 6 * len(nodes_list)
        M_data, K_data = [], []
        M_i, M_j = [], []
        K_i, K_j = [], []

        for elem in elems_list:
            DOFs_n1 = elem.locel[0]
            DOFs_n2 = elem.locel[1]
            
            for k in range(6):
                for h in range(6):
                    M_data.extend([elem.M_eS[k, h], elem.M_eS[k, h+6], elem.M_eS[k+6, h], elem.M_eS[k+6, h+6]])
                    M_i.extend([DOFs_n1[k], DOFs_n1[k], DOFs_n2[k], DOFs_n2[k]])
                    M_j.extend([DOFs_n1[h], DOFs_n2[h], DOFs_n1[h], DOFs_n2[h]])

                    K_data.extend([elem.K_eS[k, h], elem.K_eS[k, h+6], elem.K_eS[k+6, h], elem.K_eS[k+6, h+6]])
                    K_i.extend([DOFs_n1[k], DOFs_n1[k], DOFs_n2[k], DOFs_n2[k]])
                    K_j.extend([DOFs_n1[h], DOFs_n2[h], DOFs_n1[h], DOFs_n2[h]])

        self.M = sp.sparse.coo_matrix((M_data, (M_i, M_j)), shape=(n, n)).tocsr()
        self.K = sp.sparse.coo_matrix((K_data, (K_i, K_j)), shape=(n, n)).tocsr()

        # Check symmetry
        if not np.allclose((self.M - self.M.T).data, 0, atol=1e-2):
            raise ValueError("M matrix not symmetric")
        
        if not np.allclose((self.K - self.K.T).data, 0, atol=1e-2):
            raise ValueError("K matrix not symmetric")

    def addLumpedMass(self, nodes_list, nodes_lumped):
        for node in nodes_list.values():
            if node.idx in nodes_lumped:
                update = np.zeros(self.M.shape[0])
                update[node.DOF[:3]] = node.M_lumped
                
                # Mettre à jour la diagonale de la matrice M
                self.M += sp.sparse.diags(update, 0)

        print(f"Total mass : {self.computeMass()} [kg]")

    def computeMass(self):
        u = np.array([1 if (i == 0 or i % 6 == 0) else 0 for i in range(self.M.shape[0])])
        return u.T @ self.M @ u

    def removeClampedNodes(self, nodes_list, nodes_clamped):
        clamped_dofs = []
        for node in nodes_clamped:
            clamped_dofs.extend([dof for dof in nodes_list[node].DOF])

        keep = list(set(range(self.M.shape[0])) - set(clamped_dofs))
        self.M = self.M[keep][:, keep]
        self.K = self.K[keep][:, keep]

    def extractMatrices(self):
        return self.K, self.M

  
    def solve(self):
        # Solve system using sparse eigenvalue solver
        eigen_values, eigen_vectors = sp.sparse.linalg.eigsh(self.K, k=6, M=self.M, sigma=0, which='LM')

        # Sorting (by increasing values)
        eigen_values = np.sqrt(np.sort(eigen_values.real)) / (2 * np.pi)
        order = np.argsort(eigen_values)
        sorted_eigen_vectors = eigen_vectors.real[:, order]

        return eigen_values, sorted_eigen_vectors



class Node:
    
    def __init(self):
        self.idx = 0
        self.M_lumped = 0
        self.pos = None
        self.DOF = None
        

class Element:
    
    def __init(self):
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
        self.nodes = None
        self.locel = None
        self.T = None
        self.K_el = None
        self.K_eS = None
        self.M_el = None
        self.M_eS = None

    def getT(self, nodes_list):

        pos_1, pos_2 = [], []
        for i in range(3):
            pos_1.append(nodes_list[self.nodes[0]].pos[i])
            pos_2.append(nodes_list[self.nodes[1]].pos[i])
        
        # third (not aligned) point
        pos_3 = [2. + pos_2[0], 3.487 + pos_2[1], -4.562 + pos_2[2]] 

        # Check that third point is not aligned
        if collinearPoints(pos_1, pos_2, pos_3):
            pos_3 = adjustPosition(pos_1, pos_2, pos_3, epsilon=0.01)

        d_3, d_2 = [], []
        for i in range(3):
            d_2.append(pos_2[i] - pos_1[i])
            d_3.append(pos_3[i] - pos_1[i])

        # Orthogonal base (global)   
        I = np.eye(3, dtype=float)
        eX, eY, eZ = I[0], I[1], I[2]

        # Orthogonal base (local)
        ex = [(pos_2[i] - pos_1[i])/self.l for i in range(3)]
        ey = np.cross(d_3,d_2)/(np.linalg.norm(np.cross(d_3,d_2))) 
        ez = np.cross(ex,ey) 

        # Rotation matrix
        R = np.block([[np.dot(eX,ex),np.dot(eY,ex),np.dot(eZ,ex)],
                      [np.dot(eX,ey),np.dot(eY,ey),np.dot(eZ,ey)],
                      [np.dot(eX,ez),np.dot(eY,ez),np.dot(eZ,ez)]])
                
        # Transformation matrix
        self.T = sp.linalg.block_diag(R, R, R, R)

    def getK(self):

        E, G = self.E, self.G
        A = self.A
        l = self.l
        Iz, Iy = self.Iz, self.Iy
        Jx = self.Jx

        # K matrix element (local axis)
        K_el = np.array([[E*A/l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 12*E*Iz/(l**3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 12*E*Iy/(l**3), 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, G*Jx/l, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -6*E*Iy/(l**2), 0, 4*E*Iy/l, 0, 0, 0, 0, 0, 0, 0],
                         [0, 6*E*Iz/(l**2), 0, 0, 0, 4*E*Iz/l, 0, 0, 0, 0, 0, 0],
                         [-E*A/l, 0, 0, 0, 0, 0, E*A/l, 0, 0, 0, 0, 0],
                         [0, -12*E*Iz/(l**3), 0, 0, 0, -6*E*Iz/(l**2), 0, 12*E*Iz/(l**3), 0, 0, 0, 0],
                         [0, 0, -12*E*Iy/(l**3), 0, 6*E*Iy/(l**2), 0, 0, 0, 12*E*Iy/(l**3), 0, 0, 0],
                         [0 ,0 ,0, -G*Jx/l, 0, 0, 0, 0, 0, G*Jx/l, 0, 0],
                         [0, 0, -6*E*Iy/(l**2), 0, 2*E*Iy/l, 0, 0, 0, 6*E*Iy/(l**2), 0, 4*E*Iy/l, 0],
                         [0, 6*E*Iz/(l**2), 0, 0, 0, 2*E*Iz/l, 0, -6*E*Iz/(l**2), 0, 0, 0, 4*E*Iz/l]])

        self.K_el = np.tril(K_el) + np.tril(K_el, -1).T 

        # K matrix element (global axis)
        self.K_eS = self.T.T @ self.K_el @ self.T
        
        # Check symmetry
        if not np.allclose(self.K_el.T, self.K_el, atol = 1e-2):
            raise ValueError("K_el matrix not symmetric")
        
        if not np.allclose(self.K_eS.T, self.K_eS, atol = 1e-2):
            raise ValueError("K_eS matrix not symmetric")
        

    def getM(self):

        coef = self.rho*self.A*self.l
        l, r = self.l, self.r

        M_el = coef*np.array([[1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 13/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 13/35, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, r**2/3, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -11*l/210, 0, l**2/105, 0, 0, 0, 0, 0, 0, 0],
                              [0, 11*l/210, 0, 0, 0, l**2/105, 0, 0, 0, 0, 0, 0],
                              [1/6 ,0 ,0 ,0 , 0, 0, 1/3, 0, 0, 0, 0 ,0],
                              [0, 9/70, 0, 0, 0, 13*l/420, 0, 13/35, 0, 0, 0, 0],
                              [0, 0, 9/70, 0, -13*l/420, 0, 0, 0, 13/35, 0, 0, 0],
                              [0, 0, 0, r**2/6, 0, 0, 0, 0, 0, r**2/3, 0, 0],
                              [0 ,0 , 13*l/420, 0, -l**2/140, 0, 0, 0, 11*l/210, 0, l**2/105, 0],
                              [0 , -13*l/420, 0, 0, 0, -l**2/140, 0, -11*l/210, 0, 0, 0, l**2/105]])
        
        # M matrix element (local axis)
        self.M_el = np.tril(M_el) + np.tril(M_el, -1).T

        # M matrix element (global axis)
        self.M_eS = self.T.T @ self.M_el @ self.T

        # Check symmetry
        if not np.allclose(self.M_el.T, self.M_el, atol = 1e-2):
            raise ValueError("M_el matrix not symmetric")
        
        if not np.allclose(self.M_eS.T, self.M_eS, atol = 1e-2):
            raise ValueError("M_eS matrix not symmetric")
        
def collinearPoints(pos_1, pos_2, pos_3):

    vec1 = np.array([pos_2[0] - pos_1[0], pos_2[1] - pos_1[1], pos_2[2] - pos_1[2]])
    vec2 = np.array([pos_3[0] - pos_1[0], pos_3[1] - pos_1[1], pos_3[2] - pos_1[2]])
    
    return np.allclose(np.cross(vec1, vec2), np.zeros(3))

def adjustPosition(pos_1, pos_2, pos_3, epsilon=0.01):
    
    # modify "pos_3" until it is not aligned anymore
    while collinearPoints(pos_1, pos_2, pos_3):
        pos_3 = (pos_3[0] + epsilon, pos_3[1] + epsilon, pos_3[2] + epsilon)
        epsilon *= 2  
    return pos_3


def initializeGeometry(geom_data, phys_data):

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

    nodes_list = {}
    incr = 0
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
                nodes_list[idx] = node
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
            nodes_list[idx] = node
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
                        nodes_list[idx] = node
                        incr += 6
                        idx += 1
    # oh no please...
    nodes_pairs = [[0, 16], [0, 2], [1, 16], [1, 3], [2, 16], [3, 16],
                   [2, 3], [2, 4], [3, 5], [2, 17], [3, 17], [4, 17], [5, 17], [4, 5],
                   [4, 21], [5, 22], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                   [31, 32], [21, 23], [22, 24], [23, 25], [24, 26], [25, 10], [26, 11],
                   [10, 27], [11, 28], [6, 8], [6, 18], [7, 18], [7, 9], [8, 18], [8, 10],
                   [9, 11], [9, 18], [8, 19], [9, 19], [10, 19], [11, 19], [8, 9], [10, 11],
                   [27, 29], [28, 30], [29, 31], [30, 32], [31, 14], [32, 15], [12, 20],
                   [13, 20], [14, 20], [15, 20], [12, 14], [13, 15], [14, 15]]

    return nodes_list, nodes_pairs

def interpolate_pos(pos1, pos2, ratio):
    return [p1 + ratio * (p2 - p1) for p1, p2 in zip(pos1, pos2)]

def addMoreNodes(nodes_list_init, nodes_pairs_init, elem_per_beam):

    if elem_per_beam < 1:
        return nodes_list_init, nodes_pairs_init
    
    else:

        new_nodes_pairs = []

        max_dof = max(max(node.DOF) for node in nodes_list_init.values())
        max_idx = max(node.idx for node in nodes_list_init.values())
        
        for node_start_idx, node_end_idx in nodes_pairs_init:

            node_start = nodes_list_init[node_start_idx] 
            node_end = nodes_list_init[node_end_idx]

            last_idx = node_start.idx
            incr = max_dof + 1  
            
            for i in range(1, elem_per_beam + 1):
                ratio = i / (elem_per_beam + 1)
                new_pos = interpolate_pos(node_start.pos, node_end.pos, ratio)
                max_idx += 1 
                
                new_node = Node()
                new_node.idx = max_idx
                new_node.pos = new_pos
                new_node.DOF = list(range(incr, incr + 6))
                incr += 6 
                nodes_list_init[max_idx] = new_node
                
                new_nodes_pairs.append([last_idx, max_idx])
                last_idx = max_idx 

                max_dof = max(max(node.DOF) for node in nodes_list_init.values())
                max_idx = max(node.idx for node in nodes_list_init.values())

            new_nodes_pairs.append([last_idx, node_end_idx])
        
        return nodes_list_init, new_nodes_pairs
    


def createElements(nodes_pairs, nodes_list, geom_data, phys_data):

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

        elem.A, elem.l = A, np.sqrt(x*x + y*y + z*z)
        if elem.l == 0:
            raise ValueError("Lenght of elem is 0")
        elem.Iz, elem.Iy, elem.Jx, elem.r  = Iz, Iy, Jx, np.sqrt(Jx/A)
        elem.E, elem.G, elem.rho  = E, G, rho
       
        elem.getT(nodes_list)
        elem.getK()
        elem.getM()

        elems.append(elem)

    return elems





