import numpy as np
import os
import sys
from scipy import sparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from part2 import Tools_part2


dof_map = {"u": 0,
           "v": 1,
           "w": 2,
           "phi_u": 3,
           "phi_v": 4,
           "phi_w": 5}


def retainedDOF(nodes_retained, node_dof_config, nodes_clamped):

    retained_dofs = []

    for node in nodes_retained:
        DOF_init = Tools_part2.extractDOF(node, nodes_clamped)
        for dof in node_dof_config[node]:
            retained_dofs.append(DOF_init + dof_map[dof])

    return retained_dofs
