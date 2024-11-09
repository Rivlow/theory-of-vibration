import numpy as np
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from configuration.load_config import load_config
from part2.get_params_part2 import setParams as setParams2

def setParams():
    """
    Load parameters for part 3, including parts 1 and 2 parameters.
    
    Returns:
        tuple: (geom_data, phys_data, sim_data, reduction_data)
    """
    config = load_config()
    geom_data, phys_data, sim_data = setParams2()
    
    reduction_data = {
        "nodes_retained": config["reduction"]["nodes"]["retained"],
        "node_dof_config": config["reduction"]["nodes"]["dof_config"],
        "newmark": sim_data["newmark"]
    }
    
    return geom_data, phys_data, sim_data, reduction_data