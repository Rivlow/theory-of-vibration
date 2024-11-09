import numpy as np
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from configuration.load_config import load_config

def setParams():
    """
    Load parameters from config and compute derived values.
    
    Returns:
        tuple: (geom_data, phys_data)
    """
    config = load_config()
    
    D = float(config["geometry"]["beam"]["diameter"])
    e = float(config["geometry"]["beam"]["thickness"])
    d = D - 2*e
    A = np.pi * (D**2 - d**2) / 4  # [m^2]
    Iz = Iy = np.pi * (D**4 - d**4) / 64  # [m^4]
    Jx = 2 * Iy  # [m^4]
    
    E = float(config["physics"]["material"]["E"])
    nu = float(config["physics"]["material"]["nu"])
    G = E / (2 * (1 + nu))  # [Pa]
    
    M_lumped = (float(config["physics"]["mass"]["weight_per_person"]) * 
                float(config["physics"]["mass"]["number_of_people"]) / 
                float(config["physics"]["mass"]["number_of_nodes"]))  # [kg]
    
    geom_data = {
        "z_min": float(config["geometry"]["dimensions"]["z"]["min"]),
        "z_mid": float(config["geometry"]["dimensions"]["z"]["mid"]),
        "z_max": float(config["geometry"]["dimensions"]["z"]["max"]),
        "x_min": float(config["geometry"]["dimensions"]["x"]["min"]),
        "x_mid": float(config["geometry"]["dimensions"]["x"]["mid"]),
        "x_max": float(config["geometry"]["dimensions"]["x"]["max"]),
        "y_min": float(config["geometry"]["dimensions"]["y"]["min"]),
        "y_max": float(config["geometry"]["dimensions"]["y"]["max"]),
        "D": D,
        "e": e,
        "A": A,
        "Iz": Iz,
        "Iy": Iy,
        "Jx": Jx,
        "nodes_lumped": config["geometry"]["nodes"]["lumped_positions"],
        "nodes_clamped": config["geometry"]["nodes"]["clamped"]
    }
    
    phys_data = {
        "rho": float(config["physics"]["material"]["rho"]),
        "nu": nu,
        "E": E,
        "G": G,
        "M_lumped": M_lumped
    }
    
    return geom_data, phys_data