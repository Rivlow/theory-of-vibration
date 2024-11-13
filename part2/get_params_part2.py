import numpy as np
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from configuration.load_config import load_config
from part1.get_params_part1 import setParams as setParams1

def setParams():
    """
    Load parameters for part 2, including part 1 parameters.
    
    Returns:
        tuple: (geom_data, phys_data, sim_data)
    """
    config = load_config()
    geom_data, phys_data = setParams1()
    
    frequency = float(config["transient"]["excitation"]["frequency"])
    period = 1/frequency
    num_timesteps = (int(config["transient"]["time"]["num_periods"]) * 
                    int(config["transient"]["time"]["steps_per_period"]))
    
    sim_data = {
        "dt": float(config["transient"]["time"]["dt"]),
        "freq": frequency,
        "h": float(config["transient"]["excitation"]["height"]),
        "g": float(config["transient"]["excitation"]["gravity"]),
        "nodes_force": config["transient"]["nodes"]["force_application"],
        "nodes_obs": config["transient"]["nodes"]["observation"],
        "m_tot": (float(config["physics"]["mass"]["weight_per_person"]) * 
                 float(config["physics"]["mass"]["number_of_jumping_people"])),
        "t_span": np.linspace(0, 
                            float(config["transient"]["time"]["num_periods"]) * period,
                            num_timesteps),
        "newmark": {
            "gamma": float(config["transient"]["newmark"]["gamma"]),
            "beta": float(config["transient"]["newmark"]["beta"])
        },
        "damping": {
            "mode1_ratio": float(config["transient"]["damping"]["mode1_ratio"]),
            "mode2_ratio": float(config["transient"]["damping"]["mode2_ratio"])
        }
    }
    
    return geom_data, phys_data, sim_data