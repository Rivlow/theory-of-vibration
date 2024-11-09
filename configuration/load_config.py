# configuration/load_config.py
import yaml
import os
from pathlib import Path

def load_config():
    """Load configuration from YAML file."""
    # Trouver le chemin du fichier config.yaml en relatif par rapport à ce fichier
    config_path = Path(__file__).parent / 'config.yaml'
    
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {str(e)}")
