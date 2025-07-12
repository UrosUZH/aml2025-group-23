import yaml
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """
    Load YAML config file.

    Args:
        config_path (Path): Path to config YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
