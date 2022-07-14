from enum import unique
import os
import yaml
import logging
from typing import Dict


def read_yaml(path_to_yaml: str) -> Dict:
    """
    This Function reads the config.yaml from root directory and
    return configuration in dictionary.
    Returns: Dict of config
    """
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content


def create_directories(path_to_directories: list) -> None:
    """
    This Function creats folders as mentioned in the input list.
    Returns: None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")
