import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For other exceptions.

    Returns:
        ConfigBox: Parsed YAML content as ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise BoxValueError
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file '{path_to_yaml}' is empty")
    except Exception as e:
        logger.error(f"Error loading YAML file '{path_to_yaml}': {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """Creates a list of directories.

    Args:
        path_to_directories (List[Path]): List of paths of directories to create.
        verbose (bool, optional): Verbose logging. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as class attributes instead of a dictionary.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data to a binary file.

    Args:
        data (Any): Data to be saved.
        path (Path): Path to the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Loaded data.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decode_image(imgstring: str, file_name: Path):
    """Decodes a base64 image string and saves it as a file.

    Args:
        imgstring (str): Base64 encoded image string.
        file_name (Path): Path to save the decoded image file.
    """
    imgdata = base64.b64decode(imgstring)
    with open(file_name, 'wb') as f:
        f.write(imgdata)
    logger.info(f"Image decoded and saved at: {file_name}")

def encode_image_into_base64(image_path: Path) -> str:
    """Encodes an image file into a base64 string.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64 encoded image string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')