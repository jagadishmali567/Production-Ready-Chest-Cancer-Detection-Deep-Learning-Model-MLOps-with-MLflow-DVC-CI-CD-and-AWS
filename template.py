import os
from pathlib import Path
import logging

# creating the logger for the script execution status and error messages if any
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# creating the project name variable
project_name = "cnnClassifier"

# creating the directories and files required for the project structure
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

# creating the directory structure and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    # if the directory doesn't exist, create it
    if not filedir.exists():
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filepath.name}")

    # if the file exists, create an empty file if it's empty and skip otherwise
    if not filepath.exists() or filepath.stat().st_size == 0:
        try:
            filepath.touch()
            logging.info(f"Creating empty file: {filepath}")
        except Exception as e:
            logging.error(f"Failed to create file {filepath}: {e}")
    else:
        logging.info(f"{filepath.name} already exists")
