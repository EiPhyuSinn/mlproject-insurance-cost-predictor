import os 
from src.logger import get_logger
import dill

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger = get_logger()
        logger.info(f"Directory created at: {directory_path}")

def save_object(file_path: str, obj: object) -> None:
    """Save a Python object to a file using dill."""
    try:
        directory = os.path.dirname(file_path)
        create_directory_if_not_exists(directory)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger = get_logger()
        logger.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        logger = get_logger()
        logger.error(f"Failed to save object at {file_path}: {e}")
        raise