import sys
from logger import get_logger

logger = get_logger()

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        logger.error(f"Error occurred: {message}", exc_info=True)
