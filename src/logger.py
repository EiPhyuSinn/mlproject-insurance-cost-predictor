import os
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/project.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)

def get_logger():
    return logging.getLogger()
