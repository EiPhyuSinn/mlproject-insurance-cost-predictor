import logging

logging.basicConfig(
    filename="project.log",          # log file name
    level=logging.INFO,              # log level (INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_logger():
    return logging.getLogger()
