from exception import handle_exception
from logger import get_logger

logger = get_logger()

try:
    logger.info("Program started")
    x = 10 / 0   # error
except Exception as e:
    handle_exception(e)

logger.info("Program ended normally")
