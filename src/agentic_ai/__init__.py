from .utils import Logger
logger_instance = Logger(colorful_output=True)
import logging
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)

from .open_router import AIAgent
