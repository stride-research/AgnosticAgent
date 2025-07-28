from .utils import Logger
logger_instance = Logger(colorful_output=True)

from .backends.open_router import AIAgent, ToolkitBase
from .config import CONFIG_DICT