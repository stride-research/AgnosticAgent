
from typing import Dict, Any
import os
import logging

import yaml

logger = logging.getLogger(__name__)


def get_config(path: str) -> Dict[str, Any]:
      if not os.path.exists(path):
            logger.error(f"Path provided for config {path} does not exist")
            return None
      with open(path) as f:
            data = yaml.safe_load(f)
            return data

path = "src/agentic_ai/config/config.yaml"
CONFIG_DICT = get_config(path=path)