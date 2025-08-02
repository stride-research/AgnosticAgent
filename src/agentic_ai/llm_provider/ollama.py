from ..utils.core.schemas import LLMResponse, ExtraResponseSettings, ToolSpec
from ..utils.core.function_calling.openai import FunctionalToolkit, tool_registry
from .base_llm_provider import LLMProvider
from agentic_ai.utils import add_context_to_log

import json
import os
import logging
import base64
import asyncio
import aiofiles
import time
import inspect
from typing import Optional, Any, Tuple, List, Dict, Type, Callable, Coroutine
import concurrent.futures


from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Ollama(LLMProvider):
        ...