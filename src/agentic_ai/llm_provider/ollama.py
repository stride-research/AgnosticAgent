from ..utils.core.schemas import LLMResponse, ExtraResponseSettings, ToolSpec
from ..utils.core.function_calling import FunctionalToolkit, tool_registry
from .base import LLMProvider
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

DEV_INSTRUCTIONS = """
            You must use the provided tools.
            - When a tool is relevant, **immediately call the tool without any conversational filler or "thinking out loud" text.**
            - If you have successfully gathered all necessary information from tool calls to answer the user's request, provide the final answer directly.
            - If something is not sufficiently clear, ask for clarifications.
            - If you are needing a tool but you dont have access to it, you have to sttop and specify that you need it.
            - If there is any logical error in the request, express it, then stop.
            """

class Ollama(LLMProvider):
        ...