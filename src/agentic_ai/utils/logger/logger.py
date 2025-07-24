import logging
import logging.handlers
import queue
import sys
from pythonjsonlogger import jsonlogger
import contextvars
from contextlib import contextmanager
from .colorfulFormatter import ColoredJSONFormatter


class ContextAwareQueueHandler(logging.handlers.QueueHandler):
    """
    Injects dynamic fields before enqueing
    """
    def prepare(self, record):
        context = LOG_CONTEXT.get()
        for key, value in context.items():
            setattr(record, key, value)
        return super().prepare(record)

class Logger():
    def __init__(self, colorful_output=True) -> None:
        self.colorful_output = colorful_output
        queue_handler = self.__set_up_queue_handler()
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)
        self.root_logger.addHandler(queue_handler)
                
    def __set_up_queue_handler(self):
        log_queue = queue.Queue(-1)
        console_handler = self.__bind_handlers()
        listener = logging.handlers.QueueListener(log_queue, console_handler)
        listener.start()
        
        queue_handler = ContextAwareQueueHandler(log_queue)
        return queue_handler

    def __bind_handlers(self) -> logging.Handler:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        formatter = self.__bind_formatter()
        stream_handler.setFormatter(formatter)
        return stream_handler
    
    def __bind_formatter(self):
        if not self.colorful_output:
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={"levelname": "level", "asctime": "time"},
            )
            return formatter
        else:
            formatter = ColoredJSONFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                rename_fields={"levelname": "level", "asctime": "time"},
            )
        
            return formatter

LOG_CONTEXT = contextvars.ContextVar("log_context", default={})         

@contextmanager
def add_context_to_log(**kwargs):
    """
    A context manager to add dynamic data to logs.
    """
    current_context = LOG_CONTEXT.get()
    new_context = {**current_context, **kwargs}
    
    token = LOG_CONTEXT.set(new_context)
    try:
        yield
    finally:
        LOG_CONTEXT.reset(token)