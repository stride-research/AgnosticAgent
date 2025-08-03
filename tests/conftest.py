import logging

import pytest


@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging handlers after each test."""
    yield
    # Clean up any remaining handlers
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
