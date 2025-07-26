""" Create some piece of code that runs x times given y exeception """
from typing import Type, List, Dict, Callable
import logging 
import asyncio

from openai import APIStatusError
from pydantic import BaseModel


logger = logging.getLogger(__name__)

class ErrorAllowance(BaseModel):
    n_of_occurrences: int = 0
    n_of_allowances: int = 0

    def incremenet_occurence(self) -> None:
        self.n_of_occurrences += 1
    
    def has_allowance_remaining(self):
        return self.n_of_occurrences <= self.n_of_allowances


from typing import Type, List, Dict, Callable
import logging
import asyncio
import httpx

from openai import APIStatusError
from pydantic import BaseModel

# Configure logging to see the output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class ErrorAllowance(BaseModel):
    n_of_occurrences: int = 0
    n_of_allowances: int = 0

    def increment_occurrence(self) -> None: # Corrected typo
        self.n_of_occurrences += 1

    def has_allowance_remaining(self) -> bool:
        # Reruns as many times as n_of_allowances. Calls func that + 1
        return self.n_of_occurrences <= self.n_of_allowances


class ExceptionRetryController:
    """
    Controls execution with retries based on allowed exception types and specific conditions.
    """
    def __init__(self, error_allowances: Dict[Type[Exception], int]) -> None:
        self.error_record = {
            err_type: ErrorAllowance(n_of_allowances=count)
            for err_type, count in error_allowances.items()
        }
        self.total_interactions = 0

    async def _resolve_APIStatusError(self, e: APIStatusError, time_to_wait_between_retries: int) -> bool:
        status_code = e.status_code
        logger.debug(f"Caught APIStatusError with status code: {status_code}")

        if 500 <= status_code < 600:
            if APIStatusError in self.error_record:
                api_error_allowance = self.error_record[APIStatusError]
                api_error_allowance.increment_occurrence()

                if api_error_allowance.has_allowance_remaining():
                    logger.warning(
                        f"Caught APIStatusError (5xx: {status_code}) and message: {e.message}. "
                        f"Occurrences: {api_error_allowance.n_of_occurrences}, "
                        f"Allowances: {api_error_allowance.n_of_allowances}. Retrying..."
                    )
                    await asyncio.sleep(time_to_wait_between_retries)
                    return False
                else:
                    logger.exception(
                        f"Caught APIStatusError (5xx: {status_code}). "
                        f"Maximum allowances ({api_error_allowance.n_of_allowances}) exceeded. "
                        "No more retries."
                    )
                    return True
            else:
                logger.critical(
                    f"Caught APIStatusError (5xx: {status_code}) but no allowances. Message is: {e.message}"
                    "defined for APIStatusError. Execution stopped."
                )
                return True
        else:
            logger.exception(
                f"Caught APIStatusError (non-5xx: {status_code}). "
                "Not configured for retry on this status code type. Re-raising."
            )
            return True


    async def execute_with_retries(self, func: Callable, time_to_wait_between_retries: int = 3, *args, **kwargs):
        while True:
            self.total_interactions += 1 
            logger.info(f"Attempting execution. Total interactions: {self.total_interactions}")

            try:
                result = await func(*args, **kwargs)
                logger.info("Function executed successfully.")
                return result

            except Exception as e:
                exception_type = type(e)
                error_info = self.error_record.get(exception_type)
                print(f"EXCEPTION TYPE IS: {exception_type}")
                print(isinstance(e, APIStatusError))

                if isinstance(e, APIStatusError):
                    raise_APIStatusError_exception = await self._resolve_APIStatusError(e, time_to_wait_between_retries=time_to_wait_between_retries)
                    if raise_APIStatusError_exception:
                        raise e
                    else: 
                        continue

                elif error_info: 
                    error_info.increment_occurrence()
                    logger.debug(f"Has allowance remaining: {error_info.has_allowance_remaining()}")
                    if error_info.has_allowance_remaining():
                        logger.warning(
                            f"Caught {exception_type.__name__}. "
                            f"Occurrences: {error_info.n_of_occurrences}, "
                            f"Allowances: {error_info.n_of_allowances}. Retrying..."
                        )
                        await asyncio.sleep(time_to_wait_between_retries)
                        continue
                    else:
                        logger.error(
                            f"Caught {exception_type.__name__}. "
                            f"Maximum allowances ({error_info.n_of_allowances}) exceeded. "
                            "No more retries."
                        )
                        raise e 
                else:
                    # Unhandled general exception
                    logger.critical(
                        f"Caught an unhandled exception: {exception_type.__name__}. "
                        "No allowances defined for this error type. "
                        "Execution stopped."
                    )
                    raise e 

# key: error, value: numer of reattempts
my_error_allowances = {
    APIStatusError: 3 # For 5xx APIStatusErrors 
}

exception_controller_executor = ExceptionRetryController(my_error_allowances)
