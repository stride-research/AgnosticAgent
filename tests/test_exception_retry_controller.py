import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIStatusError

from agnostic_agent.utils.fault_tolerance.exception_retry_controller import \
    ExceptionRetryController


# Helper function to create a dummy httpx.Request for APIStatusError
def create_mock_api_error(status_code: int, message: str = "Fake error"):
    """Creates a mock APIStatusError with a dummy httpx.Request attached."""
    dummy_request = httpx.Request("GET", f"http://example.com/api/{status_code}")
    return APIStatusError(
        message=message,
        response=httpx.Response(status_code=status_code, request=dummy_request),
        body=None
    )

@pytest.fixture
def clean_controller():
    """Provides a fresh ExceptionRetryController instance for each test."""
    return ExceptionRetryController({
        ValueError: 2,  # Allow 2 retries for ValueError
        APIStatusError: 3, # Allow 3 retries for 5xx APIStatusErrors
        KeyError: 1 # Example for another type of defined exception
    })

@pytest.mark.asyncio
async def test_successful_execution(clean_controller):
    """
    Tests that a function executes successfully on the first attempt
    when no exceptions occur.
    """
    mock_func = AsyncMock(return_value="Successful result")
    result = await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert result == "Successful result"
    mock_func.assert_called_once()
    assert clean_controller.total_interactions == 1

@pytest.mark.asyncio
async def test_value_error_retries_then_success(clean_controller, mocker):
    """
    Tests that a ValueError is retried the allowed number of times
    and eventually succeeds.
    """
    # Mock asyncio.sleep to prevent actual delays during tests
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    # Configure the mock function to raise ValueError twice, then succeed
    mock_func = AsyncMock(side_effect=[
        ValueError("First ValueError attempt"),
        ValueError("Second ValueError attempt"),
        "Expected Success!"
    ])

    result = await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert result == "Expected Success!"
    assert mock_func.call_count == 3  # 2 retries + 1 success = 3 calls
    assert clean_controller.total_interactions == 3
    # Check that occurrences were recorded correctly
    assert clean_controller.error_record[ValueError].n_of_occurrences == 2
    assert mock_sleep.call_count == 2 # Sleep should be called for each retry

@pytest.mark.asyncio
async def test_value_error_allowances_exceeded(clean_controller, mocker):
    """
    Tests that a ValueError is re-raised if the allowed number of retries is exceeded.
    """
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        ValueError("Attempt 1"),
        ValueError("Attempt 2"),
        ValueError("Attempt 3 - Should fail") # Exceeds allowance of 2 (0-indexed occurrences vs allowance)
    ])

    with pytest.raises(ValueError, match="Attempt 3 - Should fail"):
        await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 3
    assert clean_controller.total_interactions == 3
    assert clean_controller.error_record[ValueError].n_of_occurrences == 3
    assert mock_sleep.call_count == 2 # Sleep should be called for the first two retries

@pytest.mark.asyncio
async def test_api_status_error_5xx_retries_then_success(clean_controller, mocker):
    """
    Tests that a 5xx APIStatusError is retried the allowed number of times
    and eventually succeeds.
    """
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        create_mock_api_error(500, "Internal Server Error 1"),
        create_mock_api_error(502, "Bad Gateway 2"),
        create_mock_api_error(504, "Gateway Timeout 3"),
        "API Call Success!"
    ])

    result = await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert result == "API Call Success!"
    assert mock_func.call_count == 4 # 3 retries + 1 success = 4 calls
    assert clean_controller.total_interactions == 4
    assert clean_controller.error_record[APIStatusError].n_of_occurrences == 3
    assert mock_sleep.call_count == 3 # Sleep should be called for each of the 3 retries

@pytest.mark.asyncio
async def test_api_status_error_5xx_allowances_exceeded(clean_controller, mocker):
    """
    Tests that a 5xx APIStatusError is re-raised if its allowances are exceeded.
    """
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        create_mock_api_error(500, "Server Error A"),
        create_mock_api_error(502, "Server Error B"),
        create_mock_api_error(503, "Server Error C"),
        create_mock_api_error(504, "Server Error D - Should fail") # Exceeds allowance of 3
    ])

    with pytest.raises(APIStatusError, match="Server Error D - Should fail"):
        await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 4
    assert clean_controller.total_interactions == 4
    assert clean_controller.error_record[APIStatusError].n_of_occurrences == 4
    assert mock_sleep.call_count == 3

@pytest.mark.asyncio
async def test_api_status_error_non_5xx_no_retry(clean_controller, mocker):
    """
    Tests that a non-5xx APIStatusError (e.g., 4xx) is not retried and is re-raised immediately.
    """
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        create_mock_api_error(401, "Unauthorized Access") # Not a 5xx error
    ])

    with pytest.raises(APIStatusError, match="Unauthorized Access"):
        await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 1
    assert clean_controller.total_interactions == 1
    # The APIStatusError allowance counter should not increment for non-5xx errors
    assert clean_controller.error_record[APIStatusError].n_of_occurrences == 0
    assert mock_sleep.call_count == 0

@pytest.mark.asyncio
async def test_unhandled_exception_no_retry(clean_controller, mocker):
    """
    Tests that an exception type not defined in error_allowances is re-raised immediately.
    """
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        TypeError("This is an unexpected error") # Not in configured allowances
    ])

    with pytest.raises(TypeError, match="This is an unexpected error"):
        await clean_controller.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 1
    assert clean_controller.total_interactions == 1
    assert mock_sleep.call_count == 0

@pytest.mark.asyncio
async def test_exception_with_zero_allowance(mocker):
    """
    Tests that an exception with 0 allowances defined will cause immediate re-raising
    after the first occurrence.
    """
    controller_zero_allowance = ExceptionRetryController({ValueError: 0})
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        ValueError("No retry allowed here")
    ])

    with pytest.raises(ValueError, match="No retry allowed here"):
        await controller_zero_allowance.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 1
    assert controller_zero_allowance.total_interactions == 1
    # Occurrences should still be incremented even if allowance is 0
    assert controller_zero_allowance.error_record[ValueError].n_of_occurrences == 1
    assert mock_sleep.call_count == 0 # No sleep if no allowances for retry

@pytest.mark.asyncio
async def test_api_status_error_not_in_allowances_config(mocker):
    """
    Tests that if APIStatusError is NOT in the initial error_allowances
    and a 5xx APIStatusError occurs, it's re-raised immediately.
    """
    controller_no_api_allowance = ExceptionRetryController({ValueError: 1}) # APIStatusError not in config
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mock_func = AsyncMock(side_effect=[
        create_mock_api_error(500, "API Error not explicitly configured")
    ])

    with pytest.raises(APIStatusError, match="API Error not explicitly configured"):
        await controller_no_api_allowance.execute_with_retries(mock_func, time_to_wait_between_retries=0.01)

    assert mock_func.call_count == 1
    assert controller_no_api_allowance.total_interactions == 1
    assert mock_sleep.call_count == 0
    # Ensure APIStatusError is not present in the error_record if it wasn't initially configured
    assert APIStatusError not in controller_no_api_allowance.error_record