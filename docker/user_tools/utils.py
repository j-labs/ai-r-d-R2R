import asyncio
import functools

from core import logger


def async_timeout(timeout):
    """Decorator to add timeout to async functions"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(
                    f"Function {func.__name__} timed out after {timeout} seconds"
                )
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

        return wrapper

    return decorator
