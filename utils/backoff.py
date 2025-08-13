#!/usr/bin/env python3
"""
Lightweight exponential backoff utilities for resiliency around flaky MT5 calls.
"""

import time
import random
from typing import Callable, TypeVar, Any, Tuple


F = TypeVar("F", bound=Callable[..., Any])


def backoff(
    retries: int = 3,
    base_delay_seconds: float = 0.25,
    max_delay_seconds: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[type, ...] = (Exception,),
):
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        retries: Maximum number of retries.
        base_delay_seconds: Initial delay between retries.
        max_delay_seconds: Maximum delay between retries.
        jitter: Whether to add random jitter to delay.
        retry_on: Exception types to retry on.
    """

    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            delay = base_delay_seconds
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # type: ignore[misc]
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(min(max_delay_seconds, delay + (random.uniform(0, delay) if jitter else 0)))
                    delay = min(max_delay_seconds, delay * 2)
        return wrapper  # type: ignore[return-value]

    return decorator


