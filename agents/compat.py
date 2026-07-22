"""Compatibility helpers for lightweight/custom model adapters."""

import inspect
from collections.abc import Callable
from typing import Any


def supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop optional artifacts unsupported by a legacy adapter signature."""
    signature = inspect.signature(callable_obj)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in signature.parameters}
