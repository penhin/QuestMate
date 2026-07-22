"""Retrieval specialist boundary."""

from collections.abc import Awaitable, Callable
from typing import Any


class RetrievalAgent:
    """Executes the retrieval pipeline supplied by the orchestrator."""

    def __init__(self, retrieve: Callable[..., Awaitable[Any]]) -> None:
        self._retrieve = retrieve

    async def investigate(self, **kwargs: Any) -> Any:
        return await self._retrieve(**kwargs)
