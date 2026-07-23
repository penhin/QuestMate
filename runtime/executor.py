"""Small generic executor used by the runtime lifecycle."""

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TypeVar

from runtime.context import AgentContext


Result = TypeVar("Result")


class RuntimeExecutor:
    async def execute(self, context: AgentContext, operation: Callable[[], Awaitable[Result]]) -> Result:
        started = perf_counter()
        try:
            result = await operation()
        except Exception:
            context.trace.record("request", started, outcome="error")
            raise
        context.trace.record("request", started)
        return result
