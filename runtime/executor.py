"""Small generic executor used by the runtime lifecycle."""

from collections.abc import AsyncIterator, Awaitable, Callable
from time import perf_counter
from typing import TypeVar

from runtime.context import AgentContext, activate_context, deactivate_context


Result = TypeVar("Result")


class RuntimeExecutor:
    async def execute(self, context: AgentContext, operation: Callable[[], Awaitable[Result]]) -> Result:
        started = perf_counter()
        token = activate_context(context)
        try:
            result = await operation()
        except Exception:
            context.trace.record("request", started, outcome="error")
            raise
        finally:
            deactivate_context(token)
        context.trace.record("request", started)
        return result

    async def stream(
        self,
        context: AgentContext,
        operation: Callable[[], AsyncIterator[Result]],
    ) -> AsyncIterator[Result]:
        started = perf_counter()
        token = activate_context(context)
        try:
            async for item in operation():
                yield item
        except Exception:
            context.trace.record("request_stream", started, outcome="error")
            raise
        else:
            context.trace.record("request_stream", started)
        finally:
            deactivate_context(token)
