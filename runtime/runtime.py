"""Runtime lifecycle facade; deliberately unaware of game workflow content."""

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

from runtime.context import AgentContext
from runtime.executor import RuntimeExecutor


Result = TypeVar("Result")
logger = structlog.get_logger()


class QuestRuntime:
    def __init__(self, *, executor: RuntimeExecutor | None = None) -> None:
        self._executor = executor or RuntimeExecutor()

    async def execute(
        self,
        *,
        user_id: str | None,
        tools: dict[str, Any],
        operation: Callable[[], Awaitable[Result]],
    ) -> Result:
        context = AgentContext(user_id=user_id, tools=tools)
        result = await self._executor.execute(context, operation)
        usage = getattr(result, "usage", None)
        if isinstance(usage, dict):
            context.trace.record_usage(usage)
        logger.info(
            "runtime.request_completed",
            request_id=context.request_id,
            event_count=len(context.trace.events),
            token_usage=context.trace.token_usage.get("model_calls", 0),
        )
        return result
