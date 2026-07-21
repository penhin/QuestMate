"""Bounded specialist agents used by the QuestMate orchestrator.

The specialists deliberately exchange existing typed domain objects instead of
free-form chat messages.  They cannot call each other; only ``QuestAgent``
chooses the sequence, owns request timeouts, and owns model/search budgets.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any

from schemas import (
    ChatRequest,
    GameResolution,
    InvestigationState,
    SearchPlan,
    SessionMessage,
    Source,
)


@dataclass(frozen=True)
class AgentTrace:
    """Aggregate-safe record of one specialist hand-off."""

    agent: str
    action: str
    source_count: int = 0
    refined: bool = False


class IdentityAgent:
    """Owns game identity context and post-retrieval recovery only."""

    def __init__(
        self,
        *,
        initial_context: Callable[[ChatRequest], Awaitable[GameResolution]],
        recover_context: Callable[..., Awaitable[GameResolution]],
    ) -> None:
        self._initial_context = initial_context
        self._recover_context = recover_context

    async def initial(self, request: ChatRequest) -> GameResolution:
        return await self._initial_context(request)

    async def recover(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        current: GameResolution,
    ) -> GameResolution:
        return await self._recover_context(request=request, sources=sources, current=current)


class PlanningAgent:
    """Turns a player request into the bounded retrieval artifact ``SearchPlan``."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def plan(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        game_resolution: GameResolution,
    ) -> SearchPlan:
        return await self._llm.plan_search(
            request=request, history=history, game_resolution=game_resolution
        )


class EvidenceAgent:
    """Assesses evidence gaps; it cannot perform retrieval itself."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.supports_update_investigation = callable(
            getattr(llm, "update_investigation", None)
        )

    async def update_investigation(self, **kwargs: Any) -> InvestigationState:
        update = getattr(self._llm, "update_investigation", None)
        if not callable(update):
            raise RuntimeError("Underlying LLM does not support investigation updates")
        return await update(**_supported_kwargs(update, kwargs))

    async def refine_search_plan(self, **kwargs: Any) -> SearchPlan | None:
        refine = getattr(self._llm, "refine_search_plan", None)
        if not callable(refine):
            return None
        return await refine(**_supported_kwargs(refine, kwargs))


class RetrievalAgent:
    """Executes the retrieval pipeline supplied by the orchestrator."""

    def __init__(self, retrieve: Callable[..., Awaitable[Any]]) -> None:
        self._retrieve = retrieve

    async def investigate(self, **kwargs: Any) -> Any:
        return await self._retrieve(**kwargs)


class AnswerAgent:
    """Renders an answer solely from verified sources and investigation state."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def answer(self, **kwargs: Any) -> str:
        return await self._llm.answer(**_supported_kwargs(self._llm.answer, kwargs))

    async def stream_answer(self, **kwargs: Any):
        async for chunk in self._llm.stream_answer(
            **_supported_kwargs(self._llm.stream_answer, kwargs)
        ):
            yield chunk


def _supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep optional cross-agent artifacts compatible with lightweight adapters."""
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in signature.parameters}
