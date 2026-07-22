"""Search-planning specialist boundary."""

from typing import Any

from schemas import ChatRequest, GameResolution, SearchPlan, SessionMessage


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
