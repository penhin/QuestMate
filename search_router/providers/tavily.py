"""Tavily provider adapter for SearchRouter's fallback contract.

The adapter receives explicit callables instead of a search backend, so router
code has no dependency on legacy backend-private methods.
"""

from collections.abc import Awaitable, Callable

from schemas import GameResolution, SearchPlan, Source


class TavilyProvider:
    name = "tavily"

    def __init__(
        self,
        *,
        search: Callable[..., Awaitable[list[Source]]],
        usage_snapshot: Callable[[], dict[str, int]],
    ) -> None:
        self._search = search
        self._usage_snapshot = usage_snapshot

    def usage_snapshot(self) -> dict[str, int]:
        return self._usage_snapshot()

    async def search(
        self,
        *,
        query: str,
        game: str,
        max_results: int,
        plan: SearchPlan | None,
        game_resolution: GameResolution,
    ) -> list[Source]:
        return await self._search(
            query=query,
            game=game,
            max_results=max_results,
            plan=plan,
            game_resolution=game_resolution,
        )
