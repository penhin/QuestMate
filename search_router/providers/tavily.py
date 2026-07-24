"""Tavily provider adapter for SearchRouter's fallback contract.

The underlying backend remains in ``search.TavilySearchProvider`` during the
compatibility migration because it also owns game identity and MediaWiki
indexing.  Router code only sees this adapter, so the backend can be moved
without changing routing behavior or public APIs.
"""

from typing import Any

from schemas import GameResolution, SearchPlan, Source


class TavilyProvider:
    name = "tavily"

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def usage_snapshot(self) -> dict[str, int]:
        return self._backend._tavily_usage_snapshot()

    async def search(
        self,
        *,
        query: str,
        game: str,
        max_results: int,
        plan: SearchPlan | None,
        game_resolution: GameResolution,
    ) -> list[Source]:
        return await self._backend._search_with_tavily(
            query,
            game,
            max_results=max_results,
            plan=plan,
            game_resolution=game_resolution,
            skip_mediawiki=True,
        )
