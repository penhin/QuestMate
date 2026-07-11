from typing import Protocol

from tavily import TavilyClient

from config import Settings, get_settings
from schemas import Source


class SearchProvider(Protocol):
    async def search(self, query: str, game: str, max_results: int | None = None) -> list[Source]:
        ...


class TavilySearchProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = TavilyClient(api_key=self.settings.tavily_api_key) if self.settings.tavily_api_key else None

    async def search(self, query: str, game: str, max_results: int | None = None) -> list[Source]:
        if self._client is None:
            return []

        search_query = f"{game} 攻略 {query}"
        result = self._client.search(
            query=search_query,
            max_results=max_results or self.settings.search_max_results,
            include_answer=False,
            include_raw_content=False,
        )
        return [
            Source(
                title=item.get("title") or item.get("url", "Untitled"),
                url=item["url"],
                snippet=item.get("content"),
                score=item.get("score"),
            )
            for item in result.get("results", [])
            if item.get("url")
        ]

