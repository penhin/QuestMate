from dataclasses import dataclass
from typing import Any, Protocol

from tavily import TavilyClient

from config import Settings, get_settings
from schemas import Source


@dataclass(frozen=True)
class SearchRoute:
    source_type: str
    trust_score: float
    trust_label: str
    query_template: str


class SearchProvider(Protocol):
    async def search(self, query: str, game: str, max_results: int | None = None) -> list[Source]:
        ...


class TavilySearchProvider:
    routes = (
        SearchRoute("official", 0.95, "官方", "{game} 官方 攻略 {query}"),
        SearchRoute("wiki", 0.8, "百科", "{game} wiki 攻略 {query}"),
        SearchRoute("community", 0.55, "社区", "{game} 论坛 reddit steam community {query}"),
        SearchRoute("web", 0.45, "网页", "{game} 攻略 {query}"),
    )

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or (TavilyClient(api_key=self.settings.tavily_api_key) if self.settings.tavily_api_key else None)

    async def search(self, query: str, game: str, max_results: int | None = None) -> list[Source]:
        if self._client is None:
            return []

        total_results = max_results or self.settings.search_max_results
        per_route_results = min(3, max(1, total_results))
        sources_by_url: dict[str, Source] = {}

        for route in self.routes:
            search_query = route.query_template.format(game=game, query=query)
            result = self._client.search(
                query=search_query,
                max_results=per_route_results,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = item.get("url")
                if not url:
                    continue

                raw_score = float(item.get("score") or 0)
                weighted_score = raw_score * 0.7 + route.trust_score * 0.3
                source = Source(
                    title=item.get("title") or url,
                    url=url,
                    snippet=item.get("content"),
                    score=weighted_score,
                    source_type=route.source_type,
                    trust_score=route.trust_score,
                    trust_label=route.trust_label,
                )
                current = sources_by_url.get(url)
                if current is None or (source.score or 0) > (current.score or 0):
                    sources_by_url[url] = source

        return sorted(
            sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )[:total_results]
