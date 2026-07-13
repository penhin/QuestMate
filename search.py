from dataclasses import dataclass
from typing import Any, Protocol

from tavily import TavilyClient

from config import Settings, get_settings
from query_tokens import question_relevance_tokens, relevance_tokens
from schemas import PlannedSearchQuery, SearchPlan, Source


@dataclass(frozen=True)
class SearchSource:
    source_type: str
    trust_score: float
    trust_label: str
    domains: tuple[str, ...] = ()
    query_templates: tuple[str, ...] = ()


class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
    ) -> list[Source]:
        ...


class TavilySearchProvider:
    sources = {
        "official": SearchSource(
            "official",
            0.95,
            "官方",
            query_templates=(
                "{game} official {query}",
                "{game} patch notes update {query}",
            ),
        ),
        "wiki": SearchSource(
            "wiki",
            0.8,
            "百科",
            domains=("fandom.com", "wiki.gg", "fextralife.com"),
        ),
        "community": SearchSource(
            "community",
            0.55,
            "社区",
            domains=("reddit.com", "steamcommunity.com"),
        ),
        "web": SearchSource(
            "web",
            0.45,
            "网页",
            query_templates=(
                "{game} guide {query}",
                "{game} 攻略 {query}",
            ),
        ),
    }
    fallback_plan = SearchPlan(
        intent="general",
        queries=(
            PlannedSearchQuery(source_type="wiki", query="{question}"),
            PlannedSearchQuery(source_type="community", query="{question}"),
            PlannedSearchQuery(source_type="web", query="{question}"),
        ),
    )

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or (TavilyClient(api_key=self.settings.tavily_api_key) if self.settings.tavily_api_key else None)

    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
    ) -> list[Source]:
        if self._client is None:
            return []

        total_results = max_results or self.settings.search_max_results
        per_query_results = min(2, max(1, total_results))
        sources_by_url: dict[str, Source] = {}
        search_queries = self._build_search_queries(game=game, question=query, plan=plan)

        for search_query, search_source in search_queries:
            result = self._client.search(
                query=search_query,
                max_results=per_query_results,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = item.get("url")
                if not url:
                    continue
                if not self._is_relevant_result(item=item, game=game, question=query):
                    continue

                raw_score = float(item.get("score") or 0)
                weighted_score = raw_score * 0.7 + search_source.trust_score * 0.3
                source = Source(
                    title=item.get("title") or url,
                    url=url,
                    snippet=item.get("content"),
                    score=weighted_score,
                    source_type=search_source.source_type,
                    trust_score=search_source.trust_score,
                    trust_label=search_source.trust_label,
                )
                current = sources_by_url.get(url)
                if current is None or (source.score or 0) > (current.score or 0):
                    sources_by_url[url] = source

        return sorted(
            sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )[:total_results]

    def _build_search_queries(
        self,
        *,
        game: str,
        question: str,
        plan: SearchPlan | None,
    ) -> list[tuple[str, SearchSource]]:
        planned_queries = list((plan or self.fallback_plan).queries)[:4] or list(self.fallback_plan.queries)
        built: list[tuple[str, SearchSource]] = []
        seen: set[str] = set()

        for planned in planned_queries:
            source = self.sources.get(planned.source_type, self.sources["web"])
            query = planned.query.replace("{question}", question).strip()

            candidates: list[str] = []
            candidates.extend(f"site:{domain} {game} {query}" for domain in source.domains)
            candidates.extend(template.format(game=game, query=query) for template in source.query_templates)
            if not candidates:
                candidates.append(f"{game} {query}")

            for candidate in candidates:
                normalized = " ".join(candidate.split())
                if normalized in seen:
                    continue
                built.append((normalized, source))
                seen.add(normalized)
                if len(built) >= 8:
                    return built

        return built

    @staticmethod
    def _is_relevant_result(*, item: dict[str, Any], game: str, question: str) -> bool:
        text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "url", "content")
        ).lower()
        game_tokens = relevance_tokens(game)
        question_tokens = question_relevance_tokens(question)

        has_game_match = not game_tokens or any(token in text for token in game_tokens)
        if not has_game_match:
            return False

        if not question_tokens:
            return True

        return any(token in text for token in question_tokens)
