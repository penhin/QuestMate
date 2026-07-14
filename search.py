from dataclasses import dataclass
from urllib.parse import urlparse
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
    search_noise_tokens = {
        "fandom",
        "fextralife",
        "wiki",
        "guide",
        "strategy",
        "weakness",
        "timing",
        "location",
        "merchant",
        "questline",
        "walkthrough",
        "build",
        "stats",
        "weapons",
        "talismans",
        "official",
        "patch",
        "notes",
        "update",
    }
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
        intent = (plan.intent if plan else "general") or "general"

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
                relevance_score = self._result_relevance_score(
                    item=item,
                    game=game,
                    question=f"{query} {search_query}",
                )
                if relevance_score <= 0:
                    continue

                raw_score = float(item.get("score") or 0)
                intent_score = self._intent_source_boost(intent=intent, source_type=search_source.source_type)
                domain_score = self._domain_quality_score(str(url))
                version_score = self._version_safety_score(
                    intent=intent,
                    source_type=search_source.source_type,
                    text=f"{item.get('title') or ''} {item.get('url') or ''} {item.get('content') or ''}",
                )
                weighted_score = (
                    raw_score * 0.25
                    + search_source.trust_score * 0.25
                    + relevance_score * 0.35
                    + intent_score * 0.1
                    + domain_score * 0.03
                    + version_score * 0.02
                )
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
        return TavilySearchProvider._result_relevance_score(item=item, game=game, question=question) > 0

    @staticmethod
    def _result_relevance_score(*, item: dict[str, Any], game: str, question: str) -> float:
        text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "url", "content")
        ).lower()
        if not TavilySearchProvider._is_same_game_surface(text=text, game=game, question=question):
            return 0

        game_tokens = relevance_tokens(game)
        game_token_set = set(game_tokens)
        question_tokens = [
            token
            for token in question_relevance_tokens(question)
            if token not in game_token_set and token not in TavilySearchProvider.search_noise_tokens
        ]

        has_game_match = not game_tokens or any(token in text for token in game_tokens)
        if not has_game_match:
            return 0

        if not question_tokens:
            return 0.45

        matched = sum(1 for token in question_tokens if token in text)
        if matched == 0:
            return 0

        title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).lower()
        focused_matches = sum(1 for token in question_tokens if token in title_url)
        coverage = matched / max(len(question_tokens), 1)
        focus_bonus = min(focused_matches * 0.12, 0.3)
        return min(1.0, 0.35 + coverage * 0.45 + focus_bonus)

    @staticmethod
    def _is_same_game_surface(*, text: str, game: str, question: str) -> bool:
        normalized_game = game.lower()
        normalized_question = question.lower()
        if "elden ring" in normalized_game and "nightreign" in text and "nightreign" not in normalized_question:
            return False
        return True

    @staticmethod
    def _intent_source_boost(*, intent: str, source_type: str) -> float:
        preferred = {
            "boss_strategy": {"wiki": 0.8, "community": 1.0, "web": 0.45, "official": 0.2},
            "item_location": {"wiki": 1.0, "web": 0.65, "community": 0.35, "official": 0.2},
            "quest_step": {"wiki": 1.0, "web": 0.65, "community": 0.45, "official": 0.2},
            "build": {"community": 1.0, "wiki": 0.65, "web": 0.45, "official": 0.2},
            "patch": {"official": 1.0, "wiki": 0.55, "web": 0.45, "community": 0.25},
            "lore": {"wiki": 0.9, "web": 0.65, "community": 0.35, "official": 0.2},
        }
        return preferred.get(intent, {}).get(source_type, 0.4)

    @staticmethod
    def _domain_quality_score(url: str) -> float:
        domain = urlparse(url).netloc.lower()
        if any(value in domain for value in ("wiki.gg", "fandom.com", "fextralife.com")):
            return 0.9
        if any(value in domain for value in ("bandainamco", "playstation.com", "steampowered.com")):
            return 0.85
        if any(value in domain for value in ("reddit.com", "steamcommunity.com")):
            return 0.55
        return 0.4

    @staticmethod
    def _version_safety_score(*, intent: str, source_type: str, text: str) -> float:
        lowered = text.lower()
        has_version_signal = any(
            token in lowered
            for token in ("patch", "version", "update", "1.", "版本", "补丁", "更新")
        )
        version_sensitive = intent in {"patch", "build", "boss_strategy"}
        if version_sensitive and source_type == "official":
            return 1.0
        if version_sensitive and has_version_signal:
            return 0.85
        if version_sensitive:
            return 0.45
        if intent in {"item_location", "quest_step", "lore"}:
            return 0.75
        return 0.55
