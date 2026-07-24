"""Deterministic SearXNG-first routing over existing evidence contracts."""

from dataclasses import dataclass
from typing import Any

import structlog

from quality_policy import PROGRESSIVE_STRICT_SOURCE_TARGET, STABLE_FACT_INTENTS
from schemas import GameResolution, SearchPlan, Source
from search_router.health import ProviderHealth
from search_router.providers import SearxngProvider, TavilyProvider

logger = structlog.get_logger()


@dataclass(frozen=True)
class SearchRouteDecision:
    provider: str
    reason: str
    fallback_chain: tuple[str, ...]
    cache_eligible: bool
    budget_remaining: int


class SearchRouter:
    """Route open-web recall without changing SearchProvider's public API."""

    def __init__(
        self, *, legacy_tavily: Any, searxng: SearxngProvider, settings: Any,
        tavily: TavilyProvider | None = None,
    ) -> None:
        self.legacy_tavily = legacy_tavily
        self.tavily = tavily or TavilyProvider(legacy_tavily)
        self.searxng = searxng
        self.settings = settings
        self.health = ProviderHealth(cooldown_seconds=settings.search_provider_cooldown_seconds)
        self.last_decision: SearchRouteDecision | None = None

    def usage_snapshot(self) -> dict[str, int]:
        usage = dict(self.tavily.usage_snapshot())
        usage["searxng_calls"] = self.searxng.calls
        return usage

    async def search(self, *, query: str, game: str, max_results: int, plan: SearchPlan | None, game_resolution: GameResolution) -> list[Source]:
        intent = plan.intent if plan else "general"
        aliases = list((plan.aliases if plan else [])[:6])
        entities = list((plan.named_entity_groups if plan else [])[:4])
        direct = await self.legacy_tavily._search_mediawiki_sources(
            game=game, question=query, aliases=aliases,
            planned_queries=[item.query for item in (plan.queries if plan else [])],
            game_aliases=game_resolution.aliases,
            database_domains=list(game_resolution.database_domains), intent=intent,
            version_sensitive=bool(plan and plan.version_sensitive), max_results=max_results,
            named_entity_groups=entities,
        )
        enough = len(direct) >= min(PROGRESSIVE_STRICT_SOURCE_TARGET, max_results)
        if intent in STABLE_FACT_INTENTS and enough and not (plan and plan.version_sensitive):
            self._decision("mediawiki", "direct_database_evidence", 0)
            return direct[:max_results]
        if self.searxng.configured and self.health.available("searxng"):
            try:
                queries = self.legacy_tavily._build_search_queries(
                    game=game, question=query, plan=plan,
                    database_domains=tuple(game_resolution.database_domains),
                    game_aliases=tuple(game_resolution.aliases),
                )
                results: list[Source] = []
                for search_query, _policy in queries[:self.settings.searxng_max_queries_per_request]:
                    results.extend(await self.searxng.search(search_query, max_results=max_results))
                merged = self._dedupe([*direct, *results])[:max_results]
                if len(merged) >= min(PROGRESSIVE_STRICT_SOURCE_TARGET, max_results) or results:
                    self.health.succeeded("searxng")
                    self._decision("searxng", "default_open_web", self.settings.tavily_fallback_max_calls)
                    return merged
            except Exception as exc:
                self.health.failed("searxng")
                logger.warning("search_router.provider_failed", provider="searxng", error_type=type(exc).__name__)
        if self.health.available("tavily") and self.settings.tavily_fallback_max_calls > 0:
            try:
                tavily = await self.tavily.search(
                    query=query, game=game, max_results=max_results,
                    plan=plan, game_resolution=game_resolution,
                )
                self.health.succeeded("tavily")
                self._decision("tavily", "searxng_insufficient_or_unavailable", 0)
                return self._dedupe([*direct, *tavily])[:max_results]
            except Exception as exc:
                self.health.failed("tavily")
                logger.warning("search_router.provider_failed", provider="tavily", error_type=type(exc).__name__)
        self._decision("conservative", "providers_unavailable", 0)
        return direct[:max_results]

    def _decision(self, provider: str, reason: str, budget_remaining: int) -> None:
        self.last_decision = SearchRouteDecision(
            provider=provider, reason=reason,
            fallback_chain=("mediawiki", "searxng", "tavily", "conservative"),
            cache_eligible=True, budget_remaining=budget_remaining,
        )
        logger.info("search_router.decision", provider=provider, reason=reason, budget_remaining=budget_remaining)

    @staticmethod
    def _dedupe(sources: list[Source]) -> list[Source]:
        selected: dict[str, Source] = {}
        for source in sources:
            key = str(source.url).rstrip("/").casefold()
            if key not in selected or (source.score or 0) > (selected[key].score or 0):
                selected[key] = source
        return sorted(selected.values(), key=lambda source: (source.score or 0, source.trust_score), reverse=True)
