import asyncio
from datetime import datetime, timezone
import re
from urllib.parse import urlparse, urlunparse
from typing import Any, Protocol

import structlog
from tavily import TavilyClient

from config import Settings, get_settings
from game_resolution import GameResolver
from quality_policy import (
    EXTERNAL_SEARCH_ATTEMPTS,
    PROGRESSIVE_STRICT_SOURCE_TARGET,
    SEARCH_NOISE_TOKENS,
    SOURCE_POLICIES,
    SourcePolicy,
    STABLE_FACT_INTENTS,
    VERSION_SCORE_POLICY,
    VERSION_SENSITIVE_INTENTS,
    VERSION_SIGNAL_TOKENS,
    source_domain_limit,
)
from query_tokens import question_relevance_tokens
from retrieval import (
    build_search_queries,
    is_high_quality_source,
    matches_game_name,
)
from retrieval.mediawiki_retriever import MediaWikiRetriever
from retrieval.source_builder import build_source
from schemas import GameResolution, PlannedSearchQuery, SearchPlan, Source
from mediawiki_client import MediaWikiClient
from search_cache import CachedSearchClient, RedisSearchCache, TTLSearchCache
from source_registry import GameSourceRegistry, game_source_registry


logger = structlog.get_logger()


class SearchProvider(Protocol):
    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        ...

    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
        game_resolution: GameResolution | None = None,
    ) -> list[Source]:
        ...


class ContentIndex(Protocol):
    async def index_content(self, **kwargs: Any) -> dict[str, Any]:
        ...


class TavilySearchProvider:
    search_noise_tokens = SEARCH_NOISE_TOKENS
    sources = SOURCE_POLICIES
    fallback_plan = SearchPlan(
        intent="general",
        queries=(
            PlannedSearchQuery(source_type="wiki", query="{question}"),
            PlannedSearchQuery(source_type="community", query="{question}"),
            PlannedSearchQuery(source_type="web", query="{question}"),
        ),
    )

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        mediawiki_client: Any | None = None,
        source_registry: GameSourceRegistry | None = None,
        content_index: ContentIndex | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        upstream_client = client or (
            TavilyClient(api_key=self.settings.tavily_api_key)
            if self.settings.tavily_api_key
            else None
        )
        self._client = None
        local_cache = TTLSearchCache(
            ttl_seconds=self.settings.tavily_search_cache_ttl_seconds,
            max_entries=self.settings.tavily_search_cache_max_entries,
        )
        shared_cache: Any = local_cache
        if client is None and self.settings.search_cache_use_redis:
            shared_cache = RedisSearchCache(
                redis_url=self.settings.redis_url,
                fallback=local_cache,
            )
        if upstream_client is not None:
            self._client = CachedSearchClient(upstream_client, shared_cache)
        self._game_resolver = GameResolver(self._client) if self._client is not None else None
        self._source_registry = source_registry if source_registry is not None else (
            game_source_registry if client is None else None
        )
        self._content_index = content_index
        self._mediawiki_client = mediawiki_client
        if self._mediawiki_client is None and client is None and self.settings.mediawiki_direct_search:
            self._mediawiki_client = MediaWikiClient()
        self._mediawiki_cache = shared_cache
        self._wiki_retriever = MediaWikiRetriever(
            client=self._mediawiki_client,
            cache=self._mediawiki_cache,
            settings=self.settings,
            source_policy=self.sources["wiki"],
            content_index=self._content_index,
            best_passage=self._best_evidence_passage,
            canonical_key=self._canonical_source_key,
            extract_version=self._extract_game_version,
        )

    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
        game_resolution: GameResolution | None = None,
    ) -> list[Source]:
        if self._client is None:
            return []
        usage_before = (self._client.upstream_calls, self._client.cache_hits)

        game_resolution = game_resolution or await self.resolve_game(game=game, question=query)
        total_results = max_results or self.settings.search_max_results
        per_query_results = min(6, max(2, total_results))
        strict_sources_by_url: dict[str, Source] = {}
        relaxed_sources_by_url: dict[str, Source] = {}
        search_queries = self._build_search_queries(
            game=game,
            question=query,
            plan=plan,
            database_domains=tuple(game_resolution.database_domains),
            game_aliases=tuple(game_resolution.aliases),
        )
        intent = (plan.intent if plan else "general") or "general"
        aliases = list((plan.aliases if plan else [])[:6])
        game_aliases = list(game_resolution.aliases)
        min_strict_results = min(PROGRESSIVE_STRICT_SOURCE_TARGET, total_results)

        direct_wiki_sources = await self._search_mediawiki_sources(
            game=game,
            question=query,
            aliases=aliases,
            planned_queries=[item.query for item in (plan.queries if plan else [])],
            game_aliases=game_aliases,
            database_domains=list(game_resolution.database_domains),
            intent=intent,
            max_results=total_results,
        )
        if direct_wiki_sources:
            self._log_search_usage(
                game=game,
                usage_before=usage_before,
                source_count=len(direct_wiki_sources),
                route="mediawiki",
            )
            return direct_wiki_sources[:total_results]

        first_wave_size = min(self.settings.tavily_first_wave_queries, len(search_queries))
        max_query_count = min(self.settings.tavily_max_queries_per_request, len(search_queries))

        await self._collect_sources(
            search_queries=search_queries[:first_wave_size],
            per_query_results=per_query_results,
            game=game,
            query=query,
            aliases=aliases,
            game_aliases=game_aliases,
            intent=intent,
            strict_sources_by_url=strict_sources_by_url,
            relaxed_sources_by_url=relaxed_sources_by_url,
            database_domains=list(game_resolution.database_domains),
        )

        if len(strict_sources_by_url) < min_strict_results and first_wave_size < max_query_count:
            await self._collect_sources(
                search_queries=search_queries[first_wave_size:max_query_count],
                per_query_results=per_query_results,
                game=game,
                query=query,
                aliases=aliases,
                game_aliases=game_aliases,
                intent=intent,
                strict_sources_by_url=strict_sources_by_url,
                relaxed_sources_by_url=relaxed_sources_by_url,
                database_domains=list(game_resolution.database_domains),
            )

        strict_ranked_sources = sorted(
            strict_sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )
        relaxed_ranked_sources = sorted(
            relaxed_sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )
        selected_sources = self._balanced_sources(
            strict_sources=strict_ranked_sources,
            relaxed_sources=relaxed_ranked_sources,
            total_results=total_results,
            min_strict_results=min_strict_results,
        )
        self._log_search_usage(
            game=game,
            usage_before=usage_before,
            source_count=len(selected_sources),
            route="tavily",
        )
        return selected_sources

    def _log_search_usage(
        self,
        *,
        game: str,
        usage_before: tuple[int, int],
        source_count: int,
        route: str,
    ) -> None:
        if self._client is None:
            return
        paid_before, hits_before = usage_before
        logger.info(
            "search.usage",
            game=game,
            route=route,
            tavily_paid_calls=max(0, self._client.upstream_calls - paid_before),
            tavily_cache_hits=max(0, self._client.cache_hits - hits_before),
            source_count=source_count,
        )

    async def _search_mediawiki_sources(
        self,
        *,
        game: str,
        question: str,
        aliases: list[str],
        planned_queries: list[str],
        game_aliases: list[str],
        database_domains: list[str],
        intent: str,
        max_results: int,
    ) -> list[Source]:
        return await self._wiki_retriever.search(
            game=game,
            question=question,
            aliases=aliases,
            planned_queries=planned_queries,
            game_aliases=game_aliases,
            database_domains=database_domains,
            max_results=max_results,
        )

    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        if self._client is None:
            return GameResolution(input_name=game, confirmed_name=game, confidence=0)
        if self._source_registry is not None:
            cached_resolution = await self._source_registry.get_resolution(game)
            if cached_resolution is not None and cached_resolution.is_confirmed:
                logger.info("source_registry.hit", game=game)
                return cached_resolution
        usage_before = (self._client.upstream_calls, self._client.cache_hits)
        try:
            resolution = await asyncio.wait_for(
                asyncio.to_thread(self._game_resolver.resolve, game=game, question=question),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            resolution = GameResolution(input_name=game, confirmed_name=game, confidence=0)
        self._log_search_usage(
            game=game,
            usage_before=usage_before,
            source_count=len(resolution.platform_urls) + len(resolution.database_domains),
            route="identity",
        )
        if self._source_registry is not None:
            await self._source_registry.upsert_resolution(resolution)
        return resolution

    async def _collect_sources(
        self,
        *,
        search_queries: list[tuple[str, SourcePolicy]],
        per_query_results: int,
        game: str,
        query: str,
        aliases: list[str],
        game_aliases: list[str],
        intent: str,
        strict_sources_by_url: dict[str, Source],
        relaxed_sources_by_url: dict[str, Source],
        database_domains: list[str],
    ) -> None:
        results = await self._fetch_search_results(search_queries, per_query_results)
        for (search_query, search_source), result in zip(search_queries, results, strict=True):
            for item in result.get("results", []):
                url = str(item.get("url") or "").strip()
                if search_source.source_type == "wiki":
                    result_domain = urlparse(str(url)).netloc.lower()
                    known_database = any(
                        result_domain == domain or result_domain.endswith(f".{domain}")
                        for domain in database_domains
                    )
                    title_url = f"{item.get('title') or ''} {url}".lower()
                    if not known_database and not matches_game_name(
                        text=title_url,
                        game_names=[game, *game_aliases],
                    ):
                        continue
                search_context = f"{query} {search_query} {' '.join(aliases)}"
                built = build_source(
                    item=item,
                    source_policy=search_source,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                    intent=intent,
                    best_passage=self._best_evidence_passage,
                    evidence_max_chars=self.settings.evidence_passage_max_chars,
                    version_safety_score=self._version_safety_score,
                    extract_version=self._extract_game_version,
                    parse_datetime=self._parse_source_datetime,
                )
                if built is None:
                    continue
                source = built.source
                source_key = self._canonical_source_key(str(url))
                current = relaxed_sources_by_url.get(source_key)
                if current is None or (source.score or 0) > (current.score or 0):
                    relaxed_sources_by_url[source_key] = source
                if is_high_quality_source(
                    item=built.searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                    source_type=search_source.source_type,
                ):
                    current = strict_sources_by_url.get(source_key)
                    if current is None or (source.score or 0) > (current.score or 0):
                        strict_sources_by_url[source_key] = source

    async def _fetch_search_results(
        self,
        search_queries: list[tuple[str, SourcePolicy]],
        per_query_results: int,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.settings.tavily_max_concurrency)

        async def fetch(query: str) -> dict[str, Any]:
            for attempt in range(EXTERNAL_SEARCH_ATTEMPTS):
                try:
                    async with semaphore:
                        return await asyncio.wait_for(
                            asyncio.to_thread(
                                self._client.search,
                                query=query,
                                max_results=per_query_results,
                                include_answer=False,
                                include_raw_content=self.settings.search_include_raw_content,
                            ),
                            timeout=self.settings.external_request_timeout_seconds,
                        )
                except Exception:
                    if attempt + 1 < EXTERNAL_SEARCH_ATTEMPTS:
                        await asyncio.sleep(0.15)
            return {"results": []}

        return await asyncio.gather(*(fetch(query) for query, _ in search_queries))

    @staticmethod
    def _best_evidence_passage(content: str, *, question: str, max_chars: int = 1600) -> str:
        """Return a compact passage around the best entity matches in a page."""
        cleaned = re.sub(r"\s+", " ", content).strip()
        if not cleaned or len(cleaned) <= max_chars:
            return cleaned

        tokens = question_relevance_tokens(question)
        candidates: list[str] = [cleaned[:max_chars]]
        lowered = cleaned.lower()
        for token in tokens:
            start = 0
            for _ in range(3):
                position = lowered.find(token, start)
                if position < 0:
                    break
                window_start = max(0, position - max_chars // 4)
                window_end = min(len(cleaned), window_start + max_chars)
                candidates.append(cleaned[window_start:window_end].strip())
                start = position + len(token)

        def passage_score(passage: str) -> tuple[int, int]:
            lowered_passage = passage.lower()
            matched = sum(1 for token in tokens if token in lowered_passage)
            occurrences = sum(lowered_passage.count(token) for token in tokens)
            return matched, occurrences

        return max(candidates, key=passage_score)

    def _build_search_queries(
        self,
        *,
        game: str,
        question: str,
        plan: SearchPlan | None,
        database_domains: tuple[str, ...] = (),
        game_aliases: tuple[str, ...] = (),
    ) -> list[tuple[str, SourcePolicy]]:
        selected_plan = plan or self.fallback_plan
        if not selected_plan.queries:
            selected_plan = self.fallback_plan
        return build_search_queries(
            game=game,
            question=question,
            plan=selected_plan,
            sources=self.sources,
            database_domains=database_domains,
            game_aliases=game_aliases,
        )

    @staticmethod
    def _canonical_source_key(url: str) -> str:
        parsed = urlparse(url)
        if any(value in parsed.netloc.lower() for value in ("steamcommunity.com", "reddit.com")):
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return url

    @staticmethod
    def _limit_source_diversity(sources: list[Source], *, total_results: int) -> list[Source]:
        selected: list[Source] = []
        domain_counts: dict[str, int] = {}
        for source in sources:
            domain = urlparse(str(source.url)).netloc.lower()
            limit = source_domain_limit(domain)
            if domain_counts.get(domain, 0) >= limit:
                continue
            selected.append(source)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected) >= total_results:
                return selected
        return selected

    @classmethod
    def _balanced_sources(
        cls,
        *,
        strict_sources: list[Source],
        relaxed_sources: list[Source],
        total_results: int,
        min_strict_results: int,
    ) -> list[Source]:
        selected = cls._limit_source_diversity(strict_sources, total_results=total_results)
        if len(selected) >= min_strict_results or len(selected) >= total_results:
            return selected

        selected_keys = {cls._canonical_source_key(str(source.url)) for source in selected}
        fill_sources = [
            source
            for source in relaxed_sources
            if cls._canonical_source_key(str(source.url)) not in selected_keys
        ]
        combined = selected + fill_sources
        return cls._limit_source_diversity(combined, total_results=total_results)

    @staticmethod
    def _version_safety_score(*, intent: str, source_type: str, text: str) -> float:
        lowered = text.lower()
        has_version_signal = any(token in lowered for token in VERSION_SIGNAL_TOKENS)
        version_sensitive = intent in VERSION_SENSITIVE_INTENTS
        if version_sensitive and source_type == "official":
            return VERSION_SCORE_POLICY.official_sensitive
        if version_sensitive and has_version_signal:
            return VERSION_SCORE_POLICY.versioned_sensitive
        if version_sensitive:
            return VERSION_SCORE_POLICY.undated_sensitive
        if intent in STABLE_FACT_INTENTS:
            return VERSION_SCORE_POLICY.stable_fact
        return VERSION_SCORE_POLICY.default

    @staticmethod
    def _extract_game_version(text: str) -> str | None:
        match = re.search(r"(?:patch|version|ver\.?|v|补丁|版本)\s*([0-9]+(?:\.[0-9]+){1,3})", text, re.I)
        return match.group(1) if match else None

    @staticmethod
    def _parse_source_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
