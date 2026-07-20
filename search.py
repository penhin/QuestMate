import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import re
from urllib.parse import urlparse, urlunparse
from typing import Any, Protocol

import structlog
from tavily import TavilyClient

from config import Settings, get_settings
from game_resolution import GameResolver, is_candidate_identity_url, select_game_candidate
from quality_policy import (
    EXTERNAL_SEARCH_ATTEMPTS,
    MAX_PAID_SEARCH_CALLS_PER_REQUEST,
    PROGRESSIVE_STRICT_SOURCE_TARGET,
    SOURCE_POLICIES,
    SourcePolicy,
    STABLE_FACT_INTENTS,
    VERSION_SCORE_POLICY,
    VERSION_SENSITIVE_INTENTS,
    is_version_sensitive_question,
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
from retrieval.source_quality import required_entity_groups_for_query
from retrieval.wiki_domains import has_strong_wiki_page_signal
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

    def usage_snapshot(self) -> dict[str, int]:
        """Return process counters for request-scoped delta accounting."""
        if self._client is None:
            return {"tavily_paid_calls": 0, "tavily_cache_hits": 0}
        return self._client.request_usage()

    def usage_scope(self):
        if self._client is None:
            return nullcontext()
        return self._client.usage_scope(max_paid_calls=MAX_PAID_SEARCH_CALLS_PER_REQUEST)

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
        confirmed_aliases = list(dict.fromkeys([
            *(
                [game_resolution.confirmed_name]
                if game_resolution.confirmed_name
                and game_resolution.confirmed_name.casefold() != game.casefold()
                else []
            ),
            *game_resolution.aliases,
        ]))
        search_queries = self._build_search_queries(
            game=game,
            question=query,
            plan=plan,
            database_domains=tuple(game_resolution.database_domains),
            game_aliases=tuple(confirmed_aliases),
        )
        intent = (plan.intent if plan else "general") or "general"
        version_sensitive = bool(plan and plan.version_sensitive) or is_version_sensitive_question(query)
        aliases = list((plan.aliases if plan else [])[:6])
        named_entity_groups = list((plan.named_entity_groups if plan else [])[:4])
        game_aliases = confirmed_aliases
        min_strict_results = min(PROGRESSIVE_STRICT_SOURCE_TARGET, total_results)

        direct_wiki_sources = await self._search_mediawiki_sources(
            game=game,
            question=query,
            aliases=aliases,
            planned_queries=[item.query for item in (plan.queries if plan else [])],
            game_aliases=game_aliases,
            database_domains=list(game_resolution.database_domains),
            intent=intent,
            version_sensitive=version_sensitive,
            max_results=total_results,
            named_entity_groups=named_entity_groups,
        )
        # Stable fact lookups can finish on a sufficiently populated direct
        # database result. Version-sensitive, strategic, mechanical, general,
        # and refinement requests keep the independent open-web portfolio.
        can_finish_on_database = (
            intent in STABLE_FACT_INTENTS
            and not version_sensitive
            and len(direct_wiki_sources) >= min_strict_results
            and not (plan and plan.refinement)
        )
        if can_finish_on_database:
            self._log_search_usage(
                game=game,
                usage_before=usage_before,
                source_count=len(direct_wiki_sources),
                route="mediawiki",
            )
            return direct_wiki_sources[:total_results]
        for source in direct_wiki_sources:
            source_key = self._canonical_source_key(str(source.url))
            current = relaxed_sources_by_url.get(source_key)
            if current is None or (source.score or 0) > (current.score or 0):
                relaxed_sources_by_url[source_key] = source
                strict_sources_by_url[source_key] = source

        if not search_queries:
            selected = self._balanced_sources(
                strict_sources=list(strict_sources_by_url.values()),
                relaxed_sources=list(relaxed_sources_by_url.values()),
                total_results=total_results,
                min_strict_results=min_strict_results,
            )
            self._log_search_usage(
                game=game,
                usage_before=usage_before,
                source_count=len(selected),
                route="mediawiki",
            )
            return selected

        max_query_count = min(self.settings.tavily_max_queries_per_request, len(search_queries))
        if not game_resolution.is_confirmed:
            max_query_count = min(
                max_query_count,
                max(1, self.settings.tavily_max_queries_per_request - self.settings.tavily_unconfirmed_identity_reserve),
            )
        # Relation-sensitive plans reserve a query slot for the one permitted
        # investigation refinement.  A broad first wave can find an entity,
        # yet still leave its requested condition unsupported; spending the
        # entire search budget before that check prevents recovery.
        relation_verification = bool(plan and plan.requires_relation_verification)
        search_depth = (
            self.settings.tavily_relation_search_depth
            if relation_verification
            else self.settings.tavily_search_depth
        )
        first_wave_limit = 2 if relation_verification else self.settings.tavily_first_wave_queries
        first_wave_size = min(first_wave_limit, max_query_count)

        await self._collect_sources(
            search_queries=search_queries[:first_wave_size],
            per_query_results=per_query_results,
            game=game,
            query=query,
            aliases=aliases,
            game_aliases=game_aliases,
            intent=intent,
            version_sensitive=version_sensitive,
            strict_sources_by_url=strict_sources_by_url,
            relaxed_sources_by_url=relaxed_sources_by_url,
            database_domains=list(game_resolution.database_domains),
            official_domains=self._resolution_authority_domains(game_resolution),
            named_entity_groups=named_entity_groups,
            search_depth=search_depth,
        )

        if (
            not relation_verification
            and len(strict_sources_by_url) < min_strict_results
            and first_wave_size < max_query_count
        ):
            await self._collect_sources(
                search_queries=search_queries[first_wave_size:max_query_count],
                per_query_results=per_query_results,
                game=game,
                query=query,
                aliases=aliases,
                game_aliases=game_aliases,
                intent=intent,
                version_sensitive=version_sensitive,
                strict_sources_by_url=strict_sources_by_url,
                relaxed_sources_by_url=relaxed_sources_by_url,
                database_domains=list(game_resolution.database_domains),
                official_domains=self._resolution_authority_domains(game_resolution),
                named_entity_groups=named_entity_groups,
                search_depth=search_depth,
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
            route="mediawiki+tavily" if direct_wiki_sources else "tavily",
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
        version_sensitive: bool,
        max_results: int,
        named_entity_groups: list[list[str]],
    ) -> list[Source]:
        return await self._wiki_retriever.search(
            game=game,
            question=question,
            aliases=aliases,
            planned_queries=planned_queries,
            game_aliases=game_aliases,
            database_domains=database_domains,
            max_results=max_results,
            named_entity_groups=named_entity_groups,
        )

    async def wait_for_background_tasks(self) -> None:
        await self._wiki_retriever.wait_for_background_tasks()

    async def get_cached_game_resolution(self, game: str) -> GameResolution | None:
        """Return only a previously server-verified, unambiguous identity.

        This is deliberately separate from ``resolve_game``: callers can use
        it as a no-search fast path without treating a client hint or a search
        result's source label as identity proof.
        """
        if self._source_registry is None:
            return None
        resolution = await self._source_registry.get_resolution(game)
        if resolution is not None and resolution.is_confirmed and not resolution.ambiguous:
            logger.info("source_registry.hit", game=game)
            return resolution
        return None

    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        if self._client is None:
            return GameResolution(input_name=game, confirmed_name=game, confidence=0)
        usage_before = (self._client.upstream_calls, self._client.cache_hits)
        resolution = await self._resolve_game_with_retry(game=game, question=question)
        self._log_search_usage(
            game=game,
            usage_before=usage_before,
            source_count=len(resolution.platform_urls) + len(resolution.database_domains),
            route="identity",
        )
        if self._source_registry is not None and resolution.is_confirmed and not resolution.ambiguous:
            await self._source_registry.upsert_resolution(resolution)
        return resolution

    async def _resolve_game_with_retry(
        self,
        *,
        game: str,
        question: str | None,
    ) -> GameResolution:
        """Bound identity lookup separately from the slower guide workflow."""
        for attempt in range(1, self.settings.identity_resolution_attempts + 1):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._game_resolver.resolve, game=game, question=question),
                    timeout=self.settings.identity_resolution_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "identity_resolution.failed",
                    stage="identity_resolution",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
                if attempt < self.settings.identity_resolution_attempts:
                    await asyncio.sleep(0.15)
        return GameResolution(input_name=game, confirmed_name=game, confidence=0)

    async def select_game_candidate(
        self,
        *,
        game: str,
        selected_url: str,
        question: str | None = None,
    ) -> GameResolution:
        """Validate a UI candidate against a fresh identity result, bypassing registry ambiguity."""
        if self._client is None or not is_candidate_identity_url(selected_url):
            return GameResolution(input_name=game, confirmed_name=game, confidence=0)
        usage_before = (self._client.upstream_calls, self._client.cache_hits)
        discovered = await self._resolve_game_with_retry(game=game, question=question)
        selected = select_game_candidate(discovered, selected_url=selected_url)
        self._log_search_usage(
            game=game,
            usage_before=usage_before,
            source_count=len(selected.platform_urls) if selected is not None else 0,
            route="identity-selection",
        )
        if selected is not None:
            return selected
        # The selected opaque identity disappeared from the fresh result set.
        # Never reinterpret that user choice as a different currently-ranked
        # game; return the fresh candidates for explicit confirmation instead.
        return GameResolution(
            input_name=game,
            confirmed_name=game,
            confidence=0,
            candidates=discovered.candidates,
            ambiguous=bool(discovered.candidates),
        )

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
        version_sensitive: bool,
        strict_sources_by_url: dict[str, Source],
        relaxed_sources_by_url: dict[str, Source],
        database_domains: list[str],
        official_domains: list[str],
        named_entity_groups: list[list[str]],
        search_depth: str,
    ) -> None:
        results = await self._fetch_search_results(
            search_queries, per_query_results, search_depth=search_depth
        )
        for (search_query, search_source), result in zip(search_queries, results, strict=True):
            required_entity_groups = required_entity_groups_for_query(
                named_entity_groups,
                search_query,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "").strip()
                effective_source = self._effective_source_policy(
                    configured=search_source,
                    url=url,
                    database_domains=database_domains,
                    official_domains=official_domains,
                )
                if effective_source.source_type == "wiki":
                    result_domain = urlparse(str(url)).netloc.lower()
                    known_database = any(
                        result_domain == domain or result_domain.endswith(f".{domain}")
                        for domain in database_domains
                    )
                    title_url = f"{item.get('title') or ''} {url}"
                    if not known_database and not matches_game_name(
                        text=title_url,
                        game_names=[game, *game_aliases],
                    ):
                        continue
                # Score a page against the one outbound query that produced
                # it.  That query may use either the player's localized name
                # or one translated alias, but never requires both to appear
                # on the same page.
                search_context = search_query
                query_confirms_game = any(
                    name.strip() and name.casefold() in search_query.casefold()
                    for name in [game, *game_aliases]
                )
                built = build_source(
                    item=item,
                    source_policy=effective_source,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                    intent=intent,
                    version_sensitive=version_sensitive,
                    best_passage=self._best_evidence_passage,
                    evidence_max_chars=self.settings.evidence_passage_max_chars,
                    version_safety_score=self._version_safety_score,
                    extract_version=self._extract_game_version,
                    parse_datetime=self._parse_source_datetime,
                    required_entity_groups=required_entity_groups,
                    query_confirms_game=query_confirms_game,
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
                    source_type=effective_source.source_type,
                    required_entity_groups=required_entity_groups,
                ):
                    current = strict_sources_by_url.get(source_key)
                    if current is None or (source.score or 0) > (current.score or 0):
                        strict_sources_by_url[source_key] = source

    def _effective_source_policy(
        self,
        *,
        configured: SourcePolicy,
        url: str,
        database_domains: list[str],
        official_domains: list[str],
    ) -> SourcePolicy:
        """Classify the actual result page; query intent is only a weak prior."""
        parsed = urlparse(url)
        domain = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")

        def matches_any(candidates: list[str]) -> bool:
            normalized = [
                value.casefold().split(":", 1)[0].strip(".").removeprefix("www.")
                for value in candidates
                if value
            ]
            return any(domain == value or domain.endswith(f".{value}") for value in normalized)

        if matches_any(["reddit.com", "steamcommunity.com"]):
            return self.sources["community"]
        if has_strong_wiki_page_signal(domain, url=url):
            return self.sources["wiki"]
        if matches_any(official_domains):
            return self.sources["official"]
        if configured.source_type in {"official", "wiki", "community"}:
            return self.sources["web"]
        return configured

    @staticmethod
    def _resolution_authority_domains(resolution: GameResolution) -> list[str]:
        domains: list[str] = []
        for url in [*resolution.official_urls, *resolution.platform_urls]:
            domain = urlparse(str(url)).netloc.casefold().split(":", 1)[0].removeprefix("www.")
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    async def _fetch_search_results(
        self,
        search_queries: list[tuple[str, SourcePolicy]],
        per_query_results: int,
        *,
        search_depth: str,
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
                                search_depth=search_depth,
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
        anchors = TavilySearchProvider._evidence_anchor_phrases(question)
        candidates: list[str] = [cleaned[:max_chars]]
        lowered = cleaned.lower()
        for anchor in anchors:
            start = 0
            while (position := lowered.find(anchor, start)) >= 0:
                candidates.append(
                    TavilySearchProvider._evidence_window(cleaned, focus=position, max_chars=max_chars)
                )
                start = position + len(anchor)
        for token in tokens:
            start = 0
            for _ in range(20):
                position = lowered.find(token, start)
                if position < 0:
                    break
                candidates.append(
                    TavilySearchProvider._evidence_window(cleaned, focus=position, max_chars=max_chars)
                )
                start = position + len(token)

        def passage_score(passage: str) -> tuple[int, int, int]:
            lowered_passage = passage.lower()
            anchor_matches = sum(1 for anchor in anchors if anchor in lowered_passage)
            matched = sum(1 for token in tokens if token in lowered_passage)
            occurrences = sum(lowered_passage.count(token) for token in tokens)
            return anchor_matches, matched, occurrences

        focused = max(candidates, key=passage_score)
        return TavilySearchProvider._combine_page_lead(
            cleaned,
            focused=focused,
            anchors=anchors,
            tokens=tokens,
            max_chars=max_chars,
        )

    @staticmethod
    def _combine_page_lead(
        content: str,
        *,
        focused: str,
        anchors: list[str],
        tokens: list[str],
        max_chars: int,
    ) -> str:
        """Preserve a guide page's overview while retaining its distant focused claim."""
        focused_position = content.find(focused[: min(120, len(focused))])
        if focused_position < max_chars // 2:
            return focused[:max_chars]

        lead_budget = max_chars * 3 // 5
        lead = content[:lead_budget]
        boundaries = [lead.rfind(mark) for mark in ".!?。！？;；"]
        boundary = max(boundaries, default=-1)
        if boundary >= lead_budget // 2:
            lead = lead[: boundary + 1]
        remaining = max_chars - len(lead) - 2
        if remaining <= 0:
            return lead[:max_chars]

        lowered = focused.casefold()
        positions = [lowered.find(value) for value in [*anchors, *tokens] if value]
        positions = [position for position in positions if position >= 0]
        focus = min(positions, default=0)
        detail = TavilySearchProvider._evidence_window(focused, focus=focus, max_chars=remaining)
        return f"{lead}\n\n{detail}"[:max_chars]

    @staticmethod
    def _evidence_window(content: str, *, focus: int, max_chars: int) -> str:
        """Keep the focused claim's subject by starting at a nearby sentence boundary."""
        search_start = max(0, focus - max_chars // 2)
        prefix = content[search_start:focus]
        boundaries = [prefix.rfind(mark) for mark in ".!?。！？;；"]
        boundary = max(boundaries, default=-1)
        window_start = search_start + boundary + 1 if boundary >= 0 else search_start
        while window_start < focus and content[window_start].isspace():
            window_start += 1
        return content[window_start : window_start + max_chars].strip()

    @staticmethod
    def _evidence_anchor_phrases(value: str) -> list[str]:
        """Extract identifier-bearing phrases without knowing any game's entities."""
        stop_words = {
            "access", "enter", "exact", "find", "guide", "how", "into", "location",
            "outside", "requirements", "route", "the", "to", "where",
        }
        words = re.findall(r"[a-z][a-z'-]*|[a-z]*\d[a-z0-9._-]*|\d{1,6}", value.casefold())
        anchors: list[str] = []
        for index, word in enumerate(words):
            if not any(char.isdigit() for char in word):
                continue
            left = max(0, index - 2)
            right = min(len(words), index + 3)
            local_pairs = [
                (position, words[position])
                for position in range(left, right)
                if words[position] not in stop_words
            ]
            local = [token for _position, token in local_pairs]
            identifier_index = next(
                local_index for local_index, (position, _token) in enumerate(local_pairs) if position == index
            )
            for start in range(max(0, identifier_index - 2), identifier_index + 1):
                for end in range(identifier_index + 1, min(len(local), identifier_index + 3) + 1):
                    phrase = " ".join(local[start:end])
                    if phrase != word and phrase not in anchors:
                        anchors.append(phrase)
        return sorted(anchors, key=len, reverse=True)[:12]

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
    def _version_safety_score(
        *,
        intent: str,
        source_type: str,
        text: str,
        version_sensitive: bool = False,
    ) -> float:
        lowered = text.lower()
        has_version_signal = is_version_sensitive_question(lowered)
        version_sensitive = version_sensitive or intent in VERSION_SENSITIVE_INTENTS
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
