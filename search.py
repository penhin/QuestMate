import asyncio
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from threading import Lock
from time import monotonic
from urllib.parse import quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from typing import Any, Protocol

from redis import Redis
import structlog
from tavily import TavilyClient

from config import Settings, get_settings
from game_resolution import GameResolver
from quality_policy import (
    RELEVANCE_SCORE_POLICY,
    EXTERNAL_SEARCH_ATTEMPTS,
    MAX_QUERIES_PER_PLANNED_QUERY,
    MAX_SEARCH_QUERIES,
    PROGRESSIVE_STRICT_SOURCE_TARGET,
    SEARCH_NOISE_TOKENS,
    SEARCH_RESULT_WEIGHTS,
    SOURCE_POLICIES,
    SourcePolicy,
    STABLE_FACT_INTENTS,
    VERSION_SCORE_POLICY,
    VERSION_SENSITIVE_INTENTS,
    VERSION_SIGNAL_TOKENS,
    domain_quality,
    intent_source_preference,
    source_domain_limit,
)
from query_tokens import question_relevance_tokens, relevance_tokens
from schemas import GameResolution, PlannedSearchQuery, SearchPlan, Source


logger = structlog.get_logger()


class TTLSearchCache:
    """Small thread-safe LRU cache for paid search responses."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._values: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        if self.ttl_seconds <= 0:
            return None
        now = monotonic()
        with self._lock:
            cached = self._values.get(key)
            if cached is None:
                return None
            expires_at, value = cached
            if expires_at <= now:
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._values[key] = (monotonic() + self.ttl_seconds, deepcopy(value))
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)


class RedisSearchCache:
    """Persistent JSON cache with a local fallback when Redis is unavailable."""

    def __init__(self, *, redis_url: str, fallback: TTLSearchCache) -> None:
        self._redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        self._fallback = fallback

    @staticmethod
    def _redis_key(key: str) -> str:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return f"questmate:search:v1:{digest}"

    def get(self, key: str) -> Any | None:
        local = self._fallback.get(key)
        if local is not None:
            return local
        try:
            payload = self._redis.get(self._redis_key(key))
            if payload is None:
                return None
            value = json.loads(payload)
        except Exception:
            return None
        self._fallback.set(key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        self._fallback.set(key, value)
        if self._fallback.ttl_seconds <= 0:
            return
        try:
            self._redis.setex(
                self._redis_key(key),
                self._fallback.ttl_seconds,
                json.dumps(value, ensure_ascii=False, default=str),
            )
        except Exception:
            return


class CachedSearchClient:
    """Cache identical Tavily calls and expose credit-relevant counters."""

    def __init__(self, client: Any, cache: Any) -> None:
        self._client = client
        self._cache = cache
        self.upstream_calls = 0
        self.cache_hits = 0

    def search(self, **kwargs: Any) -> dict[str, Any]:
        key = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        result = self._client.search(**kwargs)
        self.upstream_calls += 1
        self._cache.set(key, result)
        return result


class MediaWikiClient:
    """Query game-specific MediaWiki installations without paid search."""

    user_agent = "QuestMate/0.1 (local game guide search)"

    def search(self, *, domain: str, query: str, max_results: int) -> dict[str, Any]:
        search_query = " ".join(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", query)) or query
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": search_query,
                "gsrlimit": max_results,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "format": "json",
                "formatversion": 2,
                "origin": "*",
            }
        )
        request = Request(
            f"https://{domain}/api.php?{params}",
            headers={"User-Agent": self.user_agent},
        )
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
        pages = sorted(
            payload.get("query", {}).get("pages", []),
            key=lambda page: int(page.get("index") or 9999),
        )
        results = []
        for page in pages:
            title = str(page.get("title") or "").strip()
            if not title:
                continue
            revisions = page.get("revisions") or []
            content = ""
            if revisions:
                content = str(revisions[0].get("slots", {}).get("main", {}).get("content") or "")
            results.append(
                {
                    "title": title,
                    "url": f"https://{domain}/wiki/{quote(title.replace(' ', '_'))}",
                    "content": self._clean_wikitext(content)[:6000],
                    "score": 0.9,
                }
            )
        return {"results": results}

    @staticmethod
    def _clean_wikitext(content: str) -> str:
        cleaned = re.sub(r"<!--.*?-->|<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", " ", content, flags=re.S | re.I)
        cleaned = re.sub(r"\[\[(?:File|Image):[^\]]+\]\]", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", cleaned)
        cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
        for _ in range(3):
            cleaned = re.sub(r"\{\{[^{}]*\}\}", " ", cleaned)
        cleaned = re.sub(r"'{2,}|={2,}|\[https?://\S+\s*([^\]]*)\]", r" \1 ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()


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
        self._mediawiki_client = mediawiki_client
        if self._mediawiki_client is None and client is None and self.settings.mediawiki_direct_search:
            self._mediawiki_client = MediaWikiClient()
        self._mediawiki_cache = shared_cache

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
        game_aliases: list[str],
        database_domains: list[str],
        intent: str,
        max_results: int,
    ) -> list[Source]:
        if self._mediawiki_client is None or not database_domains:
            return []
        wiki_domains = [
            domain
            for domain in database_domains
            if self._game_resolver is not None and self._game_resolver.is_supported_database_domain(domain)
        ][:2]
        if not wiki_domains:
            return []

        wiki_query = " ".join(aliases[:2]).strip() or question

        async def fetch(domain: str) -> dict[str, Any]:
            cache_key = f"mediawiki:{domain}:{wiki_query.casefold()}:{max_results}"
            cached = self._mediawiki_cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._mediawiki_client.search,
                        domain=domain,
                        query=wiki_query,
                        max_results=max_results,
                    ),
                    timeout=self.settings.external_request_timeout_seconds,
                )
            except Exception:
                return {"results": []}
            self._mediawiki_cache.set(cache_key, result)
            return result

        results = await asyncio.gather(*(fetch(domain) for domain in wiki_domains))
        sources_by_url: dict[str, Source] = {}
        search_context = f"{question} {' '.join(aliases)}"
        wiki_policy = self.sources["wiki"]
        for result in results:
            for item in result.get("results", []):
                evidence = self._best_evidence_passage(
                    str(item.get("content") or ""),
                    question=search_context,
                    max_chars=self.settings.evidence_passage_max_chars,
                )
                searchable_item = {**item, "content": evidence}
                relevance_score = self._result_relevance_score(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                )
                if relevance_score <= 0 or not self._is_high_quality_source(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                    source_type="wiki",
                ):
                    continue
                url = str(item.get("url") or "")
                source = Source(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("content") or "")[:600],
                    score=min(1, 0.7 + relevance_score * 0.3),
                    source_type="wiki",
                    trust_score=wiki_policy.trust_score,
                    trust_label=wiki_policy.trust_label,
                    evidence=evidence,
                    fetched_at=datetime.now(timezone.utc),
                    game_version=self._extract_game_version(evidence),
                )
                sources_by_url[self._canonical_source_key(url)] = source
        return sorted(
            sources_by_url.values(),
            key=lambda source: source.score or 0,
            reverse=True,
        )

    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        if self._client is None:
            return GameResolution(input_name=game, confirmed_name=game, confidence=0)
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
                url = item.get("url")
                if not url:
                    continue
                if search_source.source_type == "wiki":
                    result_domain = urlparse(str(url)).netloc.lower()
                    known_database = any(
                        result_domain == domain or result_domain.endswith(f".{domain}")
                        for domain in database_domains
                    )
                    title_url = f"{item.get('title') or ''} {url}".lower()
                    if not known_database and not self._matches_game_name(
                        text=title_url,
                        game_names=[game, *game_aliases],
                    ):
                        continue
                search_context = f"{query} {search_query} {' '.join(aliases)}"
                raw_content = str(item.get("raw_content") or "")
                evidence = self._best_evidence_passage(
                    raw_content or str(item.get("content") or ""),
                    question=search_context,
                    max_chars=self.settings.evidence_passage_max_chars,
                )
                searchable_item = {
                    **item,
                    "content": f"{item.get('content') or ''} {evidence}",
                }
                relevance_score = self._result_relevance_score(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                )
                if relevance_score <= 0:
                    continue

                raw_score = float(item.get("score") or 0)
                intent_score = self._intent_source_boost(intent=intent, source_type=search_source.source_type)
                domain_score = self._domain_quality_score(str(url))
                version_score = self._version_safety_score(
                    intent=intent,
                    source_type=search_source.source_type,
                    text=f"{item.get('title') or ''} {item.get('url') or ''} {evidence}",
                )
                weighted_score = (
                    raw_score * SEARCH_RESULT_WEIGHTS.retrieval
                    + search_source.trust_score * SEARCH_RESULT_WEIGHTS.trust
                    + relevance_score * SEARCH_RESULT_WEIGHTS.relevance
                    + intent_score * SEARCH_RESULT_WEIGHTS.intent
                    + domain_score * SEARCH_RESULT_WEIGHTS.domain
                    + version_score * SEARCH_RESULT_WEIGHTS.version
                )
                source = Source(
                    title=item.get("title") or url,
                    url=url,
                    snippet=item.get("content"),
                    score=weighted_score,
                    source_type=search_source.source_type,
                    trust_score=search_source.trust_score,
                    trust_label=search_source.trust_label,
                    evidence=evidence,
                    published_at=self._parse_source_datetime(item.get("published_date") or item.get("published_at")),
                    fetched_at=datetime.now(timezone.utc),
                    game_version=self._extract_game_version(
                        f"{item.get('title') or ''} {item.get('content') or ''} {evidence}"
                    ),
                )
                source_key = self._canonical_source_key(str(url))
                current = relaxed_sources_by_url.get(source_key)
                if current is None or (source.score or 0) > (current.score or 0):
                    relaxed_sources_by_url[source_key] = source
                if self._is_high_quality_source(
                    item=searchable_item,
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
        planned_queries = list((plan or self.fallback_plan).queries)[:4] or list(self.fallback_plan.queries)
        aliases = list((plan.aliases if plan else [])[:3])
        built: list[tuple[str, SourcePolicy]] = []
        seen: set[str] = set()

        for planned in planned_queries:
            source = self.sources.get(planned.source_type, self.sources["web"])
            query = planned.query.replace("{question}", question).strip()

            candidates: list[str] = []
            if source.source_type == "wiki":
                candidates.extend(f"site:{domain} {game} {query}" for domain in database_domains)
                for alias in aliases:
                    if alias.lower() not in query.lower():
                        candidates.extend(
                            f"site:{domain} {game} {alias} {query}"
                            for domain in database_domains
                        )
                for game_alias in game_aliases[:3]:
                    if game_alias.lower() != game.lower():
                        candidates.extend(f"site:{domain} {game_alias} {query}" for domain in database_domains)
            for domain_index, domain in enumerate(source.domains):
                candidates.append(f"site:{domain} {game} {query}")
                if domain_index == 0:
                    for alias in aliases:
                        if alias.lower() not in query.lower():
                            candidates.append(f"site:{domain} {game} {alias} {query}")
                for game_alias in game_aliases[:3]:
                    if game_alias.lower() != game.lower():
                        candidates.append(f"site:{domain} {game_alias} {query}")
            for game_alias in game_aliases[:3]:
                if game_alias.lower() != game.lower():
                    candidates.extend(template.format(game=game_alias, query=query) for template in source.query_templates)
            candidates.extend(template.format(game=game, query=query) for template in source.query_templates)
            if not candidates:
                candidates.append(f"{game} {query}")
            for alias in aliases:
                if alias.lower() not in query.lower():
                    candidates.append(f"{game} {alias} {query}")

            added_for_plan = 0
            for candidate in candidates:
                normalized = " ".join(candidate.split())
                if normalized in seen:
                    continue
                built.append((normalized, source))
                seen.add(normalized)
                added_for_plan += 1
                if len(built) >= MAX_SEARCH_QUERIES:
                    return built
                if added_for_plan >= MAX_QUERIES_PER_PLANNED_QUERY:
                    break

        return built

    @staticmethod
    def _is_relevant_result(*, item: dict[str, Any], game: str, question: str) -> bool:
        return TavilySearchProvider._result_relevance_score(item=item, game=game, question=question) > 0

    @staticmethod
    def _result_relevance_score(
        *,
        item: dict[str, Any],
        game: str,
        question: str,
        game_aliases: list[str] | None = None,
    ) -> float:
        text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "url", "content")
        ).lower()
        if not TavilySearchProvider._is_same_game_surface(text=text, game=game, question=question):
            return 0
        if TavilySearchProvider._is_low_value_page(text=text, question=question):
            return 0

        game_names = [game, *(game_aliases or [])]
        game_token_set = set(relevance_tokens(" ".join(game_names)))
        question_tokens = [
            token
            for token in question_relevance_tokens(question)
            if token not in game_token_set and token not in TavilySearchProvider.search_noise_tokens
        ]

        has_game_match = TavilySearchProvider._matches_game_name(
            text=text,
            game_names=game_names,
        )
        if not has_game_match:
            return 0

        if not question_tokens:
            return RELEVANCE_SCORE_POLICY.no_entity_score

        matched = sum(1 for token in question_tokens if token in text)
        latin_entity_tokens = [token for token in question_tokens if re.fullmatch(r"[a-z0-9]+", token)]
        minimum_matches = 2 if len(latin_entity_tokens) >= 2 else 1
        if matched < minimum_matches:
            return 0

        title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).lower()
        focused_matches = sum(1 for token in question_tokens if token in title_url)
        coverage = matched / max(len(question_tokens), 1)
        focus_bonus = min(
            focused_matches * RELEVANCE_SCORE_POLICY.title_match_bonus,
            RELEVANCE_SCORE_POLICY.title_bonus_cap,
        )
        return min(
            1.0,
            RELEVANCE_SCORE_POLICY.base_score
            + coverage * RELEVANCE_SCORE_POLICY.coverage_weight
            + focus_bonus,
        )

    @staticmethod
    def _matches_game_name(*, text: str, game_names: list[str]) -> bool:
        for game_name in game_names:
            normalized = game_name.lower().strip()
            if normalized and normalized in text:
                return True
            tokens = [token for token in relevance_tokens(normalized) if token != normalized]
            if tokens and all(token in text for token in tokens):
                return True
        return not any(game_name.strip() for game_name in game_names)

    @staticmethod
    def _is_same_game_surface(*, text: str, game: str, question: str) -> bool:
        normalized_game = game.lower()
        normalized_question = question.lower()
        if "elden ring" in normalized_game and "nightreign" in text and "nightreign" not in normalized_question:
            return False
        return True

    @staticmethod
    def _is_low_value_page(*, text: str, question: str) -> bool:
        lowered_question = question.lower()
        if "villains.fandom.com" in text and not any(token in lowered_question for token in ("lore", "剧情", "背景")):
            return True
        if any(value in text for value in ("all-fiction-battles", "vs battles wiki", "battle wiki")) and not any(
            token in lowered_question for token in ("lore", "剧情", "背景")
        ):
            return True
        if "reddit - the heart of the internet" in text:
            return True
        if "reddit.com/r/eldenring/comments" not in text and any(
            value in text for value in ("reddit.com/r/eldenring", "reddit - the heart of the internet")
        ):
            return True
        if "steamcommunity.com/app" in text and "/discussions/" not in text:
            return True
        return False

    @staticmethod
    def _is_high_quality_source(
        *,
        item: dict[str, Any],
        game: str,
        question: str,
        source_type: str,
        game_aliases: list[str] | None = None,
    ) -> bool:
        if source_type == "official":
            return True

        title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).lower()
        game_token_set = set(relevance_tokens(" ".join([game, *(game_aliases or [])])))
        entity_tokens = [
            token
            for token in question_relevance_tokens(question)
            if token not in game_token_set and token not in TavilySearchProvider.search_noise_tokens
        ]
        title_entity_matches = sum(1 for token in entity_tokens if token in title_url)

        if source_type == "wiki":
            return title_entity_matches > 0
        if source_type == "community":
            return title_entity_matches > 0 and any(value in title_url for value in ("comments", "discussions"))
        return title_entity_matches > 0

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
    def _intent_source_boost(*, intent: str, source_type: str) -> float:
        return intent_source_preference(intent, source_type)

    @staticmethod
    def _domain_quality_score(url: str) -> float:
        return domain_quality(urlparse(url).netloc)

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
