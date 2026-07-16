"""Direct MediaWiki retrieval, link expansion, and knowledge indexing."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import re
import time
from typing import Any, Protocol

import structlog

from config import Settings
from mediawiki_client import MediaWikiClient
from quality_policy import SEARCH_NOISE_TOKENS, SourcePolicy
from query_tokens import question_relevance_tokens, relevance_tokens
from retrieval.evidence_pool import merge_source_evidence
from retrieval.relevance import is_high_quality_source, result_relevance_score
from retrieval.source_quality import page_authority_score, required_entity_groups_for_query
from retrieval.wiki_domains import (
    is_safe_wiki_host,
    normalize_wiki_host,
    resolves_to_public_addresses,
)
from schemas import Source

logger = structlog.get_logger()
MEDIAWIKI_FAILURE_COOLDOWN_SECONDS = 300
MAX_BACKGROUND_INDEX_BATCHES = 8
MAX_CONCURRENT_INDEX_WRITES = 4
BACKGROUND_INDEX_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class ContentIndex(Protocol):
    async def index_content(self, **kwargs: Any) -> dict[str, Any]: ...


class MediaWikiRetriever:
    def __init__(
        self,
        *,
        client: Any | None,
        cache: Any,
        settings: Settings,
        source_policy: SourcePolicy,
        content_index: ContentIndex | None,
        best_passage: Callable[..., str],
        canonical_key: Callable[[str], str],
        extract_version: Callable[[str], str | None],
    ) -> None:
        self.client = client
        self.cache = cache
        self.settings = settings
        self.source_policy = source_policy
        self.content_index = content_index
        self.best_passage = best_passage
        self.canonical_key = canonical_key
        self.extract_version = extract_version
        self._domain_retry_after: dict[str, float] = {}
        self._background_index_tasks: set[asyncio.Task[None]] = set()
        self._background_index_urls: set[str] = set()
        self._background_index_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INDEX_WRITES)
        # The built-in adapter's network behavior is known, so it also gets a
        # public-DNS check immediately before urllib performs the request.
        # Custom injected adapters remain responsible for their own transport.
        self._require_public_dns = isinstance(client, MediaWikiClient)

    async def search(
        self,
        *,
        game: str,
        question: str,
        aliases: list[str],
        planned_queries: list[str],
        game_aliases: list[str],
        database_domains: list[str],
        max_results: int,
        named_entity_groups: list[list[str]] | None = None,
    ) -> list[Source]:
        if self.client is None or not database_domains:
            return []
        domains = self._rank_database_domains(database_domains, game=game)[:2]
        queries = self._select_search_queries(
            question=question,
            aliases=aliases,
            planned_queries=planned_queries,
        )
        results = await asyncio.gather(
            *(self._fetch_search(domain, query, max_results) for domain in domains for query in queries)
        )
        # Every result keeps the query that produced it. During a refinement or
        # compound plan, one dependency page must not be required to mention
        # entities from a different parallel query.
        fallback_context = f"{question} {' '.join(aliases)}".strip()
        results.extend(
            await self._expand_results(
                results=results,
                search_context=fallback_context,
                game=game,
                game_aliases=game_aliases,
            )
        )

        sources_by_url: dict[str, Source] = {}
        content_by_url: dict[str, str] = {}
        for result in results:
            context = str(result.get("_query") or "").strip() or fallback_context
            required_entity_groups = required_entity_groups_for_query(
                named_entity_groups or [],
                context,
            )
            for item in result.get("results", []):
                content = str(item.get("content") or "")
                evidence = self.best_passage(
                    content,
                    question=context,
                    max_chars=self.settings.evidence_passage_max_chars,
                )
                searchable_item = {**item, "content": evidence}
                relevance = result_relevance_score(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=context,
                    required_entity_groups=required_entity_groups,
                )
                if relevance <= 0 or not is_high_quality_source(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=context,
                    source_type="wiki",
                    required_entity_groups=required_entity_groups,
                ):
                    continue
                url = str(item.get("url") or "")
                key = self.canonical_key(url)
                candidate = Source(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=content[:600],
                    score=min(1, 0.7 + relevance * 0.3),
                    source_type="wiki",
                    trust_score=page_authority_score(
                        item=searchable_item,
                        source_prior=self.source_policy.trust_score,
                    ),
                    trust_label=self.source_policy.trust_label,
                    evidence=evidence,
                    fetched_at=datetime.now(timezone.utc),
                    game_version=self.extract_version(evidence),
                )
                current = sources_by_url.get(key)
                if current is None:
                    sources_by_url[key] = candidate
                elif (candidate.score or 0) >= (current.score or 0):
                    sources_by_url[key] = merge_source_evidence(
                        preferred=candidate,
                        other=current,
                    )
                else:
                    sources_by_url[key] = merge_source_evidence(
                        preferred=current,
                        other=candidate,
                    )
                if len(content) > len(content_by_url.get(key, "")):
                    content_by_url[key] = content
        ranked = sorted(sources_by_url.values(), key=lambda source: source.score or 0, reverse=True)
        self._schedule_auto_index(game=game, sources=ranked, content_by_url=content_by_url)
        return ranked

    def _schedule_auto_index(
        self,
        *,
        game: str,
        sources: list[Source],
        content_by_url: dict[str, str],
    ) -> None:
        if self.content_index is None or not self.settings.wiki_auto_index_enabled or not sources:
            return
        if len(self._background_index_tasks) >= MAX_BACKGROUND_INDEX_BATCHES:
            logger.warning(
                "knowledge.wiki_auto_index_backpressure",
                pending_batches=len(self._background_index_tasks),
            )
            return
        selected: list[Source] = []
        selected_keys: set[str] = set()
        for source in sources:
            key = self.canonical_key(str(source.url))
            if not key or key in self._background_index_urls or key in selected_keys:
                continue
            selected.append(source)
            selected_keys.add(key)
            if len(selected) >= self.settings.wiki_auto_index_pages_per_query:
                break
        if not selected:
            return
        self._background_index_urls.update(selected_keys)
        task = asyncio.create_task(
            self._auto_index(game=game, sources=selected, content_by_url=content_by_url)
        )
        self._background_index_tasks.add(task)

        def complete(done: asyncio.Task[None]) -> None:
            self._background_index_tasks.discard(done)
            self._background_index_urls.difference_update(selected_keys)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.warning("knowledge.wiki_auto_index_failed", error=str(error))

        task.add_done_callback(complete)

    async def wait_for_background_tasks(
        self,
        timeout_seconds: float = BACKGROUND_INDEX_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        """Drain pending writes up to a deadline, then cancel stragglers."""
        deadline = time.monotonic() + max(0, timeout_seconds)
        while self._background_index_tasks:
            pending = tuple(self._background_index_tasks)
            remaining = max(0, deadline - time.monotonic())
            _done, still_pending = await asyncio.wait(pending, timeout=remaining)
            if not still_pending:
                continue
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
            self._background_index_tasks.difference_update(still_pending)
            logger.warning(
                "knowledge.wiki_auto_index_shutdown_timeout",
                cancelled_batches=len(still_pending),
            )
            break

    @staticmethod
    def _select_search_query(*, question: str, aliases: list[str], planned_queries: list[str]) -> str:
        return MediaWikiRetriever._select_search_queries(
            question=question,
            aliases=aliases,
            planned_queries=planned_queries,
        )[0]

    @staticmethod
    def _rank_database_domains(domains: list[str], *, game: str) -> list[str]:
        """Prefer a game-specific host over broad encyclopedias without knowing providers in advance."""
        game_key = re.sub(r"[^a-z0-9]", "", game.casefold())
        unique = list(dict.fromkeys(
            host
            for domain in domains
            if (host := normalize_wiki_host(domain)) is not None and is_safe_wiki_host(host)
        ))

        def specificity(entry: tuple[int, str]) -> tuple[int, int]:
            index, domain = entry
            domain_key = re.sub(r"[^a-z0-9]", "", domain.casefold().split(":", 1)[0])
            return (1 if len(game_key) >= 4 and game_key in domain_key else 0, -index)

        return [domain for _index, domain in sorted(enumerate(unique), key=specificity, reverse=True)]

    @staticmethod
    def _select_search_queries(*, question: str, aliases: list[str], planned_queries: list[str]) -> list[str]:
        raw_planned = list(dict.fromkeys(
            " ".join(query.split())
            for query in planned_queries
            if query.strip()
        ))[:2]
        if not raw_planned:
            return [" ".join(aliases[:2]).strip() or question]

        normalized = [
            MediaWikiRetriever._normalize_mixed_language_query(query)
            for query in raw_planned
        ]
        # Keep different planned semantics in the two-query budget.  When only
        # one mixed-language target exists, retain its original form as an
        # independent fallback so normalization can never erase an entity.
        primary = list(dict.fromkeys(normalized))[:2]
        if len(raw_planned) == 1 and normalized[0].casefold() != raw_planned[0].casefold():
            primary.append(raw_planned[0])
        return list(dict.fromkeys(primary))[:2]

    @staticmethod
    def _normalize_mixed_language_query(query: str) -> str:
        """Prefer a Latin entity only when doing so cannot erase a CJK focus."""
        compact = " ".join(query.split())
        if not re.search(r"[\u3400-\u9fff]", compact) or not re.search(r"[a-z]", compact, re.IGNORECASE):
            return compact
        latin_parts = re.findall(r"[a-z][a-z'-]*|[a-z]*\d[a-z0-9._-]*|\d{1,6}", compact, re.IGNORECASE)
        normalized = " ".join(latin_parts).strip()
        lexical_parts = [part for part in latin_parts if any(character.isalpha() for character in part)]
        cjk_focus = [
            token
            for token in question_relevance_tokens(compact)
            if re.fullmatch(r"[\u3400-\u9fff]{4,}", token)
        ]
        # A lone acronym/version is usually a modifier, not proof that the CJK
        # portion is expendable (for example "月光钥匙 DLC" or "... v2.0").
        # Likewise, a compact CJK focus surviving generic question cleanup is
        # an entity candidate and must not be removed merely because two Latin
        # modifiers are present.
        return (
            normalized
            if len(normalized) >= 3 and len(lexical_parts) >= 2 and not cjk_focus
            else compact
        )

    async def _fetch_search(self, domain: str, query: str, max_results: int) -> dict[str, Any]:
        host = normalize_wiki_host(domain)
        if host is None or not is_safe_wiki_host(host):
            logger.warning("mediawiki.unsafe_domain_rejected", domain=domain)
            return {"results": [], "_domain": "", "_query": query}
        cache_key = f"mediawiki:v2:{host}:{query.casefold()}:{max_results}"
        cached = await asyncio.to_thread(self.cache.get, cache_key)
        if cached is not None:
            return {**cached, "_domain": host, "_query": query}
        if self._domain_retry_after.get(host, 0) > time.monotonic():
            return {"results": [], "_domain": host, "_query": query}
        domain = await self._safe_request_domain(host)
        if not domain:
            return {"results": [], "_domain": "", "_query": query}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.client.search, domain=domain, query=query, max_results=max_results),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            self._domain_retry_after[domain] = time.monotonic() + MEDIAWIKI_FAILURE_COOLDOWN_SECONDS
            return {"results": [], "_domain": domain, "_query": query}
        self._domain_retry_after.pop(domain, None)
        result["_domain"] = domain
        result["_query"] = query
        await asyncio.to_thread(self.cache.set, cache_key, result)
        return result

    async def _expand_results(
        self,
        *,
        results: list[dict[str, Any]],
        search_context: str,
        game: str,
        game_aliases: list[str],
    ) -> list[dict[str, Any]]:
        limit = self.settings.wiki_link_expansion_pages_per_query
        if limit <= 0 or not hasattr(self.client, "fetch_pages"):
            return []
        game_tokens = set(relevance_tokens(" ".join([game, *game_aliases])))
        candidates_by_key: dict[tuple[str, str], tuple[int, str, str, str]] = {}
        for result in results:
            result_context = str(result.get("_query") or "").strip() or search_context
            query_tokens = [
                token
                for token in question_relevance_tokens(result_context)
                if token not in game_tokens and token not in SEARCH_NOISE_TOKENS
            ]
            domain = str(result.get("_domain") or "")
            for item_index, item in enumerate(result.get("results", [])):
                focused_evidence = self.best_passage(
                    str(item.get("content") or ""),
                    question=result_context,
                    max_chars=self.settings.evidence_passage_max_chars,
                ).casefold()
                for title in item.get("links") or []:
                    normalized = str(title).strip()
                    key = (domain, normalized.casefold())
                    if (
                        not domain
                        or not normalized
                        or normalized.casefold() == str(item.get("title") or "").strip().casefold()
                    ):
                        continue
                    score = self._linked_entity_relevance_score(
                        normalized,
                        focused_evidence=focused_evidence,
                        query_tokens=query_tokens,
                    )
                    if score:
                        score += max(0, 3 - item_index)
                    if score:
                        candidate = (score, domain, normalized, result_context)
                        current = candidates_by_key.get(key)
                        if current is None or score > current[0]:
                            candidates_by_key[key] = candidate
        selected = sorted(candidates_by_key.values(), key=lambda item: item[0], reverse=True)[:limit]
        titles_by_context: dict[tuple[str, str], list[str]] = {}
        for _score, domain, title, result_context in selected:
            titles_by_context.setdefault((domain, result_context), []).append(title)
        expanded = await asyncio.gather(
            *(
                self._fetch_pages(domain, titles, query_context=result_context)
                for (domain, result_context), titles in titles_by_context.items()
            )
        )
        logger.info(
            "mediawiki.expanded",
            selected_links=len(selected),
            selected_titles=[title for _score, _domain, title, _context in selected],
            fetched_pages=sum(len(result.get("results", [])) for result in expanded),
        )
        return list(expanded)

    @staticmethod
    def _linked_entity_relevance_score(
        title: str,
        *,
        focused_evidence: str,
        query_tokens: list[str],
    ) -> int:
        """Rank links by entity overlap and their local relevance to the query."""
        normalized = title.casefold()
        title_tokens = set(relevance_tokens(normalized))
        query_token_set = set(query_tokens)
        direct_overlap = len(title_tokens & query_token_set)
        position = focused_evidence.find(normalized)
        if position < 0:
            return direct_overlap * 4
        prior_boundaries = [focused_evidence.rfind(mark, 0, position) for mark in ".!?。！？;；"]
        left = max(prior_boundaries, default=-1) + 1
        following = [
            boundary
            for mark in ".!?。！？;；"
            if (boundary := focused_evidence.find(mark, position + len(normalized))) >= 0
        ]
        right = min(following, default=min(len(focused_evidence), position + len(normalized) + 180))
        context_tokens = set(question_relevance_tokens(focused_evidence[left:right]))
        contextual_overlap = len(context_tokens & query_token_set)
        return direct_overlap * 4 + contextual_overlap * 2 + 1

    async def _safe_request_domain(self, domain: str) -> str:
        host = normalize_wiki_host(domain)
        if host is None or not is_safe_wiki_host(host):
            logger.warning("mediawiki.unsafe_domain_rejected", domain=domain)
            return ""
        if not self._require_public_dns:
            return host
        try:
            is_public = await asyncio.wait_for(
                asyncio.to_thread(resolves_to_public_addresses, host),
                timeout=min(3.0, float(self.settings.external_request_timeout_seconds)),
            )
        except (TimeoutError, OSError):
            is_public = False
        if not is_public:
            logger.warning("mediawiki.non_public_domain_rejected", domain=host)
            return ""
        return host

    async def _fetch_pages(
        self,
        domain: str,
        titles: list[str],
        *,
        query_context: str = "",
    ) -> dict[str, Any]:
        host = normalize_wiki_host(domain)
        if host is None or not is_safe_wiki_host(host):
            logger.warning("mediawiki.unsafe_domain_rejected", domain=domain)
            return {"results": [], "_domain": "", "_query": query_context}
        cache_key = f"mediawiki-pages:v2:{host}:{'|'.join(title.casefold() for title in titles)}"
        cached = await asyncio.to_thread(self.cache.get, cache_key)
        if cached is not None:
            return {**cached, "_domain": host, "_query": query_context}
        domain = await self._safe_request_domain(host)
        if not domain:
            return {"results": [], "_domain": "", "_query": query_context}
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(self.client.fetch_pages, domain=domain, titles=titles),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            return {"results": [], "_domain": domain, "_query": query_context}
        payload["_domain"] = domain
        payload["_query"] = query_context
        await asyncio.to_thread(self.cache.set, cache_key, payload)
        return payload

    async def _auto_index(self, *, game: str, sources: list[Source], content_by_url: dict[str, str]) -> None:
        if self.content_index is None or not self.settings.wiki_auto_index_enabled:
            return
        results = await asyncio.gather(
            *(
                self._index_source(game=game, source=source, content_by_url=content_by_url)
                for source in sources
            ),
            return_exceptions=True,
        )
        logger.info(
            "knowledge.wiki_auto_index",
            game=game,
            attempted=len(sources),
            ready=sum(1 for result in results if isinstance(result, dict) and result.get("status") in {"ready", "cached"}),
            failed=sum(1 for result in results if isinstance(result, BaseException)),
        )

    async def _index_source(
        self,
        *,
        game: str,
        source: Source,
        content_by_url: dict[str, str],
    ) -> dict[str, Any]:
        assert self.content_index is not None
        async with self._background_index_semaphore:
            return await self.content_index.index_content(
                url=str(source.url),
                game=game,
                content=content_by_url.get(self.canonical_key(str(source.url)), ""),
                title=source.title,
                source_type="wiki",
                game_version=source.game_version,
                published_at=source.published_at,
                skip_if_fresh=True,
            )
