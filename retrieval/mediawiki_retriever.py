"""Direct MediaWiki retrieval, link expansion, and knowledge indexing."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import re
import time
from typing import Any, Protocol

import structlog

from config import Settings
from quality_policy import SEARCH_NOISE_TOKENS, SourcePolicy
from query_tokens import question_relevance_tokens, relevance_tokens
from retrieval.relevance import is_high_quality_source, result_relevance_score
from schemas import Source

logger = structlog.get_logger()
MEDIAWIKI_FAILURE_COOLDOWN_SECONDS = 300


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
        context = f"{question} {' '.join(aliases)} {' '.join(planned_queries[:2])}".strip()
        results.extend(
            await self._expand_results(results=results, search_context=context, game=game, game_aliases=game_aliases)
        )

        sources_by_url: dict[str, Source] = {}
        content_by_url: dict[str, str] = {}
        for result in results:
            for item in result.get("results", []):
                content = str(item.get("content") or "")
                evidence = self.best_passage(
                    content,
                    question=context,
                    max_chars=self.settings.evidence_passage_max_chars,
                )
                searchable_item = {**item, "content": evidence}
                relevance = result_relevance_score(
                    item=searchable_item, game=game, game_aliases=game_aliases, question=context
                )
                if relevance <= 0 or not is_high_quality_source(
                    item=searchable_item,
                    game=game,
                    game_aliases=game_aliases,
                    question=context,
                    source_type="wiki",
                ):
                    continue
                url = str(item.get("url") or "")
                key = self.canonical_key(url)
                sources_by_url[key] = Source(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=content[:600],
                    score=min(1, 0.7 + relevance * 0.3),
                    source_type="wiki",
                    trust_score=self.source_policy.trust_score,
                    trust_label=self.source_policy.trust_label,
                    evidence=evidence,
                    fetched_at=datetime.now(timezone.utc),
                    game_version=self.extract_version(evidence),
                )
                content_by_url[key] = content
        ranked = sorted(sources_by_url.values(), key=lambda source: source.score or 0, reverse=True)
        await self._auto_index(game=game, sources=ranked, content_by_url=content_by_url)
        return ranked

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
        unique = list(dict.fromkeys(domain for domain in domains if domain))

        def specificity(entry: tuple[int, str]) -> tuple[int, int]:
            index, domain = entry
            domain_key = re.sub(r"[^a-z0-9]", "", domain.casefold().split(":", 1)[0])
            return (1 if len(game_key) >= 4 and game_key in domain_key else 0, -index)

        return [domain for _index, domain in sorted(enumerate(unique), key=specificity, reverse=True)]

    @staticmethod
    def _select_search_queries(*, question: str, aliases: list[str], planned_queries: list[str]) -> list[str]:
        planned = list(dict.fromkeys(
            MediaWikiRetriever._normalize_mixed_language_query(query)
            for query in planned_queries
            if query.strip()
        ))[:2]
        return planned or [" ".join(aliases[:2]).strip() or question]

    @staticmethod
    def _normalize_mixed_language_query(query: str) -> str:
        """Use the Latin entity portion when a query mixes CJK instructions with English wiki names."""
        compact = " ".join(query.split())
        if not re.search(r"[\u3400-\u9fff]", compact) or not re.search(r"[a-z]", compact, re.IGNORECASE):
            return compact
        latin_parts = re.findall(r"[a-z][a-z'-]*|[a-z]*\d[a-z0-9._-]*|\d{1,6}", compact, re.IGNORECASE)
        normalized = " ".join(latin_parts).strip()
        return normalized if len(normalized) >= 3 else compact

    async def _fetch_search(self, domain: str, query: str, max_results: int) -> dict[str, Any]:
        if self._domain_retry_after.get(domain, 0) > time.monotonic():
            return {"results": [], "_domain": domain}
        cache_key = f"mediawiki:v2:{domain}:{query.casefold()}:{max_results}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {**cached, "_domain": domain}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.client.search, domain=domain, query=query, max_results=max_results),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            self._domain_retry_after[domain] = time.monotonic() + MEDIAWIKI_FAILURE_COOLDOWN_SECONDS
            return {"results": [], "_domain": domain}
        self._domain_retry_after.pop(domain, None)
        result["_domain"] = domain
        self.cache.set(cache_key, result)
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
        query_tokens = [
            token
            for token in question_relevance_tokens(search_context)
            if token not in game_tokens and token not in SEARCH_NOISE_TOKENS
        ]
        candidates: list[tuple[int, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for result in results:
            domain = str(result.get("_domain") or "")
            for item_index, item in enumerate(result.get("results", [])):
                focused_evidence = self.best_passage(
                    str(item.get("content") or ""),
                    question=search_context,
                    max_chars=self.settings.evidence_passage_max_chars,
                ).casefold()
                for title in item.get("links") or []:
                    normalized = str(title).strip()
                    key = (domain, normalized.casefold())
                    if (
                        not domain
                        or not normalized
                        or normalized.casefold() == str(item.get("title") or "").strip().casefold()
                        or key in seen
                    ):
                        continue
                    seen.add(key)
                    score = sum(1 for token in query_tokens if token in normalized.casefold())
                    dependency_score = self._explicit_dependency_score(normalized, focused_evidence)
                    if dependency_score:
                        score += dependency_score + max(0, 3 - item_index)
                    if score:
                        candidates.append((score, domain, normalized))
        candidates.sort(reverse=True)
        selected = candidates[:limit]
        titles_by_domain: dict[str, list[str]] = {}
        for _score, domain, title in selected:
            titles_by_domain.setdefault(domain, []).append(title)
        expanded = await asyncio.gather(
            *(self._fetch_pages(domain, titles) for domain, titles in titles_by_domain.items())
        )
        logger.info(
            "mediawiki.expanded",
            selected_links=len(selected),
            selected_titles=[title for _score, _domain, title in selected],
            fetched_pages=sum(len(result.get("results", [])) for result in expanded),
        )
        return list(expanded)

    @staticmethod
    def _explicit_dependency_score(title: str, focused_evidence: str) -> int:
        """Prioritize linked entities named in a sentence that expresses a dependency."""
        normalized = title.casefold()
        position = focused_evidence.find(normalized)
        if position < 0:
            return 0
        prior_boundaries = [focused_evidence.rfind(mark, 0, position) for mark in ".!?。！？;；"]
        left = max(prior_boundaries, default=-1) + 1
        following = [
            boundary
            for mark in ".!?。！？;；"
            if (boundary := focused_evidence.find(mark, position + len(normalized))) >= 0
        ]
        right = min(following, default=min(len(focused_evidence), position + len(normalized) + 180))
        context = focused_evidence[left:right]
        dependency_cues = (
            "access", "acquire", "enter", "get ", "key", "need", "obtain",
            "open", "prerequisite", "require", "retrieve", "unlock", "use ",
            "进入", "前置", "取得", "开启", "获得", "解锁", "需要", "钥匙",
        )
        return 12 if any(cue in context for cue in dependency_cues) else 2

    async def _fetch_pages(self, domain: str, titles: list[str]) -> dict[str, Any]:
        cache_key = f"mediawiki-pages:v2:{domain}:{'|'.join(title.casefold() for title in titles)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {**cached, "_domain": domain}
        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(self.client.fetch_pages, domain=domain, titles=titles),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            return {"results": [], "_domain": domain}
        payload["_domain"] = domain
        self.cache.set(cache_key, payload)
        return payload

    async def _auto_index(self, *, game: str, sources: list[Source], content_by_url: dict[str, str]) -> None:
        if self.content_index is None or not self.settings.wiki_auto_index_enabled:
            return
        selected = sources[: self.settings.wiki_auto_index_pages_per_query]
        results = await asyncio.gather(
            *(
                self.content_index.index_content(
                    url=str(source.url),
                    game=game,
                    content=content_by_url.get(self.canonical_key(str(source.url)), ""),
                    title=source.title,
                    source_type="wiki",
                    game_version=source.game_version,
                    published_at=source.published_at,
                    skip_if_fresh=True,
                )
                for source in selected
            ),
            return_exceptions=True,
        )
        logger.info(
            "knowledge.wiki_auto_index",
            game=game,
            attempted=len(selected),
            ready=sum(1 for result in results if isinstance(result, dict) and result.get("status") in {"ready", "cached"}),
            failed=sum(1 for result in results if isinstance(result, BaseException)),
        )
