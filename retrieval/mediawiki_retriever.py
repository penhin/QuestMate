"""Direct MediaWiki retrieval, link expansion, and knowledge indexing."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog

from config import Settings
from quality_policy import SEARCH_NOISE_TOKENS, SourcePolicy
from query_tokens import question_relevance_tokens, relevance_tokens
from retrieval.relevance import is_high_quality_source, result_relevance_score
from schemas import Source

logger = structlog.get_logger()


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
        domains = [domain for domain in database_domains if domain][:2]
        query = " ".join(aliases[:2]).strip() or (planned_queries[0] if planned_queries else question)
        results = await asyncio.gather(*(self._fetch_search(domain, query, max_results) for domain in domains))
        context = f"{question} {' '.join(aliases)}"
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

    async def _fetch_search(self, domain: str, query: str, max_results: int) -> dict[str, Any]:
        cache_key = f"mediawiki:{domain}:{query.casefold()}:{max_results}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {**cached, "_domain": domain}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.client.search, domain=domain, query=query, max_results=max_results),
                timeout=self.settings.external_request_timeout_seconds,
            )
        except Exception:
            return {"results": [], "_domain": domain}
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
            for item in result.get("results", []):
                for title in item.get("links") or []:
                    normalized = str(title).strip()
                    key = (domain, normalized.casefold())
                    if not domain or not normalized or key in seen:
                        continue
                    seen.add(key)
                    score = sum(1 for token in query_tokens if token in normalized.casefold())
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
            fetched_pages=sum(len(result.get("results", [])) for result in expanded),
        )
        return list(expanded)

    async def _fetch_pages(self, domain: str, titles: list[str]) -> dict[str, Any]:
        cache_key = f"mediawiki-pages:{domain}:{'|'.join(title.casefold() for title in titles)}"
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
