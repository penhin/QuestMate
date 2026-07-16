"""Convert upstream result dictionaries into scored evidence sources."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from quality_policy import SEARCH_RESULT_WEIGHTS, SourcePolicy, domain_quality, intent_source_preference
from retrieval.relevance import result_relevance_score
from schemas import Source


@dataclass(frozen=True)
class BuiltSource:
    source: Source
    searchable_item: dict[str, Any]


def build_source(
    *,
    item: dict[str, Any],
    source_policy: SourcePolicy,
    game: str,
    game_aliases: list[str],
    question: str,
    intent: str,
    best_passage: Callable[..., str],
    evidence_max_chars: int,
    version_safety_score: Callable[..., float],
    extract_version: Callable[[str], str | None],
    parse_datetime: Callable[[Any], datetime | None],
) -> BuiltSource | None:
    url = str(item.get("url") or "").strip()
    if not url.startswith(("https://", "http://")):
        return None
    raw_content = str(item.get("raw_content") or "")
    evidence = best_passage(
        raw_content or str(item.get("content") or ""),
        question=question,
        max_chars=evidence_max_chars,
    )
    searchable_item = {**item, "content": f"{item.get('content') or ''} {evidence}"}
    relevance = result_relevance_score(
        item=searchable_item,
        game=game,
        game_aliases=game_aliases,
        question=question,
    )
    if relevance <= 0:
        return None
    version = version_safety_score(
        intent=intent,
        source_type=source_policy.source_type,
        text=f"{item.get('title') or ''} {url} {evidence}",
    )
    weighted_score = (
        float(item.get("score") or 0) * SEARCH_RESULT_WEIGHTS.retrieval
        + source_policy.trust_score * SEARCH_RESULT_WEIGHTS.trust
        + relevance * SEARCH_RESULT_WEIGHTS.relevance
        + intent_source_preference(intent, source_policy.source_type) * SEARCH_RESULT_WEIGHTS.intent
        + domain_quality(urlparse(url).netloc) * SEARCH_RESULT_WEIGHTS.domain
        + version * SEARCH_RESULT_WEIGHTS.version
    )
    return BuiltSource(
        source=Source(
            title=item.get("title") or url,
            url=url,
            snippet=item.get("content"),
            score=weighted_score,
            source_type=source_policy.source_type,
            trust_score=source_policy.trust_score,
            trust_label=source_policy.trust_label,
            evidence=evidence,
            published_at=parse_datetime(item.get("published_date") or item.get("published_at")),
            fetched_at=datetime.now(timezone.utc),
            game_version=extract_version(f"{item.get('title') or ''} {item.get('content') or ''} {evidence}"),
        ),
        searchable_item=searchable_item,
    )
