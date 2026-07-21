"""Merge, score, and diversify evidence independently of agent state."""

from collections.abc import Sequence
from urllib.parse import urlparse, urlunparse

from quality_policy import (
    EVIDENCE_POOL_WEIGHTS,
    MAX_MERGED_EVIDENCE_CHARS,
    VERSION_SENSITIVE_INTENTS,
    source_domain_limit,
)
from query_tokens import question_relevance_tokens
from retrieval.source_quality import token_in_text
from schemas import Source


DIRECT_EVIDENCE_BONUS = 0.3


def canonical_source_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", "", ""))


def merge_source_evidence(*, preferred: Source, other: Source) -> Source:
    passages: list[str] = []
    for passage in (preferred.evidence or preferred.snippet or "", other.evidence or other.snippet or ""):
        cleaned = passage.strip()
        if not cleaned or any(cleaned == existing or cleaned in existing for existing in passages):
            continue
        passages = [existing for existing in passages if existing not in cleaned]
        passages.append(cleaned)
    evidence = "\n\n".join(passages)[:MAX_MERGED_EVIDENCE_CHARS]
    return preferred.model_copy(
        update={"evidence": evidence or preferred.evidence, "snippet": evidence[:600] if evidence else preferred.snippet}
    )


def source_rank(
    *,
    source: Source,
    query: str | Sequence[str],
    intent: str,
    version_sensitive: bool = False,
    entity_groups: list[list[str]] | None = None,
) -> float:
    text = f"{source.title} {source.evidence or source.snippet or ''}".casefold()
    evidence_text = (source.evidence or source.snippet or "").casefold()
    groups = entity_groups or []
    query_coverage = _query_coverage(query=query, text=text)
    query_evidence_coverage = _query_coverage(query=query, text=evidence_text)
    if groups:
        coverage = sum(
            1 for group in groups if any(token_in_text(value.casefold(), text) for value in group)
        ) / len(groups)
        evidence_coverage = sum(
            1 for group in groups if any(token_in_text(value.casefold(), evidence_text) for value in group)
        ) / len(groups)
        # Entity coverage prevents an unrelated page from entering the pool,
        # while planner query surfaces distinguish a relation-bearing passage
        # from an entity-only overview. A translated surface is evaluated as
        # an alternative rather than diluted by the player's original wording.
        # Prefer the explicit entity route whenever it matches. Only fall
        # back to a planner surface when the player-facing entity spelling is
        # absent from the page, which is the expected translated-alias case.
        # This preserves the established ranking for same-language evidence.
        if coverage == 0:
            coverage = query_coverage
        if evidence_coverage == 0:
            evidence_coverage = query_evidence_coverage
    else:
        # Fallback plans intentionally preserve the full user query. This is a
        # weak, vocabulary-free tie-breaker rather than an eligibility gate.
        coverage = query_coverage
        evidence_coverage = query_evidence_coverage
    retrieval_score = min(max(source.score or 0.5, 0), 1)
    version_score = 1.0 if source.game_version or source.published_at else 0.0
    if not version_sensitive and intent not in VERSION_SENSITIVE_INTENTS:
        version_score = 0.5
    return (
        coverage * EVIDENCE_POOL_WEIGHTS.relevance
        + retrieval_score * EVIDENCE_POOL_WEIGHTS.retrieval
        + source.trust_score * EVIDENCE_POOL_WEIGHTS.trust
        + version_score * EVIDENCE_POOL_WEIGHTS.version
        # A title can match the question while the excerpt only covers a
        # broad overview. Prefer the page whose actual evidence passage
        # repeats the user's target, without filtering out lower-prior sites.
        + evidence_coverage * DIRECT_EVIDENCE_BONUS
    )


def _query_coverage(*, query: str | Sequence[str], text: str) -> float:
    """Return the strongest lexical coverage across request-derived surfaces."""
    variants = [query] if isinstance(query, str) else list(query)
    scores: list[float] = []
    for variant in variants:
        tokens = question_relevance_tokens(str(variant))
        if tokens:
            scores.append(sum(1 for token in tokens if token_in_text(token, text)) / len(tokens))
    return max(scores, default=0.0)


def rank_sources(
    *,
    sources: list[Source],
    query: str | Sequence[str],
    intent: str,
    max_results: int,
    version_sensitive: bool = False,
    entity_groups: list[list[str]] | None = None,
) -> list[Source]:
    ranked_by_url: dict[str, tuple[float, Source]] = {}
    for source in sources:
        key = canonical_source_url(str(source.url))
        rank = source_rank(
            source=source,
            query=query,
            intent=intent,
            version_sensitive=version_sensitive,
            entity_groups=entity_groups,
        )
        current = ranked_by_url.get(key)
        if current is None or rank > current[0]:
            preferred = source if current is None else merge_source_evidence(preferred=source, other=current[1])
            ranked_by_url[key] = (rank, preferred)
        else:
            ranked_by_url[key] = (current[0], merge_source_evidence(preferred=current[1], other=source))

    selected: list[Source] = []
    domain_counts: dict[str, int] = {}
    for _rank, source in sorted(ranked_by_url.values(), key=lambda item: item[0], reverse=True):
        domain = urlparse(str(source.url)).netloc.casefold()
        if domain_counts.get(domain, 0) >= source_domain_limit(domain):
            continue
        selected.append(source)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= max_results:
            break
    return selected
