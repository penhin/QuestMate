"""Deterministic, source-indexed atomic claims for answer citation binding."""

import re

from query_tokens import question_relevance_tokens
from retrieval.source_quality import token_in_text
from schemas import CitationClaim, Source


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;.])\s*|\n+(?:[-*•]\s*)?")

def build_citation_claims(
    *,
    question: str,
    sources: list[Source],
    eligible_source_indexes: set[int],
    entity_groups: list[list[str]] | None = None,
    aliases: list[str] | None = None,
    max_claims: int = 8,
) -> list[CitationClaim]:
    """Split direct evidence into bounded claims without an extra model call.

    The caller supplies only sources that passed its direct-evidence gate.  We
    retain short, question-relevant sentences rather than treating an entire
    page as one claim, which keeps citations attached to the narrowest
    available evidence span.
    """
    tokens = question_relevance_tokens(question)
    groups = entity_groups or []
    surface_aliases = [value.casefold() for value in (aliases or []) if value.strip()]
    claims: list[CitationClaim] = []
    for source_index, source in enumerate(sources, start=1):
        if source_index not in eligible_source_indexes:
            continue
        passages = _passages(source.evidence or source.snippet or "")
        ranked = sorted(
            enumerate(passages),
            key=lambda item: (
                -_passage_score(item[1], tokens, groups, surface_aliases)[0],
                len(item[1]),
                item[0],
            ),
        )
        selected = 0
        for position, passage in ranked:
            if selected >= 3 or len(claims) >= max_claims:
                break
            if _passage_score(passage, tokens, groups, surface_aliases)[0] <= 0:
                continue
            claims.append(CitationClaim(
                claim_id=f"C{source_index}_{position + 1}",
                source_index=source_index,
                statement=passage,
            ))
            selected += 1
    return claims


def _passages(evidence: str) -> list[str]:
    values: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(evidence):
        cleaned = " ".join(part.split()).strip(" -•")
        if 24 <= len(cleaned) <= 700:
            values.append(cleaned)
    if not values:
        cleaned = " ".join(evidence.split())[:700]
        if cleaned:
            values.append(cleaned)
    return list(dict.fromkeys(values))


def _passage_score(
    passage: str,
    tokens: list[str],
    entity_groups: list[list[str]] | None = None,
    aliases: list[str] | None = None,
) -> tuple[int, int]:
    lowered = passage.casefold()
    if entity_groups:
        matches = sum(
            1 for group in entity_groups if any(token_in_text(value.casefold(), lowered) for value in group)
        )
        # Aliases are alternate surfaces of a requested entity, not extra
        # relation endpoints. They make translated evidence selectable without
        # weakening the distinct entity-group requirements.
        alias_match = any(token_in_text(value, lowered) for value in aliases or [])
        return matches + int(alias_match), -len(passage)
    matches = sum(
        1
        for token in tokens
        if token_in_text(token, lowered)
    )
    # The caller keeps the original passage position as the final tie-breaker,
    # so duplicated/translated page bodies cannot displace earlier evidence.
    return matches, -len(passage)
