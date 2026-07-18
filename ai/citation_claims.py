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
    max_claims: int = 8,
) -> list[CitationClaim]:
    """Split direct evidence into bounded claims without an extra model call.

    The caller supplies only sources that passed its direct-evidence gate.  We
    retain short, question-relevant sentences rather than treating an entire
    page as one claim, which keeps citations attached to the narrowest
    available evidence span.
    """
    tokens = question_relevance_tokens(question)
    claims: list[CitationClaim] = []
    for source_index, source in enumerate(sources, start=1):
        if source_index not in eligible_source_indexes:
            continue
        passages = _passages(source.evidence or source.snippet or "")
        ranked = sorted(
            enumerate(passages),
            key=lambda item: _passage_score(item[1], tokens),
            reverse=True,
        )
        selected = 0
        for position, passage in ranked:
            if selected >= 3 or len(claims) >= max_claims:
                break
            if _passage_score(passage, tokens)[0] <= 0:
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


def _passage_score(passage: str, tokens: list[str]) -> tuple[int, int]:
    lowered = passage.casefold()
    matches = sum(1 for token in tokens if token_in_text(token, lowered))
    # Prefer compact evidence over a page-wide navigation or boilerplate blob
    # when both repeat the same entity names.
    return matches, -len(passage)
