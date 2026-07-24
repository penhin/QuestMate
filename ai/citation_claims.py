"""Deterministic, source-indexed atomic claims for answer citation binding."""

import re

from query_tokens import question_relevance_tokens
from retrieval.source_quality import token_in_text
from schemas import CitationClaim, Source


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;.])\s*|\n+(?:[-*•]\s*)?")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
_BREADCRUMB = re.compile(r"^[^。！？!?]{0,180}(?:>|›|»)[^。！？!?]{0,180}$")
_TRUNCATED_SNIPPET = re.compile(r"(?:\.\.\.|…)(?:[\]）)】\"'”’]*)$")
_URL_OR_LINK_FRAGMENT = re.compile(
    r"(?:https?://|www\.|\]\(|\)\s*>|(?:[a-z0-9-]+\.)+(?:com|cn|net|org|io|gg)/)",
    re.IGNORECASE,
)
_NAVIGATION_PREFIX = re.compile(r"^(?:首页|手机游戏|游戏攻略|综合篇|作者[：:]|来源[：:]|发布时间[：:])")
_PAGE_META_OR_PROMOTION = re.compile(
    r"^(?:下面请看|一起来看看|更新于\d{4}-\d{1,2}-\d{1,2}|jpg\)\s*更新于|"
    r"有很多小伙伴问|本期流程.{0,80}(?:三连|关注|点赞)|.*(?:交流群|更多攻略).{0,80})"
)
_ORDERED_STEP = re.compile(r"(?:^|\s)(?:第\s*\d+\s*步|step\s*\d+)\b", re.IGNORECASE)


def claim_ids_cover_entity_groups(
    *,
    claims: list[CitationClaim],
    claim_ids: list[str],
    entity_groups: list[list[str]],
) -> bool:
    """Check that a relation block cites evidence for every named endpoint.

    This is deliberately a coverage guard, not a semantic inference engine:
    it prevents a model from presenting a relationship while citing only one
    side of it. The answer prompt remains responsible for requiring that the
    relationship itself is stated by the cited evidence.
    """
    if len(entity_groups) < 2:
        return True
    selected = [claim for claim in claims if claim.claim_id in set(claim_ids)]
    return all(
        any(
            any(token_in_text(name.casefold(), claim.statement.casefold()) for name in group)
            for claim in selected
        )
        for group in entity_groups
    )

def build_citation_claims(
    *,
    question: str,
    sources: list[Source],
    eligible_source_indexes: set[int],
    entity_groups: list[list[str]] | None = None,
    aliases: list[str] | None = None,
    evidence_queries: list[str] | None = None,
    max_claims: int = 8,
) -> list[CitationClaim]:
    """Split direct evidence into bounded claims without an extra model call.

    The caller supplies only sources that passed its direct-evidence gate.  We
    retain short, question-relevant sentences rather than treating an entire
    page as one claim, which keeps citations attached to the narrowest
    available evidence span.
    """
    # Planner queries may preserve an alias or relationship surface in the
    # evidence language when the player's question is translated.  They are
    # ranking hints only: a query never makes an ineligible source factual
    # evidence, and all terms remain request-derived rather than vocabulary
    # maintained by the application.
    tokens = list(dict.fromkeys([
        *question_relevance_tokens(question),
        *(token for query in evidence_queries or [] for token in question_relevance_tokens(query)),
    ]))
    groups = entity_groups or []
    surface_aliases = [value.casefold() for value in (aliases or []) if value.strip()]
    # Build a per-source ranked queue first.  Taking three passages from the
    # first pages can exhaust the ledger before a later, independently
    # eligible source contributes the prerequisite or outcome needed for the
    # answer.  A round-robin allocation preserves evidence-chain coverage
    # without increasing the claim or model-call budget.
    ranked_by_source: list[tuple[int, list[tuple[int, str]]]] = []
    for source_index, source in enumerate(sources, start=1):
        if source_index not in eligible_source_indexes:
            continue
        passages = _passages(source.evidence or source.snippet or "")
        candidates = list(enumerate(passages))
        # A relationship is often expressed across adjacent sentences: the
        # first names an object and the next describes its condition, outcome,
        # or sequence with a pronoun. Keep that original two-sentence span as
        # one auditable Claim only when it covers more requested endpoints than
        # either sentence alone. This preserves source wording without
        # guessing an action vocabulary or adding a model/search call.
        if len(groups) >= 2:
            for position in range(len(passages) - 1):
                combined = f"{passages[position]} {passages[position + 1]}"
                combined_matches = _passage_score(combined, tokens, groups, surface_aliases)[0]
                separate_matches = max(
                    _passage_score(passages[position], tokens, groups, surface_aliases)[0],
                    _passage_score(passages[position + 1], tokens, groups, surface_aliases)[0],
                )
                if combined_matches > separate_matches:
                    candidates.append((len(passages) + position, combined))
        ranked = sorted(
            candidates,
            key=lambda item: (
                -_passage_score(item[1], tokens, groups, surface_aliases)[0],
                len(item[1]),
                item[0],
            ),
        )
        selected_passages: list[tuple[int, str]] = []
        for position, passage in ranked:
            if len(selected_passages) >= 3:
                break
            if _passage_score(passage, tokens, groups, surface_aliases)[0] <= 0:
                continue
            selected_passages.append((position, passage))
        if selected_passages:
            ranked_by_source.append((source_index, selected_passages))

    claims: list[CitationClaim] = []
    for passage_offset in range(3):
        for source_index, passages in ranked_by_source:
            if passage_offset >= len(passages) or len(claims) >= max_claims:
                continue
            position, passage = passages[passage_offset]
            claims.append(CitationClaim(
                claim_id=f"C{source_index}_{position + 1}",
                source_index=source_index,
                statement=passage,
            ))
        if len(claims) >= max_claims:
            break
    return claims


def _passages(evidence: str) -> list[str]:
    # A provider may split ``...`` into a final single-dot fragment below.
    # Reject the whole result up front so a truncated search preview cannot
    # accidentally become a Claim after sentence splitting.
    if _TRUNCATED_SNIPPET.search(evidence.strip()):
        return []
    values: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(evidence):
        cleaned = " ".join(part.split()).strip(" -•")
        # Short sentences often contain the decisive fact (a binary condition,
        # location, or version value). Relevance and entity coverage are
        # checked later, so do not discard them merely because a longer,
        # unrelated sentence shares the same evidence passage.
        if 8 <= len(cleaned) <= 700 and _is_direct_evidence_passage(cleaned):
            values.append(cleaned)
    if not values:
        cleaned = " ".join(evidence.split())[:700]
        if cleaned and _is_direct_evidence_passage(cleaned):
            values.append(cleaned)
    return list(dict.fromkeys(values))


def _is_direct_evidence_passage(passage: str) -> bool:
    """Reject page chrome and incomplete snippets before they enter the Claim ledger.

    Search results frequently prepend Markdown headings or navigation paths, and
    some providers return a sentence cut off with an ellipsis.  Neither form is
    an auditable statement of game fact, even when it happens to contain the
    player's entity name.
    """
    if _MARKDOWN_HEADING.match(passage):
        return False
    if _BREADCRUMB.match(passage):
        return False
    if _TRUNCATED_SNIPPET.search(passage):
        return False
    if _URL_OR_LINK_FRAGMENT.search(passage):
        return False
    if passage.startswith(("[", "]", "(", ")", ">", "|")):
        return False
    if _NAVIGATION_PREFIX.match(passage):
        return False
    if "#" in passage or _PAGE_META_OR_PROMOTION.match(passage):
        return False
    return True


def _passage_score(
    passage: str,
    tokens: list[str],
    entity_groups: list[list[str]] | None = None,
    aliases: list[str] | None = None,
) -> tuple[int, int]:
    lowered = passage.casefold()
    # Entity coverage decides whether a passage may establish an endpoint, but
    # it does not by itself identify the requested relationship. Among
    # passages with equal endpoint coverage, prefer the one sharing more
    # question-derived terms. This works for unseen actions and languages
    # because every term is taken from the player's question, not a maintained
    # guide vocabulary.
    relevance_matches = sum(1 for token in tokens if token_in_text(token, lowered))
    # A numbered action is useful primary evidence for a request that itself
    # asks for steps, while page introductions often repeat the same entity
    # names without advancing the route.
    ordered_step_bonus = 8 if _ORDERED_STEP.search(passage) and any(
        token_in_text(token, "步骤 steps step questline route") for token in tokens
    ) else 0
    if entity_groups:
        matches = sum(
            1 for group in entity_groups if any(token_in_text(value.casefold(), lowered) for value in group)
        )
        # Aliases are alternate surfaces of a requested entity, not extra
        # relation endpoints. They make translated evidence selectable without
        # weakening the distinct entity-group requirements.
        alias_match = any(token_in_text(value, lowered) for value in aliases or [])
        return (matches + int(alias_match)) * 100 + min(relevance_matches + ordered_step_bonus, 99), -len(passage)
    matches = relevance_matches + ordered_step_bonus
    # The caller keeps the original passage position as the final tie-breaker,
    # so duplicated/translated page bodies cannot displace earlier evidence.
    return matches, -len(passage)
