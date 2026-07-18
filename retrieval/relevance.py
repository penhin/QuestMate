"""Game-agnostic relevance and source-page filtering."""

import re
from typing import Any
from urllib.parse import urlparse

from quality_policy import RELEVANCE_SCORE_POLICY
from retrieval.source_quality import (
    all_named_entities_match,
    matches_game_identity,
    minimum_entity_matches,
    page_source_quality,
    required_entity_groups_match,
    source_entity_groups,
    source_entity_tokens,
    token_in_text,
)

LORE_MARKERS = ("lore", "剧情", "背景")
NON_GAMEPLAY_PAGE_MARKERS = (
    "character biography",
    "fiction battles",
    "power scaling",
    "vs battles wiki",
)


def matches_game_name(*, text: str, game_names: list[str]) -> bool:
    return matches_game_identity(text=text, game_names=game_names)


def is_low_value_page(*, text: str, question: str) -> bool:
    """Reject generic indexes using URL shape, never a game or community name."""
    lowered_question = question.casefold()
    if "reddit - the heart of the internet" in text:
        return True
    if any(marker in text for marker in NON_GAMEPLAY_PAGE_MARKERS) and not any(
        marker in lowered_question for marker in LORE_MARKERS
    ):
        return True

    url_match = re.search(r"https?://[^\s]+", text)
    if not url_match:
        return False
    parsed = urlparse(url_match.group(0).rstrip(".,);]"))
    host = parsed.netloc.casefold().removeprefix("www.")
    path = parsed.path.casefold().rstrip("/")
    if host == "reddit.com" or host.endswith(".reddit.com"):
        return "/comments/" not in f"{path}/"
    if host == "steamcommunity.com" and path.startswith("/app/"):
        useful_sections = ("/discussions/", "/guides/", "/sharedfiles/")
        return not any(section in f"{path}/" for section in useful_sections)
    return False


def result_relevance_score(
    *,
    item: dict[str, Any],
    game: str,
    question: str,
    game_aliases: list[str] | None = None,
    required_entity_groups: list[list[str]] | None = None,
    query_confirms_game: bool = False,
) -> float:
    raw_text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content"))
    text = raw_text.casefold()
    if is_low_value_page(text=text, question=question):
        return 0

    game_names = [game, *(game_aliases or [])]
    required_entity_groups = required_entity_groups or []
    question_tokens = source_entity_tokens(question=question, game_names=game_names)
    entity_groups = (
        []
        if required_entity_groups
        else source_entity_groups(question=question, game_names=game_names)
    )
    # A localized guide page may not repeat the original store title in its
    # title, URL, or excerpt.  It can still be useful if the *trusted query*
    # explicitly bound the result to this game and the page independently
    # satisfies the requested entity checks below.  This only admits a
    # relaxed candidate; strict source quality still requires page identity.
    if not matches_game_name(text=raw_text, game_names=game_names) and not query_confirms_game:
        return 0
    if required_entity_groups and not required_entity_groups_match(
        groups=required_entity_groups,
        text=text,
    ):
        return 0
    if not question_tokens:
        return RELEVANCE_SCORE_POLICY.no_entity_score

    if entity_groups and not all_named_entities_match(groups=entity_groups, text=text):
        return 0
    matched = sum(1 for token in question_tokens if token_in_text(token, text))
    minimum_matches = minimum_entity_matches(question_tokens, entity_groups)
    if not required_entity_groups and matched < minimum_matches:
        return 0

    title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).casefold()
    focused_matches = sum(1 for token in question_tokens if token_in_text(token, title_url))
    coverage = matched / max(len(question_tokens), 1)
    if required_entity_groups:
        # A translated alias can satisfy the structured entity route without
        # sharing characters with the original question.
        coverage = max(coverage, 0.8)
    focus_bonus = min(
        focused_matches * RELEVANCE_SCORE_POLICY.title_match_bonus,
        RELEVANCE_SCORE_POLICY.title_bonus_cap,
    )
    return min(
        1.0,
        RELEVANCE_SCORE_POLICY.base_score + coverage * RELEVANCE_SCORE_POLICY.coverage_weight + focus_bonus,
    )


def is_high_quality_source(
    *,
    item: dict[str, Any],
    game: str,
    question: str,
    source_type: str,
    game_aliases: list[str] | None = None,
    required_entity_groups: list[list[str]] | None = None,
) -> bool:
    from quality_policy import SOURCE_EVIDENCE_QUALITY_POLICY, SOURCE_POLICIES

    relevance = result_relevance_score(
        item=item,
        game=game,
        game_aliases=game_aliases,
        question=question,
        required_entity_groups=required_entity_groups,
    )
    if relevance <= 0:
        return False
    policy = SOURCE_POLICIES.get(source_type, SOURCE_POLICIES["web"])
    quality, signals = page_source_quality(
        item=item,
        source_prior=policy.trust_score,
        game=game,
        game_aliases=game_aliases,
        question=question,
        relevance=relevance,
        evidence=str(item.get("content") or ""),
        required_entity_groups=required_entity_groups,
    )
    if source_type == "official":
        return (
            signals.game_identity > 0
            and signals.evidence_support >= SOURCE_EVIDENCE_QUALITY_POLICY.minimum_evidence_support
        )
    return (
        quality >= SOURCE_EVIDENCE_QUALITY_POLICY.strict_threshold
        and signals.evidence_support >= SOURCE_EVIDENCE_QUALITY_POLICY.minimum_evidence_support
    )
