"""Game-agnostic relevance and source-page filtering."""

import re
from typing import Any
from urllib.parse import urlparse

from quality_policy import RELEVANCE_SCORE_POLICY, SEARCH_NOISE_TOKENS
from query_tokens import question_relevance_tokens, relevance_tokens

LORE_MARKERS = ("lore", "剧情", "背景")
LOW_VALUE_KNOWLEDGE_HOSTS = ("villains.fandom.com",)
LOW_VALUE_KNOWLEDGE_MARKERS = ("all-fiction-battles", "vs battles wiki", "battle wiki")


def matches_game_name(*, text: str, game_names: list[str]) -> bool:
    for game_name in game_names:
        normalized = game_name.casefold().strip()
        if normalized and normalized in text:
            return True
        tokens = [token for token in relevance_tokens(normalized) if token != normalized]
        if tokens and all(token in text for token in tokens):
            return True
    return not any(game_name.strip() for game_name in game_names)


def is_low_value_page(*, text: str, question: str) -> bool:
    """Reject generic indexes using URL shape, never a game or community name."""
    lowered_question = question.casefold()
    if "reddit - the heart of the internet" in text:
        return True
    if any(host in text for host in LOW_VALUE_KNOWLEDGE_HOSTS) and not any(
        marker in lowered_question for marker in LORE_MARKERS
    ):
        return True
    if any(marker in text for marker in LOW_VALUE_KNOWLEDGE_MARKERS) and not any(
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
) -> float:
    text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).casefold()
    if is_low_value_page(text=text, question=question):
        return 0

    game_names = [game, *(game_aliases or [])]
    game_token_set = set(relevance_tokens(" ".join(game_names)))
    question_tokens = [
        token
        for token in question_relevance_tokens(question)
        if token not in game_token_set and token not in SEARCH_NOISE_TOKENS
    ]
    if not matches_game_name(text=text, game_names=game_names):
        return 0
    if not question_tokens:
        return RELEVANCE_SCORE_POLICY.no_entity_score

    matched = sum(1 for token in question_tokens if token in text)
    latin_entity_tokens = [token for token in question_tokens if re.fullmatch(r"[a-z0-9]+", token)]
    minimum_matches = 2 if len(latin_entity_tokens) >= 2 else 1
    if matched < minimum_matches:
        return 0

    title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).casefold()
    focused_matches = sum(1 for token in question_tokens if token in title_url)
    coverage = matched / max(len(question_tokens), 1)
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
) -> bool:
    if source_type == "official":
        return True
    title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).casefold()
    game_token_set = set(relevance_tokens(" ".join([game, *(game_aliases or [])])))
    entity_tokens = [
        token
        for token in question_relevance_tokens(question)
        if token not in game_token_set and token not in SEARCH_NOISE_TOKENS
    ]
    title_entity_matches = sum(1 for token in entity_tokens if token in title_url)
    if source_type == "wiki":
        return title_entity_matches > 0
    if source_type == "community":
        return title_entity_matches > 0 and any(value in title_url for value in ("comments", "discussions"))
    return title_entity_matches > 0
