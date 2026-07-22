"""Confidence policy for independently discovered game-identity signals."""

from quality_policy import GAME_RESOLUTION_POLICY
from query_tokens import relevance_tokens


def resolution_confidence(
    *,
    game: str,
    aliases: list[str],
    platform_urls: list[str],
    official_urls: list[str],
    identity_urls: list[str],
    database_domains: list[str],
) -> float:
    score = GAME_RESOLUTION_POLICY.base_confidence
    if aliases:
        score += GAME_RESOLUTION_POLICY.alias_bonus
    if platform_urls:
        score += GAME_RESOLUTION_POLICY.platform_bonus
    if official_urls:
        score += GAME_RESOLUTION_POLICY.official_bonus
    if identity_urls:
        score += GAME_RESOLUTION_POLICY.identity_candidate_bonus
    if database_domains:
        score += GAME_RESOLUTION_POLICY.database_bonus
    if identity_urls and not platform_urls and not official_urls:
        score = min(score, GAME_RESOLUTION_POLICY.confirmed_threshold - 0.01)
    if not relevance_tokens(game):
        score -= GAME_RESOLUTION_POLICY.invalid_name_penalty
    return max(0, min(1, score))
