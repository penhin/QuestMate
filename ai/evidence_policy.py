"""Evidence sufficiency and version-safety policy for guide answers."""

from query_tokens import is_query_entity_token, question_relevance_tokens
from schemas import ChatRequest, SearchIntent, SearchPlan, Source


def has_question_specific_sources(*, question: str, sources: list[Source]) -> bool:
    primary_question, separator, alias_text = question.partition("\nALIASES:")
    tokens = [token for token in question_relevance_tokens(primary_question) if is_query_entity_token(token)]
    alias_groups = [
        [token for token in question_relevance_tokens(alias) if is_query_entity_token(token)]
        for alias in alias_text.split("|")
        if separator and alias.strip()
    ]
    alias_groups = [group for group in alias_groups if group]
    if not tokens and not alias_groups:
        return False

    minimum_matches = 1 if len(tokens) <= 1 else max(2, (len(tokens) + 2) // 3)
    for source in sources:
        source_text = f"{source.title} {source.url} {source.evidence or source.snippet or ''}".casefold()
        primary_match = tokens and sum(1 for token in tokens if token in source_text) >= minimum_matches
        alias_match = any(all(token in source_text for token in group) for group in alias_groups)
        if primary_match or alias_match:
            return True
    return False


def evidence_level(*, question: str, sources: list[Source]) -> str:
    if not sources:
        return "none"
    return "direct" if has_question_specific_sources(question=question, sources=sources) else "game_only"


def evidence_policy_for_level(level: str) -> str:
    if level == "direct":
        return "Sources directly mention the requested entity. Answer with sourced concrete details and note uncertainty where needed."
    if level == "game_only":
        return (
            "Sources appear to cover the game but not the requested entity. Do not provide concrete item effects, "
            "locations, materials, NPCs, or step-by-step instructions. Say the direct evidence was not found and ask "
            "for more context."
        )
    return (
        "No usable sources were found. Do not infer a gameplay answer from genre conventions. Say reliable "
        "information was not found and ask for the original title, screenshot, area name, or more context."
    )


def version_evidence_status(*, intent: SearchIntent, sources: list[Source]) -> str:
    if intent not in {"patch", "build", "boss_strategy", "game_mechanic"}:
        return "not_version_sensitive"
    versioned = [source for source in sources if source.game_version or source.published_at]
    official_versioned = [source for source in versioned if source.source_type == "official"]
    if intent == "patch":
        return (
            "verified_official_version"
            if official_versioned
            else "insufficient: no official source with a version number or publication date"
        )
    if official_versioned:
        return "official_version_context"
    if versioned:
        return "dated_non_official_context: state that the recommendation may differ by version"
    return "unknown_version: do not describe balance, AI behavior, or build strength as current fact"


def evidence_question(*, request: ChatRequest, plan: SearchPlan | None) -> str:
    aliases = " | ".join((plan.aliases if plan else [])[:6])
    return request.question if not aliases else f"{request.question}\nALIASES:{aliases}"


def has_unsupported_specifics(*, answer: str, sources: list[Source], question: str) -> bool:
    if evidence_level(question=question, sources=sources) == "direct":
        return False
    lowered = answer.casefold()
    uncertainty_markers = (
        "通常", "一般", "可能", "推断", "合理推断", "常规设计", "based on", "usually", "likely", "probably",
    )
    concrete_markers = (
        "npc", "材料", "地点", "区域", "房间", "机关", "交互", "步骤", "路线", "地标",
        "奖励", "数值", "最大值", "指定", "先", "然后", "再",
    )
    return any(marker in lowered for marker in uncertainty_markers) and any(
        marker in lowered for marker in concrete_markers
    )
