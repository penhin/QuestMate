"""Evidence sufficiency and version-safety policy for guide answers."""

import json
import re
from math import ceil

from query_tokens import (
    exact_identifiers,
    is_query_entity_token,
    minimum_cjk_ngram_matches,
    question_named_entity_groups,
    question_relevance_tokens,
)
from retrieval.source_quality import required_entity_groups_match, token_in_text
from schemas import ChatRequest, SearchIntent, SearchPlan, Source


def has_question_specific_sources(*, question: str, sources: list[Source]) -> bool:
    primary_question, aliases, required_entity_groups = _evidence_question_metadata(question)
    tokens = [token for token in question_relevance_tokens(primary_question) if is_query_entity_token(token)]
    entity_groups = [] if required_entity_groups else question_named_entity_groups(primary_question)
    alias_groups = [
        [token for token in question_relevance_tokens(alias) if is_query_entity_token(token)]
        for alias in aliases
        if alias.strip()
    ]
    alias_groups = [group for group in alias_groups if group]
    if not tokens and not entity_groups and not alias_groups and not required_entity_groups:
        return False

    required_identifiers = exact_identifiers(primary_question)
    focus_tokens = _fallback_focus_tokens(primary_question, tokens)
    minimum_matches = _minimum_direct_matches(tokens)
    for source in sources:
        source_text = f"{source.title} {source.url} {source.evidence or source.snippet or ''}".casefold()
        if required_identifiers and not all(
            token_in_text(identifier, source_text)
            for identifier in required_identifiers
        ):
            continue
        if required_entity_groups:
            if required_entity_groups_match(groups=required_entity_groups, text=source_text):
                return True
            continue
        if any(all(token_in_text(token, source_text) for token in group) for group in alias_groups):
            return True

        # Quoted, titled, and determiner-bound names are high-confidence entity
        # groups.  Every endpoint must occur in the same source; distributing a
        # relationship across unrelated pages is not direct evidence.
        if entity_groups:
            if all(
                all(token_in_text(token, source_text) for token in group)
                for group in entity_groups
            ):
                return True
            continue

        matched = sum(1 for token in tokens if token_in_text(token, source_text))
        if matched < minimum_matches:
            continue
        # When capitalization provides no entity hint, use question structure
        # to keep a generic predicate ("find", "obtain", etc.) from passing the
        # gate by itself.  This deliberately does not enumerate action verbs.
        if focus_tokens and not any(token_in_text(token, source_text) for token in focus_tokens):
            continue
        return True
    return False


def requires_semantic_relation_judgment(question: str) -> bool:
    """Return whether co-occurrence alone cannot establish the requested fact.

    Two independently identified endpoints imply a relationship or comparison
    whose predicate must be supported semantically.  The endpoint extractor is
    based on naming and question grammar, so this does not encode any game's
    entities or a fixed list of actions.  Alias metadata is deliberately
    excluded: aliases are alternate names, not additional relation endpoints.
    """
    primary_question, _aliases, required_entity_groups = _evidence_question_metadata(question)
    return len(required_entity_groups or question_named_entity_groups(primary_question)) >= 2


def _minimum_direct_matches(tokens: list[str]) -> int:
    if not tokens:
        return 1
    script_floor = minimum_cjk_ngram_matches(tokens)
    if script_floor > 1:
        return script_floor
    latin_tokens = [token for token in tokens if re.fullmatch(r"[a-z0-9]+", token)]
    if len(latin_tokens) <= 2:
        return 1
    return max(2, ceil(len(latin_tokens) * 0.6))


def _fallback_focus_tokens(question: str, tokens: list[str]) -> list[str]:
    """Infer an uncapitalized English focus without a game/action vocabulary."""
    identifiers = set(exact_identifiers(question))
    latin = [
        token
        for token in tokens
        if re.fullmatch(r"[a-z0-9]+", token) and token not in identifiers
    ]
    if not latin:
        return []

    normalized = " ".join(question.casefold().split())
    # Passive/location forms put the subject immediately after the auxiliary:
    # "where is moonstone acquired" / "where does moonstone drop".
    if re.match(r"^(?:where|when|who|what|which)\s+(?:is|are|was|were)\b", normalized):
        return latin[:1]
    if re.match(
        r"^(?:where|when|who|what|which)\s+(?:do|does|did)\s+"
        r"(?!(?:i|we|you|one|players?)\b)",
        normalized,
    ):
        return latin[:1]
    if re.match(
        r"^(?:where|when|who|what|which)\s+"
        r"(?:can|could|may|might|must|should|will|would)\s+"
        r"(?!(?:i|we|you|one|players?)\b)",
        normalized,
    ):
        return latin[:1]
    # Active requests normally place the requested object last: "where can I
    # obtain moonstone".  For longer yes/no relations retain both boundaries;
    # the coverage floor still requires the relationship's other endpoint.
    if len(latin) <= 2 or re.match(r"^(?:where|when|who|what|which|how)\b", normalized):
        return latin[-1:]
    return list(dict.fromkeys([latin[0], latin[-1]]))


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


def version_evidence_status(
    *,
    intent: SearchIntent,
    sources: list[Source],
    version_sensitive: bool = False,
    question: str = "",
) -> str:
    if not version_sensitive and intent not in {"patch", "build", "boss_strategy", "game_mechanic"}:
        return "not_version_sensitive"
    versioned = [
        source
        for source in sources
        if (source.game_version or source.published_at)
        and _version_source_matches_question(question=question, source=source)
    ]
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


def _version_source_matches_question(*, question: str, source: Source) -> bool:
    """Require version metadata and target evidence to occur on the same page.

    A dated news post must not make an undated entity page current merely because
    both survived retrieval.  High-confidence names and exact identifiers are
    strict; for scripts without reliable word boundaries, a small n-gram overlap
    keeps paraphrased long-tail sources usable while failing safely on unrelated
    pages.
    """
    if not question.strip():
        # Compatibility for callers that only need the coarse legacy status.
        return True

    primary_question, aliases, required_entity_groups = _evidence_question_metadata(question)
    source_text = (
        f"{source.title} {source.url} {source.evidence or source.snippet or ''} "
        f"{source.game_version or ''}"
    ).casefold()
    for identifier in exact_identifiers(primary_question):
        alternatives = [identifier]
        if identifier.startswith("v") and len(identifier) > 1 and identifier[1].isdigit():
            alternatives.append(identifier[1:])
        if not any(token_in_text(value, source_text) for value in alternatives):
            return False

    if required_entity_groups:
        return required_entity_groups_match(groups=required_entity_groups, text=source_text)

    identity_routes: list[list[list[str]]] = []
    primary_groups = question_named_entity_groups(primary_question)
    if primary_groups:
        identity_routes.append(primary_groups)
    for alias in aliases:
        alias_tokens = [
            token
            for token in question_relevance_tokens(alias)
            if is_query_entity_token(token)
        ]
        if alias_tokens:
            identity_routes.append([alias_tokens])
    if identity_routes:
        return any(
            all(
                all(token_in_text(token, source_text) for token in group)
                for group in route
            )
            for route in identity_routes
        )

    focus_tokens = [
        token
        for token in question_relevance_tokens(primary_question)
        if is_query_entity_token(token) and token not in exact_identifiers(primary_question)
    ]
    if not focus_tokens:
        return True
    return any(_version_focus_token_matches(token=token, text=source_text) for token in focus_tokens)


def _version_focus_token_matches(*, token: str, text: str) -> bool:
    if not re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
        return token_in_text(token, text)
    grams = [token[index:index + 2] for index in range(len(token) - 1)]
    required = min(3, max(1, ceil(len(grams) * 0.4)))
    return sum(gram in text for gram in grams) >= required


def evidence_question(*, request: ChatRequest, plan: SearchPlan | None) -> str:
    game_key = " ".join(request.game.casefold().split())
    question_key = " ".join(request.question.casefold().split())
    named_entity_groups = [
        group
        for group in (plan.named_entity_groups if plan else [])[:4]
        # The game is an identity boundary, not an answer entity. Planner
        # output can redundantly add it as a required group, which rejects
        # localized pages that otherwise directly answer the question.
        if not any(" ".join(value.casefold().split()) == game_key for value in group)
        # A translated entity invented by the planner is a useful retrieval
        # alias but cannot become a mandatory answer-evidence endpoint. Keep
        # only groups anchored in the wording the player actually supplied.
        and any(" ".join(value.casefold().split()) in question_key for value in group)
    ]
    metadata = {
        "aliases": list((plan.aliases if plan else [])[:6]),
        "named_entity_groups": named_entity_groups,
    }
    if not metadata["aliases"] and not metadata["named_entity_groups"]:
        return request.question
    serialized = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"{request.question}\nEVIDENCE_METADATA:{serialized}"


def _evidence_question_metadata(question: str) -> tuple[str, list[str], list[list[str]]]:
    primary_question, separator, serialized = question.rpartition("\nEVIDENCE_METADATA:")
    if separator:
        try:
            metadata = json.loads(serialized)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        raw_aliases = metadata.get("aliases")
        aliases = [value for value in raw_aliases if isinstance(value, str)] if isinstance(raw_aliases, list) else []
        raw_groups = metadata.get("named_entity_groups")
        groups = [
            [value for value in group if isinstance(value, str) and value.strip()]
            for group in raw_groups
            if isinstance(group, list)
        ] if isinstance(raw_groups, list) else []
        return primary_question, aliases[:6], [group[:4] for group in groups if group][:4]

    # Compatibility for stored/test contexts created before structured groups.
    primary_question, legacy_separator, alias_text = question.partition("\nALIASES:")
    aliases = alias_text.split("|") if legacy_separator else []
    return primary_question, aliases[:6], []


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
