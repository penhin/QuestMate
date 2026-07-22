"""Defensive JSON-shape handling for model-produced search plans."""

import json


SEARCH_INTENTS = {
    "game_identity", "boss_strategy", "item_location", "item_usage", "quest_step",
    "game_mechanic", "build", "patch", "lore", "general",
}
SEARCH_SOURCE_TYPES = {"official", "wiki", "community", "web"}


def first_json_object(content: str) -> object:
    """Extract a complete object from model wrappers without delimiter assumptions."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            nested, _ = decoder.raw_decode(value.lstrip())
            if isinstance(nested, dict):
                return nested
    raise ValueError("No complete JSON object found")


def coerce_search_plan_data(data: object) -> dict:
    """Accept harmless JSON-shape variation without inventing plan facts."""
    if not isinstance(data, dict):
        raise TypeError("search plan must be an object")
    normalized = dict(data)
    for field in ("safety_refusal", "version_sensitive", "requires_relation_verification"):
        if not isinstance(normalized.get(field), bool):
            normalized[field] = False
    if normalized.get("intent") not in SEARCH_INTENTS:
        normalized["intent"] = "general"
    groups = normalized.get("named_entity_groups")
    if isinstance(groups, dict):
        groups = [groups]
    if isinstance(groups, list):
        normalized_groups: list[list[str]] = []
        for value in groups:
            if isinstance(value, str):
                candidates = [value]
            elif isinstance(value, dict):
                candidates = value.get("names", value.get("aliases", value.get("entity", [])))
                candidates = [candidates] if isinstance(candidates, str) else candidates
            else:
                candidates = value
            if not isinstance(candidates, list):
                continue
            cleaned = [item.strip() for item in candidates if isinstance(item, str) and item.strip()]
            if cleaned:
                normalized_groups.append(cleaned[:4])
        normalized["named_entity_groups"] = normalized_groups[:4]
    for field in ("aliases", "answer_requirements", "missing_info"):
        if isinstance(normalized.get(field), str):
            normalized[field] = [normalized[field]]
        elif not isinstance(normalized.get(field), list):
            normalized[field] = []
    queries = normalized.get("queries")
    if isinstance(queries, list):
        normalized["queries"] = [
            {"source_type": "web", "query": value}
            if isinstance(value, str)
            else {
                "source_type": value.get("source_type", value.get("type", "web")),
                "query": value.get("query", value.get("text", "")),
            }
            if isinstance(value, dict)
            else value
            for value in queries
        ]
        for query in normalized["queries"]:
            if isinstance(query, dict) and query.get("source_type") not in SEARCH_SOURCE_TYPES:
                query["source_type"] = "web"
    return normalized
