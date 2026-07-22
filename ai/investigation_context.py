"""Bounded prompt serialization for iterative investigation state."""

import json

from schemas import InvestigationState


def investigation_context(investigation: InvestigationState | None) -> str:
    """Serialize investigation state within the planner's fixed prompt budget."""
    if investigation is None:
        return "Not provided."
    max_chars = 7000
    data = investigation.model_dump(mode="json")
    data["goal"] = str(data.get("goal") or "")[:700]
    data["known_facts"] = [
        {**fact, "statement": str(fact.get("statement") or "")[:350]}
        for fact in data.get("known_facts", [])[-10:]
    ]
    data["evidence_gaps"] = [
        {
            **gap,
            "description": str(gap.get("description") or "")[:240],
            "query_hint": str(gap.get("query_hint") or "")[:180] or None,
        }
        for gap in data.get("evidence_gaps", [])[:6]
    ]
    data["unresolved_questions"] = [
        str(value)[:240] for value in data.get("unresolved_questions", [])[:6]
    ]
    data["attempted_queries"] = [
        str(value)[:180] for value in data.get("attempted_queries", [])[-10:]
    ]
    data["aliases"] = [str(value)[:80] for value in data.get("aliases", [])[:6]]
    data["next_queries"] = [
        {**query, "query": str(query.get("query") or "")[:180]}
        for query in data.get("next_queries", [])[:2]
        if isinstance(query, dict)
    ]

    gap_descriptions = {
        " ".join(str(gap.get("description") or "").casefold().split())
        for gap in data["evidence_gaps"]
    }
    data["unresolved_questions"] = [
        value
        for value in data["unresolved_questions"]
        if " ".join(value.casefold().split()) not in gap_descriptions
    ]

    def serialize() -> str:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    serialized = serialize()
    while len(serialized) > max_chars and data["known_facts"]:
        data["known_facts"].pop(0)
        serialized = serialize()
    while len(serialized) > max_chars and data["attempted_queries"]:
        data["attempted_queries"].pop(0)
        serialized = serialize()
    while len(serialized) > max_chars and data["aliases"]:
        data["aliases"].pop()
        serialized = serialize()
    while len(serialized) > max_chars and data["unresolved_questions"]:
        data["unresolved_questions"].pop()
        serialized = serialize()
    while len(serialized) > max_chars and len(data["evidence_gaps"]) > 1:
        data["evidence_gaps"].pop()
        serialized = serialize()
    while len(serialized) > max_chars and data["next_queries"]:
        data["next_queries"].pop()
        serialized = serialize()
    if len(serialized) > max_chars:
        overflow = len(serialized) - max_chars
        goal = str(data.get("goal") or "")
        data["goal"] = goal[:max(80, len(goal) - overflow)]
        serialized = serialize()
    if len(serialized) > max_chars and data["evidence_gaps"]:
        gap = data["evidence_gaps"][0]
        gap["query_hint"] = None
        gap["description"] = str(gap.get("description") or "")[:120]
        serialized = serialize()
    if len(serialized) > max_chars:
        data = {
            "goal": str(data.get("goal") or "")[:80],
            "known_facts": [],
            "evidence_gaps": data.get("evidence_gaps", [])[:1],
            "unresolved_questions": [],
            "attempted_queries": [],
            "next_queries": [],
            "aliases": [],
            "complete": bool(data.get("complete")),
            "hop_count": int(data.get("hop_count") or 0),
            "stop_reason": data.get("stop_reason"),
        }
        serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return serialized
