"""Validation helpers for model-produced investigation and answer-judge state."""

import json
from collections.abc import Callable

from pydantic import ValidationError

from query_tokens import exact_identifiers
from schemas import AnswerCompletenessAssessment, EvidenceFact, InvestigationState, PlannedSearchQuery


def parse_investigation_state(
    content: str,
    *,
    previous: InvestigationState,
    question: str,
    source_count: int,
    sanitize_text: Callable[[str], str],
    sanitize_aliases: Callable[[list[str]], list[str]],
) -> InvestigationState:
    try:
        data = _json_object(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return previous.model_copy(update={"next_queries": [], "stop_reason": "insufficient_evidence"})

    raw_facts = data.get("known_facts", [])
    raw_facts = raw_facts if isinstance(raw_facts, list) else []
    known_facts: list[EvidenceFact] = []
    for raw_fact in raw_facts[:12]:
        if not isinstance(raw_fact, dict):
            continue
        statement = str(raw_fact.get("statement") or "").strip()[:500]
        indexes = [
            index for index in raw_fact.get("source_indexes", [])
            if isinstance(index, int) and 1 <= index <= source_count
        ][:6]
        if statement and indexes:
            known_facts.append(EvidenceFact(statement=statement, source_indexes=list(dict.fromkeys(indexes))))

    attempted = list(dict.fromkeys(previous.attempted_queries))[:16]
    normalized_attempts = {" ".join(query.casefold().split()) for query in attempted}
    raw_queries = data.get("next_queries", [])
    raw_queries = raw_queries if isinstance(raw_queries, list) else []
    next_queries: list[PlannedSearchQuery] = []
    for raw_query in raw_queries[:2]:
        if not isinstance(raw_query, dict):
            continue
        query = sanitize_text(str(raw_query.get("query") or ""))[:240].strip()
        if not query:
            continue
        missing_identifiers = [
            identifier for identifier in exact_identifiers(question)
            if identifier.casefold() not in query.casefold()
        ]
        if missing_identifiers:
            query = f"{query} {' '.join(missing_identifiers)}"[:240].strip()
        if " ".join(query.casefold().split()) in normalized_attempts:
            continue
        source_type = raw_query.get("source_type", "web")
        if source_type not in {"official", "wiki", "community", "web"}:
            source_type = "web"
        next_queries.append(PlannedSearchQuery(source_type=source_type, query=query))

    raw_unresolved = data.get("unresolved_questions", [])
    raw_unresolved = raw_unresolved if isinstance(raw_unresolved, list) else []
    unresolved = [
        str(value).strip()[:300]
        for value in raw_unresolved[:6]
        if str(value).strip()
    ]
    complete = bool(data.get("complete")) and not unresolved and bool(known_facts)
    if complete:
        next_queries = []
    raw_aliases = data.get("aliases", [])
    raw_aliases = raw_aliases if isinstance(raw_aliases, list) else []
    return InvestigationState(
        goal=str(data.get("goal") or previous.goal).strip()[:1000] or previous.goal,
        known_facts=known_facts,
        unresolved_questions=list(dict.fromkeys(unresolved)),
        attempted_queries=attempted,
        next_queries=next_queries,
        aliases=sanitize_aliases([*previous.aliases, *raw_aliases]),
        complete=complete,
        hop_count=previous.hop_count,
        stop_reason="complete" if complete else "needs_search" if next_queries else "insufficient_evidence",
    )


def parse_answer_completeness(content: str) -> AnswerCompletenessAssessment:
    try:
        assessment = AnswerCompletenessAssessment.model_validate(_json_object(content))
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
        return AnswerCompletenessAssessment(
            complete=False,
            gaps=["无法可靠完成最终答案完整性检查"],
        )
    if assessment.gaps or assessment.unsupported_claims or assessment.irrelevant_details:
        return assessment.model_copy(update={"complete": False})
    return assessment


def ensure_investigation_query(
    state: InvestigationState,
    *,
    question: str,
    sanitize_text: Callable[[str], str],
) -> InvestigationState:
    """Repair an incomplete actionable state when the model omitted its next query."""
    if state.complete or state.next_queries:
        return state
    attempted = {" ".join(query.casefold().split()) for query in state.attempted_queries}
    identifiers = exact_identifiers(question)
    candidates = [
        *state.unresolved_questions,
        *(fact.statement for fact in reversed(state.known_facts)),
        question,
    ]
    for candidate in candidates:
        base = sanitize_text(candidate).strip()[:190]
        if not base:
            continue
        missing_identifiers = [value for value in identifiers if value not in base.casefold()]
        query = f"{base} prerequisite access route {' '.join(missing_identifiers)}"[:240].strip()
        if " ".join(query.casefold().split()) in attempted:
            continue
        return state.model_copy(
            update={
                "next_queries": [PlannedSearchQuery(source_type="wiki", query=query)],
                "stop_reason": "needs_search",
            }
        )
    return state


def _json_object(content: str) -> dict:
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    value = json.loads(content[start:end])
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value
