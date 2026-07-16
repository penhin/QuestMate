"""Validation helpers for model-produced investigation and answer-judge state."""

import json
from collections.abc import Callable

from pydantic import ValidationError

from query_tokens import exact_identifiers
from schemas import (
    AnswerCompletenessAssessment,
    EvidenceFact,
    EvidenceGap,
    InvestigationState,
    PlannedSearchQuery,
)


EVIDENCE_GAP_KINDS = {
    "game_identity",
    "entity_identity",
    "premise",
    "direct_answer",
    "prerequisite",
    "acquisition",
    "access_route",
    "ordered_actions",
    "outcome",
    "version",
    "conflict",
    "semantic_distinction",
    "other",
}


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
        raw_indexes = raw_fact.get("source_indexes", [])
        raw_indexes = raw_indexes if isinstance(raw_indexes, list) else []
        indexes = [
            index for index in raw_indexes
            if isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= source_count
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
        source_type = str(raw_query.get("source_type") or "web")
        if source_type not in {"official", "wiki", "community", "web"}:
            source_type = "web"
        next_queries.append(PlannedSearchQuery(source_type=source_type, query=query))

    raw_gaps_value = data.get("evidence_gaps", [])
    gaps_container_valid = isinstance(raw_gaps_value, list)
    raw_gaps = raw_gaps_value if gaps_container_valid else []
    evidence_gaps: list[EvidenceGap] = []
    seen_gaps: set[tuple[str, str]] = set()
    for raw_gap in raw_gaps[:6]:
        if not isinstance(raw_gap, dict):
            continue
        description = str(raw_gap.get("description") or "").strip()[:300]
        if not description:
            continue
        kind = str(raw_gap.get("kind") or "other")
        kind = kind if kind in EVIDENCE_GAP_KINDS else "other"
        query_hint = sanitize_text(str(raw_gap.get("query_hint") or "")).strip()[:240] or None
        source_type = str(raw_gap.get("source_type") or "web")
        source_type = source_type if source_type in {"official", "wiki", "community", "web"} else "web"
        try:
            priority = min(5, max(1, int(raw_gap.get("priority") or 3)))
        except (TypeError, ValueError):
            priority = 3
        key = (kind, " ".join(description.casefold().split()))
        if key in seen_gaps:
            continue
        seen_gaps.add(key)
        evidence_gaps.append(
            EvidenceGap(
                kind=kind,
                description=description,
                query_hint=query_hint,
                source_type=source_type,
                priority=priority,
            )
        )
    evidence_gaps.sort(key=lambda gap: gap.priority, reverse=True)

    raw_unresolved = data.get("unresolved_questions", [])
    raw_unresolved = raw_unresolved if isinstance(raw_unresolved, list) else []
    unresolved = [
        str(value).strip()[:300]
        for value in raw_unresolved[:6]
        if str(value).strip()
    ]
    if not evidence_gaps:
        evidence_gaps = [
            # These compatibility gaps were not explicitly prioritized by the
            # model.  Keep the description usable as a fallback, but do not
            # let it displace a more specific model-produced query.
            EvidenceGap(kind="other", description=value)
            for value in unresolved
        ]
    if evidence_gaps:
        unresolved = list(dict.fromkeys([*unresolved, *(gap.description for gap in evidence_gaps)]))[:6]
    # Model JSON is untrusted even in JSON mode.  In particular, ``"false"`` is
    # truthy in Python and must never terminate an investigation.  Any declared
    # gap, including one whose shape could not be parsed, also keeps the state
    # open so malformed output fails closed.
    complete = (
        data.get("complete") is True
        and gaps_container_valid
        and not unresolved
        and not evidence_gaps
        and not raw_gaps
        and bool(known_facts)
    )
    if complete:
        next_queries = []
    raw_aliases = data.get("aliases", [])
    raw_aliases = raw_aliases if isinstance(raw_aliases, list) else []
    raw_aliases = [value for value in raw_aliases if isinstance(value, str)]
    state = InvestigationState(
        goal=str(data.get("goal") or previous.goal).strip()[:1000] or previous.goal,
        known_facts=known_facts,
        evidence_gaps=evidence_gaps,
        unresolved_questions=list(dict.fromkeys(unresolved)),
        attempted_queries=attempted,
        next_queries=next_queries,
        aliases=sanitize_aliases([*previous.aliases, *raw_aliases]),
        complete=complete,
        hop_count=previous.hop_count,
        stop_reason="complete" if complete else "needs_search" if next_queries else "insufficient_evidence",
    )
    return ensure_investigation_query(
        state,
        question=question,
        sanitize_text=sanitize_text,
        enforce_gap_priority=bool(raw_gaps),
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
    enforce_gap_priority: bool = True,
) -> InvestigationState:
    """Repair an incomplete state from its highest-priority typed evidence gap."""
    if state.complete:
        return state
    attempted = {" ".join(query.casefold().split()) for query in state.attempted_queries}
    identifiers = exact_identifiers(question)

    def planned_query(candidate: str, source_type: str) -> PlannedSearchQuery | None:
        base = sanitize_text(candidate).strip()[:190]
        if not base:
            return None
        missing_identifiers = [
            value for value in identifiers
            if value.casefold() not in base.casefold()
        ]
        query = f"{base} {' '.join(missing_identifiers)}"[:240].strip()
        if " ".join(query.casefold().split()) in attempted:
            return None
        return PlannedSearchQuery(source_type=source_type, query=query)

    ordered_gaps = sorted(state.evidence_gaps, key=lambda value: value.priority, reverse=True)
    highest_gap = ordered_gaps[0] if ordered_gaps else None
    # A typed query_hint is the investigation model's explicit search target.
    # Put the highest-priority one first even when the model also emitted a
    # lower-priority next_query.  This makes priority operational rather than
    # merely descriptive.
    if highest_gap is not None and (highest_gap.query_hint or enforce_gap_priority):
        priority_query = planned_query(
            highest_gap.query_hint or highest_gap.description,
            highest_gap.source_type,
        )
        if priority_query is not None:
            normalized_priority = " ".join(priority_query.query.casefold().split())
            remaining = [
                query for query in state.next_queries
                if " ".join(query.query.casefold().split()) != normalized_priority
            ]
            return state.model_copy(
                update={
                    "next_queries": [priority_query, *remaining][:2],
                    "stop_reason": "needs_search",
                }
            )

    if state.next_queries:
        return state
    gap_candidates = [
        (gap.query_hint or gap.description, gap.source_type)
        for gap in ordered_gaps
    ]
    candidates = [
        *gap_candidates,
        *((value, "web") for value in state.unresolved_questions),
        (question, "web"),
    ]
    for candidate, source_type in candidates:
        query = planned_query(candidate, source_type)
        if query is None:
            continue
        return state.model_copy(
            update={
                "next_queries": [query],
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
