"""Validation and normalization for initial and refinement search plans."""

import json
from collections.abc import Callable

import structlog
from pydantic import ValidationError

from ai.search_plan_json import coerce_search_plan_data, first_json_object
from ai.search_plan_sanitization import (
    sanitize_aliases,
    sanitize_answer_requirements,
    sanitize_named_entity_groups,
    sanitize_search_text,
)
from quality_policy import is_version_sensitive_question
from schemas import PlannedSearchQuery, SearchPlan


logger = structlog.get_logger()


def parse_search_plan(
    content: str, *, fallback_question: str, fallback_plan: Callable[..., SearchPlan],
    fallback_subject: Callable[[str], str],
) -> SearchPlan:
    try:
        plan = SearchPlan.model_validate(coerce_search_plan_data(first_json_object(content)))
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        error_fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()][:4] if isinstance(exc, ValidationError) else []
        logger.warning("llm.search_plan_parse_failed", error_type=type(exc).__name__, error_fields=error_fields)
        return fallback_plan(question=fallback_question)
    if plan.safety_refusal:
        return SearchPlan(safety_refusal=True)
    if not plan.queries:
        return fallback_plan(question=fallback_question)
    sanitized_queries = [
        PlannedSearchQuery(source_type=query.source_type, query=sanitized)
        for query in plan.queries[:4]
        if (sanitized := sanitize_search_text(query.query))
    ]
    if not sanitized_queries:
        return fallback_plan(question=fallback_question)
    intent = plan.intent or "general"
    if intent in {"item_location", "item_usage", "quest_step", "game_mechanic"} and not any(
        query.source_type == "web" for query in sanitized_queries
    ):
        web_query = PlannedSearchQuery(source_type="web", query=fallback_subject(sanitize_search_text(fallback_question)))
        if len(sanitized_queries) >= 4:
            sanitized_queries[-1] = web_query
        else:
            sanitized_queries.append(web_query)
    aliases = sanitize_aliases(plan.aliases)
    return SearchPlan(
        intent=intent,
        safety_refusal=False,
        version_sensitive=plan.version_sensitive or is_version_sensitive_question(fallback_question),
        requires_relation_verification=plan.requires_relation_verification,
        named_entity_groups=sanitize_named_entity_groups(
            plan.named_entity_groups, question=fallback_question, aliases=aliases,
            queries=[query.query for query in sanitized_queries],
        ),
        aliases=aliases,
        queries=sanitized_queries,
        answer_requirements=sanitize_answer_requirements(plan.answer_requirements),
        missing_info=[value.strip() for value in plan.missing_info if value.strip()][:4],
    )
