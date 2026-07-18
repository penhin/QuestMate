"""Vocabulary-free fallback planning.

The normal planner is responsible for intent and evidence requirements.  When
it is unavailable, preserve the user's wording rather than guessing an intent
from a maintained action-word list.
"""

import re

from quality_policy import is_version_sensitive_question
from schemas import PlannedSearchQuery, SearchPlan


def fallback_search_subject(question: str) -> str:
    """Return an explicitly delimited entity, otherwise the complete question."""
    compact = " ".join(question.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})).split())
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", compact)
    if quoted:
        return max((" ".join(value.split()).strip() for value in quoted), key=len)
    return compact.strip("。！？?! ") or question


def fallback_search_plan(*, question: str) -> SearchPlan:
    """Use a bounded, source-diverse copy of the original user query."""
    raw_question = " ".join(question.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})).split()).strip()
    subject = fallback_search_subject(raw_question)
    queries = [
        PlannedSearchQuery(source_type=source_type, query=raw_question[:240] or subject[:240])
        for source_type in ("wiki", "community", "web")
    ]
    return SearchPlan(
        intent="general",
        version_sensitive=is_version_sensitive_question(question),
        queries=queries,
    )


def is_short_followup(question: str) -> bool:
    """Use only message shape for history expansion; never infer its intent."""
    return False
