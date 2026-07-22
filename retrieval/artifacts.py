"""Typed values exchanged between retrieval stages and orchestration."""

from dataclasses import dataclass

from retrieval.pipeline import RetrievalStage
from schemas import InvestigationState, SearchPlan, Source


@dataclass(frozen=True)
class RetrievalBatch:
    """One complete recall-to-selection pass, before answer generation."""

    sources: list[Source]
    stages: list[RetrievalStage]


@dataclass(frozen=True)
class RetrievalOutcome:
    """The selected evidence, updated plan, and bounded investigation result."""

    sources: list[Source]
    plan: SearchPlan
    investigation: InvestigationState
    refined: bool
    stages: list[RetrievalStage]
