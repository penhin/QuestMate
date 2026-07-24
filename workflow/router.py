"""Route typed search plans to bounded execution workflows."""

from enum import StrEnum
from typing import Any

from schemas import SearchPlan
from router import RouteDecision


class WorkflowKind(StrEnum):
    EVIDENCE_RESEARCH = "evidence_research"
    VERIFIED_RESEARCH = "verified_research"


class WorkflowRouter:
    """Select a reasoning chain from planner artifacts, never raw prompt text.

    The normal path writes after bounded evidence retrieval. Version-sensitive
    and relation-heavy questions add a deterministic verification checkpoint
    before answer generation, so one agent can execute distinct workflows
    while retaining a single public request API.
    """

    @staticmethod
    def classify(plan: SearchPlan) -> WorkflowKind:
        if (
            plan.version_sensitive
            or plan.requires_relation_verification
            or len(plan.named_entity_groups) >= 2
        ):
            return WorkflowKind.VERIFIED_RESEARCH
        return WorkflowKind.EVIDENCE_RESEARCH

    def next_after_research(self, state: dict[str, Any]) -> str:
        plan = state["search_plan"]
        return "verification" if self.classify(plan) is WorkflowKind.VERIFIED_RESEARCH else "writer"

    @staticmethod
    def task_workflow(route: RouteDecision) -> str:
        """Compatibility bridge until task graphs replace the shared graph."""
        return route.intent
