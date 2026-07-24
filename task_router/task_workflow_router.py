"""Map validated planning artifacts onto QuestMate task workflows."""

from schemas import GameResolution, SearchPlan
from task_router.schema import TaskRouteConstraints, TaskRouteDecision, WorkflowName


class TaskWorkflowRouter:
    _GUIDE_INTENTS = frozenset({"quest_step", "item_location", "item_usage"})

    def route(self, *, plan: SearchPlan, game_resolution: GameResolution) -> TaskRouteDecision:
        workflow = self.workflow_for(plan)
        entities = list(dict.fromkeys(alias.strip() for group in plan.named_entity_groups for alias in group if alias.strip()))[:16]
        return TaskRouteDecision(
            workflow=workflow,
            confidence=self._confidence(plan),
            game=game_resolution.confirmed_name or game_resolution.input_name,
            entities=entities,
            constraints=TaskRouteConstraints(
                version_sensitive=plan.version_sensitive,
                requires_relation_verification=plan.requires_relation_verification,
                safety_refusal=plan.safety_refusal,
                answer_requirements=plan.answer_requirements,
                missing_info=plan.missing_info,
            ),
            search_intent=plan.intent,
        )

    @classmethod
    def workflow_for(cls, plan: SearchPlan) -> WorkflowName:
        if plan.intent == "build":
            return "build"
        if plan.intent in cls._GUIDE_INTENTS:
            return "guide"
        return "analysis"

    @staticmethod
    def _confidence(plan: SearchPlan) -> float:
        if plan.safety_refusal:
            return 1.0
        return 0.6 if plan.intent == "general" else 0.9
