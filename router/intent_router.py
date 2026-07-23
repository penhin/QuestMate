"""Map validated planning artifacts onto QuestMate task workflows."""

from schemas import GameResolution, SearchPlan

from router.schema import RouteConstraints, RouteDecision, WorkflowName


class IntentRouter:
    """A deterministic router; it never inspects raw user prompt text.

    ``SearchPlan`` remains the single LLM-produced planning artifact.  This
    layer converts its validated intent into the coarser task workflow needed
    by the executor, so routing cannot introduce an extra model call or alter
    retrieval semantics.
    """

    _GUIDE_INTENTS = frozenset({"quest_step", "item_location", "item_usage"})

    def route(self, *, plan: SearchPlan, game_resolution: GameResolution) -> RouteDecision:
        workflow = self.workflow_for(plan)
        entities = list(dict.fromkeys(
            alias.strip()
            for group in plan.named_entity_groups
            for alias in group
            if alias.strip()
        ))[:16]
        return RouteDecision(
            intent=workflow,
            confidence=self._confidence(plan),
            game=game_resolution.confirmed_name or game_resolution.input_name,
            entities=entities,
            constraints=RouteConstraints(
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
        """Report decision confidence, not fabricated model-classifier odds."""
        if plan.safety_refusal:
            return 1.0
        if plan.intent == "general":
            return 0.6
        return 0.9
