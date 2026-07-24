from agent import QuestAgent
from schemas import ChatRequest, GameResolution, SearchPlan
from task_router import TaskWorkflowRouter
from workflows.verification import EvidencePath, EvidenceVerificationRouter


def test_orchestrator_graph_has_bounded_specialist_handoffs() -> None:
    class Search:
        async def search(self, *_args, **_kwargs):
            return []

        async def resolve_game(self, game, **_kwargs):
            raise AssertionError(f"identity recovery should not run in this construction test: {game}")

    class LLM:
        async def plan_search(self, **_kwargs):
            return SearchPlan()

        async def answer(self, **_kwargs):
            return "answer"

    graph = QuestAgent(search_provider=Search(), llm=LLM()).graph.get_graph()

    assert {
        "identity_agent", "planning_agent", "retrieval_evidence_agents",
        "task_router", "guide_workflow", "build_workflow", "analysis_workflow",
        "answer_agent",
    } <= set(graph.nodes)


def test_evidence_verification_router_adds_checkpoint_only_for_complex_paths() -> None:
    router = EvidenceVerificationRouter()

    assert router.classify(SearchPlan(intent="item_location")) is EvidencePath.RESEARCH
    assert router.classify(SearchPlan(intent="patch", version_sensitive=True)) is EvidencePath.VERIFIED_RESEARCH
    assert router.classify(SearchPlan(
        intent="general", named_entity_groups=[["A"], ["B"]],
    )) is EvidencePath.VERIFIED_RESEARCH


def test_task_router_emits_typed_task_workflow_decision() -> None:
    decision = TaskWorkflowRouter().route(
        plan=SearchPlan(
            intent="build",
            version_sensitive=True,
            named_entity_groups=[["Moonblade", "月刃"]],
            answer_requirements=["recommend a build"],
        ),
        game_resolution=GameResolution(
            input_name="Example Adventure", confirmed_name="Example Adventure"
        ),
    )

    assert decision.model_dump() == {
        "workflow": "build",
        "confidence": 0.9,
        "game": "Example Adventure",
        "entities": ["Moonblade", "月刃"],
        "constraints": {
            "version_sensitive": True,
            "requires_relation_verification": False,
            "safety_refusal": False,
            "answer_requirements": ["recommend a build"],
            "missing_info": [],
        },
        "search_intent": "build",
    }


def test_task_router_maps_guide_and_analysis_intents_without_raw_question_rules() -> None:
    router = TaskWorkflowRouter()
    game = GameResolution(input_name="Example Adventure")

    assert router.route(plan=SearchPlan(intent="quest_step"), game_resolution=game).workflow == "guide"
    assert router.route(plan=SearchPlan(intent="game_mechanic"), game_resolution=game).workflow == "analysis"
