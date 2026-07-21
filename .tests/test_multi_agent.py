from agent import QuestAgent
from multi_agent import AnswerAgent, EvidenceAgent
from schemas import ChatRequest, SearchPlan


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

    assert {"identity_agent", "planning_agent", "retrieval_evidence_agents", "answer_agent"} <= set(graph.nodes)


async def test_evidence_agent_falls_back_to_legacy_refinement_contract() -> None:
    class LegacyLLM:
        async def refine_search_plan(self, *, request, plan, sources, history, game_resolution=None):
            return plan.model_copy(update={"refinement": True})

    evidence = EvidenceAgent(LegacyLLM())
    request = ChatRequest(game="Example Adventure", question="Where is the relay?")
    refined = await evidence.refine_search_plan(
        request=request,
        plan=SearchPlan(),
        sources=[],
        history=[],
        game_resolution=None,
        investigation="artifact not accepted by legacy contract",
    )

    assert evidence.supports_update_investigation is False
    assert refined is not None and refined.refinement is True


async def test_answer_agent_filters_optional_artifacts_for_legacy_signature() -> None:
    class LegacyLLM:
        async def answer(self, *, request, sources, plan=None):
            return f"{request.game}:{len(sources)}:{plan.intent if plan else 'none'}"

    answer = await AnswerAgent(LegacyLLM()).answer(
        request=ChatRequest(game="Example Adventure", question="Where is the relay?"),
        sources=[],
        plan=SearchPlan(intent="item_location"),
        investigation="new cross-agent artifact",
    )

    assert answer == "Example Adventure:0:item_location"
