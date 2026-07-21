from types import SimpleNamespace

from agent import QuestAgent
from retrieval.pipeline import RetrievalStage, fuse_and_rank_evidence
from retrieval.coordinator import RetrievalCoordinator
from schemas import ChatRequest, GameResolution, PlannedSearchQuery, SearchPlan, Source
from schemas import Source


def test_pipeline_keeps_direct_passage_when_duplicate_url_has_broad_local_chunk() -> None:
    pool = fuse_and_rank_evidence(
        groups={
            "knowledge": [
                Source(
                    title="Moonstone guide",
                    url="https://guides.example/moonstone",
                    evidence="Moonstone is a useful crafting material used in several upgrades.",
                    score=0.98,
                    trust_score=0.9,
                )
            ],
            "web": [
                Source(
                    title="Moonstone guide",
                    url="https://guides.example/moonstone",
                    evidence="Moonstone is acquired from the observatory chest after the bridge puzzle.",
                    score=0.55,
                    trust_score=0.45,
                )
            ],
        },
        query="Where is Moonstone acquired?",
        intent="item_location",
        max_results=4,
    )

    assert pool.candidate_count == 2
    assert pool.fused_count == 1
    assert pool.channels == {"knowledge": 1, "web": 1}
    assert len(pool.sources) == 1
    assert "observatory chest" in (pool.sources[0].evidence or "")


def test_pipeline_ranks_direct_page_above_broader_higher_trust_page() -> None:
    pool = fuse_and_rank_evidence(
        groups={
            "knowledge": [
                Source(
                    title="Moonstone overview",
                    url="https://guides.example/overview",
                    evidence="Moonstone is a crafting material in the game.",
                    score=0.99,
                    trust_score=0.95,
                )
            ],
            "web": [
                Source(
                    title="Observatory route",
                    url="https://independent.example/route",
                    evidence="Moonstone is acquired from the observatory chest.",
                    score=0.5,
                    trust_score=0.45,
                )
            ],
        },
        query="Where is Moonstone acquired?",
        intent="item_location",
        max_results=2,
    )

    assert [source.title for source in pool.sources] == ["Observatory route", "Moonstone overview"]


async def test_investigation_records_stages_and_reuses_pipeline_for_refinement() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class Search:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str, game: str, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [Source(
                    title="Quartz Relay overview",
                    url="https://guide.example/relay",
                    evidence="Quartz Relay is an item used in the archive.",
                )]
            return [Source(
                title="Quartz Relay route",
                url="https://guide.example/relay",
                evidence="Quartz Relay opens the Azure Gate after the signal puzzle.",
            )]

    class Refiner:
        async def refine_search_plan(self, **kwargs):
            return SearchPlan(
                intent="game_mechanic",
                aliases=["Quartz Relay", "Azure Gate"],
                queries=[PlannedSearchQuery(source_type="wiki", query="Quartz Relay Azure Gate signal puzzle")],
                refinement=True,
            )

    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(), search_provider=Search(), llm=Refiner(), max_results=4, max_hops=1
    )
    request = ChatRequest(
        game="Example Adventure",
        question="Does Quartz Relay open Azure Gate after the signal puzzle?",
    )
    outcome = await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(
            intent="game_mechanic",
            requires_relation_verification=True,
            aliases=["Quartz Relay", "Azure Gate"],
            queries=[PlannedSearchQuery(source_type="wiki", query="Quartz Relay Azure Gate")],
        ),
        game_resolution=GameResolution(input_name=request.game, confirmed_name=request.game, confidence=1),
    )

    names = [stage.name for stage in outcome.stages]
    assert names[:4] == ["candidate_build", "passage_fusion", "rerank_and_select", "evidence_assessment"]
    assert names[-1] == "adaptive_decision"
    assert names.count("adaptive_decision") == 1
    assert names.count("adaptive_refinement") == 1
    assert names.count("candidate_build") == 3  # initial, refinement, accumulated merge
    assert outcome.refined is True
    assert "opens the Azure Gate" in (outcome.sources[0].evidence or "")


async def test_agent_passes_initial_pipeline_stages_to_current_coordinator() -> None:
    source = Source(
        title="Example Adventure Moonstone guide",
        url="https://guides.example/moonstone",
        evidence="Moonstone is acquired from the observatory chest.",
    )
    expected_stages = [RetrievalStage("candidate_build", 2, 1)]

    class Retrieval:
        async def retrieve_batch(self, *_args, **_kwargs):
            return SimpleNamespace(sources=[source], stages=expected_stages)

        async def investigate(self, **kwargs):
            assert kwargs["initial_stages"] == expected_stages
            return SimpleNamespace(sources=kwargs["initial_sources"], plan=kwargs["plan"])

    class Provider:
        async def resolve_game(self, *_args, **_kwargs):
            raise AssertionError("direct title-bearing evidence should not trigger recovery")

    agent = object.__new__(QuestAgent)
    agent.retrieval = Retrieval()
    agent.search_provider = Provider()
    request = ChatRequest(game="Example Adventure", question="Where is Moonstone acquired?")
    outcome, resolution = await agent._retrieve_after_identity_check(
        request=request,
        history=[],
        plan=SearchPlan(intent="item_location"),
        game_resolution=GameResolution(
            input_name=request.game, confirmed_name=request.game, confidence=1
        ),
    )

    assert outcome.sources == [source]
    assert resolution.is_confirmed
