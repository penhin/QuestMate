import json

from llm import GuideLLM
from ai.investigation import ensure_investigation_query
from retrieval.coordinator import RetrievalCoordinator
from schemas import ChatRequest, EvidenceFact, GameResolution, InvestigationState, PlannedSearchQuery, SearchPlan, Source


def test_investigation_parser_keeps_only_cited_facts_and_new_queries() -> None:
    previous = InvestigationState(
        goal="打开区域 B2 的 35 号门",
        attempted_queries=["B2 35 gate requirements"],
    )
    content = json.dumps(
        {
            "goal": previous.goal,
            "known_facts": [
                {"statement": "门需要中继令牌。", "source_indexes": [1]},
                {"statement": "无来源猜测。", "source_indexes": [9]},
            ],
            "unresolved_questions": ["如何获得中继令牌？"],
            "next_queries": [
                {"source_type": "wiki", "query": "relay token acquisition route"},
            ],
            "aliases": ["Relay Token"],
            "complete": False,
        }
    )

    state = GuideLLM._parse_investigation_state(
        content,
        previous=previous,
        question="如何打开区域 B2 的 35 号门？",
        source_count=1,
    )

    assert [fact.statement for fact in state.known_facts] == ["门需要中继令牌。"]
    assert state.next_queries[0].source_type == "wiki"
    assert "b2" in state.next_queries[0].query.casefold()
    assert "35" in state.next_queries[0].query
    assert state.stop_reason == "needs_search"


def test_investigation_parser_keeps_two_distinct_gap_queries() -> None:
    previous = InvestigationState(goal="Reach the objective")
    content = json.dumps(
        {
            "goal": previous.goal,
            "known_facts": [{"statement": "A key and passage are required.", "source_indexes": [1]}],
            "unresolved_questions": ["Where is the key?", "How is the passage opened?"],
            "next_queries": [
                {"source_type": "wiki", "query": "required key location"},
                {"source_type": "wiki", "query": "hidden passage access"},
            ],
            "aliases": [],
            "complete": False,
        }
    )

    state = GuideLLM._parse_investigation_state(
        content,
        previous=previous,
        question="How do I reach the objective?",
        source_count=1,
    )

    assert [query.query for query in state.next_queries] == ["required key location", "hidden passage access"]


async def test_investigation_follows_dependencies_until_path_is_complete() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class DependencySearch:
        def __init__(self):
            self.calls = 0

        async def search(self, query: str, game: str, **kwargs):
            self.calls += 1
            evidence = (
                "The gate requires a relay token."
                if self.calls == 1
                else "The relay token is in the maintenance passage."
                if self.calls == 2
                else "Restore auxiliary power to enter the maintenance passage."
            )
            return [
                Source(
                    title=f"Dependency {self.calls}",
                    url=f"https://example.com/dependency-{self.calls}",
                    evidence=evidence,
                )
            ]

    class Investigator:
        def __init__(self):
            self.calls = 0

        async def update_investigation(self, *, investigation, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return investigation.model_copy(
                    update={
                        "known_facts": [EvidenceFact(statement="The gate requires a relay token.", source_indexes=[1])],
                        "unresolved_questions": ["Where is the relay token?"],
                        "next_queries": [PlannedSearchQuery(source_type="wiki", query="relay token route")],
                        "aliases": ["relay token"],
                        "stop_reason": "needs_search",
                    }
                )
            if self.calls == 2:
                return investigation.model_copy(
                    update={
                        "unresolved_questions": ["How is the maintenance passage opened?"],
                        "next_queries": [PlannedSearchQuery(source_type="wiki", query="maintenance passage access")],
                        "aliases": ["relay token", "maintenance passage"],
                        "stop_reason": "needs_search",
                    }
                )
            return investigation.model_copy(
                update={
                    "unresolved_questions": [],
                    "next_queries": [],
                    "complete": True,
                    "stop_reason": "complete",
                }
            )

    search = DependencySearch()
    investigator = Investigator()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(),
        search_provider=search,
        llm=investigator,
        max_results=8,
        max_hops=2,
    )
    request = ChatRequest(game="Unseen Niche Game", question="How do I open gate ZX-35?")
    resolution = GameResolution(
        input_name=request.game,
        confirmed_name=request.game,
        aliases=[request.game],
        confidence=0.9,
    )

    outcome = await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(
            intent="game_mechanic",
            queries=[{"source_type": "wiki", "query": "ZX-35 gate requirements"}],
        ),
        game_resolution=resolution,
    )

    assert search.calls == 3
    assert investigator.calls == 3
    assert outcome.investigation.complete is True
    assert outcome.investigation.hop_count == 2
    assert outcome.investigation.stop_reason == "complete"
    assert {source.title for source in outcome.sources} == {"Dependency 1", "Dependency 2", "Dependency 3"}


async def test_direct_single_entity_evidence_skips_expensive_investigation() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class Search:
        async def search(self, query: str, game: str, **kwargs):
            return [
                Source(
                    title="Moonstone acquisition",
                    url="https://example.com/moonstone",
                    evidence="Moonstone is acquired from the observatory chest.",
                    source_type="wiki",
                )
            ]

    class Investigator:
        def __init__(self):
            self.calls = 0

        async def update_investigation(self, **kwargs):
            self.calls += 1
            raise AssertionError("direct, single-entity evidence should use the fast path")

    investigator = Investigator()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(), search_provider=Search(), llm=investigator, max_results=5
    )
    request = ChatRequest(game="Example Game", question="Where is Moonstone acquired?")

    outcome = await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(intent="item_location", aliases=["Moonstone"]),
        game_resolution=GameResolution(
            input_name=request.game, confirmed_name=request.game, confidence=1
        ),
    )

    assert investigator.calls == 0
    assert len(outcome.sources) == 1
    assert outcome.investigation.hop_count == 0


async def test_relation_verification_plan_uses_one_bounded_investigation_check() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class Search:
        async def search(self, query: str, game: str, **kwargs):
            return [Source(
                title="Amber Relay guide",
                url="https://example.com/amber-relay",
                evidence="The Amber Relay is found in the old tower.",
                source_type="wiki",
            )]

    class Investigator:
        def __init__(self):
            self.calls = 0

        async def update_investigation(self, **kwargs):
            self.calls += 1
            return kwargs["investigation"].model_copy(update={
                "complete": False,
                "next_queries": [],
                "stop_reason": "insufficient_evidence",
            })

    investigator = Investigator()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(), search_provider=Search(), llm=investigator, max_results=5
    )
    request = ChatRequest(game="Example Game", question="Does the Amber Relay open the Blue Gate?")

    outcome = await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(
            intent="general",
            requires_relation_verification=True,
            named_entity_groups=[["Amber Relay"], ["Blue Gate"]],
        ),
        game_resolution=GameResolution(
            input_name=request.game, confirmed_name=request.game, confidence=1
        ),
    )

    assert investigator.calls == 1
    assert outcome.investigation.stop_reason == "insufficient_evidence"


async def test_refinement_handoff_bounds_model_generated_gap_lists_to_plan_schema() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class Search:
        def __init__(self) -> None:
            self.refinement_missing_info: list[str] | None = None

        async def search(self, query: str, game: str, **kwargs):
            plan = kwargs.get("plan")
            if plan and plan.refinement:
                self.refinement_missing_info = plan.missing_info
                return [Source(
                    title="Refined relay route",
                    url="https://example.com/refined-relay",
                    evidence="The Amber Relay opens the Blue Gate.",
                )]
            return [Source(
                title="Initial relay route",
                url="https://example.com/initial-relay",
                evidence="The Amber Relay is in the archive.",
            )]

    class Investigator:
        async def update_investigation(self, *, investigation, **_kwargs):
            return investigation.model_copy(update={
                "unresolved_questions": [f"gap {index}" for index in range(6)],
                "next_queries": [PlannedSearchQuery(source_type="wiki", query="Amber Relay Blue Gate")],
                "stop_reason": "needs_search",
            })

    search = Search()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(), search_provider=search, llm=Investigator(), max_results=5
    )
    request = ChatRequest(game="Synthetic Adventure", question="Does the Amber Relay open the Blue Gate?")

    await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(
            intent="general",
            requires_relation_verification=True,
            named_entity_groups=[["Amber Relay"], ["Blue Gate"]],
        ),
        game_resolution=GameResolution(
            input_name=request.game, confirmed_name=request.game, confidence=1
        ),
    )

    assert search.refinement_missing_info == ["gap 0", "gap 1", "gap 2", "gap 3"]


async def test_answer_completeness_judge_reports_missing_access_step() -> None:
    class JudgeProvider:
        async def complete(self, **kwargs):
            return json.dumps(
                {
                    "complete": False,
                    "gaps": ["没有说明如何进入隐藏区域"],
                    "unsupported_claims": [],
                }
            )

    llm = GuideLLM(provider=JudgeProvider())
    request = ChatRequest(game="Unseen Niche Game", question="How do I obtain the key?")
    investigation = InvestigationState(
        goal=request.question,
        unresolved_questions=["How is the hidden area reached?"],
        stop_reason="insufficient_evidence",
    )
    assessment = await llm.assess_answer_completeness(
        request=request,
        sources=[
            Source(
                title="Key location",
                url="https://example.com/key",
                evidence="The key is inside a hidden area.",
            )
        ],
        answer="钥匙在隐藏区域。[1]",
        plan=SearchPlan(intent="item_location"),
        investigation=investigation,
    )

    assert assessment.complete is False
    assert assessment.gaps == ["没有说明如何进入隐藏区域"]


def test_answer_completeness_rejects_sourced_but_irrelevant_details() -> None:
    assessment = GuideLLM._parse_answer_completeness(
        json.dumps(
            {
                "complete": True,
                "gaps": [],
                "unsupported_claims": [],
                "irrelevant_details": ["与目标无关的房间战利品"],
            }
        )
    )

    assert assessment.complete is False
    assert assessment.irrelevant_details == ["与目标无关的房间战利品"]


def test_incomplete_action_state_gets_generic_query_from_its_own_gap() -> None:
    state = InvestigationState(
        goal="Open gate ZX-35",
        unresolved_questions=["Where is the relay token?"],
        attempted_queries=["gate ZX-35 requirements"],
        stop_reason="insufficient_evidence",
    )

    repaired = ensure_investigation_query(
        state,
        question="How do I open gate ZX-35?",
        sanitize_text=lambda value: value,
    )

    assert len(repaired.next_queries) == 1
    assert "relay token" in repaired.next_queries[0].query.casefold()
    assert "zx-35" in repaired.next_queries[0].query.casefold()
    assert repaired.stop_reason == "needs_search"


async def test_investigation_stops_early_when_initial_evidence_is_not_direct() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class RepeatingSearch:
        def __init__(self):
            self.calls = 0

        async def search(self, query: str, game: str, **kwargs):
            self.calls += 1
            return [
                Source(
                    title="Same rule page",
                    url="https://example.com/rule",
                    evidence="The same unchanged evidence.",
                )
            ]

    class PersistentInvestigator:
        def __init__(self):
            self.calls = 0

        async def update_investigation(self, *, investigation, **kwargs):
            self.calls += 1
            return investigation.model_copy(
                update={
                    "unresolved_questions": ["Need another detail"],
                    "next_queries": [PlannedSearchQuery(source_type="wiki", query=f"detail attempt {self.calls}")],
                    "stop_reason": "needs_search",
                }
            )

    search = RepeatingSearch()
    investigator = PersistentInvestigator()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(),
        search_provider=search,
        llm=investigator,
        max_results=5,
        max_hops=2,
    )
    request = ChatRequest(game="Unseen Game", question="How does this rule work?")
    outcome = await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(queries=[{"source_type": "web", "query": request.question}]),
        game_resolution=GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=[request.game],
            confidence=0.9,
        ),
    )

    assert search.calls == 1
    assert investigator.calls == 0
    assert outcome.refined is False
    assert outcome.investigation.stop_reason is None


async def test_wiki_domain_discovered_from_evidence_is_used_for_refinement() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class DomainCapturingSearch:
        def __init__(self):
            self.domains = []

        async def search(self, query: str, game: str, *, game_resolution, **kwargs):
            self.domains.append(list(game_resolution.database_domains))
            call = len(self.domains)
            return [
                Source(
                    title=f"Wiki dependency {call}",
                    url=f"https://unseen-game.example/wiki/dependency-{call}",
                    source_type="wiki",
                    evidence=f"Dependency evidence {call}",
                )
            ]

    class OneHopInvestigator:
        def __init__(self):
            self.calls = 0

        async def update_investigation(self, *, investigation, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return investigation.model_copy(
                    update={
                        "unresolved_questions": ["Find prerequisite"],
                        "next_queries": [PlannedSearchQuery(source_type="wiki", query="prerequisite route")],
                        "stop_reason": "needs_search",
                    }
                )
            return investigation.model_copy(
                update={"complete": True, "unresolved_questions": [], "next_queries": [], "stop_reason": "complete"}
            )

    search = DomainCapturingSearch()
    coordinator = RetrievalCoordinator(
        knowledge=EmptyKnowledge(),
        search_provider=search,
        llm=OneHopInvestigator(),
        max_results=5,
        max_hops=2,
    )
    request = ChatRequest(game="Unseen Game", question="How do I reach the hidden route?")
    await coordinator.investigate(
        request=request,
        history=[],
        plan=SearchPlan(intent="quest_step", queries=[{"source_type": "wiki", "query": request.question}]),
        game_resolution=GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=[request.game],
            confidence=0.9,
        ),
    )

    assert search.domains == [[], ["unseen-game.example"]]
