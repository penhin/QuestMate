import pytest

from ai.fallback_planning import fallback_search_plan
from config import Settings
from quality_policy import MAX_QUERIES_PER_PLANNED_QUERY, MAX_SEARCH_QUERIES, SOURCE_POLICIES
from retrieval.coordinator import merge_search_plans
from retrieval.query_builder import build_search_queries
from schemas import GameResolution, SearchPlan
from search import TavilySearchProvider


def test_query_portfolio_pairs_known_database_with_open_original_question() -> None:
    question = "如何进入35号房，并确认 v2.01 的前置条件？"
    plan = SearchPlan(
        intent="game_mechanic",
        aliases=["Room 35"],
        queries=[
            {
                "source_type": "wiki",
                "query": "Room 35 access prerequisite",
            }
        ],
    )

    queries = build_search_queries(
        game="冷门游戏",
        question=question,
        plan=plan,
        sources=SOURCE_POLICIES,
        database_domains=("niche.example.org",),
        game_aliases=("Niche Game",),
    )

    assert len(queries) == MAX_QUERIES_PER_PLANNED_QUERY
    assert queries[0][0].startswith("site:niche.example.org 冷门游戏")
    open_query, open_policy = queries[1]
    assert "site:" not in open_query
    assert open_policy.source_type == "web"
    assert "Niche Game" in open_query
    assert "Room 35" in open_query
    assert question in open_query
    assert '"35"' in open_query
    assert '"v2.01"' in open_query


def test_configured_source_domains_do_not_exclude_open_web_search() -> None:
    plan = SearchPlan(
        intent="general",
        aliases=["Azure Relay"],
        queries=[{"source_type": "community", "query": "Azure Relay signal behavior"}],
    )

    queries = build_search_queries(
        game="Unfamiliar Game",
        question="What happens when the Azure Relay loses its signal?",
        plan=plan,
        sources=SOURCE_POLICIES,
    )

    assert queries[0][0].startswith("site:reddit.com")
    assert "site:" not in queries[1][0]
    assert "What happens when" in queries[1][0]


def test_default_first_wave_interleaves_independent_planned_semantics() -> None:
    plan = SearchPlan(
        intent="item_location",
        aliases=["Amber Relay"],
        queries=[
            {"source_type": "wiki", "query": "Amber Relay recovery after lost"},
            {"source_type": "web", "query": "Amber Relay first acquisition source"},
            {"source_type": "community", "query": "Amber Relay bug workaround"},
        ],
    )

    queries = build_search_queries(
        game="Unfamiliar Game",
        question="How is the Amber Relay first acquired?",
        plan=plan,
        sources=SOURCE_POLICIES,
    )

    assert "recovery after lost" in queries[0][0]
    assert "first acquisition source" in queries[1][0]
    assert "site:" not in queries[1][0]


def test_initial_entity_group_keeps_planner_alias_for_cross_language_retrieval() -> None:
    plan = SearchPlan(
        aliases=["Stone Key"],
        named_entity_groups=[["青石钥匙"]],
        queries=[{"source_type": "wiki", "query": "青石钥匙 使用条件"}],
    )

    queries = build_search_queries(
        game="示例游戏",
        question="青石钥匙如何使用？",
        plan=plan,
        sources=SOURCE_POLICIES,
    )

    assert any("Stone Key" in query for query, _source in queries)


def test_query_portfolio_stays_within_global_and_refinement_limits() -> None:
    plan = SearchPlan(
        queries=[
            {"source_type": source_type, "query": f"entity relation {index}"}
            for index, source_type in enumerate(
                ("wiki", "community", "web", "official", "wiki", "web")
            )
        ]
    )

    queries = build_search_queries(
        game="Any Game",
        question="How is the entity related to the ending?",
        plan=plan,
        sources=SOURCE_POLICIES,
    )
    assert len(queries) <= MAX_SEARCH_QUERIES

    refinement = plan.model_copy(update={"refinement": True})
    refined_queries = build_search_queries(
        game="Any Game",
        question="How is the entity related to the ending?",
        plan=refinement,
        sources=SOURCE_POLICIES,
    )
    assert len(refined_queries) == 1
    assert "site:" not in refined_queries[0][0]


def test_refinement_open_query_does_not_reattach_unrelated_original_entities() -> None:
    plan = SearchPlan(
        refinement=True,
        aliases=["Blue Gate"],
        named_entity_groups=[
            ["Amber Relay"],
            ["Blue Gate"],
        ],
        queries=[{"source_type": "web", "query": "Amber Relay power source"}],
    )

    queries = build_search_queries(
        game="Example Game",
        question="Does the Amber Relay open the Blue Gate?",
        plan=plan,
        sources=SOURCE_POLICIES,
    )

    assert len(queries) == 1
    assert "Amber Relay" in queries[0][0]
    assert "Blue Gate" not in queries[0][0]


def test_plan_merge_unions_overlapping_alias_groups_for_one_entity() -> None:
    merged = merge_search_plans(
        SearchPlan(
            named_entity_groups=[
                ["蓝色大门", "Blue Gate"],
                ["琥珀继电器", "Amber Relay"],
            ],
        ),
        SearchPlan(
            named_entity_groups=[
                ["蓝色大门", "Azure Gate"],
            ],
        ),
    )

    assert merged.named_entity_groups == [
        ["蓝色大门", "Blue Gate", "Azure Gate"],
        ["琥珀继电器", "Amber Relay"],
    ]


@pytest.mark.asyncio
async def test_first_wave_never_exceeds_request_query_budget() -> None:
    class CountingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def search(self, **kwargs):
            self.calls.append((kwargs["query"], kwargs.get("search_depth")))
            return {"results": []}

    client = CountingClient()
    provider = TavilySearchProvider(
        settings=Settings(
            mediawiki_direct_search=False,
            search_cache_use_redis=False,
            tavily_first_wave_queries=4,
            tavily_max_queries_per_request=1,
        ),
        client=client,
    )
    await provider.search(
        "How are the two factions related?",
        "Unknown Game",
        plan=SearchPlan(
            intent="general",
            queries=[
                {"source_type": "wiki", "query": "faction relation"},
                {"source_type": "community", "query": "ending consequence"},
                {"source_type": "web", "query": "inheritance rule"},
            ],
        ),
        game_resolution=GameResolution(
            input_name="Unknown Game",
            confirmed_name="Unknown Game",
            confidence=1,
        ),
    )

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_unconfirmed_identity_reserves_one_paid_query_for_recovery() -> None:
    class CountingClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, **kwargs):
            self.calls.append(kwargs["query"])
            return {"results": []}

    client = CountingClient()
    provider = TavilySearchProvider(
        settings=Settings(
            mediawiki_direct_search=False,
            search_cache_use_redis=False,
            tavily_first_wave_queries=3,
            tavily_max_queries_per_request=4,
            tavily_unconfirmed_identity_reserve=1,
        ),
        client=client,
    )
    await provider.search(
        "How does the Quartz Relay work?",
        "Synthetic Adventure",
        plan=SearchPlan(
            queries=[
                {"source_type": "wiki", "query": "Synthetic Adventure Quartz Relay"},
                {"source_type": "community", "query": "Quartz Relay behavior"},
                {"source_type": "web", "query": "Quartz Relay guide"},
                {"source_type": "web", "query": "Quartz Relay source"},
            ],
        ),
        game_resolution=GameResolution(input_name="Synthetic Adventure", confirmed_name="Synthetic Adventure"),
    )

    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_relation_verification_keeps_all_independent_first_wave_routes_within_budget() -> None:
    class CountingClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def search(self, **kwargs):
            self.calls.append((kwargs["query"], kwargs.get("search_depth")))
            return {"results": []}

    client = CountingClient()
    provider = TavilySearchProvider(
        settings=Settings(
            mediawiki_direct_search=False,
            search_cache_use_redis=False,
            tavily_first_wave_queries=3,
            tavily_max_queries_per_request=4,
        ),
        client=client,
    )
    await provider.search(
        "Does the Amber Relay open the Blue Gate?",
        "Example Game",
        plan=SearchPlan(
            requires_relation_verification=True,
            queries=[
                {"source_type": "wiki", "query": "Amber Relay Blue Gate condition"},
                {"source_type": "community", "query": "Amber Relay Blue Gate report"},
                {"source_type": "web", "query": "Amber Relay Blue Gate"},
            ],
        ),
        game_resolution=GameResolution(
            input_name="Example Game", confirmed_name="Example Game", confidence=1
        ),
    )

    assert len(client.calls) == 3
    assert {depth for _query, depth in client.calls} == {"advanced"}


def test_fallback_plan_keeps_novel_question_as_an_open_candidate() -> None:
    question = "蓝色天气会不会改变两个阵营之间的声望继承关系？"

    plan = fallback_search_plan(question=question)

    assert plan.intent == "general"
    assert any(query.source_type == "web" and query.query == question for query in plan.queries)
    assert {query.source_type for query in plan.queries} >= {"wiki", "community", "web"}


def test_fallback_latin_fragment_never_replaces_unknown_relationship() -> None:
    question = "这个效果在 DLC 后会不会改变阵营声望继承关系？"

    plan = fallback_search_plan(question=question)

    assert plan.aliases == []
    assert all("阵营声望继承关系" in query.query for query in plan.queries)


def test_fallback_never_classifies_a_relation_from_fixed_signals() -> None:
    plan = fallback_search_plan(question="Moon Seal 在哪里用？")

    assert plan.intent == "general"
    assert plan.aliases == []
    assert all(query.query == "Moon Seal 在哪里用？" for query in plan.queries)


def test_fallback_plan_bounds_long_unclassified_questions() -> None:
    question = "一个尚未分类的新关系问题" * 80

    plan = fallback_search_plan(question=question)

    assert plan.queries
    assert all(0 < len(query.query) <= 240 for query in plan.queries)


def test_long_open_query_keeps_identity_and_relation_at_opposite_edges() -> None:
    relation_tail = "FINAL_RELATION_TOKEN"
    question = f"How does the unknown condition {'context ' * 100}{relation_tail}?"
    plan = SearchPlan(
        intent="general",
        aliases=["Artifact ZX-900"],
        queries=[{"source_type": "web", "query": "Artifact ZX-900 state transition"}],
    )

    queries = build_search_queries(
        game="Unfamiliar Game",
        question=question,
        plan=plan,
        sources=SOURCE_POLICIES,
    )

    open_query = next(query for query, policy in queries if policy.source_type == "web")
    assert len(open_query) <= 500
    assert open_query.startswith("Unfamiliar Game")
    assert relation_tail in open_query
