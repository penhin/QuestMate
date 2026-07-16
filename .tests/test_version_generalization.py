import pytest
from datetime import datetime, timezone

from ai.fallback_planning import fallback_search_plan
from config import Settings
from quality_policy import is_version_sensitive_question
from retrieval.coordinator import merge_search_plans
from schemas import GameResolution, SearchPlan, Source
from search import TavilySearchProvider
from llm import GuideLLM
from schemas import ChatRequest


def test_version_dimension_is_orthogonal_to_location_intent() -> None:
    plan = fallback_search_plan(question="当前版本月光钥匙在哪？")

    assert plan.intent == "item_location"
    assert plan.version_sensitive
    assert is_version_sensitive_question("Moon Key location in v2.01")
    assert is_version_sensitive_question("Moon Key location in version 2.01")
    assert is_version_sensitive_question("Moon Key location in 2.1.3")
    assert not is_version_sensitive_question("Where is room 12?")
    assert not is_version_sensitive_question("Where is the Dispatch Key?")
    assert not is_version_sensitive_question("Move to coordinates 12.5, 30.2")
    assert not is_version_sensitive_question("Does this have a 3.5-star rating?")


def test_refinement_cannot_drop_initial_version_dimension() -> None:
    merged = merge_search_plans(
        SearchPlan(intent="item_location", version_sensitive=True),
        SearchPlan(intent="item_location", version_sensitive=False, refinement=True),
    )

    assert merged.version_sensitive


@pytest.mark.asyncio
async def test_versioned_stable_fact_does_not_stop_after_direct_wiki(monkeypatch) -> None:
    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, **_kwargs):
            self.calls += 1
            return {"results": []}

    client = CountingClient()
    provider = TavilySearchProvider(
        settings=Settings(
            mediawiki_direct_search=False,
            search_cache_use_redis=False,
            tavily_first_wave_queries=1,
            tavily_max_queries_per_request=1,
        ),
        client=client,
    )
    direct_sources = [
        Source(
            title=f"Moon Key Wiki {index}",
            url=f"https://example.fandom.com/wiki/Moon_Key_{index}",
            evidence="The Moon Key is in the old tower.",
            source_type="wiki",
            score=0.9,
        )
        for index in range(2)
    ]

    async def direct_wiki(**_kwargs):
        return direct_sources

    monkeypatch.setattr(provider, "_search_mediawiki_sources", direct_wiki)

    await provider.search(
        "当前版本月光钥匙在哪？",
        "Example Adventure",
        plan=SearchPlan(
            intent="item_location",
            version_sensitive=True,
            queries=[{"source_type": "official", "query": "current Moon Key location"}],
        ),
        game_resolution=GameResolution(
            input_name="Example Adventure",
            confirmed_name="Example Adventure",
            confidence=1,
        ),
    )

    assert client.calls == 1


def test_version_scoring_uses_orthogonal_flag() -> None:
    assert TavilySearchProvider._version_safety_score(
        intent="item_location",
        version_sensitive=True,
        source_type="official",
        text="Current version notes",
    ) == 1.0


def test_model_plan_cannot_drop_explicit_version_signal() -> None:
    plan = GuideLLM._parse_search_plan(
        '{"intent":"item_location","version_sensitive":false,"aliases":["Moon Key"],'
        '"queries":[{"source_type":"wiki","query":"Moon Key location"}],"missing_info":[]}',
        fallback_question="当前版本月光钥匙在哪？",
    )

    assert plan.intent == "item_location"
    assert plan.version_sensitive


def test_current_version_location_requires_dated_context() -> None:
    request = ChatRequest(game="Example Adventure", question="当前版本 Moon Key 在哪？")
    plan = SearchPlan(intent="item_location", version_sensitive=True, aliases=["Moon Key"])
    undated = Source(
        title="Moon Key",
        url="https://example.com/moon-key",
        evidence="Example Adventure places the Moon Key in the old tower.",
    )
    dated = undated.model_copy(update={"published_at": datetime(2026, 1, 2, tzinfo=timezone.utc)})

    assert GuideLLM._should_return_conservative_answer(
        request=request,
        sources=[undated],
        plan=plan,
        game_resolution=GameResolution(
            input_name="Example Adventure",
            confirmed_name="Example Adventure",
            confidence=1,
        ),
    )
    assert not GuideLLM._should_return_conservative_answer(
        request=request,
        sources=[dated],
        plan=plan,
        game_resolution=GameResolution(
            input_name="Example Adventure",
            confirmed_name="Example Adventure",
            confidence=1,
        ),
    )


def test_unrelated_dated_page_cannot_make_an_undated_target_current() -> None:
    request = ChatRequest(game="Example Adventure", question="当前版本 Moon Key 在哪？")
    plan = SearchPlan(intent="item_location", version_sensitive=True, aliases=["Moon Key"])
    target = Source(
        title="Moon Key",
        url="https://example.com/moon-key",
        evidence="Example Adventure places the Moon Key in the old tower.",
    )
    unrelated_dated = Source(
        title="Example Adventure tournament announcement",
        url="https://example.com/tournament",
        evidence="Registration for the Example Adventure tournament is open.",
        published_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert GuideLLM._should_return_conservative_answer(
        request=request,
        sources=[target, unrelated_dated],
        plan=plan,
        game_resolution=GameResolution(
            input_name="Example Adventure",
            confirmed_name="Example Adventure",
            confidence=1,
        ),
    )
