from types import SimpleNamespace

import pytest

from schemas import GameResolution, SearchPlan, Source
from search_router.router import SearchRouter
from search_router.providers import TavilyProvider


def source(url: str) -> Source:
    return Source(title="Moon Key", url=url, evidence="Moon Key evidence", score=0.8)


class Legacy:
    async def _search_mediawiki_sources(self, **_kwargs):
        return []

    def _build_search_queries(self, **_kwargs):
        return [("Moon Key location", None)]

    def _tavily_usage_snapshot(self):
        return {"tavily_paid_calls": 0, "tavily_cache_hits": 0}

    async def _search_with_tavily(self, *_args, **kwargs):
        assert kwargs["skip_mediawiki"] is True
        return [source("https://tavily.example/moon-key")]


class Searx:
    configured = True
    calls = 0

    async def search(self, *_args, **_kwargs):
        self.calls += 1
        return [source("https://searx.example/moon-key")]


class FailingSearx(Searx):
    async def search(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("unavailable")


def settings():
    return SimpleNamespace(
        search_provider_cooldown_seconds=60,
        searxng_max_queries_per_request=3,
        tavily_fallback_max_calls=1,
    )


@pytest.mark.asyncio
async def test_router_prefers_searxng_before_tavily() -> None:
    router = SearchRouter(search_backend=Legacy(), searxng=Searx(), settings=settings())
    results = await router.search(
        query="Moon Key location", game="Example Adventure", max_results=3,
        plan=SearchPlan(intent="item_location"), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://searx.example/moon-key"
    assert router.last_decision and router.last_decision.provider == "searxng"


@pytest.mark.asyncio
async def test_router_falls_back_to_tavily_and_cools_failed_searxng() -> None:
    router = SearchRouter(search_backend=Legacy(), searxng=FailingSearx(), settings=settings())
    results = await router.search(
        query="Moon Key location", game="Example Adventure", max_results=3,
        plan=SearchPlan(intent="build"), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://tavily.example/moon-key"
    assert router.last_decision and router.last_decision.provider == "tavily"
    assert router.health.available("searxng") is False


@pytest.mark.asyncio
async def test_tavily_adapter_hides_legacy_backend_details() -> None:
    backend = Legacy()
    results = await TavilyProvider(backend).search(
        query="Moon Key", game="Example Adventure", max_results=2,
        plan=SearchPlan(), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://tavily.example/moon-key"
