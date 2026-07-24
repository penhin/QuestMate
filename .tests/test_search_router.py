from types import SimpleNamespace

import pytest

from schemas import GameResolution, SearchPlan, Source
from search_router.router import SearchRouter
from search_router.providers import MediaWikiProvider, TavilyProvider


def source(url: str) -> Source:
    return Source(title="Moon Key", url=url, evidence="Moon Key evidence", score=0.8)


class Wiki:
    async def search(self, **_kwargs):
        return []


class DirectWiki:
    async def search(self, **_kwargs):
        return [
            source("https://wiki.example/moon-key"),
            source("https://wiki.example/moon-key-route"),
        ]

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


def tavily_provider() -> TavilyProvider:
    async def search(**_kwargs):
        return [source("https://tavily.example/moon-key")]

    return TavilyProvider(
        search=search,
        usage_snapshot=lambda: {"tavily_paid_calls": 0, "tavily_cache_hits": 0},
    )


def router(*, searxng):
    return SearchRouter(
        mediawiki=MediaWikiProvider(Wiki()),
        searxng=searxng,
        tavily=tavily_provider(),
        build_queries=lambda **_kwargs: [("Moon Key location", None)],
        settings=settings(),
    )


@pytest.mark.asyncio
async def test_router_keeps_sufficient_stable_wiki_evidence_inside_provider_route() -> None:
    searxng = Searx()
    routed = SearchRouter(
        mediawiki=MediaWikiProvider(DirectWiki()),
        searxng=searxng,
        tavily=tavily_provider(),
        build_queries=lambda **_kwargs: [("Moon Key location", None)],
        settings=settings(),
    )

    results = await routed.search(
        query="Moon Key location", game="Example Adventure", max_results=2,
        plan=SearchPlan(intent="item_location"), game_resolution=GameResolution(input_name="Example Adventure"),
    )

    assert len(results) == 2
    assert searxng.calls == 0
    assert routed.last_decision and routed.last_decision.provider == "mediawiki"


@pytest.mark.asyncio
async def test_router_prefers_searxng_before_tavily() -> None:
    routed = router(searxng=Searx())
    results = await routed.search(
        query="Moon Key location", game="Example Adventure", max_results=3,
        plan=SearchPlan(intent="item_location"), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://searx.example/moon-key"
    assert routed.last_decision and routed.last_decision.provider == "searxng"


@pytest.mark.asyncio
async def test_router_falls_back_to_tavily_and_cools_failed_searxng() -> None:
    routed = router(searxng=FailingSearx())
    results = await routed.search(
        query="Moon Key location", game="Example Adventure", max_results=3,
        plan=SearchPlan(intent="build"), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://tavily.example/moon-key"
    assert routed.last_decision and routed.last_decision.provider == "tavily"
    assert routed.health.available("searxng") is False


@pytest.mark.asyncio
async def test_tavily_adapter_accepts_only_explicit_provider_contract() -> None:
    results = await tavily_provider().search(
        query="Moon Key", game="Example Adventure", max_results=2,
        plan=SearchPlan(), game_resolution=GameResolution(input_name="Example Adventure"),
    )
    assert str(results[0].url) == "https://tavily.example/moon-key"
