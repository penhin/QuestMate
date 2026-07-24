from types import SimpleNamespace

import pytest

from schemas import GameResolution, SearchPlan, Source
from search_router.router import SearchRouter
from search_router.providers import MediaWikiProvider, TavilyProvider


def source(url: str) -> Source:
    return Source(title="Moon Key", url=url, evidence="Moon Key evidence", score=0.8)


class Wiki:
    def __init__(self):
        self.database_domains = []

    async def search(self, **_kwargs):
        self.database_domains = _kwargs["database_domains"]
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


@pytest.mark.asyncio
async def test_router_derives_capability_probed_wiki_candidates_from_english_plan() -> None:
    wiki = Wiki()
    routed = SearchRouter(
        mediawiki=MediaWikiProvider(wiki),
        searxng=Searx(),
        tavily=tavily_provider(),
        build_queries=lambda **_kwargs: [("Elden Ring Ranni questline", None)],
        settings=settings(),
    )

    await routed.search(
        query="菈妮支线步骤", game="艾尔登法环", max_results=2,
        plan=SearchPlan(
            aliases=["Ranni"],
            queries=[{"source_type": "wiki", "query": "Elden Ring Ranni questline"}],
        ),
        game_resolution=GameResolution(input_name="艾尔登法环", confirmed_name="艾尔登法环", confidence=1),
    )

    assert "eldenring.fandom.com" in wiki.database_domains
    assert "eldenring.wiki.gg" in wiki.database_domains
