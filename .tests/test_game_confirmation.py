import pytest

from agent import QuestAgent
from config import Settings
from schemas import ChatRequest, GameCandidate, GameResolution
from search import TavilySearchProvider


@pytest.mark.asyncio
async def test_candidate_confirmation_uses_server_validated_identity_url() -> None:
    selected_url = "https://store.steampowered.com/app/202/Shared_Name/"

    class Provider:
        def __init__(self) -> None:
            self.selected = None

        async def select_game_candidate(self, **kwargs):
            self.selected = kwargs["selected_url"]
            return GameResolution(
                input_name="Shared Name",
                confirmed_name="Shared Name",
                platform_urls=[selected_url],
                confidence=0.9,
            )

        async def resolve_game(self, *_args, **_kwargs):
            raise AssertionError("validated selection should not fall back to name-only resolution")

    provider = Provider()
    agent = object.__new__(QuestAgent)
    agent.search_provider = provider

    resolution = await agent._resolve_request_game(ChatRequest(
        game="Shared Name",
        question="Where is the item?",
        metadata={"confirmed_game": True, "selected_game_url": selected_url},
    ))

    assert provider.selected == selected_url
    assert [str(url) for url in resolution.platform_urls] == [selected_url]


@pytest.mark.asyncio
async def test_candidate_confirmation_never_substitutes_a_different_fresh_result() -> None:
    selected_url = "https://store.steampowered.com/app/201/Shared_Name/"
    other_url = "https://store.steampowered.com/app/202/Shared_Name/"

    class Provider:
        async def select_game_candidate(self, **_kwargs):
            return GameResolution(
                input_name="Shared Name",
                confirmed_name="Shared Name",
                platform_urls=[other_url],
                confidence=0.9,
            )

        async def resolve_game(self, *_args, **_kwargs):
            raise AssertionError("a rejected opaque selection must not fall back")

    agent = object.__new__(QuestAgent)
    agent.search_provider = Provider()

    resolution = await agent._resolve_request_game(ChatRequest(
        game="Shared Name",
        question="Where is the item?",
        metadata={"confirmed_game": True, "selected_game_url": selected_url},
    ))

    assert not resolution.is_confirmed
    assert [str(url) for url in resolution.platform_urls] == [other_url]


@pytest.mark.asyncio
async def test_provider_returns_confirmation_state_when_selected_url_disappears(monkeypatch) -> None:
    selected_url = "https://store.steampowered.com/app/201/Shared_Name/"
    other_url = "https://store.steampowered.com/app/202/Shared_Name/"

    class EmptyClient:
        def search(self, **_kwargs):
            return {"results": []}

    provider = TavilySearchProvider(
        settings=Settings(search_cache_use_redis=False),
        client=EmptyClient(),
    )
    fresh = GameResolution(
        input_name="Shared Name",
        confirmed_name="Shared Name",
        confidence=0.9,
        candidates=[GameCandidate(
            name="Shared Name",
            platform_urls=[other_url],
            confidence=0.9,
        )],
    )
    monkeypatch.setattr(provider._game_resolver, "resolve", lambda **_kwargs: fresh)

    resolution = await provider.select_game_candidate(
        game="Shared Name",
        selected_url=selected_url,
    )

    assert not resolution.is_confirmed
    assert resolution.ambiguous
    assert [str(url) for url in resolution.candidates[0].platform_urls] == [other_url]


@pytest.mark.asyncio
async def test_ambiguous_identity_is_never_persisted_to_registry(monkeypatch) -> None:
    class EmptyClient:
        def search(self, **_kwargs):
            return {"results": []}

    class Registry:
        def __init__(self) -> None:
            self.writes = 0

        async def get_resolution(self, _game):
            return None

        async def upsert_resolution(self, _resolution):
            self.writes += 1

    registry = Registry()
    provider = TavilySearchProvider(
        settings=Settings(search_cache_use_redis=False),
        client=EmptyClient(),
        source_registry=registry,
    )
    ambiguous = GameResolution(
        input_name="Shared Name",
        confirmed_name="Shared Name",
        confidence=0.9,
        ambiguous=True,
        candidates=[
            GameCandidate(
                name="Shared Name",
                platform_urls=["https://store.steampowered.com/app/201/Shared_Name/"],
                confidence=0.9,
            ),
            GameCandidate(
                name="Shared Name",
                platform_urls=["https://store.steampowered.com/app/202/Shared_Name/"],
                confidence=0.9,
            ),
        ],
    )
    monkeypatch.setattr(provider._game_resolver, "resolve", lambda **_kwargs: ambiguous)

    resolved = await provider.resolve_game("Shared Name")

    assert resolved.ambiguous
    assert registry.writes == 0
