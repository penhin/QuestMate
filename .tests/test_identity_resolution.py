import pytest

from agents.identity_resolution import IdentityResolver
from schemas import ChatRequest, GameResolution


class _Provider:
    def __init__(self, cached: GameResolution | None = None) -> None:
        self.cached = cached
        self.resolve_calls = 0

    async def get_cached_game_resolution(self, _game: str) -> GameResolution | None:
        return self.cached

    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        self.resolve_calls += 1
        return GameResolution(
            input_name=game,
            confirmed_name="ELDEN RING",
            aliases=["Elden Ring"],
            database_domains=["eldenring.fandom.com"],
            confidence=0.85,
        )


@pytest.mark.asyncio
async def test_initial_context_uses_cached_verified_wiki_profile() -> None:
    cached = GameResolution(
        input_name="艾尔登法环",
        confirmed_name="ELDEN RING",
        aliases=["Elden Ring"],
        database_domains=["eldenring.fandom.com"],
        confidence=0.85,
    )
    provider = _Provider(cached)

    resolution = await IdentityResolver(provider).initial_context(
        ChatRequest(game="艾尔登法环", question="菈妮支线步骤")
    )

    assert resolution.database_domains == ["eldenring.fandom.com"]
    assert provider.resolve_calls == 0


@pytest.mark.asyncio
async def test_initial_context_starts_without_discovery_when_cache_misses() -> None:
    provider = _Provider()

    resolution = await IdentityResolver(provider).initial_context(
        ChatRequest(game="艾尔登法环", question="菈妮支线步骤")
    )

    assert resolution.confirmed_name == "艾尔登法环"
    assert provider.resolve_calls == 0
