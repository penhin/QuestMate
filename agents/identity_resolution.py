"""Identity-resolution policy kept outside the request orchestrator."""

from typing import Any

from config import get_settings
from game_resolution import (
    is_candidate_identity_url,
    resolution_matches_selected_url,
    select_game_candidate,
)
from retrieval.wiki_domains import normalize_wiki_host
from schemas import ChatRequest, GameResolution, Source


class IdentityResolver:
    """Resolves explicit player choices and safely recovers missing identity."""

    def __init__(self, search_provider: Any) -> None:
        self.search_provider = search_provider

    async def resolve_request_game(self, request: ChatRequest) -> GameResolution:
        confirmed = self.confirmed_resolution_from_request(request)
        if confirmed is None:
            return await self.search_provider.resolve_game(request.game, question=request.question)
        settings = get_settings()
        if (
            settings.allow_evaluation_retrieval_hints
            and not settings.is_production
            and request.metadata.get("evaluation") is True
        ):
            return confirmed
        selected_url = request.metadata.get("selected_game_url")
        if isinstance(selected_url, str) and len(selected_url) <= 500 and is_candidate_identity_url(selected_url):
            selector = getattr(self.search_provider, "select_game_candidate", None)
            if callable(selector):
                selected = await selector(game=request.game, selected_url=selected_url, question=request.question)
                if (
                    selected.is_confirmed
                    and not selected.ambiguous
                    and resolution_matches_selected_url(selected, selected_url=selected_url)
                ):
                    return selected
                return selected.model_copy(update={"confidence": 0, "ambiguous": bool(selected.candidates)})

        discovered = await self.search_provider.resolve_game(request.game, question=request.question)
        if isinstance(selected_url, str) and is_candidate_identity_url(selected_url):
            selected = select_game_candidate(discovered, selected_url=selected_url)
            if selected is not None:
                return selected
            return GameResolution(
                input_name=request.game,
                confirmed_name=request.game,
                confidence=0,
                candidates=discovered.candidates,
                ambiguous=bool(discovered.candidates),
            )
        return discovered

    async def initial_context(self, request: ChatRequest) -> GameResolution:
        if request.metadata.get("confirmed_game") is True or request.metadata.get("selected_game_url"):
            return await self.resolve_request_game(request)
        # A player-selected game name is sufficient to start a conversation,
        # but it is not enough to know which game wiki can be queried. Reuse a
        # server-verified profile first; otherwise resolve once and persist the
        # result so later requests do not repeatedly spend discovery budget.
        cached_resolution = getattr(self.search_provider, "get_cached_game_resolution", None)
        if callable(cached_resolution):
            cached = await cached_resolution(request.game)
            if cached is not None and cached.is_confirmed and not cached.ambiguous:
                return cached
        discovered = await self.resolve_request_game(request)
        if discovered.is_confirmed or discovered.ambiguous or discovered.candidates:
            return discovered
        # A transient identity-search failure must not make an ordinary chat
        # request unusable. Retrieval will remain conservative without known
        # database domains and can recover identity later if it finds nothing.
        return GameResolution(input_name=request.game, confirmed_name=request.game, confidence=1)

    async def recover_if_needed(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        current: GameResolution,
    ) -> GameResolution:
        if current.ambiguous or not current.is_confirmed or sources or request.metadata.get("confirmed_game") is True:
            return current
        cached_resolution = getattr(self.search_provider, "get_cached_game_resolution", None)
        if callable(cached_resolution):
            cached = await cached_resolution(request.game)
            if cached is not None and cached.is_confirmed and not cached.ambiguous:
                return cached
        recovered = await self.resolve_request_game(request)
        return recovered if (recovered.is_confirmed or recovered.ambiguous or recovered.candidates) else current

    @staticmethod
    def confirmed_resolution_from_request(request: ChatRequest) -> GameResolution | None:
        if request.metadata.get("confirmed_game") is not True:
            return None
        settings = get_settings()
        allow_hints = settings.allow_evaluation_retrieval_hints and not settings.is_production
        aliases = request.metadata.get("game_aliases") if allow_hints else None
        database_domains = request.metadata.get("database_domains") if allow_hints else None
        safe_aliases = [
            normalized
            for value in (aliases if isinstance(aliases, list) else [])[:8]
            if isinstance(value, str)
            and (normalized := " ".join(value.split()).strip())
            and len(normalized) <= 120
            and not any(marker in normalized.casefold() for marker in ("http://", "https://", "site:"))
        ]
        safe_domains = [
            host
            for value in (database_domains if isinstance(database_domains, list) else [])[:8]
            if isinstance(value, str) and (host := normalize_wiki_host(value)) is not None
        ]
        return GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=list(dict.fromkeys(safe_aliases)),
            database_domains=list(dict.fromkeys(safe_domains)),
            confidence=1,
            ambiguous=False,
        )
