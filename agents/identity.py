"""Game identity specialist boundary."""

from collections.abc import Awaitable, Callable

from schemas import ChatRequest, GameResolution, Source


class IdentityAgent:
    """Owns game identity context and post-retrieval recovery only."""

    def __init__(
        self,
        *,
        initial_context: Callable[[ChatRequest], Awaitable[GameResolution]],
        recover_context: Callable[..., Awaitable[GameResolution]],
    ) -> None:
        self._initial_context = initial_context
        self._recover_context = recover_context

    async def initial(self, request: ChatRequest) -> GameResolution:
        return await self._initial_context(request)

    async def recover(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        current: GameResolution,
    ) -> GameResolution:
        return await self._recover_context(request=request, sources=sources, current=current)
