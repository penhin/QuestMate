"""Answer-rendering specialist boundary."""

from typing import Any

class AnswerAgent:
    """Renders an answer solely from verified sources and investigation state."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def answer(self, **kwargs: Any) -> str:
        return await self._llm.answer(**kwargs)

    async def stream_answer(self, **kwargs: Any):
        async for chunk in self._llm.stream_answer(**kwargs):
            yield chunk
