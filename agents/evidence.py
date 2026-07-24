"""Evidence-assessment specialist boundary."""

from typing import Any

from schemas import InvestigationState


class EvidenceAgent:
    """Assesses evidence gaps; it cannot perform retrieval itself."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
    async def update_investigation(self, **kwargs: Any) -> InvestigationState:
        return await self._llm.update_investigation(**kwargs)
