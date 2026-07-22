"""Evidence-assessment specialist boundary."""

from typing import Any

from agents.compat import supported_kwargs
from schemas import InvestigationState, SearchPlan


class EvidenceAgent:
    """Assesses evidence gaps; it cannot perform retrieval itself."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.supports_update_investigation = callable(
            getattr(llm, "update_investigation", None)
        )

    async def update_investigation(self, **kwargs: Any) -> InvestigationState:
        update = getattr(self._llm, "update_investigation", None)
        if not callable(update):
            raise RuntimeError("Underlying LLM does not support investigation updates")
        return await update(**supported_kwargs(update, kwargs))

    async def refine_search_plan(self, **kwargs: Any) -> SearchPlan | None:
        refine = getattr(self._llm, "refine_search_plan", None)
        if not callable(refine):
            return None
        return await refine(**supported_kwargs(refine, kwargs))
