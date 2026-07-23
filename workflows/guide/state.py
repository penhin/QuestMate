"""State local to the guide task workflow."""

from typing import TypedDict

from agents import AgentTrace
from schemas import GameResolution, InvestigationState, SearchPlan, Source


class GuideState(TypedDict):
    """Only guide execution artifacts; request runtime data stays outside."""

    game: GameResolution
    quest: str
    location: str
    requirements: list[str]
    search_plan: SearchPlan
    evidence: list[Source]
    investigation: InvestigationState
    answer: str
    timings_ms: dict[str, int]
    agent_trace: list[AgentTrace]
