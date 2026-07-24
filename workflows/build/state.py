"""State local to build-oriented workflow execution."""

from typing import TypedDict

from agents import AgentTrace
from schemas import GameResolution, InvestigationState, SearchPlan, Source


class BuildState(TypedDict):
    game: GameResolution
    level: str
    stats: list[str]
    equipment: list[str]
    candidates: list[str]
    requirements: list[str]
    search_plan: SearchPlan
    evidence: list[Source]
    investigation: InvestigationState
    answer: str
    timings_ms: dict[str, int]
    agent_trace: list[AgentTrace]
