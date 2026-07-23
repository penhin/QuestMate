"""State local to mechanic and comparison analysis."""

from typing import TypedDict

from agents import AgentTrace
from schemas import GameResolution, InvestigationState, SearchPlan, Source


class AnalysisState(TypedDict):
    game: GameResolution
    mechanic: str
    comparison: list[str]
    reasoning_requirements: list[str]
    search_plan: SearchPlan
    evidence: list[Source]
    investigation: InvestigationState
    answer: str
    timings_ms: dict[str, int]
    agent_trace: list[AgentTrace]
