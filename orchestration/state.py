"""LangGraph state shared by the orchestrator nodes."""

from typing import TypedDict

from agents import AgentTrace
from task_router import TaskRouteDecision
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, SessionMessage, Source


class QuestAgentState(TypedDict):
    request: ChatRequest
    history: list[SessionMessage]
    game_resolution: GameResolution
    search_plan: SearchPlan
    route: TaskRouteDecision
    sources: list[Source]
    investigation: InvestigationState
    answer: str
    timings_ms: dict[str, int]
    agent_trace: list[AgentTrace]
