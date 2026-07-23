"""LangGraph for build and stat requests."""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, StateGraph

from retrieval.artifacts import RetrievalOutcome
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, SessionMessage
from workflow import WorkflowRouter
from workflows.build.nodes.answer import answer
from workflows.build.nodes.retrieval import retrieve
from workflows.build.nodes.verification import next_after_research, verify
from workflows.build.state import BuildState


class BuildWorkflow:
    """A build-specific graph over shared retrieval and evidence services."""

    def __init__(self, *, retrieve_after_identity_check: Callable[..., Awaitable[tuple[RetrievalOutcome, GameResolution]]], render_answer: Callable[..., Awaitable[str]], safety_refusal_message: Callable[[], str], verification_router: WorkflowRouter) -> None:
        self._retrieve_after_identity_check = retrieve_after_identity_check
        self._render_answer = render_answer
        self._safety_refusal_message = safety_refusal_message
        self._verification_router = verification_router

    async def run(self, *, request: ChatRequest, history: list[SessionMessage], game_resolution: GameResolution, search_plan: SearchPlan, timings_ms: dict[str, int], agent_trace: list[Any]) -> BuildState:
        graph = self._build_graph(request=request, history=history)
        entities = list(dict.fromkeys(alias for group in search_plan.named_entity_groups for alias in group))
        return await graph.ainvoke({
            "game": game_resolution, "level": "", "stats": [], "equipment": entities,
            "candidates": entities, "requirements": search_plan.answer_requirements,
            "search_plan": search_plan, "evidence": [],
            "investigation": InvestigationState(goal=request.question), "answer": "",
            "timings_ms": timings_ms, "agent_trace": agent_trace,
        })

    def _build_graph(self, *, request: ChatRequest, history: list[SessionMessage]):
        graph = StateGraph(BuildState)
        graph.add_node("build_retrieval", lambda state: retrieve(state, request=request, history=history, retrieve_after_identity_check=self._retrieve_after_identity_check))
        graph.add_node("build_verification", lambda state: verify(state, router=self._verification_router))
        graph.add_node("build_answer", lambda state: answer(state, request=request, history=history, render_answer=self._render_answer, safety_refusal_message=self._safety_refusal_message))
        graph.set_entry_point("build_retrieval")
        graph.add_conditional_edges("build_retrieval", lambda state: next_after_research(state, router=self._verification_router), {"verification": "build_verification", "writer": "build_answer"})
        graph.add_edge("build_verification", "build_answer")
        graph.add_edge("build_answer", END)
        return graph.compile()
