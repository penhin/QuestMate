"""LangGraph for build and stat requests."""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, StateGraph

from retrieval.artifacts import RetrievalOutcome
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, SessionMessage
from workflows.verification import EvidenceVerificationRouter
from workflows.build.nodes.answer import answer
from workflows.build.nodes.retrieval import retrieve
from workflows.build.nodes.verification import next_after_research, verify
from workflows.build.state import BuildState
from workflows.streaming import stream_workflow_answer


class BuildWorkflow:
    """A build-specific graph over shared retrieval and evidence services."""

    def __init__(self, *, retrieve_after_identity_check: Callable[..., Awaitable[tuple[RetrievalOutcome, GameResolution]]], render_answer: Callable[..., Awaitable[str]], safety_refusal_message: Callable[[], str], verification_router: EvidenceVerificationRouter) -> None:
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

    async def stream(self, **kwargs: Any) -> Any:
        request: ChatRequest = kwargs["request"]
        history: list[SessionMessage] = kwargs["history"]
        plan: SearchPlan = kwargs["search_plan"]
        entities = list(dict.fromkeys(alias for group in plan.named_entity_groups for alias in group))
        state: BuildState = {
            "game": kwargs["game_resolution"], "level": "", "stats": [], "equipment": entities,
            "candidates": entities, "requirements": plan.answer_requirements, "search_plan": plan,
            "evidence": [], "investigation": InvestigationState(goal=request.question), "answer": "",
            "timings_ms": kwargs["timings_ms"], "agent_trace": kwargs["agent_trace"],
        }
        async for event in stream_workflow_answer(
            state=state, request=request, history=history,
            retrieve_node=retrieve, verify_node=verify, stream_answer=kwargs["stream_answer"],
            safety_refusal_message=self._safety_refusal_message,
            retrieve_after_identity_check=self._retrieve_after_identity_check,
            verification_router=self._verification_router, workflow_name="build",
        ):
            yield event

    def _build_graph(self, *, request: ChatRequest, history: list[SessionMessage]):
        graph = StateGraph(BuildState)
        async def retrieval_node(state: BuildState):
            return await retrieve(state, request=request, history=history, retrieve_after_identity_check=self._retrieve_after_identity_check)
        async def verification_node(state: BuildState):
            return await verify(state, router=self._verification_router)
        async def answer_node(state: BuildState):
            return await answer(state, request=request, history=history, render_answer=self._render_answer, safety_refusal_message=self._safety_refusal_message)
        graph.add_node("build_retrieval", retrieval_node)
        graph.add_node("build_verification", verification_node)
        graph.add_node("build_answer", answer_node)
        graph.set_entry_point("build_retrieval")
        graph.add_conditional_edges("build_retrieval", lambda state: next_after_research(state, router=self._verification_router), {"verification": "build_verification", "writer": "build_answer"})
        graph.add_edge("build_verification", "build_answer")
        graph.add_edge("build_answer", END)
        return graph.compile()
