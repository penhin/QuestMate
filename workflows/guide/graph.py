"""LangGraph for route-oriented game-guide questions."""

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, StateGraph

from retrieval.artifacts import RetrievalOutcome
from schemas import ChatRequest, ChatResponse, GameResolution, InvestigationState, SearchPlan, SessionMessage
from workflow import WorkflowRouter
from workflows.guide.nodes.answer import answer
from workflows.guide.nodes.retrieval import retrieve
from workflows.guide.nodes.verification import next_after_research, verify
from workflows.guide.state import GuideState
from workflows.streaming import stream_workflow_answer


class GuideWorkflow:
    """Run existing evidence capabilities through a guide-specific graph."""

    def __init__(
        self,
        *,
        retrieve_after_identity_check: Callable[..., Awaitable[tuple[RetrievalOutcome, GameResolution]]],
        render_answer: Callable[..., Awaitable[str]],
        safety_refusal_message: Callable[[], str],
        verification_router: WorkflowRouter,
    ) -> None:
        self._retrieve_after_identity_check = retrieve_after_identity_check
        self._render_answer = render_answer
        self._safety_refusal_message = safety_refusal_message
        self._verification_router = verification_router

    async def run(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        game_resolution: GameResolution,
        search_plan: SearchPlan,
        timings_ms: dict[str, int],
        agent_trace: list[Any],
    ) -> GuideState:
        graph = self._build_graph(request=request, history=history)
        return await graph.ainvoke({
            "game": game_resolution,
            "quest": request.question if search_plan.intent == "quest_step" else "",
            "location": request.question if search_plan.intent in {"item_location", "item_usage"} else "",
            "requirements": search_plan.answer_requirements,
            "search_plan": search_plan,
            "evidence": [],
            "investigation": InvestigationState(goal=request.question),
            "answer": "",
            "timings_ms": timings_ms,
            "agent_trace": agent_trace,
        })

    async def stream(self, **kwargs: Any) -> Any:
        request: ChatRequest = kwargs["request"]
        history: list[SessionMessage] = kwargs["history"]
        search_plan: SearchPlan = kwargs["search_plan"]
        state: GuideState = {
            "game": kwargs["game_resolution"],
            "quest": request.question if search_plan.intent == "quest_step" else "",
            "location": request.question if search_plan.intent in {"item_location", "item_usage"} else "",
            "requirements": search_plan.answer_requirements, "search_plan": search_plan,
            "evidence": [], "investigation": InvestigationState(goal=request.question), "answer": "",
            "timings_ms": kwargs["timings_ms"], "agent_trace": kwargs["agent_trace"],
        }
        async for event in stream_workflow_answer(
            state=state, request=request, history=history,
            retrieve_node=retrieve, verify_node=verify, stream_answer=kwargs["stream_answer"],
            safety_refusal_message=self._safety_refusal_message,
            retrieve_after_identity_check=self._retrieve_after_identity_check,
            verification_router=self._verification_router, workflow_name="guide",
        ):
            yield event

    def _build_graph(self, *, request: ChatRequest, history: list[SessionMessage]):
        graph = StateGraph(GuideState)
        async def retrieval_node(state: GuideState):
            return await retrieve(state, request=request, history=history, retrieve_after_identity_check=self._retrieve_after_identity_check)
        async def verification_node(state: GuideState):
            return await verify(state, router=self._verification_router)
        async def answer_node(state: GuideState):
            return await answer(state, request=request, history=history, render_answer=self._render_answer, safety_refusal_message=self._safety_refusal_message)
        graph.add_node(
            "guide_retrieval",
            retrieval_node,
        )
        graph.add_node(
            "guide_verification",
            verification_node,
        )
        graph.add_node(
            "guide_answer",
            answer_node,
        )
        graph.set_entry_point("guide_retrieval")
        graph.add_conditional_edges(
            "guide_retrieval",
            lambda state: next_after_research(state, router=self._verification_router),
            {"verification": "guide_verification", "writer": "guide_answer"},
        )
        graph.add_edge("guide_verification", "guide_answer")
        graph.add_edge("guide_answer", END)
        return graph.compile()
