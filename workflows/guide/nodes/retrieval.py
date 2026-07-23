"""Evidence retrieval node for guide tasks."""

from collections.abc import Awaitable, Callable
from time import perf_counter

from agents import AgentTrace
from retrieval.artifacts import RetrievalOutcome
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, SessionMessage
from runtime import active_context
from workflows.guide.state import GuideState


async def retrieve(
    state: GuideState,
    *,
    request: ChatRequest,
    history: list[SessionMessage],
    retrieve_after_identity_check: Callable[..., Awaitable[tuple[RetrievalOutcome, GameResolution]]],
) -> GuideState:
    started = perf_counter()
    plan = state["search_plan"]
    if plan.safety_refusal:
        result = {
            **state,
            "evidence": [],
            "investigation": InvestigationState(goal=request.question),
        }
        if context := active_context():
            context.trace.record("node.retrieval", started)
        return result
    outcome, game = await retrieve_after_identity_check(
        request=request,
        history=history,
        plan=plan,
        game_resolution=state["game"],
        timings_ms=state["timings_ms"],
    )
    result = {
        **state,
        "game": game,
        "search_plan": outcome.plan,
        "evidence": outcome.sources,
        "investigation": outcome.investigation,
        "agent_trace": [*state["agent_trace"], AgentTrace(
            "guide_retrieval", "investigate", len(outcome.sources), outcome.refined
        )],
    }
    if context := active_context():
        context.trace.record("node.retrieval", started)
    return result
