"""Citation-preserving answer node for guide tasks."""

from collections.abc import Awaitable, Callable
from time import perf_counter

from agents import AgentTrace
from schemas import ChatRequest, SessionMessage
from workflows.guide.state import GuideState


async def answer(
    state: GuideState,
    *,
    request: ChatRequest,
    history: list[SessionMessage],
    render_answer: Callable[..., Awaitable[str]],
    safety_refusal_message: Callable[[], str],
) -> GuideState:
    if state["search_plan"].safety_refusal:
        return {**state, "answer": safety_refusal_message()}
    started = perf_counter()
    rendered = await render_answer(
        request=request,
        sources=state["evidence"],
        plan=state["search_plan"],
        game_resolution=state["game"],
        history=history,
        investigation=state["investigation"],
    )
    return {
        **state,
        "answer": rendered,
        "timings_ms": {**state["timings_ms"], "answer": round((perf_counter() - started) * 1000)},
        "agent_trace": [*state["agent_trace"], AgentTrace(
            "guide_answer", "render", len(state["evidence"])
        )],
    }
