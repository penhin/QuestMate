"""Shared streaming bridge for task workflows; no game-specific decisions."""

from collections.abc import AsyncIterator, Awaitable, Callable
from time import perf_counter
from typing import Any

from agents import AgentTrace
from runtime import active_context
from schemas import ChatRequest, SessionMessage
from workflows.verification import EvidencePath, EvidenceVerificationRouter


async def stream_workflow_answer(
    *,
    state: dict[str, Any],
    request: ChatRequest,
    history: list[SessionMessage],
    retrieve_node: Callable[..., Awaitable[dict[str, Any]]],
    verify_node: Callable[..., Awaitable[dict[str, Any]]],
    stream_answer: Callable[..., AsyncIterator[str]],
    safety_refusal_message: Callable[[], str],
    retrieve_after_identity_check: Callable[..., Awaitable[Any]],
    verification_router: EvidenceVerificationRouter,
    workflow_name: str,
) -> AsyncIterator[tuple[str, dict[str, Any] | str]]:
    """Run shared evidence nodes, then preserve token/chunk streaming.

    The task graph and this streaming bridge use the same typed state and node
    functions.  LangGraph's normal state-node contract returns one update, so
    its streaming adapter lives here rather than buffering the model response.
    """
    state = await retrieve_node(
        state,
        request=request,
        history=history,
        retrieve_after_identity_check=retrieve_after_identity_check,
    )
    yield "state", state
    if verification_router.classify(state["search_plan"]) is EvidencePath.VERIFIED_RESEARCH:
        state = await verify_node(state, router=verification_router)
    if state["search_plan"].safety_refusal:
        state = {**state, "answer": safety_refusal_message()}
        yield "result", state
        return
    started = perf_counter()
    chunks: list[str] = []
    async for chunk in stream_answer(
        request=request,
        sources=state["evidence"],
        plan=state["search_plan"],
        game_resolution=state["game"],
        history=history,
        investigation=state["investigation"],
    ):
        chunks.append(chunk)
        yield "chunk", chunk
    state = {
        **state,
        "answer": "".join(chunks),
        "timings_ms": {**state["timings_ms"], "answer": round((perf_counter() - started) * 1000)},
        "agent_trace": [*state["agent_trace"], AgentTrace(
            f"{workflow_name}_answer", "stream", len(state["evidence"])
        )],
    }
    if context := active_context():
        context.trace.record("node.stream_answer", started)
    yield "result", state
