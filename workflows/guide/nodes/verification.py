"""Explicit verification checkpoint for evidence-heavy guide requests."""

from agents import AgentTrace
from time import perf_counter
from runtime import active_context
from workflow import WorkflowKind, WorkflowRouter
from workflows.guide.state import GuideState


async def verify(state: GuideState, *, router: WorkflowRouter) -> GuideState:
    started = perf_counter()
    workflow = router.classify(state["search_plan"])
    result = {
        **state,
        "agent_trace": [*state["agent_trace"], AgentTrace(
            "guide_verification", workflow.value, len(state["evidence"])
        )],
    }
    if context := active_context():
        context.trace.record("node.verification", started)
    return result


def next_after_research(state: GuideState, *, router: WorkflowRouter) -> str:
    return (
        "verification"
        if router.classify(state["search_plan"]) is WorkflowKind.VERIFIED_RESEARCH
        else "writer"
    )
