"""Explicit verification checkpoint for evidence-heavy guide requests."""

from agents import AgentTrace
from workflow import WorkflowKind, WorkflowRouter
from workflows.guide.state import GuideState


async def verify(state: GuideState, *, router: WorkflowRouter) -> GuideState:
    workflow = router.classify(state["search_plan"])
    return {
        **state,
        "agent_trace": [*state["agent_trace"], AgentTrace(
            "guide_verification", workflow.value, len(state["evidence"])
        )],
    }


def next_after_research(state: GuideState, *, router: WorkflowRouter) -> str:
    return (
        "verification"
        if router.classify(state["search_plan"]) is WorkflowKind.VERIFIED_RESEARCH
        else "writer"
    )
