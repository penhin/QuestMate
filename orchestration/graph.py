"""LangGraph construction kept separate from the API-facing agent facade."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from orchestration.state import QuestAgentState


def build_request_graph(
    *,
    identity_node: Callable[..., Any],
    planning_node: Callable[..., Any],
    routing_node: Callable[..., Any] | None = None,
    guide_workflow_node: Callable[..., Any] | None = None,
    build_workflow_node: Callable[..., Any] | None = None,
    analysis_workflow_node: Callable[..., Any] | None = None,
    task_workflow_router: Callable[..., str] | None = None,
    retrieval_node: Callable[..., Any],
    answer_node: Callable[..., Any],
    verification_node: Callable[..., Any] | None = None,
    workflow_router: Callable[..., str] | None = None,
):
    graph = StateGraph(QuestAgentState)
    graph.add_node("identity_agent", identity_node)
    graph.add_node("planning_agent", planning_node)
    if routing_node is not None:
        graph.add_node("workflow_router", routing_node)
    if guide_workflow_node is not None:
        graph.add_node("guide_workflow", guide_workflow_node)
    if build_workflow_node is not None:
        graph.add_node("build_workflow", build_workflow_node)
    if analysis_workflow_node is not None:
        graph.add_node("analysis_workflow", analysis_workflow_node)
    graph.add_node("retrieval_evidence_agents", retrieval_node)
    if verification_node is not None:
        graph.add_node("verification_agent", verification_node)
    graph.add_node("answer_agent", answer_node)
    graph.set_entry_point("identity_agent")
    graph.add_edge("identity_agent", "planning_agent")
    if routing_node is not None and guide_workflow_node is not None and build_workflow_node is not None and analysis_workflow_node is not None and task_workflow_router is not None:
        graph.add_edge("planning_agent", "workflow_router")
        graph.add_conditional_edges(
            "workflow_router",
            task_workflow_router,
            {
                "guide": "guide_workflow",
                "build": "build_workflow",
                "analysis": "analysis_workflow",
            },
        )
        graph.add_edge("guide_workflow", END)
        graph.add_edge("build_workflow", END)
        graph.add_edge("analysis_workflow", END)
    elif routing_node is not None:
        graph.add_edge("planning_agent", "workflow_router")
        graph.add_edge("workflow_router", "retrieval_evidence_agents")
    else:
        graph.add_edge("planning_agent", "retrieval_evidence_agents")
    if workflow_router is not None and verification_node is not None:
        graph.add_conditional_edges(
            "retrieval_evidence_agents",
            workflow_router,
            {"verification": "verification_agent", "writer": "answer_agent"},
        )
        graph.add_edge("verification_agent", "answer_agent")
    else:
        graph.add_edge("retrieval_evidence_agents", "answer_agent")
    graph.add_edge("answer_agent", END)
    return graph.compile()
