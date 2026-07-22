"""LangGraph construction kept separate from the API-facing agent facade."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from orchestration.state import QuestAgentState


def build_request_graph(
    *,
    identity_node: Callable[..., Any],
    planning_node: Callable[..., Any],
    retrieval_node: Callable[..., Any],
    answer_node: Callable[..., Any],
):
    graph = StateGraph(QuestAgentState)
    graph.add_node("identity_agent", identity_node)
    graph.add_node("planning_agent", planning_node)
    graph.add_node("retrieval_evidence_agents", retrieval_node)
    graph.add_node("answer_agent", answer_node)
    graph.set_entry_point("identity_agent")
    graph.add_edge("identity_agent", "planning_agent")
    graph.add_edge("planning_agent", "retrieval_evidence_agents")
    graph.add_edge("retrieval_evidence_agents", "answer_agent")
    graph.add_edge("answer_agent", END)
    return graph.compile()
