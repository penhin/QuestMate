"""Compatibility exports for the specialist-agent package.

New code should import from :mod:`agents`; this module preserves the initial
multi-agent import path for local integrations.
"""

from agents import AnswerAgent, AgentTrace, EvidenceAgent, IdentityAgent, PlanningAgent, RetrievalAgent

__all__ = [
    "AgentTrace",
    "AnswerAgent",
    "EvidenceAgent",
    "IdentityAgent",
    "PlanningAgent",
    "RetrievalAgent",
]
