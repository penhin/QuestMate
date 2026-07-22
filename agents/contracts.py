"""Small, aggregate-safe artifacts shared by specialist workers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTrace:
    agent: str
    action: str
    source_count: int = 0
    refined: bool = False
