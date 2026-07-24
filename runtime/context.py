"""Request-scoped runtime context."""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4

from runtime.trace import RuntimeTrace


@dataclass
class AgentContext:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    tools: Mapping[str, Any] = field(default_factory=dict)
    memory: Any = None
    trace: RuntimeTrace = field(default_factory=RuntimeTrace)


_active_context: ContextVar[AgentContext | None] = ContextVar("questmate_agent_context", default=None)


def activate_context(context: AgentContext):
    return _active_context.set(context)


def deactivate_context(token) -> None:
    _active_context.reset(token)


def active_context() -> AgentContext | None:
    return _active_context.get()
