"""Request-scoped runtime context."""

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
