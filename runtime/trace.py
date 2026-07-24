"""Aggregate-safe execution trace, with no prompt or evidence payloads."""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(frozen=True)
class TraceEvent:
    name: str
    duration_ms: int
    outcome: str = "ok"


@dataclass
class RuntimeTrace:
    started_at: float = field(default_factory=perf_counter)
    events: list[TraceEvent] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)

    def record(self, name: str, started: float, *, outcome: str = "ok") -> None:
        self.events.append(TraceEvent(name, round((perf_counter() - started) * 1000), outcome))

    def record_usage(self, usage: dict[str, int]) -> None:
        self.token_usage = {
            key: max(0, int(value))
            for key, value in usage.items()
            if isinstance(value, int)
        }
