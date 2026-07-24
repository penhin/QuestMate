"""Small provider circuit breaker with aggregate-safe state."""

from time import monotonic


class ProviderHealth:
    def __init__(self, *, cooldown_seconds: int) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._retry_after: dict[str, float] = {}

    def available(self, provider: str) -> bool:
        return monotonic() >= self._retry_after.get(provider, 0)

    def failed(self, provider: str) -> None:
        self._retry_after[provider] = monotonic() + self.cooldown_seconds

    def succeeded(self, provider: str) -> None:
        self._retry_after.pop(provider, None)
