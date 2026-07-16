"""Cache implementations and usage counters for external search clients."""

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from hashlib import sha256
import json
from threading import Lock
from time import monotonic
from typing import Any

from redis import Redis


class TTLSearchCache:
    """Small thread-safe LRU cache for paid search responses."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._values: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        if self.ttl_seconds <= 0:
            return None
        now = monotonic()
        with self._lock:
            cached = self._values.get(key)
            if cached is None:
                return None
            expires_at, value = cached
            if expires_at <= now:
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._values[key] = (monotonic() + self.ttl_seconds, deepcopy(value))
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)


class RedisSearchCache:
    """Persistent JSON cache with a local fallback when Redis is unavailable."""

    def __init__(
        self,
        *,
        redis_url: str,
        fallback: TTLSearchCache,
        failure_cooldown_seconds: float = 30.0,
    ) -> None:
        self._redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        self._fallback = fallback
        self._failure_cooldown_seconds = max(0.0, failure_cooldown_seconds)
        self._circuit_lock = Lock()
        self._retry_after = 0.0
        self._half_open_probe = False

    @staticmethod
    def _redis_key(key: str) -> str:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return f"questmate:search:v1:{digest}"

    def get(self, key: str) -> Any | None:
        if self._fallback.ttl_seconds <= 0:
            return None
        local = self._fallback.get(key)
        if local is not None:
            return local
        if not self._begin_redis_attempt():
            return None
        try:
            payload = self._redis.get(self._redis_key(key))
        except Exception:
            self._record_redis_failure()
            return None
        self._record_redis_success()
        if payload is None:
            return None
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
        self._fallback.set(key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        self._fallback.set(key, value)
        if self._fallback.ttl_seconds <= 0:
            return
        if not self._begin_redis_attempt():
            return
        try:
            self._redis.set(
                self._redis_key(key),
                json.dumps(value, ensure_ascii=False, default=str),
                ex=self._fallback.ttl_seconds,
            )
        except Exception:
            self._record_redis_failure()
            return
        self._record_redis_success()

    def _begin_redis_attempt(self) -> bool:
        """Allow normal traffic while closed and one probe after a failure."""
        with self._circuit_lock:
            if self._retry_after == 0:
                return True
            if monotonic() < self._retry_after or self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def _record_redis_failure(self) -> None:
        with self._circuit_lock:
            self._retry_after = monotonic() + self._failure_cooldown_seconds
            self._half_open_probe = False

    def _record_redis_success(self) -> None:
        with self._circuit_lock:
            self._retry_after = 0.0
            self._half_open_probe = False


class CachedSearchClient:
    """Cache identical search calls and expose credit-relevant counters."""

    def __init__(self, client: Any, cache: Any) -> None:
        self._client = client
        self._cache = cache
        self.upstream_calls = 0
        self.cache_hits = 0
        self._flight_lock = Lock()
        self._flights: dict[str, Future[dict[str, Any]]] = {}

    def search(self, **kwargs: Any) -> dict[str, Any]:
        key = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
        cached = self._cache.get(key)
        if cached is not None:
            self._increment_cache_hits()
            return cached

        with self._flight_lock:
            flight = self._flights.get(key)
            if flight is None:
                flight = Future()
                self._flights[key] = flight
                leader = True
            else:
                leader = False

        if not leader:
            result = flight.result()
            self._increment_cache_hits()
            return deepcopy(result)

        try:
            # A preceding leader may have populated the cache between our
            # optimistic read and registration of this flight.
            cached = self._cache.get(key)
            if cached is not None:
                self._increment_cache_hits()
                flight.set_result(cached)
                return cached
            result = self._client.search(**kwargs)
            with self._flight_lock:
                self.upstream_calls += 1
            self._cache.set(key, result)
            flight.set_result(deepcopy(result))
            return result
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        finally:
            with self._flight_lock:
                if self._flights.get(key) is flight:
                    self._flights.pop(key, None)

    def _increment_cache_hits(self) -> None:
        with self._flight_lock:
            self.cache_hits += 1
