"""Cache implementations and usage counters for external search clients."""

from collections import OrderedDict
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

    def __init__(self, *, redis_url: str, fallback: TTLSearchCache) -> None:
        self._redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        self._fallback = fallback

    @staticmethod
    def _redis_key(key: str) -> str:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return f"questmate:search:v1:{digest}"

    def get(self, key: str) -> Any | None:
        local = self._fallback.get(key)
        if local is not None:
            return local
        try:
            payload = self._redis.get(self._redis_key(key))
            if payload is None:
                return None
            value = json.loads(payload)
        except Exception:
            return None
        self._fallback.set(key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        self._fallback.set(key, value)
        if self._fallback.ttl_seconds <= 0:
            return
        try:
            self._redis.setex(
                self._redis_key(key),
                self._fallback.ttl_seconds,
                json.dumps(value, ensure_ascii=False, default=str),
            )
        except Exception:
            return


class CachedSearchClient:
    """Cache identical search calls and expose credit-relevant counters."""

    def __init__(self, client: Any, cache: Any) -> None:
        self._client = client
        self._cache = cache
        self.upstream_calls = 0
        self.cache_hits = 0

    def search(self, **kwargs: Any) -> dict[str, Any]:
        key = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        result = self._client.search(**kwargs)
        self.upstream_calls += 1
        self._cache.set(key, result)
        return result
