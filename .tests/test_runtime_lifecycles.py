from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import asyncio
import time

import pytest
import search_cache as search_cache_module
import tasks as tasks_module
from agent import QuestAgent
from knowledge import KnowledgeStore, knowledge_store
from schemas import InvestigationState, SearchPlan
from search_cache import CachedSearchClient, RedisSearchCache, TTLSearchCache
from source_registry import game_source_registry
from storage import conversation_store, shared_database


def test_api_stores_share_one_database_pool() -> None:
    assert conversation_store.database is shared_database
    assert knowledge_store.database is shared_database
    assert game_source_registry.database is shared_database


def test_celery_index_task_uses_and_closes_a_fresh_store_per_event_loop(monkeypatch) -> None:
    stores = []

    class RecordingStore:
        def __init__(self) -> None:
            self.closed = False
            self.calls = []
            stores.append(self)

        async def index_url(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "ready", "document_id": f"doc-{len(stores)}"}

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(tasks_module, "KnowledgeStore", RecordingStore)

    first = tasks_module.index_url.run(
        "https://guides.example/one",
        "Example Game",
        published_at="2026-07-16T12:00:00Z",
    )
    second = tasks_module.index_url.run(
        "https://guides.example/two",
        "Example Game",
    )

    assert first["document_id"] == "doc-1"
    assert second["document_id"] == "doc-2"
    assert len(stores) == 2
    assert all(store.closed for store in stores)
    assert stores[0].calls[0]["url"] == "https://guides.example/one"
    assert stores[0].calls[0]["published_at"].isoformat() == "2026-07-16T12:00:00+00:00"


async def test_owned_knowledge_store_closes_embedding_and_database_pools() -> None:
    closed = []

    class Embeddings:
        async def aclose(self) -> None:
            closed.append("embeddings")

    class Engine:
        async def dispose(self) -> None:
            closed.append("database")

    store = object.__new__(KnowledgeStore)
    store.embeddings = Embeddings()
    store.database = type("DatabaseStub", (), {"engine": Engine()})()
    store._owns_database = True

    await store.aclose()

    assert closed == ["embeddings", "database"]


def test_redis_failure_opens_short_circuit_and_skips_repeated_timeouts() -> None:
    class FailingRedis:
        def __init__(self) -> None:
            self.get_calls = 0
            self.set_calls = 0

        def get(self, _key):
            self.get_calls += 1
            raise TimeoutError("redis unavailable")

        def set(self, *_args, **_kwargs):
            self.set_calls += 1
            raise TimeoutError("redis unavailable")

    redis = FailingRedis()
    cache = RedisSearchCache(
        redis_url="redis://localhost:6379/0",
        fallback=TTLSearchCache(ttl_seconds=60, max_entries=16),
        failure_cooldown_seconds=30,
    )
    cache._redis = redis

    assert cache.get("first-miss") is None
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(cache.get, [f"later-miss-{index}" for index in range(8)])) == [None] * 8
    cache.set("write-during-outage", {"results": []})

    assert redis.get_calls == 1
    assert redis.set_calls == 0
    assert cache.get("write-during-outage") == {"results": []}


def test_redis_circuit_allows_one_recovery_probe(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(search_cache_module, "monotonic", lambda: now[0])

    class FailingRedis:
        def get(self, _key):
            raise TimeoutError("redis unavailable")

    class HealthyRedis:
        def __init__(self) -> None:
            self.get_calls = 0

        def get(self, _key):
            self.get_calls += 1
            return None

    cache = RedisSearchCache(
        redis_url="redis://localhost:6379/0",
        fallback=TTLSearchCache(ttl_seconds=60, max_entries=16),
        failure_cooldown_seconds=5,
    )
    cache._redis = FailingRedis()
    assert cache.get("opens-circuit") is None

    healthy = HealthyRedis()
    cache._redis = healthy
    now[0] = 104
    assert cache.get("still-open") is None
    assert healthy.get_calls == 0

    now[0] = 106
    assert cache.get("half-open-probe") is None
    assert cache.get("closed-again") is None
    assert healthy.get_calls == 2


def test_identical_concurrent_search_misses_share_one_upstream_call() -> None:
    workers = 8
    start = Barrier(workers)

    class SlowSearch:
        def __init__(self) -> None:
            self.calls = 0
            self.lock = Lock()

        def search(self, **_kwargs):
            with self.lock:
                self.calls += 1
            time.sleep(0.05)
            return {"results": [{"title": "shared"}]}

    upstream = SlowSearch()
    client = CachedSearchClient(
        upstream,
        TTLSearchCache(ttl_seconds=60, max_entries=16),
    )

    def run(_index: int):
        start.wait()
        return client.search(query="same query", max_results=5)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, range(workers)))

    assert results == [{"results": [{"title": "shared"}]}] * workers
    assert upstream.calls == 1
    assert client.upstream_calls == 1
    assert client.cache_hits == workers - 1


def test_search_usage_scope_reports_request_local_cost_only() -> None:
    class Search:
        def search(self, **_kwargs):
            return {"results": []}

    client = CachedSearchClient(Search(), TTLSearchCache(ttl_seconds=60, max_entries=4))

    with client.usage_scope():
        client.search(query="synthetic", max_results=2)
        client.search(query="synthetic", max_results=2)
        assert client.request_usage() == {"tavily_paid_calls": 1, "tavily_cache_hits": 1}

    assert client.request_usage() == {"tavily_paid_calls": 0, "tavily_cache_hits": 0}


def test_search_usage_scope_enforces_paid_call_budget() -> None:
    class Search:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, **_kwargs):
            self.calls += 1
            return {"results": [{"title": "result"}]}

    upstream = Search()
    client = CachedSearchClient(upstream, TTLSearchCache(ttl_seconds=60, max_entries=4))

    with client.usage_scope(max_paid_calls=1):
        first = client.search(query="synthetic-one", max_results=2)
        second = client.search(query="synthetic-two", max_results=2)
        assert first["results"]
        assert second == {"results": []}
        assert client.request_usage()["tavily_paid_calls"] == 1

    assert upstream.calls == 1


def test_three_model_calls_are_reported_as_a_complex_evidence_path() -> None:
    class Search:
        def usage_snapshot(self):
            return {"tavily_paid_calls": 0, "tavily_cache_hits": 0}

    class LLM:
        def request_usage(self):
            return {"model_calls": 3}

    agent = object.__new__(QuestAgent)
    agent.search_provider = Search()
    agent.llm = LLM()

    usage = agent._request_usage(
        {"tavily_paid_calls": 0, "tavily_cache_hits": 0},
        InvestigationState(goal="Verify a conditional relationship"),
        SearchPlan(),
    )

    assert usage["model_calls"] == 3
    assert usage["complex_evidence_path"] == 1


async def test_cancelled_background_index_is_marked_failed() -> None:
    store = object.__new__(KnowledgeStore)
    marked = []

    async def init_schema():
        return None

    async def is_fresh(_url):
        return False

    async def upsert(**_kwargs):
        return "document-1"

    async def persist(**_kwargs):
        raise asyncio.CancelledError()

    async def mark_failed(**kwargs):
        marked.append(kwargs)

    store.init_schema = init_schema
    store._is_fresh = is_fresh
    store._upsert_document = upsert
    store._persist_content = persist
    store._mark_failed = mark_failed

    with pytest.raises(asyncio.CancelledError):
        await store.index_content(
            url="https://example.com/wiki/item",
            game="Example Game",
            content="content",
        )

    assert marked and marked[0]["document_id"] == "document-1"
