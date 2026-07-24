import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.parse import urlparse

import mediawiki_client as mediawiki_module
from mediawiki_client import MediaWikiClient
from quality_policy import SOURCE_POLICIES
from retrieval.relevance import is_low_value_page, result_relevance_score
from retrieval.mediawiki_retriever import MAX_BACKGROUND_INDEX_BATCHES, MediaWikiRetriever
from retrieval.wiki_domains import is_probable_wiki_domain, is_safe_wiki_host
from query_tokens import question_relevance_tokens
from schemas import Source


def test_wiki_discovery_accepts_hosted_and_independent_candidates() -> None:
    assert is_probable_wiki_domain("small-game.miraheze.org")
    assert is_probable_wiki_domain("guide.smallgame.example", url="https://guide.smallgame.example/wiki/Home")
    assert is_probable_wiki_domain("smallgame.wiki")
    assert not is_probable_wiki_domain("store.steampowered.com", url="https://store.steampowered.com/app/1")


def test_wiki_discovery_rejects_ambiguous_endpoints_and_unsafe_hosts() -> None:
    assert not is_probable_wiki_domain(
        "archive.example",
        url="https://archive.example/api.php?action=query",
    )
    assert not is_probable_wiki_domain(
        "archive.example",
        url="https://archive.example/index.php?title=Main_Page",
    )
    assert not is_probable_wiki_domain(
        "archive.example",
        url="https://different.example/wiki/Main_Page",
    )
    for host in (
        "localhost",
        "127.0.0.1",
        "[::1]",
        "metadata.service.internal",
        "public.example@127.0.0.1",
    ):
        assert not is_safe_wiki_host(host)


def test_reddit_index_filter_is_not_bound_to_a_specific_game_community() -> None:
    assert is_low_value_page(
        text="Community https://www.reddit.com/r/ObscureGame/",
        question="Where is the key?",
    )
    assert not is_low_value_page(
        text="Key route https://www.reddit.com/r/ObscureGame/comments/abc/key_route",
        question="Where is the key?",
    )


def test_relevance_policy_generalizes_to_an_unseen_game_name() -> None:
    score = result_relevance_score(
        item={
            "title": "Hidden Key | Example Obscure Game Wiki",
            "url": "https://example-obscure-game.wiki/wiki/Hidden_Key",
            "content": "The Hidden Key is behind the library wall in Example Obscure Game.",
        },
        game="Example Obscure Game",
        question="Where is the Hidden Key?",
    )
    assert score > 0


def test_production_retrieval_policies_do_not_name_fixture_games() -> None:
    production_text = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for root in (Path("retrieval"), Path("ai"))
        for path in root.glob("*.py")
    )
    for fixture_name in ("elden ring", "nightreign", "look outside", "goose goose duck"):
        assert fixture_name not in production_text


def test_mediawiki_client_probes_independent_wiki_api_paths(monkeypatch) -> None:
    calls: list[str] = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if urlparse(request.full_url).path == "/api.php":
            raise URLError("root API unavailable")
        return Response(b'{"query":{"pages":[]}}')

    monkeypatch.setattr(mediawiki_module, "_open_url", fake_urlopen)

    payload = MediaWikiClient()._request_payload(domain="independent.example", params="action=query")

    assert payload == {"query": {"pages": []}}
    assert any("/api.php" in url for url in calls)
    assert any("/w/api.php" in url for url in calls)


def test_short_entity_nouns_remain_relevance_tokens() -> None:
    assert "rod" in question_relevance_tokens("How do I use the Void Rod?")
    assert "key" in question_relevance_tokens("Where is the hidden key?")


def test_mediawiki_client_keeps_late_page_sections_for_passage_selection(monkeypatch) -> None:
    client = MediaWikiClient()
    long_content = "Early section. " * 600 + "Late dependency: hidden route requires the relay token."
    monkeypatch.setattr(
        client,
        "_request_payload",
        lambda **kwargs: {
            "query": {
                "pages": [{
                    "index": 1,
                    "title": "Long Guide",
                    "revisions": [{"slots": {"main": {"content": long_content}}}],
                }]
            }
        },
    )

    result = client.search(domain="long-guide.example", query="relay token", max_results=1)

    assert "Late dependency" in result["results"][0]["content"]


def test_mediawiki_refinement_query_takes_priority_over_accumulated_aliases() -> None:
    query = MediaWikiRetriever._select_search_queries(
        question="How do I open the final door?",
        aliases=["Final Door", "Required Key"],
        planned_queries=["hidden passage access prerequisite"],
    )[0]

    assert query == "hidden passage access prerequisite"


def test_mediawiki_can_probe_two_distinct_gaps_in_parallel() -> None:
    queries = MediaWikiRetriever._select_search_queries(
        question="How do I open the final door?",
        aliases=["Final Door"],
        planned_queries=["required key location", "hidden passage access", "optional lore"],
    )

    assert queries == ["required key location", "hidden passage access"]


def test_mediawiki_prioritizes_domain_matching_game_identity() -> None:
    domains = MediaWikiRetriever._rank_database_domains(
        ["en.wikipedia.org", "general-pc-wiki.example", "unseenpuzzlegame.miraheze.org"],
        game="Unseen Puzzle Game",
    )

    assert domains[0] == "unseenpuzzlegame.miraheze.org"


def test_mediawiki_preserves_mixed_language_query_without_instruction_vocabulary() -> None:
    query = MediaWikiRetriever._normalize_mixed_language_query(
        "如何进入 Apartment 12 并获取 Apt 35 Key 的准确位置"
    )

    assert "Apartment 12" in query
    assert "Apt 35 Key" in query
    assert "如何进入" in query


def test_mediawiki_prioritizes_link_by_generic_local_query_relevance() -> None:
    evidence = (
        "To open the final gate, retrieve the Relay Token from the Maintenance Annex. "
        "A Decorative Painting hangs nearby."
    ).casefold()

    relevant = MediaWikiRetriever._linked_entity_relevance_score(
        "Maintenance Annex",
        focused_evidence=evidence,
        query_tokens=["final", "gate"],
    )
    decoration = MediaWikiRetriever._linked_entity_relevance_score(
        "Decorative Painting",
        focused_evidence=evidence,
        query_tokens=["final", "gate"],
    )

    assert relevant > decoration


async def test_mediawiki_rejects_private_host_before_calling_adapter() -> None:
    class RecordingClient:
        def __init__(self):
            self.calls = 0

        def search(self, **kwargs):
            self.calls += 1
            return {"results": []}

    class EmptyCache:
        def get(self, key):
            return None

        def set(self, key, value):
            return None

    client = RecordingClient()
    retriever = MediaWikiRetriever(
        client=client,
        cache=EmptyCache(),
        settings=SimpleNamespace(external_request_timeout_seconds=1),
        source_policy=SOURCE_POLICIES["wiki"],
        content_index=None,
        best_passage=lambda content, **kwargs: content,
        canonical_key=lambda url: url,
        extract_version=lambda text: None,
    )

    result = await retriever._fetch_search("127.0.0.1", "entity", 3)

    assert result["results"] == []
    assert client.calls == 0


async def test_mediawiki_auto_index_is_deduplicated_bounded_and_cancelled() -> None:
    class BlockingIndex:
        def __init__(self):
            self.calls = 0
            self.started = asyncio.Event()

        async def index_content(self, **kwargs):
            self.calls += 1
            self.started.set()
            await asyncio.Event().wait()

    class EmptyCache:
        def get(self, key):
            return None

        def set(self, key, value):
            return None

    index = BlockingIndex()
    retriever = MediaWikiRetriever(
        client=object(),
        cache=EmptyCache(),
        settings=SimpleNamespace(
            external_request_timeout_seconds=1,
            wiki_auto_index_enabled=True,
            wiki_auto_index_pages_per_query=1,
        ),
        source_policy=SOURCE_POLICIES["wiki"],
        content_index=index,
        best_passage=lambda content, **kwargs: content,
        canonical_key=lambda url: url,
        extract_version=lambda text: None,
    )

    first = Source(title="Page 0", url="https://game.wiki/wiki/Page_0")
    retriever._schedule_auto_index(game="Game", sources=[first], content_by_url={str(first.url): "zero"})
    retriever._schedule_auto_index(game="Game", sources=[first], content_by_url={str(first.url): "zero"})
    for index_number in range(1, MAX_BACKGROUND_INDEX_BATCHES + 4):
        source = Source(title=f"Page {index_number}", url=f"https://game.wiki/wiki/Page_{index_number}")
        retriever._schedule_auto_index(
            game="Game",
            sources=[source],
            content_by_url={str(source.url): str(index_number)},
        )

    await asyncio.wait_for(index.started.wait(), timeout=1)
    assert len(retriever._background_index_tasks) == MAX_BACKGROUND_INDEX_BATCHES
    assert len(retriever._background_index_urls) == MAX_BACKGROUND_INDEX_BATCHES

    await retriever.wait_for_background_tasks(timeout_seconds=0.01)
    await asyncio.sleep(0)

    assert not retriever._background_index_tasks
    assert not retriever._background_index_urls
    assert index.calls <= 4


async def test_mediawiki_domain_failure_opens_temporary_circuit_breaker() -> None:
    class FailingClient:
        def __init__(self):
            self.calls = 0

        def search(self, **kwargs):
            self.calls += 1
            raise OSError("wiki unavailable")

    class EmptyCache:
        def get(self, key):
            return None

        def set(self, key, value):
            return None

    client = FailingClient()
    retriever = MediaWikiRetriever(
        client=client,
        cache=EmptyCache(),
        settings=SimpleNamespace(external_request_timeout_seconds=1),
        source_policy=SOURCE_POLICIES["wiki"],
        content_index=None,
        best_passage=lambda content, **kwargs: content,
        canonical_key=lambda url: url,
        extract_version=lambda text: None,
    )

    first = await retriever._fetch_search("unavailable-wiki.example", "entity", 3)
    second = await retriever._fetch_search("unavailable-wiki.example", "another entity", 3)

    assert first["results"] == second["results"] == []
    assert client.calls == 1
