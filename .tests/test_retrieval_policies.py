from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.parse import urlparse

import mediawiki_client as mediawiki_module
from mediawiki_client import MediaWikiClient
from quality_policy import SOURCE_POLICIES
from retrieval.relevance import is_low_value_page, result_relevance_score
from retrieval.mediawiki_retriever import MediaWikiRetriever
from retrieval.wiki_domains import is_probable_wiki_domain
from query_tokens import question_relevance_tokens


def test_wiki_discovery_accepts_hosted_and_independent_candidates() -> None:
    assert is_probable_wiki_domain("small-game.miraheze.org")
    assert is_probable_wiki_domain("guide.smallgame.example", url="https://guide.smallgame.example/wiki/Home")
    assert is_probable_wiki_domain("smallgame.wiki")
    assert not is_probable_wiki_domain("store.steampowered.com", url="https://store.steampowered.com/app/1")


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

    monkeypatch.setattr(mediawiki_module, "urlopen", fake_urlopen)

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
    query = MediaWikiRetriever._select_search_query(
        question="How do I open the final door?",
        aliases=["Final Door", "Required Key"],
        planned_queries=["hidden passage access prerequisite"],
    )

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


def test_mediawiki_removes_cjk_instructions_from_english_entity_query() -> None:
    query = MediaWikiRetriever._normalize_mixed_language_query(
        "如何进入 Apartment 12 并获取 Apt 35 Key 的准确位置"
    )

    assert query == "Apartment 12 Apt 35 Key"


def test_mediawiki_prioritizes_linked_entity_in_dependency_sentence() -> None:
    evidence = (
        "To open the final gate, retrieve the Relay Token from the Maintenance Annex. "
        "A Decorative Painting hangs nearby."
    ).casefold()

    dependency = MediaWikiRetriever._explicit_dependency_score("Maintenance Annex", evidence)
    decoration = MediaWikiRetriever._explicit_dependency_score("Decorative Painting", evidence)

    assert dependency > decoration


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
