from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

import mediawiki_client as mediawiki_module
from mediawiki_client import MediaWikiClient
from retrieval.relevance import is_low_value_page, result_relevance_score
from retrieval.wiki_domains import is_probable_wiki_domain


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
