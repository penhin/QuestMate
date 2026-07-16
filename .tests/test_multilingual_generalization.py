from types import SimpleNamespace

from ai.evidence_policy import has_question_specific_sources
from quality_policy import SOURCE_POLICIES
from query_tokens import question_relevance_tokens
from retrieval.mediawiki_retriever import MediaWikiRetriever
from retrieval.relevance import result_relevance_score
from schemas import Source


def test_unseen_cjk_and_hangul_predicates_do_not_swallow_the_entity() -> None:
    chinese_tokens = question_relevance_tokens("如何打开蓝色大门")
    korean_tokens = question_relevance_tokens("문스톤은 어디서 얻나요")
    japanese_tokens = question_relevance_tokens("ひかりはどこで使う")

    assert {"蓝色", "色大", "大门"}.issubset(chinese_tokens)
    assert {"문스", "스톤"}.issubset(korean_tokens)
    assert {"ひか", "かり"}.issubset(japanese_tokens)
    assert result_relevance_score(
        item={
            "title": "蓝色大门 | Example Game Wiki",
            "url": "https://example-game.fandom.com/wiki/Blue_Gate",
            "content": "Example Game 中，蓝色大门的解锁条件是完成日蚀仪式。",
        },
        game="Example Game",
        question="如何打开蓝色大门",
    ) > 0


def test_cjk_relationship_requires_both_endpoint_anchors() -> None:
    question = "超级琥珀能量继电器能打开蓝色大门吗？"
    partial = {
        "title": "超级琥珀能量继电器 | Example Game Wiki",
        "url": "https://example-game.fandom.com/wiki/Amber_Relay",
        "content": "Example Game 的超级琥珀能量继电器位于旧塔顶层。",
    }
    complete = {
        **partial,
        "content": "Example Game 中，超级琥珀能量继电器可以开启蓝色大门。",
    }

    # Without a structured planner signal, terminal-question edges remain soft
    # recall features: character position alone cannot distinguish a second
    # entity from a paraphrasable predicate.
    assert result_relevance_score(
        item=partial,
        game="Example Game",
        question=question,
    ) > 0
    assert result_relevance_score(
        item=partial,
        game="Example Game",
        question=question,
        required_entity_groups=[
            ["超级琥珀能量继电器"],
            ["蓝色大门"],
        ],
    ) == 0
    assert result_relevance_score(
        item=complete,
        game="Example Game",
        question=question,
        required_entity_groups=[
            ["超级琥珀能量继电器"],
            ["蓝色大门"],
        ],
    ) > 0
    assert result_relevance_score(
        item={
            "title": "문스톤 | Example Game Wiki",
            "url": "https://example-game.fandom.com/wiki/Moonstone",
            "content": "Example Game에서 문스톤 획득 위치는 수정 동굴입니다.",
        },
        game="Example Game",
        question="문스톤은 어디서 얻나요",
    ) > 0
    assert result_relevance_score(
        item={
            "title": "ひかり | Example Game Wiki",
            "url": "https://example-game.fandom.com/wiki/Hikari",
            "content": "Example Gameでは、ひかりを灯台の祭壇で使います。",
        },
        game="Example Game",
        question="ひかりはどこで使う",
    ) > 0


def test_cjk_binary_relation_requires_both_edges_without_an_action_table() -> None:
    entity_only = {
        "title": "超级琥珀核心继电器 | Example Game Wiki",
        "url": "https://example-game.fandom.com/wiki/Amber_Core_Relay",
        "content": "Example Game 的超级琥珀核心继电器是一件可在遗迹中获得的装置。",
    }
    complete_relation = {
        "title": "超级琥珀核心继电器与蓝色大门 | Example Game Wiki",
        "url": "https://example-game.fandom.com/wiki/Amber_Core_Relay",
        "content": "Example Game 中，蓝色大门由超级琥珀核心继电器供能后才会解锁。",
    }

    # The extraction is based on the two clause edges, so unseen predicates do
    # not need to be added to a verb list and may be paraphrased by the source.
    for question in (
        "超级琥珀核心继电器打开蓝色大门吗？",
        "超级琥珀核心继电器连接蓝色大门吗？",
        "超级琥珀核心继电器通向蓝色大门吗？",
    ):
        assert result_relevance_score(
            item=entity_only,
            game="Example Game",
            question=question,
            required_entity_groups=[
                ["超级琥珀核心继电器"],
                ["蓝色大门"],
            ],
        ) == 0
        assert result_relevance_score(
            item=complete_relation,
            game="Example Game",
            question=question,
            required_entity_groups=[
                ["超级琥珀核心继电器"],
                ["蓝色大门"],
            ],
        ) > 0


def test_cjk_single_entity_predicate_remains_a_soft_recall_signal() -> None:
    assert result_relevance_score(
        item={
            "title": "超级琥珀核心继电器 | Example Game Wiki",
            "url": "https://example-game.fandom.com/wiki/Amber_Core_Relay",
            "content": "Example Game 的超级琥珀核心继电器在夜晚会散发柔和光芒。",
        },
        game="Example Game",
        question="超级琥珀核心继电器发出荧光吗？",
    ) > 0


def test_cjk_entity_internal_modal_character_is_not_a_relation_boundary() -> None:
    entity_only = {
        "title": "超级能量继电器 | Example Game Wiki",
        "url": "https://example-game.fandom.com/wiki/Energy_Relay",
        "content": "Example Game 的超级能量继电器是一件古代装置。",
    }
    complete_relation = {
        **entity_only,
        "content": "Example Game 中，蓝色大门会响应超级能量继电器发出的信号。",
    }
    question = "超级能量继电器与蓝色大门有什么关系？"

    assert result_relevance_score(
        item=entity_only,
        game="Example Game",
        question=question,
    ) == 0
    assert result_relevance_score(
        item=complete_relation,
        game="Example Game",
        question=question,
    ) > 0


def test_direct_evidence_gate_uses_lowercase_entity_not_literal_predicate() -> None:
    direct = Source(
        title="Moonstone | Example Game Wiki",
        url="https://example-game.fandom.com/wiki/Moonstone",
        evidence="Moonstone can be found in the Crystal Cave.",
    )
    predicate_only = Source(
        title="Merchant item guide",
        url="https://example-game.fandom.com/wiki/Merchant",
        evidence="You can obtain items from this merchant.",
    )

    for question in (
        "Where can I obtain moonstone?",
        "Where can I find moonstone?",
        "Where is moonstone acquired?",
        "Where does moonstone drop?",
    ):
        assert has_question_specific_sources(question=question, sources=[direct])
        assert not has_question_specific_sources(question=question, sources=[predicate_only])


def test_mediawiki_mixed_language_normalization_never_drops_the_only_cjk_entity() -> None:
    for query in ("如何获得月光钥匙 DLC", "月光钥匙在哪里 v2.0", "月光钥匙 DLC Update"):
        normalized = MediaWikiRetriever._normalize_mixed_language_query(query)
        selected = MediaWikiRetriever._select_search_queries(
            question=query,
            aliases=[],
            planned_queries=[query],
        )

        assert "月光钥匙" in normalized
        assert any("月光钥匙" in candidate for candidate in selected)


async def test_mediawiki_filters_each_parallel_result_against_its_own_query() -> None:
    class Cache:
        def get(self, _key):
            return None

        def set(self, _key, _value):
            return None

    class Client:
        def search(self, *, domain, query, max_results):
            assert domain == "example-game.fandom.com"
            assert max_results == 4
            if "Moonstone" in query:
                return {
                    "results": [{
                        "title": "Moonstone | Example Game Wiki",
                        "url": "https://example-game.fandom.com/wiki/Moonstone",
                        "content": "In Example Game, Moonstone is found inside the Crystal Cave.",
                        "score": 0.9,
                    }]
                }
            return {
                "results": [{
                    "title": "Sun Key | Example Game Wiki",
                    "url": "https://example-game.fandom.com/wiki/Sun_Key",
                    "content": "In Example Game, use the Sun Key at the observatory lock.",
                    "score": 0.9,
                }]
            }

    retriever = MediaWikiRetriever(
        client=Client(),
        cache=Cache(),
        settings=SimpleNamespace(
            external_request_timeout_seconds=1,
            evidence_passage_max_chars=1600,
            wiki_link_expansion_pages_per_query=0,
            wiki_auto_index_enabled=False,
        ),
        source_policy=SOURCE_POLICIES["wiki"],
        content_index=None,
        best_passage=lambda content, **_kwargs: content,
        canonical_key=lambda url: url,
        extract_version=lambda _text: None,
    )

    sources = await retriever.search(
        game="Example Game",
        question="Answer both requested facts",
        aliases=[],
        planned_queries=["Where is Moonstone?", "How do I use the Sun Key?"],
        game_aliases=[],
        database_domains=["example-game.fandom.com"],
        max_results=4,
    )

    assert {source.title for source in sources} == {
        "Moonstone | Example Game Wiki",
        "Sun Key | Example Game Wiki",
    }


async def test_mediawiki_merges_distinct_passages_from_the_same_page() -> None:
    class Cache:
        def get(self, _key):
            return None

        def set(self, _key, _value):
            return None

    content = (
        "In Example Game, Moonstone is found inside the Crystal Cave. "
        "The Sun Key opens the observatory lock after the eclipse ritual."
    )

    class Client:
        def search(self, *, domain, query, max_results):
            return {
                "results": [{
                    "title": "Keys and stones | Example Game Wiki",
                    "url": "https://example-game.fandom.com/wiki/Keys_and_stones",
                    "content": content,
                    "score": 0.9,
                }]
            }

    def best_passage(_content: str, *, question: str, **_kwargs) -> str:
        if "Moonstone" in question:
            return "In Example Game, Moonstone is found inside the Crystal Cave."
        return "The Sun Key opens the observatory lock after the eclipse ritual."

    retriever = MediaWikiRetriever(
        client=Client(),
        cache=Cache(),
        settings=SimpleNamespace(
            external_request_timeout_seconds=1,
            evidence_passage_max_chars=1600,
            wiki_link_expansion_pages_per_query=0,
            wiki_auto_index_enabled=False,
        ),
        source_policy=SOURCE_POLICIES["wiki"],
        content_index=None,
        best_passage=best_passage,
        canonical_key=lambda url: url,
        extract_version=lambda _text: None,
    )

    sources = await retriever.search(
        game="Example Game",
        question="Answer both requested facts",
        aliases=[],
        planned_queries=["Where is Moonstone?", "How do I use the Sun Key?"],
        game_aliases=[],
        database_domains=["example-game.fandom.com"],
        max_results=4,
    )

    assert len(sources) == 1
    assert "Moonstone is found" in sources[0].evidence
    assert "Sun Key opens" in sources[0].evidence
