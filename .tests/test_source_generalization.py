from quality_policy import DEFAULT_DOMAIN_QUALITY, SOURCE_POLICIES, domain_quality
from retrieval.relevance import is_high_quality_source, result_relevance_score
from retrieval.source_builder import build_source
from retrieval.evidence_pool import rank_sources
from retrieval.wiki_domains import is_probable_wiki_domain
from schemas import Source


def test_direct_evidence_outranks_a_higher_trust_title_only_match() -> None:
    sources = rank_sources(
        sources=[
            Source(
                title="Moonstone acquired location guide",
                url="https://example.com/overview",
                evidence="General combat overview and enemy behavior.",
                score=0.98,
                trust_score=0.95,
            ),
            Source(
                title="Observatory route",
                url="https://example.com/route",
                evidence="Moonstone is acquired from the observatory chest.",
                score=0.5,
                trust_score=0.45,
            ),
        ],
        query="Where is Moonstone acquired?",
        intent="item_location",
        max_results=2,
    )

    assert [source.title for source in sources] == ["Observatory route", "Moonstone acquired location guide"]


def _build(item: dict[str, object], *, policy_name: str = "web"):
    return build_source(
        item=item,
        source_policy=SOURCE_POLICIES[policy_name],
        game="Example Adventure",
        game_aliases=[],
        question="Where is the Amber Relay?",
        intent="item_location",
        best_passage=lambda content, **_kwargs: content,
        evidence_max_chars=1600,
        version_safety_score=lambda **_kwargs: 0.75,
        extract_version=lambda _text: None,
        parse_datetime=lambda _value: None,
    )


def test_unknown_domain_with_direct_evidence_is_useful_without_inflated_authority() -> None:
    built = _build({
        "title": "Amber Relay location guide",
        "url": "https://independent-guides.example/amber-relay",
        "content": "In Example Adventure, the Amber Relay is beneath the observatory stairs.",
        "score": 0.8,
    })

    assert built is not None
    assert 0.4 <= built.source.trust_score <= 0.6
    assert built.source.score is not None and built.source.score >= 0.7


def test_keyword_repetition_does_not_turn_unknown_publisher_into_high_trust() -> None:
    built = _build({
        "title": "Amber Relay Amber Relay location",
        "url": "https://seo-pages.example/amber-relay",
        "content": (
            "Example Adventure Amber Relay location. "
            "Example Adventure Amber Relay location beneath the observatory stairs."
        ),
        "score": 0.95,
    })

    assert built is not None
    assert built.source.trust_score < 0.6


def test_direct_evidence_can_make_generic_unknown_title_strict() -> None:
    item = {
        "title": "Complete item walkthrough",
        "url": "https://long-tail.example/items/relay",
        "content": "Example Adventure players find the Amber Relay beneath the observatory stairs.",
        "score": 0.7,
    }

    assert is_high_quality_source(
        item=item,
        game="Example Adventure",
        question="Where is the Amber Relay?",
        source_type="web",
    )


def test_official_identity_without_supporting_passage_is_not_strict_evidence() -> None:
    item = {
        "title": "Example Adventure Amber Relay",
        "url": "https://store.steampowered.com/app/101/example-adventure",
        "content": "",
        "score": 0.9,
    }

    assert not is_high_quality_source(
        item=item,
        game="Example Adventure",
        question="Where is the Amber Relay?",
        source_type="official",
    )


def test_known_domain_is_not_a_substitute_for_page_evidence() -> None:
    superficial_known = {
        "title": "Amber Relay | Example Adventure Wiki",
        "url": "https://example-adventure.fandom.com/wiki/Amber_Relay",
        "content": "Example Adventure item index.",
        "score": 0.9,
    }
    direct_unknown = {
        "title": "Complete item walkthrough",
        "url": "https://long-tail.example/items/relay",
        "content": "Example Adventure players find the Amber Relay beneath the observatory stairs.",
        "score": 0.7,
    }

    assert not is_high_quality_source(
        item=superficial_known,
        game="Example Adventure",
        question="Where is the Amber Relay?",
        source_type="wiki",
    )
    assert is_high_quality_source(
        item=direct_unknown,
        game="Example Adventure",
        question="Where is the Amber Relay?",
        source_type="web",
    )


def test_unknown_domain_still_requires_correct_game_and_entity_evidence() -> None:
    wrong_game = {
        "title": "Amber Relay location",
        "url": "https://long-tail.example/amber-relay",
        "content": "A different adventure places the Amber Relay beside a tower.",
    }
    generic = {
        "title": "Example Adventure walkthrough",
        "url": "https://long-tail.example/walkthrough",
        "content": "A broad overview of Example Adventure characters and chapters.",
    }

    assert result_relevance_score(
        item=wrong_game,
        game="Example Adventure",
        question="Where is the Amber Relay?",
    ) == 0
    assert not is_high_quality_source(
        item=generic,
        game="Example Adventure",
        question="Where is the Amber Relay?",
        source_type="web",
    )


def test_query_bound_localized_result_can_be_a_relaxed_candidate_only() -> None:
    localized = {
        "title": "月光钥匙获取攻略",
        "url": "https://long-tail.example/moon-key",
        "content": "月光钥匙位于观星台的宝箱中。",
    }

    assert result_relevance_score(
        item=localized,
        game="Example Adventure",
        question="月光钥匙在哪里获得？",
    ) == 0
    assert result_relevance_score(
        item=localized,
        game="Example Adventure",
        question="月光钥匙在哪里获得？",
        query_confirms_game=True,
    ) > 0
    assert not is_high_quality_source(
        item=localized,
        game="Example Adventure",
        question="月光钥匙在哪里获得？",
        source_type="web",
    )


def test_paraphrased_relations_keep_recall_but_require_each_named_endpoint() -> None:
    acquired = {
        "title": "Example Adventure Excalibur",
        "url": "https://long-tail.example/excalibur",
        "content": "Obtain Excalibur from the lake in Example Adventure.",
    }
    full_relation = {
        "title": "Example Adventure Amber Relay and Blue Gate",
        "url": "https://long-tail.example/relay-gate",
        "content": "The Amber Relay opens the Blue Gate in Example Adventure.",
    }
    partial_relation = {
        "title": "Example Adventure Amber Relay",
        "url": "https://long-tail.example/relay",
        "content": "The Amber Relay is found below the stairs in Example Adventure.",
    }
    chinese_paraphrase = {
        "title": "Example Adventure 月光钥匙与蓝色大门",
        "url": "https://long-tail.example/moon-key",
        "content": "Example Adventure 中，月光钥匙用于开启蓝色大门。",
    }

    assert result_relevance_score(
        item=acquired,
        game="Example Adventure",
        question="Where is Excalibur acquired?",
    ) > 0
    assert result_relevance_score(
        item=full_relation,
        game="Example Adventure",
        question="Does the Amber Relay open the Blue Gate?",
    ) > 0
    assert result_relevance_score(
        item=partial_relation,
        game="Example Adventure",
        question="Does the Amber Relay open the Blue Gate?",
    ) == 0
    assert result_relevance_score(
        item=chinese_paraphrase,
        game="Example Adventure",
        question="月光钥匙能打开蓝色大门吗？",
    ) > 0


def test_known_guide_domains_are_small_priors_not_required_membership() -> None:
    assert domain_quality("gamefaqs.gamespot.com") > DEFAULT_DOMAIN_QUALITY
    assert domain_quality("guides.neoseeker.com") > DEFAULT_DOMAIN_QUALITY
    assert domain_quality("fandom.com.attacker.example") == DEFAULT_DOMAIN_QUALITY
    assert domain_quality("unknown-guide.example") == DEFAULT_DOMAIN_QUALITY


def test_only_unambiguous_independent_mediawiki_paths_are_probe_candidates() -> None:
    assert is_probable_wiki_domain(
        "archive.example",
        url="https://archive.example/w/index.php?title=Main_Page",
    )
    assert not is_probable_wiki_domain(
        "archive.example",
        url="https://archive.example/api.php?action=query",
    )


def test_source_policy_offers_open_wiki_discovery_templates() -> None:
    templates = SOURCE_POLICIES["wiki"].query_templates
    assert any("site:" not in template and "wiki" in template for template in templates)
