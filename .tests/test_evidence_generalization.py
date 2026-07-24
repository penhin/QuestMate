import json

import pytest

from ai.investigation import ensure_investigation_query, parse_investigation_state
from agents.identity_resolution import IdentityResolver
from config import Settings
from game_resolution import (
    GameResolver,
    identity_matches_game,
    is_platform_product_url,
    is_supported_platform_domain,
    matches_game_text,
    select_game_candidate,
)
from llm import GuideLLM
from quality_policy import SOURCE_POLICIES
from retrieval.source_quality import matches_game_identity
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, Source
from search import TavilySearchProvider


def test_identity_and_entity_tokens_support_non_latin_game_titles() -> None:
    from query_tokens import question_relevance_tokens

    assert identity_matches_game(
        title="九日 Nine Sols — 공식 게임",
        url="https://example.com/nine-sols",
        game="九日 Nine Sols",
    )
    assert identity_matches_game(
        title="Диско Элизиум — официальный сайт",
        url="https://example.com/disco",
        game="Диско Элизиум",
    )
    assert "엘리베이터" in question_relevance_tokens("엘리베이터 잠금 해제 방법")
    japanese_tokens = question_relevance_tokens("月の鍵はどこで使う？")
    assert "月の鍵" in japanese_tokens
    assert "はどこで" in japanese_tokens
    assert is_supported_platform_domain("store.steampowered.com")
    assert not is_supported_platform_domain("steampowered.com.attacker.example")
    assert not matches_game_text(
        text="A character watches the scenery outside the window.",
        game="Look Outside",
        game_aliases=[],
    )
    assert matches_game_identity(
        text="Nine Sols game guide for Eigong",
        game_names=["九日 Nine Sols"],
    )
    assert matches_game_identity(
        text="Niche Quest Wiki and walkthrough",
        game_names=["niche quest"],
    )
    assert matches_game_identity(
        text="niche quest wiki and walkthrough",
        game_names=["Niche Quest"],
    )
    assert matches_game_identity(
        text="ドラゴンクエストIII ゲーム攻略",
        game_names=["ドラゴンクエスト III"],
    )
    assert not matches_game_identity(
        text="九日天气预报与气候趋势",
        game_names=["九日 Nine Sols"],
    )
    for sequel, earlier_page in (
        ("Hades II", "Hades boon guide"),
        ("Portal 2", "Portal walkthrough"),
        ("The Witcher 3", "The Witcher quest guide"),
        ("Risk of Rain 2", "Risk of Rain item guide"),
    ):
        assert not matches_game_identity(text=earlier_page, game_names=[sequel])
    for roman_title, numeric_page in (
        ("Hades II", "Hades 2 boon guide"),
        ("Dragon Quest III", "Dragon Quest 3 walkthrough"),
        ("Civilization IV", "Civilization 4 strategy"),
    ):
        assert matches_game_identity(text=numeric_page, game_names=[roman_title])
    for title, unrelated in (
        ("Control", "controller setup instructions"),
        ("Inside", "look inside the container"),
        ("Rust", "trust settings reference"),
        ("It", "it may happen later"),
    ):
        assert not matches_game_identity(text=unrelated, game_names=[title])
    assert not matches_game_identity(
        text="Control https://fandom.com.attacker.example/control",
        game_names=["Control"],
    )


def test_game_identity_only_merges_cross_store_candidates_with_explicit_alias_link() -> None:
    class StoreResults:
        def __init__(self, results: list[dict[str, object]]) -> None:
            self.results = results

        def search(self, **_kwargs):
            return {"results": self.results}

    mirrors = GameResolver(StoreResults([
        {
            "title": "Mirror Quest on Steam",
            "url": "https://store.steampowered.com/app/101/Mirror_Quest/",
            "content": "Mirror Quest is an adventure game.",
            "score": 0.9,
        },
        {
            "title": "Mirror Quest on GOG.com",
            "url": "https://www.gog.com/en/game/mirror_quest",
            "content": "Mirror Quest is an adventure game.",
            "score": 0.85,
        },
    ])).discover_game_identity(game="Mirror Quest")

    assert len(mirrors.candidates) == 2
    assert mirrors.ambiguous

    bilingual = GameResolver(StoreResults([
        {
            "title": "九日 Nine Sols on Steam",
            "url": "https://store.steampowered.com/app/1809540/Nine_Sols/",
            "content": "九日 Nine Sols is an action adventure.",
            "score": 0.9,
        },
        {
            "title": "Nine Sols on GOG.com",
            "url": "https://www.gog.com/en/game/nine_sols",
            "content": "九日 Nine Sols is an action adventure.",
            "score": 0.88,
        },
    ])).discover_game_identity(game="九日 Nine Sols")

    assert len(bilingual.candidates) == 1
    assert len(bilingual.candidates[0].platform_urls) == 2
    assert not bilingual.ambiguous

    collision = GameResolver(StoreResults([
        {
            "title": "Shared Name on Steam",
            "url": "https://store.steampowered.com/app/201/Shared_Name/",
            "content": "Shared Name is a video game.",
            "score": 0.9,
        },
        {
            "title": "Shared Name on Steam",
            "url": "https://store.steampowered.com/app/202/Shared_Name/",
            "content": "Shared Name is another video game.",
            "score": 0.9,
        },
    ])).discover_game_identity(game="Shared Name")

    assert len(collision.candidates) == 2
    assert collision.ambiguous

    selected = select_game_candidate(
        collision,
        selected_url="https://store.steampowered.com/app/202/Shared_Name/",
    )
    assert selected is not None and selected.is_confirmed
    assert [str(url) for url in selected.platform_urls] == [
        "https://store.steampowered.com/app/202/Shared_Name/"
    ]
    assert not selected.ambiguous


def test_platform_identity_rejects_catalog_and_tag_pages() -> None:
    assert is_platform_product_url("https://store.steampowered.com/app/101/Mirror_Quest/")
    assert is_platform_product_url("https://studio.itch.io/mirror-quest")
    assert not is_platform_product_url("https://itch.io/games/tag-echo")
    assert not is_platform_product_url("https://studio.itch.io/games/tag-echo")

    class CatalogResult:
        def search(self, **_kwargs):
            return {
                "results": [{
                    "title": "Top games tagged Echo - itch.io",
                    "url": "https://itch.io/games/tag-echo",
                    "content": "Echo and other browser games.",
                    "score": 0.95,
                }]
            }

    resolution = GameResolver(CatalogResult()).discover_game_identity(game="Echo")
    assert resolution.platform_urls == []
    assert resolution.candidates == []
    assert not resolution.is_confirmed


def test_google_play_product_identity_includes_application_id() -> None:
    from game_resolution import same_platform_resource

    first = "https://play.google.com/store/apps/details?id=com.example.first"
    second = "https://play.google.com/store/apps/details?id=com.example.second"

    assert same_platform_resource(first, first)
    assert not same_platform_resource(first, second)


def test_store_product_identity_ignores_display_slug_variants() -> None:
    from game_resolution import same_platform_resource

    assert same_platform_resource(
        "https://store.steampowered.com/app/123/Game_Name/",
        "https://store.steampowered.com/app/123/",
    )


def test_official_site_identity_supports_games_outside_known_storefronts() -> None:
    class OfficialResult:
        def search(self, **_kwargs):
            return {
                "results": [{
                    "title": "Niche Quest - Official Website",
                    "url": "https://niche-quest.example/home",
                    "content": "Niche Quest official game website from the developer.",
                    "score": 0.9,
                }]
            }

    resolution = GameResolver(OfficialResult()).discover_game_identity(game="Niche Quest")

    assert resolution.platform_urls == []
    assert resolution.official_urls == []
    assert [str(url) for url in resolution.identity_urls] == [
        "https://niche-quest.example/home"
    ]
    assert resolution.candidates[0].name == "Niche Quest"
    assert not resolution.is_confirmed


def test_seo_copy_cannot_self_declare_an_official_authority() -> None:
    from game_resolution import is_generic_official_identity_result

    assert not is_generic_official_identity_result(
        item={
            "title": "Niche Quest",
            "url": "https://seo-attacker.example/niche-quest",
            "content": "Niche Quest developer and publisher information; official game guide.",
        },
        game="Niche Quest",
    )


def test_unverified_official_copy_and_unprobed_wiki_cannot_auto_confirm_game() -> None:
    class AttackerResults:
        def search(self, **_kwargs):
            return {
                "results": [
                    {
                        "title": "Niche Quest - Official Website",
                        "url": "https://seo-attacker.example/niche-quest",
                        "content": "Niche Quest official game website from the developer.",
                        "score": 0.99,
                    },
                    {
                        "title": "Niche Quest Wiki",
                        "url": "https://attacker-wiki.example/wiki/Niche_Quest",
                        "content": "Niche Quest community wiki database.",
                        "score": 0.99,
                    },
                ]
            }

    resolution = GameResolver(AttackerResults()).resolve(game="Niche Quest")

    assert not resolution.is_confirmed
    assert resolution.official_urls == []
    assert [str(url) for url in resolution.identity_urls] == [
        "https://seo-attacker.example/niche-quest"
    ]


def test_store_marketing_copy_does_not_replace_exact_requested_title() -> None:
    class StoreResult:
        def search(self, **_kwargs):
            return {
                "results": [{
                    "title": "Alan Wake 2 | Download and Buy Today - Epic Games Store",
                    "url": "https://store.epicgames.com/en-US/p/alan-wake-2",
                    "content": "Alan Wake 2 is an official horror game.",
                    "score": 0.9,
                }]
            }

    resolution = GameResolver(StoreResult()).discover_game_identity(game="Alan Wake 2")

    assert resolution.confirmed_name == "Alan Wake 2"
    assert resolution.candidates[0].name == "Alan Wake 2"
    assert resolution.aliases == []


def test_result_source_type_is_classified_from_actual_domain() -> None:
    provider = TavilySearchProvider(
        settings=Settings(mediawiki_direct_search=False, search_cache_use_redis=False),
    )

    wiki = provider._effective_source_policy(
        configured=SOURCE_POLICIES["web"],
        url="https://small-game.fandom.com/wiki/Artifact",
        database_domains=[],
        official_domains=[],
    )
    community = provider._effective_source_policy(
        configured=SOURCE_POLICIES["web"],
        url="https://www.reddit.com/r/game/comments/123",
        database_domains=[],
        official_domains=[],
    )
    spoofed = provider._effective_source_policy(
        configured=SOURCE_POLICIES["official"],
        url="https://official.example.attacker.test/patch",
        database_domains=[],
        official_domains=["OFFICIAL.EXAMPLE"],
    )
    unknown_official = provider._effective_source_policy(
        configured=SOURCE_POLICIES["official"],
        url="https://random-blog.example/patch-notes",
        database_domains=[],
        official_domains=[],
    )
    known_official = provider._effective_source_policy(
        configured=SOURCE_POLICIES["web"],
        url="https://news.official.example/patch-notes",
        database_domains=[],
        official_domains=["official.example"],
    )
    name_only_wiki = provider._effective_source_policy(
        configured=SOURCE_POLICIES["wiki"],
        url="https://definitelynotawiki.example/article",
        database_domains=["definitelynotawiki.example"],
        official_domains=[],
    )

    assert wiki.source_type == "wiki"
    assert community.source_type == "community"
    assert spoofed.source_type == "web"
    assert unknown_official.source_type == "web"
    assert known_official.source_type == "official"
    assert name_only_wiki.source_type == "web"


def test_confirmed_game_request_does_not_trust_client_supplied_retrieval_hosts() -> None:
    resolution = IdentityResolver.confirmed_resolution_from_request(ChatRequest(
        game="Chosen Game",
        question="Where is the item?",
        metadata={
            "confirmed_game": True,
            "game_aliases": ["Injected Alias"],
            "database_domains": ["127.0.0.1", "internal.attacker.test"],
        },
    ))

    assert resolution is not None and resolution.is_confirmed
    assert resolution.aliases == []
    assert resolution.database_domains == []


def test_typed_semantic_gap_drives_query_without_access_template() -> None:
    state = parse_investigation_state(
        json.dumps(
            {
                "goal": "How is the ship first obtained?",
                "known_facts": [
                    {"statement": "The current source only explains recovering a lost ship.", "source_indexes": [1]}
                ],
                "evidence_gaps": [
                    {
                        "kind": "semantic_distinction",
                        "description": "The first acquisition method is missing.",
                        "query_hint": "first ship acquisition forged title",
                        "source_type": "web",
                        "priority": 5,
                    }
                ],
                "unresolved_questions": [],
                "next_queries": [],
                "aliases": [],
                "complete": False,
            }
        ),
        previous=InvestigationState(goal="How is the ship first obtained?"),
        question="How is the ship first obtained?",
        source_count=1,
        sanitize_text=lambda value: value,
        sanitize_aliases=lambda values: values,
    )

    repaired = ensure_investigation_query(
        state,
        question="How is the ship first obtained?",
        sanitize_text=lambda value: value,
    )

    assert repaired.next_queries[0].query == "first ship acquisition forged title"
    assert repaired.next_queries[0].source_type == "web"
    assert "prerequisite access route" not in repaired.next_queries[0].query


def test_highest_priority_gap_query_preempts_unrelated_model_query() -> None:
    state = parse_investigation_state(
        json.dumps(
            {
                "goal": "Find how the ship is first obtained.",
                "known_facts": [
                    {"statement": "One source describes repainting the ship.", "source_indexes": [1]}
                ],
                "evidence_gaps": [
                    {
                        "kind": "other",
                        "description": "The available paint colors are unknown.",
                        "query_hint": "ship paint color list",
                        "source_type": "community",
                        "priority": 1,
                    },
                    {
                        "kind": "semantic_distinction",
                        "description": "The first acquisition method is still missing.",
                        "query_hint": "first ship acquisition forged title",
                        "source_type": "wiki",
                        "priority": 5,
                    },
                ],
                "unresolved_questions": [],
                "next_queries": [
                    {"source_type": "community", "query": "ship paint color list"}
                ],
                "aliases": [],
                "complete": False,
            }
        ),
        previous=InvestigationState(goal="Find how the ship is first obtained."),
        question="Find how the ship is first obtained.",
        source_count=1,
        sanitize_text=lambda value: value,
        sanitize_aliases=lambda values: values,
    )

    assert state.next_queries[0].query == "first ship acquisition forged title"
    assert state.next_queries[0].source_type == "wiki"
    assert state.next_queries[1].query == "ship paint color list"


def test_highest_priority_gap_description_preempts_query_without_hint() -> None:
    state = InvestigationState(
        goal="Determine the rule outcome.",
        evidence_gaps=[
            {
                "kind": "premise",
                "description": "Verify whether the final participant still counts as active",
                "priority": 5,
                "source_type": "wiki",
            },
            {
                "kind": "other",
                "description": "Find cosmetic colors",
                "priority": 1,
            },
        ],
        next_queries=[{"source_type": "web", "query": "cosmetic color list"}],
    )

    repaired = ensure_investigation_query(
        state,
        question="Who wins when the final participant leaves?",
        sanitize_text=lambda value: value,
    )

    assert repaired.next_queries[0].query.startswith("Verify whether the final participant")
    assert repaired.next_queries[0].source_type == "wiki"


@pytest.mark.asyncio
async def test_investigation_semantically_checks_direct_non_action_evidence() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "goal": "Who wins under this rule?",
                    "known_facts": [{"statement": "The rule names the winning role.", "source_indexes": [1]}],
                    "evidence_gaps": [],
                    "unresolved_questions": [],
                    "next_queries": [],
                    "aliases": [],
                    "complete": True,
                }
            )

    provider = Provider()
    llm = GuideLLM(provider=provider)
    request = ChatRequest(game="Unseen Social Game", question="If the final unmarked player leaves, who wins?")
    state = await llm.update_investigation(
        request=request,
        plan=SearchPlan(intent="general"),
        sources=[
            Source(
                title="Winning rule",
                url="https://example.com/rule",
                evidence="The final unmarked player leaves and the named role wins.",
            )
        ],
        investigation=InvestigationState(goal=request.question),
    )

    assert provider.calls == 1
    assert state.complete is True


@pytest.mark.asyncio
async def test_refinement_wiki_result_does_not_block_open_web_search() -> None:
    class SearchClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search(self, **kwargs):
            self.calls.append(kwargs["query"])
            return {
                "results": [
                    {
                        "title": "Niche Game Artifact ZX-17 first acquisition",
                        "url": "https://gamefaqs.example/niche-game/artifact-zx-17",
                        "content": "Niche Game guide: Artifact ZX-17 is first acquired from the archivist.",
                        "score": 0.9,
                    }
                ]
            }

    client = SearchClient()
    provider = TavilySearchProvider(
        settings=Settings(
            search_cache_use_redis=False,
            mediawiki_direct_search=False,
            tavily_first_wave_queries=1,
            tavily_max_queries_per_request=1,
        ),
        client=client,
    )

    async def direct_wiki(**kwargs):
        return [
            Source(
                title="Artifact ZX-17 recovery",
                url="https://niche-game.example/wiki/artifact-zx-17",
                evidence="This page explains how to recover Artifact ZX-17 after it is lost.",
                source_type="wiki",
                trust_score=0.8,
                score=0.8,
            )
        ]

    provider._router.mediawiki.search = direct_wiki
    sources = await provider.search(
        "How is Artifact ZX-17 first acquired?",
        "Niche Game",
        plan=SearchPlan(
            intent="item_location",
            queries=[{"source_type": "web", "query": "Artifact ZX-17 first acquisition"}],
            refinement=True,
        ),
        game_resolution=GameResolution(
            input_name="Niche Game",
            confirmed_name="Niche Game",
            aliases=["Niche Game"],
            confidence=1,
        ),
    )

    assert client.calls
    assert any("gamefaqs.example" in str(source.url) for source in sources)
    assert any("niche-game.example" in str(source.url) for source in sources)
