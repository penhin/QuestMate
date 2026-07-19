import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

import agent as agent_module
import main
from config import Settings
from quality_policy import (
    EVIDENCE_POOL_WEIGHTS,
    GAME_RESOLUTION_POLICY,
    RELEVANCE_SCORE_POLICY,
    SEARCH_RESULT_WEIGHTS,
    SOURCE_POLICIES,
)
from knowledge import KnowledgeStore, chunk_text, keyword_terms, parse_published_at
from agent import QuestAgent
from game_resolution import GameResolver, identity_matches_game
from llm import GuideLLM
from schemas import ChatRequest, ChatResponse, FeedbackRating, FeedbackRequest, GameResolution, SearchPlan, Source
from search import MediaWikiClient, TavilySearchProvider
from retrieval.relevance import result_relevance_score
from retrieval.source_builder import build_source
from storage import InMemoryConversationStore, PostgresConversationStore


client = TestClient(main.app)


def test_production_environment_aliases_are_fail_closed() -> None:
    assert Settings(app_env="production").is_production
    assert Settings(app_env=" PROD ").is_production
    assert Settings(app_env="staging").is_production
    assert Settings(app_env="prodution").is_production
    assert not Settings(app_env="development").is_production
    assert not Settings(app_env=" test ").is_production


@pytest.fixture(autouse=True)
def isolated_conversation_store(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(main, "conversation_store", store)
    monkeypatch.setattr(agent_module, "conversation_store", store)
    yield store


class EmptySearchProvider:
    async def resolve_game(self, game: str, question: str | None = None):
        return GameResolution(input_name=game, confirmed_name=game, confidence=0.8)

    async def search(self, query: str, game: str, max_results: int | None = None, plan=None, game_resolution=None):
        return []


class FastIdentitySearchClient:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, **kwargs):
        self.queries.append(kwargs["query"])
        if kwargs["query"].startswith('"Look Outside" video game'):
            return {
                "results": [
                    {
                        "title": "Look Outside on Steam",
                        "url": "https://store.steampowered.com/app/3373660/Look_Outside/",
                        "content": "Look Outside is an indie RPG.",
                        "score": 0.9,
                    },
                    {
                        "title": "Look Outside Wiki",
                        "url": "https://look-outside.fandom.com/wiki/Look_Outside_Wiki",
                        "content": "Look Outside wiki database.",
                        "score": 0.8,
                    },
                    {
                        "title": "Smooch Mode | Look Outside Wiki",
                        "url": "https://look-outside.fandom.com/wiki/Smooch_Mode",
                        "content": "Look Outside mode guide.",
                        "score": 0.85,
                    },
                ]
            }
        return {"results": []}


class CountingEmptySearchClient:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, **kwargs):
        self.queries.append(kwargs["query"])
        return {"results": []}


class RelativeUrlSearchClient:
    def search(self, **kwargs):
        return {
            "results": [{
                "title": "Redirect result",
                "url": "/goto?url=invalid",
                "content": "Crystal Project Golden Quintar breeding guide.",
                "score": 0.9,
            }]
        }


class DirectWikiClient:
    def __init__(self):
        self.queries: list[tuple[str, str]] = []

    def search(self, *, domain: str, query: str, max_results: int):
        self.queries.append((domain, query))
        return {
            "results": [{
                "title": "Pavol",
                "url": "https://felvidek.fandom.com/wiki/Pavol",
                "content": "Pavol is the protagonist of Felvidek and the player character from the start.",
                "score": 0.9,
            }]
        }


class ExpandingWikiClient:
    def search(self, *, domain: str, query: str, max_results: int):
        return {
            "results": [
                {
                    "title": "Area index",
                    "url": f"https://{domain}/wiki/Area_index",
                    "content": "Niche Game area overview.",
                    "links": ["Artifact ZX-17", "Unrelated history"],
                    "score": 0.8,
                }
            ]
        }

    def fetch_pages(self, *, domain: str, titles: list[str]):
        assert titles == ["Artifact ZX-17"]
        return {
            "results": [
                {
                    "title": "Artifact ZX-17",
                    "url": f"https://{domain}/wiki/Artifact_ZX-17",
                    "content": "Niche Game Artifact ZX-17 opens the sealed gate.",
                    "links": [],
                    "score": 0.9,
                }
            ]
        }


class RecordingContentIndex:
    def __init__(self):
        self.documents = []

    async def index_content(self, **kwargs):
        self.documents.append(kwargs)
        return {"status": "ready", "document_id": "doc", "chunk_count": 1}


async def test_knowledge_query_vectors_are_reused() -> None:
    class CountingEmbeddings:
        def __init__(self):
            self.calls = 0

        async def embed(self, texts):
            self.calls += 1
            return [[0.25, 0.75]]

    store = KnowledgeStore(settings=Settings())
    embeddings = CountingEmbeddings()
    store.embeddings = embeddings

    first = await store._query_vector("  Same   Question ")
    second = await store._query_vector("Same Question")
    case_variant = await store._query_vector("same question")

    assert first == second == case_variant == [0.25, 0.75]
    assert embeddings.calls == 2
    await store.database.engine.dispose()


def test_game_resolver_fast_identity_confirms_niche_game() -> None:
    client = FastIdentitySearchClient()
    resolution = GameResolver(client).resolve(game="Look Outside")

    assert resolution.is_confirmed
    assert resolution.confirmed_name == "Look Outside"
    assert resolution.database_domains == ["look-outside.fandom.com"]
    assert "Smooch Mode" not in resolution.aliases
    assert len(client.queries) == 1


async def test_tavily_identity_search_is_cached() -> None:
    client = FastIdentitySearchClient()
    provider = TavilySearchProvider(client=client)

    first = await provider.resolve_game("Look Outside")
    second = await provider.resolve_game("Look Outside")

    assert first == second
    assert len(client.queries) == 1
    assert provider._client.upstream_calls == 1
    assert provider._client.cache_hits == 1


async def test_source_registry_never_skips_current_identity_search() -> None:
    class CachedRegistry:
        def __init__(self) -> None:
            self.writes = []

        async def get_resolution(self, game: str):
            return GameResolution(
                input_name=game,
                confirmed_name="Registered Game",
                aliases=["Registered Alias"],
                platform_urls=["https://store.steampowered.com/app/1/Registered_Game/"],
                database_domains=["registered-game.fandom.com"],
                confidence=0.9,
            )

        async def upsert_resolution(self, resolution):
            self.writes.append(resolution)

    client = CountingEmptySearchClient()
    registry = CachedRegistry()
    provider = TavilySearchProvider(client=client, source_registry=registry)

    resolution = await provider.resolve_game("Registered Alias")

    assert not resolution.is_confirmed
    assert client.queries
    assert registry.writes == []


async def test_live_identity_is_saved_to_source_registry() -> None:
    class EmptyRegistry:
        def __init__(self):
            self.saved = []

        async def get_resolution(self, game: str):
            return None

        async def upsert_resolution(self, resolution):
            self.saved.append(resolution)

    registry = EmptyRegistry()
    provider = TavilySearchProvider(client=FastIdentitySearchClient(), source_registry=registry)

    resolution = await provider.resolve_game("Look Outside")

    assert resolution.is_confirmed
    assert registry.saved == [resolution]


async def test_game_identity_fallback_has_a_hard_query_cap() -> None:
    client = CountingEmptySearchClient()
    provider = TavilySearchProvider(
        settings=Settings(tavily_search_cache_ttl_seconds=0),
        client=client,
    )

    resolution = await provider.resolve_game("Unknown Niche Game")

    assert resolution.is_confirmed is False
    assert len(client.queries) <= 4
    assert client.queries[-2:] == [
        "Unknown Niche Game wiki",
        "Unknown Niche Game official wiki",
    ]


async def test_progressive_search_caps_paid_queries() -> None:
    client = CountingEmptySearchClient()
    settings = Settings(
        tavily_first_wave_queries=2,
        tavily_max_queries_per_request=4,
        tavily_search_cache_ttl_seconds=0,
    )
    provider = TavilySearchProvider(settings=settings, client=client)
    plan = SearchPlan(
        intent="item_usage",
        aliases=["Loaded Dice"],
        queries=[
            {"source_type": "wiki", "query": "Loaded Dice item use"},
            {"source_type": "community", "query": "Loaded Dice where use"},
            {"source_type": "web", "query": "Loaded Dice walkthrough"},
        ],
    )

    sources = await provider.search(
        "Loaded Dice 有什么用？",
        "Escape from Ever After",
        plan=plan,
        game_resolution=GameResolution(
            input_name="Escape from Ever After",
            confirmed_name="Escape from Ever After",
            confidence=0.8,
        ),
    )

    assert sources == []
    assert len(client.queries) == 4


async def test_search_skips_relative_result_urls() -> None:
    provider = TavilySearchProvider(client=RelativeUrlSearchClient())

    sources = await provider.search(
        "Golden Quintar 怎么培育？",
        "Crystal Project",
        plan=SearchPlan(
            intent="game_mechanic",
            aliases=["Golden Quintar"],
            queries=[{"source_type": "wiki", "query": "Golden Quintar breeding"}],
        ),
        game_resolution=GameResolution(
            input_name="Crystal Project",
            confirmed_name="Crystal Project",
            confidence=0.8,
        ),
    )

    assert sources == []


async def test_single_direct_mediawiki_hit_keeps_independent_search_route() -> None:
    tavily = CountingEmptySearchClient()
    mediawiki = DirectWikiClient()
    provider = TavilySearchProvider(client=tavily, mediawiki_client=mediawiki)
    plan = SearchPlan(
        intent="quest_step",
        aliases=["Pavol"],
        queries=[{"source_type": "wiki", "query": "Pavol recruit party"}],
    )

    sources = await provider.search(
        "Pavol 怎么加入队伍？",
        "Felvidek",
        plan=plan,
        game_resolution=GameResolution(
            input_name="Felvidek",
            confirmed_name="Felvidek",
            database_domains=["felvidek.fandom.com"],
            confidence=0.8,
        ),
    )

    assert [source.title for source in sources] == ["Pavol"]
    assert tavily.queries
    assert mediawiki.queries == [("felvidek.fandom.com", "Pavol recruit party")]


async def test_direct_mediawiki_ranks_pages_against_current_gap_query() -> None:
    class GapWikiClient:
        def search(self, *, domain: str, query: str, max_results: int):
            return {
                "results": [{
                    "title": "Relay Token",
                    "url": f"https://{domain}/wiki/Relay_Token",
                    "content": "Unseen Game relay token acquisition route behind the maintenance wall.",
                    "score": 0.9,
                }]
            }

    tavily = CountingEmptySearchClient()
    provider = TavilySearchProvider(client=tavily, mediawiki_client=GapWikiClient())
    sources = await provider.search(
        "How do I open the final gate?",
        "Unseen Game",
        plan=SearchPlan(
            intent="quest_step",
            aliases=["Final Gate"],
            queries=[{"source_type": "wiki", "query": "relay token acquisition route"}],
            refinement=True,
        ),
        game_resolution=GameResolution(
            input_name="Unseen Game",
            confirmed_name="Unseen Game",
            database_domains=["unseen-game.example"],
            confidence=0.8,
        ),
    )

    assert [source.title for source in sources] == ["Relay Token"]
    assert len(tavily.queries) == 1
    assert "site:" not in tavily.queries[0]


def test_evidence_passage_prefers_late_identifier_phrase_over_numeric_noise() -> None:
    content = (
        "Coordinates 35 35 35 and unrelated measurements. " * 80
        + "The Apt 35 Key is conspicuously placed in the center of the open floor."
    )

    passage = TavilySearchProvider._best_evidence_passage(
        content,
        question="Room 35 Apt 35 Key exact location",
        max_chars=500,
    )

    assert "Apt 35 Key is conspicuously placed" in passage


def test_evidence_passage_keeps_subject_at_sentence_boundary() -> None:
    content = (
        "Unrelated corridor measurements and repeated room numbers. " * 50
        + "Apartment 12 contains a concealed annex. "
        + "The Apt 35 Key is placed in the center of that annex."
    )

    passage = TavilySearchProvider._best_evidence_passage(
        content,
        question="Apt 35 Key exact location",
        max_chars=240,
    )

    assert passage.startswith("The Apt 35 Key") or "\n\nThe Apt 35 Key" in passage


def test_evidence_passage_combines_page_prerequisite_with_distant_target() -> None:
    content = (
        "The sealed annex entrance requires Solvent before it can be entered. "
        "Speak to the four witnesses in order to obtain the Annex Key. "
        + "Background history with no actionable details. " * 80
        + "The ZX-35 Token is placed in the center of the column hall."
    )

    passage = TavilySearchProvider._best_evidence_passage(
        content,
        question="ZX-35 Token exact location annex",
        max_chars=700,
    )

    assert "requires Solvent" in passage
    assert "obtain the Annex Key" in passage
    assert "ZX-35 Token is placed" in passage
    assert len(passage) <= 700


async def test_mediawiki_pages_are_auto_indexed_for_future_questions() -> None:
    tavily = CountingEmptySearchClient()
    mediawiki = DirectWikiClient()
    content_index = RecordingContentIndex()
    provider = TavilySearchProvider(
        client=tavily,
        mediawiki_client=mediawiki,
        content_index=content_index,
    )
    plan = SearchPlan(
        intent="quest_step",
        aliases=["Pavol"],
        queries=[{"source_type": "wiki", "query": "Pavol recruit party"}],
    )

    await provider.search(
        "Pavol 怎么加入队伍？",
        "Felvidek",
        plan=plan,
        game_resolution=GameResolution(
            input_name="Felvidek",
            confirmed_name="Felvidek",
            aliases=["Felvidek"],
            database_domains=["felvidek.fandom.com"],
            confidence=0.8,
        ),
    )
    await provider.wait_for_background_tasks()

    assert len(content_index.documents) == 1
    assert content_index.documents[0]["title"] == "Pavol"
    assert "protagonist" in content_index.documents[0]["content"]
    assert content_index.documents[0]["skip_if_fresh"] is True


async def test_mediawiki_auto_index_does_not_block_search_results() -> None:
    class BlockingContentIndex:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def index_content(self, **kwargs):
            self.started.set()
            await self.release.wait()
            return {"status": "ready", "document_id": "doc", "chunk_count": 1}

    content_index = BlockingContentIndex()
    provider = TavilySearchProvider(
        client=CountingEmptySearchClient(),
        mediawiki_client=DirectWikiClient(),
        content_index=content_index,
    )
    plan = SearchPlan(
        intent="quest_step",
        aliases=["Pavol"],
        queries=[{"source_type": "wiki", "query": "Pavol recruit party"}],
    )

    sources = await provider.search(
        "Pavol 怎么加入队伍？",
        "Felvidek",
        plan=plan,
        game_resolution=GameResolution(
            input_name="Felvidek",
            confirmed_name="Felvidek",
            database_domains=["felvidek.fandom.com"],
            confidence=0.8,
        ),
    )

    assert [source.title for source in sources] == ["Pavol"]
    await asyncio.wait_for(content_index.started.wait(), timeout=1)
    content_index.release.set()
    await provider.wait_for_background_tasks()


async def test_mediawiki_expands_only_question_matching_links() -> None:
    provider = TavilySearchProvider(
        client=CountingEmptySearchClient(),
        mediawiki_client=ExpandingWikiClient(),
    )
    plan = SearchPlan(
        intent="item_usage",
        aliases=["Artifact ZX-17"],
        queries=[{"source_type": "wiki", "query": "Artifact ZX-17 use"}],
    )

    sources = await provider.search(
        "ZX-17 有什么用？",
        "Niche Game",
        plan=plan,
        game_resolution=GameResolution(
            input_name="Niche Game",
            confirmed_name="Niche Game",
            aliases=["Niche Game"],
            database_domains=["niche-game.fandom.com"],
            confidence=0.8,
        ),
    )

    assert [source.title for source in sources] == ["Artifact ZX-17"]
    assert "sealed gate" in (sources[0].evidence or "")


def test_mediawiki_wikitext_cleaner_removes_markup() -> None:
    content = """
    {{Infobox character|name=Pavol}}
    == Location ==
    [[File:Pavol.png|thumb]]
    '''Pavol''' waits in [[Hollow Basin|the Hollow Basin]].<ref>Hidden note</ref>
    """

    cleaned = MediaWikiClient._clean_wikitext(content)

    assert "Pavol waits in the Hollow Basin." in cleaned
    assert "Infobox" not in cleaned
    assert "File:" not in cleaned
    assert "Hidden note" not in cleaned


def test_game_identity_requires_exact_compact_name() -> None:
    assert identity_matches_game(
        title="Look Outside Wiki",
        url="https://look-outside.fandom.com",
        game="Look Outside",
    )
    assert not identity_matches_game(
        title="A character looks outside",
        url="https://bully.fandom.com/wiki/Kissing",
        game="Look Outside",
    )


def test_game_resolver_enriches_confirmed_store_identity_with_niche_wiki() -> None:
    class StoreThenWikiClient:
        def search(self, **kwargs):
            if kwargs["query"] == '"Felvidek" video game official store wiki':
                return {
                    "results": [{
                        "title": "Felvidek on Steam",
                        "url": "https://store.steampowered.com/app/2299900/Felvidek/",
                        "score": 0.9,
                    }]
                }
            if kwargs["query"] == '"Felvidek" wiki':
                return {
                    "results": [{
                        "title": "Pavol | Felvidek Wiki",
                        "url": "https://felvidek.fandom.com/wiki/Pavol",
                        "score": 0.8,
                    }]
                }
            return {"results": []}

    resolution = GameResolver(StoreThenWikiClient()).resolve(game="Felvidek")

    assert resolution.is_confirmed
    assert resolution.database_domains == ["felvidek.fandom.com"]


def test_knowledge_chunking_preserves_order_and_overlap() -> None:
    content = "第一段资料。" * 180
    chunks = chunk_text(content, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].startswith("第一段资料")
    terms = keyword_terms("女武神 Malenia 怎么打")
    assert "malenia" in terms
    assert {"女武", "武神"}.issubset(terms)


def test_question_tokens_preserve_cjk_spans_without_intent_vocabulary() -> None:
    from query_tokens import question_relevance_tokens

    tokens = question_relevance_tokens("玛莲妮亚怎么打，弱点是什么？")
    assert "玛莲妮亚怎么打" in tokens
    assert "玛莲" in tokens
    assert question_relevance_tokens("Malenia boss strategy weakness") == [
        "malenia", "boss", "strategy", "weakness"
    ]
    assert question_relevance_tokens("How does Pavol join the party?") == [
        "how", "does", "pavol", "join", "the", "party"
    ]


def test_quality_policy_weights_and_thresholds_are_valid() -> None:
    assert sum(
        (
            SEARCH_RESULT_WEIGHTS.relevance,
            SEARCH_RESULT_WEIGHTS.retrieval,
            SEARCH_RESULT_WEIGHTS.trust,
            SEARCH_RESULT_WEIGHTS.intent,
            SEARCH_RESULT_WEIGHTS.domain,
            SEARCH_RESULT_WEIGHTS.version,
        )
    ) == pytest.approx(1)
    assert sum(
        (
            EVIDENCE_POOL_WEIGHTS.relevance,
            EVIDENCE_POOL_WEIGHTS.retrieval,
            EVIDENCE_POOL_WEIGHTS.trust,
            EVIDENCE_POOL_WEIGHTS.version,
        )
    ) == pytest.approx(1)
    assert set(SOURCE_POLICIES) == {"official", "wiki", "community", "web"}
    assert 0 < GAME_RESOLUTION_POLICY.confirmed_threshold < 1
    assert RELEVANCE_SCORE_POLICY.base_score + RELEVANCE_SCORE_POLICY.coverage_weight <= 1


def test_shared_game_registry_has_unique_processes_and_names() -> None:
    registry_path = Path(__file__).parents[1] / "overlay" / "src" / "config" / "games.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    games = registry["games"]
    names = [game["name"] for game in games]
    processes = [process.lower() for game in games for process in game["processes"]]

    assert games
    assert len(names) == len(set(names))
    assert len(processes) == len(set(processes))
    assert all(process.endswith(".exe") for process in processes)


class LocalKnowledge:
    async def retrieve(self, *, game: str, query: str):
        return [
            Source(
                title="本地女武神资料",
                url="https://example.com/malenia",
                snippet="本地索引内容",
                source_type="wiki",
                trust_score=0.8,
                trust_label="百科",
            )
        ]


class BrokenKnowledge:
    async def retrieve(self, *, game: str, query: str):
        raise OSError("knowledge database unavailable")


async def test_agent_merges_local_knowledge_before_web_results() -> None:
    agent = QuestAgent(search_provider=EmptySearchProvider(), knowledge=LocalKnowledge())
    sources = await agent._retrieve_sources(
        "女武神怎么打？",
        "Elden Ring",
        plan=SearchPlan(),
        game_resolution=GameResolution(input_name="Elden Ring", confirmed_name="Elden Ring", confidence=0.8),
    )

    assert [source.title for source in sources] == ["本地女武神资料"]


async def test_agent_reranks_local_and_web_sources_together() -> None:
    class WebSearchProvider(EmptySearchProvider):
        async def search(self, query: str, game: str, max_results=None, plan=None, game_resolution=None):
            return [
                Source(
                    title="Malenia strategy",
                    url="https://example.org/malenia",
                    evidence="Malenia weakness phase dodge strategy",
                    score=0.9,
                    trust_score=0.8,
                )
            ]

    agent = QuestAgent(search_provider=WebSearchProvider(), knowledge=LocalKnowledge())
    sources = await agent._retrieve_sources(
        "Malenia weakness",
        "Elden Ring",
        plan=SearchPlan(intent="boss_strategy", aliases=["Malenia"]),
        game_resolution=GameResolution(input_name="Elden Ring", confirmed_name="Elden Ring", confidence=0.8),
    )

    assert sources[0].title == "Malenia strategy"


def test_agent_merges_complementary_passages_from_same_page() -> None:
    sources = QuestAgent._rank_sources(
        sources=[
            Source(
                title="Access guide",
                url="https://example.com/wiki/access",
                evidence="The objective requires a special token.",
                score=0.8,
            ),
            Source(
                title="Access guide",
                url="https://example.com/wiki/access",
                evidence="Reach the token through the concealed maintenance passage.",
                score=0.9,
            ),
        ],
        query="special token maintenance passage",
        intent="game_mechanic",
    )

    assert len(sources) == 1
    assert "requires a special token" in (sources[0].evidence or "")
    assert "concealed maintenance passage" in (sources[0].evidence or "")


async def test_agent_keeps_web_results_when_local_dimension_fails() -> None:
    class WebSearchProvider(EmptySearchProvider):
        async def search(self, query: str, game: str, max_results=None, plan=None, game_resolution=None):
            return [
                Source(
                    title="Malenia Wiki",
                    url="https://example.org/malenia",
                    evidence="Malenia weakness phase guide",
                    score=0.9,
                )
            ]

    agent = QuestAgent(search_provider=WebSearchProvider(), knowledge=BrokenKnowledge())
    sources = await agent._retrieve_sources(
        "Malenia weakness",
        "Elden Ring",
        plan=SearchPlan(intent="boss_strategy", aliases=["Malenia"]),
        game_resolution=GameResolution(input_name="Elden Ring", confirmed_name="Elden Ring", confidence=0.8),
    )

    assert [source.title for source in sources] == ["Malenia Wiki"]


async def test_agent_runs_one_model_driven_refinement_pass() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class StagedSearchProvider(EmptySearchProvider):
        def __init__(self):
            self.plans = []

        async def search(self, query: str, game: str, max_results=None, plan=None, game_resolution=None):
            self.plans.append(plan)
            if len(self.plans) == 1:
                return [Source(title="Generic guide", url="https://example.com/game", evidence="Game overview")]
            return [
                Source(
                    title="Translated Entity 35",
                    url="https://example.com/entity-35",
                    evidence="Translated Entity 35 exact instructions",
                )
            ]

    class RefinementLLM:
        def __init__(self):
            self.calls = 0

        async def refine_search_plan(self, *, request, plan, sources, history, game_resolution=None):
            self.calls += 1
            if self.calls > 1:
                return None
            return SearchPlan(
                intent=plan.intent,
                aliases=["Translated Entity 35"],
                queries=[{"source_type": "wiki", "query": "Translated Entity 35 exact instructions"}],
                refinement=True,
            )

    search_provider = StagedSearchProvider()
    agent = QuestAgent(search_provider=search_provider, llm=RefinementLLM(), knowledge=EmptyKnowledge())
    request = ChatRequest(game="Unknown Niche Game", question="如何进入35号区域？")
    resolution = GameResolution(
        input_name=request.game,
        confirmed_name=request.game,
        aliases=[request.game],
        confidence=0.8,
    )

    sources, plan, refined = await agent._retrieve_with_refinement(
        request=request,
        history=[],
        plan=SearchPlan(queries=[{"source_type": "web", "query": request.question}]),
        game_resolution=resolution,
    )

    assert refined is True
    assert len(search_provider.plans) == 2
    assert "Translated Entity 35" in plan.aliases
    assert sources[0].title == "Translated Entity 35"


async def test_agent_bounds_dependency_refinement_to_one_hop_before_answering() -> None:
    class EmptyKnowledge:
        async def retrieve(self, *, game: str, query: str):
            return []

    class DependencySearchProvider(EmptySearchProvider):
        def __init__(self):
            self.calls = 0

        async def search(self, query: str, game: str, max_results=None, plan=None, game_resolution=None):
            self.calls += 1
            evidence = (
                "The final gate requires a relay token."
                if self.calls == 1
                else "The relay token is behind the maintenance passage."
                if self.calls == 2
                else "The maintenance passage opens after restoring auxiliary power."
            )
            return [
                Source(
                    title=f"Dependency {self.calls}",
                    url=f"https://example.com/dependency-{self.calls}",
                    evidence=evidence,
                )
            ]

    class DependencyLLM:
        def __init__(self):
            self.calls = 0

        async def refine_search_plan(self, *, request, plan, sources, history, game_resolution=None):
            self.calls += 1
            if self.calls == 1:
                return SearchPlan(
                    intent=plan.intent,
                    aliases=["relay token"],
                    queries=[{"source_type": "wiki", "query": "relay token acquisition route"}],
                    refinement=True,
                )
            if self.calls == 2:
                return SearchPlan(
                    intent=plan.intent,
                    aliases=["maintenance passage"],
                    queries=[{"source_type": "wiki", "query": "maintenance passage access prerequisite"}],
                    refinement=True,
                )
            return None

    search_provider = DependencySearchProvider()
    llm = DependencyLLM()
    agent = QuestAgent(search_provider=search_provider, llm=llm, knowledge=EmptyKnowledge())
    request = ChatRequest(game="Niche Game", question="如何打开最终大门？")

    sources, plan, refined = await agent._retrieve_with_refinement(
        request=request,
        history=[],
        plan=SearchPlan(
            intent="game_mechanic",
            queries=[{"source_type": "wiki", "query": "final gate opening requirements"}],
        ),
        game_resolution=GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=[request.game],
            confidence=0.8,
        ),
    )

    assert refined is True
    assert search_provider.calls == 2
    assert llm.calls == 1
    assert {source.title for source in sources} == {"Dependency 1", "Dependency 2"}
    assert "relay token" in plan.aliases
    assert "maintenance passage" not in plan.aliases


def test_patch_answers_require_dated_official_evidence() -> None:
    request = ChatRequest(game="Elden Ring", question="Malenia patch 改了什么？")
    plan = SearchPlan(intent="patch")
    wiki_source = Source(
        title="Malenia patch notes",
        url="https://example.com/malenia-patch",
        snippet="Malenia patch notes",
        source_type="wiki",
        trust_score=0.8,
        trust_label="百科",
        game_version="1.12",
    )
    official_source = wiki_source.model_copy(
        update={
            "source_type": "official",
            "trust_score": 0.95,
            "trust_label": "官方",
            "published_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
        }
    )

    assert GuideLLM._should_return_conservative_answer(request=request, sources=[wiki_source], plan=plan)
    assert not GuideLLM._should_return_conservative_answer(request=request, sources=[official_source], plan=plan)
    assert parse_published_at("2026-07-01T12:00:00Z") == datetime(2026, 7, 1, 12, tzinfo=timezone.utc)


async def test_storage_requires_explicit_opt_in_for_ephemeral_fallback() -> None:
    class BrokenDatabase:
        settings = Settings(allow_in_memory_storage=False)

        async def init_schema(self) -> None:
            raise OSError("database unavailable")

    store = PostgresConversationStore(database=BrokenDatabase())
    with pytest.raises(RuntimeError, match="Postgres is unavailable"):
        await store.init_schema()


class AmbiguousGameSearchProvider:
    def __init__(self):
        self.search_called = False

    async def resolve_game(self, game: str, question: str | None = None):
        return GameResolution(
            input_name=game,
            confirmed_name="",
            confidence=0.45,
            ambiguous=True,
            candidates=[
                {
                    "name": "Afterwards",
                    "tags": ["解谜"],
                    "platform_urls": ["https://store.steampowered.com/app/1/Afterwards/"],
                    "confidence": 0.72,
                },
                {
                    "name": "Afterwards Survival",
                    "tags": ["生存"],
                    "platform_urls": ["https://example.com/afterwards-survival"],
                    "confidence": 0.66,
                },
            ],
        )

    async def search(self, query: str, game: str, max_results: int | None = None, plan=None, game_resolution=None):
        self.search_called = True
        return []


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "QuestMate"


def test_chat_request_schema_defaults() -> None:
    request = ChatRequest(game="Elden Ring", question="新手先打哪里？")

    assert request.stream is False
    assert request.session_id is None
    assert request.metadata == {}


def test_chat_endpoint_returns_fallback_answer_without_sources(monkeypatch) -> None:
    empty_search = EmptySearchProvider()
    monkeypatch.setattr(main.quest_agent, "search_provider", empty_search)
    monkeypatch.setattr(main.quest_agent.retrieval, "search_provider", empty_search)

    response = client.post(
        "/api/chat",
        json={"game": "Elden Ring", "question": "新手先打哪里？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert "Elden Ring" in body["answer"]
    assert body["sources"] == []
    assert body["is_new"] is True


def test_chat_endpoint_streams_status_chunks_and_done(monkeypatch) -> None:
    monkeypatch.setattr(main.quest_agent, "search_provider", EmptySearchProvider())

    with client.stream(
        "POST",
        "/api/chat",
        json={"game": "Elden Ring", "question": "新手先打哪里？", "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: status" in body
    assert "event: chunk" in body
    assert "event: done" in body
    assert "理解问题" in body
    assert "类型：" in body
    assert "来源筛选" in body
    assert "Elden Ring" in body


def test_cors_allows_tauri_dev_origin() -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:1420",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:1420"


def test_feedback_schema() -> None:
    feedback = FeedbackRequest(
        session_id=uuid4(),
        rating=FeedbackRating.helpful,
        comment="引用有帮助",
    )

    assert feedback.rating == FeedbackRating.helpful
    assert feedback.comment == "引用有帮助"


def test_session_list_rename_and_delete(monkeypatch) -> None:
    monkeypatch.setattr(main.quest_agent, "search_provider", EmptySearchProvider())

    chat = client.post(
        "/api/chat",
        json={"game": "Elden Ring", "question": "女武神怎么打？"},
    )
    session_id = chat.json()["session_id"]

    sessions = client.get("/api/sessions")
    assert sessions.status_code == 200
    assert sessions.json()["sessions"][0]["session_id"] == session_id

    renamed = client.patch(f"/api/sessions/{session_id}", json={"title": "女武神打法"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "女武神打法"

    deleted = client.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 204


class FakeSearchClient:
    def __init__(self):
        self.queries = []

    def search(self, **kwargs):
        query = kwargs["query"]
        self.queries.append(query)
        if "official" in query:
            return {
                "results": [
                    {
                        "title": "Official guide",
                        "url": "https://example.com/official",
                        "content": "Official answer",
                        "score": 0.7,
                    }
                ]
            }
        if "site:fandom.com" in query:
            return {
                "results": [
                    {
                        "title": "Fandom guide",
                        "url": "https://eldenring.fandom.com/wiki/Malenia",
                        "content": "新手先打哪里 wiki answer",
                        "score": 0.9,
                    }
                ]
            }
        if "Malenia" in query:
            return {
                "results": [
                    {
                        "title": "Malenia Blade of Miquella",
                        "url": "https://eldenring.wiki.fextralife.com/Malenia+Blade+of+Miquella",
                        "content": "Elden Ring boss weakness phase strategy.",
                        "score": 0.9,
                    }
                ]
            }
        return {"results": []}


class DatabaseDiscoverySearchClient:
    def __init__(self):
        self.queries = []

    def search(self, **kwargs):
        query = kwargs["query"]
        self.queries.append(query)
        if query == "Look Outside wiki":
            return {
                "results": [
                    {
                        "title": "Look Outside Wiki",
                        "url": "https://lookoutside.fandom.com/wiki/Look_Outside_Wiki",
                        "content": "Look Outside community wiki database.",
                        "score": 0.8,
                    }
                ]
            }
        if query.startswith("site:lookoutside.fandom.com"):
            return {
                "results": [
                    {
                        "title": "Kissing Mode | Look Outside Wiki",
                        "url": "https://lookoutside.fandom.com/wiki/Kissing_Mode",
                        "content": "Look Outside kissing mode unlock enable trigger steps.",
                        "score": 0.9,
                    }
                ]
            }
        return {"results": []}


class SteamAliasSearchClient:
    def __init__(self):
        self.queries = []

    def search(self, **kwargs):
        query = kwargs["query"]
        self.queries.append(query)
        if "逃出从此以后" in query and "official store" in query:
            return {
                "results": [
                    {
                        "title": "Afterwards on Steam",
                        "url": "https://store.steampowered.com/app/123456/Afterwards/",
                        "content": "逃出从此以后 is a puzzle game on Steam.",
                        "score": 0.9,
                    }
                ]
            }
        if "Afterwards" in query and "loaded dice" in query:
            return {
                "results": [
                    {
                        "title": "Loaded Dice | Afterwards Wiki",
                        "url": "https://afterwards.fandom.com/wiki/Loaded_Dice",
                        "content": "Afterwards loaded dice item use effect puzzle interaction.",
                        "score": 0.9,
                    }
                ]
            }
        return {"results": []}


class DuplicateGameCandidateSearchClient:
    def search(self, **kwargs):
        query = kwargs["query"]
        if "official store" in query:
            return {
                "results": [
                    {
                        "title": "逃出从此以后",
                        "url": "https://store.steampowered.com/app/1/Escape_from_Ever_After/",
                        "content": "逃出从此以后 RPG 冒险 动作",
                        "score": 0.9,
                    },
                    {
                        "title": "在Steam 上购买逃出从此以后立省25%",
                        "url": "https://store.steampowered.com/app/1/Escape_from_Ever_After/",
                        "content": "逃出从此以后 RPG 冒险",
                        "score": 0.88,
                    },
                    {
                        "title": "逃出从此以后 - 所有游戏",
                        "url": "https://store.steampowered.com/app/1/Escape_from_Ever_After/",
                        "content": "逃出从此以后 RPG 冒险 动作",
                        "score": 0.85,
                    },
                ]
            }
        return {"results": []}


async def test_search_routes_use_global_sources_and_trust() -> None:
    client = FakeSearchClient()
    provider = TavilySearchProvider(client=client)

    sources = await provider.search("新手先打哪里？", "Elden Ring", max_results=5)

    assert any(query.startswith("site:fandom.com Elden Ring") for query in client.queries)
    assert [source.source_type for source in sources] == ["wiki"]
    assert sources[0].trust_label == "百科"


async def test_search_keeps_results_matched_by_planned_query_alias() -> None:
    client = FakeSearchClient()
    provider = TavilySearchProvider(client=client)
    plan = SearchPlan(
        intent="boss_strategy",
        queries=[
            {
                "source_type": "wiki",
                "query": "Malenia Blade of Miquella boss weakness phase",
            }
        ],
    )

    sources = await provider.search("女武神怎么打？", "Elden Ring", max_results=5, plan=plan)

    assert sources
    assert sources[0].title == "Malenia Blade of Miquella"


def test_search_builds_alias_queries_from_plan() -> None:
    provider = TavilySearchProvider(client=FakeSearchClient())
    plan = SearchPlan(
        intent="boss_strategy",
        aliases=["Malenia Blade of Miquella"],
        queries=[{"source_type": "community", "query": "boss weakness phase"}],
    )

    queries = provider._build_search_queries(game="Elden Ring", question="女武神怎么打？", plan=plan)

    assert any("Malenia Blade of Miquella" in query for query, _source in queries)


def test_refinement_plan_expands_to_only_one_paid_query() -> None:
    provider = TavilySearchProvider(client=FakeSearchClient())
    plan = SearchPlan(
        intent="game_mechanic",
        aliases=["maintenance passage"],
        queries=[{"source_type": "wiki", "query": "maintenance passage access prerequisite"}],
        refinement=True,
    )

    queries = provider._build_search_queries(
        game="Niche Game",
        question="如何打开最终大门？",
        plan=plan,
        database_domains=("niche-game.fandom.com",),
    )

    assert len(queries) == 1


async def test_search_discovers_specific_wiki_database_for_niche_games() -> None:
    client = DatabaseDiscoverySearchClient()
    provider = TavilySearchProvider(client=client)
    plan = SearchPlan(
        intent="game_mechanic",
        aliases=["Kissing Mode"],
        queries=[{"source_type": "wiki", "query": "kissing mode unlock enable trigger"}],
    )

    sources = await provider.search("如何开启亲亲模式？", "Look Outside", max_results=5, plan=plan)

    assert any(query == "Look Outside wiki" for query in client.queries)
    assert any(query.startswith("site:lookoutside.fandom.com") for query in client.queries)
    assert sources
    assert "lookoutside.fandom.com" in str(sources[0].url)


async def test_search_uses_steam_alias_for_official_chinese_titles() -> None:
    client = SteamAliasSearchClient()
    provider = TavilySearchProvider(client=client)
    plan = SearchPlan(
        intent="item_usage",
        aliases=["loaded dice"],
        queries=[{"source_type": "wiki", "query": "loaded dice item use effect"}],
    )

    sources = await provider.search("灌铅骰子有什么用？", "逃出从此以后", max_results=5, plan=plan)

    assert any("逃出从此以后" in query and "official store" in query for query in client.queries)
    assert any("Afterwards" in query and "loaded dice" in query for query in client.queries)
    assert sources
    assert "afterwards.fandom.com" in str(sources[0].url)


async def test_search_resolves_game_before_question_search() -> None:
    client = SteamAliasSearchClient()
    provider = TavilySearchProvider(client=client)

    resolution = await provider.resolve_game("逃出从此以后", question="灌铅骰子作用")

    assert resolution.is_confirmed
    assert resolution.confirmed_name == "Afterwards"
    assert "Afterwards" in resolution.aliases
    assert resolution.platform_urls


async def test_game_resolution_deduplicates_store_marketing_candidates() -> None:
    provider = TavilySearchProvider(client=DuplicateGameCandidateSearchClient())

    resolution = await provider.resolve_game("逃出从此以后", question="灌铅骰子作用")

    assert [candidate.name for candidate in resolution.candidates] == ["逃出从此以后"]
    assert resolution.candidates[0].tags == ["RPG", "冒险", "动作"]


def test_search_filters_unrelated_game_results() -> None:
    item = {
        "title": "Dead by Daylight Wiki",
        "url": "https://deadbydaylight.fandom.com/wiki/Keys",
        "content": "A horror game wiki page.",
    }

    assert not result_relevance_score(
        item=item,
        game="艾尔登法环",
        question="地下墓室钥匙在哪里",
    )


def test_search_filters_generic_game_page_when_question_does_not_match() -> None:
    item = {
        "title": "Elden Ring Wiki - Fextralife",
        "url": "https://eldenring.wiki.fextralife.com/Elden+Ring+Wiki",
        "content": "General Elden Ring wiki guide and game information.",
    }

    assert not result_relevance_score(
        item=item,
        game="Elden Ring",
        question="随便测试一下",
    )


def test_search_ignores_site_noise_tokens() -> None:
    item = {
        "title": "Metyr, Mother of Fingers | Elden Ring Wiki - Fandom",
        "url": "https://eldenring.fandom.com/wiki/Metyr,_Mother_of_Fingers",
        "content": "A boss page for Elden Ring.",
    }

    assert not result_relevance_score(
        item=item,
        game="Elden Ring",
        question="site:fandom.com Elden Ring Malenia Blade of Miquella boss weakness phase",
    )


def test_search_filters_nightreign_when_question_is_base_elden_ring() -> None:
    item = {
        "title": "White Horn | Nightreign Wiki Elden Ring",
        "url": "https://eldenringnightreign.wiki.fextralife.com/White+Horn",
        "content": "Nightreign item page.",
    }

    assert not result_relevance_score(
        item=item,
        game="Elden Ring",
        question="Malenia Blade of Miquella boss weakness phase",
    )


def test_search_does_not_filter_pages_by_a_fixed_strategy_vocabulary() -> None:
    item = {
        "title": "Malenia | Villains Wiki | Fandom",
        "url": "https://villains.fandom.com/wiki/Malenia",
        "content": "Malenia Blade of Miquella Elden Ring character biography.",
    }

    assert result_relevance_score(
        item=item,
        game="Elden Ring",
        question="Malenia Blade of Miquella boss weakness phase",
    ) > 0


def test_search_filters_battle_wiki_for_strategy() -> None:
    item = {
        "title": "Malenia, Blade of Miquella | All Fiction Battles Wiki | Fandom",
        "url": "https://all-fiction-battles.fandom.com/wiki/Malenia,_Blade_of_Miquella",
        "content": "Malenia Blade of Miquella power scaling.",
    }

    assert not result_relevance_score(
        item=item,
        game="Elden Ring",
        question="Malenia Blade of Miquella boss weakness phase",
    )


def test_search_filters_generic_reddit_result_title() -> None:
    item = {
        "title": "Reddit - The heart of the internet",
        "url": "https://www.reddit.com/r/Eldenring/comments/abc/example",
        "content": "Malenia Blade of Miquella discussion.",
    }

    assert not result_relevance_score(
        item=item,
        game="Elden Ring",
        question="Malenia Blade of Miquella boss weakness phase",
    )


def test_search_canonicalizes_community_urls_for_deduplication() -> None:
    assert TavilySearchProvider._canonical_source_key(
        "https://steamcommunity.com/app/1245620/discussions/0/3426690213922967942?l=koreana"
    ) == "https://steamcommunity.com/app/1245620/discussions/0/3426690213922967942"


def test_search_limits_source_diversity() -> None:
    sources = [
        Source(title=f"Reddit {index}", url=f"https://www.reddit.com/r/Eldenring/comments/{index}", score=1 - index * 0.01)
        for index in range(4)
    ]

    selected = TavilySearchProvider._limit_source_diversity(sources, total_results=5)

    assert len(selected) == 2


def test_search_balances_strict_and_relaxed_sources() -> None:
    strict_sources = [
        Source(title="Malenia Wiki", url="https://eldenring.wiki.fextralife.com/Malenia", score=0.95),
    ]
    relaxed_sources = [
        strict_sources[0],
        Source(title="Malenia Steam", url="https://steamcommunity.com/app/1245620/discussions/0/1", score=0.7),
        Source(title="Malenia Reddit", url="https://www.reddit.com/r/Eldenring/comments/1", score=0.65),
    ]

    selected = TavilySearchProvider._balanced_sources(
        strict_sources=strict_sources,
        relaxed_sources=relaxed_sources,
        total_results=5,
        min_strict_results=3,
    )

    assert [source.title for source in selected] == ["Malenia Wiki", "Malenia Steam", "Malenia Reddit"]


def test_search_rerank_prefers_question_entity_in_title() -> None:
    focused = {
        "title": "Malenia Blade of Miquella Strategy",
        "url": "https://eldenring.wiki.fextralife.com/Malenia+Blade+of+Miquella",
        "content": "Elden Ring boss weakness and phase guide.",
    }
    generic = {
        "title": "Elden Ring Wiki",
        "url": "https://eldenring.wiki.fextralife.com/Elden+Ring+Wiki",
        "content": "General Elden Ring guide with Malenia mentioned once.",
    }

    assert result_relevance_score(
        item=focused,
        game="Elden Ring",
        question="Malenia 怎么打",
    ) > result_relevance_score(
        item=generic,
        game="Elden Ring",
        question="Malenia 怎么打",
    )


def test_search_extracts_relevant_passage_from_raw_content() -> None:
    content = "通用介绍。" * 300 + "玛莲妮亚弱点与二阶段水鸟乱舞躲避方法。" + "其他内容。" * 300

    evidence = TavilySearchProvider._best_evidence_passage(content, question="玛莲妮亚怎么打")

    assert "玛莲妮亚弱点" in evidence
    assert len(evidence) <= 1600


def test_source_evidence_preserves_relevant_search_summary_before_raw_navigation() -> None:
    built = build_source(
        item={
            "title": "Unseen Puzzle Game",
            "url": "https://example.com/wiki/unseen-puzzle",
            "content": "The Arc Rod picks up floor tiles and places them over reachable holes.",
            "raw_content": "Navigation menu and edit links. " * 300,
            "score": 0.8,
        },
        source_policy=SOURCE_POLICIES["wiki"],
        game="Unseen Puzzle Game",
        game_aliases=[],
        question="Arc Rod use",
        intent="item_usage",
        best_passage=TavilySearchProvider._best_evidence_passage,
        evidence_max_chars=500,
        version_safety_score=lambda **kwargs: 0.5,
        extract_version=lambda text: None,
        parse_datetime=lambda value: None,
    )

    assert built is not None
    assert "picks up floor tiles" in built.source.evidence


def test_search_version_sensitive_sources_prefer_official_or_versioned() -> None:
    official_score = TavilySearchProvider._version_safety_score(
        intent="patch",
        source_type="official",
        text="Elden Ring patch notes version 1.12",
    )
    old_community_score = TavilySearchProvider._version_safety_score(
        intent="patch",
        source_type="community",
        text="old build guide",
    )
    stable_location_score = TavilySearchProvider._version_safety_score(
        intent="item_location",
        source_type="wiki",
        text="Stonesword Key location",
    )

    assert official_score > old_community_score
    assert stable_location_score > old_community_score




class ContextCapturingLLM:
    def __init__(self):
        self.plan_history = []
        self.answer_history = []
        self.plan_game_resolution = None
        self.title_calls = 0

    async def plan_search(self, *, request, history, game_resolution=None):
        self.plan_history = history
        self.plan_game_resolution = game_resolution
        return SearchPlan()

    async def answer(self, *, request, sources, plan, history, game_resolution=None):
        self.answer_history = history
        return "带上下文回答"

    async def refine_search_plan(self, *, request, plan, sources, history, game_resolution=None):
        return None

    async def improve_answer(self, *, request, sources, answer, plan, history, game_resolution=None):
        return answer

    async def summarize_title(self, *, request, answer):
        self.title_calls += 1
        return "上下文会话"


async def test_agent_passes_recent_history_to_llm(monkeypatch) -> None:
    store = InMemoryConversationStore()
    session_id = uuid4()
    await store.save_chat(
        ChatRequest(game="Elden Ring", question="女武神怎么打？"),
        ChatResponse(session_id=session_id, answer="先熟悉二阶段。", sources=[], title="女武神打法"),
    )
    monkeypatch.setattr(agent_module, "conversation_store", store)
    llm = ContextCapturingLLM()
    agent = QuestAgent(search_provider=EmptySearchProvider(), llm=llm)

    response = await agent.run(ChatRequest(game="Elden Ring", question="钥匙在哪？", session_id=session_id))

    assert [message.role for message in llm.plan_history] == ["user", "assistant"]
    assert llm.answer_history[0].content == "女武神怎么打？"
    assert llm.plan_game_resolution.is_confirmed
    assert llm.title_calls == 0
    assert response.is_new is False


async def test_agent_returns_game_candidates_after_retrieval_has_no_evidence(monkeypatch) -> None:
    store = InMemoryConversationStore()
    monkeypatch.setattr(agent_module, "conversation_store", store)
    search_provider = AmbiguousGameSearchProvider()
    agent = QuestAgent(search_provider=search_provider)

    response = await agent.run(ChatRequest(game="Afterwards", question="灌铅骰子作用"))

    assert response.needs_game_confirmation is True
    assert [candidate.name for candidate in response.game_candidates] == ["Afterwards", "Afterwards Survival"]
    assert search_provider.search_called is False


def test_agent_status_messages_explain_current_work() -> None:
    assert QuestAgent._status_for_plan_start("未知复合问题") == "理解问题：识别目标和关键关系"
    assert QuestAgent._status_for_search(SearchPlan(intent="item_location")) == "类型：物品位置；查地点/条件/路线"
    assert QuestAgent._status_for_search(SearchPlan(intent="item_usage")) == "类型：物品用途；查效果/用法/交互对象"
    assert QuestAgent._status_for_search(SearchPlan(intent="game_mechanic")) == "类型：游戏机制；查开启条件/触发方式"
    assert QuestAgent._status_for_sources([]) == "来源筛选：未找到强相关资料"


def test_session_title_fallback_uses_game_and_first_question() -> None:
    title = GuideLLM._fallback_title("Elden Ring", "女武神怎么打？需要什么装备？")

    assert title == "Elden Ring, 女武神怎么打？需要什么装备？"


def test_session_title_cleaner_enforces_game_prefix() -> None:
    title = GuideLLM._clean_title("女武神打法", fallback="Elden Ring, 女武神怎么打？", game="Elden Ring")

    assert title == "Elden Ring, 女武神打法"
