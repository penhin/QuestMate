from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

import agent as agent_module
import main
from config import Settings
from model_providers import OpenAICompatibleProvider, create_model_provider
from agent import QuestAgent
from schemas import ChatRequest, ChatResponse, FeedbackRating, FeedbackRequest, SearchPlan
from search import TavilySearchProvider
from storage import InMemoryConversationStore


client = TestClient(main.app)


@pytest.fixture(autouse=True)
def isolated_conversation_store(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(main, "conversation_store", store)
    monkeypatch.setattr(agent_module, "conversation_store", store)
    yield store


class EmptySearchProvider:
    async def search(self, query: str, game: str, max_results: int | None = None, plan=None):
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
    monkeypatch.setattr(main.quest_agent, "search_provider", EmptySearchProvider())

    response = client.post(
        "/api/chat",
        json={"game": "Elden Ring", "question": "新手先打哪里？"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert "Elden Ring" in body["answer"]
    assert body["sources"] == []


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
                        "content": "Wiki answer",
                        "score": 0.9,
                    }
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


def test_search_filters_unrelated_game_results() -> None:
    item = {
        "title": "Dead by Daylight Wiki",
        "url": "https://deadbydaylight.fandom.com/wiki/Keys",
        "content": "A horror game wiki page.",
    }

    assert not TavilySearchProvider._is_relevant_result(
        item=item,
        game="艾尔登法环",
        question="地下墓室钥匙在哪里",
    )


def test_deepseek_uses_openai_compatible_provider() -> None:
    request = ChatRequest(
        game="Elden Ring",
        question="新手先打哪里？",
        ai_provider="deepseek",
        ai_api_key="test-key",
        ai_model="deepseek-chat",
        ai_base_url="https://api.deepseek.com",
    )

    provider = create_model_provider(request=request, settings=Settings())

    assert isinstance(provider, OpenAICompatibleProvider)


class ContextCapturingLLM:
    def __init__(self):
        self.plan_history = []
        self.answer_history = []

    async def plan_search(self, *, request, history):
        self.plan_history = history
        return SearchPlan()

    async def answer(self, *, request, sources, history):
        self.answer_history = history
        return "带上下文回答"

    async def summarize_title(self, *, request, answer):
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

    await agent.run(ChatRequest(game="Elden Ring", question="钥匙在哪？", session_id=session_id))

    assert [message.role for message in llm.plan_history] == ["user", "assistant"]
    assert llm.answer_history[0].content == "女武神怎么打？"
