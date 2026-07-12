from uuid import uuid4

from fastapi.testclient import TestClient

import main
from config import Settings
from model_providers import OpenAICompatibleProvider, create_model_provider
from schemas import ChatRequest, FeedbackRating, FeedbackRequest
from search import TavilySearchProvider


client = TestClient(main.app)


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
