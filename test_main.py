from uuid import uuid4

from fastapi.testclient import TestClient

from main import app
from schemas import ChatRequest, FeedbackRating, FeedbackRequest


client = TestClient(app)


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


def test_chat_endpoint_returns_fallback_answer_without_api_keys() -> None:
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
