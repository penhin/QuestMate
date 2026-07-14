from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json

import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import QuestAgent, quest_agent
from config import Settings, get_settings
from knowledge import knowledge_store
from schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    KnowledgeDocumentStatus,
    KnowledgeIndexRequest,
    KnowledgeIndexResponse,
    RenameSessionRequest,
    SessionResponse,
    SessionSummary,
    SessionsResponse,
)
from storage import conversation_store
from tasks import index_url
from uuid import UUID

logger = structlog.get_logger()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await conversation_store.init_schema()
        try:
            await knowledge_store.init_schema()
        except Exception:
            logger.warning("knowledge.schema_unavailable")
        yield

    app = FastAPI(title="QuestMate", version="0.1.0", lifespan=lifespan)
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "env": settings.app_env}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, agent: QuestAgent = Depends(lambda: quest_agent)):
        if request.stream:
            async def events() -> AsyncIterator[str]:
                try:
                    async for event_type, payload in agent.stream(request):
                        if event_type == "done" and isinstance(payload, ChatResponse):
                            body = payload.model_dump(mode="json")
                        else:
                            body = {"value": payload}
                        yield f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
                except Exception as exc:
                    logger.exception("chat.stream_failed")
                    body = {"message": str(exc)}
                    yield f"event: error\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        response = await agent.run(request)
        logger.info("chat.completed", session_id=str(response.session_id), source_count=len(response.sources))
        return response

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: UUID) -> SessionResponse:
        return await conversation_store.get_session(session_id)

    @app.get("/api/sessions", response_model=SessionsResponse)
    async def list_sessions() -> SessionsResponse:
        return await conversation_store.list_sessions()

    @app.patch("/api/sessions/{session_id}", response_model=SessionSummary)
    async def rename_session(session_id: UUID, request: RenameSessionRequest) -> SessionSummary:
        return await conversation_store.rename_session(session_id, request.title)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: UUID) -> None:
        await conversation_store.delete_session(session_id)

    @app.post("/api/feedback", response_model=FeedbackResponse)
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        await conversation_store.save_feedback(request)
        return FeedbackResponse()

    @app.post("/api/knowledge/documents", response_model=KnowledgeIndexResponse, status_code=status.HTTP_202_ACCEPTED)
    async def index_knowledge_document(request: KnowledgeIndexRequest) -> KnowledgeIndexResponse:
        try:
            task = index_url.delay(
                str(request.url),
                request.game,
                request.title,
                request.source_type,
            )
        except Exception as exc:
            logger.exception("knowledge.index_enqueue_failed")
            raise HTTPException(status_code=503, detail="索引队列暂不可用") from exc
        return KnowledgeIndexResponse(task_id=task.id)

    @app.get("/api/knowledge/documents", response_model=list[KnowledgeDocumentStatus])
    async def list_knowledge_documents(game: str | None = None) -> list[KnowledgeDocumentStatus]:
        try:
            return await knowledge_store.list_documents(game=game)
        except Exception as exc:
            logger.exception("knowledge.list_failed")
            raise HTTPException(status_code=503, detail="知识库暂不可用") from exc

    return app


app = create_app()
