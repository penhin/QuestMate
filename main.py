from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import secrets

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, status
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
    GameResolution,
    KnowledgeDocumentStatus,
    KnowledgeIndexRequest,
    KnowledgeIndexResponse,
    RenameSessionRequest,
    SessionResponse,
    SessionSummary,
    SessionsResponse,
)
from storage import conversation_store, shared_database
from source_registry import game_source_registry
from tasks import index_url
from outbound_http import validate_public_https_url
from uuid import UUID, uuid4

logger = structlog.get_logger()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await conversation_store.init_schema()
        try:
            await knowledge_store.init_schema()
        except Exception:
            logger.warning("knowledge.schema_unavailable")
        try:
            await game_source_registry.init_schema()
        except Exception:
            logger.warning("source_registry.schema_unavailable")
        try:
            yield
        finally:
            wait_for_indexes = getattr(quest_agent.search_provider, "wait_for_background_tasks", None)
            try:
                if callable(wait_for_indexes):
                    await wait_for_indexes()
            finally:
                try:
                    await knowledge_store.aclose()
                finally:
                    await shared_database.engine.dispose()

    app = FastAPI(title="QuestMate", version="0.1.0", lifespan=lifespan)
    settings = get_settings()

    def require_admin(x_questmate_admin_token: str | None = Header(default=None)) -> None:
        configured = settings.knowledge_admin_token
        if not configured and not settings.is_production:
            return
        if not configured or not x_questmate_admin_token or not secrets.compare_digest(
            configured,
            x_questmate_admin_token,
        ):
            raise HTTPException(status_code=403, detail="需要管理员凭据")

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

        try:
            response = await agent.run(request)
        except Exception as exc:
            # Search, storage, and model providers are external dependencies.
            # A transient failure must not be represented as a fabricated game
            # answer or an opaque HTTP 500 to interactive clients/evaluators.
            logger.exception("chat.failed", error_type=type(exc).__name__)
            return ChatResponse(
                session_id=request.session_id or uuid4(),
                answer=(
                    "这次查询的资料服务暂时不可用，因此我无法可靠确认答案。"
                    "请稍后重试；在恢复前我不会猜测具体地点、步骤或版本结论。"
                ),
                sources=[],
                is_new=request.session_id is None,
            )
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

    @app.post(
        "/api/knowledge/documents",
        response_model=KnowledgeIndexResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_admin)],
    )
    async def index_knowledge_document(request: KnowledgeIndexRequest) -> KnowledgeIndexResponse:
        try:
            safe_url = await validate_public_https_url(str(request.url))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            task = index_url.delay(
                safe_url,
                request.game,
                request.title,
                request.source_type,
                request.game_version,
                request.published_at.isoformat() if request.published_at else None,
            )
        except Exception as exc:
            logger.exception("knowledge.index_enqueue_failed")
            raise HTTPException(status_code=503, detail="索引队列暂不可用") from exc
        return KnowledgeIndexResponse(task_id=task.id)

    @app.get(
        "/api/knowledge/documents",
        response_model=list[KnowledgeDocumentStatus],
        dependencies=[Depends(require_admin)],
    )
    async def list_knowledge_documents(game: str | None = None) -> list[KnowledgeDocumentStatus]:
        try:
            return await knowledge_store.list_documents(game=game)
        except Exception as exc:
            logger.exception("knowledge.list_failed")
            raise HTTPException(status_code=503, detail="知识库暂不可用") from exc

    @app.get(
        "/api/source-registry",
        response_model=list[GameResolution],
        dependencies=[Depends(require_admin)],
    )
    async def list_source_registry() -> list[GameResolution]:
        return await game_source_registry.list_resolutions()

    return app


app = create_app()
