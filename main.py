from collections.abc import AsyncIterator

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import QuestAgent, quest_agent
from config import Settings, get_settings
from schemas import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, SessionResponse
from storage import conversation_store
from uuid import UUID

logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(title="QuestMate", version="0.1.0")
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
                response = await agent.run(request)
                yield f"data: {response.model_dump_json()}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        response = await agent.run(request)
        logger.info("chat.completed", session_id=str(response.session_id), source_count=len(response.sources))
        return response

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: UUID) -> SessionResponse:
        return await conversation_store.get_session(session_id)

    @app.post("/api/feedback", response_model=FeedbackResponse)
    async def feedback(request: FeedbackRequest) -> FeedbackResponse:
        await conversation_store.save_feedback(request)
        return FeedbackResponse()

    return app


app = create_app()
