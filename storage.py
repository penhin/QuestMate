from collections import defaultdict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from config import Settings, get_settings
from schemas import ChatRequest, ChatResponse, FeedbackRequest, SessionMessage, SessionResponse


class Database:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine: AsyncEngine = create_async_engine(self.settings.database_url, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._messages: dict[UUID, list[SessionMessage]] = defaultdict(list)
        self._feedback: list[FeedbackRequest] = []

    async def save_chat(self, request: ChatRequest, response: ChatResponse) -> None:
        self._messages[response.session_id].append(SessionMessage(role="user", content=request.question))
        self._messages[response.session_id].append(
            SessionMessage(role="assistant", content=response.answer, sources=response.sources)
        )

    async def get_session(self, session_id: UUID) -> SessionResponse:
        return SessionResponse(session_id=session_id, messages=self._messages.get(session_id, []))

    async def save_feedback(self, feedback: FeedbackRequest) -> None:
        self._feedback.append(feedback)


conversation_store = InMemoryConversationStore()

