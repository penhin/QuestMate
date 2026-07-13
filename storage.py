from collections import defaultdict
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, MetaData, String, Table, Text, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from config import Settings, get_settings
from schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    SessionMessage,
    SessionResponse,
    SessionSummary,
    SessionsResponse,
)


metadata = MetaData()


def StringColumn(name: str, length: int, *args, **kwargs):
    return Column(name, String(length), *args, **kwargs)


def DateTimeColumn(name: str):
    return Column(
        name,
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


conversation_sessions = Table(
    "conversation_sessions",
    metadata,
    StringColumn("session_id", 36, primary_key=True),
    StringColumn("title", 60, nullable=False),
    Column("title_is_custom", Boolean, nullable=False, default=False),
    DateTimeColumn("created_at"),
    DateTimeColumn("updated_at"),
)

conversation_messages = Table(
    "conversation_messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    StringColumn("session_id", 36, ForeignKey("conversation_sessions.session_id", ondelete="CASCADE"), nullable=False),
    StringColumn("role", 24, nullable=False),
    Column("content", Text, nullable=False),
    Column("sources", JSONB, nullable=False),
    DateTimeColumn("created_at"),
)

conversation_feedback = Table(
    "conversation_feedback",
    metadata,
    StringColumn("message_id", 36, primary_key=True),
    StringColumn("session_id", 36, nullable=False),
    StringColumn("rating", 24, nullable=False),
    Column("comment", Text),
    DateTimeColumn("created_at"),
)


class Database:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine: AsyncEngine = create_async_engine(self.settings.database_url, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._messages: dict[UUID, list[SessionMessage]] = defaultdict(list)
        self._titles: dict[UUID, str] = {}
        self._feedback: list[FeedbackRequest] = []

    async def init_schema(self) -> None:
        return None

    async def save_chat(self, request: ChatRequest, response: ChatResponse) -> None:
        self._messages[response.session_id].append(SessionMessage(role="user", content=request.question))
        self._messages[response.session_id].append(
            SessionMessage(role="assistant", content=response.answer, sources=response.sources)
        )
        if response.title:
            self._titles[response.session_id] = response.title

    async def get_session(self, session_id: UUID) -> SessionResponse:
        return SessionResponse(session_id=session_id, messages=self._messages.get(session_id, []))

    async def get_recent_messages(self, session_id: UUID, limit: int = 8) -> list[SessionMessage]:
        return self._messages.get(session_id, [])[-limit:]

    async def session_exists(self, session_id: UUID) -> bool:
        return session_id in self._messages

    async def list_sessions(self) -> SessionsResponse:
        summaries = []
        for session_id, messages in self._messages.items():
            updated_at = messages[-1].created_at if messages else None
            title = self._titles.get(session_id) or self._fallback_title(messages)
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    title=title,
                    message_count=len(messages),
                    updated_at=updated_at,
                )
            )

        return SessionsResponse(sessions=sort_summaries(summaries))

    async def rename_session(self, session_id: UUID, title: str) -> SessionSummary:
        self._titles[session_id] = title.strip()
        messages = self._messages.get(session_id, [])
        return SessionSummary(
            session_id=session_id,
            title=self._titles[session_id],
            message_count=len(messages),
            updated_at=messages[-1].created_at if messages else None,
        )

    async def delete_session(self, session_id: UUID) -> None:
        self._messages.pop(session_id, None)
        self._titles.pop(session_id, None)

    async def save_feedback(self, feedback: FeedbackRequest) -> None:
        self._feedback.append(feedback)

    @staticmethod
    def _fallback_title(messages: list[SessionMessage]) -> str:
        for message in messages:
            if message.role == "user" and message.content.strip():
                return message.content.strip()[:28]
        return "未命名会话"


class PostgresConversationStore:
    def __init__(self, database: Database | None = None, fallback: InMemoryConversationStore | None = None) -> None:
        self.database = database or Database()
        self.fallback = fallback or InMemoryConversationStore()
        self._using_fallback = False
        self._initialized = False

    async def init_schema(self) -> None:
        if self._initialized:
            return

        try:
            await self.database.init_schema()
            self._using_fallback = False
        except Exception:
            self._using_fallback = True
            await self.fallback.init_schema()
        finally:
            self._initialized = True

    async def save_chat(self, request: ChatRequest, response: ChatResponse) -> None:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.save_chat(request, response)

        now = datetime.now(timezone.utc)
        async with self.database.sessionmaker() as session:
            async with session.begin():
                existing = await session.execute(
                    select(conversation_sessions.c.session_id).where(
                        conversation_sessions.c.session_id == str(response.session_id)
                    )
                )
                if existing.scalar_one_or_none() is None:
                    title = response.title or self._fallback_title_from_question(request.question)
                    await session.execute(
                        insert(conversation_sessions).values(
                            session_id=str(response.session_id),
                            title=title,
                            title_is_custom=False,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    current = (
                        await session.execute(
                            select(conversation_sessions.c.title_is_custom).where(
                                conversation_sessions.c.session_id == str(response.session_id)
                            )
                        )
                    ).scalar_one_or_none()
                    values = {"updated_at": now}
                    if response.title and not current:
                        values["title"] = response.title
                    await session.execute(
                        update(conversation_sessions)
                        .where(conversation_sessions.c.session_id == str(response.session_id))
                        .values(**values)
                    )

                await session.execute(
                    insert(conversation_messages),
                    [
                        {
                            "session_id": str(response.session_id),
                            "role": "user",
                            "content": request.question,
                            "sources": [],
                            "created_at": now,
                        },
                        {
                            "session_id": str(response.session_id),
                            "role": "assistant",
                            "content": response.answer,
                            "sources": [source.model_dump(mode="json") for source in response.sources],
                            "created_at": now,
                        },
                    ],
                )

    async def get_session(self, session_id: UUID) -> SessionResponse:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.get_session(session_id)

        async with self.database.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(conversation_messages)
                    .where(conversation_messages.c.session_id == str(session_id))
                    .order_by(conversation_messages.c.id.asc())
                )
            ).mappings()

            return SessionResponse(
                session_id=session_id,
                messages=[
                    SessionMessage(
                        role=row["role"],
                        content=row["content"],
                        created_at=row["created_at"],
                        sources=row["sources"],
                    )
                    for row in rows
                ],
            )

    async def get_recent_messages(self, session_id: UUID, limit: int = 8) -> list[SessionMessage]:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.get_recent_messages(session_id, limit=limit)

        async with self.database.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(conversation_messages)
                    .where(conversation_messages.c.session_id == str(session_id))
                    .order_by(conversation_messages.c.id.desc())
                    .limit(limit)
                )
            ).mappings()

            messages = [
                SessionMessage(
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    sources=row["sources"],
                )
                for row in rows
            ]
            return list(reversed(messages))

    async def session_exists(self, session_id: UUID) -> bool:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.session_exists(session_id)

        async with self.database.sessionmaker() as session:
            existing = await session.execute(
                select(conversation_sessions.c.session_id).where(
                    conversation_sessions.c.session_id == str(session_id)
                )
            )
            return existing.scalar_one_or_none() is not None

    async def list_sessions(self) -> SessionsResponse:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.list_sessions()

        async with self.database.sessionmaker() as session:
            message_counts = (
                select(
                    conversation_messages.c.session_id,
                    func.count(conversation_messages.c.id).label("message_count"),
                )
                .group_by(conversation_messages.c.session_id)
                .subquery()
            )
            rows = (
                await session.execute(
                    select(
                        conversation_sessions.c.session_id,
                        conversation_sessions.c.title,
                        conversation_sessions.c.updated_at,
                        func.coalesce(message_counts.c.message_count, 0).label("message_count"),
                    )
                    .outerjoin(
                        message_counts,
                        conversation_sessions.c.session_id == message_counts.c.session_id,
                    )
                    .order_by(conversation_sessions.c.updated_at.desc())
                )
            ).mappings()

            return SessionsResponse(
                sessions=[
                    SessionSummary(
                        session_id=UUID(row["session_id"]),
                        title=row["title"],
                        message_count=row["message_count"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
            )

    async def rename_session(self, session_id: UUID, title: str) -> SessionSummary:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.rename_session(session_id, title)

        async with self.database.sessionmaker() as session:
            async with session.begin():
                await session.execute(
                    update(conversation_sessions)
                    .where(conversation_sessions.c.session_id == str(session_id))
                    .values(title=title.strip(), title_is_custom=True, updated_at=datetime.now(timezone.utc))
                )

        sessions = await self.list_sessions()
        return next(
            (summary for summary in sessions.sessions if summary.session_id == session_id),
            SessionSummary(session_id=session_id, title=title.strip()),
        )

    async def delete_session(self, session_id: UUID) -> None:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.delete_session(session_id)

        async with self.database.sessionmaker() as session:
            async with session.begin():
                await session.execute(
                    delete(conversation_sessions).where(conversation_sessions.c.session_id == str(session_id))
                )

    async def save_feedback(self, feedback: FeedbackRequest) -> None:
        await self.init_schema()
        if self._using_fallback:
            return await self.fallback.save_feedback(feedback)

        async with self.database.sessionmaker() as session:
            async with session.begin():
                await session.execute(
                    insert(conversation_feedback).values(
                        message_id=str(feedback.message_id),
                        session_id=str(feedback.session_id),
                        rating=feedback.rating.value,
                        comment=feedback.comment,
                        created_at=datetime.now(timezone.utc),
                    )
                )

    @staticmethod
    def _fallback_title_from_question(question: str) -> str:
        return question.strip()[:28] or "未命名会话"


def sort_summaries(summaries: list[SessionSummary]) -> list[SessionSummary]:
    return sorted(
        summaries,
        key=lambda summary: summary.updated_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


conversation_store = PostgresConversationStore()
