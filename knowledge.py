"""Durable game-guide knowledge indexing and retrieval.

The index deliberately has two retrieval paths: pgvector cosine similarity when
an embedding endpoint is configured, and keyword scoring as a deployment-safe
fallback. This keeps locally indexed material useful before embedding credentials
have been provisioned.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import trafilatura
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Table, Text, delete, func, insert, select, text, update

from config import Settings, get_settings
from schemas import KnowledgeDocumentStatus, Source
from storage import Database, metadata


knowledge_documents = Table(
    "knowledge_documents",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("url", String(2048), nullable=False, unique=True),
    Column("game", String(120), nullable=False),
    Column("title", String(300)),
    Column("source_type", String(16), nullable=False, default="web"),
    Column("status", String(16), nullable=False, default="queued"),
    Column("error", Text),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
Index("ix_knowledge_documents_game_status", knowledge_documents.c.game, knowledge_documents.c.status)

knowledge_chunks = Table(
    "knowledge_chunks",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(1536), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
Index("ix_knowledge_chunks_document", knowledge_chunks.c.document_id)


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.embedding_api_key and self.settings.embedding_base_url)

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        if not texts or not self.enabled:
            return [None] * len(texts)
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.settings.embedding_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.settings.embedding_api_key}"},
                json={"model": self.settings.embedding_model, "input": texts},
            )
            response.raise_for_status()
        data = response.json().get("data", [])
        vectors = [item.get("embedding") for item in data]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise ValueError("Embedding endpoint returned an invalid response")
        if any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
            raise ValueError(f"Embedding dimensions must equal {self.settings.embedding_dimensions}")
        return vectors


class KnowledgeStore:
    def __init__(self, database: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.embedding_dimensions != 1536:
            raise ValueError("EMBEDDING_DIMENSIONS must be 1536 until a database migration changes the vector column")
        self.database = database or Database(self.settings)
        self.embeddings = EmbeddingClient(self.settings)
        self._initialized = False
        self._available = False

    async def init_schema(self) -> None:
        if self._initialized:
            return
        async with self.database.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(metadata.create_all)
        self._initialized = True
        self._available = True

    async def index_url(self, *, url: str, game: str, title: str | None = None, source_type: str = "web") -> dict[str, Any]:
        await self.init_schema()
        document_id = await self._upsert_document(url=url, game=game, title=title, source_type=source_type, status="indexing")
        try:
            extracted_title, content = await self._fetch_and_extract(url)
            chunks = chunk_text(content)
            if not chunks:
                raise ValueError("No extractable article content found")
            vectors = await self.embeddings.embed(chunks)
            now = datetime.now(timezone.utc)
            async with self.database.sessionmaker() as session:
                async with session.begin():
                    await session.execute(delete(knowledge_chunks).where(knowledge_chunks.c.document_id == document_id))
                    await session.execute(
                        insert(knowledge_chunks),
                        [
                            {
                                "id": str(uuid4()),
                                "document_id": document_id,
                                "chunk_index": index,
                                "content": chunk,
                                "embedding": vector,
                                "created_at": now,
                            }
                            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                        ],
                    )
                    await session.execute(
                        update(knowledge_documents)
                        .where(knowledge_documents.c.id == document_id)
                        .values(title=title or extracted_title, status="ready", error=None, chunk_count=len(chunks), updated_at=now)
                    )
            return {"status": "ready", "document_id": document_id, "chunk_count": len(chunks)}
        except Exception as exc:
            async with self.database.sessionmaker() as session:
                async with session.begin():
                    await session.execute(
                        update(knowledge_documents)
                        .where(knowledge_documents.c.id == document_id)
                        .values(status="failed", error=str(exc)[:2000], updated_at=datetime.now(timezone.utc))
                    )
            raise

    async def retrieve(self, *, game: str, query: str, limit: int | None = None) -> list[Source]:
        if not self._available:
            return []
        try:
            limit = limit or self.settings.knowledge_retrieval_results
            vector = (await self.embeddings.embed([query]))[0]
            if vector is not None:
                rows = await self._vector_rows(game=game, vector=vector, limit=limit)
            else:
                rows = await self._keyword_rows(game=game, query=query, limit=limit)
            return [self._source_from_row(row) for row in rows]
        except Exception:
            # Search should remain available while the index is unavailable.
            return []

    async def list_documents(self, *, game: str | None = None, limit: int = 50) -> list[KnowledgeDocumentStatus]:
        await self.init_schema()
        statement = select(knowledge_documents).order_by(knowledge_documents.c.updated_at.desc()).limit(limit)
        if game:
            statement = statement.where(func.lower(knowledge_documents.c.game) == game.lower())
        async with self.database.sessionmaker() as session:
            rows = (await session.execute(statement)).mappings()
            return [KnowledgeDocumentStatus(**dict(row)) for row in rows]

    async def _upsert_document(self, *, url: str, game: str, title: str | None, source_type: str, status: str) -> str:
        now = datetime.now(timezone.utc)
        async with self.database.sessionmaker() as session:
            async with session.begin():
                existing = (await session.execute(select(knowledge_documents.c.id).where(knowledge_documents.c.url == url))).scalar_one_or_none()
                if existing:
                    await session.execute(update(knowledge_documents).where(knowledge_documents.c.id == existing).values(game=game, title=title, source_type=source_type, status=status, error=None, updated_at=now))
                    return existing
                document_id = str(uuid4())
                await session.execute(insert(knowledge_documents).values(id=document_id, url=url, game=game, title=title, source_type=source_type, status=status, error=None, chunk_count=0, created_at=now, updated_at=now))
                return document_id

    async def _fetch_and_extract(self, url: str) -> tuple[str | None, str]:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": "QuestMateIndexer/0.1"}) as client:
            response = await client.get(url)
            response.raise_for_status()
        downloaded = response.text
        content = trafilatura.extract(downloaded, include_comments=False, include_tables=True, favor_precision=True) or ""
        metadata = trafilatura.extract_metadata(downloaded)
        return (metadata.title if metadata else None), content

    async def _vector_rows(self, *, game: str, vector: list[float], limit: int) -> list[dict[str, Any]]:
        distance = knowledge_chunks.c.embedding.cosine_distance(vector).label("distance")
        statement = (
            select(knowledge_documents, knowledge_chunks.c.content, distance)
            .join(knowledge_chunks, knowledge_chunks.c.document_id == knowledge_documents.c.id)
            .where(knowledge_documents.c.status == "ready", func.lower(knowledge_documents.c.game) == game.lower(), knowledge_chunks.c.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        async with self.database.sessionmaker() as session:
            return [dict(row) for row in (await session.execute(statement)).mappings()]

    async def _keyword_rows(self, *, game: str, query: str, limit: int) -> list[dict[str, Any]]:
        terms = keyword_terms(query)
        statement = (
            select(knowledge_documents, knowledge_chunks.c.content)
            .join(knowledge_chunks, knowledge_chunks.c.document_id == knowledge_documents.c.id)
            .where(knowledge_documents.c.status == "ready", func.lower(knowledge_documents.c.game) == game.lower())
            .limit(80)
        )
        async with self.database.sessionmaker() as session:
            rows = [dict(row) for row in (await session.execute(statement)).mappings()]
        for row in rows:
            haystack = f"{row.get('title') or ''} {row['content']}".lower()
            row["keyword_score"] = sum(haystack.count(term) for term in terms)
        return sorted((row for row in rows if row["keyword_score"] > 0), key=lambda row: row["keyword_score"], reverse=True)[:limit]

    @staticmethod
    def _source_from_row(row: dict[str, Any]) -> Source:
        source_type = row["source_type"] if row["source_type"] in {"official", "wiki", "community", "web"} else "web"
        trust = {"official": (0.95, "官方"), "wiki": (0.8, "百科"), "community": (0.55, "社区"), "web": (0.65, "知识库")}[source_type]
        score = 1 - float(row["distance"]) if row.get("distance") is not None else min(0.95, 0.5 + row.get("keyword_score", 0) * 0.08)
        return Source(title=row.get("title") or urlparse(row["url"]).netloc, url=row["url"], snippet=row["content"][:900], score=max(0, score), source_type=source_type, trust_score=trust[0], trust_label=trust[1])


def chunk_text(content: str, *, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    cleaned = re.sub(r"\s+", " ", content).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        if end < len(cleaned):
            boundary = max(cleaned.rfind("。", start, end), cleaned.rfind(". ", start, end), cleaned.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def keyword_terms(query: str) -> list[str]:
    lowered = query.lower()
    latin = re.findall(r"[a-z0-9]{3,}", lowered)
    chinese = [lowered[index : index + 2] for index in range(len(lowered) - 1) if re.fullmatch(r"[\u4e00-\u9fff]{2}", lowered[index : index + 2])]
    return list(dict.fromkeys([*latin, *chinese]))


knowledge_store = KnowledgeStore()
