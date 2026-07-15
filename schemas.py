from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl

from quality_policy import GAME_RESOLUTION_POLICY


class Source(BaseModel):
    title: str
    url: HttpUrl
    snippet: str | None = None
    score: float | None = Field(default=None, ge=0)
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    trust_score: float = Field(default=0.5, ge=0, le=1)
    trust_label: str = "普通"
    evidence: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    game_version: str | None = Field(default=None, max_length=80)


class PlannedSearchQuery(BaseModel):
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    query: str = Field(min_length=1, max_length=240)


class GameCandidate(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list, max_length=6)
    tags: list[str] = Field(default_factory=list, max_length=5)
    platform_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
    database_domains: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0, ge=0, le=1)


class GameResolution(BaseModel):
    input_name: str = ""
    confirmed_name: str = ""
    aliases: list[str] = Field(default_factory=list, max_length=8)
    platform_urls: list[HttpUrl] = Field(default_factory=list, max_length=6)
    official_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
    database_domains: list[str] = Field(default_factory=list, max_length=8)
    candidates: list[GameCandidate] = Field(default_factory=list, max_length=6)
    confidence: float = Field(default=0, ge=0, le=1)
    ambiguous: bool = False

    @property
    def is_confirmed(self) -> bool:
        return self.confidence >= GAME_RESOLUTION_POLICY.confirmed_threshold and bool(
            self.confirmed_name or self.aliases or self.platform_urls or self.database_domains
        )


SearchIntent = Literal[
    "boss_strategy",
    "item_location",
    "item_usage",
    "quest_step",
    "game_mechanic",
    "build",
    "patch",
    "lore",
    "general",
]


class SearchPlan(BaseModel):
    intent: SearchIntent = "general"
    aliases: list[str] = Field(default_factory=list, max_length=6)
    queries: list[PlannedSearchQuery] = Field(default_factory=list, max_length=6)
    missing_info: list[str] = Field(default_factory=list, max_length=4)
    refinement: bool = False


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    game: str = Field(min_length=1, max_length=120)
    session_id: UUID | None = None
    stream: bool = False
    ai_provider: Literal["anthropic", "deepseek"] = "anthropic"
    ai_api_key: str | None = Field(default=None, max_length=400)
    ai_model: str | None = Field(default=None, max_length=120)
    ai_base_url: str | None = Field(default=None, max_length=300)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[Source] = Field(default_factory=list)
    title: str | None = None
    is_new: bool = False
    needs_game_confirmation: bool = False
    game_candidates: list[GameCandidate] = Field(default_factory=list)


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[Source] = Field(default_factory=list)


class SessionResponse(BaseModel):
    session_id: UUID
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: UUID
    title: str
    message_count: int = 0
    updated_at: datetime | None = None


class SessionsResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)


class FeedbackRating(str, Enum):
    helpful = "helpful"
    unhelpful = "unhelpful"
    inaccurate = "inaccurate"


class FeedbackRequest(BaseModel):
    session_id: UUID
    message_id: UUID = Field(default_factory=uuid4)
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    accepted: bool = True


class KnowledgeIndexRequest(BaseModel):
    url: HttpUrl
    game: str = Field(min_length=1, max_length=120)
    title: str | None = Field(default=None, max_length=300)
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    game_version: str | None = Field(default=None, max_length=80)
    published_at: datetime | None = None


class KnowledgeIndexResponse(BaseModel):
    task_id: str
    status: Literal["queued"] = "queued"


class KnowledgeDocumentStatus(BaseModel):
    url: HttpUrl
    game: str
    title: str | None = None
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    game_version: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    status: Literal["queued", "indexing", "ready", "failed"]
    chunk_count: int = 0
    error: str | None = None
    updated_at: datetime | None = None
