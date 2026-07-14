from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class Source(BaseModel):
    title: str
    url: HttpUrl
    snippet: str | None = None
    score: float | None = Field(default=None, ge=0)
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    trust_score: float = Field(default=0.5, ge=0, le=1)
    trust_label: str = "普通"


class PlannedSearchQuery(BaseModel):
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    query: str = Field(min_length=1, max_length=240)


SearchIntent = Literal["boss_strategy", "item_location", "quest_step", "build", "patch", "lore", "general"]


class SearchPlan(BaseModel):
    intent: SearchIntent = "general"
    aliases: list[str] = Field(default_factory=list, max_length=6)
    queries: list[PlannedSearchQuery] = Field(default_factory=list, max_length=4)
    missing_info: list[str] = Field(default_factory=list, max_length=4)


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
