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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    game: str = Field(min_length=1, max_length=120)
    session_id: UUID | None = None
    stream: bool = False
    ai_provider: Literal["anthropic"] = "anthropic"
    ai_api_key: str | None = Field(default=None, max_length=400)
    ai_model: str | None = Field(default=None, max_length=120)
    ai_base_url: str | None = Field(default=None, max_length=300)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[Source] = Field(default_factory=list)


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[Source] = Field(default_factory=list)


class SessionResponse(BaseModel):
    session_id: UUID
    messages: list[SessionMessage] = Field(default_factory=list)


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
