from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal
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
    official_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
    identity_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
    database_domains: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0, ge=0, le=1)


class GameResolution(BaseModel):
    input_name: str = ""
    confirmed_name: str = ""
    aliases: list[str] = Field(default_factory=list, max_length=8)
    platform_urls: list[HttpUrl] = Field(default_factory=list, max_length=6)
    official_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
    identity_urls: list[HttpUrl] = Field(default_factory=list, max_length=4)
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

NamedEntityAliasGroup = Annotated[list[str], Field(min_length=1, max_length=4)]


class SearchPlan(BaseModel):
    intent: SearchIntent = "general"
    version_sensitive: bool = False
    named_entity_groups: list[NamedEntityAliasGroup] = Field(default_factory=list, max_length=4)
    aliases: list[str] = Field(default_factory=list, max_length=6)
    queries: list[PlannedSearchQuery] = Field(default_factory=list, max_length=6)
    missing_info: list[str] = Field(default_factory=list, max_length=4)
    refinement: bool = False


class EvidenceFact(BaseModel):
    statement: str = Field(min_length=1, max_length=500)
    source_indexes: list[int] = Field(default_factory=list, max_length=6)


EvidenceGapKind = Literal[
    "game_identity",
    "entity_identity",
    "premise",
    "direct_answer",
    "prerequisite",
    "acquisition",
    "access_route",
    "ordered_actions",
    "outcome",
    "version",
    "conflict",
    "semantic_distinction",
    "other",
]


class EvidenceGap(BaseModel):
    """A typed missing link that can drive a materially different search."""

    kind: EvidenceGapKind = "other"
    description: str = Field(min_length=1, max_length=300)
    query_hint: str | None = Field(default=None, max_length=240)
    source_type: Literal["official", "wiki", "community", "web"] = "web"
    priority: int = Field(default=3, ge=1, le=5)


class InvestigationState(BaseModel):
    """Request-scoped state for following an evidence dependency chain."""

    goal: str = Field(min_length=1, max_length=1000)
    known_facts: list[EvidenceFact] = Field(default_factory=list, max_length=12)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list, max_length=6)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=6)
    attempted_queries: list[str] = Field(default_factory=list, max_length=16)
    next_queries: list[PlannedSearchQuery] = Field(default_factory=list, max_length=2)
    aliases: list[str] = Field(default_factory=list, max_length=6)
    complete: bool = False
    hop_count: int = Field(default=0, ge=0, le=10)
    stop_reason: Literal["complete", "needs_search", "budget_exhausted", "insufficient_evidence"] | None = None


class AnswerCompletenessAssessment(BaseModel):
    complete: bool = False
    gaps: list[str] = Field(default_factory=list, max_length=6)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=6)
    irrelevant_details: list[str] = Field(default_factory=list, max_length=6)


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
