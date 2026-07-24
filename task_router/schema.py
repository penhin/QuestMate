"""Typed task-routing artifacts shared by QuestMate workflows."""

from typing import Literal

from pydantic import BaseModel, Field

from schemas import SearchIntent

WorkflowName = Literal["guide", "build", "analysis"]


class RouteConstraints(BaseModel):
    version_sensitive: bool = False
    requires_relation_verification: bool = False
    safety_refusal: bool = False
    answer_requirements: list[str] = Field(default_factory=list, max_length=4)
    missing_info: list[str] = Field(default_factory=list, max_length=4)


class RouteDecision(BaseModel):
    intent: WorkflowName = "analysis"
    confidence: float = Field(default=0, ge=0, le=1)
    game: str = Field(default="", max_length=120)
    entities: list[str] = Field(default_factory=list, max_length=16)
    constraints: RouteConstraints = Field(default_factory=RouteConstraints)
    search_intent: SearchIntent = "general"
