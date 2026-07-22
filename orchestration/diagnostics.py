"""Evaluation-only aggregate diagnostics; never include prompts or source bodies."""

import re
from typing import Any

from agents import AgentTrace
from ai.evidence_policy import evidence_level, evidence_question
from schemas import ChatRequest, GameResolution, SearchPlan, Source


def evaluation_diagnostics(
    *,
    request: ChatRequest,
    path: str,
    llm: Any,
    sources: list[Source] | None = None,
    plan: SearchPlan | None = None,
    game_resolution: GameResolution | None = None,
    answer: str = "",
    agent_trace: list[AgentTrace] | None = None,
) -> dict[str, str | int]:
    if not request.metadata.get("evaluation"):
        return {}
    source_list = sources or []
    level = "none"
    if source_list:
        level = evidence_level(question=evidence_question(request=request, plan=plan), sources=source_list)
    diagnostics: dict[str, str | int] = {
        "path": path,
        "evidence_level": level,
        "source_count": len(source_list),
        "citation_count": len(re.findall(r"\[(\d+)\]", answer)),
        "agent_handoffs": len(agent_trace or []),
    }
    if path != "answer":
        return diagnostics
    conservative = getattr(llm, "_should_return_conservative_answer", None)
    if callable(conservative):
        diagnostics["policy_conservative"] = int(bool(conservative(
            request=request, sources=source_list, plan=plan, game_resolution=game_resolution
        )))
    claim_context = getattr(llm, "_citation_claim_context", None)
    if callable(claim_context):
        try:
            context = claim_context(
                question=evidence_question(request=request, plan=plan),
                sources=source_list,
                entity_groups=plan.named_entity_groups if plan else None,
                aliases=plan.aliases if plan else None,
            )
            diagnostics["claim_count"] = context.count("<claim id=")
        except (TypeError, ValueError):
            diagnostics["claim_count"] = 0
    return diagnostics
