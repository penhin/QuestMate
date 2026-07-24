"""Deterministic claim eligibility and citation-safe answer rendering."""

import json
import re
from collections.abc import Callable

import structlog

from ai.citation_claims import build_citation_claims, claim_ids_cover_entity_groups
from ai.evidence_policy import evidence_entity_groups, evidence_question, has_question_specific_sources
from ai.search_plan_json import first_json_object
from retrieval.source_quality import token_in_text
from schemas import ChatRequest, CitationClaim, SearchPlan, Source


logger = structlog.get_logger()
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def order_citations_by_appearance(answer: str, sources: list[Source]) -> tuple[str, list[Source]]:
    """Renumber citations and sources in the order a player encounters them."""
    encountered = [
        int(match.group(1))
        for match in _CITATION_PATTERN.finditer(answer)
        if 1 <= int(match.group(1)) <= len(sources)
    ]
    ordered_indexes = list(dict.fromkeys(encountered))
    if not ordered_indexes:
        return answer, sources
    remap = {source_index: display_index for display_index, source_index in enumerate(ordered_indexes, start=1)}

    def replace(match: re.Match[str]) -> str:
        source_index = int(match.group(1))
        return f"[{remap[source_index]}]" if source_index in remap else match.group(0)

    ordered_sources = [sources[index - 1] for index in ordered_indexes]
    ordered_sources.extend(
        source for index, source in enumerate(sources, start=1)
        if index not in remap
    )
    return _CITATION_PATTERN.sub(replace, answer), ordered_sources


def claim_entity_groups(*, request: ChatRequest, plan: SearchPlan | None) -> list[list[str]]:
    return evidence_entity_groups(evidence_question(request=request, plan=plan))


def claim_evidence_queries(plan: SearchPlan | None) -> list[str]:
    return [query.query for query in plan.queries[:4] if query.query.strip()] if plan else []


def claim_source_has_direct_body(
    *, question: str, source: Source, entity_groups: list[list[str]] | None, aliases: list[str] | None,
) -> bool:
    body = source.evidence or source.snippet or ""
    if not body.strip():
        return False
    lowered = body.casefold()
    if entity_groups:
        return any(any(token_in_text(name.casefold(), lowered) for name in group) for group in entity_groups)
    if any(token_in_text(alias.casefold(), lowered) for alias in aliases or []):
        return True
    return has_question_specific_sources(
        question=question, sources=[source.model_copy(update={"title": ""})], include_url=False,
    )


def claim_eligible_source_indexes(
    *, question: str, sources: list[Source], entity_groups: list[list[str]] | None, aliases: list[str] | None = None,
) -> set[int]:
    """Keep direct evidence, plus one anchored side of a multi-entity chain."""
    eligible = {
        index for index, source in enumerate(sources, start=1)
        if claim_source_has_direct_body(
            question=question, source=source, entity_groups=entity_groups, aliases=aliases,
        )
    }
    if not entity_groups or len(entity_groups) < 2:
        for index, source in enumerate(sources, start=1):
            source_text = (source.evidence or source.snippet or "").casefold()
            if any(token_in_text(alias.casefold(), source_text) for alias in aliases or []):
                eligible.add(index)
        return eligible
    for index, source in enumerate(sources, start=1):
        source_text = (source.evidence or source.snippet or "").casefold()
        if any(any(token_in_text(name.casefold(), source_text) for name in group) for group in entity_groups):
            eligible.add(index)
    return eligible


def _claims(*, request: ChatRequest, sources: list[Source], plan: SearchPlan | None) -> list[CitationClaim]:
    question = evidence_question(request=request, plan=plan)
    groups = claim_entity_groups(request=request, plan=plan)
    return build_citation_claims(
        question=question,
        sources=sources,
        eligible_source_indexes=claim_eligible_source_indexes(
            question=question, sources=sources, entity_groups=groups, aliases=plan.aliases if plan else None,
        ),
        entity_groups=groups,
        aliases=plan.aliases if plan else None,
        evidence_queries=claim_evidence_queries(plan),
    )


def citation_claim_context(
    *, question: str, sources: list[Source], entity_groups: list[list[str]] | None = None,
    aliases: list[str] | None = None, evidence_queries: list[str] | None = None,
) -> str:
    claims = build_citation_claims(
        question=question,
        sources=sources,
        eligible_source_indexes=claim_eligible_source_indexes(
            question=question, sources=sources, entity_groups=entity_groups, aliases=aliases,
        ),
        entity_groups=entity_groups,
        aliases=aliases,
        evidence_queries=evidence_queries,
    )
    return "\n".join(
        f'<claim id="{claim.claim_id}" source_indexes="[{claim.source_index}]">{claim.statement}</claim>'
        for claim in claims
    )


def render_claim_bound_answer(
    *, answer: str, request: ChatRequest, sources: list[Source], plan: SearchPlan | None,
) -> str:
    claims = _claims(request=request, sources=sources, plan=plan)
    claim_sources = {claim.claim_id: claim.source_index for claim in claims}

    def render(match: re.Match[str]) -> str:
        source_index = int(match.group(1))
        claim_id = match.group(2)
        return f"[{source_index}]" if claim_sources.get(claim_id) == source_index else ""

    return re.sub(r"\[(\d+)\]\{(C\d+_\d+)\}", render, answer).strip()


def claim_ledger_fallback(claims: list[CitationClaim]) -> str:
    return "\n".join(["已核实的资料：", *[f"- {claim.statement}[{claim.source_index}]" for claim in claims[:4]]])


def render_structured_answer(
    *, answer: str, request: ChatRequest, sources: list[Source], plan: SearchPlan | None,
    conservative_answer: Callable[..., str],
) -> str:
    """Render source citations from model-selected Claim IDs, never raw indexes."""
    try:
        data = first_json_object(answer)
        blocks = data.get("blocks") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise ValueError("missing blocks")
    except (json.JSONDecodeError, ValueError, TypeError):
        claims = _claims(request=request, sources=sources, plan=plan)
        logger.info("llm.answer_render", format="fallback", claim_count=len(claims))
        return claim_ledger_fallback(claims) if claims else render_claim_bound_answer(
            answer=answer, request=request, sources=sources, plan=plan,
        )

    claims = _claims(request=request, sources=sources, plan=plan)
    claim_sources = {claim.claim_id: claim.source_index for claim in claims}
    relation_groups = claim_entity_groups(request=request, plan=plan)
    rendered: list[str] = []
    requirements = plan.answer_requirements if plan else []
    covered_requirements: set[int] = set()
    unbound_blocks = 0
    for block in blocks[:8]:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        claim_ids = block.get("claim_ids")
        if not text or not isinstance(claim_ids, list):
            continue
        requirement_indexes = block.get("requirement_indexes")
        if requirements:
            if not isinstance(requirement_indexes, list):
                unbound_blocks += 1
                continue
            valid_requirements = [
                index for index in requirement_indexes
                if isinstance(index, int) and 0 <= index < len(requirements)
            ]
            if not valid_requirements:
                unbound_blocks += 1
                continue
        else:
            valid_requirements = []
        indexes = [claim_sources[claim_id] for claim_id in claim_ids if claim_id in claim_sources]
        if not indexes or not claim_ids_cover_entity_groups(
            claims=claims,
            claim_ids=[claim_id for claim_id in claim_ids if isinstance(claim_id, str)],
            entity_groups=relation_groups,
        ):
            unbound_blocks += 1
            continue
        rendered.append(f"{text}{''.join(f'[{index}]' for index in dict.fromkeys(indexes))}")
        covered_requirements.update(valid_requirements)
    if requirements and set(range(len(requirements))) - covered_requirements:
        rendered = []
    if rendered:
        logger.info(
            "llm.answer_render", format="structured", block_count=len(blocks),
            bound_block_count=len(rendered), unbound_block_count=unbound_blocks, claim_count=len(claims),
        )
        return "\n\n".join(rendered)
    logger.info(
        "llm.answer_render", format="structured", block_count=len(blocks),
        bound_block_count=0, unbound_block_count=unbound_blocks, claim_count=len(claims),
    )
    return claim_ledger_fallback(claims) if claims else conservative_answer(
        request=request, sources=sources, plan=plan,
    )
