"""Coordinate local, live, and iterative evidence retrieval."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import structlog

from quality_policy import MAX_INVESTIGATION_HOPS
from retrieval.evidence_pool import canonical_source_url, rank_sources
from schemas import ChatRequest, EvidenceGap, GameResolution, InvestigationState, SearchPlan, SessionMessage, Source

logger = structlog.get_logger()


class KnowledgeRetriever(Protocol):
    async def retrieve(self, *, game: str, query: str) -> list[Source]: ...


class LiveRetriever(Protocol):
    async def search(self, query: str, game: str, **kwargs: Any) -> list[Source]: ...


class PlanRefiner(Protocol):
    async def refine_search_plan(self, **kwargs: Any) -> SearchPlan | None: ...


@dataclass(frozen=True)
class RetrievalOutcome:
    sources: list[Source]
    plan: SearchPlan
    investigation: InvestigationState
    refined: bool


class RetrievalCoordinator:
    def __init__(
        self,
        *,
        knowledge: KnowledgeRetriever,
        search_provider: LiveRetriever,
        llm: PlanRefiner,
        max_results: int,
        max_hops: int = MAX_INVESTIGATION_HOPS,
    ) -> None:
        self.knowledge = knowledge
        self.search_provider = search_provider
        self.llm = llm
        self.max_results = max_results
        self.max_hops = max_hops

    async def retrieve_with_refinement(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        plan: SearchPlan,
        game_resolution: GameResolution,
    ) -> tuple[list[Source], SearchPlan, bool]:
        outcome = await self.investigate(
            request=request,
            history=history,
            plan=plan,
            game_resolution=game_resolution,
        )
        return outcome.sources, outcome.plan, outcome.refined

    async def investigate(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        plan: SearchPlan,
        game_resolution: GameResolution,
    ) -> RetrievalOutcome:
        merged_sources = await self.retrieve_sources(
            request.question, request.game, plan=plan, game_resolution=game_resolution
        )
        merged_plan = plan
        refined = False
        active_game_resolution = self._with_discovered_wiki_domains(game_resolution, merged_sources)
        investigation = InvestigationState(
            goal=request.question,
            evidence_gaps=[
                EvidenceGap(kind="other", description=value, query_hint=value)
                for value in plan.missing_info
            ],
            unresolved_questions=plan.missing_info,
            attempted_queries=[query.query for query in plan.queries],
            aliases=plan.aliases,
        )

        update_investigation = getattr(self.llm, "update_investigation", None)
        if callable(update_investigation):
            for hop in range(0, self.max_hops + 1):
                investigation = await update_investigation(
                    request=request,
                    plan=merged_plan,
                    sources=merged_sources,
                    investigation=investigation,
                    history=history,
                    game_resolution=active_game_resolution,
                )
                investigation = InvestigationState.model_validate(investigation.model_dump())
                if investigation.complete or not investigation.next_queries:
                    break
                if hop >= self.max_hops:
                    investigation = investigation.model_copy(
                        update={"next_queries": [], "stop_reason": "budget_exhausted"}
                    )
                    break

                refined_plan = SearchPlan(
                    intent=merged_plan.intent,
                    version_sensitive=merged_plan.version_sensitive,
                    named_entity_groups=merged_plan.named_entity_groups,
                    aliases=investigation.aliases,
                    queries=investigation.next_queries,
                    missing_info=investigation.unresolved_questions,
                    refinement=True,
                )
                refined_sources = await self.retrieve_sources(
                    request.question,
                    request.game,
                    plan=refined_plan,
                    game_resolution=active_game_resolution,
                    include_knowledge=False,
                )
                if not self._has_novel_evidence(existing=merged_sources, candidates=refined_sources):
                    investigation = investigation.model_copy(
                        update={"next_queries": [], "stop_reason": "insufficient_evidence"}
                    )
                    logger.info(
                        "retrieval.investigation_stopped",
                        game=request.game,
                        reason="no_novel_evidence",
                        hop=investigation.hop_count + 1,
                    )
                    break
                merged_plan = merge_search_plans(merged_plan, refined_plan)
                merged_sources = rank_sources(
                    sources=[*merged_sources, *refined_sources],
                    query=f"{request.question} {' '.join(merged_plan.aliases)}".strip(),
                    intent=merged_plan.intent,
                    max_results=self.max_results,
                    version_sensitive=merged_plan.version_sensitive,
                )
                attempted_queries = list(dict.fromkeys([
                    *investigation.attempted_queries,
                    *(query.query for query in investigation.next_queries),
                ]))[:16]
                investigation = investigation.model_copy(
                    update={
                        "attempted_queries": attempted_queries,
                        "next_queries": [],
                        "hop_count": investigation.hop_count + 1,
                    }
                )
                active_game_resolution = self._with_discovered_wiki_domains(
                    active_game_resolution, merged_sources
                )
                refined = True
                logger.info(
                    "retrieval.investigation_hop",
                    game=request.game,
                    hop=investigation.hop_count,
                    new_source_count=len(refined_sources),
                    merged_source_count=len(merged_sources),
                    unresolved_count=len(investigation.unresolved_questions),
                )
            merged_plan = merged_plan.model_copy(
                update={
                    "aliases": list(dict.fromkeys([*merged_plan.aliases, *investigation.aliases]))[:6],
                    "missing_info": list(dict.fromkeys([
                        *investigation.unresolved_questions,
                        *(gap.description for gap in investigation.evidence_gaps),
                    ]))[:4],
                }
            )
            return RetrievalOutcome(merged_sources, merged_plan, investigation, refined)

        # Compatibility path for lightweight/custom LLM implementations.
        for hop in range(1, self.max_hops + 1):
            refined_plan = await self.llm.refine_search_plan(
                request=request,
                plan=merged_plan,
                sources=merged_sources,
                history=history,
                game_resolution=active_game_resolution,
            )
            if refined_plan is None:
                break
            refined_sources = await self.retrieve_sources(
                request.question,
                request.game,
                plan=refined_plan,
                game_resolution=active_game_resolution,
                include_knowledge=False,
            )
            if not self._has_novel_evidence(existing=merged_sources, candidates=refined_sources):
                logger.info(
                    "retrieval.investigation_stopped",
                    game=request.game,
                    reason="no_novel_evidence",
                    hop=hop,
                )
                break
            merged_plan = merge_search_plans(merged_plan, refined_plan)
            merged_sources = rank_sources(
                sources=[*merged_sources, *refined_sources],
                query=f"{request.question} {' '.join(merged_plan.aliases)}".strip(),
                intent=merged_plan.intent,
                max_results=self.max_results,
                version_sensitive=merged_plan.version_sensitive,
            )
            refined = True
            active_game_resolution = self._with_discovered_wiki_domains(
                active_game_resolution, merged_sources
            )
            logger.info(
                "retrieval.investigation_hop",
                game=request.game,
                hop=hop,
                new_source_count=len(refined_sources),
                merged_source_count=len(merged_sources),
            )
        investigation = investigation.model_copy(
            update={
                "attempted_queries": [query.query for query in merged_plan.queries][:16],
                "aliases": merged_plan.aliases,
                "unresolved_questions": merged_plan.missing_info,
                "hop_count": self.max_hops if refined else 0,
                "stop_reason": "insufficient_evidence" if merged_plan.missing_info else None,
            }
        )
        return RetrievalOutcome(merged_sources, merged_plan, investigation, refined)

    @staticmethod
    def _with_discovered_wiki_domains(
        resolution: GameResolution,
        sources: list[Source],
    ) -> GameResolution:
        domains = list(resolution.database_domains)
        for source in sources:
            if source.source_type != "wiki":
                continue
            domain = urlparse(str(source.url)).netloc.casefold().removeprefix("www.")
            if domain and domain not in domains:
                domains.append(domain)
        return resolution.model_copy(update={"database_domains": domains[:8]})

    @staticmethod
    def _has_novel_evidence(*, existing: list[Source], candidates: list[Source]) -> bool:
        existing_by_url: dict[str, str] = {}
        for source in existing:
            key = canonical_source_url(str(source.url))
            evidence = " ".join((source.evidence or source.snippet or "").casefold().split())
            existing_by_url[key] = f"{existing_by_url.get(key, '')} {evidence}".strip()
        for source in candidates:
            key = canonical_source_url(str(source.url))
            evidence = " ".join((source.evidence or source.snippet or "").casefold().split())
            if key not in existing_by_url:
                return True
            known = existing_by_url[key]
            if evidence and evidence not in known:
                return True
        return False

    async def retrieve_sources(
        self,
        question: str,
        game: str,
        *,
        plan: SearchPlan,
        game_resolution: GameResolution,
        include_knowledge: bool = True,
    ) -> list[Source]:
        calls = []
        dimensions = []
        if include_knowledge:
            calls.append(self.knowledge.retrieve(game=game, query=question))
            dimensions.append("knowledge")
        calls.append(self.search_provider.search(question, game, plan=plan, game_resolution=game_resolution))
        dimensions.append("web")
        results = await asyncio.gather(*calls, return_exceptions=True)
        groups: list[list[Source]] = []
        for dimension, result in zip(dimensions, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "retrieval.dimension_failed", dimension=dimension, game=game, error_type=type(result).__name__
                )
                groups.append([])
            else:
                groups.append(result)
        selected = rank_sources(
            sources=[source for group in groups for source in group],
            query=f"{question} {' '.join(plan.aliases)}".strip(),
            intent=plan.intent,
            max_results=self.max_results,
            version_sensitive=plan.version_sensitive,
        )
        logger.info(
            "retrieval.completed",
            game=game,
            intent=plan.intent,
            dimensions={dimension: len(group) for dimension, group in zip(dimensions, groups, strict=True)},
            selected_count=len(selected),
            selected_source_types=[source.source_type for source in selected],
        )
        return selected


def merge_search_plans(initial: SearchPlan, refined: SearchPlan) -> SearchPlan:
    aliases = list(dict.fromkeys([*initial.aliases, *refined.aliases]))[:6]
    named_entity_groups: list[list[str]] = []
    for group in [*initial.named_entity_groups, *refined.named_entity_groups]:
        cleaned = list(dict.fromkeys(name.strip() for name in group if name.strip()))
        normalized = {" ".join(name.casefold().split()) for name in cleaned}
        if not normalized:
            continue
        overlapping = [
            index
            for index, existing in enumerate(named_entity_groups)
            if normalized.intersection(
                " ".join(name.casefold().split()) for name in existing
            )
        ]
        if not overlapping:
            named_entity_groups.append(cleaned[:4])
            continue
        target = overlapping[0]
        named_entity_groups[target] = list(dict.fromkeys([
            *named_entity_groups[target],
            *cleaned,
        ]))[:4]
        # Alias overlap is transitive. Collapse any additional connected
        # groups instead of turning one entity into multiple AND requirements.
        for index in reversed(overlapping[1:]):
            named_entity_groups[target] = list(dict.fromkeys([
                *named_entity_groups[target],
                *named_entity_groups[index],
            ]))[:4]
            named_entity_groups.pop(index)
    queries = []
    seen_queries: set[str] = set()
    for query in [*initial.queries, *refined.queries]:
        normalized = " ".join(query.query.casefold().split())
        if normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        queries.append(query)
    return SearchPlan(
        intent=initial.intent,
        version_sensitive=initial.version_sensitive or refined.version_sensitive,
        named_entity_groups=named_entity_groups[:4],
        aliases=aliases,
        queries=queries[:6],
        missing_info=list(dict.fromkeys([*initial.missing_info, *refined.missing_info]))[:4],
    )
