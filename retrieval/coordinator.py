"""Coordinate local, live, and iterative evidence retrieval."""

import asyncio
from typing import Any, Protocol

import structlog

from quality_policy import MAX_INVESTIGATION_HOPS
from retrieval.evidence_pool import rank_sources
from schemas import ChatRequest, GameResolution, SearchPlan, SessionMessage, Source

logger = structlog.get_logger()


class KnowledgeRetriever(Protocol):
    async def retrieve(self, *, game: str, query: str) -> list[Source]: ...


class LiveRetriever(Protocol):
    async def search(self, query: str, game: str, **kwargs: Any) -> list[Source]: ...


class PlanRefiner(Protocol):
    async def refine_search_plan(self, **kwargs: Any) -> SearchPlan | None: ...


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
        merged_sources = await self.retrieve_sources(
            request.question, request.game, plan=plan, game_resolution=game_resolution
        )
        merged_plan = plan
        refined = False
        for hop in range(1, self.max_hops + 1):
            refined_plan = await self.llm.refine_search_plan(
                request=request,
                plan=merged_plan,
                sources=merged_sources,
                history=history,
                game_resolution=game_resolution,
            )
            if refined_plan is None:
                break
            refined_sources = await self.retrieve_sources(
                request.question,
                request.game,
                plan=refined_plan,
                game_resolution=game_resolution,
                include_knowledge=False,
            )
            merged_plan = merge_search_plans(merged_plan, refined_plan)
            merged_sources = rank_sources(
                sources=[*merged_sources, *refined_sources],
                query=f"{request.question} {' '.join(merged_plan.aliases)}".strip(),
                intent=merged_plan.intent,
                max_results=self.max_results,
            )
            refined = True
            logger.info(
                "retrieval.investigation_hop",
                game=request.game,
                hop=hop,
                new_source_count=len(refined_sources),
                merged_source_count=len(merged_sources),
            )
        return merged_sources, merged_plan, refined

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
        aliases=aliases,
        queries=queries[:6],
        missing_info=list(dict.fromkeys([*initial.missing_info, *refined.missing_info]))[:4],
    )
