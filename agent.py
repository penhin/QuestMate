import asyncio
from collections.abc import AsyncIterator
from typing import Literal, TypedDict
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from langgraph.graph import END, StateGraph
import structlog

from knowledge import KnowledgeStore, knowledge_store
from config import get_settings
from llm import GuideLLM
from quality_policy import (
    EVIDENCE_POOL_WEIGHTS,
    HIGH_TRUST_THRESHOLD,
    VERSION_SENSITIVE_INTENTS,
    source_domain_limit,
)
from query_tokens import question_relevance_tokens
from schemas import ChatRequest, ChatResponse, GameResolution, SearchPlan, SessionMessage, Source
from search import SearchProvider, TavilySearchProvider
from storage import conversation_store


AgentStreamEvent = tuple[Literal["status", "chunk", "done"], str | ChatResponse]
logger = structlog.get_logger()


class QuestAgentState(TypedDict):
    request: ChatRequest
    history: list[SessionMessage]
    game_resolution: GameResolution
    search_plan: SearchPlan
    sources: list[Source]
    answer: str


class QuestAgent:
    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        llm: GuideLLM | None = None,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        self.search_provider = search_provider or TavilySearchProvider()
        self.llm = llm or GuideLLM()
        self.knowledge = knowledge or knowledge_store
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QuestAgentState)
        graph.add_node("resolve_game", self._resolve_game)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("resolve_game")
        graph.add_edge("resolve_game", "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def run(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid4()
        request = request.model_copy(update={"session_id": session_id})
        is_new_session = not await conversation_store.session_exists(session_id)
        history = await conversation_store.get_recent_messages(session_id, limit=8)
        game_resolution = self._confirmed_resolution_from_request(request) or await self.search_provider.resolve_game(
            request.game,
            question=request.question,
        )
        if self._needs_game_confirmation(game_resolution):
            return ChatResponse(
                session_id=session_id,
                answer="我还不能确定你要查的是哪一款游戏。请选择一个候选游戏，或选择“都不是”。",
                sources=[],
                is_new=is_new_session,
                needs_game_confirmation=True,
                game_candidates=game_resolution.candidates,
            )
        state = await self.graph.ainvoke(
            {
                "request": request,
                "history": history,
                "game_resolution": game_resolution,
                "search_plan": SearchPlan(),
                "sources": [],
                "answer": "",
            }
        )
        answer = await self.llm.improve_answer(
            request=request,
            sources=state["sources"],
            answer=state["answer"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            history=history,
        )
        title = await self.llm.summarize_title(request=request, answer=answer) if is_new_session else None
        response = ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=state["sources"],
            title=title,
            is_new=is_new_session,
        )
        await conversation_store.save_chat(request, response)
        return response

    async def stream(self, request: ChatRequest) -> AsyncIterator[AgentStreamEvent]:
        session_id = request.session_id or uuid4()
        request = request.model_copy(update={"session_id": session_id})
        is_new_session = not await conversation_store.session_exists(session_id)
        history = await conversation_store.get_recent_messages(session_id, limit=8)

        yield ("status", "确认游戏：查平台页和资料库")
        game_resolution = self._confirmed_resolution_from_request(request) or await self.search_provider.resolve_game(
            request.game,
            question=request.question,
        )

        yield ("status", self._status_for_game_resolution(game_resolution))
        if self._needs_game_confirmation(game_resolution):
            response = ChatResponse(
                session_id=session_id,
                answer="我还不能确定你要查的是哪一款游戏。请选择一个候选游戏，或选择“都不是”。",
                sources=[],
                title=None,
                is_new=is_new_session,
                needs_game_confirmation=True,
                game_candidates=game_resolution.candidates,
            )
            yield ("done", response)
            return
        yield ("status", self._status_for_plan_start(request.question))
        search_plan = await self.llm.plan_search(request=request, history=history, game_resolution=game_resolution)

        yield ("status", self._status_for_search(search_plan))
        sources, search_plan, refined = await self._retrieve_with_refinement(
            request=request,
            history=history,
            plan=search_plan,
            game_resolution=game_resolution,
        )

        if refined:
            yield ("status", "证据不足：已换一种表述补充检索")
        yield ("status", self._status_for_sources(sources))
        yield ("status", "整理答案：保留来源，核对版本")
        chunks: list[str] = []
        async for chunk in self.llm.stream_answer(
            request=request,
            sources=sources,
            plan=search_plan,
            game_resolution=game_resolution,
            history=history,
        ):
            chunks.append(chunk)
            yield ("chunk", chunk)

        answer = "".join(chunks)
        improved_answer = await self.llm.improve_answer(
            request=request,
            sources=sources,
            answer=answer,
            plan=search_plan,
            game_resolution=game_resolution,
            history=history,
        )
        if improved_answer != answer:
            answer = improved_answer
        title = await self.llm.summarize_title(request=request, answer=answer) if is_new_session else None
        response = ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=sources,
            title=title,
            is_new=is_new_session,
        )
        await conversation_store.save_chat(request, response)
        yield ("done", response)

    async def _plan(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        search_plan = await self.llm.plan_search(
            request=request,
            history=state["history"],
            game_resolution=state["game_resolution"],
        )
        return {**state, "search_plan": search_plan}

    async def _resolve_game(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        existing = state.get("game_resolution")
        if existing is not None:
            return state
        game_resolution = await self.search_provider.resolve_game(request.game, question=request.question)
        return {**state, "game_resolution": game_resolution}

    async def _search(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        sources, search_plan, _refined = await self._retrieve_with_refinement(
            request=request,
            history=state["history"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
        )
        return {**state, "sources": sources, "search_plan": search_plan}

    async def _retrieve_with_refinement(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        plan: SearchPlan,
        game_resolution: GameResolution,
    ) -> tuple[list[Source], SearchPlan, bool]:
        sources = await self._retrieve_sources(
            request.question,
            request.game,
            plan=plan,
            game_resolution=game_resolution,
        )
        refined_plan = await self.llm.refine_search_plan(
            request=request,
            plan=plan,
            sources=sources,
            history=history,
            game_resolution=game_resolution,
        )
        if refined_plan is None:
            return sources, plan, False

        refined_sources = await self._retrieve_sources(
            request.question,
            request.game,
            plan=refined_plan,
            game_resolution=game_resolution,
            include_knowledge=False,
        )
        merged_plan = self._merge_search_plans(plan, refined_plan)
        merged_sources = self._rank_sources(
            sources=[*sources, *refined_sources],
            query=f"{request.question} {' '.join(merged_plan.aliases)}".strip(),
            intent=merged_plan.intent,
        )
        logger.info(
            "retrieval.refined",
            game=request.game,
            initial_source_count=len(sources),
            refined_source_count=len(refined_sources),
            merged_source_count=len(merged_sources),
        )
        return merged_sources, merged_plan, True

    async def _retrieve_sources(
        self,
        question: str,
        game: str,
        *,
        plan: SearchPlan,
        game_resolution: GameResolution,
        include_knowledge: bool = True,
    ) -> list[Source]:
        """Rank local and live evidence in one pool, retaining URL diversity."""
        retrieval_calls = []
        dimensions = []
        if include_knowledge:
            retrieval_calls.append(self.knowledge.retrieve(game=game, query=question))
            dimensions.append("knowledge")
        retrieval_calls.append(
            self.search_provider.search(
                question,
                game,
                plan=plan,
                game_resolution=game_resolution,
            )
        )
        dimensions.append("web")
        retrievals = await asyncio.gather(*retrieval_calls, return_exceptions=True)
        source_groups: list[list[Source]] = []
        for dimension, result in zip(dimensions, retrievals, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "retrieval.dimension_failed",
                    dimension=dimension,
                    game=game,
                    error_type=type(result).__name__,
                )
                source_groups.append([])
            else:
                source_groups.append(result)
        query = f"{question} {' '.join(plan.aliases)}".strip()
        return self._rank_sources(
            sources=[source for group in source_groups for source in group],
            query=query,
            intent=plan.intent,
        )

    @staticmethod
    def _merge_search_plans(initial: SearchPlan, refined: SearchPlan) -> SearchPlan:
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
            queries=queries[:4],
            missing_info=list(dict.fromkeys([*initial.missing_info, *refined.missing_info]))[:4],
        )

    @staticmethod
    def _rank_sources(*, sources: list[Source], query: str, intent: str) -> list[Source]:
        ranked_by_url: dict[str, tuple[float, Source]] = {}
        for source in sources:
            key = QuestAgent._canonical_source_url(str(source.url))
            rank = QuestAgent._source_rank(source=source, query=query, intent=intent)
            current = ranked_by_url.get(key)
            if current is None or rank > current[0]:
                ranked_by_url[key] = (rank, source)

        ranked = sorted(ranked_by_url.values(), key=lambda item: item[0], reverse=True)
        selected: list[Source] = []
        domain_counts: dict[str, int] = {}
        for _rank, source in ranked:
            domain = urlparse(str(source.url)).netloc.lower()
            domain_limit = source_domain_limit(domain)
            if domain_counts.get(domain, 0) >= domain_limit:
                continue
            selected.append(source)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected) >= get_settings().search_max_results:
                break
        return selected

    @staticmethod
    def _canonical_source_url(url: str) -> str:
        parsed = urlparse(url)
        return urlunparse(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", "")
        )

    @staticmethod
    def _source_rank(*, source: Source, query: str, intent: str) -> float:
        text = f"{source.title} {source.evidence or source.snippet or ''}".lower()
        tokens = question_relevance_tokens(query)
        coverage = sum(1 for token in tokens if token in text) / max(len(tokens), 1)
        retrieval_score = min(max(source.score or 0.5, 0), 1)
        version_score = 1.0 if source.game_version or source.published_at else 0.0
        if intent not in VERSION_SENSITIVE_INTENTS:
            version_score = 0.5
        return (
            coverage * EVIDENCE_POOL_WEIGHTS.relevance
            + retrieval_score * EVIDENCE_POOL_WEIGHTS.retrieval
            + source.trust_score * EVIDENCE_POOL_WEIGHTS.trust
            + version_score * EVIDENCE_POOL_WEIGHTS.version
        )

    async def _answer(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        answer = await self.llm.answer(
            request=request,
            sources=state["sources"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            history=state["history"],
        )
        return {**state, "answer": answer}

    @staticmethod
    def _status_for_plan_start(question: str) -> str:
        if any(token in question for token in ("版本", "补丁", "更新", "削弱", "增强")):
            return "理解问题：先查版本变化"
        if any(token in question for token in ("怎么打", "打不过", "弱点", "Boss", "boss")):
            return "理解问题：查弱点和打法"
        if any(token in question for token in ("有什么用", "有啥用", "作用", "用途", "用来", "怎么用")):
            return "理解问题：查物品用途"
        if any(token in question for token in ("在哪", "哪里", "获得", "获取", "钥匙", "位置")):
            return "理解问题：查位置和获取方式"
        if any(token in question for token in ("任务", "支线", "下一步", "NPC", "npc")):
            return "理解问题：查任务步骤"
        if any(token in question for token in ("模式", "开启", "打开", "解锁", "触发", "机制")):
            return "理解问题：查开启条件"
        return "理解问题：规划查询"

    @staticmethod
    def _status_for_search(search_plan: SearchPlan) -> str:
        labels = {
            "boss_strategy": "类型：Boss 打法；查弱点/阶段/社区打法",
            "item_location": "类型：物品位置；查地点/条件/路线",
            "item_usage": "类型：物品用途；查效果/用法/交互对象",
            "quest_step": "类型：任务步骤；查 NPC/触发/顺序",
            "game_mechanic": "类型：游戏机制；查开启条件/触发方式",
            "build": "类型：配装；查数值/装备/版本",
            "patch": "类型：版本变化；优先官方补丁",
            "lore": "类型：剧情背景；查事实和解释",
        }
        return labels.get(search_plan.intent, "类型：通用问题；筛选相关来源")

    @staticmethod
    def _status_for_game_resolution(game_resolution: GameResolution) -> str:
        if game_resolution.is_confirmed:
            entries = len(game_resolution.platform_urls) + len(game_resolution.database_domains)
            if entries:
                return f"确认游戏：找到 {entries} 个入口"
            return "确认游戏：名称已匹配，继续检索"
        return "确认游戏：资料入口不足，谨慎回答"

    @staticmethod
    def _needs_game_confirmation(game_resolution: GameResolution) -> bool:
        return (
            game_resolution.ambiguous
            or (not game_resolution.is_confirmed and bool(game_resolution.candidates))
            or len(game_resolution.candidates) > 1
        )

    @staticmethod
    def _confirmed_resolution_from_request(request: ChatRequest) -> GameResolution | None:
        if request.metadata.get("confirmed_game") is not True:
            return None
        aliases = request.metadata.get("game_aliases")
        database_domains = request.metadata.get("database_domains")
        return GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=aliases if isinstance(aliases, list) else [],
            database_domains=database_domains if isinstance(database_domains, list) else [],
            confidence=1,
            ambiguous=False,
        )

    @staticmethod
    def _status_for_sources(sources: list[Source]) -> str:
        if not sources:
            return "来源筛选：未找到强相关资料"
        trusted_count = sum(1 for source in sources if source.trust_score >= HIGH_TRUST_THRESHOLD)
        if trusted_count:
            return f"来源筛选：保留 {len(sources)} 个，{trusted_count} 个高可信"
        return f"来源筛选：保留 {len(sources)} 个，交叉核对"


quest_agent = QuestAgent()
