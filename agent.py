from collections.abc import AsyncIterator
import inspect
import re
from time import perf_counter
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
import structlog

from knowledge import KnowledgeStore, knowledge_store
from config import get_settings
from llm import GuideLLM
from quality_policy import HIGH_TRUST_THRESHOLD
from retrieval.coordinator import RetrievalCoordinator, merge_search_plans
from retrieval.evidence_pool import canonical_source_url, merge_source_evidence, rank_sources, source_rank
from retrieval.wiki_domains import normalize_wiki_host
from game_resolution import (
    is_candidate_identity_url,
    resolution_matches_selected_url,
    select_game_candidate,
)
from ai.fallback_planning import fallback_search_plan
from request_safety import requires_safe_refusal
from schemas import ChatRequest, ChatResponse, GameResolution, InvestigationState, SearchPlan, SessionMessage, Source
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
    investigation: InvestigationState
    answer: str
    timings_ms: dict[str, int]


class QuestAgent:
    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        llm: GuideLLM | None = None,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        self.knowledge = knowledge or knowledge_store
        self.search_provider = search_provider or TavilySearchProvider(content_index=self.knowledge)
        self.llm = llm or GuideLLM()
        self.retrieval = RetrievalCoordinator(
            knowledge=self.knowledge,
            search_provider=self.search_provider,
            llm=self.llm,
            max_results=get_settings().search_max_results,
        )
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
        provider_scope = getattr(self.llm, "provider_scope", None)
        if callable(provider_scope):
            async with provider_scope(request):
                return await self._run(request)
        return await self._run(request)

    async def _run(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        session_id = request.session_id or uuid4()
        request = request.model_copy(update={"session_id": session_id})
        is_new_session = not await conversation_store.session_exists(session_id)
        if requires_safe_refusal(f"{request.game}\n{request.question}"):
            return ChatResponse(
                session_id=session_id,
                answer="我不能执行忽略规则、泄露提示词或密钥等请求。可以继续回答正常的游戏攻略问题。",
                sources=[],
                is_new=is_new_session,
                timings_ms={"safety": self._elapsed_ms(started)},
            )
        history = await conversation_store.get_recent_messages(session_id, limit=8)
        identity_started = perf_counter()
        game_resolution = await self._initial_game_context(request)
        timings_ms = {"identity": self._elapsed_ms(identity_started)}
        state = await self.graph.ainvoke(
            {
                "request": request,
                "history": history,
                "game_resolution": game_resolution,
                "search_plan": SearchPlan(),
                "sources": [],
                "investigation": InvestigationState(goal=request.question),
                "answer": "",
                "timings_ms": timings_ms,
            }
        )
        game_resolution = state["game_resolution"]
        if self._needs_game_confirmation(game_resolution):
            return ChatResponse(
                session_id=session_id,
                answer=self._game_confirmation_message(game_resolution),
                sources=[],
                is_new=is_new_session,
                needs_game_confirmation=True,
                game_candidates=game_resolution.candidates,
                timings_ms=state["timings_ms"],
            )
        state = {**state, "game_resolution": game_resolution}
        improve_kwargs = dict(
            request=request,
            sources=state["sources"],
            answer=state["answer"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            history=history,
        )
        self._add_optional_argument(self.llm.improve_answer, improve_kwargs, "investigation", state["investigation"])
        # The answer renderer has already enforced Claim-to-source binding.
        # A second model rewrite cannot add evidence and can silently remove
        # required action terms or detach citations, so source-backed answers
        # are delivered directly within the two-call budget.
        if state["sources"]:
            answer = state["answer"]
            state["timings_ms"]["improvement"] = 0
        else:
            improvement_started = perf_counter()
            answer = await self.llm.improve_answer(**improve_kwargs)
            state["timings_ms"]["improvement"] = self._elapsed_ms(improvement_started)
        title = self._initial_title(request) if is_new_session else None
        response = ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=state["sources"],
            title=title,
            is_new=is_new_session,
            timings_ms=state["timings_ms"],
        )
        await conversation_store.save_chat(request, response)
        return response

    async def stream(self, request: ChatRequest) -> AsyncIterator[AgentStreamEvent]:
        provider_scope = getattr(self.llm, "provider_scope", None)
        if callable(provider_scope):
            async with provider_scope(request):
                async for event in self._stream(request):
                    yield event
            return
        async for event in self._stream(request):
            yield event

    async def _stream(self, request: ChatRequest) -> AsyncIterator[AgentStreamEvent]:
        started = perf_counter()
        session_id = request.session_id or uuid4()
        request = request.model_copy(update={"session_id": session_id})
        is_new_session = not await conversation_store.session_exists(session_id)

        if requires_safe_refusal(f"{request.game}\n{request.question}"):
            response = ChatResponse(
                session_id=session_id,
                answer="我不能执行忽略规则、泄露提示词或密钥等请求。可以继续回答正常的游戏攻略问题。",
                sources=[],
                is_new=is_new_session,
                timings_ms={"safety": self._elapsed_ms(started)},
            )
            yield ("done", response)
            return
        history = await conversation_store.get_recent_messages(session_id, limit=8)

        identity_started = perf_counter()
        game_resolution = await self._initial_game_context(request)
        timings_ms = {"identity": self._elapsed_ms(identity_started)}
        yield ("status", self._status_for_plan_start(request.question))
        planning_started = perf_counter()
        search_plan = await self.llm.plan_search(request=request, history=history, game_resolution=game_resolution)
        timings_ms["planning"] = self._elapsed_ms(planning_started)

        yield ("status", self._status_for_search(search_plan))
        outcome, game_resolution = await self._retrieve_after_identity_check(
            request=request,
            history=history,
            plan=search_plan,
            game_resolution=game_resolution,
            timings_ms=timings_ms,
        )
        sources, search_plan, refined = outcome.sources, outcome.plan, outcome.refined

        game_resolution = await self._recover_game_identity_if_needed(
            request=request,
            sources=sources,
            current=game_resolution,
        )
        if self._needs_game_confirmation(game_resolution):
            response = ChatResponse(
                session_id=session_id,
                answer=self._game_confirmation_message(game_resolution),
                sources=[],
                title=None,
                is_new=is_new_session,
                needs_game_confirmation=True,
                game_candidates=game_resolution.candidates,
                timings_ms=timings_ms,
            )
            yield ("done", response)
            return

        if refined:
            yield ("status", "证据扩展：已针对关键证据缺口补查")
        yield ("status", self._status_for_sources(sources))
        yield ("status", "整理答案：保留来源，核对版本")
        chunks: list[str] = []
        answer_started = perf_counter()
        stream_kwargs = dict(
            request=request,
            sources=sources,
            plan=search_plan,
            game_resolution=game_resolution,
            history=history,
        )
        self._add_optional_argument(self.llm.stream_answer, stream_kwargs, "investigation", outcome.investigation)
        async for chunk in self.llm.stream_answer(**stream_kwargs):
            chunks.append(chunk)
            yield ("chunk", chunk)

        answer = "".join(chunks)
        timings_ms["answer"] = self._elapsed_ms(answer_started)
        improve_kwargs = dict(
            request=request,
            sources=sources,
            answer=answer,
            plan=search_plan,
            game_resolution=game_resolution,
            history=history,
        )
        self._add_optional_argument(self.llm.improve_answer, improve_kwargs, "investigation", outcome.investigation)
        improvement_started = perf_counter()
        improved_answer = await self.llm.improve_answer(**improve_kwargs)
        timings_ms["improvement"] = self._elapsed_ms(improvement_started)
        if improved_answer != answer:
            answer = improved_answer
        title = self._initial_title(request) if is_new_session else None
        response = ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=sources,
            title=title,
            is_new=is_new_session,
            timings_ms=timings_ms,
        )
        await conversation_store.save_chat(request, response)
        yield ("done", response)

    async def _plan(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        started = perf_counter()
        search_plan = await self.llm.plan_search(
            request=request,
            history=state["history"],
            game_resolution=state["game_resolution"],
        )
        return {**state, "search_plan": search_plan, "timings_ms": {
            **state["timings_ms"], "planning": self._elapsed_ms(started)
        }}

    async def _resolve_game(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        existing = state.get("game_resolution")
        if existing is not None:
            return state
        game_resolution = await self.search_provider.resolve_game(request.game, question=request.question)
        return {**state, "game_resolution": game_resolution}

    async def _search(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        outcome, game_resolution = await self._retrieve_after_identity_check(
            request=request,
            history=state["history"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            timings_ms=state["timings_ms"],
        )
        return {
            **state,
            "sources": outcome.sources,
            "search_plan": outcome.plan,
            "investigation": outcome.investigation,
            "game_resolution": game_resolution,
        }

    async def _retrieve_after_identity_check(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        plan: SearchPlan,
        game_resolution: GameResolution,
        timings_ms: dict[str, int] | None = None,
    ) -> tuple[object, GameResolution]:
        """Run a cheap retrieval wave before paying for full investigation.

        Direct guide evidence can continue the answer path. If the first wave
        has no evidence at all, recover identity before asking the model to
        answer. A cached profile is never used as a substitute for the current
        request's identity decision.
        """
        resolved = game_resolution
        # Game identity is a safety boundary, not a retrieval result.  Before
        # using an unconfirmed title, resolve it once so a plausible guide
        # page cannot silently select one of several games sharing that name.
        # This is an identity-state rule, not a title, game, or action-word
        # heuristic. Retrieval callers that already supplied a confirmed
        # identity avoid the extra lookup.
        if (
            request.metadata.get("confirmed_game") is not True
            and not request.metadata.get("selected_game_url")
        ):
            resolution_started = perf_counter()
            resolved = await self._resolve_request_game(request)
            if timings_ms is not None:
                timings_ms["identity_resolution"] = self._elapsed_ms(resolution_started)
            if self._needs_game_confirmation(resolved):
                outcome = await self.retrieval.investigate(
                    request=request,
                    history=history,
                    plan=plan,
                    game_resolution=resolved,
                    initial_sources=[],
                )
                return outcome, resolved
        retrieval_started = perf_counter()
        initial_sources = await self.retrieval.retrieve_sources(
            request.question,
            request.game,
            plan=plan,
            game_resolution=resolved,
        )
        if timings_ms is not None:
            timings_ms["retrieval_initial"] = self._elapsed_ms(retrieval_started)
        if not initial_sources:
            resolution_started = perf_counter()
            resolved = await self._resolve_request_game(request)
            if timings_ms is not None:
                timings_ms["identity_resolution"] = self._elapsed_ms(resolution_started)
            if self._needs_game_confirmation(resolved):
                # A concrete competing candidate is actionable ambiguity.  An
                # empty discovery response is only an upstream/retrieval miss;
                # preserve the supplied title and continue to the conservative
                # evidence path instead of forcing an unnecessary UI detour.
                if resolved.ambiguous or resolved.candidates:
                    outcome = await self.retrieval.investigate(
                        request=request,
                        history=history,
                        plan=plan,
                        game_resolution=resolved,
                        initial_sources=[],
                    )
                    return outcome, resolved
                resolved = game_resolution
        if (
            not initial_sources
            and resolved.is_confirmed
            and self._resolution_added_identity(game_resolution, resolved)
        ):
            # The first wave may have been built before aliases/official
            # identity were known, or its model-generated query may be too
            # narrow. Retry once with a local plan that preserves the full
            # question. This improves recall without spending another LLM
            # call or fabricating an answer without evidence.
            fallback_plan = fallback_search_plan(question=request.question)
            retry_started = perf_counter()
            initial_sources = await self.retrieval.retrieve_sources(
                request.question,
                request.game,
                plan=fallback_plan,
                game_resolution=resolved,
            )
            if timings_ms is not None:
                timings_ms["retrieval_retry"] = self._elapsed_ms(retry_started)
            if initial_sources:
                plan = merge_search_plans(plan, fallback_plan)
        investigation_started = perf_counter()
        outcome = await self.retrieval.investigate(
            request=request,
            history=history,
            plan=plan,
            game_resolution=resolved,
            initial_sources=initial_sources,
        )
        if timings_ms is not None:
            timings_ms["investigation"] = self._elapsed_ms(investigation_started)
        return outcome, resolved

    @staticmethod
    def _resolution_added_identity(before: GameResolution, after: GameResolution) -> bool:
        """Whether resolution added usable server-discovered identity evidence."""
        before_values = {
            str(value).casefold()
            for value in [
                *before.aliases,
                *before.platform_urls,
                *before.official_urls,
                *before.identity_urls,
                *before.database_domains,
            ]
        }
        after_values = {
            str(value).casefold()
            for value in [
                *after.aliases,
                *after.platform_urls,
                *after.official_urls,
                *after.identity_urls,
                *after.database_domains,
            ]
        }
        return bool(after_values - before_values)

    async def _retrieve_with_refinement(
        self,
        *,
        request: ChatRequest,
        history: list[SessionMessage],
        plan: SearchPlan,
        game_resolution: GameResolution,
    ) -> tuple[list[Source], SearchPlan, bool]:
        return await self.retrieval.retrieve_with_refinement(
            request=request, history=history, plan=plan, game_resolution=game_resolution
        )

    async def _retrieve_sources(
        self,
        question: str,
        game: str,
        *,
        plan: SearchPlan,
        game_resolution: GameResolution,
        include_knowledge: bool = True,
    ) -> list[Source]:
        return await self.retrieval.retrieve_sources(
            question,
            game,
            plan=plan,
            game_resolution=game_resolution,
            include_knowledge=include_knowledge,
        )

    @staticmethod
    def _merge_search_plans(initial: SearchPlan, refined: SearchPlan) -> SearchPlan:
        return merge_search_plans(initial, refined)

    @staticmethod
    def _rank_sources(*, sources: list[Source], query: str, intent: str) -> list[Source]:
        return rank_sources(sources=sources, query=query, intent=intent, max_results=get_settings().search_max_results)

    @staticmethod
    def _merge_source_evidence(*, preferred: Source, other: Source) -> Source:
        return merge_source_evidence(preferred=preferred, other=other)

    @staticmethod
    def _canonical_source_url(url: str) -> str:
        return canonical_source_url(url)

    @staticmethod
    def _source_rank(*, source: Source, query: str, intent: str) -> float:
        return source_rank(source=source, query=query, intent=intent)

    async def _answer(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        started = perf_counter()
        answer_kwargs = dict(
            request=request,
            sources=state["sources"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            history=state["history"],
        )
        self._add_optional_argument(self.llm.answer, answer_kwargs, "investigation", state["investigation"])
        answer = await self.llm.answer(**answer_kwargs)
        return {**state, "answer": answer, "timings_ms": {
            **state["timings_ms"], "answer": self._elapsed_ms(started)
        }}

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((perf_counter() - started) * 1000)

    @staticmethod
    def _add_optional_argument(callable_obj, kwargs: dict, name: str, value) -> None:
        if name in inspect.signature(callable_obj).parameters:
            kwargs[name] = value

    @staticmethod
    def _status_for_plan_start(question: str) -> str:
        # Intent-specific status follows after model/fallback planning.  Before
        # that point a keyword table would only create misleading certainty for
        # novel or compound questions.
        return "理解问题：识别目标和关键关系"

    @staticmethod
    def _initial_title(request: ChatRequest) -> str:
        """Keep title generation off the critical answer path."""
        return request.question.strip()[:40] or request.game.strip() or "未命名会话"

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
            entries = (
                len(game_resolution.platform_urls)
                + len(game_resolution.official_urls)
                + len(game_resolution.identity_urls)
                + len(game_resolution.database_domains)
            )
            if entries:
                return f"确认游戏：找到 {entries} 个入口"
            return "确认游戏：名称已匹配，继续检索"
        return "确认游戏：资料入口不足，谨慎回答"

    @staticmethod
    def _needs_game_confirmation(game_resolution: GameResolution) -> bool:
        # An unverified title is an identity problem even when search returned
        # no safe candidates. Continuing to gameplay retrieval in that state
        # silently converts a missing identity into a conservative gameplay
        # answer, which makes the UI and evaluation unable to ask the user for
        # the one detail that would resolve it.
        return game_resolution.ambiguous or not game_resolution.is_confirmed

    @staticmethod
    def _game_confirmation_message(game_resolution: GameResolution) -> str:
        if game_resolution.candidates:
            return "我还不能确定你要查的是哪一款游戏。请选择一个候选游戏，或选择“都不是”。"
        return (
            f"我还没有找到能可靠确认《{game_resolution.input_name}》的游戏入口。"
            "请提供 Steam/itch.io 链接、原文游戏名、开发商或商店页截图，我再继续查。"
        )

    async def _resolve_request_game(self, request: ChatRequest) -> GameResolution:
        confirmed = self._confirmed_resolution_from_request(request)
        if confirmed is None:
            return await self.search_provider.resolve_game(
                request.game,
                question=request.question,
            )
        settings = get_settings()
        if (
            settings.allow_evaluation_retrieval_hints
            and not settings.is_production
            and request.metadata.get("evaluation") is True
        ):
            return confirmed
        selected_url = request.metadata.get("selected_game_url")
        if (
            isinstance(selected_url, str)
            and len(selected_url) <= 500
            and is_candidate_identity_url(selected_url)
        ):
            selector = getattr(self.search_provider, "select_game_candidate", None)
            if callable(selector):
                selected = await selector(
                    game=request.game,
                    selected_url=selected_url,
                    question=request.question,
                )
                if (
                    selected.is_confirmed
                    and not selected.ambiguous
                    and resolution_matches_selected_url(
                        selected,
                        selected_url=selected_url,
                    )
                ):
                    return selected
                # A stale or mismatched opaque selection must not fall through
                # to name-only resolution and silently select another game.
                return selected.model_copy(
                    update={
                        "confidence": 0,
                        "ambiguous": bool(selected.candidates),
                    }
                )

        # A production confirmation is a user choice, not authority for client-
        # supplied aliases or hosts. Re-resolve the name and accept only an
        # unambiguous identity or a URL that occurs in the fresh candidate set.
        discovered = await self.search_provider.resolve_game(
            request.game,
            question=request.question,
        )
        if isinstance(selected_url, str) and is_candidate_identity_url(selected_url):
            selected = select_game_candidate(discovered, selected_url=selected_url)
            if selected is not None:
                return selected
            return GameResolution(
                input_name=request.game,
                confirmed_name=request.game,
                confidence=0,
                candidates=discovered.candidates,
                ambiguous=bool(discovered.candidates),
            )
        if discovered.is_confirmed and not discovered.ambiguous:
            return discovered
        return discovered

    async def _initial_game_context(self, request: ChatRequest) -> GameResolution:
        """Use the user-supplied title for retrieval; resolve only explicit choices.

        Identity search is an exception path, not a prerequisite for every
        guide question. Retrieval can often establish the title directly from
        source evidence without paying an additional discovery request.
        """
        if request.metadata.get("confirmed_game") is True or request.metadata.get("selected_game_url"):
            return await self._resolve_request_game(request)
        return GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            confidence=1,
        )

    async def _recover_game_identity_if_needed(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        current: GameResolution,
    ) -> GameResolution:
        """Ask for identity confirmation only after retrieval cannot proceed."""
        if current.ambiguous or not current.is_confirmed:
            return current
        if sources or request.metadata.get("confirmed_game") is True:
            return current
        recovered = await self._resolve_request_game(request)
        # A retrieval miss is not evidence that the title is ambiguous.  In
        # particular, an identity provider can have a transient miss while a
        # clearly named game simply lacks indexed material for this question.
        # Only replace the optimistic request context when discovery actually
        # found a usable identity or competing candidates.  The former enables
        # the one cheap retry; the latter is the only case that should interrupt
        # a normal guide question and ask the player to choose a game.
        if recovered.is_confirmed or recovered.ambiguous or recovered.candidates:
            return recovered
        return current

    @staticmethod
    def _confirmed_resolution_from_request(request: ChatRequest) -> GameResolution | None:
        if request.metadata.get("confirmed_game") is not True:
            return None
        settings = get_settings()
        allow_hints = (
            settings.allow_evaluation_retrieval_hints
            and not settings.is_production
        )
        aliases = request.metadata.get("game_aliases") if allow_hints else None
        database_domains = request.metadata.get("database_domains") if allow_hints else None
        safe_aliases = [
            normalized
            for value in (aliases if isinstance(aliases, list) else [])[:8]
            if isinstance(value, str)
            and (normalized := " ".join(value.split()).strip())
            and len(normalized) <= 120
            and not any(marker in normalized.casefold() for marker in ("http://", "https://", "site:"))
        ]
        safe_domains = [
            host
            for value in (database_domains if isinstance(database_domains, list) else [])[:8]
            if isinstance(value, str)
            and (host := normalize_wiki_host(value)) is not None
        ]
        return GameResolution(
            input_name=request.game,
            confirmed_name=request.game,
            aliases=list(dict.fromkeys(safe_aliases)),
            database_domains=list(dict.fromkeys(safe_domains)),
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
