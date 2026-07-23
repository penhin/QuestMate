import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
import inspect
from time import perf_counter
from typing import Literal
from uuid import uuid4

import structlog

from knowledge import KnowledgeStore, knowledge_store
from config import get_settings
from llm import GuideLLM
from retrieval.coordinator import RetrievalCoordinator, merge_search_plans
from retrieval.evidence_pool import canonical_source_url, merge_source_evidence, rank_sources, source_rank
from retrieval.pipeline import RetrievalStage
from retrieval.source_quality import matches_game_identity
from ai.fallback_planning import fallback_search_plan
from agents import (
    AgentTrace,
    AnswerAgent,
    EvidenceAgent,
    IdentityAgent,
    IdentityResolver,
    PlanningAgent,
    RetrievalAgent,
)
from request_safety import requires_safe_refusal
from orchestration.diagnostics import evaluation_diagnostics
from orchestration.graph import build_request_graph
from orchestration.state import QuestAgentState
from orchestration import status as orchestration_status
from router import IntentRouter, RouteDecision
from runtime import QuestRuntime
from workflow import WorkflowRouter
from workflows.guide import GuideWorkflow
from workflows.build import BuildWorkflow
from workflows.analysis import AnalysisWorkflow
from schemas import ChatRequest, ChatResponse, GameResolution, InvestigationState, SearchPlan, SessionMessage, Source
from search import SearchProvider, TavilySearchProvider
from storage import conversation_store


AgentStreamEvent = tuple[Literal["status", "chunk", "done"], str | ChatResponse]
logger = structlog.get_logger()


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
        self.evidence_agent = EvidenceAgent(self.llm)
        self.retrieval = RetrievalCoordinator(
            knowledge=self.knowledge,
            search_provider=self.search_provider,
            llm=self.evidence_agent,
            max_results=get_settings().search_max_results,
            max_investigation_model_calls=1,
        )
        self.identity_resolver = IdentityResolver(self.search_provider)
        self.identity_agent = IdentityAgent(
            initial_context=self._initial_game_context,
            recover_context=self._recover_game_identity_if_needed,
        )
        self.planning_agent = PlanningAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self._retrieve_after_identity_check)
        self.answer_agent = AnswerAgent(self.llm)
        self.intent_router = IntentRouter()
        self.runtime = QuestRuntime()
        self.workflow_router = WorkflowRouter()
        self.guide_workflow = GuideWorkflow(
            retrieve_after_identity_check=self._retrieve_after_identity_check,
            render_answer=self._render_answer,
            safety_refusal_message=self._safety_refusal_message,
            verification_router=self.workflow_router,
        )
        self.build_workflow = BuildWorkflow(
            retrieve_after_identity_check=self._retrieve_after_identity_check,
            render_answer=self._render_answer,
            safety_refusal_message=self._safety_refusal_message,
            verification_router=self.workflow_router,
        )
        self.analysis_workflow = AnalysisWorkflow(
            retrieve_after_identity_check=self._retrieve_after_identity_check,
            render_answer=self._render_answer,
            safety_refusal_message=self._safety_refusal_message,
            verification_router=self.workflow_router,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        return build_request_graph(
            identity_node=self._resolve_game,
            planning_node=self._plan,
            routing_node=self._route,
            guide_workflow_node=self._run_guide_workflow,
            build_workflow_node=self._run_build_workflow,
            analysis_workflow_node=self._run_analysis_workflow,
            task_workflow_router=self._select_task_workflow,
            retrieval_node=self._search,
            verification_node=self._verify,
            workflow_router=self.workflow_router.next_after_research,
            answer_node=self._answer,
        )

    async def run(self, request: ChatRequest) -> ChatResponse:
        with self._search_usage_scope():
            provider_scope = getattr(self.llm, "provider_scope", None)
            if callable(provider_scope):
                async with provider_scope(request):
                    return await self._run_in_runtime(request)
            return await self._run_in_runtime(request)

    async def _run_in_runtime(self, request: ChatRequest) -> ChatResponse:
        user_id = request.metadata.get("user_id")
        return await self.runtime.execute(
            user_id=user_id if isinstance(user_id, str) else None,
            tools={"search": self.search_provider, "knowledge": self.knowledge},
            operation=lambda: self._run_with_timeout(request),
        )

    async def _run_with_timeout(self, request: ChatRequest) -> ChatResponse:
        """Keep a stalled upstream from consuming an unbounded user request."""
        started = perf_counter()
        usage_before = self._search_usage_snapshot()
        try:
            async with asyncio.timeout(get_settings().agent_request_timeout_seconds):
                return await self._run(request)
        except TimeoutError:
            session_id = request.session_id or uuid4()
            return ChatResponse(
                session_id=session_id,
                answer=(
                    "当前未能在时限内取得足以核实的游戏资料，因此不提供未经证实的攻略结论。"
                    "请稍后重试，或补充具体版本、地点或物品名称。"
                ),
                sources=[],
                timings_ms={"request_timeout": self._elapsed_ms(started)},
                usage=self._request_usage(usage_before),
                diagnostics=self._evaluation_diagnostics(request=request, path="request_timeout"),
            )

    async def _run(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        usage_before = self._search_usage_snapshot()
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
                usage=self._request_usage(usage_before),
                diagnostics=self._evaluation_diagnostics(request=request, path="safety_gate"),
            )
        history = await conversation_store.get_recent_messages(session_id, limit=8)
        state = await self.graph.ainvoke(
            {
                "request": request,
                "history": history,
                "game_resolution": GameResolution(input_name=request.game),
                "search_plan": SearchPlan(),
                "route": RouteDecision(),
                "sources": [],
                "investigation": InvestigationState(goal=request.question),
                "answer": "",
                "timings_ms": {},
                "agent_trace": [],
            }
        )
        game_resolution = state["game_resolution"]
        response_path = "planner_safety_gate" if state["search_plan"].safety_refusal else "answer"
        if state["search_plan"].safety_refusal:
            state["answer"] = self._safety_refusal_message()
        if self._needs_game_confirmation(game_resolution):
            return ChatResponse(
                session_id=session_id,
                answer=self._game_confirmation_message(game_resolution),
                sources=[],
                is_new=is_new_session,
                needs_game_confirmation=True,
                game_candidates=game_resolution.candidates,
                timings_ms=state["timings_ms"],
                usage=self._request_usage(
                    usage_before, state["investigation"], state["search_plan"]
                ),
                diagnostics=self._evaluation_diagnostics(
                    request=request, path="game_confirmation", agent_trace=state["agent_trace"]
                ),
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
        if state["search_plan"].safety_refusal:
            answer = state["answer"]
            state["timings_ms"]["improvement"] = 0
        elif state["sources"]:
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
            usage=self._request_usage(
                usage_before, state["investigation"], state["search_plan"]
            ),
            diagnostics=self._evaluation_diagnostics(
                request=request,
                path=response_path,
                sources=state["sources"],
                plan=state["search_plan"],
                game_resolution=state["game_resolution"],
                answer=answer,
                agent_trace=state["agent_trace"],
            ),
        )
        await conversation_store.save_chat(request, response)
        return response

    async def stream(self, request: ChatRequest) -> AsyncIterator[AgentStreamEvent]:
        with self._search_usage_scope():
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
        game_resolution = await self.identity_agent.initial(request)
        timings_ms = {"identity": self._elapsed_ms(identity_started)}
        yield ("status", self._status_for_plan_start(request.question))
        planning_started = perf_counter()
        search_plan = await self.planning_agent.plan(
            request=request, history=history, game_resolution=game_resolution
        )
        timings_ms["planning"] = self._elapsed_ms(planning_started)
        route = self.intent_router.route(plan=search_plan, game_resolution=game_resolution)

        if search_plan.safety_refusal:
            response = ChatResponse(
                session_id=session_id,
                answer=self._safety_refusal_message(),
                sources=[],
                is_new=is_new_session,
                timings_ms=timings_ms,
            )
            await conversation_store.save_chat(request, response)
            yield ("done", response)
            return

        yield ("status", self._status_for_search(search_plan))
        yield ("status", self._status_for_route(route))
        outcome, game_resolution = await self.retrieval_agent.investigate(
            request=request,
            history=history,
            plan=search_plan,
            game_resolution=game_resolution,
            timings_ms=timings_ms,
        )
        sources, search_plan, refined = outcome.sources, outcome.plan, outcome.refined

        game_resolution = await self.identity_agent.recover(
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
        self._add_optional_argument(
            self.answer_agent.stream_answer, stream_kwargs, "investigation", outcome.investigation
        )
        async for chunk in self.answer_agent.stream_answer(**stream_kwargs):
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
        search_plan = await self.planning_agent.plan(
            request=request,
            history=state["history"],
            game_resolution=state["game_resolution"],
        )
        return {**state, "search_plan": search_plan, "timings_ms": {
            **state["timings_ms"], "planning": self._elapsed_ms(started)
        }, "agent_trace": [*state["agent_trace"], AgentTrace("planning", "plan")]}

    async def _route(self, state: QuestAgentState) -> QuestAgentState:
        """Create the typed task-workflow hand-off after planning.

        Phase 1 keeps the existing retrieval graph intact.  Subsequent phases
        replace its common tail with the selected task graph.
        """
        route = self.intent_router.route(
            plan=state["search_plan"], game_resolution=state["game_resolution"]
        )
        return {
            **state,
            "route": route,
            "agent_trace": [*state["agent_trace"], AgentTrace("workflow_router", route.intent)],
        }

    @staticmethod
    def _select_task_workflow(state: QuestAgentState) -> str:
        return state["route"].intent

    async def _run_guide_workflow(self, state: QuestAgentState) -> QuestAgentState:
        guide_state = await self.guide_workflow.run(
            request=state["request"],
            history=state["history"],
            game_resolution=state["game_resolution"],
            search_plan=state["search_plan"],
            timings_ms=state["timings_ms"],
            agent_trace=state["agent_trace"],
        )
        return {
            **state,
            "game_resolution": guide_state["game"],
            "search_plan": guide_state["search_plan"],
            "sources": guide_state["evidence"],
            "investigation": guide_state["investigation"],
            "answer": guide_state["answer"],
            "timings_ms": guide_state["timings_ms"],
            "agent_trace": guide_state["agent_trace"],
        }

    async def _run_build_workflow(self, state: QuestAgentState) -> QuestAgentState:
        build_state = await self.build_workflow.run(
            request=state["request"], history=state["history"],
            game_resolution=state["game_resolution"], search_plan=state["search_plan"],
            timings_ms=state["timings_ms"], agent_trace=state["agent_trace"],
        )
        return {
            **state, "game_resolution": build_state["game"],
            "search_plan": build_state["search_plan"], "sources": build_state["evidence"],
            "investigation": build_state["investigation"], "answer": build_state["answer"],
            "timings_ms": build_state["timings_ms"], "agent_trace": build_state["agent_trace"],
        }

    async def _run_analysis_workflow(self, state: QuestAgentState) -> QuestAgentState:
        analysis_state = await self.analysis_workflow.run(
            request=state["request"], history=state["history"],
            game_resolution=state["game_resolution"], search_plan=state["search_plan"],
            timings_ms=state["timings_ms"], agent_trace=state["agent_trace"],
        )
        return {
            **state, "game_resolution": analysis_state["game"],
            "search_plan": analysis_state["search_plan"], "sources": analysis_state["evidence"],
            "investigation": analysis_state["investigation"], "answer": analysis_state["answer"],
            "timings_ms": analysis_state["timings_ms"], "agent_trace": analysis_state["agent_trace"],
        }

    async def _resolve_game(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        existing = state.get("game_resolution")
        if existing is not None and existing.is_confirmed:
            return state
        started = perf_counter()
        game_resolution = await self.identity_agent.initial(request)
        return {**state, "game_resolution": game_resolution, "timings_ms": {
            **state["timings_ms"], "identity": self._elapsed_ms(started)
        }, "agent_trace": [*state["agent_trace"], AgentTrace("identity", "initial")]}

    async def _search(self, state: QuestAgentState) -> QuestAgentState:
        if state["search_plan"].safety_refusal:
            return {**state, "sources": [], "investigation": InvestigationState(goal=state["request"].question)}
        request = state["request"]
        outcome, game_resolution = await self.retrieval_agent.investigate(
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
            "agent_trace": [*state["agent_trace"], AgentTrace(
                "retrieval_evidence", "investigate", len(outcome.sources), outcome.refined
            )],
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
        # Only an already-known conflicting identity blocks retrieval.  For an
        # otherwise unverified title, the first evidence wave is safer and
        # cheaper than treating a discovery miss as user-facing ambiguity.
        # Retrieval still never chooses between explicit competing candidates.
        if game_resolution.ambiguous:
            outcome = await self.retrieval.investigate(
                request=request,
                history=history,
                plan=plan,
                game_resolution=game_resolution,
                initial_sources=[],
            )
            return outcome, game_resolution
        retrieval_started = perf_counter()
        initial_sources, initial_stages = await self._retrieve_initial_batch(
            request.question,
            request.game,
            plan=plan,
            game_resolution=resolved,
        )
        if timings_ms is not None:
            timings_ms["retrieval_initial"] = self._elapsed_ms(retrieval_started)
        # A query can return a perfectly coherent answer for a *different*
        # game when the supplied title is short, misspelled, or shared.  The
        # relaxed retrieval path intentionally permits localized pages whose
        # title is established by the query, so do not use its presence as
        # identity proof.  Recover identity only when no retained passage
        # independently carries the requested title or a previously confirmed
        # alias.  This keeps ordinary title-bearing evidence on the fast path
        # while preventing cross-game evidence from being rendered as fact.
        if not initial_sources or not self._sources_establish_game_identity(
            sources=initial_sources,
            resolution=game_resolution,
        ):
            resolution_started = perf_counter()
            resolved = await self._recover_game_identity_if_needed(
                request=request,
                sources=[],
                current=game_resolution,
            )
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
                # Discovery did not identify a competing title, but the
                # retained pages still failed the source-local identity gate.
                # Do not let those pages become a fallback answer.
                if initial_sources:
                    initial_sources = []
                    initial_stages = []
                resolved = game_resolution
            elif initial_sources and not self._sources_establish_game_identity(
                sources=initial_sources,
                resolution=resolved,
            ):
                # A unique candidate that still does not match any retained
                # page cannot safely bind those pages to the user's game.
                # Continue through the conservative evidence path rather than
                # presenting a plausible answer from another title.
                initial_sources = []
                initial_stages = []
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
            initial_sources, initial_stages = await self._retrieve_initial_batch(
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
        investigate_kwargs = dict(
            request=request,
            history=history,
            plan=plan,
            game_resolution=resolved,
            initial_sources=initial_sources,
        )
        self._add_optional_argument(
            self.retrieval.investigate, investigate_kwargs, "initial_stages", initial_stages
        )
        outcome = await self.retrieval.investigate(**investigate_kwargs)
        if timings_ms is not None:
            timings_ms["investigation"] = self._elapsed_ms(investigation_started)
        return outcome, resolved

    async def _retrieve_initial_batch(
        self,
        question: str,
        game: str,
        *,
        plan: SearchPlan,
        game_resolution: GameResolution,
    ) -> tuple[list[Source], list[RetrievalStage]]:
        """Use pipeline batches when available without breaking custom retrievers."""
        retrieve_batch = getattr(self.retrieval, "retrieve_batch", None)
        if callable(retrieve_batch):
            batch = await retrieve_batch(
                question, game, plan=plan, game_resolution=game_resolution
            )
            return list(batch.sources), list(batch.stages)
        sources = await self.retrieval.retrieve_sources(
            question, game, plan=plan, game_resolution=game_resolution
        )
        return sources, []

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

    @staticmethod
    def _sources_establish_game_identity(
        *, sources: list[Source], resolution: GameResolution
    ) -> bool:
        """Require a retained page to name this game before it can answer it.

        The test is intentionally structural and source-local: canonical title
        plus server-confirmed aliases are accepted, but a matching search query
        or a source type is never treated as proof of identity.
        """
        names = list(dict.fromkeys([
            resolution.input_name,
            resolution.confirmed_name or "",
            *resolution.aliases,
        ]))
        return any(
            matches_game_identity(
                text=f"{source.title} {source.url} {source.evidence or source.snippet or ''}",
                game_names=names,
            )
            for source in sources
        )

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
        if state["search_plan"].safety_refusal:
            return {**state, "answer": self._safety_refusal_message()}
        started = perf_counter()
        answer = await self._render_answer(
            request=state["request"],
            sources=state["sources"],
            plan=state["search_plan"],
            game_resolution=state["game_resolution"],
            history=state["history"],
            investigation=state["investigation"],
        )
        return {**state, "answer": answer, "timings_ms": {
            **state["timings_ms"], "answer": self._elapsed_ms(started)
        }, "agent_trace": [*state["agent_trace"], AgentTrace(
            "answer", "render", len(state["sources"])
        )]}

    async def _render_answer(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        plan: SearchPlan,
        game_resolution: GameResolution,
        history: list[SessionMessage],
        investigation: InvestigationState,
    ) -> str:
        answer_kwargs = dict(
            request=request,
            sources=sources,
            plan=plan,
            game_resolution=game_resolution,
            history=history,
        )
        self._add_optional_argument(self.answer_agent.answer, answer_kwargs, "investigation", investigation)
        return await self.answer_agent.answer(**answer_kwargs)

    async def _verify(self, state: QuestAgentState) -> QuestAgentState:
        """Checkpoint required by verification-oriented workflows.

        Evidence policies remain deterministic and answer-side enforcement
        stays in ``GuideLLM``; this node makes that verification boundary an
        explicit, inspectable execution step.
        """
        workflow = self.workflow_router.classify(state["search_plan"])
        return {
            **state,
            "agent_trace": [*state["agent_trace"], AgentTrace(
                "verification", workflow.value, len(state["sources"])
            )],
        }

    @staticmethod
    def _safety_refusal_message() -> str:
        return "我不能协助绕过安全限制、获取受保护信息或实施不当行为。可以继续回答正常的游戏攻略问题。"

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return round((perf_counter() - started) * 1000)

    def _evaluation_diagnostics(
        self,
        *,
        request: ChatRequest,
        path: str,
        sources: list[Source] | None = None,
        plan: SearchPlan | None = None,
        game_resolution: GameResolution | None = None,
        answer: str = "",
        agent_trace: list[AgentTrace] | None = None,
    ) -> dict[str, str | int]:
        return evaluation_diagnostics(
            request=request,
            path=path,
            llm=self.llm,
            sources=sources,
            plan=plan,
            game_resolution=game_resolution,
            answer=answer,
            agent_trace=agent_trace,
        )

    def _search_usage_snapshot(self) -> dict[str, int]:
        snapshot = getattr(self.search_provider, "usage_snapshot", None)
        if not callable(snapshot):
            return {"tavily_paid_calls": 0, "tavily_cache_hits": 0}
        values = snapshot()
        if not isinstance(values, dict):
            return {"tavily_paid_calls": 0, "tavily_cache_hits": 0}
        return {
            key: max(0, int(values.get(key, 0)))
            for key in ("tavily_paid_calls", "tavily_cache_hits")
        }

    def _search_usage_scope(self):
        factory = getattr(self.search_provider, "usage_scope", None)
        return factory() if callable(factory) else nullcontext()

    def _request_usage(
        self,
        before: dict[str, int],
        investigation: InvestigationState | None = None,
        plan: SearchPlan | None = None,
    ) -> dict[str, int]:
        after = self._search_usage_snapshot()
        usage = {
            key: max(0, after.get(key, 0) - before.get(key, 0))
            for key in ("tavily_paid_calls", "tavily_cache_hits")
        }
        model_usage = getattr(self.llm, "request_usage", None)
        values = model_usage() if callable(model_usage) else {}
        usage["model_calls"] = max(0, int(values.get("model_calls", 0))) if isinstance(values, dict) else 0
        usage["investigation_hops"] = investigation.hop_count if investigation is not None else 0
        usage["complex_evidence_path"] = int(bool(
            usage["model_calls"] > 2
            or
            investigation
            and (
                investigation.hop_count > 0
                or plan is not None and (
                    plan.version_sensitive
                    or bool(plan.missing_info)
                    or len(plan.named_entity_groups) >= 2
                )
            )
        ))
        return usage

    @staticmethod
    def _add_optional_argument(callable_obj, kwargs: dict, name: str, value) -> None:
        signature = inspect.signature(callable_obj)
        parameters = signature.parameters.values()
        if name in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ):
            kwargs[name] = value

    @staticmethod
    def _status_for_plan_start(question: str) -> str:
        return orchestration_status.plan_start(question)

    @staticmethod
    def _initial_title(request: ChatRequest) -> str:
        """Keep title generation off the critical answer path."""
        return request.question.strip()[:40] or request.game.strip() or "未命名会话"

    @staticmethod
    def _status_for_search(search_plan: SearchPlan) -> str:
        return orchestration_status.search(search_plan)

    @staticmethod
    def _status_for_route(route: RouteDecision) -> str:
        labels = {
            "guide": "任务工作流：攻略与路线",
            "build": "任务工作流：配装与属性",
            "analysis": "任务工作流：机制与分析",
        }
        return labels[route.intent]

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
        return await self._identity_resolver().resolve_request_game(request)

    async def _initial_game_context(self, request: ChatRequest) -> GameResolution:
        return await self._identity_resolver().initial_context(request)

    async def _recover_game_identity_if_needed(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        current: GameResolution,
    ) -> GameResolution:
        return await self._identity_resolver().recover_if_needed(
            request=request, sources=sources, current=current
        )

    def _identity_resolver(self) -> IdentityResolver:
        """Lazily construct the specialist for legacy lightweight instances."""
        resolver = getattr(self, "identity_resolver", None)
        if resolver is None:
            resolver = IdentityResolver(self.search_provider)
            self.identity_resolver = resolver
        elif resolver.search_provider is not self.search_provider:
            resolver.search_provider = self.search_provider
        return resolver

    @staticmethod
    def _confirmed_resolution_from_request(request: ChatRequest) -> GameResolution | None:
        return IdentityResolver.confirmed_resolution_from_request(request)

    @staticmethod
    def _status_for_sources(sources: list[Source]) -> str:
        return orchestration_status.sources(sources)


quest_agent = QuestAgent()
