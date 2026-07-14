from collections.abc import AsyncIterator
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from llm import GuideLLM
from schemas import ChatRequest, ChatResponse, SearchPlan, SessionMessage, Source
from search import SearchProvider, TavilySearchProvider
from storage import conversation_store


AgentStreamEvent = tuple[Literal["status", "chunk", "done"], str | ChatResponse]


class QuestAgentState(TypedDict):
    request: ChatRequest
    history: list[SessionMessage]
    search_plan: SearchPlan
    sources: list[Source]
    answer: str


class QuestAgent:
    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        llm: GuideLLM | None = None,
    ) -> None:
        self.search_provider = search_provider or TavilySearchProvider()
        self.llm = llm or GuideLLM()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QuestAgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def run(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid4()
        request = request.model_copy(update={"session_id": session_id})
        is_new_session = not await conversation_store.session_exists(session_id)
        history = await conversation_store.get_recent_messages(session_id, limit=8)
        state = await self.graph.ainvoke(
            {"request": request, "history": history, "search_plan": SearchPlan(), "sources": [], "answer": ""}
        )
        answer = await self.llm.improve_answer(
            request=request,
            sources=state["sources"],
            answer=state["answer"],
            plan=state["search_plan"],
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

        yield ("status", self._status_for_plan_start(request.question))
        search_plan = await self.llm.plan_search(request=request, history=history)

        yield ("status", self._status_for_search(search_plan))
        sources = await self.search_provider.search(request.question, request.game, plan=search_plan)

        yield ("status", self._status_for_sources(sources))
        yield ("status", "整理答案：保留来源，核对版本")
        chunks: list[str] = []
        async for chunk in self.llm.stream_answer(request=request, sources=sources, plan=search_plan, history=history):
            chunks.append(chunk)
            yield ("chunk", chunk)

        answer = "".join(chunks)
        improved_answer = await self.llm.improve_answer(
            request=request,
            sources=sources,
            answer=answer,
            plan=search_plan,
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
        search_plan = await self.llm.plan_search(request=request, history=state["history"])
        return {**state, "search_plan": search_plan}

    async def _search(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        sources = await self.search_provider.search(request.question, request.game, plan=state["search_plan"])
        return {**state, "sources": sources}

    async def _answer(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        answer = await self.llm.answer(
            request=request,
            sources=state["sources"],
            plan=state["search_plan"],
            history=state["history"],
        )
        return {**state, "answer": answer}

    @staticmethod
    def _status_for_plan_start(question: str) -> str:
        if any(token in question for token in ("版本", "补丁", "更新", "削弱", "增强")):
            return "理解问题：先查版本变化"
        if any(token in question for token in ("怎么打", "打不过", "弱点", "Boss", "boss")):
            return "理解问题：查弱点和打法"
        if any(token in question for token in ("在哪", "哪里", "获得", "获取", "钥匙", "位置")):
            return "理解问题：查位置和获取方式"
        if any(token in question for token in ("任务", "支线", "下一步", "NPC", "npc")):
            return "理解问题：查任务步骤"
        return "理解问题：规划查询"

    @staticmethod
    def _status_for_search(search_plan: SearchPlan) -> str:
        labels = {
            "boss_strategy": "类型：Boss 打法；查弱点/阶段/社区打法",
            "item_location": "类型：物品位置；查地点/条件/路线",
            "quest_step": "类型：任务步骤；查 NPC/触发/顺序",
            "build": "类型：配装；查数值/装备/版本",
            "patch": "类型：版本变化；优先官方补丁",
            "lore": "类型：剧情背景；查事实和解释",
        }
        return labels.get(search_plan.intent, "类型：通用问题；筛选相关来源")

    @staticmethod
    def _status_for_sources(sources: list[Source]) -> str:
        if not sources:
            return "来源筛选：未找到强相关资料"
        trusted_count = sum(1 for source in sources if source.trust_score >= 0.8)
        if trusted_count:
            return f"来源筛选：保留 {len(sources)} 个，{trusted_count} 个高可信"
        return f"来源筛选：保留 {len(sources)} 个，交叉核对"


quest_agent = QuestAgent()
