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
        title = await self.llm.summarize_title(request=request, answer=state["answer"]) if is_new_session else None
        response = ChatResponse(
            session_id=session_id,
            answer=state["answer"],
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

        yield ("status", "规划查询")
        search_plan = await self.llm.plan_search(request=request, history=history)

        yield ("status", "检索资料")
        sources = await self.search_provider.search(request.question, request.game, plan=search_plan)

        yield ("status", "生成回答")
        chunks: list[str] = []
        async for chunk in self.llm.stream_answer(request=request, sources=sources, history=history):
            chunks.append(chunk)
            yield ("chunk", chunk)

        answer = "".join(chunks)
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
        answer = await self.llm.answer(request=request, sources=state["sources"], history=state["history"])
        return {**state, "answer": answer}


quest_agent = QuestAgent()
