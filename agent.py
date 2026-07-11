from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from llm import ClaudeClient
from schemas import ChatRequest, ChatResponse, Source
from search import SearchProvider, TavilySearchProvider
from storage import conversation_store


class QuestAgentState(TypedDict):
    request: ChatRequest
    sources: list[Source]
    answer: str


class QuestAgent:
    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        llm: ClaudeClient | None = None,
    ) -> None:
        self.search_provider = search_provider or TavilySearchProvider()
        self.llm = llm or ClaudeClient()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(QuestAgentState)
        graph.add_node("search", self._search)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("search")
        graph.add_edge("search", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def run(self, request: ChatRequest) -> ChatResponse:
        state = await self.graph.ainvoke({"request": request, "sources": [], "answer": ""})
        response = ChatResponse(
            session_id=request.session_id or uuid4(),
            answer=state["answer"],
            sources=state["sources"],
        )
        await conversation_store.save_chat(request, response)
        return response

    async def _search(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        sources = await self.search_provider.search(request.question, request.game)
        return {**state, "sources": sources}

    async def _answer(self, state: QuestAgentState) -> QuestAgentState:
        request = state["request"]
        answer = await self.llm.answer(request=request, sources=state["sources"])
        return {**state, "answer": answer}


quest_agent = QuestAgent()
