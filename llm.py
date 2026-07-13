import json

from pydantic import ValidationError

from config import Settings, get_settings
from model_providers import ModelProvider, create_model_provider
from schemas import ChatRequest, PlannedSearchQuery, SearchPlan, SessionMessage, Source


class GuideLLM:
    def __init__(self, settings: Settings | None = None, provider: ModelProvider | None = None) -> None:
        self.settings = settings or get_settings()
        self._provider = provider

    async def plan_search(self, *, request: ChatRequest, history: list[SessionMessage] | None = None) -> SearchPlan:
        provider = self._provider or create_model_provider(request=request, settings=self.settings)
        if provider is None:
            return self._fallback_search_plan(question=request.question)

        try:
            content = await provider.complete(
                max_tokens=700,
                temperature=0,
                system=self._search_planner_system_prompt(),
                user=self._planner_user_prompt(request=request, history=history or []),
                json_mode=True,
            )
            return self._parse_search_plan(content, fallback_question=request.question)
        except Exception:
            return self._fallback_search_plan(question=request.question)

    async def answer(
        self,
        *,
        request: ChatRequest,
        sources: list[Source],
        history: list[SessionMessage] | None = None,
    ) -> str:
        provider = self._provider or create_model_provider(request=request, settings=self.settings)
        if provider is None:
            return self._fallback_answer(game=request.game, question=request.question, sources=sources)

        return await provider.complete(
            max_tokens=2400,
            temperature=0.2,
            system=(
                "You are QuestMate, a precise game guide assistant. "
                "Answer in Chinese with practical, actionable steps. "
                "Use relevant provided sources first, ignore clearly unrelated sources, and prefer higher-credibility "
                "sources when sources disagree. If sources are weak or incomplete, still give a useful general answer "
                "from game knowledge, but clearly mark the unsupported parts as uncertain. "
                "Do not answer with only a refusal unless the question is impossible to understand."
            ),
            user=self._answer_user_prompt(request=request, sources=sources, history=history or []),
        )

    async def summarize_title(self, *, request: ChatRequest, answer: str) -> str:
        provider = self._provider or create_model_provider(request=request, settings=self.settings)
        if provider is None:
            return self._fallback_title(request.question)

        try:
            title = await provider.complete(
                max_tokens=32,
                temperature=0,
                system="Summarize this game guide chat as a short session title. Return only the title.",
                user=f"Game: {request.game}\nQuestion: {request.question}\nAnswer:\n{answer[:1200]}",
            )
            return self._clean_title(title, fallback=request.question)
        except Exception:
            return self._fallback_title(request.question)

    @staticmethod
    def _fallback_answer(*, game: str, question: str, sources: list[Source]) -> str:
        source_note = f"已检索到 {len(sources)} 个来源。" if sources else "当前未配置 Anthropic/Tavily API key，返回骨架占位回答。"
        return f"关于《{game}》的问题：{question}\n\n{source_note}"

    @classmethod
    def _clean_title(cls, title: str, *, fallback: str) -> str:
        cleaned = title.strip().strip("\"'“”‘’")
        if not cleaned:
            return cls._fallback_title(fallback)
        return cleaned[:40]

    @staticmethod
    def _fallback_title(question: str) -> str:
        return question.strip()[:28] or "未命名会话"

    @staticmethod
    @staticmethod
    def _planner_user_prompt(*, request: ChatRequest, history: list[SessionMessage]) -> str:
        context = GuideLLM._history_context(history)
        return (
            f"Game: {request.game}\n"
            f"Recent conversation:\n{context or 'No prior messages.'}\n\n"
            f"Current question: {request.question}"
        )

    @staticmethod
    def _answer_user_prompt(*, request: ChatRequest, sources: list[Source], history: list[SessionMessage]) -> str:
        source_context = "\n".join(
            (
                f"- {source.title}: {source.url}\n"
                f"  可信度: {source.trust_label} ({source.trust_score:.2f})\n"
                f"  {source.snippet or ''}"
            )
            for source in sources
        )
        return (
            f"Game: {request.game}\n"
            f"Recent conversation:\n{GuideLLM._history_context(history) or 'No prior messages.'}\n\n"
            f"Current question: {request.question}\n\n"
            f"Sources:\n{source_context or 'No sources were found.'}"
        )

    @staticmethod
    def _history_context(history: list[SessionMessage]) -> str:
        return "\n".join(
            f"{message.role}: {message.content[:600]}"
            for message in history[-8:]
            if message.content.strip()
        )

    @staticmethod
    def _search_planner_system_prompt() -> str:
        return (
            "You plan web searches for a game guide assistant. "
            "Return only compact JSON with keys: intent, queries, missing_info. "
            "queries must contain 2 to 4 objects with source_type and query. "
            "source_type must be one of official, wiki, community, web. "
            "Use English keywords when useful, keep named entities exact, and do not include site: filters. "
            "Use wiki for facts, locations, NPCs, bosses, items, and quest steps. "
            "Use community for strategies, builds, timing, and player tactics. "
            "Use official only for patches, versions, events, outages, or balance changes. "
            "For boss strategy, include boss aliases, weakness, phase, dodge timing, and recommended build terms. "
            "Example: for Elden Ring 女武神怎么打, query wiki 'Malenia Blade of Miquella boss weakness phase 2' "
            "and community 'Malenia strategy recommended build dodge timing'."
        )

    @classmethod
    def _parse_search_plan(cls, content: str, *, fallback_question: str) -> SearchPlan:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("No JSON object found")
            data = json.loads(content[start:end])
            plan = SearchPlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return cls._fallback_search_plan(question=fallback_question)

        if not plan.queries:
            return cls._fallback_search_plan(question=fallback_question)

        return SearchPlan(
            intent=plan.intent or "general",
            queries=plan.queries[:4],
            missing_info=plan.missing_info[:4],
        )

    @staticmethod
    def _fallback_search_plan(*, question: str) -> SearchPlan:
        lowered = question.lower()
        queries: list[PlannedSearchQuery] = []

        if any(token in lowered for token in ("patch", "version", "update", "版本", "补丁", "更新")):
            queries.append(PlannedSearchQuery(source_type="official", query=f"{question} patch notes update"))
        if any(token in lowered for token in ("boss", "build", "打法", "配装", "怎么打")):
            queries.append(PlannedSearchQuery(source_type="community", query=f"{question} strategy build tips"))

        queries.extend(
            [
                PlannedSearchQuery(source_type="wiki", query=f"{question} wiki guide"),
                PlannedSearchQuery(source_type="web", query=question),
            ]
        )

        return SearchPlan(intent="general", queries=queries[:4], missing_info=[])
