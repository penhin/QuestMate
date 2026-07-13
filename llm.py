import json

from pydantic import ValidationError

from config import Settings, get_settings
from model_providers import ModelProvider, create_model_provider
from query_tokens import is_query_entity_token, relevance_tokens
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
            return self._fallback_title(request.game, request.question)

        try:
            title = await provider.complete(
                max_tokens=32,
                temperature=0,
                system=(
                    "Generate a short game guide session title. "
                    "The fixed format is: {game}, {short question summary}. "
                    "Compress the question summary to 4-10 Chinese characters or 2-5 English words. "
                    "Return only the title."
                ),
                user=f"Game: {request.game}\nFirst question: {request.question}",
            )
            return self._clean_title(title, fallback=self._fallback_title(request.game, request.question), game=request.game)
        except Exception:
            return self._fallback_title(request.game, request.question)

    @staticmethod
    def _fallback_answer(*, game: str, question: str, sources: list[Source]) -> str:
        if not sources:
            return (
                f"关于《{game}》的问题：{question}\n\n"
                "我没有找到能直接回答这个问题的有效资料。可以换成更具体的问题，比如地点、Boss 名、道具名或任务名。"
            )

        if not GuideLLM._has_question_specific_sources(question=question, sources=sources):
            return (
                f"关于《{game}》的问题：{question}\n\n"
                "我找到了一些游戏资料，但它们没有直接覆盖这个问题。请补充更具体的名称或场景，我再继续查。"
            )

        return (
            f"关于《{game}》的问题：{question}\n\n"
            "我找到了相关资料，但当前没有可用模型来整理完整答案。你可以先查看下方来源。"
        )

    @classmethod
    def _clean_title(cls, title: str, *, fallback: str, game: str | None = None) -> str:
        cleaned = title.strip().strip("\"'“”‘’")
        if not cleaned:
            return fallback
        game_name = (game or "").strip()
        if game_name:
            separator = "，" if "，" in cleaned and "," not in cleaned else ","
            if separator in cleaned:
                cleaned = cleaned.split(separator, 1)[1].strip()
            return f"{game_name}, {cleaned or fallback.split(',', 1)[-1].strip()}"[:40]
        return cleaned[:40]

    @staticmethod
    def _fallback_title(game: str, question: str) -> str:
        summary = question.strip()[:16] or "未命名"
        return f"{game.strip() or '游戏'}, {summary}"

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
    def _has_question_specific_sources(*, question: str, sources: list[Source]) -> bool:
        tokens = [
            token
            for token in GuideLLM._question_tokens(question)
            if is_query_entity_token(token)
        ]
        if not tokens:
            return False

        source_text = " ".join(
            f"{source.title} {source.url} {source.snippet or ''}".lower()
            for source in sources
        )
        return any(token in source_text for token in tokens)

    @staticmethod
    def _question_tokens(question: str) -> list[str]:
        return relevance_tokens(question)

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
            "For boss strategy, include boss aliases, weakness, phase, dodge timing, and recommended build terms."
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
