from anthropic import AsyncAnthropic
import httpx

from config import Settings, get_settings
from schemas import ChatRequest, Source


class ClaudeClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key) if self.settings.anthropic_api_key else None

    async def answer(self, *, request: ChatRequest, sources: list[Source]) -> str:
        if request.ai_provider == "deepseek" and request.ai_api_key:
            return await self._answer_with_deepseek(request=request, sources=sources)

        client = self._client
        model = self.settings.anthropic_model

        if request.ai_api_key:
            client_kwargs = {"api_key": request.ai_api_key}
            if request.ai_base_url:
                client_kwargs["base_url"] = request.ai_base_url.rstrip("/")
            client = AsyncAnthropic(**client_kwargs)
            model = request.ai_model or model
        elif request.ai_model:
            model = request.ai_model

        if client is None:
            return self._fallback_answer(game=request.game, question=request.question, sources=sources)

        source_context = "\n".join(
            (
                f"- {source.title}: {source.url}\n"
                f"  可信度: {source.trust_label} ({source.trust_score:.2f})\n"
                f"  {source.snippet or ''}"
            )
            for source in sources
        )
        message = await client.messages.create(
            model=model,
            max_tokens=1200,
            system=(
                "You are QuestMate, a precise game guide assistant. "
                "Answer in Chinese. Use only provided sources when sources are available. "
                "Prefer higher-credibility sources when sources disagree. "
                "Call out uncertainty and include concise citations by source title."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Game: {request.game}\n"
                        f"Question: {request.question}\n\n"
                        f"Sources:\n{source_context or 'No sources were found.'}"
                    ),
                }
            ],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")

    async def _answer_with_deepseek(self, *, request: ChatRequest, sources: list[Source]) -> str:
        source_context = "\n".join(
            (
                f"- {source.title}: {source.url}\n"
                f"  可信度: {source.trust_label} ({source.trust_score:.2f})\n"
                f"  {source.snippet or ''}"
            )
            for source in sources
        )
        base_url = (request.ai_base_url or "https://api.deepseek.com").rstrip("/")
        model = request.ai_model or "deepseek-chat"

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {request.ai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are QuestMate, a precise game guide assistant. "
                                "Answer in Chinese. Use only provided sources when sources are available. "
                                "Prefer higher-credibility sources when sources disagree. "
                                "Call out uncertainty and include concise citations by source title."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Game: {request.game}\n"
                                f"Question: {request.question}\n\n"
                                f"Sources:\n{source_context or 'No sources were found.'}"
                            ),
                        },
                    ],
                    "max_tokens": 1200,
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()

        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _fallback_answer(*, game: str, question: str, sources: list[Source]) -> str:
        source_note = f"已检索到 {len(sources)} 个来源。" if sources else "当前未配置 Anthropic/Tavily API key，返回骨架占位回答。"
        return f"关于《{game}》的问题：{question}\n\n{source_note}"
