from collections.abc import AsyncIterator
import json
from typing import Protocol
from urllib.parse import urlparse

from anthropic import AsyncAnthropic
import httpx

from config import Settings
from schemas import ChatRequest


class ModelProvider(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        ...

    async def stream_complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        ...


class AnthropicProvider:
    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

        self.client = AsyncAnthropic(**client_kwargs)
        self.model = model

    async def aclose(self) -> None:
        await self.client.close()

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")

    async def stream_complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


class OpenAICompatibleProvider:
    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None

    def _http_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(timeout=60)
        return self.client

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        response = await self._http_client().post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()

        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    async def stream_complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with self._http_client().stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield str(delta)


def create_model_provider(*, request: ChatRequest, settings: Settings) -> ModelProvider | None:
    if request.ai_provider == "deepseek":
        if not request.ai_api_key:
            return None
        base_url = request.ai_base_url or "https://api.deepseek.com"
        if not _model_endpoint_allowed(
            base_url,
            built_in_hosts={"api.deepseek.com"},
            settings=settings,
        ):
            return None
        return OpenAICompatibleProvider(
            api_key=request.ai_api_key,
            model=request.ai_model or "deepseek-chat",
            base_url=base_url,
        )

    if request.ai_api_key:
        base_url = request.ai_base_url
        if base_url and not _model_endpoint_allowed(
            base_url,
            built_in_hosts={"api.anthropic.com"},
            settings=settings,
        ):
            return None
        return AnthropicProvider(
            api_key=request.ai_api_key,
            model=request.ai_model or settings.anthropic_model,
            base_url=base_url,
        )

    if not settings.anthropic_api_key:
        return None
    # Request-controlled endpoint/model values must never be combined with a
    # server-owned credential.
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        base_url=None,
    )


def _model_endpoint_allowed(
    base_url: str,
    *,
    built_in_hosts: set[str],
    settings: Settings,
) -> bool:
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        return False
    host = parsed.hostname.casefold().strip(".")
    return host in built_in_hosts or host in settings.allowed_custom_model_hosts
