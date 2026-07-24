"""SearXNG JSON API adapter returning QuestMate's public Source model."""

from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from schemas import Source


class SearxngProvider:
    def __init__(self, *, base_url: str, timeout_seconds: int, max_results: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.calls = 0

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def search(self, query: str, *, max_results: int | None = None) -> list[Source]:
        if not self.configured:
            return []
        self.calls += 1
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=False) as client:
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json", "language": "all"},
                headers={"X-Forwarded-For": "127.0.0.1"},
            )
            response.raise_for_status()
        selected: list[Source] = []
        for item in response.json().get("results", [])[: max_results or self.max_results]:
            url = str(item.get("url") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                continue
            content = str(item.get("content") or "").strip()
            title = str(item.get("title") or url).strip()
            if not title:
                continue
            try:
                selected.append(Source(
                    title=title[:500], url=url, snippet=content[:600] or None,
                    evidence=content[:1600] or None, score=0.45,
                    source_type="web", trust_score=0.45, trust_label="普通",
                    fetched_at=datetime.now(timezone.utc),
                ))
            except ValueError:
                continue
        return selected
