"""Free direct search adapter for game-specific MediaWiki sites."""

import json
import re
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class MediaWikiClient:
    user_agent = "QuestMate/0.1 (local game guide search)"

    def search(self, *, domain: str, query: str, max_results: int) -> dict[str, Any]:
        search_query = " ".join(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", query)) or query
        params = urlencode(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": search_query,
                "gsrlimit": max_results,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "format": "json",
                "formatversion": 2,
                "origin": "*",
            }
        )
        request = Request(
            f"https://{domain}/api.php?{params}",
            headers={"User-Agent": self.user_agent},
        )
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
        pages = sorted(
            payload.get("query", {}).get("pages", []),
            key=lambda page: int(page.get("index") or 9999),
        )
        results = []
        for page in pages:
            title = str(page.get("title") or "").strip()
            if not title:
                continue
            revisions = page.get("revisions") or []
            content = ""
            if revisions:
                content = str(revisions[0].get("slots", {}).get("main", {}).get("content") or "")
            results.append(
                {
                    "title": title,
                    "url": f"https://{domain}/wiki/{quote(title.replace(' ', '_'))}",
                    "content": self._clean_wikitext(content)[:6000],
                    "score": 0.9,
                }
            )
        return {"results": results}

    @staticmethod
    def _clean_wikitext(content: str) -> str:
        cleaned = re.sub(r"<!--.*?-->|<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", " ", content, flags=re.S | re.I)
        cleaned = re.sub(r"\[\[(?:File|Image):[^\]]+\]\]", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", cleaned)
        cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
        for _ in range(3):
            cleaned = re.sub(r"\{\{[^{}]*\}\}", " ", cleaned)
        cleaned = re.sub(r"'{2,}|={2,}|\[https?://\S+\s*([^\]]*)\]", r" \1 ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()
