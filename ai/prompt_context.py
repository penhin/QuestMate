"""Bounded prompt-context serializers shared by model-facing workflows."""

import json

from schemas import GameResolution, SessionMessage, Source


def history_context(history: list[SessionMessage]) -> str:
    context = "\n".join(
        f"{message.role}: {message.content[:500]}"
        for message in history[-8:]
        if message.content.strip()
    )
    return context[-4000:]


def source_context(sources: list[Source], *, max_chars: int = 16000) -> str:
    parts: list[str] = []
    remaining = max_chars
    for index, source in enumerate(sources, start=1):
        evidence = (source.evidence or source.snippet or "")[:2800]
        part = (
            f'<source index="{index}" type="{source.source_type}" trust="{source.trust_label}" '
            f'trust_score="{source.trust_score:.2f}">\n'
            f"title: {source.title}\nurl: {source.url}\n"
            f"published_at: {source.published_at.isoformat() if source.published_at else 'unknown'}\n"
            f"fetched_at: {source.fetched_at.isoformat() if source.fetched_at else 'unknown'}\n"
            f"game_version: {source.game_version or 'unknown'}\nevidence: {evidence}\n</source>"
        )
        if len(part) > remaining:
            if remaining < 240:
                break
            part = f"{part[:remaining - 11].rstrip()}\n</source>"
        parts.append(part)
        remaining -= len(part) + 1
        if remaining <= 0:
            break
    return "\n".join(parts)


def game_resolution_context(game_resolution: GameResolution | None) -> str:
    if game_resolution is None:
        return "No game resolution was provided."
    return json.dumps(
        {
            "input_name": game_resolution.input_name,
            "confirmed_name": game_resolution.confirmed_name,
            "aliases": game_resolution.aliases,
            "platform_urls": [str(url) for url in game_resolution.platform_urls],
            "official_urls": [str(url) for url in game_resolution.official_urls],
            "identity_urls": [str(url) for url in game_resolution.identity_urls],
            "database_domains": game_resolution.database_domains,
            "confidence": game_resolution.confidence,
            "ambiguous": game_resolution.ambiguous,
        }, ensure_ascii=False,
    )
