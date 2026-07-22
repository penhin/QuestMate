"""Result-page source classification and deterministic result balancing."""

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from quality_policy import SourcePolicy
from retrieval.wiki_domains import has_strong_wiki_page_signal
from schemas import GameResolution, Source
from search_components.ranking import canonical_source_key, limit_source_diversity


def effective_source_policy(
    *, configured: SourcePolicy, url: str, database_domains: list[str], official_domains: list[str],
    sources: dict[str, SourcePolicy],
) -> SourcePolicy:
    """Classify the result URL; its query source is only a weak prior."""
    domain = urlparse(url).netloc.casefold().split(":", 1)[0].removeprefix("www.")

    def matches_any(candidates: list[str]) -> bool:
        normalized = [
            value.casefold().split(":", 1)[0].strip(".").removeprefix("www.")
            for value in candidates if value
        ]
        return any(domain == value or domain.endswith(f".{value}") for value in normalized)

    if matches_any(["reddit.com", "steamcommunity.com"]):
        return sources["community"]
    if has_strong_wiki_page_signal(domain, url=url):
        return sources["wiki"]
    if matches_any(official_domains):
        return sources["official"]
    return sources["web"] if configured.source_type in {"official", "wiki", "community"} else configured


def resolution_authority_domains(resolution: GameResolution) -> list[str]:
    domains: list[str] = []
    for url in [*resolution.official_urls, *resolution.platform_urls]:
        domain = urlparse(str(url)).netloc.casefold().split(":", 1)[0].removeprefix("www.")
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def balanced_sources(
    *, strict_sources: list[Source], relaxed_sources: list[Source], total_results: int,
    min_strict_results: int,
) -> list[Source]:
    selected = limit_source_diversity(strict_sources, total_results=total_results)
    if len(selected) >= min_strict_results or len(selected) >= total_results:
        return selected
    selected_keys = {canonical_source_key(str(source.url)) for source in selected}
    fill_sources = [
        source for source in relaxed_sources
        if canonical_source_key(str(source.url)) not in selected_keys
    ]
    return limit_source_diversity(selected + fill_sources, total_results=total_results)


def extract_game_version(text: str) -> str | None:
    match = re.search(r"(?:patch|version|ver\.?|v|补丁|版本)\s*([0-9]+(?:\.[0-9]+){1,3})", text, re.I)
    return match.group(1) if match else None


def parse_source_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
