"""Source diversity and version-aware scoring independent of search transport."""

from urllib.parse import urlparse, urlunparse

from quality_policy import STABLE_FACT_INTENTS, VERSION_SCORE_POLICY, VERSION_SENSITIVE_INTENTS, is_version_sensitive_question, source_domain_limit
from schemas import Source


def canonical_source_key(url: str) -> str:
    parsed = urlparse(url)
    if any(value in parsed.netloc.lower() for value in ("steamcommunity.com", "reddit.com")):
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    return url


def limit_source_diversity(sources: list[Source], *, total_results: int) -> list[Source]:
    selected: list[Source] = []
    domain_counts: dict[str, int] = {}
    for source in sources:
        domain = urlparse(str(source.url)).netloc.lower()
        if domain_counts.get(domain, 0) >= source_domain_limit(domain):
            continue
        selected.append(source)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(selected) >= total_results:
            break
    return selected


def version_safety_score(*, intent: str, source_type: str, text: str, version_sensitive: bool = False) -> float:
    version_sensitive = version_sensitive or intent in VERSION_SENSITIVE_INTENTS
    if version_sensitive and source_type == "official":
        return VERSION_SCORE_POLICY.official_sensitive
    if version_sensitive and is_version_sensitive_question(text.lower()):
        return VERSION_SCORE_POLICY.versioned_sensitive
    if version_sensitive:
        return VERSION_SCORE_POLICY.undated_sensitive
    if intent in STABLE_FACT_INTENTS:
        return VERSION_SCORE_POLICY.stable_fact
    return VERSION_SCORE_POLICY.default
