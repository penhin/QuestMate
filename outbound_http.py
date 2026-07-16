"""Shared validation for server-side fetches of untrusted URLs."""

import asyncio
from urllib.parse import urlparse

from retrieval.wiki_domains import (
    is_safe_wiki_host,
    normalize_wiki_host,
    resolves_to_public_addresses,
)


def normalized_public_https_url(value: str) -> str | None:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.hostname is None
    ):
        return None
    host = normalize_wiki_host(parsed.hostname)
    if host is None or not is_safe_wiki_host(host):
        return None
    return parsed._replace(netloc=host, fragment="").geturl()


async def validate_public_https_url(value: str, *, dns_timeout: float = 3.0) -> str:
    normalized = normalized_public_https_url(value)
    if normalized is None:
        raise ValueError("Only public HTTPS URLs on port 443 are allowed")
    host = urlparse(normalized).hostname
    assert host is not None
    try:
        is_public = await asyncio.wait_for(
            asyncio.to_thread(resolves_to_public_addresses, host),
            timeout=dns_timeout,
        )
    except (OSError, TimeoutError):
        is_public = False
    if not is_public:
        raise ValueError("URL host does not resolve exclusively to public addresses")
    return normalized
