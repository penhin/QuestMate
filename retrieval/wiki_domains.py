"""Generic discovery rules for candidate MediaWiki installations."""

from urllib.parse import urlparse

HOSTED_WIKI_SUFFIXES = (
    "fandom.com",
    "wiki.gg",
    "miraheze.org",
    "wikitide.org",
    "wikitide.net",
)


def is_probable_wiki_domain(domain: str, *, url: str = "") -> bool:
    """Return candidates for capability probing, not a fixed provider allowlist."""
    host = domain.casefold().split(":", 1)[0].strip(".")
    if not host:
        return False
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in HOSTED_WIKI_SUFFIXES):
        return True
    if host.endswith(".wiki") or any("wiki" in label for label in host.split(".")):
        return True
    return urlparse(url).path.casefold().startswith(("/wiki/", "/w/"))
