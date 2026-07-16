"""Conservative discovery and request-safety rules for MediaWiki hosts."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


HOSTED_WIKI_SUFFIXES = (
    "fandom.com",
    "wiki.gg",
    "miraheze.org",
    "wikitide.org",
    "wikitide.net",
)

_NON_PUBLIC_HOST_SUFFIXES = (
    ".home",
    ".internal",
    ".invalid",
    ".lan",
    ".local",
    ".localhost",
    ".test",
)
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def normalize_wiki_host(domain: str) -> str | None:
    """Return a DNS hostname only when ``domain`` cannot alter the request URL."""
    candidate = domain.strip().rstrip(".")
    if (
        not candidate
        or any(character.isspace() for character in candidate)
        or any(character in candidate for character in "/?#@")
        or ":" in candidate
    ):
        return None
    try:
        host = candidate.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        return None
    if len(host) > 253 or "." not in host:
        return None
    if not all(_HOST_LABEL.fullmatch(label) for label in host.split(".")):
        return None
    return host


def is_safe_wiki_host(domain: str) -> bool:
    """Reject malformed, local, reserved, and literal-IP MediaWiki targets."""
    host = normalize_wiki_host(domain)
    if host is None:
        return False
    if host in {"localhost", "localhost.localdomain"} or host.endswith(_NON_PUBLIC_HOST_SUFFIXES):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False


def resolves_to_public_addresses(domain: str) -> bool:
    """Resolve a production network target and require every answer to be public."""
    host = normalize_wiki_host(domain)
    if host is None or not is_safe_wiki_host(host):
        return False
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            if result and result[4]
        }
    except (OSError, UnicodeError):
        return False
    if not addresses:
        return False
    try:
        return all(ipaddress.ip_address(address).is_global for address in addresses)
    except ValueError:
        return False


def is_probable_wiki_domain(domain: str, *, url: str = "") -> bool:
    """Return conservative candidates for a later MediaWiki capability probe."""
    host = normalize_wiki_host(domain)
    if host is None or not is_safe_wiki_host(host):
        return False
    if has_strong_wiki_page_signal(host, url=url):
        return True
    labels = host.split(".")
    # A host ending in "wiki" is a discovery hint only. It must pass a
    # MediaWiki capability probe before the domain is persisted or trusted.
    if any(label.endswith("wiki") for label in labels):
        return True

    return False


def has_strong_wiki_page_signal(domain: str, *, url: str = "") -> bool:
    """Identify URL evidence strong enough to classify an individual result."""
    host = normalize_wiki_host(domain)
    if host is None or not is_safe_wiki_host(host):
        return False
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in HOSTED_WIKI_SUFFIXES):
        return True
    parsed = urlparse(url)
    if parsed.hostname:
        try:
            url_host = parsed.hostname.encode("idna").decode("ascii").casefold().rstrip(".")
        except UnicodeError:
            return False
        if url_host != host:
            return False
    path = parsed.path.casefold().rstrip("/")
    # Content paths and MediaWiki's conventional /w/index.php endpoint are
    # useful signals. A root /api.php or /index.php is common to many unrelated
    # applications and must be capability-probed before it earns wiki status.
    return parsed.path.casefold().startswith("/wiki/") or path == "/w/index.php"
