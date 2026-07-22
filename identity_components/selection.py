"""Opaque identity-URL validation and candidate selection."""

import re
from typing import Any
from urllib.parse import parse_qs, urlparse


def domain_matches(domain: str, candidate: str) -> bool:
    host = domain.casefold().split(":", 1)[0].strip(".")
    candidate = candidate.casefold().strip(".")
    return host == candidate or host.endswith(f".{candidate}")


def is_platform_product_url(url: str) -> bool:
    """Accept product identities, never storefront search/tag/catalog pages."""
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    parts = [part for part in parsed.path.casefold().split("/") if part]
    if domain_matches(host, "store.steampowered.com"):
        return len(parts) >= 2 and parts[0] == "app" and parts[1].isdigit()
    if domain_matches(host, "itch.io"):
        return host != "itch.io" and bool(parts) and parts[0] not in {"games", "jam", "jams"}
    if domain_matches(host, "gog.com"):
        return "game" in parts and parts.index("game") + 1 < len(parts)
    if domain_matches(host, "epicgames.com"):
        return "p" in parts and parts.index("p") + 1 < len(parts)
    if domain_matches(host, "playstation.com"):
        return "product" in parts and parts.index("product") + 1 < len(parts)
    if domain_matches(host, "nintendo.com"):
        return "products" in parts and parts.index("products") + 1 < len(parts)
    if domain_matches(host, "xbox.com"):
        return "store" in parts and len(parts) >= parts.index("store") + 3
    if host == "apps.apple.com" or host.endswith(".apps.apple.com"):
        return any(re.fullmatch(r"id\d+", part) for part in parts)
    if host == "play.google.com" or host.endswith(".play.google.com"):
        return parsed.path.casefold().rstrip("/") == "/store/apps/details" and bool(
            re.search(r"(?:^|&)id=[^&]+", parsed.query)
        )
    return False


def platform_resource_identity(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    families = (
        "steampowered.com", "steamcommunity.com", "itch.io", "gog.com", "epicgames.com",
        "playstation.com", "nintendo.com", "xbox.com", "apps.apple.com", "play.google.com",
    )
    family = next((value for value in families if domain_matches(host, value)), host)
    parts = [part.casefold() for part in parsed.path.split("/") if part]
    resource = f"{host}{parsed.path.rstrip('/').casefold()}"
    if domain_matches(host, "store.steampowered.com") and len(parts) >= 2 and parts[0] == "app":
        resource = f"app:{parts[1]}"
    elif domain_matches(host, "gog.com") and "game" in parts:
        resource = f"game:{parts[parts.index('game') + 1]}"
    elif domain_matches(host, "epicgames.com") and "p" in parts:
        resource = f"product:{parts[parts.index('p') + 1]}"
    elif domain_matches(host, "playstation.com") and "product" in parts:
        resource = f"product:{parts[parts.index('product') + 1]}"
    elif domain_matches(host, "nintendo.com") and "products" in parts:
        resource = f"product:{parts[parts.index('products') + 1]}"
    elif domain_matches(host, "xbox.com") and "store" in parts:
        resource = f"product:{parts[-1]}"
    elif host == "apps.apple.com" or host.endswith(".apps.apple.com"):
        resource = f"app:{next((part for part in parts if re.fullmatch(r'id\d+', part)), '')}"
    elif domain_matches(host, "play.google.com"):
        resource = f"app:{(parse_qs(parsed.query).get('id') or [''])[0].casefold()}"
    return family, resource


def canonical_identity_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    return f"https://{host}{parsed.path.rstrip('/').casefold()}" if host else ""


def is_candidate_identity_url(url: str) -> bool:
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https" and parsed.hostname is not None
        and parsed.username is None and parsed.password is None and port in {None, 443}
    )


def same_platform_resource(left_url: str, right_url: str) -> bool:
    return (
        is_platform_product_url(left_url)
        and is_platform_product_url(right_url)
        and platform_resource_identity(left_url) == platform_resource_identity(right_url)
    )


def select_game_candidate(
    resolution: Any, *, selected_url: str, confirmed_threshold: float,
) -> Any | None:
    """Resolve an opaque UI choice against fresh server-discovered candidates."""
    selected = next(
        (
            candidate for candidate in resolution.candidates
            if any(same_platform_resource(str(url), selected_url) for url in candidate.platform_urls)
            or any(is_candidate_identity_url(selected_url) and canonical_identity_url(str(url)) == canonical_identity_url(selected_url) for url in candidate.official_urls)
            or any(is_candidate_identity_url(selected_url) and canonical_identity_url(str(url)) == canonical_identity_url(selected_url) for url in candidate.identity_urls)
        ),
        None,
    )
    if selected is None:
        return None
    return resolution.model_copy(update={
        "confirmed_name": selected.name,
        "aliases": selected.aliases,
        "platform_urls": selected.platform_urls,
        "official_urls": selected.official_urls,
        "identity_urls": selected.identity_urls,
        "database_domains": selected.database_domains,
        "confidence": max(selected.confidence, confirmed_threshold),
        "ambiguous": False,
    })


def resolution_matches_selected_url(resolution: Any, *, selected_url: str) -> bool:
    if not is_candidate_identity_url(selected_url):
        return False
    if any(same_platform_resource(str(url), selected_url) for url in resolution.platform_urls):
        return True
    selected_key = canonical_identity_url(selected_url)
    return any(canonical_identity_url(str(url)) == selected_key for url in [*resolution.official_urls, *resolution.identity_urls])
