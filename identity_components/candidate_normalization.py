"""Pure normalization rules for search-result game candidates."""

import re
from urllib.parse import urlparse

from identity_components.selection import domain_matches, is_platform_product_url, platform_resource_identity
from query_tokens import relevance_tokens


def compact_identity_text(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def infer_game_tags(text: str) -> list[str]:
    tag_rules = (
        ("RPG", ("rpg", "role-playing", "角色扮演")), ("生存", ("survival", "生存")),
        ("恐怖", ("horror", "恐怖")), ("解谜", ("puzzle", "解谜", "谜题")),
        ("冒险", ("adventure", "冒险")), ("动作", ("action", "动作")),
        ("模拟", ("simulation", "simulator", "模拟")), ("策略", ("strategy", "策略")),
        ("视觉小说", ("visual novel", "视觉小说")), ("独立游戏", ("indie", "独立")),
    )
    lowered = text.casefold()
    return [tag for tag, keywords in tag_rules if any(keyword in lowered for keyword in keywords)][:5]


def is_low_value_game_candidate(*, title: str, url: str) -> bool:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.casefold().split("/") if part]
    if domain_matches(parsed.netloc, "store.steampowered.com"):
        return len(parts) < 2 or parts[0] != "app"
    return " ".join(title.casefold().split()).strip(" -|:：") in {
        "all games", "games", "所有游戏", "全部游戏", "game collection", "游戏合集",
    }


def candidate_key(*, name: str, url: str) -> str:
    parsed = urlparse(url)
    if is_platform_product_url(url):
        family, resource = platform_resource_identity(url)
        return f"{family}:{resource}"
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    path = parsed.path.rstrip("/").casefold()
    return f"url:{host}{path}" if host and path else compact_identity_text(name) or url


def choose_canonical_candidate_name(*, game: str, candidates: list[str] | tuple[str, ...], fallback: str) -> str:
    cleaned = list(dict.fromkeys(value.strip() for value in candidates if value.strip()))
    if not cleaned:
        return fallback
    game_key = compact_identity_text(game)
    exact = next((value for value in cleaned if compact_identity_text(value) == game_key), None)
    if exact:
        return exact
    game_tokens = set(relevance_tokens(game))

    def score(value: str) -> tuple[float, int, int]:
        value_tokens = set(relevance_tokens(value))
        return (
            len(game_tokens.intersection(value_tokens)) / max(len(game_tokens), 1),
            int(bool(game_key) and game_key in compact_identity_text(value)), -len(value),
        )
    return max(cleaned, key=score)


def useful_candidate_aliases(*, name: str, candidates: list[str] | tuple[str, ...]) -> list[str]:
    markers = (
        "download", "buy today", "official website", "official site", "games store", "app store",
        "google play", "playstation store", "xbox store", "epic games", "on steam", "在 steam 上",
        "立即购买", "官方网站",
    )
    aliases: list[str] = []
    name_key = compact_identity_text(name)
    for value in candidates:
        normalized = " ".join(value.split()).strip(" -|:：")
        if not normalized or compact_identity_text(normalized) == name_key or any(marker in normalized.casefold() for marker in markers):
            continue
        if normalized not in aliases:
            aliases.append(normalized)
    return aliases[:6]


def title_alias_candidates(title: str, *, url: str = "") -> tuple[str, ...]:
    cleaned = title.strip()
    for marker in (" on Steam", " on GOG.com", " on GOG", " on itch.io", " Steam", "在 Steam 上", "Steam 上的", " - Steam", " | Steam", " - GOG.com", " | GOG.com", " - itch.io", " | itch.io"):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"^在\s*steam\s*上购买", "", cleaned, flags=re.I)
    cleaned = re.sub(r"立省\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*-\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*所有游戏.*$", "", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" -|:：")
    candidates = [cleaned] if 3 <= len(cleaned) <= 80 else []
    if domain_matches(urlparse(url).netloc, "itch.io") and " by " in cleaned.casefold():
        candidates.insert(0, re.split(r"\s+by\s+", cleaned, maxsplit=1, flags=re.I)[0].strip())
    candidates.extend(
        part.strip(" -|:：") for part in re.split(r"\s[-|:：]\s|[（）()【】\[\]]", cleaned)
        if any(char.isascii() and char.isalpha() for char in part) and 3 <= len(part.strip()) <= 80
    )
    candidates.extend(
        part.strip() for part in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'_.-]*)+", cleaned)
        if 3 <= len(part.strip()) <= 80
    )
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))
