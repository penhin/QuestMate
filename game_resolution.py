import re
from typing import Any, Protocol
from urllib.parse import urlparse

from query_tokens import relevance_tokens
from schemas import GameCandidate, GameResolution


class SearchClient(Protocol):
    def search(self, **kwargs: Any) -> dict[str, Any]:
        ...


def split_title(value: str) -> list[str]:
    return re.split(r"\s[-|:：]\s|[（）()【】\[\]]", value)


class GameResolver:
    def __init__(self, client: SearchClient) -> None:
        self._client = client

    def resolve(self, *, game: str, question: str | None = None) -> GameResolution:
        candidates = list(self.discover_game_candidates(game=game))
        aliases = list(self.discover_game_aliases(game=game))
        for candidate in candidates:
            for alias in [candidate.name, *candidate.aliases]:
                if alias.lower() != game.lower() and alias not in aliases:
                    aliases.append(alias)

        database_domains = list(self.discover_database_domains(game=game, game_aliases=aliases))
        platform_urls = list(self.discover_platform_urls(game=game, game_aliases=aliases))
        confirmed_name = candidates[0].name if candidates else (aliases[0] if aliases else game)
        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            database_domains=database_domains,
        )
        ambiguous = len(candidates) > 1 and candidates[0].confidence - candidates[1].confidence < 0.2
        return GameResolution(
            input_name=game,
            confirmed_name=confirmed_name,
            aliases=aliases,
            platform_urls=platform_urls,
            database_domains=database_domains,
            candidates=candidates,
            confidence=confidence,
            ambiguous=ambiguous,
        )

    def discover_database_domains(self, *, game: str, game_aliases: list[str] | None = None) -> tuple[str, ...]:
        game_names = [game, *(game_aliases or [])]
        queries = tuple(
            f"{game_name} {suffix}"
            for game_name in game_names[:3]
            for suffix in ("fandom wiki", "wiki.gg wiki", "official wiki")
        )
        domains: list[str] = []
        for query in queries:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "")
                domain = urlparse(url).netloc.lower()
                if not self.is_supported_database_domain(domain):
                    continue
                text = self.result_text(item)
                if not matches_game_text(text=text, game=game, game_aliases=game_aliases or []):
                    continue
                if domain not in domains:
                    domains.append(domain)
                if len(domains) >= 4:
                    return tuple(domains)
        return tuple(domains)

    def discover_game_aliases(self, *, game: str) -> tuple[str, ...]:
        queries = (
            f"{game} Steam",
            f"{game} Steam official",
            f"{game} itch.io",
            f"{game} GOG",
            f"{game} game English title",
        )
        aliases: list[str] = []
        for query in queries:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                if not matches_game_text(text=self.result_text(item), game=game, game_aliases=[]):
                    continue
                for alias in title_alias_candidates(str(item.get("title") or "")):
                    if alias.lower() != game.lower() and alias not in aliases:
                        aliases.append(alias)
                    if len(aliases) >= 4:
                        return tuple(aliases)
        return tuple(aliases)

    def discover_platform_urls(self, *, game: str, game_aliases: list[str] | None = None) -> tuple[str, ...]:
        game_names = [game, *(game_aliases or [])]
        queries = tuple(
            f"{game_name} {platform}"
            for game_name in game_names[:3]
            for platform in ("Steam", "itch.io", "GOG", "Epic Games")
        )
        urls: list[str] = []
        for query in queries:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "")
                domain = urlparse(url).netloc.lower()
                if not is_supported_platform_domain(domain):
                    continue
                if not matches_game_text(text=self.result_text(item), game=game, game_aliases=game_aliases or []):
                    continue
                if url not in urls:
                    urls.append(url)
                if len(urls) >= 6:
                    return tuple(urls)
        return tuple(urls)

    def discover_game_candidates(self, *, game: str) -> tuple[GameCandidate, ...]:
        queries = (
            f"{game} Steam",
            f"{game} itch.io game",
            f"{game} GOG game",
            f"{game} Epic Games",
        )
        candidates_by_key: dict[str, GameCandidate] = {}
        for query in queries:
            result = self._client.search(
                query=query,
                max_results=4,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "")
                domain = urlparse(url).netloc.lower()
                if not is_supported_platform_domain(domain):
                    continue
                text = self.result_text(item)
                if not matches_game_text(text=text, game=game, game_aliases=[]):
                    continue
                title = str(item.get("title") or url)
                if is_low_value_game_candidate(title=title, url=url):
                    continue
                aliases = list(title_alias_candidates(title))
                name = aliases[0] if aliases else title[:80]
                canonical_key = candidate_key(name=name, url=url)
                tags = infer_game_tags(text)
                raw_score = float(item.get("score") or 0.5)
                confidence = min(1.0, 0.45 + raw_score * 0.35 + (0.15 if aliases else 0) + (0.05 if tags else 0))
                existing = candidates_by_key.get(canonical_key)
                platform_urls = [url]
                if existing:
                    platform_urls = list(dict.fromkeys([*map(str, existing.platform_urls), url]))
                    confidence = max(existing.confidence, confidence)
                    tags = list(dict.fromkeys([*existing.tags, *tags]))[:5]
                    aliases = list(dict.fromkeys([*existing.aliases, *aliases]))[:6]
                    name = existing.name if len(existing.name) <= len(name) else name
                candidates_by_key[canonical_key] = GameCandidate(
                    name=name,
                    aliases=[alias for alias in aliases if alias != name][:6],
                    tags=tags,
                    platform_urls=platform_urls[:4],
                    confidence=confidence,
                )
        return tuple(sorted(candidates_by_key.values(), key=lambda candidate: candidate.confidence, reverse=True)[:6])

    @staticmethod
    def result_text(item: dict[str, Any]) -> str:
        return " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).lower()

    @staticmethod
    def is_supported_database_domain(domain: str) -> bool:
        return (
            domain.endswith(".fandom.com")
            or domain == "fandom.com"
            or domain.endswith(".wiki.gg")
            or domain == "wiki.gg"
        )

    @staticmethod
    def resolution_confidence(
        *,
        game: str,
        aliases: list[str],
        platform_urls: list[str],
        database_domains: list[str],
    ) -> float:
        score = 0.25
        if aliases:
            score += 0.25
        if platform_urls:
            score += 0.35
        if database_domains:
            score += 0.25
        if not relevance_tokens(game):
            score -= 0.2
        return max(0, min(1, score))


def infer_game_tags(text: str) -> list[str]:
    tag_rules = (
        ("RPG", ("rpg", "role-playing", "角色扮演")),
        ("生存", ("survival", "生存")),
        ("恐怖", ("horror", "恐怖")),
        ("解谜", ("puzzle", "解谜", "谜题")),
        ("冒险", ("adventure", "冒险")),
        ("动作", ("action", "动作")),
        ("模拟", ("simulation", "simulator", "模拟")),
        ("策略", ("strategy", "策略")),
        ("视觉小说", ("visual novel", "视觉小说")),
        ("独立游戏", ("indie", "独立")),
    )
    tags: list[str] = []
    for tag, keywords in tag_rules:
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags[:5]


def is_low_value_game_candidate(*, title: str, url: str) -> bool:
    lowered = f"{title} {url}".lower()
    low_value_patterns = (
        "所有游戏",
        "全部游戏",
        "购买",
        "立省",
        "省",
        "折扣",
        "特惠",
        "sale",
        "bundle",
        "合集",
        "collection",
        "steam 上购买",
        "on sale",
    )
    return any(pattern in lowered for pattern in low_value_patterns)


def candidate_key(*, name: str, url: str) -> str:
    parsed = urlparse(url)
    if "steampowered.com" in parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "app":
            return f"steam:{parts[1]}"
    if "gog.com" in parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if parts:
            return f"gog:{parts[-1]}"
    if "itch.io" in parsed.netloc:
        return f"itch:{parsed.netloc}{parsed.path.rstrip('/')}"
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", name.lower())
    return normalized or url


def is_supported_platform_domain(domain: str) -> bool:
    return any(
        value in domain
        for value in (
            "store.steampowered.com",
            "steamcommunity.com",
            "itch.io",
            "gog.com",
            "epicgames.com",
        )
    )


def title_alias_candidates(title: str) -> tuple[str, ...]:
    cleaned = title.strip()
    for marker in (" on Steam", " Steam", "在 Steam 上", "Steam 上的", " - Steam", " | Steam"):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"^在\s*steam\s*上购买", "", cleaned, flags=re.I)
    cleaned = re.sub(r"立省\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*-\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*所有游戏.*$", "", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" -|:：")
    candidates = [cleaned] if 3 <= len(cleaned) <= 80 else []
    ascii_parts = [
        part.strip(" -|:：")
        for part in split_title(cleaned)
        if any(char.isascii() and char.isalpha() for char in part) and 3 <= len(part.strip()) <= 80
    ]
    candidates.extend(ascii_parts)
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def matches_game_text(*, text: str, game: str, game_aliases: list[str]) -> bool:
    game_tokens = relevance_tokens(" ".join([game, *game_aliases]))
    return not game_tokens or any(token in text for token in game_tokens)
