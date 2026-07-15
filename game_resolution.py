import re
from typing import Any, Protocol
from urllib.parse import urlparse

from query_tokens import relevance_tokens
from quality_policy import (
    FAST_GAME_IDENTITY_MAX_RESULTS,
    GAME_IDENTITY_CANDIDATE_QUERIES,
    GAME_IDENTITY_DATABASE_QUERIES,
    GAME_RESOLUTION_POLICY,
)
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
        fast_resolution = self.discover_game_identity(game=game)
        if fast_resolution.is_confirmed:
            if not fast_resolution.database_domains:
                database_domains = list(self.discover_database_identity(game=game))
                if database_domains:
                    confidence = self.resolution_confidence(
                        game=game,
                        aliases=fast_resolution.aliases,
                        platform_urls=[str(url) for url in fast_resolution.platform_urls],
                        database_domains=database_domains,
                    )
                    return fast_resolution.model_copy(
                        update={"database_domains": database_domains, "confidence": confidence}
                    )
            return fast_resolution

        candidates = list(self.discover_game_candidates(game=game))
        aliases: list[str] = []
        for candidate in candidates:
            for alias in [candidate.name, *candidate.aliases]:
                if alias.lower() != game.lower() and alias not in aliases:
                    aliases.append(alias)

        database_domains = list(self.discover_database_domains(game=game, game_aliases=aliases))
        platform_urls = list(
            dict.fromkeys(
                str(url)
                for candidate in candidates
                for url in candidate.platform_urls
            )
        )
        confirmed_name = candidates[0].name if candidates else (aliases[0] if aliases else game)
        candidates_by_key: dict[str, GameCandidate] = {}
        for candidate in candidates:
            key = candidate_key(name=candidate.name, url=str(candidate.platform_urls[0]))
            existing = candidates_by_key.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                candidates_by_key[key] = candidate
        candidates = sorted(
            candidates_by_key.values(),
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )

        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            database_domains=database_domains,
        )
        ambiguous = (
            len(candidates) > 1
            and candidates[0].confidence - candidates[1].confidence
            < GAME_RESOLUTION_POLICY.ambiguity_margin
        )
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

    def discover_game_identity(self, *, game: str) -> GameResolution:
        """Confirm an exact store/wiki identity with one broad search request."""
        result = self._client.search(
            query=f'"{game}" game Steam itch.io GOG wiki',
            max_results=FAST_GAME_IDENTITY_MAX_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )
        aliases: list[str] = []
        platform_urls: list[str] = []
        database_domains: list[str] = []
        candidates: list[GameCandidate] = []

        for item in result.get("results", []):
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            domain = urlparse(url).netloc.lower()
            if not identity_matches_game(title=title, url=url, game=game):
                continue

            if self.is_supported_database_domain(domain) and domain not in database_domains:
                database_domains.append(domain)

            if is_supported_platform_domain(domain):
                # Article titles describe mechanics, characters, or items and
                # must never become aliases for the game itself. Only store
                # identity pages are authoritative enough to supply aliases.
                item_aliases = list(title_alias_candidates(title))
                if is_low_value_game_candidate(title=title, url=url):
                    continue
                if url not in platform_urls:
                    platform_urls.append(url)
                name = item_aliases[0] if item_aliases else game
                raw_score = float(item.get("score") or 0.5)
                candidates.append(
                    GameCandidate(
                        name=name,
                        aliases=[alias for alias in item_aliases if alias != name][:6],
                        tags=infer_game_tags(self.result_text(item)),
                        platform_urls=[url],
                        confidence=min(
                            1.0,
                            GAME_RESOLUTION_POLICY.candidate_base
                            + raw_score * GAME_RESOLUTION_POLICY.candidate_search_weight
                            + (GAME_RESOLUTION_POLICY.candidate_alias_bonus if item_aliases else 0),
                        ),
                    )
                )

        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            database_domains=database_domains,
        )
        if candidates:
            # Every candidate in the fast path has already passed an exact
            # identity check for the requested game. Store mirrors and
            # marketing-title variants therefore describe one game, not
            # multiple ambiguous games.
            primary = min(candidates, key=lambda candidate: len(candidate.name))
            exact_name = next(
                (candidate.name for candidate in candidates if candidate.name.casefold() == game.casefold()),
                primary.name,
            )
            merged_aliases = list(
                dict.fromkeys(
                    alias
                    for candidate in candidates
                    for alias in [candidate.name, *candidate.aliases]
                    if alias.casefold() != exact_name.casefold()
                )
            )
            candidates = [
                GameCandidate(
                    name=exact_name,
                    aliases=merged_aliases[:6],
                    tags=list(dict.fromkeys(tag for candidate in candidates for tag in candidate.tags))[:5],
                    platform_urls=platform_urls[:6],
                    confidence=max(candidate.confidence for candidate in candidates),
                )
            ]
        confirmed_name = candidates[0].name if candidates else game
        ambiguous = (
            len(candidates) > 1
            and candidates[0].confidence - candidates[1].confidence
            < GAME_RESOLUTION_POLICY.ambiguity_margin
        )
        return GameResolution(
            input_name=game,
            confirmed_name=confirmed_name,
            aliases=aliases[:8],
            platform_urls=platform_urls[:6],
            database_domains=database_domains[:8],
            candidates=candidates[:6],
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
        for query in queries[:GAME_IDENTITY_DATABASE_QUERIES]:
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

    def discover_database_identity(self, *, game: str) -> tuple[str, ...]:
        """Find a game-specific wiki with one exact identity query."""
        result = self._client.search(
            query=f'"{game}" wiki fandom wiki.gg',
            max_results=4,
            include_answer=False,
            include_raw_content=False,
        )
        domains: list[str] = []
        for item in result.get("results", []):
            url = str(item.get("url") or "")
            domain = urlparse(url).netloc.lower()
            if not self.is_supported_database_domain(domain):
                continue
            if not identity_matches_game(
                title=str(item.get("title") or ""),
                url=url,
                game=game,
            ):
                continue
            if domain not in domains:
                domains.append(domain)
        return tuple(domains[:4])

    def discover_game_aliases(self, *, game: str) -> tuple[str, ...]:
        queries = (
            f"{game} Steam",
            f"{game} Steam official",
            f"{game} itch.io",
            f"{game} GOG",
            f"{game} game English title",
        )
        aliases: list[str] = []
        for query in queries[:GAME_IDENTITY_CANDIDATE_QUERIES]:
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
        for query in queries[:GAME_IDENTITY_CANDIDATE_QUERIES]:
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
        for query in queries[:GAME_IDENTITY_CANDIDATE_QUERIES]:
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
                confidence = min(
                    1.0,
                    GAME_RESOLUTION_POLICY.candidate_base
                    + raw_score * GAME_RESOLUTION_POLICY.candidate_search_weight
                    + (GAME_RESOLUTION_POLICY.candidate_alias_bonus if aliases else 0)
                    + (GAME_RESOLUTION_POLICY.candidate_tag_bonus if tags else 0),
                )
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
        score = GAME_RESOLUTION_POLICY.base_confidence
        if aliases:
            score += GAME_RESOLUTION_POLICY.alias_bonus
        if platform_urls:
            score += GAME_RESOLUTION_POLICY.platform_bonus
        if database_domains:
            score += GAME_RESOLUTION_POLICY.database_bonus
        if not relevance_tokens(game):
            score -= GAME_RESOLUTION_POLICY.invalid_name_penalty
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
        "games like",
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


def identity_matches_game(*, title: str, url: str, game: str) -> bool:
    compact_game = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", game.lower())
    compact_surface = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", f"{title} {url}".lower())
    return len(compact_game) >= 3 and compact_game in compact_surface
