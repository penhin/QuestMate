import re
from difflib import SequenceMatcher
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from query_tokens import relevance_tokens
from retrieval.wiki_domains import is_probable_wiki_domain
from retrieval.source_quality import matches_game_identity
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
                        official_urls=[str(url) for url in fast_resolution.official_urls],
                        identity_urls=[str(url) for url in fast_resolution.identity_urls],
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
        official_urls = list(dict.fromkeys(
            str(url)
            for candidate in candidates
            for url in candidate.official_urls
        ))
        identity_urls = list(dict.fromkeys(
            str(url)
            for candidate in candidates
            for url in candidate.identity_urls
        ))
        confirmed_name = candidates[0].name if candidates else (aliases[0] if aliases else game)
        candidates = list(merge_cross_platform_candidates(candidates))
        if candidates:
            primary = candidates[0]
            confirmed_name = primary.name
            aliases = list(dict.fromkeys([
                *([primary.name] if primary.name.casefold() != game.casefold() else []),
                *primary.aliases,
            ]))[:8]
            platform_urls = [str(url) for url in primary.platform_urls]
            official_urls = [str(url) for url in primary.official_urls]
            identity_urls = [str(url) for url in primary.identity_urls]

        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            official_urls=official_urls,
            identity_urls=identity_urls,
            database_domains=database_domains,
        )
        ambiguous = (
            len(candidates) > 1
            and candidates[0].confidence - candidates[1].confidence
            < GAME_RESOLUTION_POLICY.ambiguity_margin
        ) or self._candidate_requires_confirmation(game=game, candidates=candidates)
        return GameResolution(
            input_name=game,
            confirmed_name=confirmed_name,
            aliases=aliases,
            platform_urls=platform_urls,
            official_urls=official_urls,
            identity_urls=identity_urls,
            database_domains=database_domains,
            candidates=candidates,
            confidence=confidence,
            ambiguous=ambiguous,
        )

    def discover_game_identity(self, *, game: str) -> GameResolution:
        """Confirm an exact store/wiki identity with one broad search request."""
        result = self._client.search(
            query=f'"{game}" video game official store wiki',
            max_results=FAST_GAME_IDENTITY_MAX_RESULTS,
            include_answer=False,
            include_raw_content=False,
        )
        aliases: list[str] = []
        platform_urls: list[str] = []
        official_urls: list[str] = []
        identity_urls: list[str] = []
        database_domains: list[str] = []
        candidates: list[GameCandidate] = []

        for item in result.get("results", []):
            title = str(item.get("title") or "")
            url = str(item.get("url") or "")
            domain = urlparse(url).netloc.lower()
            if not matches_game_text(text=self.result_text(item), game=game, game_aliases=[]):
                continue

            if self.is_supported_database_domain(domain, url=url) and domain not in database_domains:
                database_domains.append(domain)

            if is_supported_platform_domain(domain) and is_platform_product_url(url):
                # Article titles describe mechanics, characters, or items and
                # must never become aliases for the game itself. Only store
                # identity pages are authoritative enough to supply aliases.
                item_aliases = list(title_alias_candidates(title, url=url))
                if is_low_value_game_candidate(title=title, url=url):
                    continue
                if url not in platform_urls:
                    platform_urls.append(url)
                name = choose_canonical_candidate_name(game=game, candidates=item_aliases, fallback=game)
                useful_aliases = useful_candidate_aliases(name=name, candidates=item_aliases)
                raw_score = float(item.get("score") or 0.5)
                candidates.append(
                    GameCandidate(
                        name=name,
                        aliases=useful_aliases,
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
            elif is_generic_official_identity_result(item=item, game=game):
                item_aliases = list(title_alias_candidates(title, url=url))
                name = choose_canonical_candidate_name(game=game, candidates=item_aliases, fallback=game)
                useful_aliases = useful_candidate_aliases(name=name, candidates=item_aliases)
                if url not in identity_urls:
                    identity_urls.append(url)
                raw_score = float(item.get("score") or 0.5)
                candidates.append(
                    GameCandidate(
                        name=name,
                        aliases=useful_aliases,
                        tags=infer_game_tags(self.result_text(item)),
                        identity_urls=[url],
                        confidence=min(
                            1.0,
                            GAME_RESOLUTION_POLICY.candidate_base
                            + raw_score * GAME_RESOLUTION_POLICY.candidate_search_weight,
                        ),
                    )
                )

        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            official_urls=official_urls,
            identity_urls=identity_urls,
            database_domains=database_domains,
        )
        if candidates:
            candidates = list(merge_cross_platform_candidates(candidates))
            primary = candidates[0]
            exact_name = next(
                (candidate.name for candidate in candidates if candidate.name.casefold() == game.casefold()),
                primary.name,
            )
            primary_aliases = list(dict.fromkeys([
                *([primary.name] if primary.name.casefold() != game.casefold() else []),
                *primary.aliases,
            ]))
            aliases = primary_aliases[:8]
            platform_urls = [str(url) for url in primary.platform_urls]
            official_urls = [str(url) for url in primary.official_urls]
            identity_urls = [str(url) for url in primary.identity_urls]
        confirmed_name = exact_name if candidates else game
        confidence = self.resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            official_urls=official_urls,
            identity_urls=identity_urls,
            database_domains=database_domains,
        )
        ambiguous = (
            len(candidates) > 1
            and candidates[0].confidence - candidates[1].confidence
            < GAME_RESOLUTION_POLICY.ambiguity_margin
        ) or self._candidate_requires_confirmation(game=game, candidates=candidates)
        return GameResolution(
            input_name=game,
            confirmed_name=confirmed_name,
            aliases=aliases[:8],
            platform_urls=platform_urls[:6],
            official_urls=official_urls[:4],
            identity_urls=identity_urls[:4],
            database_domains=database_domains[:8],
            candidates=candidates[:6],
            confidence=confidence,
            ambiguous=ambiguous,
        )

    def discover_database_domains(self, *, game: str, game_aliases: list[str] | None = None) -> tuple[str, ...]:
        game_names = [game, *(game_aliases or [])]
        queries = tuple(
            f"{game_name} {suffix}"
            for suffix in ("wiki", "official wiki", "community database", "walkthrough wiki")
            for game_name in game_names[:3]
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
                if not self.is_supported_database_domain(domain, url=url):
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
            query=f'"{game}" wiki',
            max_results=4,
            include_answer=False,
            include_raw_content=False,
        )
        domains: list[str] = []
        for item in result.get("results", []):
            url = str(item.get("url") or "")
            domain = urlparse(url).netloc.lower()
            if not self.is_supported_database_domain(domain, url=url):
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

    def discover_game_candidates(self, *, game: str) -> tuple[GameCandidate, ...]:
        queries = (
            f'"{game}" video game official store',
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
                is_platform = is_supported_platform_domain(domain) and is_platform_product_url(url)
                is_official = is_generic_official_identity_result(item=item, game=game)
                if not is_platform and not is_official:
                    continue
                text = self.result_text(item)
                title = str(item.get("title") or url)
                if not matches_game_text(text=text, game=game, game_aliases=[]) and not is_near_game_identity(
                    title=title, url=url, game=game
                ):
                    continue
                if is_low_value_game_candidate(title=title, url=url):
                    continue
                aliases = list(title_alias_candidates(title, url=url))
                name = choose_canonical_candidate_name(game=game, candidates=aliases, fallback=title[:80])
                aliases = useful_candidate_aliases(name=name, candidates=aliases)
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
                platform_urls = [url] if is_platform else []
                identity_urls = [url] if is_official else []
                # Search-result copy may nominate a generic official site as
                # an identity candidate, but only a separately verified
                # authority is allowed into official_urls.
                official_urls = []
                if existing:
                    platform_urls = list(dict.fromkeys([*map(str, existing.platform_urls), url]))
                    if not is_platform:
                        platform_urls = [str(value) for value in existing.platform_urls]
                    official_urls = list(dict.fromkeys([
                        *map(str, existing.official_urls),
                    ]))
                    identity_urls = list(dict.fromkeys([
                        *map(str, existing.identity_urls),
                        *(identity_urls if is_official else []),
                    ]))
                    confidence = max(existing.confidence, confidence)
                    tags = list(dict.fromkeys([*existing.tags, *tags]))[:5]
                    aliases = list(dict.fromkeys([*existing.aliases, *aliases]))[:6]
                    name = existing.name if len(existing.name) <= len(name) else name
                candidates_by_key[canonical_key] = GameCandidate(
                    name=name,
                    aliases=[alias for alias in aliases if alias != name][:6],
                    tags=tags,
                    platform_urls=platform_urls[:4],
                    official_urls=official_urls[:4],
                    identity_urls=identity_urls[:4],
                    confidence=confidence,
                )
        return merge_cross_platform_candidates(list(candidates_by_key.values()))[:6]

    @staticmethod
    def _candidate_requires_confirmation(*, game: str, candidates: list[GameCandidate]) -> bool:
        """Require a choice for fuzzy titles and distinct same-name products.

        Search ranking is relevance, not identity authority.  Even a large
        score gap must not silently choose between two distinct product IDs
        that both match the title the user entered.
        """
        if not candidates:
            return False

        def matches_input(candidate: GameCandidate) -> bool:
            return any(
                identity_names_equivalent(game, name)
                for name in (candidate.name, *candidate.aliases)
            )

        if not matches_input(candidates[0]):
            return True
        return sum(matches_input(candidate) for candidate in candidates) > 1

    @staticmethod
    def result_text(item: dict[str, Any]) -> str:
        return " ".join(str(item.get(field) or "") for field in ("title", "url", "content"))

    @staticmethod
    def is_supported_database_domain(domain: str, *, url: str = "") -> bool:
        """Identify candidates broadly; MediaWikiClient performs the capability check."""
        return is_probable_wiki_domain(domain, url=url)

    @staticmethod
    def resolution_confidence(
        *,
        game: str,
        aliases: list[str],
        platform_urls: list[str],
        official_urls: list[str],
        identity_urls: list[str],
        database_domains: list[str],
    ) -> float:
        score = GAME_RESOLUTION_POLICY.base_confidence
        if aliases:
            score += GAME_RESOLUTION_POLICY.alias_bonus
        if platform_urls:
            score += GAME_RESOLUTION_POLICY.platform_bonus
        if official_urls:
            score += GAME_RESOLUTION_POLICY.official_bonus
        if identity_urls:
            score += GAME_RESOLUTION_POLICY.identity_candidate_bonus
        if database_domains:
            score += GAME_RESOLUTION_POLICY.database_bonus
        # A page calling itself official is useful as a selectable candidate,
        # but not enough to silently confirm an identity without an independent
        # store or database signal.
        if identity_urls and not platform_urls and not official_urls:
            score = min(score, GAME_RESOLUTION_POLICY.confirmed_threshold - 0.01)
        if not relevance_tokens(game):
            score -= GAME_RESOLUTION_POLICY.invalid_name_penalty
        return max(0, min(1, score))


def infer_game_tags(text: str) -> list[str]:
    text = text.casefold()
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
    parsed = urlparse(url)
    host = parsed.netloc.casefold()
    path_parts = [part for part in parsed.path.casefold().split("/") if part]
    # Steam catalog, bundle, and package pages are not game identities. An app
    # page remains valid even when its localized title contains sale copy.
    if domain_matches(host, "store.steampowered.com"):
        return len(path_parts) < 2 or path_parts[0] != "app"
    normalized_title = " ".join(title.casefold().split()).strip(" -|:：")
    return normalized_title in {
        "all games",
        "games",
        "所有游戏",
        "全部游戏",
        "game collection",
        "游戏合集",
    }


def candidate_key(*, name: str, url: str) -> str:
    parsed = urlparse(url)
    if is_platform_product_url(url):
        family, resource = _platform_resource_identity(url)
        return f"{family}:{resource}"
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    path = parsed.path.rstrip("/").casefold()
    return f"url:{host}{path}" if host and path else compact_identity_text(name) or url


def merge_cross_platform_candidates(
    candidates: list[GameCandidate],
) -> tuple[GameCandidate, ...]:
    """Merge mirror store pages while preserving genuine same-name ambiguity.

    Exact product URLs are duplicates. Cross-store pages merge only when one
    explicitly supplies the other's canonical name as an alias; a shared title
    alone is not enough because unrelated games often have the same name.
    """
    merged: list[GameCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        match_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _same_cross_platform_identity(existing, candidate)
            ),
            None,
        )
        if match_index is None:
            merged.append(candidate)
            continue
        existing = merged[match_index]
        all_names = list(dict.fromkeys([
            existing.name,
            *existing.aliases,
            candidate.name,
            *candidate.aliases,
        ]))
        preferred_name = min((existing.name, candidate.name), key=len)
        merged[match_index] = existing.model_copy(update={
            "name": preferred_name,
            "aliases": [name for name in all_names if name != preferred_name][:6],
            "tags": list(dict.fromkeys([*existing.tags, *candidate.tags]))[:5],
            "platform_urls": list(dict.fromkeys([
                *map(str, existing.platform_urls),
                *map(str, candidate.platform_urls),
            ]))[:4],
            "official_urls": list(dict.fromkeys([
                *map(str, existing.official_urls),
                *map(str, candidate.official_urls),
            ]))[:4],
            "identity_urls": list(dict.fromkeys([
                *map(str, existing.identity_urls),
                *map(str, candidate.identity_urls),
            ]))[:4],
            "database_domains": list(dict.fromkeys([
                *existing.database_domains,
                *candidate.database_domains,
            ]))[:4],
            "confidence": max(existing.confidence, candidate.confidence),
        })
    return tuple(sorted(merged, key=lambda item: item.confidence, reverse=True))


def _same_cross_platform_identity(left: GameCandidate, right: GameCandidate) -> bool:
    left_resources = {_platform_resource_identity(str(url)) for url in left.platform_urls}
    right_resources = {_platform_resource_identity(str(url)) for url in right.platform_urls}
    if left_resources.intersection(right_resources):
        return True
    left_official = {_canonical_identity_url(str(url)) for url in left.official_urls}
    right_official = {_canonical_identity_url(str(url)) for url in right.official_urls}
    if left_official.intersection(right_official):
        return True
    left_identity = {_canonical_identity_url(str(url)) for url in left.identity_urls}
    right_identity = {_canonical_identity_url(str(url)) for url in right.identity_urls}
    if left_identity.intersection(right_identity):
        return True
    left_families = {family for family, _resource in left_resources}
    right_families = {family for family, _resource in right_resources}
    if left_families.intersection(right_families):
        return False
    left_aliases = {compact_identity_text(alias) for alias in left.aliases}
    right_aliases = {compact_identity_text(alias) for alias in right.aliases}
    return (
        compact_identity_text(left.name) in right_aliases
        or compact_identity_text(right.name) in left_aliases
    )


def _platform_resource_identity(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    families = (
        "steampowered.com",
        "steamcommunity.com",
        "itch.io",
        "gog.com",
        "epicgames.com",
        "playstation.com",
        "nintendo.com",
        "xbox.com",
        "apps.apple.com",
        "play.google.com",
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
        app_id = next((part for part in parts if re.fullmatch(r"id\d+", part)), "")
        resource = f"app:{app_id}"
    elif domain_matches(host, "play.google.com"):
        app_id = (parse_qs(parsed.query).get("id") or [""])[0].casefold()
        resource = f"app:{app_id}"
    return family, resource


def same_platform_resource(left_url: str, right_url: str) -> bool:
    """Compare server-discovered product identities without trusting display names."""
    return (
        is_platform_product_url(left_url)
        and is_platform_product_url(right_url)
        and _platform_resource_identity(left_url) == _platform_resource_identity(right_url)
    )


def _canonical_identity_url(url: str) -> str:
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
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
    )


def select_game_candidate(
    resolution: GameResolution,
    *,
    selected_url: str,
) -> GameResolution | None:
    """Resolve an opaque UI choice against fresh server-discovered candidates."""
    selected = next(
        (
            candidate
            for candidate in resolution.candidates
            if any(same_platform_resource(str(url), selected_url) for url in candidate.platform_urls)
            or any(
                is_candidate_identity_url(selected_url)
                and _canonical_identity_url(str(url)) == _canonical_identity_url(selected_url)
                for url in candidate.official_urls
            )
            or any(
                is_candidate_identity_url(selected_url)
                and _canonical_identity_url(str(url)) == _canonical_identity_url(selected_url)
                for url in candidate.identity_urls
            )
        ),
        None,
    )
    if selected is None:
        return None
    return GameResolution(
        input_name=resolution.input_name,
        confirmed_name=selected.name,
        aliases=selected.aliases,
        platform_urls=selected.platform_urls,
        official_urls=selected.official_urls,
        identity_urls=selected.identity_urls,
        database_domains=selected.database_domains,
        confidence=max(selected.confidence, GAME_RESOLUTION_POLICY.confirmed_threshold),
        ambiguous=False,
    )


def resolution_matches_selected_url(
    resolution: GameResolution,
    *,
    selected_url: str,
) -> bool:
    """Verify that a returned resolution represents the user's opaque choice."""
    if not is_candidate_identity_url(selected_url):
        return False
    if any(same_platform_resource(str(url), selected_url) for url in resolution.platform_urls):
        return True
    selected_key = _canonical_identity_url(selected_url)
    return any(
        _canonical_identity_url(str(url)) == selected_key
        for url in [*resolution.official_urls, *resolution.identity_urls]
    )


def choose_canonical_candidate_name(
    *,
    game: str,
    candidates: list[str] | tuple[str, ...],
    fallback: str,
) -> str:
    """Prefer the title fragment closest to the requested identity over store copy."""
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
        overlap = len(game_tokens.intersection(value_tokens)) / max(len(game_tokens), 1)
        contains = int(bool(game_key) and game_key in compact_identity_text(value))
        return overlap, contains, -len(value)

    return max(cleaned, key=score)


def useful_candidate_aliases(
    *,
    name: str,
    candidates: list[str] | tuple[str, ...],
) -> list[str]:
    """Keep alternate identities while removing storefront presentation copy."""
    marketing_markers = (
        "download",
        "buy today",
        "official website",
        "official site",
        "games store",
        "app store",
        "google play",
        "playstation store",
        "xbox store",
        "epic games",
        "on steam",
        "在 steam 上",
        "立即购买",
        "官方网站",
    )
    aliases: list[str] = []
    name_key = compact_identity_text(name)
    for value in candidates:
        normalized = " ".join(value.split()).strip(" -|:：")
        key = compact_identity_text(normalized)
        lowered = normalized.casefold()
        if not normalized or key == name_key or any(marker in lowered for marker in marketing_markers):
            continue
        if normalized not in aliases:
            aliases.append(normalized)
    return aliases[:6]


def is_generic_official_identity_result(*, item: dict[str, Any], game: str) -> bool:
    """Admit exact official-site identities without requiring a storefront allowlist."""
    title = str(item.get("title") or "")
    url = str(item.get("url") or "")
    parsed = urlparse(url)
    if not is_candidate_identity_url(url):
        return False
    host = parsed.netloc.casefold()
    if is_supported_platform_domain(host) or is_probable_wiki_domain(host, url=url):
        return False
    if any(domain_matches(host, domain) for domain in ("reddit.com", "steamcommunity.com", "youtube.com")):
        return False
    aliases = title_alias_candidates(title, url=url)
    if not any(compact_identity_text(alias) == compact_identity_text(game) for alias in aliases):
        return False
    identity_text = f"{title} {item.get('content') or ''}".casefold()
    normalized_title = title.casefold()
    official_markers = (
        "official site",
        "official website",
        "official game",
        "官方网站",
        "官方站点",
        "公式サイト",
        "공식 사이트",
        "developer",
        "publisher",
    )
    title_markers = (
        "official",
        "官方网站",
        "官方站点",
        "公式サイト",
        "공식 사이트",
    )
    return any(marker in normalized_title for marker in title_markers) and any(
        marker in identity_text for marker in official_markers
    )


def is_supported_platform_domain(domain: str) -> bool:
    return any(
        domain_matches(domain, value)
        for value in (
            "store.steampowered.com",
            "itch.io",
            "gog.com",
            "epicgames.com",
            "playstation.com",
            "nintendo.com",
            "xbox.com",
            "apps.apple.com",
            "play.google.com",
        )
    )


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


def domain_matches(domain: str, candidate: str) -> bool:
    host = domain.casefold().split(":", 1)[0].strip(".")
    candidate = candidate.casefold().strip(".")
    return host == candidate or host.endswith(f".{candidate}")


def title_alias_candidates(title: str, *, url: str = "") -> tuple[str, ...]:
    cleaned = title.strip()
    for marker in (
        " on Steam",
        " on GOG.com",
        " on GOG",
        " on itch.io",
        " Steam",
        "在 Steam 上",
        "Steam 上的",
        " - Steam",
        " | Steam",
        " - GOG.com",
        " | GOG.com",
        " - itch.io",
        " | itch.io",
    ):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r"^在\s*steam\s*上购买", "", cleaned, flags=re.I)
    cleaned = re.sub(r"立省\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*-\s*\d+%.*$", "", cleaned)
    cleaned = re.sub(r"\s*所有游戏.*$", "", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" -|:：")
    candidates = [cleaned] if 3 <= len(cleaned) <= 80 else []
    if domain_matches(urlparse(url).netloc, "itch.io") and " by " in cleaned.casefold():
        candidates.insert(0, re.split(r"\s+by\s+", cleaned, maxsplit=1, flags=re.I)[0].strip())
    ascii_parts = [
        part.strip(" -|:：")
        for part in split_title(cleaned)
        if any(char.isascii() and char.isalpha() for char in part) and 3 <= len(part.strip()) <= 80
    ]
    candidates.extend(ascii_parts)
    candidates.extend(
        part.strip()
        for part in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_.-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'_.-]*)+", cleaned)
        if 3 <= len(part.strip()) <= 80
    )
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))


def matches_game_text(*, text: str, game: str, game_aliases: list[str]) -> bool:
    return matches_game_identity(text=text, game_names=[game, *game_aliases])


def identity_matches_game(*, title: str, url: str, game: str) -> bool:
    return matches_game_identity(text=f"{title} {url}", game_names=[game])


def identity_names_equivalent(left: str, right: str) -> bool:
    return compact_identity_text(left) == compact_identity_text(right)


def is_near_game_identity(*, title: str, url: str, game: str) -> bool:
    """Accept a likely typo only as a candidate from a verifiable identity URL."""
    if not is_candidate_identity_url(url):
        return False
    target = compact_identity_text(game)
    if len(target) < 5:
        return False
    return any(
        SequenceMatcher(a=target, b=compact_identity_text(candidate)).ratio() >= 0.84
        for candidate in title_alias_candidates(title, url=url)
        if compact_identity_text(candidate)
    )


def compact_identity_text(value: str) -> str:
    """Normalize identity text without discarding non-Latin game titles."""
    return "".join(char for char in value.casefold() if char.isalnum())
