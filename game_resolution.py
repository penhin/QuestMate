import re
from difflib import SequenceMatcher
from typing import Any, Protocol
from urllib.parse import urlparse

from query_tokens import relevance_tokens
from identity_components.scoring import resolution_confidence
from identity_components.candidate_normalization import (
    candidate_key as normalized_candidate_key,
    choose_canonical_candidate_name as normalized_candidate_name,
    infer_game_tags as normalized_game_tags,
    is_low_value_game_candidate as normalized_low_value_candidate,
    title_alias_candidates as normalized_title_aliases,
    useful_candidate_aliases as normalized_candidate_aliases,
)
from identity_components.selection import (
    canonical_identity_url as _selection_canonical_identity_url,
    domain_matches as _selection_domain_matches,
    is_candidate_identity_url as _selection_is_candidate_identity_url,
    is_platform_product_url as _selection_is_platform_product_url,
    platform_resource_identity as _selection_platform_resource_identity,
    resolution_matches_selected_url as _selection_resolution_matches_selected_url,
    same_platform_resource as _selection_same_platform_resource,
    select_game_candidate as _selection_select_game_candidate,
)
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
                item_aliases = list(normalized_title_aliases(title, url=url))
                if normalized_low_value_candidate(title=title, url=url):
                    continue
                if url not in platform_urls:
                    platform_urls.append(url)
                name = normalized_candidate_name(game=game, candidates=item_aliases, fallback=game)
                useful_aliases = normalized_candidate_aliases(name=name, candidates=item_aliases)
                raw_score = float(item.get("score") or 0.5)
                candidates.append(
                    GameCandidate(
                        name=name,
                        aliases=useful_aliases,
                        tags=normalized_game_tags(self.result_text(item)),
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
                item_aliases = list(normalized_title_aliases(title, url=url))
                name = normalized_candidate_name(game=game, candidates=item_aliases, fallback=game)
                useful_aliases = normalized_candidate_aliases(name=name, candidates=item_aliases)
                if url not in identity_urls:
                    identity_urls.append(url)
                raw_score = float(item.get("score") or 0.5)
                candidates.append(
                    GameCandidate(
                        name=name,
                        aliases=useful_aliases,
                        tags=normalized_game_tags(self.result_text(item)),
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
                if normalized_low_value_candidate(title=title, url=url):
                    continue
                aliases = list(normalized_title_aliases(title, url=url))
                name = normalized_candidate_name(game=game, candidates=aliases, fallback=title[:80])
                aliases = normalized_candidate_aliases(name=name, candidates=aliases)
                canonical_key = normalized_candidate_key(name=name, url=url)
                tags = normalized_game_tags(text)
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
        return resolution_confidence(
            game=game,
            aliases=aliases,
            platform_urls=platform_urls,
            official_urls=official_urls,
            identity_urls=identity_urls,
            database_domains=database_domains,
        )


def infer_game_tags(text: str) -> list[str]:
    return normalized_game_tags(text)


def is_low_value_game_candidate(*, title: str, url: str) -> bool:
    return normalized_low_value_candidate(title=title, url=url)


def candidate_key(*, name: str, url: str) -> str:
    return normalized_candidate_key(name=name, url=url)


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
    return _selection_platform_resource_identity(url)


def same_platform_resource(left_url: str, right_url: str) -> bool:
    """Compare server-discovered product identities without trusting display names."""
    return _selection_same_platform_resource(left_url, right_url)


def _canonical_identity_url(url: str) -> str:
    return _selection_canonical_identity_url(url)


def is_candidate_identity_url(url: str) -> bool:
    return _selection_is_candidate_identity_url(url)


def select_game_candidate(
    resolution: GameResolution,
    *,
    selected_url: str,
) -> GameResolution | None:
    """Resolve an opaque UI choice against fresh server-discovered candidates."""
    return _selection_select_game_candidate(
        resolution,
        selected_url=selected_url,
        confirmed_threshold=GAME_RESOLUTION_POLICY.confirmed_threshold,
    )


def resolution_matches_selected_url(
    resolution: GameResolution,
    *,
    selected_url: str,
) -> bool:
    """Verify that a returned resolution represents the user's opaque choice."""
    return _selection_resolution_matches_selected_url(resolution, selected_url=selected_url)


def choose_canonical_candidate_name(
    *,
    game: str,
    candidates: list[str] | tuple[str, ...],
    fallback: str,
) -> str:
    return normalized_candidate_name(game=game, candidates=candidates, fallback=fallback)


def useful_candidate_aliases(
    *,
    name: str,
    candidates: list[str] | tuple[str, ...],
) -> list[str]:
    return normalized_candidate_aliases(name=name, candidates=candidates)


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
    return _selection_is_platform_product_url(url)


def domain_matches(domain: str, candidate: str) -> bool:
    return _selection_domain_matches(domain, candidate)


def title_alias_candidates(title: str, *, url: str = "") -> tuple[str, ...]:
    return normalized_title_aliases(title, url=url)


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
