import re
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse
from typing import Any, Protocol

from tavily import TavilyClient

from config import Settings, get_settings
from query_tokens import question_relevance_tokens, relevance_tokens
from schemas import GameCandidate, GameResolution, PlannedSearchQuery, SearchPlan, Source


def re_split_title(value: str) -> list[str]:
    return re.split(r"\s[-|:：]\s|[（）()【】\[\]]", value)


@dataclass(frozen=True)
class SearchSource:
    source_type: str
    trust_score: float
    trust_label: str
    domains: tuple[str, ...] = ()
    query_templates: tuple[str, ...] = ()


class SearchProvider(Protocol):
    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        ...

    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
        game_resolution: GameResolution | None = None,
    ) -> list[Source]:
        ...


class TavilySearchProvider:
    search_noise_tokens = {
        "fandom",
        "fextralife",
        "wiki",
        "guide",
        "strategy",
        "weakness",
        "timing",
        "location",
        "merchant",
        "questline",
        "walkthrough",
        "build",
        "stats",
        "weapons",
        "talismans",
        "official",
        "patch",
        "notes",
        "update",
    }
    sources = {
        "official": SearchSource(
            "official",
            0.95,
            "官方",
            query_templates=(
                "{game} official {query}",
                "{game} patch notes update {query}",
            ),
        ),
        "wiki": SearchSource(
            "wiki",
            0.8,
            "百科",
            domains=("fandom.com", "wiki.gg", "fextralife.com"),
        ),
        "community": SearchSource(
            "community",
            0.55,
            "社区",
            domains=("reddit.com", "steamcommunity.com"),
        ),
        "web": SearchSource(
            "web",
            0.45,
            "网页",
            query_templates=(
                "{game} guide {query}",
                "{game} 攻略 {query}",
            ),
        ),
    }
    fallback_plan = SearchPlan(
        intent="general",
        queries=(
            PlannedSearchQuery(source_type="wiki", query="{question}"),
            PlannedSearchQuery(source_type="community", query="{question}"),
            PlannedSearchQuery(source_type="web", query="{question}"),
        ),
    )

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client or (TavilyClient(api_key=self.settings.tavily_api_key) if self.settings.tavily_api_key else None)

    async def search(
        self,
        query: str,
        game: str,
        max_results: int | None = None,
        plan: SearchPlan | None = None,
        game_resolution: GameResolution | None = None,
    ) -> list[Source]:
        if self._client is None:
            return []

        game_resolution = game_resolution or await self.resolve_game(game=game, question=query)
        total_results = max_results or self.settings.search_max_results
        per_query_results = min(4, max(2, total_results))
        strict_sources_by_url: dict[str, Source] = {}
        relaxed_sources_by_url: dict[str, Source] = {}
        search_queries = self._build_search_queries(
            game=game,
            question=query,
            plan=plan,
            database_domains=tuple(game_resolution.database_domains),
            game_aliases=tuple(game_resolution.aliases),
        )
        intent = (plan.intent if plan else "general") or "general"
        aliases = list((plan.aliases if plan else [])[:6])
        game_aliases = list(game_resolution.aliases)
        min_strict_results = min(3, total_results)

        self._collect_sources(
            search_queries=search_queries,
            per_query_results=per_query_results,
            game=game,
            query=query,
            aliases=aliases,
            game_aliases=game_aliases,
            intent=intent,
            strict_sources_by_url=strict_sources_by_url,
            relaxed_sources_by_url=relaxed_sources_by_url,
        )

        if len(strict_sources_by_url) < min_strict_results and intent != "patch":
            database_domains = tuple(game_resolution.database_domains) or self._discover_database_domains(
                game=game,
                game_aliases=game_aliases,
            )
            if database_domains or game_aliases:
                database_queries = self._build_search_queries(
                    game=game,
                    question=query,
                    plan=plan,
                    database_domains=database_domains,
                    game_aliases=tuple(game_aliases),
                )
                self._collect_sources(
                    search_queries=database_queries,
                    per_query_results=per_query_results,
                    game=game,
                    query=query,
                    aliases=aliases,
                    game_aliases=game_aliases,
                    intent=intent,
                    strict_sources_by_url=strict_sources_by_url,
                    relaxed_sources_by_url=relaxed_sources_by_url,
                )

        strict_ranked_sources = sorted(
            strict_sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )
        relaxed_ranked_sources = sorted(
            relaxed_sources_by_url.values(),
            key=lambda source: ((source.score or 0), source.trust_score),
            reverse=True,
        )
        return self._balanced_sources(
            strict_sources=strict_ranked_sources,
            relaxed_sources=relaxed_ranked_sources,
            total_results=total_results,
            min_strict_results=min_strict_results,
        )

    async def resolve_game(self, game: str, question: str | None = None) -> GameResolution:
        if self._client is None:
            return GameResolution(input_name=game, confirmed_name=game, confidence=0)

        candidates = list(self._discover_game_candidates(game=game))
        aliases = list(self._discover_game_aliases(game=game))
        for candidate in candidates:
            for alias in [candidate.name, *candidate.aliases]:
                if alias.lower() != game.lower() and alias not in aliases:
                    aliases.append(alias)
        database_domains = list(self._discover_database_domains(game=game, game_aliases=aliases))
        platform_urls = list(self._discover_platform_urls(game=game, game_aliases=aliases))
        confirmed_name = candidates[0].name if candidates else (aliases[0] if aliases else game)
        confidence = self._resolution_confidence(
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

    def _collect_sources(
        self,
        *,
        search_queries: list[tuple[str, SearchSource]],
        per_query_results: int,
        game: str,
        query: str,
        aliases: list[str],
        game_aliases: list[str],
        intent: str,
        strict_sources_by_url: dict[str, Source],
        relaxed_sources_by_url: dict[str, Source],
    ) -> None:
        for search_query, search_source in search_queries:
            result = self._client.search(
                query=search_query,
                max_results=per_query_results,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = item.get("url")
                if not url:
                    continue
                search_context = f"{query} {search_query} {' '.join(aliases)}"
                relevance_score = self._result_relevance_score(
                    item=item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                )
                if relevance_score <= 0:
                    continue

                raw_score = float(item.get("score") or 0)
                intent_score = self._intent_source_boost(intent=intent, source_type=search_source.source_type)
                domain_score = self._domain_quality_score(str(url))
                version_score = self._version_safety_score(
                    intent=intent,
                    source_type=search_source.source_type,
                    text=f"{item.get('title') or ''} {item.get('url') or ''} {item.get('content') or ''}",
                )
                weighted_score = (
                    raw_score * 0.25
                    + search_source.trust_score * 0.25
                    + relevance_score * 0.35
                    + intent_score * 0.1
                    + domain_score * 0.03
                    + version_score * 0.02
                )
                source = Source(
                    title=item.get("title") or url,
                    url=url,
                    snippet=item.get("content"),
                    score=weighted_score,
                    source_type=search_source.source_type,
                    trust_score=search_source.trust_score,
                    trust_label=search_source.trust_label,
                )
                source_key = self._canonical_source_key(str(url))
                current = relaxed_sources_by_url.get(source_key)
                if current is None or (source.score or 0) > (current.score or 0):
                    relaxed_sources_by_url[source_key] = source
                if self._is_high_quality_source(
                    item=item,
                    game=game,
                    game_aliases=game_aliases,
                    question=search_context,
                    source_type=search_source.source_type,
                ):
                    current = strict_sources_by_url.get(source_key)
                    if current is None or (source.score or 0) > (current.score or 0):
                        strict_sources_by_url[source_key] = source

    def _discover_database_domains(self, *, game: str, game_aliases: list[str] | None = None) -> tuple[str, ...]:
        game_names = [game, *(game_aliases or [])]
        candidates = tuple(
            f"{game_name} {suffix}"
            for game_name in game_names[:3]
            for suffix in ("fandom wiki", "wiki.gg wiki", "official wiki")
        )
        domains: list[str] = []
        for query in candidates:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "")
                domain = urlparse(url).netloc.lower()
                if not self._is_supported_database_domain(domain):
                    continue
                text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).lower()
                if not self._matches_game_text(text=text, game=game, game_aliases=game_aliases or []):
                    continue
                if domain not in domains:
                    domains.append(domain)
                if len(domains) >= 4:
                    return tuple(domains)
        return tuple(domains)

    @staticmethod
    def _is_supported_database_domain(domain: str) -> bool:
        return (
            domain.endswith(".fandom.com")
            or domain == "fandom.com"
            or domain.endswith(".wiki.gg")
            or domain == "wiki.gg"
        )

    def _discover_game_aliases(self, *, game: str) -> tuple[str, ...]:
        candidates = (
            f"{game} Steam",
            f"{game} Steam official",
            f"{game} itch.io",
            f"{game} GOG",
            f"{game} game English title",
        )
        aliases: list[str] = []
        for query in candidates:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).lower()
                if not self._matches_game_text(text=text, game=game, game_aliases=[]):
                    continue
                title = str(item.get("title") or "")
                for alias in self._title_alias_candidates(title):
                    if alias.lower() != game.lower() and alias not in aliases:
                        aliases.append(alias)
                    if len(aliases) >= 4:
                        return tuple(aliases)
        return tuple(aliases)

    def _discover_platform_urls(self, *, game: str, game_aliases: list[str] | None = None) -> tuple[str, ...]:
        game_names = [game, *(game_aliases or [])]
        candidates = tuple(
            f"{game_name} {platform}"
            for game_name in game_names[:3]
            for platform in ("Steam", "itch.io", "GOG", "Epic Games")
        )
        urls: list[str] = []
        for query in candidates:
            result = self._client.search(
                query=query,
                max_results=3,
                include_answer=False,
                include_raw_content=False,
            )
            for item in result.get("results", []):
                url = str(item.get("url") or "")
                domain = urlparse(url).netloc.lower()
                if not self._is_supported_platform_domain(domain):
                    continue
                text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).lower()
                if not self._matches_game_text(text=text, game=game, game_aliases=game_aliases or []):
                    continue
                if url not in urls:
                    urls.append(url)
                if len(urls) >= 6:
                    return tuple(urls)
        return tuple(urls)

    def _discover_game_candidates(self, *, game: str) -> tuple[GameCandidate, ...]:
        queries = (
            f"{game} Steam",
            f"{game} itch.io game",
            f"{game} GOG game",
            f"{game} Epic Games",
        )
        candidates_by_name: dict[str, GameCandidate] = {}
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
                if not self._is_supported_platform_domain(domain):
                    continue
                text = " ".join(str(item.get(field) or "") for field in ("title", "url", "content")).lower()
                if not self._matches_game_text(text=text, game=game, game_aliases=[]):
                    continue
                title = str(item.get("title") or url)
                if self._is_low_value_game_candidate(title=title, url=url):
                    continue
                aliases = list(self._title_alias_candidates(title))
                name = aliases[0] if aliases else title[:80]
                canonical_key = self._candidate_key(name=name, url=url)
                tags = self._infer_game_tags(text)
                raw_score = float(item.get("score") or 0.5)
                confidence = min(1.0, 0.45 + raw_score * 0.35 + (0.15 if aliases else 0) + (0.05 if tags else 0))
                existing = candidates_by_name.get(canonical_key)
                platform_urls = [url]
                if existing:
                    platform_urls = list(dict.fromkeys([*map(str, existing.platform_urls), url]))
                    confidence = max(existing.confidence, confidence)
                    tags = list(dict.fromkeys([*existing.tags, *tags]))[:5]
                    aliases = list(dict.fromkeys([*existing.aliases, *aliases]))[:6]
                    name = existing.name if len(existing.name) <= len(name) else name
                candidates_by_name[canonical_key] = GameCandidate(
                    name=name,
                    aliases=[alias for alias in aliases if alias != name][:6],
                    tags=tags,
                    platform_urls=platform_urls[:4],
                    confidence=confidence,
                )
        return tuple(
            sorted(candidates_by_name.values(), key=lambda candidate: candidate.confidence, reverse=True)[:6]
        )

    @staticmethod
    def _infer_game_tags(text: str) -> list[str]:
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

    @staticmethod
    def _is_low_value_game_candidate(*, title: str, url: str) -> bool:
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

    @staticmethod
    def _candidate_key(*, name: str, url: str) -> str:
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

    @staticmethod
    def _is_supported_platform_domain(domain: str) -> bool:
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

    @staticmethod
    def _resolution_confidence(
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

    @staticmethod
    def _title_alias_candidates(title: str) -> tuple[str, ...]:
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
            for part in re_split_title(cleaned)
            if any(char.isascii() and char.isalpha() for char in part) and 3 <= len(part.strip()) <= 80
        ]
        candidates.extend(ascii_parts)
        return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))

    @staticmethod
    def _matches_game_text(*, text: str, game: str, game_aliases: list[str]) -> bool:
        game_tokens = relevance_tokens(" ".join([game, *game_aliases]))
        return not game_tokens or any(token in text for token in game_tokens)

    def _build_search_queries(
        self,
        *,
        game: str,
        question: str,
        plan: SearchPlan | None,
        database_domains: tuple[str, ...] = (),
        game_aliases: tuple[str, ...] = (),
    ) -> list[tuple[str, SearchSource]]:
        planned_queries = list((plan or self.fallback_plan).queries)[:4] or list(self.fallback_plan.queries)
        aliases = list((plan.aliases if plan else [])[:3])
        built: list[tuple[str, SearchSource]] = []
        seen: set[str] = set()

        for planned in planned_queries:
            source = self.sources.get(planned.source_type, self.sources["web"])
            query = planned.query.replace("{question}", question).strip()

            candidates: list[str] = []
            if source.source_type == "wiki":
                candidates.extend(f"site:{domain} {game} {query}" for domain in database_domains)
            candidates.extend(f"site:{domain} {game} {query}" for domain in source.domains)
            for game_alias in game_aliases[:3]:
                if game_alias.lower() != game.lower():
                    candidates.extend(f"site:{domain} {game_alias} {query}" for domain in source.domains)
                    candidates.extend(template.format(game=game_alias, query=query) for template in source.query_templates)
            candidates.extend(template.format(game=game, query=query) for template in source.query_templates)
            if not candidates:
                candidates.append(f"{game} {query}")
            for alias in aliases:
                if alias.lower() not in query.lower():
                    candidates.append(f"{game} {alias} {query}")

            for candidate in candidates:
                normalized = " ".join(candidate.split())
                if normalized in seen:
                    continue
                built.append((normalized, source))
                seen.add(normalized)
                if len(built) >= 8:
                    return built

        return built

    @staticmethod
    def _is_relevant_result(*, item: dict[str, Any], game: str, question: str) -> bool:
        return TavilySearchProvider._result_relevance_score(item=item, game=game, question=question) > 0

    @staticmethod
    def _result_relevance_score(
        *,
        item: dict[str, Any],
        game: str,
        question: str,
        game_aliases: list[str] | None = None,
    ) -> float:
        text = " ".join(
            str(item.get(field) or "")
            for field in ("title", "url", "content")
        ).lower()
        if not TavilySearchProvider._is_same_game_surface(text=text, game=game, question=question):
            return 0
        if TavilySearchProvider._is_low_value_page(text=text, question=question):
            return 0

        game_tokens = relevance_tokens(" ".join([game, *(game_aliases or [])]))
        game_token_set = set(game_tokens)
        question_tokens = [
            token
            for token in question_relevance_tokens(question)
            if token not in game_token_set and token not in TavilySearchProvider.search_noise_tokens
        ]

        has_game_match = not game_tokens or any(token in text for token in game_tokens)
        if not has_game_match:
            return 0

        if not question_tokens:
            return 0.45

        matched = sum(1 for token in question_tokens if token in text)
        if matched == 0:
            return 0

        title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).lower()
        focused_matches = sum(1 for token in question_tokens if token in title_url)
        coverage = matched / max(len(question_tokens), 1)
        focus_bonus = min(focused_matches * 0.12, 0.3)
        return min(1.0, 0.35 + coverage * 0.45 + focus_bonus)

    @staticmethod
    def _is_same_game_surface(*, text: str, game: str, question: str) -> bool:
        normalized_game = game.lower()
        normalized_question = question.lower()
        if "elden ring" in normalized_game and "nightreign" in text and "nightreign" not in normalized_question:
            return False
        return True

    @staticmethod
    def _is_low_value_page(*, text: str, question: str) -> bool:
        lowered_question = question.lower()
        if "villains.fandom.com" in text and not any(token in lowered_question for token in ("lore", "剧情", "背景")):
            return True
        if any(value in text for value in ("all-fiction-battles", "vs battles wiki", "battle wiki")) and not any(
            token in lowered_question for token in ("lore", "剧情", "背景")
        ):
            return True
        if "reddit - the heart of the internet" in text:
            return True
        if "reddit.com/r/eldenring/comments" not in text and any(
            value in text for value in ("reddit.com/r/eldenring", "reddit - the heart of the internet")
        ):
            return True
        if "steamcommunity.com/app" in text and "/discussions/" not in text:
            return True
        return False

    @staticmethod
    def _is_high_quality_source(
        *,
        item: dict[str, Any],
        game: str,
        question: str,
        source_type: str,
        game_aliases: list[str] | None = None,
    ) -> bool:
        if source_type == "official":
            return True

        title_url = " ".join(str(item.get(field) or "") for field in ("title", "url")).lower()
        game_token_set = set(relevance_tokens(" ".join([game, *(game_aliases or [])])))
        entity_tokens = [
            token
            for token in question_relevance_tokens(question)
            if token not in game_token_set and token not in TavilySearchProvider.search_noise_tokens
        ]
        title_entity_matches = sum(1 for token in entity_tokens if token in title_url)

        if source_type == "wiki":
            return title_entity_matches > 0
        if source_type == "community":
            return title_entity_matches > 0 and any(value in title_url for value in ("comments", "discussions"))
        return title_entity_matches > 0

    @staticmethod
    def _canonical_source_key(url: str) -> str:
        parsed = urlparse(url)
        if any(value in parsed.netloc.lower() for value in ("steamcommunity.com", "reddit.com")):
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
        return url

    @staticmethod
    def _limit_source_diversity(sources: list[Source], *, total_results: int) -> list[Source]:
        selected: list[Source] = []
        domain_counts: dict[str, int] = {}
        for source in sources:
            domain = urlparse(str(source.url)).netloc.lower()
            limit = 2 if any(value in domain for value in ("reddit.com", "steamcommunity.com")) else 3
            if domain_counts.get(domain, 0) >= limit:
                continue
            selected.append(source)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected) >= total_results:
                return selected
        return selected

    @classmethod
    def _balanced_sources(
        cls,
        *,
        strict_sources: list[Source],
        relaxed_sources: list[Source],
        total_results: int,
        min_strict_results: int,
    ) -> list[Source]:
        selected = cls._limit_source_diversity(strict_sources, total_results=total_results)
        if len(selected) >= min_strict_results or len(selected) >= total_results:
            return selected

        selected_keys = {cls._canonical_source_key(str(source.url)) for source in selected}
        fill_sources = [
            source
            for source in relaxed_sources
            if cls._canonical_source_key(str(source.url)) not in selected_keys
        ]
        combined = selected + fill_sources
        return cls._limit_source_diversity(combined, total_results=total_results)

    @staticmethod
    def _intent_source_boost(*, intent: str, source_type: str) -> float:
        preferred = {
            "boss_strategy": {"wiki": 0.8, "community": 1.0, "web": 0.45, "official": 0.2},
            "item_location": {"wiki": 1.0, "web": 0.65, "community": 0.35, "official": 0.2},
            "quest_step": {"wiki": 1.0, "web": 0.65, "community": 0.45, "official": 0.2},
            "item_usage": {"wiki": 1.0, "web": 0.65, "community": 0.45, "official": 0.25},
            "build": {"community": 1.0, "wiki": 0.65, "web": 0.45, "official": 0.2},
            "patch": {"official": 1.0, "wiki": 0.55, "web": 0.45, "community": 0.25},
            "lore": {"wiki": 0.9, "web": 0.65, "community": 0.35, "official": 0.2},
        }
        return preferred.get(intent, {}).get(source_type, 0.4)

    @staticmethod
    def _domain_quality_score(url: str) -> float:
        domain = urlparse(url).netloc.lower()
        if any(value in domain for value in ("wiki.gg", "fandom.com", "fextralife.com")):
            return 0.9
        if any(value in domain for value in ("bandainamco", "playstation.com", "steampowered.com")):
            return 0.85
        if any(value in domain for value in ("reddit.com", "steamcommunity.com")):
            return 0.55
        return 0.4

    @staticmethod
    def _version_safety_score(*, intent: str, source_type: str, text: str) -> float:
        lowered = text.lower()
        has_version_signal = any(
            token in lowered
            for token in ("patch", "version", "update", "1.", "版本", "补丁", "更新")
        )
        version_sensitive = intent in {"patch", "build", "boss_strategy"}
        if version_sensitive and source_type == "official":
            return 1.0
        if version_sensitive and has_version_signal:
            return 0.85
        if version_sensitive:
            return 0.45
        if intent in {"item_location", "item_usage", "quest_step", "lore"}:
            return 0.75
        return 0.55
