"""Build a bounded, complementary portfolio of search queries.

Search plans describe useful subjects and source preferences, but they should
not turn into a single ``site:``-restricted retrieval path.  This module keeps
source-specific queries and an open-web query together so an incomplete or
incorrect source registry cannot hide long-tail guides.
"""

from collections.abc import Iterable

from quality_policy import MAX_QUERIES_PER_PLANNED_QUERY, MAX_SEARCH_QUERIES, SourcePolicy
from query_tokens import exact_identifiers
from retrieval.source_quality import required_entity_groups_for_query
from schemas import SearchPlan


def build_search_queries(
    *,
    game: str,
    question: str,
    plan: SearchPlan,
    sources: dict[str, SourcePolicy],
    database_domains: tuple[str, ...] = (),
    game_aliases: tuple[str, ...] = (),
) -> list[tuple[str, SourcePolicy]]:
    """Expand a plan into source-specific and open-web query strategies.

    The first-wave candidates intentionally cover different planned semantics
    as well as different source routes. Intent rules therefore influence the
    portfolio without becoming an exclusion filter.
    """
    planned_query_limit = 1 if plan.refinement else 4
    planned_queries = list(plan.queries)[:planned_query_limit]
    per_planned_query_limit = 1 if plan.refinement else MAX_QUERIES_PER_PLANNED_QUERY
    entity_aliases = _unique_text(plan.aliases[:3])
    known_game_aliases = [
        alias
        for alias in _unique_text(game_aliases[:3])
        if alias.casefold() != game.casefold()
    ]
    portfolios: list[list[tuple[str, SourcePolicy]]] = []
    for planned in planned_queries:
        preferred_source = sources.get(planned.source_type, sources["web"])
        planned_query = planned.query.replace("{question}", question).strip()
        routed_entity_aliases = entity_aliases
        if plan.named_entity_groups:
            routed_entity_aliases = _unique_text(
                name
                for group in required_entity_groups_for_query(
                    plan.named_entity_groups,
                    planned_query,
                )
                for name in group
            )
            # The planner's aliases are alternative surfaces of the requested
            # entity. Keep them in an initial portfolio even when a structured
            # entity group is present; otherwise a localized user surface can
            # silently displace the translated name needed by long-tail pages.
            # A refinement query is deliberately narrower and follows one
            # dependency, so it must not reattach aliases for an unrelated
            # endpoint from the original question.
            if not plan.refinement:
                routed_entity_aliases = _unique_text([
                    *routed_entity_aliases,
                    *entity_aliases,
                ])
        candidates = _query_portfolio(
            game=game,
            question=question,
            planned_query=planned_query,
            preferred_source=preferred_source,
            web_source=sources["web"],
            database_domains=database_domains,
            game_aliases=known_game_aliases,
            entity_aliases=routed_entity_aliases,
            prefer_open=plan.refinement,
        )

        portfolios.append(candidates)

    built: list[tuple[str, SourcePolicy]] = []
    seen: set[str] = set()
    per_plan_counts = [0] * len(portfolios)
    for plan_index, candidate, source in _interleaved_candidates(portfolios):
        if per_plan_counts[plan_index] >= per_planned_query_limit:
            continue
        normalized = _normalize_query(candidate)
        dedupe_key = normalized.casefold()
        if not normalized or dedupe_key in seen:
            continue
        built.append((normalized, source))
        seen.add(dedupe_key)
        per_plan_counts[plan_index] += 1
        if len(built) >= MAX_SEARCH_QUERIES:
            return built
    return built


def _interleaved_candidates(
    portfolios: list[list[tuple[str, SourcePolicy]]],
) -> list[tuple[int, str, SourcePolicy]]:
    """Put independent planned semantics inside the first-wave budget."""
    if not portfolios:
        return []
    if len(portfolios) == 1:
        return [(0, query, source) for query, source in portfolios[0]]

    ordered: list[tuple[int, str, SourcePolicy]] = []
    used_indexes: dict[int, set[int]] = {index: set() for index in range(len(portfolios))}

    # Keep the planner's highest-priority targeted route first.
    if portfolios[0]:
        query, source = portfolios[0][0]
        ordered.append((0, query, source))
        used_indexes[0].add(0)

    # Then cover every other planned semantic with its open-web route before
    # spending another call on a variant of the first semantic.
    for plan_index, portfolio in enumerate(portfolios[1:], start=1):
        open_index = next(
            (
                index
                for index, (query, source) in enumerate(portfolio)
                if source.source_type == "web" and "site:" not in query
            ),
            0,
        )
        if portfolio:
            query, source = portfolio[open_index]
            ordered.append((plan_index, query, source))
            used_indexes[plan_index].add(open_index)

    for candidate_index in range(max((len(portfolio) for portfolio in portfolios), default=0)):
        for plan_index, portfolio in enumerate(portfolios):
            if candidate_index >= len(portfolio) or candidate_index in used_indexes[plan_index]:
                continue
            query, source = portfolio[candidate_index]
            ordered.append((plan_index, query, source))
    return ordered


def _query_portfolio(
    *,
    game: str,
    question: str,
    planned_query: str,
    preferred_source: SourcePolicy,
    web_source: SourcePolicy,
    database_domains: tuple[str, ...],
    game_aliases: list[str],
    entity_aliases: list[str],
    prefer_open: bool,
) -> list[tuple[str, SourcePolicy]]:
    """Return ordered candidates whose retrieval assumptions are independent."""
    focused_query = _with_missing_alias(planned_query, entity_aliases)
    source_specific = _source_specific_queries(
        game=game,
        query=focused_query,
        source=preferred_source,
        database_domains=database_domains,
    )

    # Prefer a confirmed alternate title for the open query.  This is useful
    # when the user supplies a localized title but long-tail material uses the
    # store/English title.  The canonical title remains in source-specific
    # candidates, so the pair covers both names without requiring extra calls.
    open_game = game_aliases[0] if game_aliases else game
    exact_terms = exact_identifiers(f"{question} {planned_query}")[:4]
    quoted_identifiers = " ".join(f'"{term}"' for term in exact_terms)
    exact_entity = _quoted_alias(entity_aliases[0]) if entity_aliases else ""
    open_query = _join_query_parts(
        open_game,
        exact_entity,
        quoted_identifiers,
        planned_query,
        # A refinement query targets one missing dependency. Re-appending the
        # complete original question here would also re-attach every original
        # entity and make a dependency page satisfy unrelated endpoints.
        question
        if not prefer_open and question.casefold() not in planned_query.casefold()
        else "",
    )

    # The open candidate deliberately carries the generic web policy.  It may
    # discover a wiki, an independent guide, or an official page; downstream
    # evidence scoring still validates game and question relevance.
    candidates: list[tuple[str, SourcePolicy]] = []
    open_first = prefer_open or preferred_source.source_type == "web"
    if open_first:
        candidates.append((open_query, web_source))
    if source_specific:
        candidates.append((source_specific[0], preferred_source))
    if not open_first:
        candidates.append((open_query, web_source))
    candidates.extend((query, preferred_source) for query in source_specific[1:])

    # These variants are normally reached only when an earlier candidate
    # deduplicates.  Keeping them here makes the builder robust for policies
    # with identical templates without increasing the configured query count.
    candidates.append((_join_query_parts(game, focused_query), web_source))
    for alias in game_aliases[1:]:
        candidates.append((_join_query_parts(alias, focused_query), web_source))
    return candidates


def _source_specific_queries(
    *,
    game: str,
    query: str,
    source: SourcePolicy,
    database_domains: tuple[str, ...],
) -> list[str]:
    candidates: list[str] = []
    if source.source_type == "wiki":
        candidates.extend(f"site:{domain} {game} {query}" for domain in _unique_text(database_domains))
    candidates.extend(f"site:{domain} {game} {query}" for domain in _unique_text(source.domains))
    return candidates


def _with_missing_alias(query: str, aliases: list[str]) -> str:
    for alias in aliases:
        if alias.casefold() not in query.casefold():
            return _join_query_parts(alias, query)
    return query


def _quoted_alias(alias: str) -> str:
    if " " in alias and not (alias.startswith('"') and alias.endswith('"')):
        return f'"{alias}"'
    return alias


def _join_query_parts(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _normalize_query(query: str, *, max_chars: int = 500) -> str:
    """Collapse whitespace and bound requests without dropping the final relation."""
    normalized = " ".join(query.split())
    if len(normalized) <= max_chars:
        return normalized
    # Open candidates place game/entity identity first and the user's exact
    # condition near the end. Keeping both edges is safer than a head-only cut,
    # which can silently turn a novel relationship into a generic entity query.
    left = (max_chars * 3) // 5
    right = max_chars - left - 1
    return f"{normalized[:left].rstrip()} {normalized[-right:].lstrip()}"[:max_chars].strip()


def _unique_text(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            unique.append(normalized)
            seen.add(key)
    return unique
