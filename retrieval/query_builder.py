"""Build bounded search queries from model plans and source policies."""

from quality_policy import MAX_QUERIES_PER_PLANNED_QUERY, MAX_SEARCH_QUERIES, SourcePolicy
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
    """Expand a search plan without coupling query policy to an HTTP adapter."""
    planned_queries = list(plan.queries)[:4]
    aliases = list(plan.aliases[:3])
    per_planned_query_limit = 1 if plan.refinement else MAX_QUERIES_PER_PLANNED_QUERY
    built: list[tuple[str, SourcePolicy]] = []
    seen: set[str] = set()

    for planned in planned_queries:
        source = sources.get(planned.source_type, sources["web"])
        query = planned.query.replace("{question}", question).strip()
        candidates: list[str] = []

        if source.source_type == "wiki":
            candidates.extend(f"site:{domain} {game} {query}" for domain in database_domains)
            for alias in aliases:
                if alias.casefold() not in query.casefold():
                    candidates.extend(f"site:{domain} {game} {alias} {query}" for domain in database_domains)
            for game_alias in game_aliases[:3]:
                if game_alias.casefold() != game.casefold():
                    candidates.extend(f"site:{domain} {game_alias} {query}" for domain in database_domains)

        for domain_index, domain in enumerate(source.domains):
            candidates.append(f"site:{domain} {game} {query}")
            if domain_index == 0:
                for alias in aliases:
                    if alias.casefold() not in query.casefold():
                        candidates.append(f"site:{domain} {game} {alias} {query}")
            for game_alias in game_aliases[:3]:
                if game_alias.casefold() != game.casefold():
                    candidates.append(f"site:{domain} {game_alias} {query}")

        for game_alias in game_aliases[:3]:
            if game_alias.casefold() != game.casefold():
                candidates.extend(template.format(game=game_alias, query=query) for template in source.query_templates)
        candidates.extend(template.format(game=game, query=query) for template in source.query_templates)
        if not candidates:
            candidates.append(f"{game} {query}")
        for alias in aliases:
            if alias.casefold() not in query.casefold():
                candidates.append(f"{game} {alias} {query}")

        added_for_plan = 0
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if normalized in seen:
                continue
            built.append((normalized, source))
            seen.add(normalized)
            added_for_plan += 1
            if len(built) >= MAX_SEARCH_QUERIES:
                return built
            if added_for_plan >= per_planned_query_limit:
                break

    return built
