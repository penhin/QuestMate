"""MediaWiki provider adapter for the deterministic search route."""

from typing import Any

from schemas import Source


class MediaWikiProvider:
    """Expose direct game-wiki retrieval without leaking backend internals."""

    name = "mediawiki"

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    async def search(
        self,
        *,
        game: str,
        question: str,
        aliases: list[str],
        planned_queries: list[str],
        game_aliases: list[str],
        database_domains: list[str],
        max_results: int,
        named_entity_groups: list[list[str]],
    ) -> list[Source]:
        return await self._retriever.search(
            game=game,
            question=question,
            aliases=aliases,
            planned_queries=planned_queries,
            game_aliases=game_aliases,
            database_domains=database_domains,
            max_results=max_results,
            named_entity_groups=named_entity_groups,
        )

