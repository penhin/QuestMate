"""Persistent game identity and source-entry registry.

Search engines are used to discover a game's durable entry points once. Later
questions can reuse the confirmed aliases, store pages, and wiki domains without
paying for identity discovery again.
"""

from datetime import datetime, timezone

import structlog
from sqlalchemy import Column, DateTime, Float, String, Table, insert, select, update
from sqlalchemy.dialects.postgresql import JSONB

from config import Settings, get_settings
from schemas import GameResolution
from storage import Database, metadata


logger = structlog.get_logger()

game_source_profiles = Table(
    "game_source_profiles",
    metadata,
    Column("game_key", String(160), primary_key=True),
    Column("canonical_name", String(120), nullable=False),
    Column("aliases", JSONB, nullable=False),
    Column("platform_urls", JSONB, nullable=False),
    Column("official_urls", JSONB, nullable=False),
    Column("database_domains", JSONB, nullable=False),
    Column("confidence", Float, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class GameSourceRegistry:
    def __init__(self, database: Database | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.database = database or Database(self.settings)
        self._initialized = False
        self._available = False

    async def init_schema(self) -> None:
        if self._initialized:
            return
        async with self.database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        self._initialized = True
        self._available = True

    async def get_resolution(self, game: str) -> GameResolution | None:
        try:
            await self.init_schema()
            game_key = self._game_key(game)
            async with self.database.sessionmaker() as session:
                direct = (
                    await session.execute(
                        select(game_source_profiles).where(game_source_profiles.c.game_key == game_key)
                    )
                ).mappings().one_or_none()
                if direct is not None:
                    return self._resolution_from_row(dict(direct), input_name=game)

                rows = (
                    await session.execute(
                        select(game_source_profiles)
                        .order_by(game_source_profiles.c.updated_at.desc())
                        .limit(200)
                    )
                ).mappings()
                lowered = game.casefold().strip()
                for row in rows:
                    names = [str(row["canonical_name"]), *map(str, row["aliases"] or [])]
                    if any(name.casefold().strip() == lowered for name in names):
                        return self._resolution_from_row(dict(row), input_name=game)
        except Exception as exc:
            logger.warning("source_registry.read_failed", error_type=type(exc).__name__)
        return None

    async def upsert_resolution(self, resolution: GameResolution) -> None:
        if not resolution.is_confirmed:
            return
        try:
            await self.init_schema()
            canonical_name = resolution.confirmed_name or resolution.input_name
            game_key = self._game_key(canonical_name)
            now = datetime.now(timezone.utc)
            async with self.database.sessionmaker() as session:
                async with session.begin():
                    existing = (
                        await session.execute(
                            select(game_source_profiles).where(game_source_profiles.c.game_key == game_key)
                        )
                    ).mappings().one_or_none()
                    values = self._merged_values(existing=dict(existing) if existing else None, resolution=resolution)
                    values["updated_at"] = now
                    if existing:
                        await session.execute(
                            update(game_source_profiles)
                            .where(game_source_profiles.c.game_key == game_key)
                            .values(**values)
                        )
                    else:
                        await session.execute(
                            insert(game_source_profiles).values(
                                game_key=game_key,
                                created_at=now,
                                **values,
                            )
                        )
            logger.info(
                "source_registry.upserted",
                game=canonical_name,
                database_count=len(values["database_domains"]),
                platform_count=len(values["platform_urls"]),
            )
        except Exception as exc:
            logger.warning("source_registry.write_failed", error_type=type(exc).__name__)

    async def list_resolutions(self, *, limit: int = 100) -> list[GameResolution]:
        try:
            await self.init_schema()
            async with self.database.sessionmaker() as session:
                rows = (
                    await session.execute(
                        select(game_source_profiles).order_by(game_source_profiles.c.updated_at.desc()).limit(limit)
                    )
                ).mappings()
                return [self._resolution_from_row(dict(row), input_name=str(row["canonical_name"])) for row in rows]
        except Exception as exc:
            logger.warning("source_registry.list_failed", error_type=type(exc).__name__)
            return []

    @staticmethod
    def _game_key(game: str) -> str:
        return " ".join(game.casefold().split())[:160]

    @staticmethod
    def _merge_unique(existing: list[str], incoming: list[str], *, limit: int) -> list[str]:
        return list(dict.fromkeys([*existing, *incoming]))[:limit]

    @classmethod
    def _merged_values(cls, *, existing: dict | None, resolution: GameResolution) -> dict:
        existing = existing or {}
        canonical_name = resolution.confirmed_name or resolution.input_name
        aliases = cls._merge_unique(
            list(existing.get("aliases") or []),
            [alias for alias in [resolution.input_name, *resolution.aliases] if alias and alias != canonical_name],
            limit=8,
        )
        return {
            "canonical_name": canonical_name,
            "aliases": aliases,
            "platform_urls": cls._merge_unique(
                list(existing.get("platform_urls") or []),
                [str(url) for url in resolution.platform_urls],
                limit=6,
            ),
            "official_urls": cls._merge_unique(
                list(existing.get("official_urls") or []),
                [str(url) for url in resolution.official_urls],
                limit=4,
            ),
            "database_domains": cls._merge_unique(
                list(existing.get("database_domains") or []),
                list(resolution.database_domains),
                limit=8,
            ),
            "confidence": max(float(existing.get("confidence") or 0), resolution.confidence),
        }

    @staticmethod
    def _resolution_from_row(row: dict, *, input_name: str) -> GameResolution:
        return GameResolution(
            input_name=input_name,
            confirmed_name=str(row["canonical_name"]),
            aliases=list(row.get("aliases") or []),
            platform_urls=list(row.get("platform_urls") or []),
            official_urls=list(row.get("official_urls") or []),
            database_domains=list(row.get("database_domains") or []),
            confidence=float(row.get("confidence") or 0),
            ambiguous=False,
        )


game_source_registry = GameSourceRegistry()
