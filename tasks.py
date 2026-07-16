import asyncio
from datetime import datetime

from celery import Celery

from config import get_settings
from knowledge import KnowledgeStore

settings = get_settings()

celery_app = Celery(
    "questmate",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="questmate.index_url")
def index_url(
    url: str,
    game: str,
    title: str | None = None,
    source_type: str = "web",
    game_version: str | None = None,
    published_at: str | None = None,
) -> dict[str, object]:
    """Fetch, extract, chunk and persist one guide page for retrieval."""
    async def run_once() -> dict[str, object]:
        # Celery invokes this synchronous task repeatedly in the same worker
        # process, while asyncio.run creates a fresh event loop each time.  A
        # per-task store prevents HTTP/DB pools created on an earlier loop from
        # being reused after that loop has closed.
        store = KnowledgeStore()
        try:
            return await store.index_url(
                url=url,
                game=game,
                title=title,
                source_type=source_type,
                game_version=game_version,
                published_at=datetime.fromisoformat(published_at.replace("Z", "+00:00")) if published_at else None,
            )
        finally:
            await store.aclose()

    return asyncio.run(run_once())
