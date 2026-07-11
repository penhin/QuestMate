from celery import Celery

from config import get_settings

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
def index_url(url: str, game: str | None = None) -> dict[str, str | None]:
    return {
        "status": "queued",
        "url": url,
        "game": game,
    }

