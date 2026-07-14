# QuestMate

游戏攻略问答服务，包含 FastAPI 后端和 Tauri 桌面悬浮窗。

## 启动后端

```bash
cp .env.example .env
docker compose up -d postgres redis
uv sync
uv run uvicorn main:app --reload
```

服务地址：`http://127.0.0.1:8000`

常用命令：

```bash
uv run pytest
uv run celery -A tasks.celery_app worker --loglevel=info
```

## 启动桌面端

```bash
cd overlay
npm install
npm run dev
```

桌面端默认请求 `http://127.0.0.1:8000`；可通过 `VITE_API_BASE_URL` 覆盖。

## 接口

- `GET /health`
- `POST /api/chat`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `PATCH /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`
- `POST /api/feedback`
- `POST /api/knowledge/documents`
- `GET /api/knowledge/documents?game=Elden%20Ring`

## 索引资料

先启动 Postgres、Redis 和 Celery worker：

```bash
docker compose up -d postgres redis
uv run celery -A tasks.celery_app worker --loglevel=info
```

提交资料：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/documents \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/guide","game":"Elden Ring","source_type":"wiki","game_version":"1.12"}'
```

嵌入接口未配置时，知识库使用关键词检索；配置 OpenAI 兼容嵌入接口后启用 pgvector 语义检索。
