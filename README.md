# QuestMate

QuestMate 是一个游戏攻略 agent 的 Python 后端骨架。它使用 FastAPI 构建 Web API，使用 Anthropic 调用 Claude，使用 LangGraph 编排 agent 流程，使用 Tavily 进行实时联网搜索，使用 Postgres + pgvector 缓存知识库，并使用 Celery + Redis 处理后台索引任务。

QuestMate is a Python backend skeleton for a game guide agent. It uses FastAPI for the web API, Anthropic for Claude calls, LangGraph for agent orchestration, Tavily for live web search, Postgres with pgvector for knowledge caching, and Celery with Redis for background indexing tasks.

## 快速开始 / Quick Start

```bash
cp .env.example .env
docker compose up -d postgres redis
uv sync
uv run uvicorn main:app --reload
```

API 将运行在 `http://127.0.0.1:8000`。

The API will be available at `http://127.0.0.1:8000`.

## 常用命令 / Useful Commands

```bash
uv run pytest
uv run celery -A tasks.celery_app worker --loglevel=info
```

## 桌面悬浮助手 / Desktop Overlay

桌面悬浮助手位于 `overlay/` 目录，使用 Tauri + React + TypeScript 构建。第一版优先支持 Windows 的窗口化和无边框全屏游戏，提供悬浮球、小弹窗和右侧抽屉。

The desktop overlay lives in `overlay/` and is built with Tauri + React + TypeScript. The first version prioritizes windowed and borderless fullscreen games on Windows, and provides a floating bubble, compact popover, and right drawer.

```bash
cd overlay
npm install
npm run dev
```

## 主要接口 / Main Endpoints

- `GET /health`: 健康检查 / health check.
- `POST /api/chat`: 提交游戏攻略问题 / ask a game guide question.
- `GET /api/sessions/{session_id}`: 获取已保存的会话历史占位数据 / fetch saved conversation history placeholder.
- `POST /api/feedback`: 提交答案反馈占位数据 / submit answer feedback placeholder.

第一版有意保持为可运行骨架。搜索、LLM、存储和后台任务的边界已经就位，后续可以逐步补全完整的检索和索引行为。

The first version is intentionally a runnable skeleton. Search, LLM, storage, and background task boundaries are in place so the full retrieval and indexing behavior can be filled in incrementally.

## 知识库索引 / Knowledge Index

知识库会抓取指定 URL、提取正文、按段落分块并写入 Postgres；配置了 OpenAI 兼容的嵌入接口后，会同时写入 pgvector 向量。未配置嵌入接口时仍可按关键词检索，便于本地开发和渐进部署。

启动依赖与 worker：

```bash
docker compose up -d postgres redis
uv run celery -A tasks.celery_app worker --loglevel=info
```

提交一篇资料：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/documents \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/guide","game":"Elden Ring","source_type":"wiki"}'
```

用 `GET /api/knowledge/documents?game=Elden%20Ring` 查看索引状态。状态为 `ready` 的资料会在每次问答时优先参与检索，再与实时网页结果去重合并。
