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

## Agent 评测

评测集与运行方式见 [evals/README.md](evals/README.md)。

## 答案质量链路

QuestMate 会先确认游戏身份并生成意图化搜索计划，再检索本地知识库与实时网页。实时搜索会抽取正文中与问题实体最相关的证据段；本地和网页证据随后按实体覆盖、检索分、来源可信度与版本信息统一重排。

回答中的具体事实、地点、任务步骤、数值、版本结论和打法建议应使用 `[1]`、`[2]` 等编号关联返回的来源。没有直接实体证据，或版本问题缺少带日期/版本号的官方来源时，Agent 会返回保守答案而不是推测。

## 搜索额度控制

实时检索采用渐进链路，避免每个问题固定执行整批付费搜索：

1. 已识别出游戏 Wiki 域名时，优先通过 MediaWiki API 免费直查实体页。
2. 仍缺证据时，Tavily 首批只执行 `TAVILY_FIRST_WAVE_QUERIES` 条查询；严格相关来源不足，再扩展到 `TAVILY_MAX_QUERIES_PER_REQUEST` 的单请求上限。
3. 相同搜索结果默认缓存 24 小时。Redis 可用时跨进程和重启复用；不可用时自动退回进程内缓存。

默认配置下，问题检索首批最多调用 2 次 Tavily，证据不足才扩展到最多 4 次；缓存命中或 MediaWiki 已提供强相关证据时为 0 次。首次识别陌生游戏可能另外产生身份检索调用，后续会命中同一缓存。

可通过 `.env` 调整 `TAVILY_SEARCH_CACHE_TTL_SECONDS`、`TAVILY_SEARCH_CACHE_MAX_ENTRIES`、`SEARCH_CACHE_USE_REDIS`、`TAVILY_FIRST_WAVE_QUERIES`、`TAVILY_MAX_QUERIES_PER_REQUEST` 和 `MEDIAWIKI_DIRECT_SEARCH`。后端的 `search.usage` 结构化日志会记录每阶段的 `tavily_paid_calls`、`tavily_cache_hits` 和检索路线，便于按请求审计额度。

## 配置边界

- `config.py`：部署环境参数，例如模型凭据、数据库、超时、并发量和结果数量。
- `quality_policy.py`：正常回答持续使用的版本化质量策略，例如来源可信度、排序权重、域名质量、版本敏感意图和游戏识别阈值。
- `overlay/src/config/games.json`：桌面端唯一的游戏进程注册表，由 TypeScript 前端和 Rust Windows 后端共同读取。
- 安全提示词、保守回答以及无模型/无搜索时的 fallback 保留在对应代码路径，便于审查行为变化。
