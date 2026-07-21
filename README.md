# QuestMate

游戏攻略问答服务，包含 FastAPI 后端和 Tauri 桌面悬浮窗。

A game-guide question-answering service with a FastAPI backend and a Tauri desktop overlay.

## 启动后端 / Run the backend

```bash
cp .env.example .env
docker compose up -d postgres redis
uv sync
uv run uvicorn main:app --reload
```

服务地址为 `http://127.0.0.1:8000`。常用命令：

The service is available at `http://127.0.0.1:8000`. Common commands:

```bash
uv run pytest
uv run celery -A tasks.celery_app worker --loglevel=info
```

## 启动桌面端 / Run the desktop overlay

```bash
cd overlay
npm install
npm run dev
```

桌面端默认请求 `http://127.0.0.1:8000`，可通过 `VITE_API_BASE_URL` 覆盖。

The overlay targets `http://127.0.0.1:8000` by default; override it with
`VITE_API_BASE_URL`.

## 接口 / API

- `GET /health`
- `POST /api/chat`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `PATCH /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`
- `POST /api/feedback`
- `POST /api/knowledge/documents`
- `GET /api/knowledge/documents?game=Elden%20Ring`
- `GET /api/source-registry`

## 索引资料 / Index knowledge sources

先启动 Postgres、Redis 和 Celery worker：

Start Postgres, Redis, and the Celery worker first:

```bash
docker compose up -d postgres redis
uv run celery -A tasks.celery_app worker --loglevel=info
```

提交资料 / Submit a source document:

```bash
export QUESTMATE_ADMIN_TOKEN='replace-with-your-admin-token'
curl -X POST http://127.0.0.1:8000/api/knowledge/documents \
  -H 'Content-Type: application/json' \
  -H "X-QuestMate-Admin-Token: $QUESTMATE_ADMIN_TOKEN" \
  -d '{"url":"https://example.com/guide","game":"Elden Ring","source_type":"wiki","game_version":"1.12"}'
```

生产环境必须设置 `KNOWLEDGE_ADMIN_TOKEN`；知识资料和来源注册表的写入及列表接口
都会校验 `X-QuestMate-Admin-Token`。开发环境未配置 token 时可在本机调用。索引器
仅抓取解析到公网地址的 HTTPS 443 页面，并逐跳重检重定向目标。

Production must set `KNOWLEDGE_ADMIN_TOKEN`; knowledge-document and
source-registry write/list endpoints validate `X-QuestMate-Admin-Token`. Local
development may call them without a configured token. The indexer fetches only
HTTPS port 443 pages resolving to public addresses and revalidates every redirect.

未配置嵌入接口时，知识库使用关键词检索；配置 OpenAI 兼容嵌入接口后会启用
pgvector 语义检索。

Without an embedding endpoint, the knowledge base uses keyword retrieval. An
OpenAI-compatible embedding endpoint enables pgvector semantic retrieval.

## Agent 评测 / Agent evaluation

评测数据与运行方式见 [evals/README.md](evals/README.md)。仓库可公开内容、私有
运行数据与 sealed holdout 的边界见 [REPOSITORY_BOUNDARY.md](REPOSITORY_BOUNDARY.md)。

See [evals/README.md](evals/README.md) for evaluation data and commands. See
[REPOSITORY_BOUNDARY.md](REPOSITORY_BOUNDARY.md) for public/private repository
boundaries and sealed-holdout rules.

## 答案质量链路 / Answer-quality flow

QuestMate 先生成意图化搜索计划并检索本地知识库和实时网页；无冲突的陌生游戏名不会
在检索前被拦截。只有出现竞争候选或证据无法建立可靠对应关系时才请求身份确认。确认的
别名、商店页和 Wiki 域名会写入来源注册表供后续复用。MediaWiki 命中页会被分块入库，
并可沿问题实体匹配的站内链接扩展一层；同一页面默认七天内不重复索引。

QuestMate first creates an intent-aware search plan and searches local knowledge
and the web; an unfamiliar title without competing candidates is not blocked
before retrieval. It requests identity confirmation only when candidates conflict
or evidence cannot establish a reliable match. Confirmed aliases, store pages,
and Wiki domains are recorded for reuse. Matching MediaWiki pages are chunked
and indexed, with one matching internal-link expansion; a page is not re-indexed
within the default seven-day window.

本地知识库与实时网页返回的内容会先作为带来源通道的段落候选进入融合层。同 URL 的互补
段落会合并，而最终排序由最直接回答问题的段落决定；这样宽泛的高分摘要不会覆盖同页的
直接证据。该阶段会记录候选数、融合后页面数和各通道数量，供评测和后续自适应检索使用。

Local knowledge and live-web results first enter a channel-aware passage-fusion
stage. Complementary passages from one URL are retained, while the most direct
passage determines final ordering so a broad high-score excerpt cannot hide a
direct answer on the same page. Candidate, fused-page, and channel counts are
logged for evaluation and later adaptive retrieval.

请求由一个受控编排器依次交给 Identity、Planning、Retrieval/Evidence 和 Answer 四个
专家。专家之间只传递 `GameResolution`、`SearchPlan`、`Source` 与 `InvestigationState`
等结构化 artifact；它们不能自行调用其他专家或扩大搜索/模型预算。编排器会记录不含
查询文本和来源内容的交接计数，便于评测协作路径。

A bounded orchestrator routes requests through Identity, Planning,
Retrieval/Evidence, and Answer specialists. Specialists exchange only typed
artifacts such as `GameResolution`, `SearchPlan`, `Source`, and
`InvestigationState`; they cannot invoke one another or expand search/model
budgets. The orchestrator records aggregate-safe hand-off counts for evaluation.

实时搜索会组合定向来源查询与不含 `site:` 的开放查询，避免固定资料站占满预算。证据
按游戏身份、问题实体、直接支持程度、来源弱先验和版本信息重排；未知站点只要直接
支持问题，也可进入高质量来源池。

Live search combines source-directed queries with open queries without `site:`
so known sites cannot consume the whole budget. Evidence is reranked by game
identity, question entities, directness, weak source priors, and version data;
an unfamiliar site can qualify when it directly supports the question.

模型用结构化证据缺口记录身份、前提、直接答案、前置条件、获取方式、路线、操作顺序、
结果、版本、冲突或语义区别。最多沿最高优先级缺口继续调查两跳，同一 URL 的互补证据段
会合并。代码只控制预算、原始标识符保留和证据边界，不维护特定游戏的词汇映射或答案规则。

The model records structured evidence gaps for identity, prerequisites, direct
answer, acquisition, route, sequence, result, version, conflicts, and semantic
distinctions. It investigates at most two further hops along the highest-priority
gap and merges complementary passages from a URL. Code governs budget, raw
identifier preservation, and evidence boundaries; it does not maintain
game-specific vocabularies or answer rules.

具体事实、地点、任务步骤、数值、版本结论和打法建议应以 `[1]`、`[2]` 等引用关联
返回来源。缺少直接实体证据，或版本问题缺少含日期/版本号的官方来源时，Agent 应保守
回答而非推测。

Concrete facts, locations, quest steps, values, version conclusions, and
strategy advice should be linked to returned sources using `[1]`, `[2]`, and so
on. The Agent responds conservatively when direct entity evidence is absent or
a version question lacks an official dated/versioned source.

## 搜索额度控制 / Search budget controls

实时检索采用渐进链路，避免每题固定执行整批付费搜索：

Live retrieval is progressive rather than running a fixed batch of paid searches:

1. 已识别 Wiki 域名时，优先使用免费 MediaWiki API 直查实体页。
   When a Wiki domain is known, query entity pages through the free MediaWiki API first.
2. 证据仍不足时，Tavily 首轮最多执行 `TAVILY_MAX_QUERIES_PER_REQUEST` 条查询。
   If evidence remains insufficient, the first Tavily wave runs at most `TAVILY_MAX_QUERIES_PER_REQUEST` queries.
3. 关键缺口仍存在时，模型最多追加两跳调查，每跳只展开一条开放 Tavily 查询。
   If a critical gap remains, the model adds at most two investigation hops, each expanding one open Tavily query.
4. 相同搜索结果默认缓存 24 小时；Redis 不可用时回退至进程内缓存。
   Identical search results are cached for 24 hours by default; Redis failure falls back to process-local cache.

默认首轮最多 2 次 Tavily 调用，最多两次缺口调查各追加 1 次，单题内容检索上限为
4 次；缓存命中或 MediaWiki 证据完整时为 0 次。陌生游戏首次身份识别最多 4 次搜索，
随后复用缓存或来源注册表。

The default first wave makes at most two Tavily calls; two gap investigations
may add one each, for a four-call content-retrieval ceiling per question. Cache
hits or complete MediaWiki evidence cost zero calls. Initial identity discovery
for an unfamiliar game may use up to four searches, then reuses cache or registry.

可在 `.env` 中调整 `TAVILY_SEARCH_CACHE_TTL_SECONDS`、`TAVILY_SEARCH_CACHE_MAX_ENTRIES`、
`SEARCH_CACHE_USE_REDIS`、`TAVILY_FIRST_WAVE_QUERIES`、`TAVILY_MAX_QUERIES_PER_REQUEST` 和
`MEDIAWIKI_DIRECT_SEARCH`。`search.usage` 结构化日志会记录每阶段的付费调用、缓存命中
与检索路线。

Configure `TAVILY_SEARCH_CACHE_TTL_SECONDS`, `TAVILY_SEARCH_CACHE_MAX_ENTRIES`,
`SEARCH_CACHE_USE_REDIS`, `TAVILY_FIRST_WAVE_QUERIES`,
`TAVILY_MAX_QUERIES_PER_REQUEST`, and `MEDIAWIKI_DIRECT_SEARCH` in `.env`.
Structured `search.usage` logs record paid calls, cache hits, and retrieval
routes by stage.

## 配置边界 / Configuration boundaries

- `config.py`：部署参数，例如模型凭据、数据库、超时、并发和结果数。
  `config.py`: deployment settings such as model credentials, database, timeouts, concurrency, and result counts.
- `quality_policy.py`：版本化的正常回答质量策略。
  `quality_policy.py`: versioned quality policy for normal answers.
- `overlay/src/config/games.json`：桌面端唯一的游戏进程注册表，由 TypeScript 与 Rust 共享。
  `overlay/src/config/games.json`: the overlay's single game-process registry, shared by TypeScript and Rust.
- 安全提示词、保守回答和无模型/无搜索 fallback 留在各自代码路径，以便审查。
  Security prompts, conservative responses, and no-model/no-search fallbacks stay in their respective code paths for review.

服务端模型密钥只发送到服务器配置的官方端点。用户自带密钥时可直连 DeepSeek 官方端点；
其他自定义 HTTPS 端点必须将精确主机名加入 `CUSTOM_MODEL_ENDPOINT_HOSTS`，不接受通配符、
用户信息或非 443 端口。`ALLOW_EVALUATION_RETRIEVAL_HINTS` 仅供隔离评测实例使用，生产
环境会忽略客户端别名和站点提示。

Server-side model keys are sent only to configured official endpoints. User
keys may use DeepSeek's official endpoint directly; other custom HTTPS endpoints
must add an exact host to `CUSTOM_MODEL_ENDPOINT_HOSTS`, with no wildcards, user
info, or non-443 ports. `ALLOW_EVALUATION_RETRIEVAL_HINTS` is for isolated
evaluation instances only; production ignores client-supplied aliases and site hints.

## 模块边界 / Module boundaries

- `search.py`：检索编排、渐进查询和来源选择 / retrieval orchestration, progressive queries, and source selection.
- `retrieval/pipeline.py`：段落候选融合、去重与最终证据池重排 / passage fusion, deduplication, and final evidence-pool reranking.
- `multi_agent.py`：受控专家 agent 与结构化交接边界 / bounded specialist agents and typed hand-offs.
- `search_cache.py`：内存/Redis 缓存与调用计数 / memory/Redis cache and call accounting.
- `source_registry.py`：别名、商店页和 Wiki 入口持久化 / persistence for aliases, store pages, and Wiki entries.
- `mediawiki_client.py`：免费 MediaWiki API 适配 / free MediaWiki API adapter.
- `llm.py`：模型调用、证据策略和回答流程 / model calls, evidence policy, and answer flow.
- `guide_prompts.py`：稳定系统提示词与回答结构 / stable system prompts and answer structure.
- `evals/dataset.py`、`evals/scoring.py`、`evals/run_evals.py`：评测数据、确定性评分和黑盒执行 / evaluation data, deterministic scoring, and black-box execution.
- `overlay/src/ui.tsx`：桌面端文案、图标和复用 UI；`App.tsx` 保留页面状态与业务交互 / overlay copy, icons, and reusable UI; `App.tsx` owns page state and interactions.
