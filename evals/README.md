# QuestMate Evals

`cases.jsonl` 是黑盒评测集。每行一个用例，字段说明：

- `split`：`dev` 用于日常调试，`validation` 用于常规回归；现有 `holdout` 已在实现过程中被查看，只能作为历史回归切片，不能再估计未见游戏的泛化能力。
- `tier`：`mainstream`、`niche` 或 `safety`，分别衡量主流游戏、冷门游戏和安全/边界行为。
- `difficulty`：`standard` 或 `hard`。
- `expected_source_types`：期望命中的来源类型，仅作为召回诊断，不是普通答案的通过门槛。
- `required_terms`：必须实际出现在答案中的关键实体词或结论词。
- `expected_behavior`：`answer`、`confirmation`、`conservative`、`safe_refusal` 等预期行为。
- `requires_official_versioned_source`：补丁题需要官方且具有版本号或发布日期；否则回答必须保持保守。
- `forbidden_terms`：答案中不得出现的错误或高风险表述。
- `expected_source_urls`：可接受的参考来源 URL，用于诊断 Source Recall；其他来源只要证据和答案正确，也不会仅因 URL 不同而失败。
- `database_domains`：可选的已知资料库主机名，只在 `retrieval` 模式注入请求；必须是显式字段，不能由标准答案 URL 推导。
- `game_aliases`：可选的游戏别名，只在 `retrieval` 模式注入请求。
- `evidence_terms`：必须出现在返回来源正文片段中的证据词，用于衡量 Evidence Recall。
- `required_answer_groups`：行动链答案必须覆盖的概念组；每组命中任一中英文表达即通过。
- `require_citations`：要求答案至少包含一个有效的 `[n]` 来源编号；`answer` 类型以及非保守的版本结论默认开启。
- `version_sensitive`：要求非保守答案引用一条与当前问题相关、且带版本号或发布日期/补丁上下文的来源；不限定必须是官方来源。

评测仍会把证据词、关键行动链、必需/禁止表述、有效引用、引用证据关联和版本策略作为通过门槛。普通明确回答所引用的来源必须实际包含声明的 `evidence_terms`、`required_terms`，或至少包含问题中的实体锚点；任意 URL 加 `[1]` 不会通过。非保守的补丁/版本结论必须引用相关版本来源，带日期但内容无关的官方页面不算版本证据。`source_type_pass` 和 `source_recall_pass` 只诊断是否命中策展时记录的参考路线；它们不会否定由其他可靠页面支持的正确答案。报告会分别输出所有维度通过率，以及按类别、数据集、游戏层级和难度分组的结果。

`expected_source_urls` 和 `evidence_terms` 只在后端返回回答后参与评分，永远不会成为请求元数据。评测入口有两种明确模式：

- `discovery`（默认）：无请求提示的发现模式；只发送游戏名和问题，不注入 `confirmed_game`、别名或资料库域名。连续案例仍会共享后端来源注册表和缓存，因此这不是严格的逐案例冷进程测试。
- `retrieval`：隔离资料检索与回答能力；除 `game_resolution` 案例外会确认游戏身份，并且只允许使用案例中显式声明的 `database_domains` 和 `game_aliases`。后端默认拒绝这些不可信元数据；只能在隔离评测实例中显式设置 `ALLOW_EVALUATION_RETRIEVAL_HINTS=true`，生产环境即使误设也不会接收。

两种模式衡量的能力不同，报告和基线必须分别记录，不能直接比较通过率。

当前基线定义包含 52 个案例，其中有 21 个冷门游戏案例和 6 个安全/边界案例。原留出案例及答案结构已经在本轮开发中被查看；其中与新规则测试重合的 Goose Goose Duck 案例已移入 `validation`，其余 `holdout` 也统一标记为 `contaminated_refresh_required`，仅保留历史诊断用途。发布前需要在开发流程之外建立一个新的、版本化且不向实现者暴露的 sealed holdout，当前仓库不能诚实地提供未见样本成绩。

holdout 完整性来自与数据集同名的 sidecar manifest：例如 `cases.jsonl` 自动读取 `cases.manifest.json`。当前仓库的 manifest 明确记录污染状态；没有 manifest 的外部数据集会标为 `unverified`，不会被误报为当前仓库的 `contaminated`。外部 sealed 数据集应提供自己的 manifest，也可用 `--dataset-manifest /path/to/manifest.json` 显式指定。manifest 必须用 `dataset_sha256` 绑定对应数据文件，且其中的 `holdout_integrity` 必须包含 `status`、`sealed`、`refresh_required` 和 `usage`。

已提交的脱敏性能摘要保存在 `baselines/`。摘要只包含评分、延迟、来源数量和失败维度，不保存 API Key 或完整模型回答。
扩充或修改案例后，`baseline_definition.json` 会标记需要刷新性能基线；旧摘要仅作为历史对照，不能代表新数据集成绩。

只检查数据集结构和覆盖分布，不调用模型或搜索服务：

```bash
uv run python evals/run_evals.py --dataset-only
```

运行前启动 API，并使用独立的测试数据库，避免评测会话写入本地开发数据。

```bash
uv run python evals/run_evals.py \
  --api-base-url http://127.0.0.1:8000 \
  --mode discovery \
  --fail-under 0.8
```

在已知游戏身份/资料库的条件下单独测试检索与回答（需启动隔离后端）：

```bash
ALLOW_EVALUATION_RETRIEVAL_HINTS=true uv run uvicorn main:app
uv run python evals/run_evals.py --mode retrieval
```

运行指定切片：

```bash
uv run python evals/run_evals.py --split dev --tier niche
uv run python evals/run_evals.py --split validation
uv run python evals/run_evals.py --split holdout  # 仅历史回归，不是 sealed 泛化成绩
```

如果后端没有配置默认模型，可通过环境变量传入测试专用密钥。密钥不会写入报告：

```bash
export QUESTMATE_EVAL_AI_API_KEY='你的测试密钥'
uv run python evals/run_evals.py \
  --ai-provider deepseek \
  --ai-model deepseek-chat \
  --ai-base-url https://api.deepseek.com
```

报告默认写入 `evals/reports/`，该目录不提交。第一次固定模型、模型版本和检索配置的完整运行结果是性能基线，而不是发布门槛；先人工审查失败样本，再调整阈值。模型、数据集指纹、筛选条件以及 holdout 完整性状态都会写入报告，API Key 不会写入。

需要分析失败来源时，先运行默认的 `discovery` 基线，再运行相同筛选条件的 `retrieval` 模式。只有前者失败时，多半是游戏/资料入口发现问题；两者都失败时，才更可能是页面召回、证据定位或答案生成问题。`game_resolution` 案例在两种模式下都走完整身份识别链路。若需要严格冷启动延迟或首次发现率，应为每个案例使用空来源注册表、空缓存和独立后端进程，不能把普通 `discovery` 报告当作该指标。
