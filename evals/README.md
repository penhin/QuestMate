# QuestMate Evals

`cases.jsonl` 是黑盒评测集。每行一个用例，字段说明：

- `split`：`dev` 用于日常调试，`validation` 用于常规回归；`holdout` 按整个游戏留出，禁止用于逐题调参，只用于检查跨游戏泛化。
- `tier`：`mainstream`、`niche` 或 `safety`，分别衡量主流游戏、冷门游戏和安全/边界行为。
- `difficulty`：`standard` 或 `hard`。
- `expected_source_types`：回答需要命中的来源类型。
- `required_terms`：答案或证据片段必须出现的实体词。
- `expected_behavior`：`answer`、`confirmation`、`conservative`、`safe_refusal` 等预期行为。
- `requires_official_versioned_source`：补丁题需要官方且具有版本号或发布日期；否则回答必须保持保守。
- `required_terms`：这些词必须实际出现在答案中，仅出现在检索来源里不算通过。
- `forbidden_terms`：答案中不得出现的错误或高风险表述。
- `expected_source_urls`：可接受的正确来源 URL；返回来源至少命中一个，用于衡量 Source Recall。
- `evidence_terms`：必须出现在返回来源正文片段中的证据词，用于衡量 Evidence Recall。
- `required_answer_groups`：行动链答案必须覆盖的概念组；每组命中任一中英文表达即通过。
- `require_citations`：要求答案至少包含一个有效的 `[n]` 来源编号；`answer` 类型默认开启。

评测还会拒绝超出来源数量的引用编号，并区分“有官方版本证据的明确回答”和“没有版本证据时的保守回答”。报告会分别输出评分维度通过率，以及按类别、数据集、游戏层级和难度分组的结果。

当前基线定义包含 52 个案例，并包含与开发/验证游戏完全不重叠的留出游戏，其中有 21 个冷门游戏案例和 6 个安全/边界案例。`baseline_definition.json` 固定数据集指纹和评测口径；首次使用固定模型跑出的完整报告才是性能基线。

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
  --fail-under 0.8
```

运行指定切片：

```bash
uv run python evals/run_evals.py --split dev --tier niche
uv run python evals/run_evals.py --split validation
uv run python evals/run_evals.py --split holdout
```

如果后端没有配置默认模型，可通过环境变量传入测试专用密钥。密钥不会写入报告：

```bash
export QUESTMATE_EVAL_AI_API_KEY='你的测试密钥'
uv run python evals/run_evals.py \
  --ai-provider deepseek \
  --ai-model deepseek-chat \
  --ai-base-url https://api.deepseek.com
```

报告默认写入 `evals/reports/`，该目录不提交。第一次固定模型、模型版本和检索配置的完整运行结果是性能基线，而不是发布门槛；先人工审查失败样本，再调整阈值。模型、数据集指纹和筛选条件会写入报告，API Key 不会写入。

为了把答案质量与游戏识别能力分开，除 `game_resolution` 类别外，评测请求会将游戏身份标记为已确认；`game_resolution` 案例仍走完整身份识别链路。这样可以减少重复付费搜索，也能明确区分“游戏没认出来”和“证据/答案质量不足”。
