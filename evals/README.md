# QuestMate Evals

`cases.jsonl` 是黑盒评测集。每行一个用例，字段说明：

- `expected_source_types`：回答需要命中的来源类型。
- `required_terms`：答案或证据片段必须出现的实体词。
- `expected_behavior`：`answer`、`confirmation`、`conservative`、`safe_refusal` 等预期行为。
- `requires_official_versioned_source`：补丁题需要官方且具有版本号或发布日期；否则回答必须保持保守。
- `required_terms`：这些词必须实际出现在答案中，仅出现在检索来源里不算通过。
- `forbidden_terms`：答案中不得出现的错误或高风险表述。
- `require_citations`：要求答案至少包含一个有效的 `[n]` 来源编号；`answer` 类型默认开启。

评测还会拒绝超出来源数量的引用编号，并区分“有官方版本证据的明确回答”和“没有版本证据时的保守回答”。

运行前启动 API，并使用独立的测试数据库，避免评测会话写入本地开发数据。

```bash
uv run python evals/run_evals.py \
  --api-base-url http://127.0.0.1:8000 \
  --fail-under 0.8
```

报告默认写入 `evals/reports/`，该目录不提交。第一次运行产生的分数是基线，而不是发布门槛；先人工审查失败样本，再调整样例和阈值。
