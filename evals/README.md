# QuestMate Evals / QuestMate 评测

此目录包含公开的开发/回归评测、确定性评分器和执行脚本；不保存 sealed holdout
或真实运行数据。完整仓库边界见 [`../REPOSITORY_BOUNDARY.md`](../REPOSITORY_BOUNDARY.md)。

This directory contains public development/regression evaluations, deterministic
scorers, and runner scripts. It does not store sealed holdouts or real runtime
data. See [`../REPOSITORY_BOUNDARY.md`](../REPOSITORY_BOUNDARY.md) for the full boundary.

## 数据集与模式 / Datasets and modes

- `dev`：日常调试；`validation`：公开回归。二者可用于调优，但不能证明未见游戏的泛化能力。
- `holdout`：仓库现有条目仅作历史诊断，已不是 sealed 泛化集。
- `discovery`（默认）：只发送游戏名和问题，评估完整发现链路。
- `retrieval`：仅用于隔离评测后端；可注入案例显式声明的别名和资料域名，不能与 discovery 的成绩混合比较。

- `dev` is for daily debugging and `validation` is for public regression; neither proves unseen-game generalization.
- Repository `holdout` entries are historical diagnostics only, not a sealed generalization set.
- `discovery` (default) sends only game and question, exercising the full discovery path.
- `retrieval` is for isolated evaluation backends only. It may inject explicitly declared aliases and source domains, so its scores must not be combined with discovery.

案例以 JSONL 保存。每行至少需要 `id`、`game`、`question` 和 `expected_behavior`；
字段校验、可选评分字段和完整性 manifest 的规则见 [`dataset.py`](dataset.py)。

Cases are JSONL. Each row needs at least `id`, `game`, `question`, and
`expected_behavior`; see [`dataset.py`](dataset.py) for validation, optional
scoring fields, and integrity-manifest rules.

## 常用命令 / Common commands

```bash
# 只检查数据集结构与分布 / Check dataset structure and distribution only
uv run python evals/run_evals.py --dataset-only

# 默认 discovery 回归 / Default discovery regression
uv run python evals/run_evals.py --split validation --mode discovery

# 隔离后端上的 retrieval 诊断 / Retrieval diagnostics on an isolated backend
ALLOW_EVALUATION_RETRIEVAL_HINTS=true uv run uvicorn main:app
uv run python evals/run_evals.py --split validation --mode retrieval
```

如果后端没有默认模型配置，可将测试专用密钥放在受限环境变量
`QUESTMATE_EVAL_AI_API_KEY` 中；密钥不应写入命令、报告或 Git。

If the backend lacks a default model configuration, place a test-only key in
the restricted `QUESTMATE_EVAL_AI_API_KEY` environment variable. Never put a
key in commands, reports, or Git.

报告默认写入已忽略的 `evals/reports/`。提交的基线只能是脱敏聚合摘要；模型、
提交、数据集指纹、评测模式和延迟范围必须一起记录。

Reports default to ignored `evals/reports/`. Committed baselines may contain
only redacted aggregates and must record the model, commit, dataset fingerprint,
evaluation mode, and latency scope.

## 质量与成本契约 / Quality and cost contract

sealed `discovery` 的验收门槛为：总体通过率至少 80%，需引用题的引用归属率至少
85%，正常攻略题的身份确认率不高于 15%，安全 tier 行为通过率为 100%，且 p95 不高于
30 秒。简单题最多两次模型调用；复杂证据路径最多三次；每题付费搜索最多四次。

The sealed `discovery` acceptance gates are: at least 80% overall pass rate,
at least 85% citation grounding among citation-required cases, at most 15%
confirmation for normal guide cases, 100% safety-tier behavior, and p95 at or
below 30 seconds. Simple cases use at most two model calls, complex evidence
paths at most three, and each request uses at most four paid searches.

`--enforce-contract` makes the runner return non-zero when any aggregate gate
fails. Use `--environment-id` to record a non-sensitive isolated-runner label.
Sealed reports include cohorts, resource aggregates, service commit, and
environment label only; they never contain per-case data.

## Sealed holdout / 密封留出集

新的 sealed holdout 必须由独立评测负责人维护在仓库外的受限位置，使用
`holdout` + `discovery` 运行，并且只向实现者提供聚合指标。创建 manifest、
权限、执行和轮换流程见：

- [移交模板 / Handoff template](SEALED_HOLDOUT_TEMPLATE.md)
- [受限运行手册 / Restricted runbook](SEALED_HOLDOUT_RUNBOOK.md)

一旦实现者看到了任一逐题题目、答案、来源 URL、case ID 或响应，该版本 holdout
即被污染，必须轮换。

A new sealed holdout must be maintained outside this repository by an
independent evaluation owner. Run it with `holdout` + `discovery` and provide
implementers aggregate metrics only. Any disclosed per-case question, answer,
URL, case ID, or response contaminates that release and requires rotation.

## 格式参考 / Format reference

[`examples/holdout_format_reference.jsonl`](examples/holdout_format_reference.jsonl)
包含五道全虚构的 JSONL 示例，展示常规回答、物品说明、版本题、未知实体和身份确认
等字段组合。该文件仅用于格式参考：它不由默认评测入口读取，不属于 dev、validation
或有效 sealed holdout，也不得用于通过率、延迟或泛化结论。

[`examples/holdout_format_reference.jsonl`](examples/holdout_format_reference.jsonl)
contains five fully fictional JSONL examples covering ordinary answers, item
usage, version-sensitive questions, unknown entities, and identity resolution.
It is format reference only: the default evaluator does not read it, it belongs
to neither dev nor validation nor a valid sealed holdout, and it must not be
used for pass-rate, latency, or generalization claims.
