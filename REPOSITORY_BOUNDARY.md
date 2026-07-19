# Repository Boundary / 仓库对外边界

QuestMate 是可公开审查的应用实现仓库，包含代码、安全示例、公开回归用例和文档；
它不是生产环境、私有知识库或 sealed 评测库。

QuestMate is a publicly reviewable application implementation repository. It
contains code, safe examples, public regression cases, and documentation; it
is not a production environment, private knowledge store, or sealed evaluation
store.

## May be committed and shared / 可提交和共享

- 应用、基础设施、测试和评测代码 / Application, infrastructure, test, and evaluation code.
- 仅含占位符的 `.env.example` / `.env.example` files with placeholders only.
- 公开 `dev`、`validation` 用例、确定性评分器和脱敏聚合基线 / Public `dev` and `validation` cases, deterministic scorers, and redacted aggregate baselines.
- 不含凭据和用户数据的合成/示例资料与配置 / Synthetic or example knowledge documents and configuration without credentials or user data.

## Must stay outside Git / 必须留在 Git 之外

- API Key、Token、Cookie、生产连接串和已填充的 `.env` / API keys, tokens, cookies, production connection strings, and populated `.env` files.
- 用户对话、反馈、数据库导出、缓存、请求日志和生产资料正文 / User conversations, feedback, database exports, caches, request logs, and production source documents.
- Sealed holdout 的题目、答案、来源 URL、别名、Case ID、逐题结果和模型回答 / Sealed-holdout questions, answers, source URLs, aliases, case IDs, per-case results, and model responses.
- 私有来源快照，以及任何包含凭据或受控访问内容的 URL/文件 / Private source snapshots and any URL or artifact containing credentials or access-controlled material.

## Evaluation boundary / 评测边界

公开 `dev` 和 `validation` 数据可用于开发与回归，但不能证明未见游戏的泛化能力。
Sealed holdout 由独立评测负责人在仓库外的受限位置持有和执行；实现者只能收到
聚合分数和失败维度分布。

Public `dev` and `validation` data may be used for development and regression,
but cannot support a claim about unseen-game generalization. An independent
evaluator owns and runs a sealed holdout outside this repository; implementers
receive only aggregate scores and failure-dimension distributions.

如果实现者看到了任一 holdout 题目、标准答案、来源 URL、Case ID 或逐题响应，
该版本即被污染，必须轮换。操作流程见
[`evals/SEALED_HOLDOUT_RUNBOOK.md`](evals/SEALED_HOLDOUT_RUNBOOK.md)。

If an implementer sees a holdout question, expected answer, source URL, case
ID, or per-case response, that release is contaminated and must be rotated.
See [`evals/SEALED_HOLDOUT_RUNBOOK.md`](evals/SEALED_HOLDOUT_RUNBOOK.md).

## Release check / 发布检查

发布分支、版本、Issue 附件或 CI 产物前，确认其中不含密钥、用户数据、sealed
材料、私有来源快照、未脱敏评测报告或运行时存储导出。发布聚合评测结果时，必须
同时声明模型、提交、公开数据集指纹、评测模式和延迟范围。

Before publishing a branch, release, issue attachment, or CI artifact, verify
that it contains no secret, user data, sealed material, private source snapshot,
unredacted evaluator report, or runtime storage export. Aggregate evaluation
results must state the model, commit, public dataset fingerprint, evaluation
mode, and latency scope.
