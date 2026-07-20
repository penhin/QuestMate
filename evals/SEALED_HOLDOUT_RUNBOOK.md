# Sealed holdout runbook / Sealed holdout 运行手册

本手册仅面向评测负责人。不要在 Agent 实现者使用的工作区运行，也不要将数据放入
Git、Issue 附件、CI 产物或共享聊天。

This runbook is for the evaluation owner only. Do not run it in an Agent
implementer's workspace or place its data in Git, issue attachments, CI
artifacts, or shared chat.

## One-time setup / 一次性准备

在受限执行器或加密卷上创建私有目录。仅评测负责人和 CI 服务账号应有读取权限。
私有 JSONL 与报告均保存在该目录；应用仓库只包含评测器代码。

Create a private directory on a restricted runner or encrypted volume. Only
the evaluation owner and CI service account should have read permission. Keep
the private JSONL and reports there; the application repository contains only
evaluator code.

数据集只能包含新的 `split: "holdout"` 用例。封存前，确认游戏、主要来源域名、
任务/机制链均不与公开 dev/validation 重叠。作者、证据快照日期和评测窗口记录在
私有追踪系统中，不写入应用仓库。

The dataset may contain only new `split: "holdout"` cases. Before sealing,
ensure its games, primary source domains, and task/mechanic chains do not
overlap public dev or validation data. Record author, evidence snapshot date,
and evaluation window in a private tracker, not this repository.

## Create and validate the manifest / 创建并校验 manifest

在已检出应用仓库、但私有数据集仍位于仓库外时执行：

From a checked-out application repository, while the private dataset remains
outside it, run:

```bash
umask 077
uv run python evals/create_sealed_manifest.py \
  --cases /secure/questmate-holdout.jsonl

uv run python evals/run_evals.py \
  --cases /secure/questmate-holdout.jsonl \
  --dataset-manifest /secure/questmate-holdout.manifest.json \
  --split holdout --mode discovery --sealed-holdout --dataset-only
```

第二个命令必须在调用 API 前成功：它证明 manifest 哈希匹配，且数据集可被视为 sealed。

The second command must succeed before an API run. It proves that the manifest
hash matches and the dataset is eligible to be treated as sealed.

## Run and publish / 运行与发布

启动使用隔离数据库/缓存和测试专用模型凭据的独立评测 API，再执行：

Start a dedicated evaluation API instance with an isolated database/cache and
test-only model credentials, then run:

当 API 使用服务器 `.env` 内的密钥时，模型由 API 的 `DEEPSEEK_MODEL`（或对应服务端配置）
决定；评测命令的 `--ai-model` 不会覆盖它。只有设置了请求专用的
`QUESTMATE_EVAL_AI_API_KEY` 时，才可在评测命令中传入 `--ai-model` 与 `--ai-base-url`。

When the API uses a server-owned `.env` key, its `DEEPSEEK_MODEL` (or equivalent
server configuration) determines the model; evaluator `--ai-model` does not
override it. Pass `--ai-model` and `--ai-base-url` only when a request-owned
`QUESTMATE_EVAL_AI_API_KEY` is configured.

```bash
uv run python evals/run_evals.py \
  --cases /secure/questmate-holdout.jsonl \
  --dataset-manifest /secure/questmate-holdout.manifest.json \
  --split holdout --mode discovery --sealed-holdout \
  --output /secure/reports/holdout-$(date -u +%Y%m%dT%H%M%SZ).json
```

输出仅为聚合结果且仅负责人可读。除通过率外，报告会给出验收维度失败数及两两共同
失败数；这些数据不会按案例、类别、问题、答案、URL 或响应分组。只发布聚合/分层得分、
延迟、来源数量和失败维度比例；不要发布私有数据集、请求日志、含问题的 API 日志或带
`results` 的常规报告。

`agent_funnel` 同样只包含聚合数据：响应路径、证据等级和是否渲染引用；其中不含问题、
答案、查询、来源、URL 或逐题标识。

The output is aggregate-only and owner-readable. It includes pass rates plus
acceptance-dimension failure counts and pairwise co-failure counts; these are
not grouped by case, category, prompt, answer, URL, or response. Publish only
aggregate and stratified scores, latency, source counts, and failure-dimension rates. Do not
publish the private dataset, request logs, API logs containing questions, or a
normal evaluator report with `results`.

`agent_funnel` is also aggregate-only: response paths, evidence levels, and
rendered-citation presence. It contains no prompt, answer, query, source, URL,
or per-case identifier.
It also groups these counters by expected behavior only, so `answer` and
`safe_refusal` failures can be diagnosed without exposing a case.

## Rotation / 轮换

如果实现者为诊断而看到单题、标准答案、参考 URL、Case ID 或响应，在私有追踪系统
中将该版本标记为已污染。不要修改 manifest 来宣称它仍为 sealed；为下一次泛化估计
创建新的私有数据集版本和 manifest。

If an implementer sees an individual question, expected answer, reference URL,
case ID, or response for diagnosis, mark that release contaminated in the
private tracker. Do not alter its manifest to claim it remains sealed; create
a new private dataset release and manifest for the next generalization estimate.
