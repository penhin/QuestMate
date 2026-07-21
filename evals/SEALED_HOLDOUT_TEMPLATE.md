# Sealed Holdout Handoff / Sealed Holdout 移交模板

本文件面向独立评测负责人。不得将 sealed 用例、答案、URL、别名或逐题报告加入本仓库。

This file is for the independent evaluation owner. Do not add sealed cases,
answers, URLs, aliases, or per-case reports to this repository.

1. 创建与公开用例在游戏族和来源域名上隔离的私有 JSONL 数据集。
   Create a private JSONL dataset with game-family and source-domain splits disjoint from public cases.
2. 创建同名 manifest，设置 `sealed: true`、`refresh_required: false`，并写入数据集 SHA-256。
   Create its sidecar manifest with `sealed: true`, `refresh_required: false`, and that dataset's SHA-256.
3. 只在受控环境中以 `--sealed-holdout` 运行；仅向实现者返回聚合通过率、引用归属率、错误数和延迟分位数。
   Run only from the controlled environment using `--sealed-holdout`; return aggregate pass rate, citation-grounding rate, error count, and latency percentiles to implementers.
4. 每道需要引用的事实题都应提供 `evidence_terms`，表示必须在被引用证据段中出现的关系锚点；不要以游戏类别或动作词表代替。
   Every citation-required factual case should declare `evidence_terms`: relationship anchors that must occur in cited evidence. Do not substitute game-category or action-word lists.
5. 任一逐题结果、来源 URL 或答案向实现者泄露后，轮换该 holdout。
   Rotate the holdout after any per-case result, source URL, or answer is revealed to an implementer.
