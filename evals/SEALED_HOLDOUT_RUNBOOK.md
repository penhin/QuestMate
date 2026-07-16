# Sealed holdout runbook

This runbook is for the evaluation owner. Do not run it in the repository
workspace used by Agent implementers, and do not place its data in Git, issue
attachments, CI artifacts, or shared chat.

## One-time setup

Create a private directory on a restricted runner or encrypted volume. Only the
evaluation owner and the CI service account should have read permission. Keep
the private JSONL and its reports there; the application repository contains
only the evaluator code.

The dataset must contain only new `split: "holdout"` cases. Before sealing,
the owner checks that its games, primary source domains, and task/quest chains
do not overlap public dev or validation data. The owner should also record the
author, evidence snapshot date, and intended evaluation window in their private
tracker, not inside the application repository.

## Create and validate the manifest

From the checked-out application repository, while the private dataset remains
outside it:

```bash
umask 077
uv run python evals/create_sealed_manifest.py \
  --cases /secure/questmate-holdout.jsonl

uv run python evals/run_evals.py \
  --cases /secure/questmate-holdout.jsonl \
  --dataset-manifest /secure/questmate-holdout.manifest.json \
  --split holdout --mode discovery --sealed-holdout --dataset-only
```

The second command must succeed before an API run. It proves the manifest hash
matches and that the dataset is eligible to be treated as sealed.

## Run and publish

Start a dedicated evaluation API instance with an isolated database/cache and
the test-only model credentials. Then run:

```bash
uv run python evals/run_evals.py \
  --cases /secure/questmate-holdout.jsonl \
  --dataset-manifest /secure/questmate-holdout.manifest.json \
  --split holdout --mode discovery --sealed-holdout \
  --output /secure/reports/holdout-$(date -u +%Y%m%dT%H%M%SZ).json
```

The output is aggregate-only and owner-readable. Publish only that report's
aggregate scores, stratified scores, latency, source counts, and failure
dimension rates. Do not publish the private dataset, request logs, API logs
containing questions, or a normal evaluator report with `results`.

## Rotation

If an implementer sees an individual question, expected answer, reference URL,
case ID, or response for diagnosis, mark that release contaminated in the
private tracker. Do not alter its manifest to claim it remains sealed; create a
new private dataset release and manifest for the next generalization estimate.
