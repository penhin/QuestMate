#!/usr/bin/env bash
set -euo pipefail

# No API server, model provider, Tavily, Redis, or private dataset required.
# Run after any identity/safety/evaluation change; reserve full E2E evaluation
# for a completed batch.
uv run pytest \
  .tests/test_game_confirmation.py \
  .tests/test_evidence_generalization.py \
  .tests/test_evals.py
