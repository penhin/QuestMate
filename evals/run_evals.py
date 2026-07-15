"""Run reproducible black-box evaluations against a QuestMate API instance."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import httpx

try:
    from evals.dataset import dataset_metadata, filter_cases, load_cases
    from evals.scoring import evaluate_case, summarize
except ModuleNotFoundError:  # Support `python evals/run_evals.py` from the repository root.
    from dataset import dataset_metadata, filter_cases, load_cases
    from scoring import evaluate_case, summarize


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases.jsonl"


async def run_case(
    client: httpx.AsyncClient,
    api_base_url: str,
    case: dict[str, Any],
    model_config: dict[str, str],
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        result = await client.post(
            f"{api_base_url.rstrip('/')}/api/chat",
            json={
                "game": case["game"],
                "question": case["question"],
                "stream": False,
                "metadata": {"evaluation": True, "evaluation_case_id": case["id"]},
                **model_config,
            },
        )
        result.raise_for_status()
        response = result.json()
        evaluation = evaluate_case(case, response)
        return {
            "case": case,
            "response": response,
            "evaluation": evaluation,
            "latency_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        }
    except Exception as exc:
        return {
            "case": case,
            "error": str(exc),
            "evaluation": {"passed": False},
            "latency_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        }


async def main_async(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    all_cases = load_cases(cases_path)
    cases = filter_cases(all_cases, split=args.split, tier=args.tier, category=args.category)
    if args.limit:
        cases = cases[: args.limit]
    metadata = dataset_metadata(cases_path, all_cases)
    if args.dataset_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    api_key = os.getenv("QUESTMATE_EVAL_AI_API_KEY", "").strip()
    model_config = {
        key: value
        for key, value in {
            "ai_provider": args.ai_provider,
            "ai_api_key": api_key,
            "ai_model": args.ai_model,
            "ai_base_url": args.ai_base_url,
        }.items()
        if value
    }
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = [await run_case(client, args.api_base_url, case, model_config) for case in cases]
    summary = summarize(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.api_base_url,
        "dataset": metadata,
        "model": {
            "provider": args.ai_provider or "backend_default",
            "model": args.ai_model or "backend_default",
            "base_url": args.ai_base_url or "backend_default",
            "request_api_key_configured": bool(api_key),
        },
        "filters": {"split": args.split, "tier": args.tier, "category": args.category},
        "summary": summary,
        "results": results,
    }
    output = Path(args.output) if args.output else ROOT / "reports" / f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), **summary}, ensure_ascii=False, indent=2))
    return 0 if summary["pass_rate"] >= args.fail_under else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("dev", "validation"))
    parser.add_argument("--tier", choices=("mainstream", "niche", "safety"))
    parser.add_argument("--category")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--ai-provider", choices=("anthropic", "deepseek"))
    parser.add_argument("--ai-model")
    parser.add_argument("--ai-base-url")
    parser.add_argument("--fail-under", type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async(parse_args())))
    except KeyboardInterrupt:
        sys.exit(130)
