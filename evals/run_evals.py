"""Run reproducible black-box evaluations against a QuestMate API instance."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import httpx

try:
    from evals.dataset import dataset_metadata, filter_cases, load_cases
    from evals.scoring import SCORING_SCHEMA_VERSION, evaluate_case, summarize
except ModuleNotFoundError:  # Support `python evals/run_evals.py` from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import dataset_metadata, filter_cases, load_cases
    from scoring import SCORING_SCHEMA_VERSION, evaluate_case, summarize


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases.jsonl"


EVALUATION_MODES = ("discovery", "retrieval")


def error_category(exc: Exception) -> str:
    """Return a stable diagnostic label without retaining request content."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connection_error"
    if isinstance(exc, httpx.RequestError):
        return "network_error"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "response_decode_error"
    return f"runtime_{type(exc).__name__}"


def aggregate_error_categories(results: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(
        str(result.get("error_category", "runtime_unknown"))
        for result in results
        if "error" in result
    ).items()))


def sealed_holdout_report(
    *,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    model: dict[str, Any],
    evaluation_mode: str,
) -> dict[str, Any]:
    """Produce a report safe to share with implementers.

    In particular, do not include questions, expected answers, URLs, case IDs,
    or model responses.  A sealed dataset is only useful while those details
    remain unavailable to the people tuning the agent.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_kind": "sealed_holdout_aggregate",
        "dataset": {
            key: metadata[key]
            for key in (
                "schema_version",
                "sha256",
                "case_count",
                "by_split",
                "by_tier",
                "by_difficulty",
                "by_category",
                "holdout_integrity",
            )
        },
        "model": model,
        "filters": {"mode": evaluation_mode, "split": "holdout"},
        "summary": summary,
    }


def validate_sealed_holdout_run(
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    mode: str,
) -> None:
    """Reject configurations that would make a sealed result misleading."""
    integrity = metadata["holdout_integrity"]
    if not integrity["sealed"] or integrity["refresh_required"]:
        raise ValueError(
            "--sealed-holdout requires a manifest declaring sealed=true and "
            "refresh_required=false"
        )
    if not cases or any(case["split"] != "holdout" for case in cases):
        raise ValueError("--sealed-holdout may run only a non-empty holdout split")
    if mode != "discovery":
        raise ValueError("--sealed-holdout requires --mode discovery")


def evaluation_database_domains(case: dict[str, Any]) -> list[str]:
    """Return only explicit retrieval hints; gold source URLs are scoring data only."""
    domains: list[str] = []
    for raw_domain in case.get("database_domains") or []:
        domain = str(raw_domain).casefold().strip().removeprefix("www.")
        if domain and domain not in domains:
            domains.append(domain)
    return domains[:8]


def evaluation_request_metadata(case: dict[str, Any], mode: str) -> dict[str, Any]:
    """Build request hints without allowing scoring fields to influence discovery."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")

    metadata: dict[str, Any] = {
        "evaluation": True,
        "evaluation_case_id": case["id"],
        "evaluation_mode": mode,
    }
    if mode == "discovery" or case.get("category") == "game_resolution":
        return metadata

    metadata["confirmed_game"] = True
    aliases = list(
        dict.fromkeys(alias.strip() for alias in case.get("game_aliases") or [])
    )[:8]
    database_domains = evaluation_database_domains(case)
    if aliases:
        metadata["game_aliases"] = aliases
    if database_domains:
        metadata["database_domains"] = database_domains
    return metadata


async def run_case(
    client: httpx.AsyncClient,
    api_base_url: str,
    case: dict[str, Any],
    model_config: dict[str, str],
    evaluation_mode: str = "discovery",
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        metadata = evaluation_request_metadata(case, evaluation_mode)
        result = await client.post(
            f"{api_base_url.rstrip('/')}/api/chat",
            json={
                "game": case["game"],
                "question": case["question"],
                "stream": False,
                "metadata": metadata,
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
            "error_category": error_category(exc),
            "evaluation": {"passed": False},
            "latency_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000),
        }


async def main_async(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    cases_path = Path(args.cases)
    all_cases = load_cases(cases_path)
    cases = filter_cases(all_cases, split=args.split, tier=args.tier, category=args.category)
    if args.case_id:
        cases = [case for case in cases if case["id"] == args.case_id]
    if args.limit:
        cases = cases[: args.limit]
    metadata = dataset_metadata(
        cases_path,
        all_cases,
        manifest_path=Path(args.dataset_manifest) if args.dataset_manifest else None,
    )
    if args.sealed_holdout:
        validate_sealed_holdout_run(metadata, cases, args.mode)
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
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_run(case: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await run_case(client, args.api_base_url, case, model_config, args.mode)

        results = list(await asyncio.gather(*(bounded_run(case) for case in cases)))
    summary = summarize(results)
    model = {
        "provider": args.ai_provider or "backend_default",
        "model": args.ai_model or "backend_default",
        "base_url": args.ai_base_url or "backend_default",
        "request_api_key_configured": bool(api_key),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.api_base_url,
        "dataset": metadata,
        "scoring_schema_version": SCORING_SCHEMA_VERSION,
        "model": model,
        "filters": {
            "mode": args.mode,
            "split": args.split,
            "tier": args.tier,
            "category": args.category,
            "case_id": args.case_id,
        },
        "evaluation_scope": {
            "mode": args.mode,
            "answer_cases": (
                "no-hint discovery without request-supplied identity, alias, or database hints; server caches may be warm"
                if args.mode == "discovery"
                else "game identity pre-confirmed; explicit case hints supplied to an opt-in evaluation backend"
            ),
            "game_resolution_cases": "full identity resolution without retrieval hints",
            "gold_source_urls": "used only after the response for scoring",
        },
        "summary": summary,
        "results": results,
    }
    sealed_error_categories: dict[str, int] | None = None
    if args.sealed_holdout:
        sealed_error_categories = aggregate_error_categories(results)
        report = sealed_holdout_report(
            metadata=metadata,
            summary=summary,
            model=model,
            evaluation_mode=args.mode,
        )
        report["error_categories"] = sealed_error_categories
    output = Path(args.output) if args.output else ROOT / "reports" / f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sealed_holdout:
        output.chmod(0o600)
    console_summary = {"report": str(output), **summary}
    if sealed_error_categories is not None:
        console_summary["error_categories"] = sealed_error_categories
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_rate"] >= args.fail_under else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument(
        "--dataset-manifest",
        help="optional integrity manifest; defaults to <cases>.manifest.json when present",
    )
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="maximum concurrent evaluation requests; raise carefully to respect model/search quotas",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("dev", "validation", "holdout"))
    parser.add_argument("--tier", choices=("mainstream", "niche", "safety"))
    parser.add_argument("--category")
    parser.add_argument("--case-id")
    parser.add_argument(
        "--mode",
        choices=EVALUATION_MODES,
        default="discovery",
        help="discovery sends no hints (shared server caches may be warm); retrieval may use explicit case hints",
    )
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument(
        "--sealed-holdout",
        action="store_true",
        help=(
            "require a fresh sealed external holdout and emit an aggregate-only, "
            "owner-readable report; requires --split holdout --mode discovery"
        ),
    )
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
