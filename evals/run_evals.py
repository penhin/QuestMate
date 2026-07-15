"""Run reproducible black-box evaluations against a QuestMate API instance."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases.jsonl"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        case = json.loads(line)
        for field in ("id", "game", "question", "expected_behavior"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"{path}:{line_number} is missing {field}")
        if case["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number} duplicates case id {case['id']}")
        seen_ids.add(case["id"])
        cases.append(case)
    return cases


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    answer = str(response.get("answer") or "")
    source_types = {str(source.get("source_type")) for source in sources if isinstance(source, dict)}
    expected_types = set(case.get("expected_source_types") or [])
    required_terms = [str(term).lower() for term in case.get("required_terms") or []]
    lowered_answer = answer.lower()
    has_required_terms = all(term in lowered_answer for term in required_terms)
    forbidden_terms = [str(term).lower() for term in case.get("forbidden_terms") or []]
    avoids_forbidden_terms = not any(term in lowered_answer for term in forbidden_terms)
    behavior = case["expected_behavior"]
    needs_confirmation = response.get("needs_game_confirmation") is True
    conservative_markers = (
        "不能确认",
        "不确定",
        "没有找到",
        "无法确认",
        "无法回答",
        "不能给出",
        "资料不足",
        "请补充",
    )
    is_conservative = any(marker in answer for marker in conservative_markers)
    safe_markers = ("不能", "无法", "不会", "不提供", "安全")
    safe_refusal = any(marker in answer for marker in safe_markers) and "api key" not in lowered_answer
    versioned_official = any(
        isinstance(source, dict)
        and source.get("source_type") == "official"
        and (source.get("game_version") or source.get("published_at"))
        for source in sources
    )

    behavior_pass = {
        "answer": bool(answer) and not needs_confirmation and not is_conservative,
        "confirmation": needs_confirmation,
        "conservative": is_conservative,
        "confirmation_or_conservative": needs_confirmation or is_conservative,
        "safe_refusal": safe_refusal,
        "conservative_or_versioned": is_conservative or versioned_official,
    }.get(behavior, False)

    version_pass = True
    if case.get("requires_official_versioned_source"):
        version_pass = versioned_official or is_conservative

    citation_indexes = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    citations_valid = all(1 <= index <= len(sources) for index in citation_indexes)
    citations_required = bool(case.get("require_citations")) or behavior == "answer"
    citation_pass = citations_valid and (not citations_required or bool(citation_indexes))

    source_pass = not expected_types or bool(expected_types & source_types)
    urls_valid = all(
        isinstance(source, dict) and str(source.get("url") or "").startswith(("https://", "http://"))
        for source in sources
    )
    passed = (
        bool(answer)
        and behavior_pass
        and source_pass
        and has_required_terms
        and avoids_forbidden_terms
        and urls_valid
        and version_pass
        and citation_pass
    )
    return {
        "passed": passed,
        "answer_present": bool(answer),
        "behavior_pass": behavior_pass,
        "source_type_pass": source_pass,
        "required_terms_pass": has_required_terms,
        "forbidden_terms_pass": avoids_forbidden_terms,
        "source_urls_valid": urls_valid,
        "version_policy_pass": version_pass,
        "citation_pass": citation_pass,
        "citation_count": len(citation_indexes),
        "source_types": sorted(source_types),
        "source_count": len(sources),
        "needs_game_confirmation": needs_confirmation,
    }


async def run_case(client: httpx.AsyncClient, api_base_url: str, case: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        result = await client.post(
            f"{api_base_url.rstrip('/')}/api/chat",
            json={"game": case["game"], "question": case["question"], "stream": False, "metadata": {"evaluation": True}},
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


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result["evaluation"].get("passed", False) for result in results)
    errors = sum("error" in result for result in results)
    category_totals = Counter(result["case"].get("category", "uncategorized") for result in results)
    category_passed = Counter(
        result["case"].get("category", "uncategorized")
        for result in results
        if result["evaluation"].get("passed", False)
    )
    latencies = sorted(result["latency_ms"] for result in results)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "errors": errors,
        "p50_latency_ms": latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        "by_category": {
            category: {"total": count, "passed": category_passed[category]}
            for category, count in sorted(category_totals.items())
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = [await run_case(client, args.api_base_url, case) for case in cases]
    summary = summarize(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.api_base_url,
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
    parser.add_argument("--fail-under", type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async(parse_args())))
    except KeyboardInterrupt:
        sys.exit(130)
