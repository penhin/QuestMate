"""Deterministic response scoring and aggregate quality metrics."""

from collections import Counter
import re
from typing import Any


SCORE_DIMENSIONS = (
    "answer_present",
    "behavior_pass",
    "source_type_pass",
    "source_recall_pass",
    "evidence_recall_pass",
    "required_terms_pass",
    "forbidden_terms_pass",
    "source_urls_valid",
    "version_policy_pass",
    "citation_pass",
)


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
    conservative_markers = ("不能确认", "不确定", "没有找到", "无法确认", "无法回答", "不能给出", "资料不足", "请补充")
    leading_answer = answer[:240]
    is_conservative = any(marker in leading_answer for marker in conservative_markers) or (
        len(answer) < 320 and any(marker in answer for marker in conservative_markers)
    )
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
    version_pass = not case.get("requires_official_versioned_source") or versioned_official or is_conservative
    citation_indexes = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    citations_valid = all(1 <= index <= len(sources) for index in citation_indexes)
    citations_required = bool(case.get("require_citations")) or behavior == "answer"
    citation_pass = citations_valid and (not citations_required or bool(citation_indexes))
    source_pass = not expected_types or bool(expected_types & source_types)
    source_urls = [
        str(source.get("url") or "").casefold().rstrip("/")
        for source in sources
        if isinstance(source, dict)
    ]
    expected_source_urls = [
        str(url).casefold().rstrip("/")
        for url in case.get("expected_source_urls") or []
    ]
    source_recall_pass = not expected_source_urls or any(
        expected == actual or expected in actual
        for expected in expected_source_urls
        for actual in source_urls
    )
    evidence_text = " ".join(
        f"{source.get('title') or ''} {source.get('evidence') or source.get('snippet') or ''}"
        for source in sources
        if isinstance(source, dict)
    ).casefold()
    evidence_terms = [str(term).casefold() for term in case.get("evidence_terms") or []]
    evidence_recall_pass = all(term in evidence_text for term in evidence_terms)
    urls_valid = all(
        isinstance(source, dict) and str(source.get("url") or "").startswith(("https://", "http://"))
        for source in sources
    )
    dimensions = {
        "answer_present": bool(answer),
        "behavior_pass": behavior_pass,
        "source_type_pass": source_pass,
        "source_recall_pass": source_recall_pass,
        "evidence_recall_pass": evidence_recall_pass,
        "required_terms_pass": has_required_terms,
        "forbidden_terms_pass": avoids_forbidden_terms,
        "source_urls_valid": urls_valid,
        "version_policy_pass": version_pass,
        "citation_pass": citation_pass,
    }
    return {
        "passed": all(dimensions.values()),
        **dimensions,
        "citation_count": len(citation_indexes),
        "source_types": sorted(source_types),
        "source_count": len(sources),
        "needs_game_confirmation": needs_confirmation,
    }


def _group_summary(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    totals = Counter(result["case"].get(field, "uncategorized") for result in results)
    passed = Counter(
        result["case"].get(field, "uncategorized")
        for result in results
        if result["evaluation"].get("passed", False)
    )
    return {
        value: {
            "total": count,
            "passed": passed[value],
            "pass_rate": round(passed[value] / count, 4),
        }
        for value, count in sorted(totals.items())
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result["evaluation"].get("passed", False) for result in results)
    latencies = sorted(result["latency_ms"] for result in results)
    dimension_rates = {
        dimension: round(
            sum(result["evaluation"].get(dimension, False) for result in results) / total,
            4,
        ) if total else 0
        for dimension in SCORE_DIMENSIONS
    }
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "errors": sum("error" in result for result in results),
        "p50_latency_ms": latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
        "average_source_count": round(
            sum(result["evaluation"].get("source_count", 0) for result in results) / total,
            2,
        ) if total else 0,
        "dimension_pass_rates": dimension_rates,
        "by_category": _group_summary(results, "category"),
        "by_split": _group_summary(results, "split"),
        "by_tier": _group_summary(results, "tier"),
        "by_difficulty": _group_summary(results, "difficulty"),
    }
