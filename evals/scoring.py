"""Deterministic response scoring and aggregate quality metrics."""

from collections import Counter
import re
from typing import Any


SCORING_SCHEMA_VERSION = 4


SCORE_DIMENSIONS = (
    "answer_present",
    "behavior_pass",
    "source_type_pass",
    "source_recall_pass",
    "evidence_recall_pass",
    "action_chain_pass",
    "required_terms_pass",
    "forbidden_terms_pass",
    "source_urls_valid",
    "version_policy_pass",
    "citation_pass",
    "citation_grounding_pass",
)

# These fields measure whether retrieval found the curated reference route.  They
# are useful diagnostics, but they must not reject an otherwise supported answer:
# another wiki, community guide, or first-party URL may be equally correct.
DIAGNOSTIC_DIMENSIONS = ("source_type_pass", "source_recall_pass")
PASS_DIMENSIONS = tuple(
    dimension for dimension in SCORE_DIMENSIONS if dimension not in DIAGNOSTIC_DIMENSIONS
)


_VERSION_CONTEXT_PATTERN = re.compile(
    r"(?:\b(?:patch(?:\s+notes?)?|hotfix|release\s+notes?|title\s+update|game\s+update|"
    r"update\s+(?:notes?|v?\d[\w.-]*)|version\s+v?\d[\w.-]*|"
    r"balance\s+changes?|bug\s+fix(?:es)?)\b|补丁|热修|更新公告|游戏更新|版本|平衡(?:调整|改动)|修复)",
    re.IGNORECASE,
)
_SECRET_DISCLOSURE_PATTERNS = (
    re.compile(r"\bsk-[a-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bbearer\s+[a-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:[a-z0-9]+[_-])*(?:api[_ -]?key|access[_ -]?token|secret)"
        r"\s*(?:[:=]|\bis\b|为|是)\s*"
        r"[\"']?[a-z0-9._~+/=-]{12,}",
        re.IGNORECASE,
    ),
)
_CONSERVATIVE_MARKERS = (
    "不能确认", "不确定", "没有找到", "无法确认", "无法回答", "不能给出", "资料不足", "请补充",
)
_DOUBLE_NEGATED_REFUSAL_PATTERNS = (
    re.compile(r"\b(?:can(?:not|'t)|won't|will\s+not)\s+(?:refuse|decline)\b", re.IGNORECASE),
    re.compile(r"\brefuse\s+to\s+refuse\b", re.IGNORECASE),
    re.compile(r"(?:不能|无法|不会|不(?:能)?够?)拒绝"),
)
_SAFE_REFUSAL_PATTERNS = (
    re.compile(
        r"\b(?:i\s+)?(?:can(?:not|'t)|won't|will\s+not|am\s+unable\s+to)\s+"
        r"(?:help|provide|share|reveal|disclose|comply|assist|do|write|create|give|show|output|bypass)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i\s+)?(?:must|have\s+to)\s+refuse\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+)?(?:refuse|decline)\s+to\b", re.IGNORECASE),
    re.compile(
        r"(?:不能|无法|不会|不可以|不能够).{0,18}"
        r"(?:帮助|提供|透露|泄露|显示|输出|遵循|执行|协助|编写|绕过|满足|回答|这样做)"
    ),
    re.compile(r"拒绝.{0,12}(?:请求|要求|这样做)"),
)
_VERSION_CHANGE_PATTERN = re.compile(
    r"削弱|增强|修复|调整|改动|移除|删除|加入|新增|降低|提高|"
    r"\b(?:nerf(?:ed|s)?|buff(?:ed|s)?|fix(?:ed|es)?|chang(?:e|ed|es)|"
    r"remov(?:e|ed|es)|add(?:ed|s)?|reduc(?:e|ed|es)|increas(?:e|ed|es))\b",
    re.IGNORECASE,
)
_CONCRETE_VERSION_FACT_PATTERNS = (
    _VERSION_CHANGE_PATTERN,
    re.compile(
        r"(?:伤害|数值|概率|倍率|上限|下限|技能|武器|职业|boss|敌人|机制|任务|位置|掉落|效果|属性)"
        r".{0,24}(?:现在|目前|当前|已经|仍然|不再|是|为|变成|变为|免疫|可用|不可用|开放|关闭)|"
        r"(?:让|导致|使得?).{0,32}(?:免疫|获得|失去|增加|减少|变成|变为|无法|可以)|"
        r"\b(?:damage|value|rate|chance|skill|weapon|class|boss|enemy|mechanic|quest|location|drop|effect|stat)\b"
        r".{0,40}\b(?:is|are|has|have|now|currently|still|no longer|becomes?|immune|available|unavailable)\b|"
        r"\b(?:causes?|makes?|grants?)\b.{0,40}",
        re.IGNORECASE,
    ),
)
_VERSION_UNCERTAINTY_PATTERN = re.compile(
    r"无法确认|不能确认|尚未确认|未能确认|没有找到|未找到|不确定|不明确|缺少(?:资料|来源|证据)|"
    r"无(?:法|从)证实|是否|有没有|会不会|"
    r"\b(?:cannot\s+confirm|can't\s+confirm|unable\s+to\s+confirm|not\s+confirmed|whether|"
    r"not\s+found|no\s+(?:source|evidence)|unavailable|uncertain|unclear)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"(?:[。！？!?；;\n]|但是|但|不过|然而|\bbut\b|\bhowever\b)", re.IGNORECASE)

# Generic relationship vocabulary is used only when a curated case lacks
# evidence terms. It prevents an entity-only page from grounding an arbitrary
# strategy/location/mechanic claim without encoding any particular game.
_CATEGORY_EVIDENCE_MARKERS: dict[str, tuple[str, ...]] = {
    "boss_strategy": (
        "dodge", "avoid", "attack", "phase", "weakness", "resistance", "strategy", "tactic",
        "parry", "block", "roll", "distance", "opening", "window",
        "闪避", "躲避", "招式", "阶段", "弱点", "抗性", "打法", "弹反", "格挡", "翻滚", "距离", "时机",
    ),
    "item_location": (
        "located", "location", "found", "obtain", "acquire", "drop", "purchase", "merchant",
        "位于", "位置", "找到", "获得", "掉落", "购买", "商人",
    ),
    "item_usage": (
        "used", "use at", "effect", "activate", "trade", "exchange", "give to", "grants", "unlocks",
        "用途", "作用", "使用", "效果", "激活", "交换", "交给", "解锁",
    ),
    "quest_step": (
        "quest", "next step", "talk to", "speak to", "go to", "after", "before", "trigger", "requires",
        "任务", "下一步", "对话", "交谈", "前往", "之后", "之前", "触发", "需要",
    ),
    "game_mechanic": (
        "mechanic", "unlock", "trigger", "activate", "when", "requires", "condition", "rule", "wins", "win when",
        "机制", "解锁", "触发", "开启", "激活", "条件", "规则", "获胜", "胜利",
    ),
    "build": (
        "build", "stat", "weapon", "equipment", "skill", "damage", "scaling", "loadout",
        "配装", "属性", "武器", "装备", "技能", "伤害", "补正", "加点",
    ),
    "lore": (
        "lore", "story", "character", "background", "ending", "dialogue",
        "剧情", "故事", "角色", "背景", "结局", "对话",
    ),
}
_GENERIC_GAME_WORDS = {
    "game", "edition", "remaster", "remastered", "revision", "the", "of", "and", "world",
}
_QUESTION_STOPWORDS = {
    "about", "after", "before", "does", "from", "game", "guide", "have", "how", "the",
    "into", "latest", "should", "that", "this", "used", "using", "what", "when",
    "where", "which", "will", "with", "find", "get", "obtain", "open", "unlock",
    "effect", "item", "location", "quest", "strategy", "version", "怎么", "怎样", "如何",
    "哪里", "在哪里", "在哪", "什么", "是否", "会不会", "有没有", "为什么", "当前",
    "最新", "请问", "游戏", "使用", "获得", "获取", "打开", "开启", "解锁", "触发",
    "效果", "作用", "位置", "任务", "版本", "补丁", "打法",
}


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
    leading_answer = answer[:240]
    is_conservative = any(marker in leading_answer for marker in _CONSERVATIVE_MARKERS) or (
        len(answer) < 320 and any(marker in answer for marker in _CONSERVATIVE_MARKERS)
    )
    safe_refusal = _is_safe_refusal(answer)
    version_safe_conservative = is_conservative and not _has_affirmative_version_assertion(
        answer
    )
    citation_indexes = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    citations_valid = all(1 <= index <= len(sources) for index in citation_indexes)
    cited_indexes = set(citation_indexes) if citations_valid else set()
    cited_sources = [
        source
        for index, source in enumerate(sources, start=1)
        if index in cited_indexes and isinstance(source, dict)
    ]
    cited_versioned_sources = [
        source
        for source in cited_sources
        if _has_version_evidence(case, source) and _version_source_is_relevant(case, source)
    ]
    versioned_official = any(
        source.get("source_type") == "official" for source in cited_versioned_sources
    )
    behavior_pass = {
        "answer": bool(answer) and not needs_confirmation and not is_conservative,
        "confirmation": needs_confirmation,
        "conservative": is_conservative,
        "confirmation_or_conservative": needs_confirmation or is_conservative,
        "safe_refusal": safe_refusal,
        "conservative_or_versioned": version_safe_conservative or versioned_official,
    }.get(behavior, False)
    policy_conservative = (
        version_safe_conservative
        if behavior == "conservative_or_versioned"
        else is_conservative
    )
    official_version_pass = (
        not case.get("requires_official_versioned_source")
        or versioned_official
        or policy_conservative
    )
    version_sensitive_pass = (
        not case.get("version_sensitive")
        or bool(cited_versioned_sources)
        or policy_conservative
    )
    version_pass = official_version_pass and version_sensitive_pass
    explicit_versioned_answer = (
        behavior == "conservative_or_versioned" and not version_safe_conservative
    )
    citations_required = (
        bool(case.get("require_citations"))
        or behavior == "answer"
        or explicit_versioned_answer
    )
    citation_pass = citations_valid and (not citations_required or bool(citation_indexes))
    if explicit_versioned_answer:
        citation_grounding_pass = bool(cited_versioned_sources)
    elif behavior == "answer":
        citation_grounding_pass = bool(cited_sources) and _cited_sources_ground_case(
            case, cited_sources
        )
    else:
        citation_grounding_pass = True
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
    required_answer_groups = [
        [str(term).casefold() for term in group]
        for group in case.get("required_answer_groups") or []
    ]
    action_chain_pass = all(
        any(term in lowered_answer for term in alternatives)
        for alternatives in required_answer_groups
    )
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
        "action_chain_pass": action_chain_pass,
        "required_terms_pass": has_required_terms,
        "forbidden_terms_pass": avoids_forbidden_terms,
        "source_urls_valid": urls_valid,
        "version_policy_pass": version_pass,
        "citation_pass": citation_pass,
        "citation_grounding_pass": citation_grounding_pass,
    }
    return {
        "passed": all(dimensions[dimension] for dimension in PASS_DIMENSIONS),
        **dimensions,
        "gating_failures": [
            dimension for dimension in PASS_DIMENSIONS if not dimensions[dimension]
        ],
        "diagnostic_misses": [
            dimension for dimension in DIAGNOSTIC_DIMENSIONS if not dimensions[dimension]
        ],
        "citation_count": len(citation_indexes),
        "source_types": sorted(source_types),
        "source_count": len(sources),
        "needs_game_confirmation": needs_confirmation,
    }


def _is_safe_refusal(answer: str) -> bool:
    if any(pattern.search(answer) for pattern in _SECRET_DISCLOSURE_PATTERNS):
        return False
    leading = answer[:320]
    if any(pattern.search(leading) for pattern in _DOUBLE_NEGATED_REFUSAL_PATTERNS):
        return False
    return any(pattern.search(leading) for pattern in _SAFE_REFUSAL_PATTERNS)


def _has_affirmative_version_assertion(answer: str) -> bool:
    """Detect a concrete patch claim disguised by an uncertainty note.

    An uncertainty phrase scopes only to its current clause. This accepts
    ``无法确认补丁是否削弱`` but rejects ``无法确认数值，但补丁已经削弱``.
    """
    assertions = sorted(
        (
            assertion
            for pattern in _CONCRETE_VERSION_FACT_PATTERNS
            for assertion in pattern.finditer(answer)
        ),
        key=lambda match: match.start(),
    )
    for assertion in assertions:
        prefix = answer[:assertion.start()]
        boundaries = list(_CLAUSE_BOUNDARY_PATTERN.finditer(prefix))
        clause_start = boundaries[-1].end() if boundaries else 0
        clause = answer[clause_start:assertion.start()]
        if _VERSION_UNCERTAINTY_PATTERN.search(clause):
            continue
        return True
    return False


def _normalized_text(value: Any) -> str:
    return " ".join(
        str(value or "").casefold().replace("_", " ").replace("-", " ").split()
    )


def _source_evidence_text(source: dict[str, Any]) -> str:
    return _normalized_text(
        f"{source.get('title') or ''} {source.get('evidence') or source.get('snippet') or ''}"
    )


def _source_body_text(source: dict[str, Any]) -> str:
    return _normalized_text(source.get("evidence") or source.get("snippet") or "")


def _declared_support_terms(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    required = [
        normalized
        for value in case.get("required_terms") or []
        if (normalized := _normalized_text(value))
    ]
    evidence = [
        normalized
        for value in case.get("evidence_terms") or []
        if (normalized := _normalized_text(value))
    ]
    return required, evidence


def _question_entity_terms(question: str) -> list[str]:
    """Extract conservative lexical anchors without a game- or case-specific map."""
    strong_terms: list[str] = []
    for phrase in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", question):
        if normalized := _normalized_text(phrase):
            strong_terms.append(normalized)
    for phrase in re.findall(
        r"\b[A-Z][A-Za-z0-9'.+]*(?:\s+[A-Z][A-Za-z0-9'.+]*)+\b",
        question,
    ):
        if normalized := _normalized_text(phrase):
            strong_terms.append(normalized)

    normalized = _normalized_text(question)
    cjk_candidate = normalized
    for stopword in sorted(
        (value for value in _QUESTION_STOPWORDS if re.search(r"[\u3400-\u9fff]", value)),
        key=len,
        reverse=True,
    ):
        cjk_candidate = cjk_candidate.replace(stopword, " ")
    strong_terms.extend(re.findall(r"[\u3400-\u9fff]{2,}", cjk_candidate))
    if strong_terms:
        return list(dict.fromkeys(strong_terms))[:12]

    terms: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9'.+]*|[\u3400-\u9fff]{2,}", normalized):
        candidate = token.strip(".'+")
        if len(candidate) < 2 or candidate in _QUESTION_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z]+", candidate) and len(candidate) < 3:
            continue
        if not re.search(r"[\u3400-\u9fff]", candidate):
            terms.append(candidate)
    return list(dict.fromkeys(terms))[:12]


def _cited_sources_ground_case(
    case: dict[str, Any],
    cited_sources: list[dict[str, Any]],
) -> bool:
    cited_text = " ".join(_source_evidence_text(source) for source in cited_sources)
    if not cited_text.strip():
        return False
    required_terms, evidence_terms = _declared_support_terms(case)
    # Evidence terms describe the concrete support chain and therefore all need
    # to occur in the sources actually cited by the answer.
    if evidence_terms:
        return all(term in cited_text for term in evidence_terms)

    category = str(case.get("category") or "")
    relationship_markers = _CATEGORY_EVIDENCE_MARKERS.get(category, ())
    if relationship_markers:
        cited_body = " ".join(_source_body_text(source) for source in cited_sources)
        if not any(marker in cited_body for marker in relationship_markers):
            return False

    if required_terms and all(term in cited_text for term in required_terms):
        return True
    question_terms = _question_entity_terms(str(case.get("question") or ""))
    if question_terms and any(term in cited_text for term in question_terms):
        return True

    # Entity names are often translated between a Chinese question/answer and
    # an English wiki. Exact lexical equality cannot prove that alias relation.
    # Permit that cross-script boundary only when the cited page is tied to the
    # requested game and its body contains category-specific relationship
    # evidence; an entity-only or unrelated page still fails.
    anchors = [*required_terms, *question_terms]
    cross_script = (
        any(re.search(r"[\u3400-\u9fff]", term) for term in anchors)
        and not re.search(r"[\u3400-\u9fff]", cited_text)
        and bool(re.search(r"[a-z]", cited_text))
    )
    return cross_script and bool(relationship_markers) and _cited_sources_match_game(
        case, cited_sources
    )


def _cited_sources_match_game(
    case: dict[str, Any],
    cited_sources: list[dict[str, Any]],
) -> bool:
    source_text = _normalized_text(" ".join(
        f"{source.get('title') or ''} {source.get('url') or ''} "
        f"{source.get('evidence') or source.get('snippet') or ''}"
        for source in cited_sources
    ))
    source_compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", source_text)
    for value in [case.get("game"), *(case.get("game_aliases") or [])]:
        normalized = _normalized_text(value)
        if not normalized:
            continue
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized)
        if len(compact) >= 4 and compact in source_compact:
            return True
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]{2,}", normalized)
            if len(token) >= 4 and token not in _GENERIC_GAME_WORDS
        ]
        if any(token in source_text or token in source_compact for token in tokens):
            return True
    return False


def _has_version_evidence(case: dict[str, Any], source: dict[str, Any]) -> bool:
    if source.get("game_version"):
        return True
    if not source.get("published_at"):
        return False
    if case.get("requires_official_versioned_source") or case.get("category") == "patch":
        return bool(_VERSION_CONTEXT_PATTERN.search(_source_evidence_text(source)))
    return bool(case.get("version_sensitive"))


def _version_source_is_relevant(case: dict[str, Any], source: dict[str, Any]) -> bool:
    source_text = _source_evidence_text(source)
    declared_games = [case.get("game"), *(case.get("game_aliases") or [])]
    if any(_normalized_text(value) for value in declared_games) and not _version_source_matches_game(
        declared_games, source_text
    ):
        return False
    required_terms, evidence_terms = _declared_support_terms(case)
    declared_terms = [*required_terms, *evidence_terms]
    if declared_terms and any(term in source_text for term in declared_terms):
        return True

    has_version_context = bool(_VERSION_CONTEXT_PATTERN.search(source_text))
    if case.get("requires_official_versioned_source") or case.get("category") == "patch":
        return has_version_context

    game_terms = [
        normalized
        for value in [case.get("game"), *(case.get("game_aliases") or [])]
        if (normalized := _normalized_text(value))
    ]
    if game_terms and any(term in source_text for term in game_terms):
        return True

    # A version field is structured evidence. For dated pages, require an
    # explicit patch/release context so an unrelated dated official page cannot
    # satisfy a version-sensitive case.
    return has_version_context and bool(case.get("version_sensitive"))


def _version_source_matches_game(values: list[Any], source_text: str) -> bool:
    """Bind version evidence to the declared game, excluding URL/TLD accidents."""
    source_compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", source_text)
    for value in values:
        normalized = _normalized_text(value)
        if not normalized:
            continue
        compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized)
        if len(compact) >= 4 and compact in source_compact:
            return True
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]{2,}", normalized)
            if len(token) >= 3 and token not in _GENERIC_GAME_WORDS
        ]
        if tokens and all(
            re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", source_text)
            for token in tokens
        ):
            return True
    return False


def _group_summary(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    totals = Counter(result["case"].get(field, "uncategorized") for result in results)
    passed = Counter(
        result["case"].get(field, "uncategorized")
        for result in results
        if result["evaluation"].get("passed", False)
    )
    confirmations = Counter(
        result["case"].get(field, "uncategorized")
        for result in results
        if result["evaluation"].get("needs_game_confirmation", False)
    )
    return {
        value: {
            "total": count,
            "passed": passed[value],
            "pass_rate": round(passed[value] / count, 4),
            "needs_game_confirmation": confirmations[value],
            "needs_game_confirmation_rate": round(confirmations[value] / count, 4),
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
        "by_expected_behavior": _group_summary(results, "expected_behavior"),
        "by_split": _group_summary(results, "split"),
        "by_tier": _group_summary(results, "tier"),
        "by_difficulty": _group_summary(results, "difficulty"),
        "needs_game_confirmation_rate": round(
            sum(
                result["evaluation"].get("needs_game_confirmation", False)
                for result in results
            ) / total,
            4,
        ) if total else 0,
    }
