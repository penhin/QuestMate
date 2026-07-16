"""Deterministic, broad-recall planning when model planning is unavailable.

The fallback is intentionally a portfolio generator rather than a miniature
rules engine.  Intent signals add useful vocabulary and source candidates, but
the original subject is always retained so an unknown relationship or a new
kind of game question can still be found by open-web retrieval.
"""

import re

from quality_policy import is_version_sensitive_question
from schemas import PlannedSearchQuery, SearchIntent, SearchPlan

# Signals are used only to rank a fallback intent.  They never filter sources
# or replace the user's wording.  Keeping overlapping groups is deliberate:
# ``where do I use`` should score as usage even though it also mentions where.
INTENT_SIGNALS: tuple[tuple[SearchIntent, tuple[str, ...]], ...] = (
    (
        "item_usage",
        (
            "有什么用", "有啥用", "作用", "用途", "用来", "在哪里用", "怎么用",
            "what does", "used for", "use for", "how to use",
        ),
    ),
    (
        "item_location",
        (
            "在哪", "哪里", "获得", "获取", "掉落", "位置", "location", "where",
            "find", "obtain", "acquire", "drop",
        ),
    ),
    (
        "quest_step",
        (
            "任务", "支线", "下一步", "加入队伍", "入队", "招募", "quest",
            "questline", "recruit", "join party",
        ),
    ),
    (
        "game_mechanic",
        (
            "模式", "开启", "打开", "解锁", "隐藏", "触发", "机制", "功能", "设置",
            "mode", "unlock", "enable", "activate", "trigger", "mechanic", "setting",
        ),
    ),
    (
        "boss_strategy",
        ("boss", "打法", "怎么打", "打不过", "弱点", "阶段", "phase", "strategy", "weakness"),
    ),
    (
        "build",
        ("build", "配装", "加点", "装备", "武器", "护符", "流派", "stats", "loadout"),
    ),
    (
        "patch",
        (
            "patch", "version", "update", "latest", "版本", "补丁", "更新", "削弱",
            "增强", "改动", "hotfix",
        ),
    ),
    ("lore", ("剧情", "结局", "背景", "设定", "lore", "ending", "backstory")),
)

# These are optional recall hints, not complete query templates.  A raw/open
# candidate is generated alongside them by ``retrieval.query_builder``.
INTENT_QUERY_FACETS: dict[SearchIntent, tuple[str, ...]] = {
    "patch": ("patch notes version changes", "版本 更新 改动"),
    "boss_strategy": ("weakness phase strategy", "dodge timing fight guide"),
    "item_location": ("location obtain source", "merchant drop route"),
    "item_usage": ("use effect interaction", "what does it do how to use"),
    "quest_step": ("quest steps requirements", "walkthrough next step reward"),
    "game_mechanic": ("mechanic unlock requirements", "enable trigger conditions"),
    "build": ("build stats equipment", "weapons skills scaling"),
    "lore": ("lore story context", "ending character relationship"),
    "general": (),
}


def infer_intent(question: str) -> SearchIntent:
    """Choose the best overlapping intent signal instead of first-match wins."""
    lowered = " ".join(question.casefold().split())
    scores: dict[SearchIntent, tuple[int, int, int]] = {}
    for intent, signals in INTENT_SIGNALS:
        matched = [signal for signal in signals if _contains_signal(lowered, signal)]
        if matched:
            # A specific phrase wins over several overlapping fragments (for
            # example ``在哪里用`` is usage, not two location matches).  The
            # other dimensions still help with genuinely compound questions.
            scores[intent] = (
                max(len(signal) for signal in matched),
                sum(len(signal) for signal in matched),
                len(matched),
            )
    if not scores:
        return "general"
    return max(scores, key=scores.__getitem__)


def _contains_signal(question: str, signal: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", signal):
        return signal in question
    return re.search(rf"(?<![a-z0-9]){re.escape(signal)}(?![a-z0-9])", question) is not None


def fallback_search_subject(question: str) -> str:
    """Keep likely entity names while avoiding guesses for unknown wording."""
    normalized = question.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    compact = " ".join(normalized.split()).strip("。！？?! ")

    quoted = [
        " ".join(phrase.split()).strip()
        for phrase in re.findall(r"['\"]([^'\"]{2,80})['\"]", compact)
    ]
    if quoted:
        return max(quoted, key=len)

    latin_phrases = [
        " ".join(phrase.split()).strip(" -_'\"")
        for phrase in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'_.-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'_.-]*)*",
            compact,
        )
    ]
    embedded = [
        phrase
        for phrase in latin_phrases
        if len(phrase) >= 3 and phrase.casefold() != compact.casefold()
    ]
    if embedded:
        return max(embedded, key=len)

    # For an all-English question, a capitalized multi-word phrase is often an
    # entity name.  If no such phrase exists, retaining the whole question is
    # safer than inventing one from a keyword list.
    capitalized = [
        phrase.strip()
        for phrase in re.findall(
            r"(?:[A-Z][A-Za-z0-9'_.-]+|[A-Z0-9][A-Z0-9_.-]{1,})(?:\s+(?:[A-Z][A-Za-z0-9'_.-]+|[A-Z0-9][A-Z0-9_.-]{1,}))+",
            compact,
        )
    ]
    if capitalized:
        return max(capitalized, key=len)
    return compact or question


def fallback_search_plan(*, question: str) -> SearchPlan:
    intent = infer_intent(question)
    subject = fallback_search_subject(question)
    raw_question = " ".join(question.split()).strip()
    # The extracted subject is useful as an alias, not as a replacement for
    # an unseen relationship. Keeping the full question in every fallback
    # route prevents tokens such as DLC, NPC, or v2.01 from swallowing the
    # actual condition the user asked about.
    query_subject = raw_question or subject
    facets = INTENT_QUERY_FACETS[intent]
    source_order = _fallback_source_order(intent)

    queries: list[PlannedSearchQuery] = []
    for index, source_type in enumerate(source_order):
        facet = facets[index] if index < len(facets) else ""
        queries.append(
            PlannedSearchQuery(
                source_type=source_type,
                query=_bounded_planned_query(query_subject, facet),
            )
        )

    # Always include an unmodified subject as a web candidate.  Deduplication
    # happens after expansion, so this costs nothing when another candidate is
    # equivalent but protects novel intents from the facet vocabulary above.
    has_compact_entity = _is_compact_entity(subject, question)
    open_subject = raw_question or subject
    queries.append(
        PlannedSearchQuery(
            source_type="web",
            query=_bounded_text(open_subject, 240),
        )
    )
    aliases = [subject] if has_compact_entity else []
    return SearchPlan(
        intent=intent,
        version_sensitive=is_version_sensitive_question(question),
        named_entity_groups=_fallback_named_entity_groups(question),
        aliases=aliases,
        queries=queries[:4],
        missing_info=[],
    )


def _fallback_named_entity_groups(question: str) -> list[list[str]]:
    """Keep only explicitly quoted fallback entities; ambiguous CJK edges stay soft."""
    normalized = question.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    entities = [
        " ".join(value.split()).strip()
        for value in re.findall(r"['\"]([^'\"]{2,80})['\"]", normalized)
    ]
    return [[value] for value in dict.fromkeys(entities) if value][:4]


def _fallback_source_order(intent: SearchIntent) -> tuple[str, ...]:
    if intent == "patch":
        return ("official", "web", "community")
    if intent in {"boss_strategy", "build"}:
        return ("community", "wiki", "web")
    return ("wiki", "community", "web")


def _bounded_planned_query(subject: str, facet: str, *, max_chars: int = 240) -> str:
    combined = " ".join(part for part in (subject, facet) if part).strip()
    if len(combined) <= max_chars:
        return combined
    if not facet:
        return _bounded_text(subject, max_chars)
    subject_budget = max(1, max_chars - len(facet) - 1)
    return f"{_bounded_text(subject, subject_budget)} {facet}".strip()


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit < 5:
        return value[:limit]
    left = (limit - 1) // 2
    right = limit - left - 1
    return f"{value[:left].rstrip()} {value[-right:].lstrip()}"[:limit].strip()


def _is_compact_entity(subject: str, question: str) -> bool:
    normalized_question = " ".join(question.split()).strip("。！？?! ")
    if subject.casefold() == normalized_question.casefold():
        return False
    return len(subject) <= 80 and len(subject.split()) <= 8


def is_short_followup(question: str) -> bool:
    lowered = question.casefold().strip()
    markers = (
        "就是", "我说的是", "这个", "那个", "该钥匙", "该区域", "该物品", "该任务",
        "该npc", "那里", "它", "上面", "刚才", "为什么没有", "怎么去", "然后呢",
        "接下来", "it is", "i mean", "same game",
    )
    return any(marker in lowered for marker in markers)
