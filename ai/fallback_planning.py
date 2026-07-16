"""Deterministic planning used only when model planning is unavailable."""

import re

from schemas import PlannedSearchQuery, SearchIntent, SearchPlan

ACTIONABLE_INVESTIGATION_INTENTS = frozenset(
    {"item_location", "item_usage", "quest_step", "game_mechanic"}
)


def requires_action_chain(*, intent: SearchIntent, question: str) -> bool:
    if intent in ACTIONABLE_INVESTIGATION_INTENTS:
        return True
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "如何", "怎么", "在哪", "哪里", "进入", "打开", "解锁", "获得", "获取",
            "触发", "下一步", "找不到", "不见", "why can't", "how to", "where is",
        )
    )


def infer_intent(question: str) -> SearchIntent:
    lowered = question.casefold()
    intent_markers: tuple[tuple[SearchIntent, tuple[str, ...]], ...] = (
        ("patch", ("patch", "version", "update", "版本", "补丁", "更新", "削弱", "增强")),
        ("boss_strategy", ("boss", "打法", "怎么打", "打不过", "弱点", "二阶段", "phase")),
        ("item_usage", ("有什么用", "有啥用", "作用", "用途", "用来", "在哪里用", "怎么用", "what does", "use for")),
        ("item_location", ("在哪", "哪里", "获得", "获取", "钥匙", "位置", "location", "where")),
        ("quest_step", ("任务", "支线", "下一步", "加入队伍", "入队", "招募", "npc", "quest", "questline", "recruit")),
        ("game_mechanic", ("模式", "开启", "打开", "解锁", "隐藏", "触发", "机制", "功能", "设置", "mode", "unlock", "enable", "activate", "trigger", "mechanic", "setting")),
        ("build", ("build", "配装", "加点", "装备", "武器", "护符", "流派")),
        ("lore", ("剧情", "结局", "背景", "lore", "ending")),
    )
    for intent, markers in intent_markers:
        if any(marker in lowered for marker in markers):
            return intent
    return "general"


def fallback_search_subject(question: str) -> str:
    """Keep entity names while dropping natural-language search instructions."""
    normalized = question.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'}))
    latin_phrases = [
        " ".join(phrase.split()).strip(" -_'\"")
        for phrase in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9'_.-]*(?:\s+[A-Za-z0-9][A-Za-z0-9'_.-]*)*",
            normalized,
        )
    ]
    candidates = [phrase for phrase in latin_phrases if len(phrase) >= 3]
    if candidates:
        return max(candidates, key=len)
    return " ".join(question.split()).strip("。！？?!") or question


def fallback_search_plan(*, question: str) -> SearchPlan:
    intent = infer_intent(question)
    subject = fallback_search_subject(question)
    templates: dict[SearchIntent, tuple[tuple[str, str], ...]] = {
        "patch": (("official", "patch notes update"),),
        "boss_strategy": (("wiki", "boss weakness phase"), ("community", "strategy dodge timing build")),
        "item_location": (("wiki", "item location merchant drop"), ("web", "map location guide")),
        "item_usage": (("wiki", "item use effect where to use puzzle"), ("community", "what does it do how to use")),
        "quest_step": (("wiki", "questline step location reward"), ("web", "walkthrough guide")),
        "game_mechanic": (("wiki", "mode mechanic unlock enable trigger"), ("community", "how to enable unlock trigger")),
        "build": (("community", "build stats weapons talismans"), ("wiki", "weapon skill scaling")),
    }
    queries = [
        PlannedSearchQuery(source_type=source_type, query=f"{subject} {suffix}")
        for source_type, suffix in templates.get(intent, ())
    ]
    queries.extend(
        [
            PlannedSearchQuery(source_type="wiki", query=f"{subject} wiki guide"),
            PlannedSearchQuery(source_type="web", query=subject),
        ]
    )
    aliases = [subject] if subject.casefold() != question.casefold() else []
    return SearchPlan(intent=intent, aliases=aliases, queries=queries[:4], missing_info=[])


def is_short_followup(question: str) -> bool:
    lowered = question.casefold().strip()
    markers = (
        "就是", "我说的是", "这个", "那个", "该钥匙", "该区域", "该物品", "该任务",
        "该npc", "那里", "它", "上面", "刚才", "为什么没有", "怎么去", "然后呢",
        "接下来", "it is", "i mean", "same game",
    )
    return any(marker in lowered for marker in markers)
