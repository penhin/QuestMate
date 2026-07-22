"""User-facing, deterministic orchestration status messages."""

from quality_policy import HIGH_TRUST_THRESHOLD
from schemas import GameResolution, SearchPlan, Source


def plan_start(_question: str) -> str:
    return "理解问题：识别目标和关键关系"


def search(plan: SearchPlan) -> str:
    labels = {
        "boss_strategy": "类型：Boss 打法；查弱点/阶段/社区打法",
        "item_location": "类型：物品位置；查地点/条件/路线",
        "item_usage": "类型：物品用途；查效果/用法/交互对象",
        "quest_step": "类型：任务步骤；查 NPC/触发/顺序",
        "game_mechanic": "类型：游戏机制；查开启条件/触发方式",
        "build": "类型：配装；查数值/装备/版本",
        "patch": "类型：版本变化；优先官方补丁",
        "lore": "类型：剧情背景；查事实和解释",
    }
    return labels.get(plan.intent, "类型：通用问题；筛选相关来源")


def sources(sources: list[Source]) -> str:
    if not sources:
        return "来源筛选：未找到强相关资料"
    trusted_count = sum(source.trust_score >= HIGH_TRUST_THRESHOLD for source in sources)
    if trusted_count:
        return f"来源筛选：保留 {len(sources)} 个，{trusted_count} 个高可信"
    return f"来源筛选：保留 {len(sources)} 个，交叉核对"


def game_resolution(resolution: GameResolution) -> str:
    if resolution.is_confirmed:
        entries = sum(map(len, (
            resolution.platform_urls,
            resolution.official_urls,
            resolution.identity_urls,
            resolution.database_domains,
        )))
        return f"确认游戏：找到 {entries} 个入口" if entries else "确认游戏：使用当前游戏名"
    return "确认游戏：需要更多身份线索"
