"""Version-controlled answer-quality policy.

Values in this module shape normal retrieval and ranking behavior. Runtime and
deployment settings belong in config.py; emergency/failure fallbacks stay next
to the code path that uses them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePolicy:
    source_type: str
    trust_score: float
    trust_label: str
    domains: tuple[str, ...] = ()
    query_templates: tuple[str, ...] = ()


SOURCE_POLICIES = {
    "official": SourcePolicy(
        "official",
        0.95,
        "官方",
        query_templates=("{game} official {query}", "{game} patch notes update {query}"),
    ),
    "wiki": SourcePolicy(
        "wiki",
        0.8,
        "百科",
        domains=("fandom.com", "wiki.gg", "fextralife.com"),
    ),
    "community": SourcePolicy(
        "community",
        0.55,
        "社区",
        domains=("reddit.com", "steamcommunity.com"),
    ),
    "web": SourcePolicy(
        "web",
        0.45,
        "网页",
        query_templates=("{game} guide {query}", "{game} 攻略 {query}"),
    ),
}

KNOWLEDGE_SOURCE_TRUST = {
    "official": (0.95, "官方"),
    "wiki": (0.8, "百科"),
    "community": (0.55, "社区"),
    "web": (0.65, "知识库"),
}

SEARCH_NOISE_TOKENS = frozenset(
    {
        "fandom",
        "fextralife",
        "wiki",
        "guide",
        "strategy",
        "weakness",
        "timing",
        "location",
        "merchant",
        "questline",
        "walkthrough",
        "build",
        "stats",
        "weapons",
        "talismans",
        "official",
        "patch",
        "notes",
        "update",
    }
)


@dataclass(frozen=True)
class RankingWeights:
    relevance: float
    retrieval: float
    trust: float
    intent: float = 0
    domain: float = 0
    version: float = 0


SEARCH_RESULT_WEIGHTS = RankingWeights(
    relevance=0.35,
    retrieval=0.25,
    trust=0.25,
    intent=0.1,
    domain=0.03,
    version=0.02,
)
EVIDENCE_POOL_WEIGHTS = RankingWeights(
    relevance=0.55,
    retrieval=0.2,
    trust=0.2,
    version=0.05,
)

INTENT_SOURCE_PREFERENCES = {
    "boss_strategy": {"wiki": 0.8, "community": 1.0, "web": 0.45, "official": 0.2},
    "item_location": {"wiki": 1.0, "web": 0.65, "community": 0.35, "official": 0.2},
    "quest_step": {"wiki": 1.0, "web": 0.65, "community": 0.45, "official": 0.2},
    "item_usage": {"wiki": 1.0, "web": 0.65, "community": 0.45, "official": 0.25},
    "build": {"community": 1.0, "wiki": 0.65, "web": 0.45, "official": 0.2},
    "patch": {"official": 1.0, "wiki": 0.55, "web": 0.45, "community": 0.25},
    "lore": {"wiki": 0.9, "web": 0.65, "community": 0.35, "official": 0.2},
}
DEFAULT_INTENT_SOURCE_PREFERENCE = 0.4

DOMAIN_QUALITY_GROUPS = (
    (("wiki.gg", "fandom.com", "fextralife.com"), 0.9),
    (("bandainamco", "playstation.com", "steampowered.com"), 0.85),
    (("reddit.com", "steamcommunity.com"), 0.55),
)
DEFAULT_DOMAIN_QUALITY = 0.4
COMMUNITY_DOMAINS = ("reddit.com", "steamcommunity.com")
COMMUNITY_DOMAIN_RESULT_LIMIT = 2
DEFAULT_DOMAIN_RESULT_LIMIT = 3
HIGH_TRUST_THRESHOLD = 0.8

VERSION_SIGNAL_TOKENS = ("patch", "version", "update", "1.", "版本", "补丁", "更新")
VERSION_SENSITIVE_INTENTS = frozenset({"patch", "build", "boss_strategy", "game_mechanic"})
STABLE_FACT_INTENTS = frozenset({"item_location", "item_usage", "quest_step", "lore"})


@dataclass(frozen=True)
class RelevanceScorePolicy:
    no_entity_score: float = 0.45
    base_score: float = 0.35
    coverage_weight: float = 0.45
    title_match_bonus: float = 0.12
    title_bonus_cap: float = 0.3


RELEVANCE_SCORE_POLICY = RelevanceScorePolicy()


@dataclass(frozen=True)
class VersionScorePolicy:
    official_sensitive: float = 1.0
    versioned_sensitive: float = 0.85
    undated_sensitive: float = 0.45
    stable_fact: float = 0.75
    default: float = 0.55


VERSION_SCORE_POLICY = VersionScorePolicy()


@dataclass(frozen=True)
class KnowledgeScorePolicy:
    keyword_base: float = 0.5
    keyword_increment: float = 0.08
    keyword_cap: float = 0.95


KNOWLEDGE_SCORE_POLICY = KnowledgeScorePolicy()


@dataclass(frozen=True)
class GameResolutionPolicy:
    confirmed_threshold: float = 0.55
    ambiguity_margin: float = 0.2
    base_confidence: float = 0.25
    alias_bonus: float = 0.25
    platform_bonus: float = 0.35
    database_bonus: float = 0.25
    invalid_name_penalty: float = 0.2
    candidate_base: float = 0.45
    candidate_search_weight: float = 0.35
    candidate_alias_bonus: float = 0.15
    candidate_tag_bonus: float = 0.05


GAME_RESOLUTION_POLICY = GameResolutionPolicy()
FAST_GAME_IDENTITY_MAX_RESULTS = 8
GAME_IDENTITY_CANDIDATE_QUERIES = 1
GAME_IDENTITY_DATABASE_QUERIES = 1
MAX_SEARCH_QUERIES = 8
MAX_QUERIES_PER_PLANNED_QUERY = 2
EXTERNAL_SEARCH_ATTEMPTS = 2
PROGRESSIVE_STRICT_SOURCE_TARGET = 2


def intent_source_preference(intent: str, source_type: str) -> float:
    return INTENT_SOURCE_PREFERENCES.get(intent, {}).get(
        source_type,
        DEFAULT_INTENT_SOURCE_PREFERENCE,
    )


def domain_quality(domain: str) -> float:
    lowered = domain.lower()
    for fragments, score in DOMAIN_QUALITY_GROUPS:
        if any(fragment in lowered for fragment in fragments):
            return score
    return DEFAULT_DOMAIN_QUALITY


def source_domain_limit(domain: str) -> int:
    if any(fragment in domain.lower() for fragment in COMMUNITY_DOMAINS):
        return COMMUNITY_DOMAIN_RESULT_LIMIT
    return DEFAULT_DOMAIN_RESULT_LIMIT
