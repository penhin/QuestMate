import re


NON_ENTITY_PHRASE_PATTERN = re.compile(
    r"(在哪里|在哪儿|哪里|哪儿|怎么打|怎么玩|怎么|什么|如何|攻略|测试|问题|"
    r"获得|获取|用来|可以|这个|那个|作用|用途|位置|地点|任务|支线|步骤|"
    r"当前版本|最新版本|版本|补丁|更新|弱点|打法|推荐|改了|改动|调整|"
    r"哪些|有没有|是否|是什么|怎么样|才能|进入|当我|如果|已经|其它|其他|"
    r"所有人|只有|一个|没被|没有被|最后|被票出去了|被票出|出去|谁会|获胜|"
    r"会赢|胜利|玩|号房)"
)
NUMBERED_LOCATION_PATTERN = re.compile(
    r"(?<!\d)(\d{1,4})\s*(号房|号公寓|房间|室|楼|层|f\b)",
    re.IGNORECASE,
)
CROSS_LANGUAGE_CONCEPTS = (
    (("感染", "传染"), ("infect", "infection")),
    (("票出", "投票出局", "被投出"), ("vote", "voted")),
    (("获胜", "胜利", "谁会赢", "谁赢"), ("win", "victory")),
)
NON_ENTITY_LATIN_TOKENS = {
    "answer",
    "boss",
    "build",
    "current",
    "does",
    "effect",
    "enable",
    "game",
    "guide",
    "how",
    "item",
    "join",
    "location",
    "mechanic",
    "mode",
    "official",
    "party",
    "puzzle",
    "patch",
    "phase",
    "quest",
    "reward",
    "step",
    "steps",
    "strategy",
    "trigger",
    "unlock",
    "update",
    "usage",
    "version",
    "what",
    "when",
    "where",
    "weakness",
}


def relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    tokens = [normalized] if len(normalized) >= 3 else []
    tokens.extend(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", normalized))
    return list(dict.fromkeys(tokens))


def question_relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    concept_tokens = [
        english_token
        for chinese_markers, english_tokens in CROSS_LANGUAGE_CONCEPTS
        if any(marker in normalized for marker in chinese_markers)
        for english_token in english_tokens
    ]
    numbered_locations = [
        f"{number}{label.lower()}"
        for number, label in NUMBERED_LOCATION_PATTERN.findall(normalized)
    ]
    location_numbers = [number for number, _label in NUMBERED_LOCATION_PATTERN.findall(normalized)]
    latin = [
        token
        for token in re.findall(r"[a-z0-9]{2,}", normalized)
        if token not in NON_ENTITY_LATIN_TOKENS and is_query_entity_token(token)
    ]
    chinese: list[str] = []
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        cleaned = NON_ENTITY_PHRASE_PATTERN.sub(" ", phrase)
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", cleaned):
            token = token.removesuffix("的人").removesuffix("时").removesuffix("了")
            if len(token) >= 2:
                chinese.append(token)
    return list(dict.fromkeys([*numbered_locations, *location_numbers, *latin, *chinese, *concept_tokens]))


def is_query_entity_token(token: str) -> bool:
    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
        return len(token) >= 2 and not NON_ENTITY_PHRASE_PATTERN.fullmatch(token)

    if re.fullmatch(r"[a-z0-9]+", token):
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)
        return token.isdigit() or (len(token) >= 4 and has_alpha) or (has_alpha and has_digit and len(token) >= 2)

    return False
