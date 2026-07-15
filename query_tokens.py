import re


NON_ENTITY_PHRASE_PATTERN = re.compile(
    r"(在哪里|在哪儿|哪里|哪儿|怎么打|怎么玩|怎么|什么|如何|攻略|测试|问题|"
    r"获得|获取|用来|可以|这个|那个|作用|用途|位置|地点|任务|支线|步骤|"
    r"当前版本|最新版本|版本|补丁|更新|弱点|打法|推荐|改了|改动|调整|"
    r"哪些|有没有|是否|是什么)"
)
EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:[a-z]*\d[a-z0-9._-]*|\d{1,6})(?![a-z0-9])",
    re.IGNORECASE,
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


def exact_identifiers(value: str) -> list[str]:
    return list(dict.fromkeys(EXACT_IDENTIFIER_PATTERN.findall(value.lower())))


def question_relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    identifiers = exact_identifiers(normalized)
    latin = [
        token
        for token in re.findall(r"[a-z0-9]{2,}", normalized)
        if token not in NON_ENTITY_LATIN_TOKENS and is_query_entity_token(token)
    ]
    chinese: list[str] = []
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        cleaned = NON_ENTITY_PHRASE_PATTERN.sub(" ", phrase)
        chinese.extend(re.findall(r"[\u4e00-\u9fff]{2,}", cleaned))
    return list(dict.fromkeys([*identifiers, *latin, *chinese]))


def is_query_entity_token(token: str) -> bool:
    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
        return len(token) >= 2 and not NON_ENTITY_PHRASE_PATTERN.fullmatch(token)

    if re.fullmatch(r"[a-z0-9]+", token):
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)
        return token.isdigit() or (len(token) >= 4 and has_alpha) or (has_alpha and has_digit and len(token) >= 2)

    return False
