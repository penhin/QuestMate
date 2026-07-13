import re


NON_ENTITY_PHRASE_PATTERN = re.compile(r"(哪里|怎么|什么|如何|攻略|测试|问题|获得|获取|用来|可以|这个|那个)")


def relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    tokens = [normalized] if len(normalized) >= 3 else []
    tokens.extend(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", normalized))
    return list(dict.fromkeys(tokens))


def question_relevance_tokens(value: str) -> list[str]:
    return [token for token in relevance_tokens(value) if is_query_entity_token(token)]


def is_query_entity_token(token: str) -> bool:
    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
        return len(token) >= 2 and not NON_ENTITY_PHRASE_PATTERN.fullmatch(token)

    if re.fullmatch(r"[a-z0-9]+", token):
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)
        return (len(token) >= 6 and has_alpha) or (has_alpha and has_digit and len(token) >= 4)

    return False
