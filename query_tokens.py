import re
from math import ceil


# Grammar-only separators are retained as a parser fallback. They do not name
# a game, item, action, or guide category.
CJK_RELATION_CONNECTORS = frozenset("与和跟同及")

EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:[a-z]+[-_]?\d[a-z0-9._-]*|\d[a-z][a-z0-9._-]*|\d{1,6})(?![a-z0-9])",
    re.IGNORECASE,
)
def relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    tokens = [normalized] if len(normalized) >= 3 else []
    tokens.extend(re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", normalized))
    tokens.extend(_other_script_tokens(normalized))
    return list(dict.fromkeys(tokens))


def exact_identifiers(value: str) -> list[str]:
    return list(dict.fromkeys(EXACT_IDENTIFIER_PATTERN.findall(value.lower())))


def question_relevance_tokens(value: str) -> list[str]:
    normalized = value.lower().strip()
    identifiers = exact_identifiers(normalized)
    latin = [token for token in re.findall(r"[a-z0-9]{2,}", normalized) if is_query_entity_token(token)]
    chinese: list[str] = []
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        # CJK does not expose word boundaries. Character n-grams avoid a
        # vocabulary-dependent segmenter and work for unseen game terminology.
        chinese.append(phrase)
        chinese.extend(_overlapping_ngrams(phrase, width=2))
    return list(dict.fromkeys([
        *identifiers,
        *latin,
        *chinese,
        *_japanese_entity_tokens(normalized),
        *_other_script_tokens(normalized),
    ]))


def question_named_entity_groups(value: str) -> list[list[str]]:
    """Return high-confidence entity groups without classifying every verb.

    These groups are deliberately sparse. Retrieval may admit a page on a
    lower-case or unseen term, while clearly named multi-entity questions must
    still mention each named side of the relationship.
    """
    groups: list[list[str]] = []
    quoted = re.findall(r"['\"]([^'\"]{2,100})['\"]", value)
    capitalized = re.findall(
        r"(?<![\w])(?:[A-Z][A-Za-z0-9'_.-]*|[A-Z0-9][A-Z0-9_.-]{1,})"
        r"(?:\s+(?:[A-Z][A-Za-z0-9'_.-]*|[A-Z0-9][A-Z0-9_.-]{1,}))*",
        value,
    )
    capitalized = re.findall(
        r"(?<!\w)(?:[A-Z][A-Za-z0-9'_.-]*\s+){1,}[A-Z][A-Za-z0-9'_.-]*",
        value,
    )
    for phrase in [*quoted, *capitalized]:
        tokens = [
            token.casefold()
            for token in re.findall(r"[a-z0-9]{2,}", phrase, re.IGNORECASE)
        ]
        if tokens:
            groups.append(list(dict.fromkeys(tokens)))

    groups.extend(_cjk_relation_endpoint_groups(value))

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            unique.append(group)
            seen.add(key)
    return unique


def _cjk_relation_endpoint_groups(value: str) -> list[list[str]]:
    """Find CJK relation endpoints from separator position, not action words."""
    groups: list[list[str]] = []
    for phrase in re.findall(r"[\u3400-\u9fff]{5,}", value):
        for index, character in enumerate(phrase):
            if character not in CJK_RELATION_CONNECTORS:
                continue
            left, right = phrase[:index], phrase[index + 1:]
            if len(left) >= 2 and len(right) >= 2:
                groups.extend((
                    _overlapping_ngrams(left[-3:], width=2),
                    _overlapping_ngrams(right[:3], width=2),
                ))
                break
    return groups


def minimum_cjk_ngram_matches(tokens: list[str]) -> int:
    """Set a script-aware recall floor without requiring exact morphology.

    CJK text is dense enough to use a higher overlap ratio.  Other scripts
    such as Hangul carry grammatical suffixes on nouns, so their character
    n-grams use a smaller floor while still requiring more than one accidental
    match for a full natural-language question.
    """
    cjk_bigrams = [token for token in tokens if re.fullmatch(r"[\u4e00-\u9fff]{2}", token)]
    if len(cjk_bigrams) >= 4:
        return min(4, max(2, ceil(len(cjk_bigrams) * 0.25)))

    other_script_bigrams = [
        token
        for token in tokens
        if len(token) == 2
        and all(character.isalpha() for character in token)
        and not re.search(r"[a-z\u4e00-\u9fff]", token, re.IGNORECASE)
    ]
    if len(other_script_bigrams) >= 4:
        return min(3, max(2, ceil(len(other_script_bigrams) * 0.25)))
    return 1


def is_query_entity_token(token: str) -> bool:
    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
        return len(token) >= 2

    if re.fullmatch(r"[a-z0-9]+", token):
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)
        return token.isdigit() or (len(token) >= 3 and has_alpha) or (has_alpha and has_digit and len(token) >= 2)

    if len(token) >= 2 and all(char.isalnum() for char in token):
        return any(char.isalpha() for char in token)

    return False


def _other_script_tokens(value: str) -> list[str]:
    """Retain entity words written outside Latin and CJK ideograph scripts."""
    candidates = re.findall(r"[^\W\d_a-z\u4e00-\u9fff]{2,}", value, re.IGNORECASE | re.UNICODE)
    expanded: list[str] = []
    for token in candidates:
        if re.search(r"[\u3040-\u30ff]", token):
            # Japanese particles attach to neighbouring scripts without
            # whitespace. Keep the noun-bearing Katakana/Hiragana runs rather
            # than the entire grammatical fragment (for example はどこで).
            script_parts = [
                *re.findall(r"[\u30a1-\u30fa\u30fc]{2,}", token),
                *re.findall(r"[\u3041-\u3096]{2,}", token),
            ]
            for part in script_parts:
                expanded.append(part)
                if len(part) >= 3:
                    expanded.extend(_overlapping_ngrams(part, width=2))
            continue
        expanded.append(token)
        # Inflected and particle-bearing words in Hangul and other scripts do
        # not necessarily repeat verbatim in a guide page. Character n-grams
        # preserve the entity stem without knowing the language's action verbs.
        if len(token) >= 3:
            expanded.extend(_overlapping_ngrams(token, width=2))
    return list(dict.fromkeys(token for token in expanded if is_query_entity_token(token)))


def _japanese_entity_tokens(value: str) -> list[str]:
    """Extract compact Japanese noun chains that whitespace tokenizers miss."""
    chains = re.findall(
        r"[\u4e00-\u9fff\u3005\u30a1-\u30fa\u30fc]{1,16}"
        r"(?:の[\u4e00-\u9fff\u3005\u30a1-\u30fa\u30fc]{1,16})+",
        value,
    )
    katakana = re.findall(r"[\u30a1-\u30fa\u30fc]{2,}", value)
    return list(dict.fromkeys([*chains, *katakana]))


def _overlapping_ngrams(value: str, *, width: int) -> list[str]:
    if len(value) <= width:
        return [value]
    return [value[index:index + width] for index in range(len(value) - width + 1)]
