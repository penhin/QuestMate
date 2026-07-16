import re
from math import ceil


NON_ENTITY_PHRASE_PATTERN = re.compile(
    r"(在哪里|在哪儿|哪里|哪儿|怎么打|怎么玩|怎么|什么|如何|攻略|测试|问题|"
    r"获得|获取|用来|可以|这个|那个|作用|用途|位置|地点|任务|支线|步骤|"
    r"当前版本|最新版本|版本|补丁|更新|弱点|打法|推荐|改了|改动|调整|"
    r"哪些|有没有|是否|是什么|使用|解除|方法)"
)
CJK_TERMINAL_QUESTION_PARTICLES = "吗么嘛呢"
CJK_RELATION_CONNECTORS = frozenset("与和跟同及")
CJK_INTERROGATIVE_TAIL_PATTERN = re.compile(r"(?:有|是|为)?(?:什么|怎样|如何|哪里|哪儿|多少).*$")
CJK_EDGE_GRAMMAR = "的是为有"
EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:[a-z]+[-_]?\d[a-z0-9._-]*|\d[a-z][a-z0-9._-]*|\d{1,6})(?![a-z0-9])",
    re.IGNORECASE,
)
NON_ENTITY_LATIN_TOKENS = {
    "and",
    "answer",
    "access",
    "acquisition",
    "boss",
    "build",
    "current",
    "can",
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
    "recruit",
    "route",
    "reward",
    "step",
    "steps",
    "strategy",
    "the",
    "trigger",
    "unlock",
    "update",
    "use",
    "usage",
    "version",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "weakness",
}


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
    latin = [
        token
        for token in re.findall(r"[a-z0-9]{2,}", normalized)
        if token not in NON_ENTITY_LATIN_TOKENS and is_query_entity_token(token)
    ]
    chinese: list[str] = []
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        cleaned = NON_ENTITY_PHRASE_PATTERN.sub(" ", phrase)
        cleaned_segments = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
        for segment in cleaned_segments:
            if len(segment) <= 6:
                chinese.append(segment)
            else:
                # Continuous CJK text has no whitespace boundary. Bigrams keep
                # entity wording matchable when the relation verb is paraphrased,
                # without requiring the page to repeat the whole question.
                chinese.extend(segment[index:index + 2] for index in range(len(segment) - 1))
        removed_ratio = 1 - sum(map(len, cleaned_segments)) / len(phrase)
        if removed_ratio < 0.4:
            # CJK does not expose word boundaries. When the existing cleanup
            # did not confidently isolate an entity, retain overlapping
            # character n-grams so an unseen predicate cannot become glued to
            # it and force an exact-sentence match.
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
    sentence_leads = {
        "a", "an", "are", "can", "could", "did", "do", "does", "how",
        "if", "is", "should", "the", "what", "when", "where", "which",
        "who", "why", "will", "would",
    }
    for phrase in [*quoted, *capitalized]:
        tokens = [
            token.casefold()
            for token in re.findall(r"[a-z0-9]{2,}", phrase, re.IGNORECASE)
            if token.casefold() not in sentence_leads
            and token.casefold() not in NON_ENTITY_LATIN_TOKENS
        ]
        if tokens:
            groups.append(list(dict.fromkeys(tokens)))

    # Lower-case noun phrases after a determiner provide a second endpoint for
    # relationship questions even when that endpoint is not title-cased.
    for phrase in re.findall(
        r"\b(?:the|a|an|this|that|these|those)\s+([a-z][a-z0-9'_.-]{2,}(?:\s+[a-z][a-z0-9'_.-]{2,})?)",
        value,
    ):
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]{2,}", phrase)
            if token not in NON_ENTITY_LATIN_TOKENS
        ]
        if tokens:
            groups.append([tokens[-1]])

    groups.extend(_cjk_relationship_endpoint_groups(value))

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            unique.append(group)
            seen.add(key)
    return unique


def _cjk_relationship_endpoint_groups(value: str) -> list[list[str]]:
    """Extract both CJK relation endpoints from grammatical structure.

    CJK entity names have neither title casing nor reliable whitespace.  A
    verb/modal table is also unsafe: a character such as ``能`` may be part of
    the entity itself, and novel predicates would remain invisible.  Instead,
    use explicit coordination (``A 与/和/跟/同/及 B``), a language-level
    structure that does not depend on the action. Terminal-question edges stay
    in the normal relevance-token stream as soft signals: without semantic
    context, ``A opens B?`` and ``A emits light?`` have the same character-level
    shape, so treating both edges as named entities would reject paraphrases.

    Adjacent two-character edge anchors are already present in the CJK
    relevance-token stream. Requiring each endpoint edge rejects an entity-only
    page while a source may still paraphrase the predicate freely.
    """
    groups: list[list[str]] = []
    for phrase in re.findall(r"[\u3400-\u9fff]{6,}", value):
        surface = phrase.rstrip(CJK_TERMINAL_QUESTION_PARTICLES)

        # Relationship questions such as ``A 与 B 有什么关系`` do not always
        # end in 吗. Remove the interrogative tail, then choose the connector
        # with the strongest two-sided span. This also skips a connector near
        # an edge when it is merely part of an entity name.
        connector_surface = CJK_INTERROGATIVE_TAIL_PATTERN.sub("", surface).rstrip(
            CJK_EDGE_GRAMMAR
        )
        connector_candidates: list[tuple[int, int, str, str]] = []
        for index, character in enumerate(connector_surface):
            if character not in CJK_RELATION_CONNECTORS:
                continue
            left = _cjk_clean_endpoint_span(connector_surface[:index], take="last")
            right = _cjk_clean_endpoint_span(connector_surface[index + 1:], take="first")
            if len(left) < 2 or len(right) < 2:
                continue
            connector_candidates.append((min(len(left), len(right)), -abs(len(left) - len(right)), left, right))
        if connector_candidates:
            _span, _balance, left, right = max(connector_candidates)
            groups.extend((
                _cjk_edge_anchor_group(left, take="last"),
                _cjk_edge_anchor_group(right, take="first"),
            ))
    return groups


def _cjk_clean_endpoint_span(value: str, *, take: str) -> str:
    segments = re.findall(
        r"[\u3400-\u9fff]{2,}",
        NON_ENTITY_PHRASE_PATTERN.sub(" ", value.strip(CJK_EDGE_GRAMMAR)),
    )
    if not segments:
        return ""
    return segments[-1] if take == "last" else segments[0]


def _cjk_edge_anchor_group(value: str, *, take: str) -> list[str]:
    """Use adjacent edge bigrams so a generic bigram cannot stand in for an endpoint."""
    edge = value[-3:] if take == "last" else value[:3]
    return list(dict.fromkeys(_overlapping_ngrams(edge, width=2)))


def minimum_cjk_ngram_matches(tokens: list[str]) -> int:
    """Set a script-aware recall floor without requiring exact morphology.

    CJK text is dense enough to use a higher overlap ratio.  Other scripts
    such as Hangul carry grammatical suffixes on nouns, so their character
    n-grams use a smaller floor while still requiring more than one accidental
    match for a full natural-language question.
    """
    cjk_bigrams = [token for token in tokens if re.fullmatch(r"[\u4e00-\u9fff]{2}", token)]
    if len(cjk_bigrams) >= 4:
        return min(5, max(2, ceil(len(cjk_bigrams) * 0.4)))

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
        return len(token) >= 2 and not NON_ENTITY_PHRASE_PATTERN.fullmatch(token)

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
    japanese_noise = {
        "する",
        "できる",
        "どこ",
        "どこで",
        "はどこで",
        "なに",
        "なんで",
        "どうやって",
        "ください",
    }
    for token in candidates:
        if re.search(r"[\u3040-\u30ff]", token):
            # Japanese particles attach to neighbouring scripts without
            # whitespace. Keep the noun-bearing Katakana/Hiragana runs rather
            # than the entire grammatical fragment (for example はどこで).
            script_parts = [
                *re.findall(r"[\u30a1-\u30fa\u30fc]{2,}", token),
                *(
                    part
                    for part in re.findall(r"[\u3041-\u3096]{2,}", token)
                    if part not in japanese_noise
                ),
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
