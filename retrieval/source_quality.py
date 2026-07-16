"""Page-level source quality derived from identity and focused evidence."""

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlparse

from quality_policy import (
    SEARCH_NOISE_TOKENS,
    SOURCE_EVIDENCE_QUALITY_POLICY,
    domain_quality,
)
from query_tokens import (
    minimum_cjk_ngram_matches,
    question_named_entity_groups,
    question_relevance_tokens,
    relevance_tokens,
)


@dataclass(frozen=True)
class SourceQualitySignals:
    game_identity: float
    entity_coverage: float
    evidence_support: float
    domain_prior: float


def token_in_text(token: str, text: str) -> bool:
    """Match Latin tokens on boundaries while retaining natural CJK matching."""
    if re.fullmatch(r"[a-z0-9]+", token):
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


def matches_game_identity(*, text: str, game_names: list[str]) -> bool:
    original_text = text
    text = text.casefold()
    text_tokens = _identity_tokens(text)
    meaningful_text_tokens = _without_connectors(text_tokens)
    for game_name in game_names:
        normalized = game_name.casefold().strip()
        if not normalized:
            continue
        name_tokens = _without_connectors(_identity_tokens(normalized))
        if not name_tokens or not _variant_tokens_match(name_tokens, text_tokens, text):
            continue
        if _contains_sequence(meaningful_text_tokens, name_tokens):
            if len(name_tokens) >= 2 and (
                not _is_latin_name(normalized)
                or _has_titled_latin_anchor(game_name.strip(), original_text)
            ):
                return True
        latin_tokens = _without_connectors(re.findall(r"[a-z0-9]+", normalized))
        if latin_tokens and _latin_sequence_in_text(latin_tokens, text):
            if len(latin_tokens) >= 2:
                if _has_titled_latin_anchor(game_name.strip(), original_text):
                    return True
            if _single_latin_identity_match(
                token=latin_tokens[0],
                original_name=game_name.strip(),
                original_text=original_text,
            ):
                return True
        for script_pattern in (r"[\u4e00-\u9fff]", r"[^\W\d_a-z\u4e00-\u9fff]"):
            script_name = "".join(re.findall(script_pattern, normalized, re.UNICODE))
            script_text = "".join(re.findall(script_pattern, text, re.UNICODE))
            if not script_name or script_name not in script_text:
                continue
            if len(script_name) >= 3 or _short_script_identity_match(script_name, text):
                return True
    return not any(game_name.strip() for game_name in game_names)


IDENTITY_CONNECTORS = frozenset({"and", "the", "of"})
def _identity_tokens(value: str) -> list[str]:
    normalized = value.casefold().replace("&", " and ")
    return re.findall(r"[^\W_]+", normalized, re.UNICODE)


def _without_connectors(tokens: list[str]) -> list[str]:
    filtered = [token for token in tokens if token not in IDENTITY_CONNECTORS]
    return filtered or tokens


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[index:index + len(needle)] == needle for index in range(len(haystack) - len(needle) + 1))


def _variant_tokens_match(
    name_tokens: list[str],
    text_tokens: list[str],
    text: str,
) -> bool:
    variants = [
        token
        for token in name_tokens
        if token.isdigit() or re.fullmatch(r"[ivxlcdm]{1,6}", token)
    ]
    return all(_variant_appears(token, text_tokens=text_tokens, text=text) for token in variants)


def _variant_appears(token: str, *, text_tokens: list[str], text: str) -> bool:
    value = _variant_number(token)
    alternatives = {token}
    if value is not None:
        alternatives.add(str(value))
        roman = _to_roman(value)
        if roman:
            alternatives.add(roman)
    return any(
        alternative in text_tokens
        or re.search(
            rf"(?<![a-z0-9]){re.escape(alternative)}(?![a-z0-9])",
            text,
        ) is not None
        for alternative in alternatives
    )


def _variant_number(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    if not re.fullmatch(r"[ivxlcdm]{1,9}", token):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(token):
        value = values[character]
        total += -value if value < previous else value
        previous = max(previous, value)
    return total if _to_roman(total) == token else None


def _to_roman(value: int) -> str | None:
    if not 1 <= value <= 3999:
        return None
    pairs = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )
    remaining = value
    result: list[str] = []
    for number, numeral in pairs:
        count, remaining = divmod(remaining, number)
        result.extend([numeral] * count)
    return "".join(result)


def _short_script_identity_match(name: str, text: str) -> bool:
    """Require local, game-specific context for ambiguous one/two-character names."""
    local_markers = (
        "攻略",
        "wiki",
        "维基",
        "百科",
        "공략",
        "위키",
        "攻略wiki",
    )
    for match in re.finditer(re.escape(name), text):
        start = max(0, match.start() - 8)
        end = min(len(text), match.end() + 8)
        if any(marker in text[start:end] for marker in local_markers):
            return True
    first_url = text.find("http")
    if first_url >= 0:
        title_surface = text[:first_url].strip(" -|:：")
        if title_surface == name or title_surface.endswith(f" {name}"):
            return True
    return any(_identity_url_contains(url, name) for url in _urls_in_text(text))


def _single_latin_identity_match(
    *,
    token: str,
    original_name: str,
    original_text: str,
) -> bool:
    if len(token) <= 2:
        return any(_identity_url_contains(url, token) for url in _urls_in_text(original_text.casefold()))
    if any(_identity_url_contains(url, token) for url in _urls_in_text(original_text.casefold())):
        return True
    escaped_name = re.escape(original_name)
    explicit_title = re.compile(
        rf"(?<![A-Za-z0-9]){escaped_name}(?![A-Za-z0-9])"
        rf"\s*(?:[|:：\-–—(]|is\s+(?:an?\s+)?)\s*"
        rf"(?:the\s+|\d{{4}}\s+)?(?:video\s+game|game\s+(?:guide|wiki)|official)",
        re.IGNORECASE,
    )
    return explicit_title.search(original_text) is not None


def _is_latin_name(value: str) -> bool:
    return re.fullmatch(r"[a-z0-9\s&'_.:()\-]+", value) is not None


def _has_titled_latin_anchor(name: str, text: str) -> bool:
    name_parts = re.findall(r"[A-Za-z0-9]+", name)
    if not name_parts:
        return False
    patterns: list[str] = []
    for part in name_parts:
        value = _variant_number(part.casefold())
        if value is None:
            patterns.append(re.escape(part))
            continue
        alternatives = {part, str(value)}
        if roman := _to_roman(value):
            alternatives.update({roman, roman.upper()})
        patterns.append("(?:" + "|".join(re.escape(item) for item in alternatives) + ")")
    pattern = r"[^A-Za-z0-9]+".join(patterns)
    return re.search(
        rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    ) is not None or any(
        _identity_url_contains(url, name.casefold()) for url in _urls_in_text(text.casefold())
    )


def _urls_in_text(text: str) -> list[str]:
    return [value.rstrip(".,);]}") for value in re.findall(r"https?://[^\s<>]+", text)]


def _identity_url_contains(url: str, name: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.casefold().split(":", 1)[0].removeprefix("www.")
    identity_hosts = (
        "steampowered.com",
        "itch.io",
        "gog.com",
        "epicgames.com",
        "playstation.com",
        "nintendo.com",
        "xbox.com",
        "apps.apple.com",
        "play.google.com",
        "fandom.com",
        "wiki.gg",
        "miraheze.org",
        "wikitide.org",
        "wikitide.net",
        "gamefaqs.gamespot.com",
        "neoseeker.com",
    )
    if not any(host == candidate or host.endswith(f".{candidate}") for candidate in identity_hosts):
        return False
    host_path = f"{parsed.netloc} {parsed.path}".casefold()
    tokens = _identity_tokens(host_path.replace("-", " ").replace("_", " ").replace("/", " "))
    name_tokens = _without_connectors(_identity_tokens(name.casefold()))
    if name_tokens and _contains_sequence(_without_connectors(tokens), name_tokens):
        return True
    compact_name = "".join(name_tokens)
    compact_segments = {
        "".join(character for character in segment.casefold() if character.isalnum())
        for segment in re.split(r"[./_-]+", f"{parsed.netloc}{parsed.path}")
    }
    return bool(compact_name) and compact_name in compact_segments


def _latin_sequence_in_text(tokens: list[str], text: str) -> bool:
    if not tokens:
        return False
    token_patterns: list[str] = []
    for token in tokens:
        value = _variant_number(token)
        alternatives = {token}
        if value is not None:
            alternatives.add(str(value))
            if roman := _to_roman(value):
                alternatives.add(roman)
        token_patterns.append("(?:" + "|".join(re.escape(item) for item in sorted(alternatives)) + ")")
    separated = r"[^a-z0-9]+".join(token_patterns)
    compact = "".join(tokens)
    return re.search(
        rf"(?<![a-z0-9])(?:{separated}|{re.escape(compact)})(?![a-z0-9])",
        text,
    ) is not None


def source_entity_tokens(*, question: str, game_names: list[str]) -> list[str]:
    game_token_set = set(relevance_tokens(" ".join(game_names))) | set(
        question_relevance_tokens(" ".join(game_names))
    )
    infrastructure_noise = {"com", "http", "https", "site", "www"}
    return [
        token
        for token in question_relevance_tokens(question)
        if token not in game_token_set
        and token not in SEARCH_NOISE_TOKENS
        and token not in infrastructure_noise
    ]


def source_entity_groups(*, question: str, game_names: list[str]) -> list[list[str]]:
    game_token_set = set(relevance_tokens(" ".join(game_names))) | set(
        question_relevance_tokens(" ".join(game_names))
    )
    groups: list[list[str]] = []
    for group in question_named_entity_groups(question):
        filtered = [
            token
            for token in group
            if token not in game_token_set and token not in SEARCH_NOISE_TOKENS
        ]
        if filtered:
            groups.append(filtered)
    return groups


def required_entity_groups_match(*, groups: list[list[str]], text: str) -> bool:
    """Require every distinct entity while allowing aliases within its group."""
    if not groups:
        return True
    return all(
        any(_entity_name_in_text(name, text) for name in group if name.strip())
        for group in groups
    )


def required_entity_groups_for_query(
    groups: list[list[str]],
    query: str,
) -> list[list[str]]:
    """Select the distinct entities explicitly carried by one planned query."""
    return [
        group
        for group in groups
        if any(_entity_name_in_text(name, query) for name in group if name.strip())
    ]


def _entity_name_in_text(name: str, text: str) -> bool:
    normalized_name = " ".join(name.casefold().split())
    normalized_text = " ".join(text.casefold().split())
    if not normalized_name:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9\s'_.:-]*", normalized_name):
        parts = re.findall(r"[a-z0-9]+", normalized_name)
        if not parts:
            return False
        pattern = r"[^a-z0-9]+".join(re.escape(part) for part in parts)
        return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", normalized_text) is not None
    return normalized_name in normalized_text


def minimum_entity_matches(tokens: list[str], groups: list[list[str]]) -> int:
    if groups:
        return len({token for group in groups for token in group})
    return minimum_cjk_ngram_matches(tokens)


def all_named_entities_match(*, groups: list[list[str]], text: str) -> bool:
    return all(all(token_in_text(token, text) for token in group) for group in groups)


def _coverage(tokens: list[str], text: str) -> float:
    if not tokens:
        return 0
    return sum(1 for token in tokens if token_in_text(token, text)) / len(tokens)


def evidence_support_score(
    *,
    evidence: str,
    question: str,
    game_names: list[str],
    required_entity_groups: list[list[str]] | None = None,
) -> float:
    """Estimate whether a focused passage, rather than only its title, supports the query."""
    lowered = " ".join(evidence.casefold().split())
    if not lowered:
        return 0
    entity_tokens = source_entity_tokens(question=question, game_names=game_names)
    if not entity_tokens:
        identity = 1.0 if matches_game_identity(text=evidence, game_names=game_names) else 0.0
        return min(0.65, identity * 0.45 + min(len(lowered) / 400, 1) * 0.2)
    required_entity_groups = required_entity_groups or []
    if required_entity_groups:
        if not required_entity_groups_match(groups=required_entity_groups, text=lowered):
            return 0
        detail = min(len(lowered) / 240, 1)
        return min(1.0, 0.85 + detail * 0.15)
    groups = source_entity_groups(question=question, game_names=game_names)
    if groups and not all_named_entities_match(groups=groups, text=lowered):
        return 0
    matched = sum(1 for token in entity_tokens if token_in_text(token, lowered))
    minimum_matches = minimum_entity_matches(entity_tokens, groups)
    focused_match = min(matched / minimum_matches, 1)
    # Passage detail is a small tie-breaker; matching the requested entities is
    # intentionally the dominant signal.  A refinement context can contain both
    # the original goal and a newly discovered dependency, so support is judged
    # against the minimum focused match instead of every token from both clauses.
    detail = min(len(lowered) / 240, 1)
    return min(1.0, focused_match * 0.85 + detail * 0.15)


def source_quality_signals(
    *,
    item: dict[str, Any],
    game: str,
    game_aliases: list[str] | None,
    question: str,
    evidence: str | None = None,
    required_entity_groups: list[list[str]] | None = None,
) -> SourceQualitySignals:
    game_names = [game, *(game_aliases or [])]
    title = str(item.get("title") or "")
    url = str(item.get("url") or "")
    content = evidence if evidence is not None else str(item.get("content") or "")
    identity_text = f"{title} {url} {content}"
    complete_text = identity_text.casefold()
    entities = source_entity_tokens(question=question, game_names=game_names)
    return SourceQualitySignals(
        game_identity=1.0 if matches_game_identity(text=identity_text, game_names=game_names) else 0.0,
        entity_coverage=_coverage(entities, complete_text) if entities else 0.45,
        evidence_support=evidence_support_score(
            evidence=content,
            question=question,
            game_names=game_names,
            required_entity_groups=required_entity_groups,
        ),
        domain_prior=domain_quality(urlparse(url).netloc),
    )


def page_source_quality(
    *,
    item: dict[str, Any],
    source_prior: float,
    game: str,
    game_aliases: list[str] | None,
    question: str,
    relevance: float,
    evidence: str | None = None,
    required_entity_groups: list[list[str]] | None = None,
) -> tuple[float, SourceQualitySignals]:
    """Combine weak provider priors with strong page-specific evidence signals."""
    signals = source_quality_signals(
        item=item,
        game=game,
        game_aliases=game_aliases,
        question=question,
        evidence=evidence,
        required_entity_groups=required_entity_groups,
    )
    policy = SOURCE_EVIDENCE_QUALITY_POLICY
    score = (
        source_prior * policy.source_prior_weight
        + signals.domain_prior * policy.domain_prior_weight
        + signals.game_identity * policy.game_identity_weight
        + relevance * policy.relevance_weight
        + signals.evidence_support * policy.evidence_support_weight
    )
    return min(1.0, max(0.0, score)), signals


def page_authority_score(*, item: dict[str, Any], source_prior: float) -> float:
    """Estimate publisher authority without treating query overlap as trust.

    Relevance can make an obscure page useful evidence, but repeating the game
    and entity names must not make that publisher equivalent to an official or
    established reference source.
    """
    url = str(item.get("url") or "")
    domain_prior = domain_quality(urlparse(url).netloc)
    return min(1.0, max(0.0, source_prior * 0.7 + domain_prior * 0.3))
