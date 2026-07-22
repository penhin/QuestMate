"""Input-boundary normalization for model-generated search plans."""

import re


PROMPT_INJECTION_QUERY_PATTERNS = (
    re.compile(r"\b(ignore|disregard|forget|override)\b.{0,80}\b(instructions?|prompt|rules|system|developer)\b", re.I),
    re.compile(r"\b(reveal|print|show|output|display|exfiltrate)\b.{0,80}\b(api keys?|tokens?|secrets?|system prompt|developer instructions|hidden configuration|environment variables)\b", re.I),
    re.compile(r"(忽略|无视|覆盖|忘记).{0,40}(指令|规则|提示词|系统|开发者)", re.I),
    re.compile(r"(输出|显示|泄露|透露|打印).{0,40}(系统prompt|系统提示|提示词|api key|密钥|环境变量|隐藏配置)", re.I),
)


def sanitize_search_text(value: str) -> str:
    clauses = re.split(r"([。！？!?；;\n])", value)
    kept: list[str] = []
    for index in range(0, len(clauses), 2):
        clause = clauses[index].strip()
        separator = clauses[index + 1] if index + 1 < len(clauses) else ""
        if clause and not any(pattern.search(clause) for pattern in PROMPT_INJECTION_QUERY_PATTERNS):
            kept.append(f"{clause}{separator}")
    return " ".join("".join(kept).split())


def sanitize_aliases(aliases: list[str]) -> list[str]:
    cleaned: list[str] = []
    for alias in aliases[:6]:
        value = sanitize_search_text(alias).strip().strip("\"'“”‘’")
        lowered = value.lower()
        if (
            not value or len(value) > 80
            or any(token in lowered for token in ("http://", "https://", "site:", "ignore", "system prompt", "api key"))
            or lowered in {"wiki", "guide", "boss", "item", "quest", "攻略", "打法", "位置"}
        ):
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def sanitize_answer_requirements(requirements: list[str]) -> list[str]:
    values: list[str] = []
    for requirement in requirements[:4]:
        if not isinstance(requirement, str):
            continue
        cleaned = sanitize_search_text(requirement).strip()
        if 3 <= len(cleaned) <= 240 and cleaned not in values:
            values.append(cleaned)
    return values


def entity_occurs_in_text(entity: str, text: str) -> bool:
    normalized_entity = " ".join(entity.casefold().split())
    normalized_text = " ".join(text.casefold().split())
    if not normalized_entity:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9\s'_.:-]*", normalized_entity):
        parts = re.findall(r"[a-z0-9]+", normalized_entity)
        pattern = r"[^a-z0-9]+".join(re.escape(part) for part in parts)
        return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", normalized_text) is not None
    compact_entity = "".join(character for character in normalized_entity if character.isalnum())
    compact_text = "".join(character for character in normalized_text if character.isalnum())
    return len(compact_entity) >= 2 and compact_entity in compact_text


def sanitize_named_entity_groups(
    groups: list[list[str]], *, question: str, aliases: list[str], queries: list[str],
) -> list[list[str]]:
    route_texts = [*aliases, *queries]
    sanitized: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for raw_group in groups[:4]:
        names = sanitize_aliases(raw_group[:4])
        grounded = [name for name in names if entity_occurs_in_text(name, question)]
        if not grounded:
            continue
        allowed = [
            name for name in names
            if name in grounded or any(entity_occurs_in_text(name, route) for route in route_texts)
        ]
        key = tuple(sorted(" ".join(name.casefold().split()) for name in allowed))
        if key and key not in seen_groups:
            seen_groups.add(key)
            sanitized.append(allowed)
    return sanitized
