"""Page-passage extraction independent of remote search execution."""

import re

from query_tokens import question_relevance_tokens


def best_evidence_passage(content: str, *, question: str, max_chars: int = 1600) -> str:
    cleaned = re.sub(r"\s+", " ", content).strip()
    if not cleaned or len(cleaned) <= max_chars:
        return cleaned
    tokens = question_relevance_tokens(question)
    anchors = evidence_anchor_phrases(question)
    candidates: list[str] = [cleaned[:max_chars]]
    lowered = cleaned.lower()
    for needle in [*anchors, *tokens]:
        start = 0
        for _ in range(20):
            position = lowered.find(needle, start)
            if position < 0:
                break
            candidates.append(evidence_window(cleaned, focus=position, max_chars=max_chars))
            start = position + len(needle)
    focused = max(
        candidates,
        key=lambda passage: (
            sum(anchor in passage.lower() for anchor in anchors),
            sum(token in passage.lower() for token in tokens),
            sum(passage.lower().count(token) for token in tokens),
        ),
    )
    return combine_page_lead(cleaned, focused=focused, anchors=anchors, tokens=tokens, max_chars=max_chars)


def combine_page_lead(content: str, *, focused: str, anchors: list[str], tokens: list[str], max_chars: int) -> str:
    if content.find(focused[: min(120, len(focused))]) < max_chars // 2:
        return focused[:max_chars]
    lead_budget = max_chars * 3 // 5
    lead = content[:lead_budget]
    boundary = max((lead.rfind(mark) for mark in ".!?。！？;；"), default=-1)
    if boundary >= lead_budget // 2:
        lead = lead[: boundary + 1]
    remaining = max_chars - len(lead) - 2
    if remaining <= 0:
        return lead[:max_chars]
    lowered = focused.casefold()
    positions = [lowered.find(value) for value in [*anchors, *tokens] if value]
    detail = evidence_window(focused, focus=min((value for value in positions if value >= 0), default=0), max_chars=remaining)
    return f"{lead}\n\n{detail}"[:max_chars]


def evidence_window(content: str, *, focus: int, max_chars: int) -> str:
    search_start = max(0, focus - max_chars // 2)
    prefix = content[search_start:focus]
    boundary = max((prefix.rfind(mark) for mark in ".!?。！？;；"), default=-1)
    start = search_start + boundary + 1 if boundary >= 0 else search_start
    while start < focus and content[start].isspace():
        start += 1
    return content[start : start + max_chars].strip()


def evidence_anchor_phrases(value: str) -> list[str]:
    stop_words = {"access", "enter", "exact", "find", "guide", "how", "into", "location", "outside", "requirements", "route", "the", "to", "where"}
    words = re.findall(r"[a-z][a-z'-]*|[a-z]*\d[a-z0-9._-]*|\d{1,6}", value.casefold())
    anchors: list[str] = []
    for index, word in enumerate(words):
        if not any(char.isdigit() for char in word):
            continue
        pairs = [(position, words[position]) for position in range(max(0, index - 2), min(len(words), index + 3)) if words[position] not in stop_words]
        local = [token for _position, token in pairs]
        identifier = next(local_index for local_index, (position, _token) in enumerate(pairs) if position == index)
        for start in range(max(0, identifier - 2), identifier + 1):
            for end in range(identifier + 1, min(len(local), identifier + 3) + 1):
                phrase = " ".join(local[start:end])
                if phrase != word and phrase not in anchors:
                    anchors.append(phrase)
    return sorted(anchors, key=len, reverse=True)[:12]
