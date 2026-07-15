"""Dataset loading, validation, filtering, and reproducibility metadata."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


VALID_SPLITS = {"dev", "validation"}
VALID_TIERS = {"mainstream", "niche", "safety"}
VALID_DIFFICULTIES = {"standard", "hard"}
VALID_BEHAVIORS = {
    "answer",
    "confirmation",
    "conservative",
    "confirmation_or_conservative",
    "safe_refusal",
    "conservative_or_versioned",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        case = json.loads(line)
        for field in ("id", "game", "question", "expected_behavior"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"{path}:{line_number} is missing {field}")
        if case["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number} duplicates case id {case['id']}")
        seen_ids.add(case["id"])
        case.setdefault("category", "uncategorized")
        case.setdefault("split", "dev")
        default_tier = "safety" if case["category"] in {
            "game_resolution",
            "unknown_entity",
            "prompt_injection",
            "out_of_scope",
        } else "mainstream"
        case.setdefault("tier", default_tier)
        case.setdefault("difficulty", "standard")
        if case["split"] not in VALID_SPLITS:
            raise ValueError(f"{path}:{line_number} has invalid split {case['split']}")
        if case["tier"] not in VALID_TIERS:
            raise ValueError(f"{path}:{line_number} has invalid tier {case['tier']}")
        if case["difficulty"] not in VALID_DIFFICULTIES:
            raise ValueError(f"{path}:{line_number} has invalid difficulty {case['difficulty']}")
        if case["expected_behavior"] not in VALID_BEHAVIORS:
            raise ValueError(f"{path}:{line_number} has invalid expected_behavior {case['expected_behavior']}")
        cases.append(case)
    return cases


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    split: str | None = None,
    tier: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if (split is None or case["split"] == split)
        and (tier is None or case["tier"] == tier)
        and (category is None or case["category"] == category)
    ]


def dataset_metadata(path: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": sha256(payload).hexdigest(),
        "case_count": len(cases),
        "by_split": dict(sorted(Counter(case["split"] for case in cases).items())),
        "by_tier": dict(sorted(Counter(case["tier"] for case in cases).items())),
        "by_difficulty": dict(sorted(Counter(case["difficulty"] for case in cases).items())),
        "by_category": dict(sorted(Counter(case["category"] for case in cases).items())),
    }
