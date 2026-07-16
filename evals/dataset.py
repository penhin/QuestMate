"""Dataset loading, validation, filtering, and reproducibility metadata."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


VALID_SPLITS = {"dev", "validation", "holdout"}
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
OPTIONAL_STRING_LIST_FIELDS = {
    "database_domains",
    "evidence_terms",
    "expected_source_types",
    "expected_source_urls",
    "forbidden_terms",
    "game_aliases",
    "required_terms",
}
OPTIONAL_BOOLEAN_FIELDS = {
    "require_citations",
    "requires_official_versioned_source",
    "version_sensitive",
}

UNVERIFIED_HOLDOUT_INTEGRITY = {
    "status": "unverified",
    "sealed": False,
    "refresh_required": True,
    "usage": "integrity was not declared by a dataset manifest",
}
HOLDOUT_INTEGRITY_FIELDS = {"status", "sealed", "refresh_required", "usage"}


def _validate_string_list(
    path: Path,
    line_number: int,
    case: dict[str, Any],
    field: str,
) -> None:
    value = case.get(field)
    if value is None:
        return
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{path}:{line_number} has invalid {field}; expected non-empty strings")


def _validate_retrieval_hints(path: Path, line_number: int, case: dict[str, Any]) -> None:
    aliases = case.get("game_aliases") or []
    domains = case.get("database_domains") or []
    if len(aliases) > 12:
        raise ValueError(f"{path}:{line_number} has more than 12 game_aliases")
    if len(domains) > 8:
        raise ValueError(f"{path}:{line_number} has more than 8 database_domains")

    normalized_aliases = [alias.casefold().strip() for alias in aliases]
    if len(normalized_aliases) != len(set(normalized_aliases)):
        raise ValueError(f"{path}:{line_number} has duplicate game_aliases")

    normalized_domains = [
        domain.casefold().strip().removeprefix("www.") for domain in domains
    ]
    if len(normalized_domains) != len(set(normalized_domains)):
        raise ValueError(f"{path}:{line_number} has duplicate database_domains")
    for domain in normalized_domains:
        if "." not in domain or any(
            marker in domain for marker in ("://", "/", "?", "#", " ")
        ):
            raise ValueError(
                f"{path}:{line_number} has invalid database_domains; expected bare hostnames"
            )


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
        for field in OPTIONAL_STRING_LIST_FIELDS:
            _validate_string_list(path, line_number, case, field)
        for field in OPTIONAL_BOOLEAN_FIELDS:
            if field in case and not isinstance(case[field], bool):
                raise ValueError(f"{path}:{line_number} has invalid {field}; expected a boolean")
        _validate_retrieval_hints(path, line_number, case)
        answer_groups = case.get("required_answer_groups")
        if answer_groups is not None and (
            not isinstance(answer_groups, list)
            or any(
                not isinstance(group, list)
                or not group
                or any(not isinstance(term, str) or not term.strip() for term in group)
                for group in answer_groups
            )
        ):
            raise ValueError(
                f"{path}:{line_number} has invalid required_answer_groups; expected non-empty string groups"
            )
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


def _validate_holdout_integrity(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != HOLDOUT_INTEGRITY_FIELDS:
        raise ValueError(
            f"{source} has invalid holdout_integrity; expected fields "
            f"{sorted(HOLDOUT_INTEGRITY_FIELDS)}"
        )
    if (
        not isinstance(value["status"], str)
        or not value["status"].strip()
        or not isinstance(value["usage"], str)
        or not value["usage"].strip()
        or not isinstance(value["sealed"], bool)
        or not isinstance(value["refresh_required"], bool)
    ):
        raise ValueError(f"{source} has invalid holdout_integrity field types")
    if value["sealed"] and value["refresh_required"]:
        raise ValueError(f"{source} cannot be sealed while refresh_required is true")
    return {
        "status": value["status"].strip(),
        "sealed": value["sealed"],
        "refresh_required": value["refresh_required"],
        "usage": value["usage"].strip(),
    }


def _dataset_manifest(
    path: Path,
    *,
    manifest_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    candidate = manifest_path or path.with_suffix(".manifest.json")
    if not candidate.exists():
        if manifest_path is not None:
            raise ValueError(f"dataset manifest does not exist: {candidate}")
        return None, None
    try:
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset manifest {candidate}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid dataset manifest {candidate}; expected an object")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported dataset manifest schema in {candidate}")
    declared_dataset = manifest.get("dataset")
    if declared_dataset is not None and declared_dataset != path.name:
        raise ValueError(
            f"dataset manifest {candidate} declares {declared_dataset!r}, expected {path.name!r}"
        )
    declared_sha256 = manifest.get("dataset_sha256")
    actual_sha256 = sha256(path.read_bytes()).hexdigest()
    if declared_sha256 != actual_sha256:
        raise ValueError(
            f"dataset manifest {candidate} fingerprint does not match {path}"
        )
    return manifest, candidate


def dataset_metadata(
    path: Path,
    cases: list[dict[str, Any]],
    *,
    manifest_path: Path | None = None,
    holdout_integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, resolved_manifest_path = _dataset_manifest(path, manifest_path=manifest_path)
    if holdout_integrity is not None:
        integrity = _validate_holdout_integrity(holdout_integrity, source="explicit override")
        integrity_source = "explicit_override"
    elif manifest is not None and "holdout_integrity" in manifest:
        integrity = _validate_holdout_integrity(
            manifest["holdout_integrity"], source=str(resolved_manifest_path)
        )
        integrity_source = "manifest"
    else:
        integrity = UNVERIFIED_HOLDOUT_INTEGRITY.copy()
        integrity_source = "default_unverified"
    payload = path.read_bytes()
    return {
        "schema_version": 4,
        "path": str(path),
        "sha256": sha256(payload).hexdigest(),
        "case_count": len(cases),
        "by_split": dict(sorted(Counter(case["split"] for case in cases).items())),
        "by_tier": dict(sorted(Counter(case["tier"] for case in cases).items())),
        "by_difficulty": dict(sorted(Counter(case["difficulty"] for case in cases).items())),
        "by_category": dict(sorted(Counter(case["category"] for case in cases).items())),
        "holdout_integrity": integrity,
        "holdout_integrity_source": integrity_source,
        "dataset_manifest": str(resolved_manifest_path) if resolved_manifest_path else None,
        "retrieval_hints": {
            "cases_with_database_domains": sum(bool(case.get("database_domains")) for case in cases),
            "cases_with_game_aliases": sum(bool(case.get("game_aliases")) for case in cases),
        },
        "scoring_gold": {
            "cases_with_expected_source_urls": sum(
                bool(case.get("expected_source_urls")) for case in cases
            ),
        },
    }
