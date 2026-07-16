"""Create the integrity manifest for a private sealed holdout dataset.

Run this only in the restricted evaluation environment.  It intentionally does
not copy, print, or upload the dataset.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

try:
    from evals.dataset import load_cases
except ModuleNotFoundError:  # Support direct execution from repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import load_cases


DEFAULT_USAGE = "unseen generalization estimate; restricted evaluation use only"


def build_sealed_manifest(cases_path: Path, *, usage: str = DEFAULT_USAGE) -> dict[str, object]:
    """Validate a private holdout and return its content-bound manifest."""
    cases = load_cases(cases_path)
    if not cases:
        raise ValueError("sealed holdout dataset must not be empty")
    if any(case["split"] != "holdout" for case in cases):
        raise ValueError("sealed holdout dataset may contain only split=holdout cases")
    normalized_usage = usage.strip()
    if not normalized_usage:
        raise ValueError("usage must not be empty")
    return {
        "schema_version": 1,
        "dataset": cases_path.name,
        "dataset_sha256": sha256(cases_path.read_bytes()).hexdigest(),
        "holdout_integrity": {
            "status": "sealed",
            "sealed": True,
            "refresh_required": False,
            "usage": normalized_usage,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="private holdout JSONL path")
    parser.add_argument("--output", help="manifest path; defaults beside --cases")
    parser.add_argument("--usage", default=DEFAULT_USAGE)
    parser.add_argument("--force", action="store_true", help="replace an existing manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases)
    output = Path(args.output) if args.output else cases_path.with_suffix(".manifest.json")
    if output.exists() and not args.force:
        raise FileExistsError(f"manifest already exists: {output}; pass --force to replace it")
    manifest = build_sealed_manifest(cases_path, usage=args.usage)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    print(json.dumps({"manifest": str(output), "dataset_sha256": manifest["dataset_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
