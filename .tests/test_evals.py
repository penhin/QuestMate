from pathlib import Path

from evals.dataset import dataset_metadata, filter_cases, load_cases
from evals.run_evals import DEFAULT_CASES
from evals.scoring import evaluate_case, summarize


def test_evaluation_suite_has_diverse_unique_cases() -> None:
    cases = load_cases(DEFAULT_CASES)

    assert len(cases) == 52
    assert len({case["id"] for case in cases}) == len(cases)
    assert {"patch", "boss_strategy", "quest_step", "prompt_injection"}.issubset(
        {case["category"] for case in cases}
    )
    assert len(filter_cases(cases, tier="niche")) >= 21
    assert len(filter_cases(cases, split="validation")) >= 10


def test_dataset_metadata_is_reproducible() -> None:
    cases = load_cases(DEFAULT_CASES)
    metadata = dataset_metadata(DEFAULT_CASES, cases)

    assert metadata["case_count"] == 52
    assert len(metadata["sha256"]) == 64
    assert metadata["by_tier"]["niche"] >= 21


def test_dataset_rejects_invalid_split(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"bad","game":"Game","question":"Question","expected_behavior":"answer","split":"hidden"}\n',
        encoding="utf-8",
    )

    try:
        load_cases(path)
    except ValueError as exc:
        assert "invalid split" in str(exc)
    else:
        raise AssertionError("invalid split was accepted")


def test_evaluation_rejects_undated_non_official_patch_answer() -> None:
    case = {
        "id": "patch",
        "expected_behavior": "conservative_or_versioned",
        "expected_source_types": ["official"],
        "required_terms": [],
        "requires_official_versioned_source": True,
    }
    response = {
        "answer": "当前版本已经削弱。",
        "sources": [{"title": "Community patch summary", "url": "https://example.com/patch", "source_type": "community"}],
    }

    result = evaluate_case(case, response)

    assert result["version_policy_pass"] is False
    assert result["passed"] is False


def test_evaluation_accepts_dated_official_patch_answer() -> None:
    case = {
        "id": "patch",
        "expected_behavior": "conservative_or_versioned",
        "expected_source_types": ["official"],
        "required_terms": ["削弱"],
        "requires_official_versioned_source": True,
    }
    response = {
        "answer": "1.12 版本削弱了对应技能。[1]",
        "sources": [{"title": "Official patch notes", "url": "https://example.com/patch", "source_type": "official", "game_version": "1.12"}],
    }

    assert evaluate_case(case, response)["passed"] is True


def test_evaluation_requires_answer_terms_and_valid_citations() -> None:
    case = {"id": "boss", "expected_behavior": "answer", "expected_source_types": ["wiki"], "required_terms": ["玛莲妮亚"]}
    source = {"title": "玛莲妮亚", "url": "https://example.com/guide", "source_type": "wiki"}

    missing_term = evaluate_case(case, {"answer": "保持距离。[1]", "sources": [source]})
    bad_citation = evaluate_case(case, {"answer": "玛莲妮亚保持距离。[2]", "sources": [source]})
    passing = evaluate_case(case, {"answer": "玛莲妮亚需要保持距离。[1]", "sources": [source]})

    assert missing_term["required_terms_pass"] is False
    assert bad_citation["citation_pass"] is False
    assert passing["passed"] is True


def test_evaluation_measures_source_and_evidence_recall_separately() -> None:
    case = {
        "id": "retrieval-gold",
        "expected_behavior": "answer",
        "expected_source_types": ["wiki"],
        "expected_source_urls": ["example.com/wiki/exact_entity"],
        "evidence_terms": ["required key", "hidden room"],
        "required_answer_groups": [["隐藏入口", "hidden entrance"], ["钥匙", "key"]],
        "required_terms": ["结论"],
    }
    correct_source = {
        "title": "Exact Entity",
        "url": "https://example.com/wiki/Exact_Entity",
        "source_type": "wiki",
        "evidence": "The required key opens the hidden room.",
    }

    passing = evaluate_case(case, {"answer": "结论：先拿钥匙，再走隐藏入口。[1]", "sources": [correct_source]})
    wrong_page = evaluate_case(
        case,
        {
            "answer": "结论：先拿钥匙，再走隐藏入口。[1]",
            "sources": [{**correct_source, "url": "https://example.com/wiki/index"}],
        },
    )
    missing_evidence = evaluate_case(
        case,
        {"answer": "结论：先拿钥匙，再走隐藏入口。[1]", "sources": [{**correct_source, "evidence": "Generic overview."}]},
    )
    incomplete_chain = evaluate_case(case, {"answer": "结论：拿到钥匙。[1]", "sources": [correct_source]})

    assert passing["source_recall_pass"] is True
    assert passing["evidence_recall_pass"] is True
    assert wrong_page["source_recall_pass"] is False
    assert missing_evidence["evidence_recall_pass"] is False
    assert incomplete_chain["action_chain_pass"] is False


def test_answer_with_late_uncertainty_note_is_not_a_conservative_refusal() -> None:
    case = {
        "id": "qualified-answer",
        "expected_behavior": "answer",
        "expected_source_types": ["wiki"],
        "required_terms": ["mantra"],
    }
    response = {
        "answer": "Mantra 系统需要先取得石板。[1]" + "具体步骤。" * 60 + "仍有少量不确定部分。",
        "sources": [{"title": "Mantra", "url": "https://example.com/mantra", "source_type": "wiki"}],
    }

    assert evaluate_case(case, response)["behavior_pass"] is True


def test_summary_reports_quality_dimensions_and_segments() -> None:
    results = [
        {"case": {"category": "boss", "split": "dev", "tier": "mainstream", "difficulty": "standard"}, "evaluation": {"passed": True, "answer_present": True, "source_count": 2}, "latency_ms": 100},
        {"case": {"category": "boss", "split": "validation", "tier": "niche", "difficulty": "hard"}, "evaluation": {"passed": False, "answer_present": True, "source_count": 0}, "latency_ms": 300},
    ]

    summary = summarize(results)

    assert summary["pass_rate"] == 0.5
    assert summary["average_source_count"] == 1
    assert summary["by_tier"]["niche"]["pass_rate"] == 0
    assert summary["dimension_pass_rates"]["answer_present"] == 1
