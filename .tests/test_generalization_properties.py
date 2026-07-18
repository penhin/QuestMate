import pytest

from retrieval.evidence_pool import source_rank
from schemas import Source


@pytest.mark.parametrize(
    ("entity", "evidence"),
    [
        ("Aster Prism", "Aster Prism is recorded in the archive."),
        ("霜环核心", "霜环核心位于旧档案库。"),
        ("蒼星の鍵", "蒼星の鍵は記録庫にある。"),
    ],
)
def test_entity_group_ranking_is_independent_of_question_vocabulary(entity: str, evidence: str) -> None:
    matching = Source(title=entity, url="https://example.com/match", evidence=evidence)
    unrelated = Source(
        title="Unrelated record", url="https://example.com/unrelated", evidence="A separate record exists."
    )
    groups = [[entity]]

    matching_rank = source_rank(
        source=matching,
        query="arbitrary wording that does not repeat the entity",
        intent="general",
        entity_groups=groups,
    )
    unrelated_rank = source_rank(
        source=unrelated,
        query="arbitrary wording that does not repeat the entity",
        intent="general",
        entity_groups=groups,
    )

    assert matching_rank > unrelated_rank
