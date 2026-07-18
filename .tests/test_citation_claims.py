from ai.citation_claims import build_citation_claims
from schemas import Source


def test_build_citation_claims_keeps_source_and_sentence_boundaries() -> None:
    claims = build_citation_claims(
        question="Where is Moonstone acquired, and is there a telescope puzzle?",
        sources=[
            Source(
                title="Moonstone route",
                url="https://example.com/moonstone",
                evidence=(
                    "Moonstone is acquired from the observatory chest. "
                    "The chest opens after the telescope puzzle. "
                    "A nearby merchant sells healing potions."
                ),
            ),
            Source(
                title="Combat overview",
                url="https://example.com/combat",
                evidence="Combat tips and enemy behavior are explained here.",
            ),
        ],
        eligible_source_indexes={1},
    )

    assert {claim.claim_id for claim in claims} == {"C1_1", "C1_2"}
    assert [claim.source_index for claim in claims] == [1, 1]
    assert {claim.statement for claim in claims} == {
        "Moonstone is acquired from the observatory chest.",
        "The chest opens after the telescope puzzle.",
    }
    assert all("merchant" not in claim.statement for claim in claims)


def test_build_citation_claims_never_uses_ineligible_source() -> None:
    claims = build_citation_claims(
        question="Where is Moonstone acquired?",
        sources=[
            Source(
                title="Moonstone route",
                url="https://example.com/moonstone",
                evidence="Moonstone is acquired from the observatory chest.",
            ),
        ],
        eligible_source_indexes=set(),
    )

    assert claims == []
