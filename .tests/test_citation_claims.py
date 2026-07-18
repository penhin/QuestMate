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
        "Moonstone route: Moonstone is acquired from the observatory chest.",
        "Moonstone route: The chest opens after the telescope puzzle.",
    }
    assert all("merchant" not in claim.statement for claim in claims)


def test_title_can_supply_entity_context_but_not_unrelated_claims() -> None:
    claims = build_citation_claims(
        question="How does Quartz Relay alter the archive state?",
        sources=[
            Source(
                title="Quartz Relay",
                url="https://example.com/relay",
                evidence="Activating it changes the archive state. A merchant sells potions nearby.",
            )
        ],
        eligible_source_indexes={1},
        entity_groups=[["Quartz Relay"]],
    )

    assert len(claims) == 1
    assert "archive state" in claims[0].statement


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


def test_build_citation_claims_keeps_earlier_evidence_for_equal_relevance() -> None:
    claims = build_citation_claims(
        question="How is the Azure Key used?",
        sources=[
            Source(
                title="Key guide",
                url="https://example.com/key",
                evidence=(
                    "The Azure Key is used at the north gate after speaking to the guard. "
                    "The Azure Key can also be mentioned in a later lore entry with several unrelated historical details."
                ),
            ),
        ],
        eligible_source_indexes={1},
        max_claims=1,
    )

    assert len(claims) == 1
    assert claims[0].claim_id == "C1_1"
    assert "north gate" in claims[0].statement
