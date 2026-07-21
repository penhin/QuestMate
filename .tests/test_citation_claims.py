from ai.citation_claims import build_citation_claims, claim_ids_cover_entity_groups
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


def test_build_citation_claims_accepts_translated_alias_without_new_entity_requirement() -> None:
    claims = build_citation_claims(
        question="青石钥匙如何使用？",
        sources=[Source(
            title="Stone Key guide",
            url="https://example.com/stone-key",
            evidence="The Stone Key opens the observatory lock.",
        )],
        eligible_source_indexes={1},
        entity_groups=[["青石钥匙"]],
        aliases=["Stone Key"],
    )

    assert [claim.source_index for claim in claims] == [1]
    assert "Stone Key" in claims[0].statement


def test_claim_group_coverage_requires_every_relation_endpoint() -> None:
    claims = build_citation_claims(
        question="Does Quartz Relay activate Azure Gate?",
        sources=[
            Source(title="Relay note", url="https://example.com/relay", evidence="Quartz Relay needs a charged core."),
            Source(title="Gate note", url="https://example.com/gate", evidence="Azure Gate opens after the relay signal."),
        ],
        eligible_source_indexes={1, 2},
        entity_groups=[["Quartz Relay"], ["Azure Gate"]],
    )

    assert not claim_ids_cover_entity_groups(
        claims=claims,
        claim_ids=["C1_1"],
        entity_groups=[["Quartz Relay"], ["Azure Gate"]],
    )
    assert claim_ids_cover_entity_groups(
        claims=claims,
        claim_ids=["C1_1", "C2_1"],
        entity_groups=[["Quartz Relay"], ["Azure Gate"]],
    )


def test_build_citation_claims_retains_adjacent_relation_context() -> None:
    claims = build_citation_claims(
        question="Does Quartz Relay activate Azure Gate?",
        sources=[Source(
            title="Signal route",
            url="https://example.com/signal",
            evidence="Quartz Relay sends a signal when charged. The Azure Gate opens after that signal.",
        )],
        eligible_source_indexes={1},
        entity_groups=[["Quartz Relay"], ["Azure Gate"]],
        max_claims=1,
    )

    assert len(claims) == 1
    assert "Quartz Relay" in claims[0].statement
    assert "Azure Gate" in claims[0].statement
