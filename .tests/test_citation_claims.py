from ai.citation_claims import build_citation_claims, claim_ids_cover_entity_groups
from ai.citation_rendering import order_citations_by_appearance, render_structured_answer
from schemas import ChatRequest, SearchPlan, Source


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


def test_citations_and_sources_follow_first_appearance_order() -> None:
    sources = [
        Source(title="First", url="https://example.com/first"),
        Source(title="Second", url="https://example.com/second"),
        Source(title="Third", url="https://example.com/third"),
        Source(title="Fourth", url="https://example.com/fourth"),
    ]

    answer, ordered = order_citations_by_appearance("结论一[4]，结论二[2]，补充[4][1]。", sources)

    assert answer == "结论一[1]，结论二[2]，补充[1][3]。"
    assert [source.title for source in ordered] == ["Fourth", "Second", "First", "Third"]


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


def test_build_citation_claims_keeps_short_direct_fact_beside_long_noise() -> None:
    claims = build_citation_claims(
        question="How does Orb open Gate?",
        sources=[Source(
            title="Mixed evidence",
            url="https://example.com/mixed",
            evidence="This overview contains unrelated background history and cosmetic lore. Orb opens Gate.",
        )],
        eligible_source_indexes={1},
        entity_groups=[["Orb"], ["Gate"]],
    )

    assert any(claim.statement == "Orb opens Gate." for claim in claims)


def test_claim_ranking_prefers_question_relation_over_entity_only_lore() -> None:
    claims = build_citation_claims(
        question="How does Amber Relay activate Blue Gate?",
        sources=[Source(
            title="Relay guide",
            url="https://example.com/relay",
            evidence=(
                "Amber Relay was built by the old observatory. "
                "Amber Relay activates Blue Gate after receiving a charged signal."
            ),
        )],
        eligible_source_indexes={1},
        entity_groups=[["Amber Relay"]],
        max_claims=1,
    )

    assert len(claims) == 1
    assert "activates Blue Gate" in claims[0].statement


def test_claim_ranking_uses_planned_evidence_language_for_translated_question() -> None:
    claims = build_citation_claims(
        question="琥珀中继器有什么效果？",
        sources=[Source(
            title="Amber Relay guide",
            url="https://example.com/relay",
            evidence=(
                "Amber Relay is an old observatory component. "
                "A charged Amber Relay activates the Blue Gate."
            ),
        )],
        eligible_source_indexes={1},
        aliases=["Amber Relay"],
        evidence_queries=["Amber Relay activate Blue Gate"],
        max_claims=1,
    )

    assert len(claims) == 1
    assert "activates the Blue Gate" in claims[0].statement


def test_claims_drop_headings_breadcrumbs_and_truncated_search_snippets() -> None:
    claims = build_citation_claims(
        question="菈妮支线步骤",
        sources=[
            Source(
                title="Guide",
                url="https://example.com/guide",
                evidence=(
                    "# 《艾尔登法环》菈妮支线任务详细完成步骤.1\n"
                    "手机游戏 > 艾尔登法环 > 攻略\n"
                    "拿到猎杀指头刀后，回去找菈妮。"
                ),
            ),
            Source(
                title="Truncated result",
                url="https://example.com/truncated",
                evidence="你不需要做那些步骤来推进菈妮的支线任务，所以不用太 ...",
            ),
        ],
        eligible_source_indexes={1, 2},
        aliases=["菈妮"],
    )

    assert [claim.statement for claim in claims] == ["拿到猎杀指头刀后，回去找菈妮。"]


def test_malformed_structured_answer_uses_player_safe_verified_fallback() -> None:
    request = ChatRequest(game="艾尔登法环", question="菈妮支线步骤")
    sources = [Source(
        title="Guide",
        url="https://example.com/guide",
        evidence="拿到猎杀指头刀后，回去找菈妮。然后可获得颠倒沙漏。",
    )]

    rendered = render_structured_answer(
        answer="模型没有按约定输出 JSON",
        request=request,
        sources=sources,
        plan=SearchPlan(aliases=["菈妮"]),
        conservative_answer=lambda **_: "保守回答",
    )

    assert rendered.startswith("目前只能直接核实以下信息：")
    assert "已核实的资料" not in rendered
    assert "#" not in rendered
    assert "拿到猎杀指头刀后，回去找菈妮。[1]" in rendered
    assert rendered.endswith("我不会补全推测。")
