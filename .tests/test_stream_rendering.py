import pytest

from agent import QuestAgent
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, Source


@pytest.mark.asyncio
async def test_stream_answer_emits_rendered_answer_not_claim_json() -> None:
    agent = object.__new__(QuestAgent)

    async def render(**_kwargs):
        return "拉妮支线请先前往菈妮魔法师塔。[2]"

    agent._render_answer = render
    chunks = [
        chunk async for chunk in agent._stream_answer(
            request=ChatRequest(game="艾尔登法环", question="拉妮支线怎么做？", stream=True),
            sources=[Source(title="Ranni guide", url="https://example.com/ranni", evidence="Ranni quest steps")],
            plan=SearchPlan(intent="quest_step"),
            game_resolution=GameResolution(input_name="艾尔登法环"),
            history=[],
            investigation=InvestigationState(goal="拉妮支线怎么做？"),
        )
    ]

    assert chunks == ["拉妮支线请先前往菈妮魔法师塔。[2]"]
