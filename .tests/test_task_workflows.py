import pytest

from retrieval.artifacts import RetrievalOutcome
from schemas import ChatRequest, GameResolution, InvestigationState, SearchPlan, Source
from workflow import WorkflowRouter
from workflows.analysis import AnalysisWorkflow
from workflows.build import BuildWorkflow
from workflows.guide import GuideWorkflow


@pytest.mark.parametrize(
    ("workflow_type", "intent"),
    [(GuideWorkflow, "quest_step"), (BuildWorkflow, "build"), (AnalysisWorkflow, "game_mechanic")],
)
async def test_task_workflows_preserve_retrieval_evidence_and_cited_answer(workflow_type, intent) -> None:
    request = ChatRequest(game="Example Adventure", question="How does the Moon Key work?")
    plan = SearchPlan(intent=intent)
    source = Source(title="Moon Key", url="https://example.com/moon-key", evidence="The Moon Key opens the gate.")

    async def retrieve(**kwargs):
        return (
            RetrievalOutcome(
                sources=[source], plan=kwargs["plan"],
                investigation=InvestigationState(goal=request.question, complete=True),
                refined=False, stages=[],
            ),
            kwargs["game_resolution"],
        )

    async def render(**kwargs):
        assert kwargs["sources"] == [source]
        return "The Moon Key opens the gate. [1]"

    workflow = workflow_type(
        retrieve_after_identity_check=retrieve,
        render_answer=render,
        safety_refusal_message=lambda: "refused",
        verification_router=WorkflowRouter(),
    )
    state = await workflow.run(
        request=request, history=[], game_resolution=GameResolution(input_name=request.game),
        search_plan=plan, timings_ms={}, agent_trace=[],
    )

    assert state["evidence"] == [source]
    assert state["answer"].endswith("[1]")
