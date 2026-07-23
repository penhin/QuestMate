from runtime import QuestRuntime


async def test_runtime_records_usage_without_exposing_request_content() -> None:
    class Result:
        usage = {"model_calls": 2, "investigation_hops": 1}

    runtime = QuestRuntime()
    result = await runtime.execute(
        user_id="player-1",
        tools={"search": object()},
        operation=lambda: _result(),
    )

    assert result.usage["model_calls"] == 2


async def _result():
    class Result:
        usage = {"model_calls": 2, "investigation_hops": 1}
    return Result()
