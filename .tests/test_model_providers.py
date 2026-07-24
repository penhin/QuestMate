from config import Settings
from model_providers import OpenAICompatibleProvider, create_model_provider
from schemas import ChatRequest


def test_omitted_provider_uses_configured_server_deepseek_default() -> None:
    provider = create_model_provider(
        request=ChatRequest(game="Example Adventure", question="Where is the key?"),
        settings=Settings(anthropic_api_key="", deepseek_api_key="server-key", deepseek_model="deepseek-chat"),
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "deepseek-chat"
