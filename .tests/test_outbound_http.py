import pytest

import outbound_http


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/guide",
        "https://localhost/guide",
        "https://127.0.0.1/guide",
        "https://user:secret@example.com/guide",
        "https://example.com:8443/guide",
        "https://example.test/guide",
    ],
)
def test_public_url_syntax_rejects_local_or_credentialed_targets(value: str) -> None:
    assert outbound_http.normalized_public_https_url(value) is None


def test_public_url_syntax_normalizes_host_and_removes_fragment() -> None:
    assert (
        outbound_http.normalized_public_https_url(
            "https://EXAMPLE.com:443/guides/item?q=1#private-fragment"
        )
        == "https://example.com/guides/item?q=1"
    )


async def test_public_url_validation_requires_exclusively_public_dns(monkeypatch) -> None:
    monkeypatch.setattr(outbound_http, "resolves_to_public_addresses", lambda _host: False)

    with pytest.raises(ValueError, match="public addresses"):
        await outbound_http.validate_public_https_url("https://example.com/guide")


async def test_public_url_validation_returns_normalized_url(monkeypatch) -> None:
    monkeypatch.setattr(outbound_http, "resolves_to_public_addresses", lambda _host: True)

    assert (
        await outbound_http.validate_public_https_url("https://EXAMPLE.com/guide#section")
        == "https://example.com/guide"
    )
