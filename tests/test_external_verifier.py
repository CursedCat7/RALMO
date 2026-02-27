"""Tests for External Verifier (factory and OpenAI verifier)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ralmo_core.external.verifier_adapter import (
    StubVerifier,
    create_verifier,
)


class TestStubVerifier:
    """Tests for StubVerifier."""

    def test_always_returns_unverified(self) -> None:
        v = StubVerifier()
        result = v.verify("test prompt", "test output")
        assert result.verified is False
        assert result.confidence == 0.0

    def test_not_available(self) -> None:
        v = StubVerifier()
        assert v.is_available() is False


class TestVerifierFactory:
    """Tests for create_verifier factory."""

    def test_create_stub(self) -> None:
        v = create_verifier(provider="stub")
        assert isinstance(v, StubVerifier)

    def test_create_stub_default(self) -> None:
        v = create_verifier()
        assert isinstance(v, StubVerifier)

    def test_create_openai(self) -> None:
        v = create_verifier(provider="openai", api_key="test-key")
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        assert isinstance(v, OpenAIVerifier)
        assert v.is_available() is True

    def test_unknown_provider_falls_back_to_stub(self) -> None:
        v = create_verifier(provider="anthropic")
        assert isinstance(v, StubVerifier)


class TestOpenAIVerifier:
    """Tests for OpenAIVerifier with mocked API."""

    def test_is_available_with_key(self) -> None:
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="test-key-123")
        assert v.is_available() is True

    def test_is_not_available_without_key(self) -> None:
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="")
        assert v.is_available() is False

    def test_parse_valid_json(self) -> None:
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="test")
        result = v._parse_response(
            '{"verified": true, "confidence": 0.95, "alternative": null}'
        )
        assert result.verified is True
        assert result.confidence == 0.95

    def test_parse_invalid_json(self) -> None:
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="test")
        result = v._parse_response("not valid json")
        assert result.verified is False
        assert result.confidence == 0.0

    def test_verify_with_mocked_api(self) -> None:
        """Test verify() with a fully mocked OpenAI client."""
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="test-key", model="gpt-4o-mini")

        # Build mock response chain
        mock_message = MagicMock()
        mock_message.content = (
            '{"verified": true, "confidence": 0.92, '
            '"alternative": null}'
        )
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # Inject mock client
        v._client = mock_client

        result = v.verify("What is 2+2?", "4")
        assert result.verified is True
        assert result.confidence == 0.92
        assert result.metadata["provider"] == "openai"
        mock_client.chat.completions.create.assert_called_once()

    def test_verify_retry_on_failure(self) -> None:
        """Test that verify retries on API failure."""
        from ralmo_core.external.openai_verifier import OpenAIVerifier

        v = OpenAIVerifier(api_key="test-key", max_retries=2)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("API error"),
            RuntimeError("API error"),
        ]
        v._client = mock_client

        with patch("time.sleep"):  # Don't actually wait
            result = v.verify("test", "test")

        assert result.verified is False
        assert result.metadata["error"] == "max_retries_exceeded"  # type: ignore
        assert mock_client.chat.completions.create.call_count == 2
