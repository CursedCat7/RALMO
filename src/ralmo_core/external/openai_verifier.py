"""OpenAI-based external verifier for cloud LLM escalation.

Implements the ExternalVerifier interface using OpenAI's Chat Completions
API. Used when the local speculative engine signals low confidence
(should_escalate=True) to verify output quality via a cloud model.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ralmo_core.external.verifier_adapter import (
    ExternalVerifier,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Verification prompt template
VERIFY_PROMPT = (
    "You are a verification assistant. Given an input prompt "
    "and a locally-generated response, evaluate whether the "
    "response is accurate, coherent, and complete.\n\n"
    "**Input Prompt:**\n{prompt}\n\n"
    "**Local Model Response:**\n{local_output}\n\n"
    "**Instructions:**\n"
    "1. Assess the quality (accuracy, completeness, coherence).\n"
    "2. If the response is adequate, set \"verified\" to true.\n"
    "3. If it has issues, set \"verified\" to false and provide "
    "an improved response.\n"
    '4. Rate your confidence from 0.0 to 1.0.\n\n'
    "Respond in this exact JSON format:\n"
    '{{"verified": true/false, "confidence": 0.0-1.0, '
    '"alternative": "improved response or null"}}'
)


class OpenAIVerifier(ExternalVerifier):
    """External verifier using OpenAI Chat Completions API.

    Sends the local model's output to a cloud LLM for quality
    verification. Supports configurable model selection and
    retry logic.

    Attributes:
        model: The OpenAI model to use for verification.
        max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ) -> None:
        """Initialize OpenAI verifier.

        Args:
            api_key: OpenAI API key. If None, reads from
                     OPENAI_API_KEY environment variable.
            model: Model identifier for verification.
            max_retries: Number of retries on failure.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._max_retries = max_retries
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                logger.error(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
                raise
        return self._client

    def verify(
        self, prompt: str, local_output: str
    ) -> VerificationResult:
        """Verify local output using OpenAI API.

        Args:
            prompt: The original input prompt.
            local_output: The locally generated text to verify.

        Returns:
            VerificationResult with verification outcome.
        """
        verification_prompt = VERIFY_PROMPT.format(
            prompt=prompt,
            local_output=local_output,
        )

        for attempt in range(self._max_retries):
            try:
                client = self._ensure_client()
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a verification assistant.",
                        },
                        {"role": "user", "content": verification_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or ""
                return self._parse_response(content)

            except Exception as e:
                logger.warning(
                    "OpenAI verification attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries - 1:
                    wait = 2**attempt
                    time.sleep(wait)

        logger.error("OpenAI verification failed after %d retries.", self._max_retries)
        return VerificationResult(
            verified=False,
            confidence=0.0,
            metadata={"provider": "openai", "error": "max_retries_exceeded"},
        )

    def _parse_response(self, content: str) -> VerificationResult:
        """Parse the LLM's JSON response into a VerificationResult."""
        import json

        try:
            data = json.loads(content)
            return VerificationResult(
                verified=bool(data.get("verified", False)),
                alternative_text=data.get("alternative"),
                confidence=float(data.get("confidence", 0.0)),
                metadata={"provider": "openai", "model": self._model},
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                "Failed to parse OpenAI response: %s. Raw: %s",
                e,
                content[:200],
            )
            return VerificationResult(
                verified=False,
                confidence=0.0,
                alternative_text=content,
                metadata={
                    "provider": "openai",
                    "parse_error": str(e),
                },
            )

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self._api_key)
