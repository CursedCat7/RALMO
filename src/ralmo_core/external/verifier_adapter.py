"""External verifier adapter for cloud LLM escalation.

Stub implementation for MVP. In future phases, this will connect to
cloud APIs (OpenAI, Anthropic, etc.) for high-confidence verification
of locally uncertain outputs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from an external verifier.

    Attributes:
        verified: Whether the external model agrees with the local output.
        alternative_text: Alternative output from the external model, if any.
        confidence: Confidence score from the external model.
        metadata: Additional provider-specific metadata.
    """

    verified: bool
    alternative_text: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] | None = None


class ExternalVerifier(ABC):
    """Abstract base class for external (cloud) verifiers.

    Future implementations will connect to cloud LLM APIs for verification
    of locally uncertain outputs.
    """

    @abstractmethod
    def verify(self, prompt: str, local_output: str) -> VerificationResult:
        """Verify a local model's output using an external model.

        Args:
            prompt: The original input prompt.
            local_output: The locally generated text to verify.

        Returns:
            VerificationResult with verification outcome.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the external verifier is configured and reachable.

        Returns:
            True if the verifier can be used.
        """
        ...


class StubVerifier(ExternalVerifier):
    """No-op verifier stub for MVP.

    Always returns None / not verified. Placeholder for future cloud
    escalation integration.
    """

    def verify(self, prompt: str, local_output: str) -> VerificationResult:
        """Stub verification — always returns unverified.

        Args:
            prompt: The original input prompt (unused).
            local_output: The locally generated text (unused).

        Returns:
            VerificationResult with verified=False.
        """
        logger.debug("StubVerifier called (no-op)")
        return VerificationResult(
            verified=False,
            alternative_text=None,
            confidence=0.0,
            metadata={"provider": "stub"},
        )

    def is_available(self) -> bool:
        """Stub is never 'available' as a real verifier."""
        return False
