"""Abstract base class for all RALMO model wrappers.

Defines the interface that DraftModel and TargetModel must implement,
covering generation, verification, logprob extraction, and KV cache management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVSnapshot:
    """Opaque snapshot of a model's KV cache state.

    Attributes:
        state: The raw KV state data (format depends on backend).
        token_count: Number of tokens in the KV cache at snapshot time.
        metadata: Optional additional metadata for debugging.
    """

    state: Any
    token_count: int
    metadata: dict[str, Any] | None = None


@dataclass
class GenerationResult:
    """Result of a model generation call.

    Attributes:
        tokens: List of generated token IDs.
        logprobs: Corresponding log-probabilities for each generated token.
        finish_reason: Why generation stopped (length, eos, etc.).
    """

    tokens: list[int]
    logprobs: list[float]
    finish_reason: str = "length"


class BaseModel(ABC):
    """Abstract base class for RALMO model wrappers.

    All model implementations (draft, target) must inherit from this class
    and implement the required methods for generation, verification,
    and KV cache management.
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load a model from the specified path.

        Args:
            model_path: Path to the model file (GGUF format).
            **kwargs: Additional model-specific loading parameters.
        """
        ...

    @abstractmethod
    def prefill(self, tokens: list[int]) -> None:
        """Evaluate tokens and populate the internal KV cache.

        Typically called once for the initial prompt, or after a full reset.

        Args:
            tokens: Input token IDs to evaluate.
        """
        ...

    @abstractmethod
    def generate(self, k: int, temperature: float = 0.8) -> GenerationResult:
        """Generate k tokens autoregressively, continuing from the current KV cache.

        Args:
            k: Number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            GenerationResult with generated tokens and their logprobs.
        """
        ...

    @abstractmethod
    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        """Compute log-probabilities and entropies of draft tokens.

        Continuously evaluates the given draft tokens on top of the
        current KV cache.

        Args:
            draft_tokens: The draft-generated token IDs to verify.

        Returns:
            Tuple of (logprobs, entropies), each a list parallel to draft_tokens.
        """
        ...

    @abstractmethod
    def generate_single(self) -> tuple[int, float]:
        """Generate a single token deterministically from the current KV cache.

        Used by the target model to generate the correct continuation after a
        draft mismatch.

        Returns:
            Tuple of (token_id, logprob).
        """
        ...

    @abstractmethod
    def snapshot_kv(self) -> KVSnapshot:
        """Take a snapshot of the current KV cache state.

        Returns:
            KVSnapshot that can be restored later.
        """
        ...

    @abstractmethod
    def restore_kv(self, snapshot: KVSnapshot) -> None:
        """Restore KV cache to a previously saved state.

        Args:
            snapshot: A KVSnapshot obtained from snapshot_kv().
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the model state, clearing KV cache and internal buffers."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs.

        Args:
            text: Input text string.

        Returns:
            List of token IDs.
        """
        ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text string.
        """
        ...
