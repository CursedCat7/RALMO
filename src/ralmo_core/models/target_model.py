"""Target model wrapper for RALMO speculative decoding.

The target model is the authoritative verifier that runs on CPU with persistent
KV cache. It evaluates the log-probabilities of draft tokens to decide acceptance
or rejection. When rejection occurs, it generates the correct continuation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from llama_cpp import Llama

from ralmo_core.models.base_model import BaseModel, GenerationResult, KVSnapshot

logger = logging.getLogger(__name__)


class TargetModel(BaseModel):
    """Target (verifier) model for speculative decoding.

    Runs on CPU with all layers loaded into RAM. Operates deterministically
    (temperature=0) and maintains persistent KV cache across verification
    rounds to avoid redundant computation.

    Attributes:
        model: The underlying llama_cpp.Llama instance.
        model_path: Path to the loaded GGUF model file.
    """

    def __init__(self) -> None:
        self.model: Llama | None = None
        self.model_path: str = ""
        self._n_ctx: int = 2048

    def load(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Load the target model from a GGUF file.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
            n_gpu_layers: Number of GPU layers (0 for CPU-only).
            seed: Random seed for reproducibility.
            **kwargs: Additional llama_cpp.Llama parameters.
        """
        logger.info(
            "Loading target model from %s (CPU-only, n_gpu_layers=%d)",
            model_path,
            n_gpu_layers,
        )
        self.model_path = model_path
        self._n_ctx = n_ctx
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            logits_all=True,
            verbose=False,
            **kwargs,
        )
        logger.info("Target model loaded successfully.")

    def prefill(self, tokens: list[int]) -> None:
        """Evaluate tokens and populate the KV cache.

        Args:
            tokens: Input token IDs.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")
        self.model.eval(tokens)

    def generate(
        self,
        k: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate k tokens deterministically from the target model.

        Used when the target needs to produce the correct continuation
        after a draft mismatch. Always uses greedy decoding (temperature=0).

        Args:
            k: Number of tokens to generate.
            temperature: Ignored; always uses deterministic decoding.

        Returns:
            GenerationResult with deterministically generated tokens.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")

        generated_tokens: list[int] = []
        logprobs: list[float] = []

        for _ in range(k):
            # Greedy sampling (temperature=0)
            token = self.model.sample(temp=0.0)

            # Extract logprob
            scores = self.model.scores  # type: ignore[attr-defined]
            if scores is not None and len(scores) > 0:
                last_logits = scores[-1]
                max_logit = max(last_logits)
                log_sum_exp = max_logit + math.log(
                    sum(math.exp(x - max_logit) for x in last_logits)
                )
                token_logprob = float(last_logits[token]) - log_sum_exp
                logprobs.append(token_logprob)
            else:
                logprobs.append(0.0)

            generated_tokens.append(token)

            if token == self.model.token_eos():
                return GenerationResult(
                    tokens=generated_tokens,
                    logprobs=logprobs,
                    finish_reason="eos",
                )

            self.model.eval([token])

        return GenerationResult(
            tokens=generated_tokens,
            logprobs=logprobs,
            finish_reason="length",
        )

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        """Verify draft tokens by computing their log-probabilities.

        This is the core verification step in speculative decoding.
        Evaluates the newly drafted tokens and returns the log-probability
        and entropy of each token.

        Args:
            draft_tokens: Draft-generated token IDs to verify.

        Returns:
            Tuple of (logprobs, entropies) lists.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")

        import math

        self.model.eval(draft_tokens)

        logprobs: list[float] = []
        entropies: list[float] = []
        scores = self.model.scores  # type: ignore[attr-defined]

        if scores is None:
            return [0.0] * len(draft_tokens), [0.0] * len(draft_tokens)

        for i, token in enumerate(draft_tokens):
            if i < len(scores):
                logits = scores[i]
                max_logit = max(logits)

                # Compute probabilities and entropy
                probs = [math.exp(x - max_logit) for x in logits]
                sum_probs = sum(probs)
                probs = [p / sum_probs for p in probs]

                token_logprob = math.log(probs[token])
                entropy = -sum(p * math.log(p) for p in probs if p > 0)

                logprobs.append(token_logprob)
                entropies.append(entropy)
            else:
                logprobs.append(0.0)
                entropies.append(0.0)

        return logprobs, entropies

    def generate_single(self) -> tuple[int, float]:
        """Generate a single token deterministically (for mismatch recovery).

        After a draft mismatch, the target generates the correct next token
        from its current KV cache.

        Returns:
            Tuple of (token_id, logprob).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")

        import math

        token = self.model.sample(temp=0.0)

        scores = self.model.scores  # type: ignore[attr-defined]
        logprob = 0.0
        if scores is not None and len(scores) > 0:
            last_logits = scores[-1]
            max_logit = max(last_logits)
            log_sum_exp = max_logit + math.log(sum(math.exp(x - max_logit) for x in last_logits))
            logprob = float(last_logits[token]) - log_sum_exp

        return token, logprob

    def snapshot_kv(self) -> KVSnapshot:
        """Snapshot the current KV cache state.

        Returns:
            KVSnapshot with the serialized KV state.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")

        try:
            state = self.model.save_state()
            n_tokens = self.model.n_tokens
            return KVSnapshot(
                state=state,
                token_count=n_tokens,
                metadata={"model": "target", "path": self.model_path},
            )
        except Exception:
            logger.warning("Native KV snapshot not available for target model.")
            return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        """Restore KV cache from a snapshot.

        Args:
            snapshot: Previously saved KVSnapshot.
        """
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")

        if snapshot.state is not None:
            try:
                self.model.load_state(snapshot.state)
            except Exception:
                logger.warning("Failed to restore target KV state, resetting.")
                self.model.reset()

    def reset(self) -> None:
        """Reset model state and clear KV cache."""
        if self.model is not None:
            self.model.reset()

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs."""
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")
        tokens = self.model.tokenize(text.encode("utf-8"))
        return list(tokens)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        if self.model is None:
            raise RuntimeError("Target model not loaded. Call load() first.")
        text_bytes: bytes = self.model.detokenize(tokens)
        return text_bytes.decode("utf-8", errors="replace")
