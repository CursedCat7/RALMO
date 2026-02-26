"""Draft model wrapper for RALMO speculative decoding.

The draft model is a lightweight, GPU-accelerated model (typically quantized Q4_K_M)
that generates candidate tokens quickly. These candidates are then verified
by the target model.
"""

from __future__ import annotations

import logging
from typing import Any

from llama_cpp import Llama

from ralmo_core.models.base_model import BaseModel, GenerationResult, KVSnapshot

logger = logging.getLogger(__name__)


class DraftModel(BaseModel):
    """Draft model for speculative decoding.

    Runs on GPU with partial layer offloading. Generates candidate tokens
    that the target model will verify. Uses temperature-based sampling
    for diverse draft proposals.

    Attributes:
        model: The underlying llama_cpp.Llama instance.
        model_path: Path to the loaded GGUF model file.
        n_gpu_layers: Number of layers offloaded to GPU.
    """

    def __init__(self) -> None:
        self.model: Llama | None = None
        self.model_path: str = ""
        self.n_gpu_layers: int = -1
        self._n_ctx: int = 2048

    def load(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Load a draft model from a GGUF file.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            seed: Random seed for reproducibility.
            **kwargs: Additional llama_cpp.Llama parameters.
        """
        logger.info("Loading draft model from %s (n_gpu_layers=%d)", model_path, n_gpu_layers)
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
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
        logger.info("Draft model loaded successfully.")

    def prefill(self, tokens: list[int]) -> None:
        """Evaluate tokens and populate the KV cache.

        Args:
            tokens: Input token IDs.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")
        self.model.eval(tokens)

    def generate(
        self,
        k: int,
        temperature: float = 0.8,
    ) -> GenerationResult:
        """Generate k draft tokens autoregressively from current KV cache.

        Args:
            k: Number of tokens to generate.
            temperature: Sampling temperature for diversity.

        Returns:
            GenerationResult with generated tokens and logprobs.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")

        generated_tokens: list[int] = []
        logprobs: list[float] = []

        for _ in range(k):
            # Sample next token
            token = self.model.sample(
                temp=temperature,
                top_k=40,
                top_p=0.95,
            )

            # Extract logprob for the sampled token
            scores = self.model.scores  # type: ignore[attr-defined]
            if scores is not None and len(scores) > 0:
                import math

                last_logits = scores[-1]
                # Compute log-softmax for the sampled token
                max_logit = max(last_logits)
                log_sum_exp = max_logit + math.log(
                    sum(math.exp(x - max_logit) for x in last_logits)
                )
                token_logprob = float(last_logits[token]) - log_sum_exp
                logprobs.append(token_logprob)
            else:
                logprobs.append(0.0)

            generated_tokens.append(token)

            # Check for EOS
            if token == self.model.token_eos():
                return GenerationResult(
                    tokens=generated_tokens,
                    logprobs=logprobs,
                    finish_reason="eos",
                )

            # Feed the token back for next iteration
            self.model.eval([token])

        return GenerationResult(
            tokens=generated_tokens,
            logprobs=logprobs,
            finish_reason="length",
        )

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        """Compute log-probabilities and entropies of draft tokens.

        Args:
            draft_tokens: Draft token IDs to evaluate.

        Returns:
            Tuple of (logprobs, entropies) for each draft token.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")

        import math

        self.model.eval(draft_tokens)

        logprobs: list[float] = []
        entropies: list[float] = []
        scores = self.model.scores  # type: ignore[attr-defined]

        if scores is None:
            return [0.0] * len(draft_tokens), [0.0] * len(draft_tokens)

        # Extract logprobs and entropies for the newly evaluated tokens
        # Note: In llama-cpp-python, self.model.scores contains the logits
        # for the tokens added in the most recent eval() call.
        # Shape: (len(draft_tokens), vocab_size).

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
        """Generate a single token from current KV cache."""
        if self.model is None:
            raise RuntimeError("Draft model not loaded.")

        import math

        token = self.model.sample(temp=0.8)

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

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")

        try:
            state = self.model.save_state()
            n_tokens = self.model.n_tokens
            return KVSnapshot(
                state=state,
                token_count=n_tokens,
                metadata={"model": "draft", "path": self.model_path},
            )
        except Exception:
            logger.warning("Native KV snapshot not available, returning empty snapshot.")
            return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        """Restore KV cache from a snapshot.

        Args:
            snapshot: Previously saved KVSnapshot.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")

        if snapshot.state is not None:
            try:
                self.model.load_state(snapshot.state)
            except Exception:
                logger.warning("Failed to restore KV state, resetting model.")
                self.model.reset()

    def reset(self) -> None:
        """Reset model state and clear KV cache."""
        if self.model is not None:
            self.model.reset()

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")
        tokens = self.model.tokenize(text.encode("utf-8"))
        return list(tokens)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text.

        Args:
            tokens: Token IDs.

        Returns:
            Decoded text.
        """
        if self.model is None:
            raise RuntimeError("Draft model not loaded. Call load() first.")
        text_bytes: bytes = self.model.detokenize(tokens)
        return text_bytes.decode("utf-8", errors="replace")
