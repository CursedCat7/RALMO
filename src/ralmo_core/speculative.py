"""Speculative decoding engine for RALMO.

Implements the core linear speculative decoding loop:
1. Draft model generates k candidate tokens
2. Target model verifies via logprob comparison
3. Accept prefix up to first mismatch
4. Target generates correct token at mismatch point
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import BaseModel
from ralmo_core.policy import AcceptDecision, BasePolicy

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeStats:
    """Statistics for a single speculative decoding run.

    Attributes:
        total_tokens: Total tokens generated in the output.
        draft_tokens_proposed: Total draft tokens proposed across all iterations.
        draft_tokens_accepted: Total draft tokens accepted.
        draft_tokens_rejected: Total draft tokens rejected.
        iterations: Number of speculative loop iterations.
        target_corrections: Number of times target produced a correction token.
        latency_ms: Total wall-clock time in milliseconds.
        acceptance_rate: Fraction of draft tokens accepted.
    """

    total_tokens: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    draft_tokens_rejected: int = 0
    iterations: int = 0
    target_corrections: int = 0
    latency_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Compute the draft token acceptance rate."""
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed


@dataclass
class SpeculativeResult:
    """Result of a speculative decoding session.

    Attributes:
        tokens: The full list of generated token IDs.
        text: Decoded text output.
        stats: Performance statistics.
        finish_reason: Why generation stopped.
    """

    tokens: list[int] = field(default_factory=list)
    text: str = ""
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)
    finish_reason: str = "length"


class SpeculativeEngine:
    """Linear speculative decoding engine.

    Coordinates the draft-verify-accept/reject loop between a draft model
    and a target model. Uses a policy module to determine acceptance
    thresholds and a KV manager for cache management.

    Attributes:
        draft_model: The fast draft model for candidate generation.
        target_model: The authoritative target model for verification.
        policy: The acceptance policy (static or adaptive).
        kv_manager: KV cache manager for snapshot/restore.
    """

    def __init__(
        self,
        draft_model: BaseModel,
        target_model: BaseModel,
        policy: BasePolicy,
        kv_manager: KVManager,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.policy = policy
        self.kv_manager = kv_manager

    def run(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 512,
        draft_temperature: float = 0.8,
    ) -> SpeculativeResult:
        """Run the speculative decoding loop.

        Core algorithm:
            while not finished:
                draft_tokens = draft.generate(k)
                target_logprobs = target.verify(draft_tokens)
                decision = policy.should_accept(target_logprobs)
                if all accepted:
                    commit_all()
                else:
                    commit_prefix(accepted_count)
                    target.generate_from_mismatch()

        Args:
            prompt_tokens: Input prompt as token IDs.
            max_tokens: Maximum number of tokens to generate.
            draft_temperature: Temperature for draft model sampling.

        Returns:
            SpeculativeResult with generated tokens, text, and stats.
        """
        start_time = time.perf_counter()
        stats = SpeculativeStats()

        # Initial KV prefill for both models
        try:
            self.draft_model.prefill(prompt_tokens)
            self.target_model.prefill(prompt_tokens)
        except Exception as e:
            logger.error("Failed to prefill models: %s", e)
            raise

        # Working context: starts with prompt, grows as tokens are committed
        context = list(prompt_tokens)
        generated: list[int] = []
        finish_reason = "length"

        k = self.policy.get_k()

        while len(generated) < max_tokens:
            stats.iterations += 1
            remaining = max_tokens - len(generated)
            current_k = min(k, remaining)

            if current_k <= 0:
                break

            # Snapshot both models before drafting
            self.kv_manager.snapshot(self.draft_model, "draft")
            self.kv_manager.snapshot(self.target_model, "target")

            # --- Step 1: Draft generates k candidate tokens ---
            logger.debug("Iteration %d: Draft generating %d tokens", stats.iterations, current_k)
            draft_result = self.draft_model.generate(
                k=current_k,
                temperature=draft_temperature,
            )
            draft_tokens = draft_result.tokens
            stats.draft_tokens_proposed += len(draft_tokens)

            if not draft_tokens:
                logger.warning("Draft model produced no tokens, stopping.")
                finish_reason = "draft_empty"
                break

            # --- Step 2: Target verifies draft tokens ---
            target_logprobs, target_entropies = self.target_model.verify(
                draft_tokens=draft_tokens,
            )

            # --- Step 3: Policy decides acceptance ---
            decision: AcceptDecision = self.policy.should_accept(target_logprobs, target_entropies)

            logger.debug(
                "Acceptance decision: %d/%d tokens accepted",
                decision.accepted_count,
                len(draft_tokens),
            )

            # --- Step 4a: Commit accepted prefix ---
            accepted_tokens = draft_tokens[: decision.accepted_count]
            if accepted_tokens:
                context.extend(accepted_tokens)
                generated.extend(accepted_tokens)
                stats.draft_tokens_accepted += len(accepted_tokens)

                # Check for EOS in accepted tokens
                if self._contains_eos(accepted_tokens):
                    finish_reason = "eos"
                    # Target's KV cache is aligned up to the EOS since it evaluated the draft string
                    # But we'll break immediately.
                    break

            # --- Step 4b: Handle rejection & Resync KV State ---
            rejected_count = len(draft_tokens) - decision.accepted_count
            stats.draft_tokens_rejected += rejected_count

            if decision.accepted_count < len(draft_tokens):
                # 1. Target Rollback
                target_restored = self.kv_manager.restore(self.target_model, "target")
                if target_restored:
                    if accepted_tokens:
                        self.target_model.prefill(accepted_tokens)
                else:
                    self.target_model.prefill(context)

                # 2. Target generates the correct token at the mismatch point
                correction_token, correction_logprob = self.target_model.generate_single()  # type: ignore[attr-defined]
                context.append(correction_token)
                generated.append(correction_token)
                stats.target_corrections += 1
                stats.total_tokens = len(generated)

                logger.debug(
                    "Target correction: token=%d, logprob=%.4f",
                    correction_token,
                    correction_logprob,
                )

                if self._is_eos(correction_token):
                    finish_reason = "eos"
                    break

                # 3. Draft Rollback (Sync with target)
                draft_restored = self.kv_manager.restore(self.draft_model, "draft")
                draft_sync_tokens = accepted_tokens + [correction_token]
                if draft_restored:
                    if draft_sync_tokens:
                        self.draft_model.prefill(draft_sync_tokens)
                else:
                    self.draft_model.prefill(context)

            else:
                # All draft tokens accepted; Target and Draft KV caches are perfectly aligned!
                stats.total_tokens = len(generated)

            # Check EOS in draft result
            if draft_result.finish_reason == "eos" and decision.accepted_count == len(draft_tokens):
                finish_reason = "eos"
                break

        # Finalize stats
        end_time = time.perf_counter()
        stats.total_tokens = len(generated)
        stats.latency_ms = (end_time - start_time) * 1000.0

        # Decode text
        try:
            text = self.target_model.detokenize(generated)
        except Exception:
            text = ""

        logger.info(
            "Speculative decoding complete: %d tokens, %.1fms, %.1f%% acceptance",
            stats.total_tokens,
            stats.latency_ms,
            stats.acceptance_rate * 100,
        )

        return SpeculativeResult(
            tokens=generated,
            text=text,
            stats=stats,
            finish_reason=finish_reason,
        )

    def _contains_eos(self, tokens: list[int]) -> bool:
        """Check if any token in the list is an EOS token."""
        try:
            eos_token = int(self.target_model.model.token_eos())  # type: ignore[attr-defined]
            return eos_token in tokens
        except Exception:
            return False

    def _is_eos(self, token: int) -> bool:
        """Check if a single token is the EOS token."""
        try:
            return token == int(self.target_model.model.token_eos())  # type: ignore[attr-defined]
        except Exception:
            return False
