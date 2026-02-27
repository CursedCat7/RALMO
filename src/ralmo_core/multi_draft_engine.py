"""Multi-Draft speculative decoding engine for RALMO.

Extends the standard SpeculativeEngine by managing multiple draft models
and using a DraftSelector to choose the best candidate before target
verification.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ralmo_core.draft_selector import BestOfNSelector, DraftSelector
from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import BaseModel, GenerationResult
from ralmo_core.policy import AcceptDecision, BasePolicy
from ralmo_core.speculative import SpeculativeResult, SpeculativeStats

logger = logging.getLogger(__name__)


class MultiDraftEngine:
    """Speculative decoding engine with multiple draft models.

    Runs k-token generation on each draft model concurrently via
    ThreadPoolExecutor, selects the best candidate via a
    DraftSelector, then verifies against the target model.

    Attributes:
        draft_models: List of draft model instances.
        target_model: The target verification model.
        policy: Acceptance policy.
        kv_manager: KV cache manager.
        selector: Strategy for choosing among draft candidates.
    """

    def __init__(
        self,
        draft_models: list[BaseModel],
        target_model: BaseModel,
        policy: BasePolicy,
        kv_manager: KVManager,
        selector: DraftSelector | None = None,
    ) -> None:
        """Initialize multi-draft engine.

        Args:
            draft_models: List of draft model instances.
            target_model: Target model for verification.
            policy: Acceptance policy.
            kv_manager: KV cache manager.
            selector: Draft selection strategy (default: BestOfNSelector).
        """
        if not draft_models:
            raise ValueError("At least one draft model is required.")

        self.draft_models = draft_models
        self.target_model = target_model
        self.policy = policy
        self.kv_manager = kv_manager
        self.selector = selector or BestOfNSelector()

    def run(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 512,
        draft_temperature: float = 0.8,
    ) -> SpeculativeResult:
        """Run the multi-draft speculative decoding loop.

        Algorithm:
            1. Each draft model generates k candidate tokens
            2. Selector picks the best candidate
            3. Target model verifies the selected draft
            4. Policy decides acceptance
            5. Commit accepted prefix, rollback rejected suffix

        Args:
            prompt_tokens: Input prompt as token IDs.
            max_tokens: Maximum number of tokens to generate.
            draft_temperature: Temperature for draft model sampling.

        Returns:
            SpeculativeResult with generated tokens, text, and stats.
        """
        start_time = time.perf_counter()
        stats = SpeculativeStats()

        # Initial KV prefill for all models
        for draft in self.draft_models:
            draft.prefill(prompt_tokens)
        self.target_model.prefill(prompt_tokens)

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

            # Snapshot all models
            for i, draft in enumerate(self.draft_models):
                self.kv_manager.snapshot(draft, f"draft_{i}")
            self.kv_manager.snapshot(self.target_model, "target")

            # Step 1: Each draft generates candidates concurrently
            candidates: list[GenerationResult] = []
            gen_k = current_k
            gen_temp = draft_temperature

            if len(self.draft_models) > 1:
                with ThreadPoolExecutor(
                    max_workers=len(self.draft_models)
                ) as executor:
                    futures = []
                    for dm in self.draft_models:
                        futures.append(
                            executor.submit(
                                dm.generate,
                                k=gen_k,
                                temperature=gen_temp,
                            )
                        )
                    candidates = [f.result() for f in futures]
            else:
                candidates = [
                    self.draft_models[0].generate(
                        k=gen_k, temperature=gen_temp
                    )
                ]

            # Step 2: Select best candidate
            draft_result = self.selector.select(candidates)
            draft_tokens = draft_result.tokens
            stats.draft_tokens_proposed += len(draft_tokens)

            if not draft_tokens:
                break

            # Step 3: Target verifies selected draft
            target_logprobs, target_entropies = self.target_model.verify(
                draft_tokens=draft_tokens,
            )

            # Step 4: Policy decides acceptance
            decision: AcceptDecision = self.policy.should_accept(
                target_logprobs, target_entropies
            )

            logger.debug(
                "Multi-draft iteration %d: %d/%d accepted",
                stats.iterations,
                decision.accepted_count,
                len(draft_tokens),
            )

            # Step 5a: Commit accepted prefix
            accepted_tokens = draft_tokens[: decision.accepted_count]
            if accepted_tokens:
                context.extend(accepted_tokens)
                generated.extend(accepted_tokens)
                stats.draft_tokens_accepted += len(accepted_tokens)

                if self._contains_eos(accepted_tokens):
                    finish_reason = "eos"
                    break

            # Step 5b: Handle rejection & resync
            rejected_count = len(draft_tokens) - decision.accepted_count
            stats.draft_tokens_rejected += rejected_count

            if decision.accepted_count < len(draft_tokens):
                # Target rollback
                target_restored = self.kv_manager.restore(
                    self.target_model, "target"
                )
                if target_restored:
                    if accepted_tokens:
                        self.target_model.prefill(accepted_tokens)
                else:
                    self.target_model.prefill(context)

                # Target correction
                correction_token, _ = self.target_model.generate_single()  # type: ignore[attr-defined]
                context.append(correction_token)
                generated.append(correction_token)
                stats.target_corrections += 1
                stats.total_tokens = len(generated)

                if self._is_eos(correction_token):
                    finish_reason = "eos"
                    break

                # Draft rollback & resync all drafts
                draft_sync_tokens = accepted_tokens + [correction_token]
                for i, draft in enumerate(self.draft_models):
                    draft_restored = self.kv_manager.restore(
                        draft, f"draft_{i}"
                    )
                    if draft_restored:
                        if draft_sync_tokens:
                            draft.prefill(draft_sync_tokens)
                    else:
                        draft.prefill(context)
            else:
                stats.total_tokens = len(generated)

            if (
                draft_result.finish_reason == "eos"
                and decision.accepted_count == len(draft_tokens)
            ):
                finish_reason = "eos"
                break

        # Finalize stats
        elapsed = time.perf_counter() - start_time
        stats.latency_ms = elapsed * 1000
        stats.total_tokens = len(generated)

        # Detokenize using the first draft model
        text = self.draft_models[0].detokenize(generated)

        return SpeculativeResult(
            tokens=generated,
            text=text,
            stats=stats,
            finish_reason=finish_reason,
        )

    def _contains_eos(self, tokens: list[int]) -> bool:
        """Check if any token is EOS."""
        if not hasattr(self.target_model, "model") or self.target_model.model is None:
            return False
        eos_id = self.target_model.model.token_eos()  # type: ignore[attr-defined]
        return bool(eos_id in tokens)

    def _is_eos(self, token: int) -> bool:
        """Check if token is EOS."""
        if not hasattr(self.target_model, "model") or self.target_model.model is None:
            return False
        return bool(token == self.target_model.model.token_eos())  # type: ignore[attr-defined]
