"""Tests for RALMO speculative decoding engine.

Uses mock models that return predetermined logprobs to verify the
acceptance/rejection logic without requiring actual model files.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import GenerationResult, KVSnapshot
from ralmo_core.policy import StaticPolicy
from ralmo_core.speculative import SpeculativeEngine


class MockDraftModel:
    """Mock draft model that generates predetermined tokens."""

    def __init__(self, token_sequence: list[int], logprobs: list[float]) -> None:
        self._token_sequence = token_sequence
        self._logprobs = logprobs
        self._call_count = 0

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(
        self, k: int, temperature: float = 0.8
    ) -> GenerationResult:
        start = self._call_count * k
        end = min(start + k, len(self._token_sequence))
        tokens = self._token_sequence[start:end]
        lps = self._logprobs[start:end]
        self._call_count += 1
        finish = "eos" if end >= len(self._token_sequence) else "length"
        return GenerationResult(tokens=tokens, logprobs=lps, finish_reason=finish)

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        return [0.0] * len(draft_tokens), [0.0] * len(draft_tokens)

    def generate_single(self) -> tuple[int, float]:
        return (0, 0.0)

    def snapshot_kv(self) -> KVSnapshot:
        return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        pass

    def reset(self) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if 32 <= t < 127)


class MockTargetModel:
    """Mock target model that returns predetermined verification logprobs."""

    def __init__(self, verification_logprobs: list[list[float]]) -> None:
        """
        Args:
            verification_logprobs: List of logprob lists, one per verify() call.
        """
        self._verification_logprobs = verification_logprobs
        self._verify_call_count = 0
        self._correction_tokens: list[tuple[int, float]] = [(99, -0.1)]

        # Mock model attribute for EOS check
        self.model = MagicMock()
        self.model.token_eos.return_value = -1  # Use -1 as EOS

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(
        self, k: int, temperature: float = 0.0
    ) -> GenerationResult:
        return GenerationResult(tokens=[99], logprobs=[-0.1], finish_reason="length")

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        if self._verify_call_count < len(self._verification_logprobs):
            lps = self._verification_logprobs[self._verify_call_count]
            self._verify_call_count += 1
            return lps[: len(draft_tokens)], [0.0] * min(len(draft_tokens), len(lps))
        return [-1.0] * len(draft_tokens), [0.0] * len(draft_tokens)

    def generate_single(self) -> tuple[int, float]:
        return (99, -0.1)

    def snapshot_kv(self) -> KVSnapshot:
        return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        pass

    def reset(self) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if 32 <= t < 127)


class TestSpeculativeEngineAllAccept:
    """Test: all draft tokens are accepted."""

    def test_all_tokens_accepted(self) -> None:
        # Draft generates 4 tokens each round, all with high logprobs from target
        draft = MockDraftModel(
            token_sequence=[10, 20, 30, 40],
            logprobs=[-0.1, -0.2, -0.3, -0.1],
        )
        target = MockTargetModel(
            verification_logprobs=[[-0.1, -0.2, -0.3, -0.1]],
        )
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)

        assert result.stats.draft_tokens_accepted == 4
        assert result.stats.draft_tokens_rejected == 0
        assert result.stats.target_corrections == 0
        assert len(result.tokens) == 4


class TestSpeculativeEnginePartialAccept:
    """Test: some draft tokens accepted, then mismatch."""

    def test_partial_acceptance(self) -> None:
        # Draft generates 4 tokens, but 3rd has low logprob from target
        draft = MockDraftModel(
            token_sequence=[10, 20, 30, 40, 50, 60, 70, 80],
            logprobs=[-0.1, -0.2, -0.3, -0.1, -0.1, -0.1, -0.1, -0.1],
        )
        target = MockTargetModel(
            verification_logprobs=[
                [-0.1, -0.2, -0.9, -0.1],  # 3rd token rejected (below -0.7)
                [-0.1, -0.2, -0.3, -0.1],  # second round all accepted
            ],
        )
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=10)

        # First round: 2 accepted + 1 correction from target
        # Second round: 4 accepted
        assert result.stats.draft_tokens_accepted >= 2
        assert result.stats.target_corrections >= 1


class TestSpeculativeEngineFullReject:
    """Test: all draft tokens rejected."""

    def test_all_tokens_rejected(self) -> None:
        draft = MockDraftModel(
            token_sequence=[10, 20, 30, 40],
            logprobs=[-0.1, -0.2, -0.3, -0.1],
        )
        target = MockTargetModel(
            verification_logprobs=[[-2.0, -2.0, -2.0, -2.0]],
        )
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)

        # No draft tokens accepted, but target produced corrections
        assert result.stats.draft_tokens_accepted == 0
        assert result.stats.target_corrections >= 1


class TestSpeculativeEngineBoundaryTau:
    """Test: tokens exactly at threshold τ."""

    def test_boundary_tau_accepted(self) -> None:
        """Tokens with logprob exactly equal to τ should be accepted."""
        draft = MockDraftModel(
            token_sequence=[10, 20, 30, 40],
            logprobs=[-0.7, -0.7, -0.7, -0.7],
        )
        target = MockTargetModel(
            verification_logprobs=[[-0.7, -0.7, -0.7, -0.7]],
        )
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)

        # All tokens exactly at τ → all accepted
        assert result.stats.draft_tokens_accepted == 4
        assert result.stats.target_corrections == 0
