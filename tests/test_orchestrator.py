"""Integration test for RALMO orchestrator with mock models."""

from __future__ import annotations

from unittest.mock import MagicMock

from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import GenerationResult, KVSnapshot
from ralmo_core.policy import StaticPolicy
from ralmo_core.speculative import SpeculativeEngine


class MockDraftForIntegration:
    """Mock draft model for integration testing."""

    def __init__(self) -> None:
        self._tokens = list(range(100, 120))  # 20 tokens
        self._call_count = 0

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(
        self, k: int, temperature: float = 0.8
    ) -> GenerationResult:
        start = self._call_count * k
        end = min(start + k, len(self._tokens))
        tokens = self._tokens[start:end]
        self._call_count += 1
        if not tokens:
            return GenerationResult(tokens=[], logprobs=[], finish_reason="eos")
        lps = [-0.2] * len(tokens)
        finish = "eos" if end >= len(self._tokens) else "length"
        return GenerationResult(tokens=tokens, logprobs=lps, finish_reason=finish)

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        return [-0.2] * len(draft_tokens), [0.0] * len(draft_tokens)

    def generate_single(self) -> tuple[int, float]:
        return (0, 0.0)

    def snapshot_kv(self) -> KVSnapshot:
        return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        pass

    def reset(self) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text[:50]]

    def detokenize(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class MockTargetForIntegration:
    """Mock target model for integration testing."""

    def __init__(self, accept_all: bool = True) -> None:
        self._accept_all = accept_all
        self.model = MagicMock()
        self.model.token_eos.return_value = -1

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(
        self, k: int, temperature: float = 0.0
    ) -> GenerationResult:
        return GenerationResult(tokens=[200], logprobs=[-0.1], finish_reason="length")

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        if self._accept_all:
            return [-0.2] * len(draft_tokens), [0.0] * len(draft_tokens)
        # Reject every other token
        logprobs = [-0.2 if i % 2 == 0 else -1.5 for i in range(len(draft_tokens))]
        return logprobs, [0.0] * len(draft_tokens)

    def generate_single(self) -> tuple[int, float]:
        return (200, -0.1)

    def snapshot_kv(self) -> KVSnapshot:
        return KVSnapshot(state=None, token_count=0)

    def restore_kv(self, snapshot: KVSnapshot) -> None:
        pass

    def reset(self) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text[:50]]

    def detokenize(self, tokens: list[int]) -> str:
        return " ".join(str(t) for t in tokens)


class TestIntegrationFullPipeline:
    """End-to-end integration test with mock models."""

    def test_full_accept_pipeline(self) -> None:
        """Full pipeline with all tokens accepted."""
        draft = MockDraftForIntegration()
        target = MockTargetForIntegration(accept_all=True)
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=16)

        assert result.stats.total_tokens > 0
        assert result.stats.acceptance_rate > 0.0
        assert result.stats.latency_ms > 0.0
        assert len(result.tokens) == result.stats.total_tokens

    def test_mixed_accept_reject_pipeline(self) -> None:
        """Pipeline with some rejections."""
        draft = MockDraftForIntegration()
        target = MockTargetForIntegration(accept_all=False)
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=8)

        assert result.stats.total_tokens > 0
        assert result.stats.target_corrections > 0
        # Should have some accepted and some rejected
        assert result.stats.draft_tokens_proposed > 0

    def test_max_tokens_respected(self) -> None:
        """Ensure max_tokens limit is respected."""
        draft = MockDraftForIntegration()
        target = MockTargetForIntegration(accept_all=True)
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        max_tokens = 8
        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=max_tokens)

        assert result.stats.total_tokens <= max_tokens

    def test_stats_consistency(self) -> None:
        """Verify stats are internally consistent."""
        draft = MockDraftForIntegration()
        target = MockTargetForIntegration(accept_all=True)
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = SpeculativeEngine(
            draft_model=draft,  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=12)
        stats = result.stats

        # Proposed = accepted + rejected
        assert stats.draft_tokens_proposed == (
            stats.draft_tokens_accepted + stats.draft_tokens_rejected
        )
        # Total tokens = accepted + corrections
        assert stats.total_tokens == stats.draft_tokens_accepted + stats.target_corrections
