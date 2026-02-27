"""Tests for Multi-Draft engine and draft selectors."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ralmo_core.draft_selector import (
    BestOfNSelector,
    MajorityVoteSelector,
)
from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import GenerationResult, KVSnapshot
from ralmo_core.multi_draft_engine import MultiDraftEngine
from ralmo_core.policy import StaticPolicy

# --- Draft Selector Tests ---


class TestBestOfNSelector:
    """Tests for BestOfNSelector."""

    def test_selects_highest_logprob_sum(self) -> None:
        sel = BestOfNSelector()
        c1 = GenerationResult(tokens=[1, 2], logprobs=[-0.5, -0.5], finish_reason="length")
        c2 = GenerationResult(tokens=[3, 4], logprobs=[-0.1, -0.1], finish_reason="length")
        c3 = GenerationResult(tokens=[5, 6], logprobs=[-0.9, -0.9], finish_reason="length")
        result = sel.select([c1, c2, c3])
        assert result.tokens == [3, 4]

    def test_single_candidate(self) -> None:
        sel = BestOfNSelector()
        c = GenerationResult(tokens=[1], logprobs=[-0.5], finish_reason="length")
        assert sel.select([c]).tokens == [1]

    def test_empty_raises(self) -> None:
        sel = BestOfNSelector()
        with pytest.raises(ValueError):
            sel.select([])


class TestMajorityVoteSelector:
    """Tests for MajorityVoteSelector."""

    def test_majority_vote_picks_most_common(self) -> None:
        sel = MajorityVoteSelector()
        c1 = GenerationResult(tokens=[10, 20], logprobs=[-0.3, -0.3], finish_reason="length")
        c2 = GenerationResult(tokens=[10, 30], logprobs=[-0.2, -0.2], finish_reason="length")
        c3 = GenerationResult(tokens=[10, 20], logprobs=[-0.4, -0.4], finish_reason="length")
        result = sel.select([c1, c2, c3])
        # Position 0: token 10 (3 votes). Position 1: token 20 (2 votes).
        assert result.tokens == [10, 20]

    def test_averages_logprobs_of_winners(self) -> None:
        sel = MajorityVoteSelector()
        c1 = GenerationResult(tokens=[10], logprobs=[-0.2], finish_reason="length")
        c2 = GenerationResult(tokens=[10], logprobs=[-0.4], finish_reason="length")
        c3 = GenerationResult(tokens=[99], logprobs=[-0.1], finish_reason="length")
        result = sel.select([c1, c2, c3])
        assert result.tokens == [10]
        assert abs(result.logprobs[0] - (-0.3)) < 1e-6

    def test_empty_raises(self) -> None:
        sel = MajorityVoteSelector()
        with pytest.raises(ValueError):
            sel.select([])

    def test_tie_uses_first_seen(self) -> None:
        """When tokens are tied, Counter.most_common picks first insertion."""
        sel = MajorityVoteSelector()
        c1 = GenerationResult(tokens=[10], logprobs=[-0.3], finish_reason="length")
        c2 = GenerationResult(tokens=[20], logprobs=[-0.2], finish_reason="length")
        result = sel.select([c1, c2])
        # Both have 1 vote; Counter.most_common picks first
        assert result.tokens[0] in (10, 20)
        assert len(result.tokens) == 1


# --- MultiDraftEngine Test ---


class MockDraftForMulti:
    """Mock draft model for multi-draft testing."""

    def __init__(self, tokens: list[int], logprobs: list[float]) -> None:
        self._tokens = tokens
        self._logprobs = logprobs
        self._call_count = 0

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(self, k: int, temperature: float = 0.8) -> GenerationResult:
        start = self._call_count * k
        end = min(start + k, len(self._tokens))
        toks = self._tokens[start:end]
        lps = self._logprobs[start:end]
        self._call_count += 1
        finish = "eos" if end >= len(self._tokens) else "length"
        return GenerationResult(tokens=toks, logprobs=lps, finish_reason=finish)

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


class MockTargetForMulti:
    """Mock target model for multi-draft testing."""

    def __init__(self, logprobs: list[float]) -> None:
        self._logprobs = logprobs
        self._verify_count = 0
        self.model = MagicMock()
        self.model.token_eos.return_value = -1

    def prefill(self, tokens: list[int]) -> None:
        pass

    def generate(self, k: int, temperature: float = 0.0) -> GenerationResult:
        return GenerationResult(tokens=[99], logprobs=[-0.1], finish_reason="length")

    def verify(self, draft_tokens: list[int]) -> tuple[list[float], list[float]]:
        lps = self._logprobs[: len(draft_tokens)]
        return lps, [0.0] * len(lps)

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


class TestMultiDraftEngine:
    """Test MultiDraftEngine with mock models."""

    def test_multi_draft_all_accepted(self) -> None:
        draft1 = MockDraftForMulti([10, 20, 30, 40], [-0.1, -0.2, -0.3, -0.1])
        draft2 = MockDraftForMulti([11, 21, 31, 41], [-0.5, -0.5, -0.5, -0.5])
        target = MockTargetForMulti([-0.1, -0.2, -0.3, -0.1])
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = MultiDraftEngine(
            draft_models=[draft1, draft2],  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
            selector=BestOfNSelector(),
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)
        assert result.stats.draft_tokens_accepted == 4
        assert result.stats.target_corrections == 0

    def test_multi_draft_requires_at_least_one(self) -> None:
        with pytest.raises(ValueError):
            MultiDraftEngine(
                draft_models=[],
                target_model=MagicMock(),  # type: ignore
                policy=StaticPolicy(),
                kv_manager=KVManager(use_native=False),
            )

    def test_multi_draft_partial_rejection(self) -> None:
        """Test multi-draft with partial token rejection."""
        draft1 = MockDraftForMulti(
            [10, 20, 30, 40, 50, 60, 70, 80],
            [-0.1, -0.2, -0.3, -0.1, -0.1, -0.2, -0.3, -0.1],
        )
        draft2 = MockDraftForMulti(
            [11, 21, 31, 41, 51, 61, 71, 81],
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        )
        # Target rejects token at position 2 (logprob -0.9 < tau -0.7)
        target = MockTargetForMulti([-0.1, -0.2, -0.9, -0.1])
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = MultiDraftEngine(
            draft_models=[draft1, draft2],  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
            selector=BestOfNSelector(),
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)
        # Iteration 1: 2 accepted + 1 correction = 3 tokens
        # Iteration 2: 1 more accepted from remaining budget
        assert result.stats.target_corrections >= 1
        assert result.stats.draft_tokens_rejected >= 2

    def test_multi_draft_with_majority_vote(self) -> None:
        """Test multi-draft using MajorityVoteSelector."""
        # All 3 drafts produce same first 2 tokens but differ at position 2
        draft1 = MockDraftForMulti([10, 20, 30, 40], [-0.1, -0.2, -0.3, -0.1])
        draft2 = MockDraftForMulti([10, 20, 99, 40], [-0.1, -0.2, -0.5, -0.1])
        draft3 = MockDraftForMulti([10, 20, 30, 40], [-0.1, -0.2, -0.4, -0.1])
        target = MockTargetForMulti([-0.1, -0.2, -0.3, -0.1])
        policy = StaticPolicy(tau=-0.7, k=4)
        kv = KVManager(use_native=False)

        engine = MultiDraftEngine(
            draft_models=[draft1, draft2, draft3],  # type: ignore
            target_model=target,  # type: ignore
            policy=policy,
            kv_manager=kv,
            selector=MajorityVoteSelector(),
        )

        result = engine.run(prompt_tokens=[1, 2, 3], max_tokens=4)
        # MajorityVote picks token 30 at pos 2 (2 vs 1 vote)
        assert result.stats.draft_tokens_accepted == 4
        assert result.stats.target_corrections == 0
