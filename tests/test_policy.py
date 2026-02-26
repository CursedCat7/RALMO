"""Tests for RALMO policy module."""

from __future__ import annotations

from ralmo_core.policy import AdaptivePolicy, StaticPolicy


class TestStaticPolicy:
    """Test the static acceptance policy."""

    def test_all_above_threshold(self) -> None:
        """All logprobs above τ → all accepted."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-0.1, -0.2, -0.3, -0.5])

        assert decision.accepted_count == 4
        assert decision.should_escalate is False

    def test_none_above_threshold(self) -> None:
        """All logprobs below τ → none accepted."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-1.0, -1.5, -2.0, -3.0])

        assert decision.accepted_count == 0

    def test_partial_acceptance(self) -> None:
        """First two above, third below → accept first two."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-0.1, -0.3, -0.9, -0.1])

        assert decision.accepted_count == 2

    def test_boundary_equal_to_tau(self) -> None:
        """Logprob exactly equal to τ → accepted (≥ threshold)."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-0.7, -0.7, -0.7, -0.7])

        assert decision.accepted_count == 4

    def test_boundary_just_below_tau(self) -> None:
        """Logprob just below τ → rejected."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-0.70001])

        assert decision.accepted_count == 0

    def test_empty_logprobs(self) -> None:
        """Empty logprobs → zero accepted."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([])

        assert decision.accepted_count == 0

    def test_single_token_accept(self) -> None:
        """Single token above threshold."""
        policy = StaticPolicy(tau=-0.7, k=1)
        decision = policy.should_accept([-0.3])

        assert decision.accepted_count == 1

    def test_single_token_reject(self) -> None:
        """Single token below threshold."""
        policy = StaticPolicy(tau=-0.7, k=1)
        decision = policy.should_accept([-1.5])

        assert decision.accepted_count == 0

    def test_get_k(self) -> None:
        """Test k accessor."""
        policy = StaticPolicy(tau=-0.5, k=8)
        assert policy.get_k() == 8

    def test_get_tau(self) -> None:
        """Test τ accessor."""
        policy = StaticPolicy(tau=-0.5, k=8)
        assert policy.get_tau() == -0.5

    def test_decision_details(self) -> None:
        """Test that decision includes diagnostic details."""
        policy = StaticPolicy(tau=-0.7, k=4)
        decision = policy.should_accept([-0.1, -0.5, -0.9])

        assert decision.details is not None
        assert decision.details["tau"] == -0.7
        assert decision.details["total_proposed"] == 3.0
        assert decision.details["min_logprob"] == -0.9
        assert decision.details["max_logprob"] == -0.1

    def test_consecutive_rejection(self) -> None:
        """Only consecutive tokens from start are accepted."""
        policy = StaticPolicy(tau=-0.7, k=4)
        # First token bad → stop immediately, even if later tokens are good
        decision = policy.should_accept([-1.0, -0.1, -0.1, -0.1])

        assert decision.accepted_count == 0

class TestAdaptivePolicy:
    """Test the adaptive acceptance policy."""

    def test_high_entropy_lowers_threshold(self) -> None:
        """High entropy -> higher tau(H) -> stricter acceptance."""
        # tau_0 = -0.7, alpha = 0.5, h_0 = 1.0
        # H = 2.0 -> tau(H) = -0.7 + 0.5 * (2.0 - 1.0) = -0.2
        policy = AdaptivePolicy(tau_0=-0.7, alpha=0.5, h_0=1.0, k=4)
        decision = policy.should_accept([-0.3], [2.0])
        # -0.3 is NOT >= -0.2
        assert decision.accepted_count == 0

    def test_low_entropy_raises_threshold(self) -> None:
        """Low entropy -> lower tau(H) -> lenient acceptance."""
        # tau_0 = -0.7, alpha = 0.5, h_0 = 1.0
        # H = 0.0 -> tau(H) = -0.7 + 0.5 * (0.0 - 1.0) = -1.2
        policy = AdaptivePolicy(tau_0=-0.7, alpha=0.5, h_0=1.0, k=4)
        decision = policy.should_accept([-0.9], [0.0])
        # -0.9 IS >= -1.2
        assert decision.accepted_count == 1

    def test_dynamic_threshold_per_token(self) -> None:
        """Threshold adapts per token based on entropy."""
        policy = AdaptivePolicy(tau_0=-0.7, alpha=0.5, h_0=1.0, k=4)

        # logprobs: [-0.6, -0.6]
        # entropies: [0.0, 2.0]
        # Token 1: tau(-0.6) with H=0.0 -> threshold is -1.2 -> accepted
        # Token 2: tau(-0.6) with H=2.0 -> threshold is -0.2 -> rejected

        decision = policy.should_accept([-0.6, -0.6], [0.0, 2.0])
        assert decision.accepted_count == 1

    def test_missing_entropies(self) -> None:
        """If entropies are missing, treats them as 0.0."""
        # H=0.0 -> tau(H) = -0.7 + 0.5 * (0.0 - 1.0) = -1.2
        policy = AdaptivePolicy(tau_0=-0.7, alpha=0.5, h_0=1.0, k=4)
        decision = policy.should_accept([-0.9, -0.9])
        # Since H=0, threshold is -1.2, so -0.9 is accepted
        assert decision.accepted_count == 2
