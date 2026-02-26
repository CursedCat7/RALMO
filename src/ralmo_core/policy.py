"""Policy module for RALMO speculative decoding.

Defines acceptance policies that determine whether draft tokens should be
accepted based on their log-probabilities from the target model.
MVP uses a static threshold; future phases will add adaptive policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AcceptDecision:
    """Result of a policy acceptance evaluation.

    Attributes:
        accepted_count: Number of consecutive tokens accepted from the start.
        should_escalate: Whether to escalate to external verifier (future use).
        details: Optional diagnostic info about the decision.
    """

    accepted_count: int
    should_escalate: bool = False
    details: dict[str, float] | None = None


class BasePolicy(ABC):
    """Abstract base class for acceptance policies.

    Subclasses implement the acceptance logic that determines how many
    draft tokens to accept based on the target model's logprob evaluation.
    """

    @abstractmethod
    def should_accept(self, target_logprobs: list[float], entropies: list[float]) -> AcceptDecision:
        """Evaluate draft tokens for acceptance.

        Args:
            target_logprobs: Log-probabilities of draft tokens as scored
                             by the target model.
            entropies: Entropy values of the target model's distribution
                       for each draft token.

        Returns:
            AcceptDecision with the number of accepted tokens.
        """
        ...

    @abstractmethod
    def get_k(self) -> int:
        """Return the number of draft tokens to propose per iteration.

        Returns:
            Number of tokens the draft model should generate.
        """
        ...

    @abstractmethod
    def get_tau(self) -> float:
        """Return the current acceptance threshold.

        Returns:
            Threshold value (logprob scale).
        """
        ...


class StaticPolicy(BasePolicy):
    """Static acceptance policy with fixed threshold and draft count.

    Accepts consecutive draft tokens whose target logprobs are at or above
    the threshold τ. Stops at the first token that falls below τ.

    This is the MVP policy. Future adaptive policies will dynamically
    adjust τ and k based on entropy, history, and other signals.

    Attributes:
        tau: Acceptance threshold on logprob scale.
        k: Number of draft tokens to propose per iteration.
    """

    def __init__(self, tau: float = -0.7, k: int = 4) -> None:
        """Initialize static policy.

        Args:
            tau: Logprob threshold for acceptance (default: -0.7).
                 More negative = more lenient acceptance.
            k: Number of draft tokens per iteration (default: 4).
        """
        self._tau = tau
        self._k = k

    def should_accept(
        self, target_logprobs: list[float], entropies: list[float] | None = None
    ) -> AcceptDecision:
        """Accept consecutive tokens above threshold τ.

        Scans logprobs left-to-right and accepts tokens until one falls
        below τ. All subsequent tokens are rejected. Ignors entropy.

        Args:
            target_logprobs: Log-probabilities from the target model.
            entropies: Entropy values (ignored by StaticPolicy).

        Returns:
            AcceptDecision with the count of accepted tokens.
        """
        accepted_count = 0
        for logprob in target_logprobs:
            if logprob >= self._tau:
                accepted_count += 1
            else:
                break

        return AcceptDecision(
            accepted_count=accepted_count,
            should_escalate=False,
            details={
                "tau": self._tau,
                "k": float(self._k),
                "total_proposed": float(len(target_logprobs)),
                "min_logprob": min(target_logprobs) if target_logprobs else 0.0,
                "max_logprob": max(target_logprobs) if target_logprobs else 0.0,
            },
        )

    def get_k(self) -> int:
        """Return the fixed draft count k."""
        return self._k

    def get_tau(self) -> float:
        """Return the fixed threshold τ."""
        return self._tau


class AdaptivePolicy(BasePolicy):
    """Adaptive acceptance policy using entropy.

    Dynamically adjusts the acceptance threshold τ based on the target model's
    entropy for each generated token:
        τ(H) = τ_0 + α(H - H_0)

    Attributes:
        tau_0: Base acceptance threshold.
        alpha: Entropy sensitivity parameter.
        h_0: Baseline entropy.
        k: Number of draft tokens to propose per iteration.
    """

    def __init__(
        self,
        tau_0: float = -0.7,
        alpha: float = 0.1,
        h_0: float = 1.0,
        k: int = 4,
    ) -> None:
        """Initialize adaptive policy."""
        self._tau_0 = tau_0
        self._alpha = alpha
        self._h_0 = h_0
        self._k = k

    def should_accept(
        self, target_logprobs: list[float], entropies: list[float] | None = None
    ) -> AcceptDecision:
        """Accept tokens based on dynamic threshold τ(H)."""
        if entropies is None:
            entropies = [0.0] * len(target_logprobs)

        accepted_count = 0
        for logprob, entropy in zip(target_logprobs, entropies, strict=False):
            dynamic_tau = self._tau_0 + self._alpha * (entropy - self._h_0)
            if logprob >= dynamic_tau:
                accepted_count += 1
            else:
                break

        final_tau = self._tau_0 + self._alpha * (entropies[max(0, accepted_count - 1)] - self._h_0)

        return AcceptDecision(
            accepted_count=accepted_count,
            should_escalate=False,
            details={
                "tau_0": self._tau_0,
                "alpha": self._alpha,
                "h_0": self._h_0,
                "k": float(self._k),
                "total_proposed": float(len(target_logprobs)),
                "min_logprob": min(target_logprobs) if target_logprobs else 0.0,
                "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
                "final_tau": final_tau,
            },
        )

    def get_k(self) -> int:
        """Return the fixed draft count k."""
        return self._k

    def get_tau(self) -> float:
        """Return the base threshold τ_0."""
        return self._tau_0
