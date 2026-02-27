"""Draft selection strategies for Multi-Draft speculative decoding.

Provides abstract and concrete selectors that choose the best draft
candidate from multiple draft model outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from ralmo_core.models.base_model import GenerationResult


class DraftSelector(ABC):
    """Abstract base class for draft candidate selection.

    Given multiple GenerationResults from different draft models,
    selects the best candidate to forward to the target model.
    """

    @abstractmethod
    def select(self, candidates: list[GenerationResult]) -> GenerationResult:
        """Select the best draft candidate.

        Args:
            candidates: GenerationResults from each draft model.

        Returns:
            The selected GenerationResult.

        Raises:
            ValueError: If candidates list is empty.
        """
        ...


class BestOfNSelector(DraftSelector):
    """Select the candidate with the highest sum of log-probabilities.

    This heuristic favors the draft sequence that the draft model
    itself is most confident about.
    """

    def select(self, candidates: list[GenerationResult]) -> GenerationResult:
        """Select candidate with highest total logprob.

        Args:
            candidates: GenerationResults from each draft model.

        Returns:
            The candidate with the highest sum of logprobs.

        Raises:
            ValueError: If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidates list.")

        best = candidates[0]
        best_score = sum(best.logprobs) if best.logprobs else float("-inf")

        for candidate in candidates[1:]:
            score = sum(candidate.logprobs) if candidate.logprobs else float("-inf")
            if score > best_score:
                best = candidate
                best_score = score

        return best


class MajorityVoteSelector(DraftSelector):
    """Select tokens by majority vote across candidates.

    For each position, picks the token that appears most frequently
    across all draft model outputs. Log-probabilities are averaged
    from the candidates that contributed the winning token.
    """

    def select(self, candidates: list[GenerationResult]) -> GenerationResult:
        """Select tokens by per-position majority vote.

        Args:
            candidates: GenerationResults from each draft model.

        Returns:
            A new GenerationResult with majority-voted tokens.

        Raises:
            ValueError: If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidates list.")

        if len(candidates) == 1:
            return candidates[0]

        # Find the minimum token length across candidates
        min_len = min(len(c.tokens) for c in candidates)
        if min_len == 0:
            return candidates[0]

        voted_tokens: list[int] = []
        voted_logprobs: list[float] = []

        for pos in range(min_len):
            # Collect tokens at this position from all candidates
            tokens_at_pos = [c.tokens[pos] for c in candidates]
            counter = Counter(tokens_at_pos)
            winner_token = counter.most_common(1)[0][0]

            # Average logprobs from candidates that produced the winner
            contributing_logprobs = [
                c.logprobs[pos]
                for c in candidates
                if c.tokens[pos] == winner_token
            ]
            avg_logprob = (
                sum(contributing_logprobs) / len(contributing_logprobs)
                if contributing_logprobs
                else 0.0
            )

            voted_tokens.append(winner_token)
            voted_logprobs.append(avg_logprob)

        # Use EOS finish_reason if any candidate hit EOS
        finish_reasons = [c.finish_reason for c in candidates]
        finish = "eos" if "eos" in finish_reasons else "length"

        return GenerationResult(
            tokens=voted_tokens,
            logprobs=voted_logprobs,
            finish_reason=finish,
        )
