"""KV Cache manager for RALMO.

Manages KV cache snapshots and restoration for both draft and target models.
Supports native llama.cpp state save/load when available, with fallback
to prefix-based recomputation simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ralmo_core.models.base_model import BaseModel, KVSnapshot

logger = logging.getLogger(__name__)


@dataclass
class KVState:
    """Tracked KV state for a model.

    Attributes:
        snapshot: The last saved KV snapshot.
        committed_tokens: Tokens that have been committed to the KV cache.
        is_dirty: Whether the KV cache has been modified since last snapshot.
    """

    snapshot: KVSnapshot | None = None
    committed_tokens: list[int] | None = None
    is_dirty: bool = False


class KVManager:
    """Manages KV cache lifecycle for RALMO models.

    Provides a unified interface for snapshotting, restoring, and committing
    KV cache states. When native llama.cpp KV state save/load is not available,
    falls back to prefix-based recomputation.

    Attributes:
        use_native: Whether to attempt native KV state operations.
    """

    def __init__(self, use_native: bool = True) -> None:
        self.use_native = use_native
        self._states: dict[str, KVState] = {}

    def snapshot(self, model: BaseModel, model_id: str = "default") -> KVSnapshot:
        """Take a snapshot of a model's KV cache.

        Args:
            model: The model whose KV cache to snapshot.
            model_id: Unique identifier for the model (e.g., 'draft', 'target').

        Returns:
            KVSnapshot that can be restored later.
        """
        if self.use_native:
            try:
                snap = model.snapshot_kv()
                self._states[model_id] = KVState(
                    snapshot=snap,
                    committed_tokens=None,
                    is_dirty=False,
                )
                logger.debug("Native KV snapshot saved for model '%s'", model_id)
                return snap
            except Exception as e:
                logger.warning(
                    "Native KV snapshot failed for '%s': %s. Using prefix fallback.",
                    model_id,
                    e,
                )

        # Fallback: return a minimal snapshot
        snap = KVSnapshot(state=None, token_count=0, metadata={"model_id": model_id})
        self._states[model_id] = KVState(
            snapshot=snap,
            committed_tokens=None,
            is_dirty=False,
        )
        return snap

    def restore(self, model: BaseModel, model_id: str = "default") -> bool:
        """Restore a model's KV cache from the last snapshot.

        Args:
            model: The model to restore.
            model_id: Identifier of the saved state to restore.

        Returns:
            True if restoration succeeded, False if fallback was used.
        """
        state = self._states.get(model_id)
        if state is None or state.snapshot is None:
            logger.warning("No snapshot found for model '%s'", model_id)
            return False

        if self.use_native and state.snapshot.state is not None:
            try:
                model.restore_kv(state.snapshot)
                state.is_dirty = False
                logger.debug("Native KV restore succeeded for model '%s'", model_id)
                return True
            except Exception as e:
                logger.warning(
                    "Native KV restore failed for '%s': %s. Using prefix fallback.",
                    model_id,
                    e,
                )

        # Fallback: reset model (caller must re-evaluate prefix)
        model.reset()
        state.is_dirty = True
        return False

    def commit(
        self,
        model: BaseModel,
        tokens: list[int],
        model_id: str = "default",
    ) -> None:
        """Record that tokens have been committed to the model's context.

        Updates the tracked state so that future snapshot/restore operations
        know the current token count.

        Args:
            model: The model that consumed the tokens.
            tokens: Token IDs that were committed.
            model_id: Model identifier.
        """
        state = self._states.get(model_id)
        if state is None:
            state = KVState()
            self._states[model_id] = state

        if state.committed_tokens is None:
            state.committed_tokens = list(tokens)
        else:
            state.committed_tokens.extend(tokens)

        state.is_dirty = True
        logger.debug(
            "Committed %d tokens for model '%s' (total: %d)",
            len(tokens),
            model_id,
            len(state.committed_tokens),
        )

    def get_committed_tokens(self, model_id: str = "default") -> list[int]:
        """Get the list of all committed tokens for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of committed token IDs, or empty list if none.
        """
        state = self._states.get(model_id)
        if state is None or state.committed_tokens is None:
            return []
        return list(state.committed_tokens)

    def clear(self, model_id: str | None = None) -> None:
        """Clear tracked KV states.

        Args:
            model_id: If provided, clear only this model's state.
                      If None, clear all states.
        """
        if model_id is not None:
            self._states.pop(model_id, None)
            logger.debug("Cleared KV state for model '%s'", model_id)
        else:
            self._states.clear()
            logger.debug("Cleared all KV states.")

    def get_state_info(self, model_id: str = "default") -> dict[str, Any]:
        """Get diagnostic information about the current KV state.

        Args:
            model_id: Model identifier.

        Returns:
            Dictionary with state information.
        """
        state = self._states.get(model_id)
        if state is None:
            return {"model_id": model_id, "exists": False}
        return {
            "model_id": model_id,
            "exists": True,
            "has_snapshot": state.snapshot is not None,
            "snapshot_token_count": state.snapshot.token_count if state.snapshot else 0,
            "committed_token_count": len(state.committed_tokens) if state.committed_tokens else 0,
            "is_dirty": state.is_dirty,
        }
