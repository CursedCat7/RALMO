"""Tests for RALMO KV Cache Manager."""

from __future__ import annotations

from unittest.mock import MagicMock

from ralmo_core.kv_manager import KVManager
from ralmo_core.models.base_model import KVSnapshot


class TestKVManagerSnapshot:
    """Test KV snapshot operations."""

    def test_native_snapshot(self) -> None:
        """Test native KV snapshot save and restore."""
        mock_model = MagicMock()
        mock_state = b"fake_kv_state"
        mock_model.snapshot_kv.return_value = KVSnapshot(
            state=mock_state,
            token_count=10,
            metadata={"model": "test"},
        )

        kv = KVManager(use_native=True)
        snap = kv.snapshot(mock_model, model_id="test")

        assert snap.state == mock_state
        assert snap.token_count == 10
        mock_model.snapshot_kv.assert_called_once()

    def test_native_restore(self) -> None:
        """Test restoring a native KV snapshot."""
        mock_model = MagicMock()
        mock_state = b"fake_kv_state"
        mock_model.snapshot_kv.return_value = KVSnapshot(
            state=mock_state,
            token_count=10,
        )

        kv = KVManager(use_native=True)
        kv.snapshot(mock_model, model_id="test")
        result = kv.restore(mock_model, model_id="test")

        assert result is True
        mock_model.restore_kv.assert_called_once()

    def test_fallback_on_native_failure(self) -> None:
        """Test fallback when native snapshot fails."""
        mock_model = MagicMock()
        mock_model.snapshot_kv.side_effect = RuntimeError("not supported")

        kv = KVManager(use_native=True)
        snap = kv.snapshot(mock_model, model_id="fallback")

        # Should fall back to empty snapshot
        assert snap.state is None
        assert snap.token_count == 0

    def test_restore_without_snapshot(self) -> None:
        """Test restore when no snapshot exists."""
        mock_model = MagicMock()
        kv = KVManager(use_native=True)
        result = kv.restore(mock_model, model_id="nonexistent")

        assert result is False


class TestKVManagerCommit:
    """Test KV commit tracking."""

    def test_commit_tokens(self) -> None:
        """Test committing tokens."""
        mock_model = MagicMock()
        kv = KVManager(use_native=False)

        kv.commit(mock_model, tokens=[1, 2, 3], model_id="test")
        committed = kv.get_committed_tokens(model_id="test")

        assert committed == [1, 2, 3]

    def test_cumulative_commit(self) -> None:
        """Test multiple commits accumulate."""
        mock_model = MagicMock()
        kv = KVManager(use_native=False)

        kv.commit(mock_model, tokens=[1, 2], model_id="test")
        kv.commit(mock_model, tokens=[3, 4], model_id="test")
        committed = kv.get_committed_tokens(model_id="test")

        assert committed == [1, 2, 3, 4]

    def test_empty_committed_tokens(self) -> None:
        """Test getting committed tokens when none exist."""
        kv = KVManager(use_native=False)
        committed = kv.get_committed_tokens(model_id="test")

        assert committed == []


class TestKVManagerClear:
    """Test clearing KV states."""

    def test_clear_specific_model(self) -> None:
        """Test clearing a specific model's state."""
        mock_model = MagicMock()
        kv = KVManager(use_native=False)
        kv.commit(mock_model, tokens=[1, 2, 3], model_id="test")
        kv.clear(model_id="test")

        assert kv.get_committed_tokens(model_id="test") == []

    def test_clear_all(self) -> None:
        """Test clearing all states."""
        mock_model = MagicMock()
        kv = KVManager(use_native=False)
        kv.commit(mock_model, tokens=[1], model_id="a")
        kv.commit(mock_model, tokens=[2], model_id="b")
        kv.clear()

        assert kv.get_committed_tokens(model_id="a") == []
        assert kv.get_committed_tokens(model_id="b") == []


class TestKVManagerStateInfo:
    """Test state info diagnostics."""

    def test_state_info_exists(self) -> None:
        """Test state info for existing state."""
        mock_model = MagicMock()
        kv = KVManager(use_native=False)
        kv.commit(mock_model, tokens=[1, 2, 3], model_id="test")

        info = kv.get_state_info(model_id="test")

        assert info["exists"] is True
        assert info["committed_token_count"] == 3
        assert info["is_dirty"] is True

    def test_state_info_not_exists(self) -> None:
        """Test state info for non-existing state."""
        kv = KVManager(use_native=False)
        info = kv.get_state_info(model_id="missing")

        assert info["exists"] is False
