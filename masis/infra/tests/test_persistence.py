"""
tests/test_persistence.py
=========================
Unit tests for masis.infra.persistence.

Covers
------
- get_checkpointer() returns InMemorySaver when POSTGRES_URL is absent.
- get_checkpointer() logs the WARNING when falling back to InMemorySaver.
- get_async_checkpointer() async path (InMemorySaver fallback).
- CheckpointerContext async context manager.
- _redact_url() correctly masks passwords.
- get_state_history_for_thread() handles empty / error conditions gracefully.
- get_state_history_for_thread_async() falls back to sync when graph lacks
  aget_state_history.
"""

from __future__ import annotations

import asyncio
import os
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from masis.infra.persistence import (
    CheckpointerContext,
    _redact_url,
    _warn_in_memory,
    get_async_checkpointer,
    get_checkpointer,
    get_state_history_for_thread,
    get_state_history_for_thread_async,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(checkpoint_id, step, values=None, next_nodes=None):
    snap = MagicMock()
    snap.config = {"configurable": {"checkpoint_id": checkpoint_id}}
    snap.metadata = {"step": step}
    snap.values = values or {"iteration_count": step}
    snap.next = next_nodes or []
    snap.created_at = "2026-03-01T00:00:00Z"
    return snap


# ---------------------------------------------------------------------------
# _redact_url
# ---------------------------------------------------------------------------

class TestRedactUrl:

    def test_redacts_password(self):
        url = "postgresql://masis:super_secret@localhost:5432/db"
        redacted = _redact_url(url)
        assert "super_secret" not in redacted
        assert "masis:***@" in redacted

    def test_no_credentials_unchanged(self):
        url = "postgresql://localhost:5432/db"
        assert _redact_url(url) == url

    def test_empty_string(self):
        assert _redact_url("") == ""


# ---------------------------------------------------------------------------
# get_checkpointer — InMemorySaver fallback (no POSTGRES_URL)
# ---------------------------------------------------------------------------

class TestGetCheckpointer:

    def test_returns_in_memory_saver_when_no_url(self, caplog):
        """ENG-11 M1 S2a: If POSTGRES_URL not set → InMemorySaver used."""
        with patch.dict(os.environ, {}, clear=False):
            env_backup = os.environ.pop("POSTGRES_URL", None)
            try:
                with caplog.at_level(logging.WARNING, logger="masis.infra.persistence"):
                    checkpointer = get_checkpointer(postgres_url=None)
            finally:
                if env_backup is not None:
                    os.environ["POSTGRES_URL"] = env_backup

        # Should be some form of InMemorySaver or equivalent stub
        assert checkpointer is not None

    def test_returns_in_memory_saver_with_explicit_none(self):
        """Passing postgres_url=None explicitly also falls back."""
        checkpointer = get_checkpointer(postgres_url=None)
        assert checkpointer is not None

    def test_warning_message_printed_on_in_memory_fallback(self, capsys):
        """ENG-11 M1 S2b: WARNING message printed when InMemorySaver used."""
        with patch.dict(os.environ, {}, clear=False):
            env_backup = os.environ.pop("POSTGRES_URL", None)
            try:
                get_checkpointer(postgres_url=None)
            finally:
                if env_backup is not None:
                    os.environ["POSTGRES_URL"] = env_backup

        captured = capsys.readouterr()
        # The warning should appear in stdout (print) or stderr
        assert "InMemorySaver" in captured.out or "InMemorySaver" in captured.err

    def test_postgres_url_attempted_when_provided(self):
        """When POSTGRES_URL is provided but Postgres is not installed, fall back."""
        with patch("masis.infra.persistence._POSTGRES_SYNC_AVAILABLE", False):
            checkpointer = get_checkpointer(postgres_url="postgresql://user:pass@localhost/db")
        assert checkpointer is not None  # Falls back to InMemorySaver


# ---------------------------------------------------------------------------
# get_async_checkpointer — async path
# ---------------------------------------------------------------------------

class TestGetAsyncCheckpointer:

    @pytest.mark.asyncio
    async def test_returns_in_memory_saver_when_no_url(self):
        with patch.dict(os.environ, {}, clear=False):
            env_backup = os.environ.pop("POSTGRES_URL", None)
            try:
                checkpointer = await get_async_checkpointer(postgres_url=None)
            finally:
                if env_backup is not None:
                    os.environ["POSTGRES_URL"] = env_backup
        assert checkpointer is not None

    @pytest.mark.asyncio
    async def test_postgres_async_not_available_falls_back(self):
        with patch("masis.infra.persistence._POSTGRES_ASYNC_AVAILABLE", False):
            checkpointer = await get_async_checkpointer(
                postgres_url="postgresql://user:pass@localhost/db"
            )
        assert checkpointer is not None


# ---------------------------------------------------------------------------
# CheckpointerContext
# ---------------------------------------------------------------------------

class TestCheckpointerContext:

    @pytest.mark.asyncio
    async def test_context_manager_returns_checkpointer(self):
        async with CheckpointerContext(postgres_url=None) as checkpointer:
            assert checkpointer is not None

    @pytest.mark.asyncio
    async def test_aexit_called_on_saver_if_supported(self):
        """If the checkpointer has __aexit__, it should be called."""
        mock_saver = MagicMock()
        mock_saver.__aexit__ = AsyncMock(return_value=False)

        ctx = CheckpointerContext(postgres_url=None)
        ctx._checkpointer = mock_saver

        await ctx.__aexit__(None, None, None)
        mock_saver.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_does_not_suppress_exceptions(self):
        """Exceptions inside the context should propagate."""
        with pytest.raises(RuntimeError, match="test error"):
            async with CheckpointerContext(postgres_url=None):
                raise RuntimeError("test error")


# ---------------------------------------------------------------------------
# get_state_history_for_thread (sync)
# ---------------------------------------------------------------------------

class TestGetStateHistoryForThread:

    def _make_graph(self, snapshots, raises=False):
        graph = MagicMock()
        if raises:
            graph.get_state_history.side_effect = RuntimeError("DB error")
        else:
            graph.get_state_history.return_value = iter(snapshots)
        return graph

    def test_returns_empty_list_on_error(self):
        graph = self._make_graph([], raises=True)
        result = get_state_history_for_thread(graph, "thread-1")
        assert result == []

    def test_returns_snapshots_for_thread(self):
        snaps = [_make_snapshot(f"ckpt-{i}", i) for i in range(3)]
        graph = self._make_graph(snaps)
        result = get_state_history_for_thread(graph, "thread-1")
        assert len(result) == 3
        assert result[0]["step"] == 0
        assert result[2]["checkpoint_id"] == "ckpt-2"

    def test_respects_limit(self):
        snaps = [_make_snapshot(f"ckpt-{i}", i) for i in range(10)]
        graph = self._make_graph(snaps)
        result = get_state_history_for_thread(graph, "thread-1", limit=3)
        assert len(result) == 3

    def test_snapshot_has_required_keys(self):
        snaps = [_make_snapshot("ckpt-0", 0, values={"iteration_count": 0}, next_nodes=["executor"])]
        graph = self._make_graph(snaps)
        result = get_state_history_for_thread(graph, "thread-1")
        entry = result[0]
        assert "checkpoint_id" in entry
        assert "step" in entry
        assert "values" in entry
        assert "next" in entry
        assert "created_at" in entry

    def test_uses_thread_id_in_config(self):
        graph = self._make_graph([])
        get_state_history_for_thread(graph, "my-thread-abc")
        graph.get_state_history.assert_called_once_with(
            {"configurable": {"thread_id": "my-thread-abc"}}
        )


# ---------------------------------------------------------------------------
# get_state_history_for_thread_async
# ---------------------------------------------------------------------------

class TestGetStateHistoryForThreadAsync:

    @pytest.mark.asyncio
    async def test_uses_aget_state_history_when_available(self):
        """ENG-11 M2 S2a: state history available via async graph method."""
        snaps = [_make_snapshot(f"ckpt-{i}", i) for i in range(2)]

        async def _async_iter():
            for s in snaps:
                yield s

        graph = MagicMock()
        graph.aget_state_history.return_value = _async_iter()

        result = await get_state_history_for_thread_async(graph, "t1")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_when_aget_unavailable(self):
        snaps = [_make_snapshot("ckpt-0", 0)]
        graph = MagicMock(spec=[])  # no aget_state_history
        graph.get_state_history = MagicMock(return_value=iter(snaps))

        result = await get_state_history_for_thread_async(graph, "t2")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        async def _bad_iter():
            raise RuntimeError("async DB error")
            yield  # unreachable — makes this an async generator

        graph = MagicMock()
        graph.aget_state_history.return_value = _bad_iter()

        result = await get_state_history_for_thread_async(graph, "t3")
        assert result == []
