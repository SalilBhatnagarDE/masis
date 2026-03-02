"""
masis.infra.persistence
=======================
State Management & Persistence layer for MASIS (ENG-11).

Responsibilities
----------------
* Provide a single `get_checkpointer()` factory that returns a LangGraph
  `BaseCheckpointSaver` implementation suitable for the current environment.
* Primary backend: `AsyncPostgresSaver` (or `PostgresSaver` for sync code)
  connected via the `POSTGRES_URL` environment variable.
* Dev-mode fallback: `InMemorySaver` when `POSTGRES_URL` is absent or
  unreachable, with a mandatory WARNING log so developers cannot silently
  deploy without durable storage.
* Expose `CheckpointerContext` – an async context-manager wrapper that handles
  connection setup/teardown and calls `checkpointer.setup()` to ensure Postgres
  tables exist before the first graph invocation.
* Expose `get_state_history_for_thread()` – thin helper used by the
  `GET /masis/trace/{thread_id}` endpoint (MF-API-04).

MF-IDs
------
MF-MEM-08  Checkpoint persistence (PostgresSaver / InMemorySaver)
MF-HITL-05 Resume handler requires durable checkpoint so state survives the
           pause/resume lifecycle.

Architecture reference
----------------------
final_architecture_and_flow.md Section 10: Persistence & PostgresSaver Setup
engineering_tasks.md ENG-11 M1 S1/S2, M2 S1/S2
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

# ---------------------------------------------------------------------------
# LangGraph checkpoint imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from langgraph.checkpoint.memory import InMemorySaver  # always available
except ImportError:  # pragma: no cover
    InMemorySaver = None  # type: ignore[assignment,misc]

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _POSTGRES_ASYNC_AVAILABLE = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore[assignment,misc]
    _POSTGRES_ASYNC_AVAILABLE = False

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    _POSTGRES_SYNC_AVAILABLE = True
except ImportError:
    PostgresSaver = None  # type: ignore[assignment,misc]
    _POSTGRES_SYNC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Phase 0 forward-compatible imports (try/except so Phase 2 stays independent)
# ---------------------------------------------------------------------------
try:
    from masis.schemas.models import MASISState  # noqa: F401  (type hints only)
except ImportError:
    MASISState = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ENV_KEY = "POSTGRES_URL"
_WARN_IN_MEMORY = (
    "WARNING: Using InMemorySaver. Data WILL be lost on process restart. "
    "Set POSTGRES_URL in your environment for production deployments."
)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_checkpointer(postgres_url: Optional[str] = None):
    """Return the best available synchronous `BaseCheckpointSaver`.

    Tries `PostgresSaver` first (requires `langgraph-checkpoint-postgres`
    installed and `POSTGRES_URL` set).  Falls back to `InMemorySaver` with a
    loud WARNING log if Postgres is unavailable.

    Parameters
    ----------
    postgres_url:
        Override the value read from the ``POSTGRES_URL`` environment variable.
        Useful in tests.

    Returns
    -------
    BaseCheckpointSaver
        A configured, ready-to-use checkpoint saver.  Caller is responsible
        for calling ``checkpointer.setup()`` before graph compilation when
        using Postgres (this is done automatically by `CheckpointerContext`).
    """
    url = postgres_url or os.environ.get(_ENV_KEY)

    if url and _POSTGRES_SYNC_AVAILABLE and PostgresSaver is not None:
        try:
            checkpointer = PostgresSaver.from_conn_string(url)
            logger.info(
                "Persistence: PostgresSaver initialised (sync). "
                "URL=%s", _redact_url(url)
            )
            # Ensure tables exist; harmless if they already do.
            checkpointer.setup()
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Persistence: PostgresSaver failed to connect (%s). "
                "Falling back to InMemorySaver.", exc
            )

    _warn_in_memory()
    return InMemorySaver()


async def get_async_checkpointer(postgres_url: Optional[str] = None):
    """Return the best available *async* `BaseCheckpointSaver`.

    Follows the same fallback logic as `get_checkpointer()` but returns an
    `AsyncPostgresSaver` when Postgres is available.  Must be awaited inside
    an already-running event loop.

    Parameters
    ----------
    postgres_url:
        Override ``POSTGRES_URL`` from the environment.

    Returns
    -------
    BaseCheckpointSaver
        A configured async-compatible checkpoint saver.
    """
    url = postgres_url or os.environ.get(_ENV_KEY)

    if url and _POSTGRES_ASYNC_AVAILABLE and AsyncPostgresSaver is not None:
        try:
            checkpointer = AsyncPostgresSaver.from_conn_string(url)
            logger.info(
                "Persistence: AsyncPostgresSaver initialised. "
                "URL=%s", _redact_url(url)
            )
            await checkpointer.setup()
            return checkpointer
        except Exception as exc:
            logger.warning(
                "Persistence: AsyncPostgresSaver failed to connect (%s). "
                "Falling back to InMemorySaver.", exc
            )

    _warn_in_memory()
    return InMemorySaver()


# ---------------------------------------------------------------------------
# Async context-manager wrapper
# ---------------------------------------------------------------------------

class CheckpointerContext:
    """Async context manager that owns the lifecycle of a LangGraph checkpointer.

    Usage
    -----
    ::

        async with CheckpointerContext() as checkpointer:
            graph = workflow.compile(checkpointer=checkpointer)
            result = await graph.ainvoke(state, config)

    On entry the context manager acquires the checkpointer (Postgres or
    InMemory) and ensures tables are ready.  On exit it cleanly releases
    any connection pool resources if the backend supports it.
    """

    def __init__(self, postgres_url: Optional[str] = None) -> None:
        self._postgres_url = postgres_url
        self._checkpointer: Any = None

    async def __aenter__(self):
        self._checkpointer = await get_async_checkpointer(self._postgres_url)
        return self._checkpointer

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # AsyncPostgresSaver implements __aexit__ for connection-pool cleanup.
        if hasattr(self._checkpointer, "__aexit__"):
            await self._checkpointer.__aexit__(exc_type, exc_val, exc_tb)
        return False  # Do not suppress exceptions


# ---------------------------------------------------------------------------
# State-history helper (used by GET /masis/trace)
# ---------------------------------------------------------------------------

def get_state_history_for_thread(
    graph,
    thread_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return an ordered list of state snapshots for *thread_id*.

    Each entry is a plain ``dict`` with keys ``checkpoint_id``, ``step``,
    ``values``, ``next``, ``created_at`` (ISO-8601 string if available).

    Parameters
    ----------
    graph:
        A compiled LangGraph graph with a checkpointer attached.
    thread_id:
        The thread whose history to retrieve.
    limit:
        Maximum number of historical snapshots to return (most-recent first).

    Returns
    -------
    list[dict]
        Ordered list of snapshot dicts, newest first.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = []

    try:
        history_iter = graph.get_state_history(config)
        for snapshot in history_iter:
            entry: dict[str, Any] = {
                "checkpoint_id": getattr(snapshot, "config", {})
                    .get("configurable", {})
                    .get("checkpoint_id", None),
                "step": getattr(snapshot, "metadata", {}).get("step", None),
                "values": dict(snapshot.values) if snapshot.values else {},
                "next": list(snapshot.next) if snapshot.next else [],
                "created_at": getattr(snapshot, "created_at", None),
            }
            snapshots.append(entry)
            if len(snapshots) >= limit:
                break
    except Exception as exc:
        logger.error(
            "get_state_history_for_thread: failed for thread=%s: %s",
            thread_id, exc
        )

    return snapshots


async def get_state_history_for_thread_async(
    graph,
    thread_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Async variant of `get_state_history_for_thread()`.

    Uses `graph.aget_state_history()` when the graph supports it, otherwise
    falls back to the synchronous `get_state_history`.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = []

    try:
        if hasattr(graph, "aget_state_history"):
            history_iter = graph.aget_state_history(config)
            async for snapshot in history_iter:
                entry: dict[str, Any] = {
                    "checkpoint_id": getattr(snapshot, "config", {})
                        .get("configurable", {})
                        .get("checkpoint_id", None),
                    "step": getattr(snapshot, "metadata", {}).get("step", None),
                    "values": dict(snapshot.values) if snapshot.values else {},
                    "next": list(snapshot.next) if snapshot.next else [],
                    "created_at": getattr(snapshot, "created_at", None),
                }
                snapshots.append(entry)
                if len(snapshots) >= limit:
                    break
        else:
            snapshots = get_state_history_for_thread(graph, thread_id, limit)
    except Exception as exc:
        logger.error(
            "get_state_history_for_thread_async: failed for thread=%s: %s",
            thread_id, exc
        )

    return snapshots


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _warn_in_memory() -> None:
    """Emit the mandatory InMemorySaver warning via both logger and print."""
    logger.warning(_WARN_IN_MEMORY)
    print(_WARN_IN_MEMORY)  # Visible even when logging is not configured


def _redact_url(url: str) -> str:
    """Return a URL with the password replaced by '***' for safe logging."""
    import re
    return re.sub(r"(?<=://)([^:]+):([^@]+)@", r"\1:***@", url)
