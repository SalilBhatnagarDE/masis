"""
masis.graph.runner
==================
High-level runner functions for the MASIS compiled graph (ENG-14 M2/M3).

This module provides the public API for invoking the MASIS pipeline:

    compile_graph()     -- Build or return the compiled graph singleton
    get_graph()         -- Return the compiled graph singleton (lazy init)
    ainvoke_graph()     -- Async invocation: query in -> final state out
    stream_graph()      -- Async generator: yields state events as they happen
    generate_thread_id() -- Create a new UUID-based thread identifier

Thread Management
-----------------
Every MASIS query runs within a LangGraph *thread* identified by a UUID.  The
thread_id is embedded in the ``config["configurable"]["thread_id"]`` dict and
is used by the checkpointer to persist state per conversation.

The runner manages thread IDs so callers (API, CLI, tests) can either supply
their own or let the runner generate one.

MF-IDs covered
--------------
MF-MEM-08  Checkpoint persistence through compiled graph's checkpointer
MF-API-01  ainvoke_graph is the backend for POST /masis/query
MF-API-05  stream_graph is the backend for GET /masis/stream
MF-SUP-17  decision_log populated via supervisor_node (verified here)

Exports
-------
compile_graph, get_graph, ainvoke_graph, stream_graph, generate_thread_id
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from masis.graph.workflow import compile_workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_compiled_graph = None
_graph_lock_initialized = False


# ---------------------------------------------------------------------------
# Thread ID management
# ---------------------------------------------------------------------------

def generate_thread_id() -> str:
    """Generate a new unique thread identifier for a MASIS query session.

    Returns a UUID-4 string.  Thread IDs are used by LangGraph's checkpointer
    to isolate state between concurrent query sessions.

    Returns
    -------
    str
        A UUID-4 string such as ``"a1b2c3d4-e5f6-7890-abcd-ef1234567890"``.
    """
    thread_id = str(uuid.uuid4())
    logger.debug("generate_thread_id: created %s", thread_id)
    return thread_id


def make_config(
    thread_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a LangGraph-compatible config dict.

    Parameters
    ----------
    thread_id : str, optional
        Existing thread ID to reuse (e.g. for HITL resume).  If ``None``,
        a new UUID is generated.
    extra : dict, optional
        Additional keys to merge into ``config["configurable"]``.

    Returns
    -------
    dict
        A config dict of the form ``{"configurable": {"thread_id": ...}}``.
    """
    if thread_id is None:
        thread_id = generate_thread_id()

    configurable: Dict[str, Any] = {"thread_id": thread_id}
    if extra:
        configurable.update(extra)

    config: Dict[str, Any] = {"configurable": configurable}
    logger.debug("make_config: %s", config)
    return config


# ---------------------------------------------------------------------------
# Graph lifecycle
# ---------------------------------------------------------------------------

def compile_graph(
    checkpointer: Optional[Any] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    force: bool = False,
):
    """Compile the MASIS graph and cache it as a module-level singleton.

    On the first call, ``compile_workflow()`` is invoked and the result is
    stored.  Subsequent calls return the cached graph unless ``force=True``.

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        Passed through to ``compile_workflow()``.  When ``None``, the default
        checkpointer selection logic applies (Postgres or InMemory).
    interrupt_before : list[str], optional
        Node names to interrupt before (HITL support).
    interrupt_after : list[str], optional
        Node names to interrupt after (HITL support).
    force : bool
        When ``True``, discard the existing singleton and recompile.  Useful
        in tests or after configuration changes.

    Returns
    -------
    CompiledGraph
        The compiled MASIS LangGraph graph.
    """
    global _compiled_graph

    if _compiled_graph is not None and not force:
        logger.debug("compile_graph: returning cached graph singleton")
        return _compiled_graph

    logger.info("compile_graph: compiling MASIS graph%s", " (forced)" if force else "")
    _compiled_graph = compile_workflow(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    logger.info("compile_graph: singleton cached successfully")
    return _compiled_graph


def get_graph():
    """Return the compiled graph singleton, compiling it lazily if needed.

    This is the preferred entry point for most consumers (API layer, CLI).
    It guarantees the graph is compiled exactly once across the process lifetime.

    Returns
    -------
    CompiledGraph
        The compiled MASIS LangGraph graph.

    Raises
    ------
    RuntimeError
        If graph compilation fails (e.g. missing imports, bad schema).
    """
    global _compiled_graph

    if _compiled_graph is not None:
        return _compiled_graph

    logger.info("get_graph: first access — compiling graph")
    try:
        return compile_graph()
    except Exception as exc:
        logger.error("get_graph: compilation failed: %s", exc, exc_info=True)
        raise RuntimeError(
            f"MASIS graph compilation failed: {exc}. "
            "Check that all node modules and schemas are importable."
        ) from exc


def reset_graph() -> None:
    """Clear the compiled graph singleton.

    Used in tests to force a fresh compilation on the next ``get_graph()`` or
    ``compile_graph()`` call.
    """
    global _compiled_graph
    _compiled_graph = None
    logger.debug("reset_graph: singleton cleared")


# ---------------------------------------------------------------------------
# Invocation helpers
# ---------------------------------------------------------------------------

def _build_initial_state(
    query: str,
    initial_state_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the initial MASISState dict for a new query.

    Populates immutable and default fields so the supervisor_node receives a
    well-formed state on the first turn.

    Parameters
    ----------
    query : str
        The user's natural-language query.

    Returns
    -------
    dict
        A dict conforming to the MASISState schema with sensible defaults.
    """
    # Import BudgetTracker for proper token budget initialization
    try:
        from masis.schemas.models import BudgetTracker
        budget = BudgetTracker()
    except ImportError:
        # Fallback: use a dict that matches the schema if schemas not available
        budget = {
            "max_tokens": 100_000,
            "total_tokens_used": 0,
            "max_cost_usd": 0.50,
            "total_cost_usd": 0.0,
        }

    state = {
        "original_query": query,
        "query_id": str(uuid.uuid4()),
        "task_dag": [],
        "stop_condition": "",
        "iteration_count": 0,
        "next_tasks": [],
        "supervisor_decision": "",
        "last_task_result": None,
        "batch_task_results": [],
        "parallel_batch_mode": False,
        "evidence_board": [],
        "critique_notes": None,
        "synthesis_output": None,
        "quality_scores": {},
        "validation_pass": False,
        "validation_round": 0,
        "token_budget": budget,
        "api_call_counts": {},
        "start_time": time.time(),
        "force_synthesize": False,
        "enable_ambiguity_hitl": False,
        "hitl_payload": None,
        "decision_log": [],
    }
    if initial_state_overrides:
        state.update(initial_state_overrides)
    return state


async def ainvoke_graph(
    query: str,
    config: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    initial_state_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Asynchronously invoke the MASIS pipeline end-to-end.

    This is the primary async entry point.  It builds the initial state from
    the query string, merges in any user-supplied config, and runs the graph
    to completion (or until an interrupt for HITL).

    Parameters
    ----------
    query : str
        The user's natural-language question.
    config : dict, optional
        LangGraph config dict.  If provided, must contain
        ``config["configurable"]["thread_id"]``.  When ``None``, a new config
        is generated automatically.
    thread_id : str, optional
        Shorthand for setting the thread_id without building the full config
        dict.  Ignored if ``config`` is already provided.

    Returns
    -------
    dict
        The final MASISState after the graph reaches END or an interrupt.

    Raises
    ------
    RuntimeError
        If the graph is not compilable.
    Exception
        Propagates any unhandled exception from graph execution after logging.
    """
    graph = get_graph()

    if config is None:
        config = make_config(thread_id=thread_id)

    actual_thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(
        "ainvoke_graph: starting query=%r thread_id=%s",
        query[:120],
        actual_thread_id,
    )

    initial_state = _build_initial_state(
        query,
        initial_state_overrides=initial_state_overrides,
    )

    try:
        result = await graph.ainvoke(initial_state, config=config)
        logger.info(
            "ainvoke_graph: completed thread_id=%s decision_log_len=%d",
            actual_thread_id,
            len(result.get("decision_log", [])),
        )
        return result
    except Exception as exc:
        logger.error(
            "ainvoke_graph: failed thread_id=%s error=%s",
            actual_thread_id,
            exc,
            exc_info=True,
        )
        raise


async def stream_graph(
    query: str,
    config: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    stream_mode: str = "values",
    initial_state_overrides: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Asynchronously stream MASIS pipeline execution events.

    Yields state snapshots or update dicts as each node completes, enabling
    real-time progress reporting via SSE (MF-API-05).

    Parameters
    ----------
    query : str
        The user's natural-language question.
    config : dict, optional
        LangGraph config dict.  When ``None``, auto-generated.
    thread_id : str, optional
        Shorthand thread_id (ignored if ``config`` is already provided).
    stream_mode : str
        LangGraph stream mode.  One of ``"values"`` (full state after each
        super-step), ``"updates"`` (only the changed keys), or ``"debug"``
        (verbose internal events).  Default: ``"values"``.

    Yields
    ------
    dict
        State snapshots or update dicts depending on ``stream_mode``.

    Raises
    ------
    RuntimeError
        If the graph is not compilable.
    """
    graph = get_graph()

    if config is None:
        config = make_config(thread_id=thread_id)

    actual_thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(
        "stream_graph: starting query=%r thread_id=%s stream_mode=%s",
        query[:120],
        actual_thread_id,
        stream_mode,
    )

    initial_state = _build_initial_state(
        query,
        initial_state_overrides=initial_state_overrides,
    )

    event_count = 0
    try:
        async for event in graph.astream(
            initial_state,
            config=config,
            stream_mode=stream_mode,
        ):
            event_count += 1
            logger.debug(
                "stream_graph: event #%d thread_id=%s keys=%s",
                event_count,
                actual_thread_id,
                list(event.keys()) if isinstance(event, dict) else type(event).__name__,
            )
            yield event
    except Exception as exc:
        logger.error(
            "stream_graph: failed at event #%d thread_id=%s error=%s",
            event_count,
            actual_thread_id,
            exc,
            exc_info=True,
        )
        raise

    logger.info(
        "stream_graph: completed thread_id=%s total_events=%d",
        actual_thread_id,
        event_count,
    )
