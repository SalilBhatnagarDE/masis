"""
masis.api.main
==============
FastAPI application for the MASIS multi-agent pipeline (ENG-15).

Endpoints
---------
POST /masis/query           -- Start new query, return thread_id (MF-API-01)
POST /masis/resume          -- Resume from HITL interrupt (MF-API-02)
GET  /masis/status/{id}     -- Poll run status (MF-API-03)
GET  /masis/trace/{id}      -- Full audit trail (MF-API-04)
GET  /masis/stream/{id}     -- SSE stream of state updates (MF-API-05)

Architecture reference
----------------------
final_architecture_and_flow.md Section 23.10
engineering_tasks.md ENG-15 M1-M5
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from masis.api.models import (
    QueryRequest,
    QueryResponse,
    ResumeRequest,
    SSEEvent,
    StatusResponse,
    TraceResponse,
    TraceEntry,
)
from masis.graph.runner import (
    ainvoke_graph,
    generate_thread_id,
    get_graph,
    make_config,
    stream_graph,
    _build_initial_state,
)
from masis.infra.persistence import (
    get_state_history_for_thread_async,
)
from masis.infra.tracing import get_tracing_callback, trace_graph_invocation
from masis.infra.metrics import (
    MetricsMiddleware,
    active_queries,
    extract_metrics_from_result,
    get_metrics_app,
    record_query_metrics,
    errors_total,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory tracking for background tasks
# ---------------------------------------------------------------------------
# Maps thread_id -> {"status": str, "task": asyncio.Task | None,
#                     "result": dict | None, "error": str | None,
#                     "start_time": float}
_active_runs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Background graph runner
# ---------------------------------------------------------------------------

async def _run_graph_background(
    query: str,
    thread_id: str,
    config: Dict[str, Any],
    model_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """Execute the MASIS graph in the background and update _active_runs.

    This is the target of ``asyncio.create_task`` called by the /query endpoint.
    It handles lifecycle tracking, metrics recording, and error capture.
    """
    run_entry = _active_runs.get(thread_id)
    if run_entry is None:
        logger.error("_run_graph_background: no entry for thread_id=%s", thread_id)
        return

    start_time = run_entry["start_time"]
    active_queries.inc()

    try:
        with trace_graph_invocation(thread_id, query) as span:
            result = await ainvoke_graph(query, config=config, thread_id=None)
            span.set_attribute("iterations", result.get("iteration_count", 0))
            span.set_attribute("status", "completed")

        run_entry["status"] = "completed"
        run_entry["result"] = result

        # Record Prometheus metrics
        metrics_data = extract_metrics_from_result(result, start_time)
        record_query_metrics(
            latency_seconds=metrics_data["latency_seconds"],
            cost_usd=metrics_data["cost_usd"],
            agent_type_counts=metrics_data["agent_type_counts"],
            fast_path_decisions=metrics_data["fast_path_decisions"],
            total_decisions=metrics_data["total_decisions"],
        )

        logger.info(
            "Background run completed: thread_id=%s latency=%.2fs",
            thread_id, metrics_data["latency_seconds"],
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        run_entry["status"] = "failed"
        run_entry["error"] = error_msg
        errors_total.labels(error_type=type(exc).__name__).inc()

        # Still record partial metrics on failure
        latency = time.time() - start_time
        record_query_metrics(
            latency_seconds=latency,
            cost_usd=0.0,
            agent_type_counts={},
            error_type=type(exc).__name__,
        )

        logger.error(
            "Background run failed: thread_id=%s error=%s",
            thread_id, error_msg, exc_info=True,
        )
    finally:
        active_queries.dec()


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the MASIS FastAPI application.

    Returns a fully wired FastAPI app with all five endpoints, Prometheus
    metrics middleware, and the /metrics scrape endpoint.

    Returns
    -------
    FastAPI
        The configured application instance.
    """
    app = FastAPI(
        title="MASIS API",
        description=(
            "Multi-Agent Supervised Intelligence System -- "
            "REST API for query orchestration, HITL resume, status polling, "
            "audit trail retrieval, and real-time event streaming."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware ---
    app.add_middleware(MetricsMiddleware)

    # --- Mount Prometheus /metrics ---
    metrics_asgi = get_metrics_app()
    app.mount("/metrics", metrics_asgi)

    # --- Register endpoint handlers ---
    _register_endpoints(app)

    return app


def _register_endpoints(app: FastAPI) -> None:
    """Attach all MASIS endpoint handlers to the FastAPI app."""

    # -----------------------------------------------------------------------
    # MF-API-01: POST /masis/query -- Start a new query
    # -----------------------------------------------------------------------
    @app.post(
        "/masis/query",
        response_model=QueryResponse,
        summary="Start a new MASIS query",
        description="Accept a query, generate a thread_id, start graph processing in the background.",
        tags=["Query"],
    )
    async def start_query(request: QueryRequest) -> QueryResponse:
        thread_id = generate_thread_id()
        config = make_config(thread_id=thread_id)

        # Merge model overrides into config if provided
        if request.model_overrides:
            config["configurable"]["model_overrides"] = request.model_overrides

        # Register in active runs tracker
        _active_runs[thread_id] = {
            "status": "processing",
            "task": None,
            "result": None,
            "error": None,
            "start_time": time.time(),
            "query": request.query,
        }

        # Start graph execution in background
        task = asyncio.create_task(
            _run_graph_background(
                query=request.query,
                thread_id=thread_id,
                config=config,
                model_overrides=request.model_overrides,
            )
        )
        _active_runs[thread_id]["task"] = task

        logger.info(
            "POST /masis/query: started thread_id=%s query=%r",
            thread_id, request.query[:100],
        )

        return QueryResponse(
            thread_id=thread_id,
            status="processing",
            message="Query accepted and processing started.",
        )

    # -----------------------------------------------------------------------
    # MF-API-02: POST /masis/resume -- Resume from HITL pause
    # -----------------------------------------------------------------------
    @app.post(
        "/masis/resume",
        summary="Resume a paused MASIS query",
        description="Resume a query that was paused at a human-in-the-loop interrupt.",
        tags=["Query"],
    )
    async def resume_query(request: ResumeRequest) -> JSONResponse:
        thread_id = request.thread_id
        config = make_config(thread_id=thread_id)

        try:
            from langgraph.types import Command
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="langgraph.types.Command is not available. "
                       "Ensure langgraph>=0.2 is installed.",
            )

        resume_value = {"action": request.action, **request.data}

        logger.info(
            "POST /masis/resume: thread_id=%s action=%s",
            thread_id, request.action,
        )

        try:
            graph = get_graph()
            result = await graph.ainvoke(
                Command(resume=resume_value),
                config=config,
            )

            # Update active runs if tracked
            if thread_id in _active_runs:
                # Determine if completed or still processing
                next_nodes = []
                try:
                    state = await graph.aget_state(config)
                    next_nodes = list(state.next) if state.next else []
                except Exception:
                    pass

                if not next_nodes:
                    _active_runs[thread_id]["status"] = "completed"
                    _active_runs[thread_id]["result"] = result
                else:
                    _active_runs[thread_id]["status"] = "processing"

            # Serialise result safely
            serialisable_result = _safe_serialise(result)

            return JSONResponse(
                content={
                    "thread_id": thread_id,
                    "status": "resumed",
                    "result": serialisable_result,
                },
                status_code=200,
            )

        except Exception as exc:
            logger.error(
                "POST /masis/resume: failed thread_id=%s error=%s",
                thread_id, exc, exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to resume query: {type(exc).__name__}: {exc}",
            )

    # -----------------------------------------------------------------------
    # MF-API-03: GET /masis/status/{thread_id} -- Check run status
    # -----------------------------------------------------------------------
    @app.get(
        "/masis/status/{thread_id}",
        response_model=StatusResponse,
        summary="Check MASIS query status",
        description="Return the current status, iteration count, and task progress.",
        tags=["Status"],
    )
    async def get_status(thread_id: str) -> StatusResponse:
        # First check if graph state is available
        try:
            graph = get_graph()
            config = make_config(thread_id=thread_id)
            state = await graph.aget_state(config)
        except Exception as exc:
            logger.warning(
                "GET /masis/status: aget_state failed for thread_id=%s: %s",
                thread_id, exc,
            )
            state = None

        # Build response from graph state if available
        if state is not None and state.values:
            values = state.values
            task_dag = values.get("task_dag", [])
            next_nodes = list(state.next) if state.next else []
            is_interrupted = len(next_nodes) > 0 and values.get("hitl_payload") is not None

            # Count done tasks
            tasks_done = 0
            current_task_id = None
            tasks_total = 0
            if isinstance(task_dag, list):
                tasks_total = len(task_dag)
                for task in task_dag:
                    task_status = _get_task_status(task)
                    if task_status == "done":
                        tasks_done += 1
                    elif task_status == "running":
                        current_task_id = _get_task_id(task)

            # Determine overall status
            supervisor_decision = values.get("supervisor_decision", "")
            if is_interrupted:
                overall_status = "paused"
            elif supervisor_decision in ("", "continue", "dispatch"):
                overall_status = "processing"
            elif supervisor_decision in ("ready_for_validation", "force_synthesize"):
                overall_status = "processing"
            elif not next_nodes and tasks_done == tasks_total and tasks_total > 0:
                overall_status = "completed"
            else:
                # Check in-memory tracker
                run_entry = _active_runs.get(thread_id, {})
                overall_status = run_entry.get("status", "processing")

            return StatusResponse(
                thread_id=thread_id,
                status=overall_status,
                iteration_count=values.get("iteration_count", 0),
                tasks_total=tasks_total,
                tasks_done=tasks_done,
                current_task=current_task_id,
                supervisor_decision=str(supervisor_decision),
                is_interrupted=is_interrupted,
                next_nodes=next_nodes,
            )

        # Fallback: check in-memory tracker
        run_entry = _active_runs.get(thread_id)
        if run_entry is not None:
            return StatusResponse(
                thread_id=thread_id,
                status=run_entry["status"],
                iteration_count=0,
                tasks_total=0,
                tasks_done=0,
                current_task=None,
                supervisor_decision="",
                is_interrupted=False,
                next_nodes=[],
            )

        # Thread not found
        return StatusResponse(
            thread_id=thread_id,
            status="unknown",
        )

    # -----------------------------------------------------------------------
    # MF-API-04: GET /masis/trace/{thread_id} -- Full audit trail
    # -----------------------------------------------------------------------
    @app.get(
        "/masis/trace/{thread_id}",
        response_model=TraceResponse,
        summary="Get full audit trail",
        description="Return the complete checkpoint history, DAG, decisions, and quality scores.",
        tags=["Trace"],
    )
    async def get_trace(thread_id: str) -> TraceResponse:
        try:
            graph = get_graph()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        snapshots = await get_state_history_for_thread_async(
            graph, thread_id, limit=100,
        )

        if not snapshots:
            return TraceResponse(
                thread_id=thread_id,
                total_steps=0,
                decision_log=[],
                quality_scores={},
                task_dag=[],
                checkpoints=[],
                final_answer=None,
            )

        # Most recent snapshot is first
        latest = snapshots[0] if snapshots else {}
        latest_values = latest.get("values", {})

        # Extract decision log from latest state
        decision_log = latest_values.get("decision_log", [])
        if not isinstance(decision_log, list):
            decision_log = []

        # Extract quality scores
        quality_scores = latest_values.get("quality_scores", {})
        if not isinstance(quality_scores, dict):
            quality_scores = {}

        # Extract task DAG
        task_dag_raw = latest_values.get("task_dag", [])
        task_dag_serialised: list = []
        if isinstance(task_dag_raw, list):
            for task in task_dag_raw:
                task_dag_serialised.append(_safe_serialise(task))

        # Extract final answer from synthesis_output
        final_answer = None
        synthesis = latest_values.get("synthesis_output")
        if isinstance(synthesis, dict):
            final_answer = synthesis.get("answer")
        elif hasattr(synthesis, "answer"):
            final_answer = getattr(synthesis, "answer", None)

        # Build checkpoint entries
        checkpoints = []
        for snap in snapshots:
            values = snap.get("values", {})
            # Summarise values (exclude large fields for readability)
            summary = {
                "iteration_count": values.get("iteration_count", 0),
                "supervisor_decision": values.get("supervisor_decision", ""),
                "force_synthesize": values.get("force_synthesize", False),
            }
            created_at = snap.get("created_at")
            if created_at is not None and not isinstance(created_at, str):
                created_at = str(created_at)

            checkpoints.append(TraceEntry(
                checkpoint_id=snap.get("checkpoint_id"),
                step=snap.get("step"),
                next_nodes=snap.get("next", []),
                created_at=created_at,
                values_summary=summary,
            ))

        return TraceResponse(
            thread_id=thread_id,
            total_steps=len(snapshots),
            decision_log=_safe_serialise(decision_log),
            quality_scores=_safe_serialise(quality_scores),
            task_dag=task_dag_serialised,
            checkpoints=checkpoints,
            final_answer=final_answer,
        )

    # -----------------------------------------------------------------------
    # MF-API-05: GET /masis/stream/{thread_id} -- SSE streaming
    # -----------------------------------------------------------------------
    @app.get(
        "/masis/stream/{thread_id}",
        summary="Stream MASIS query events via SSE",
        description=(
            "Server-Sent Events stream of typed events during query execution. "
            "Events: plan_created, task_started, task_completed, hitl_required, "
            "answer_ready, error, heartbeat."
        ),
        tags=["Stream"],
    )
    async def stream_events(thread_id: str) -> StreamingResponse:
        run_entry = _active_runs.get(thread_id)

        async def event_generator():
            """Generate SSE events from the graph execution or state polling."""
            # If we have a known running query, stream its updates
            if run_entry is not None and run_entry.get("query"):
                query = run_entry["query"]
                config = make_config(thread_id=thread_id)
                event_count = 0
                prev_iteration = -1
                prev_decision = ""

                try:
                    graph = get_graph()
                    initial_state = _build_initial_state(query)

                    async for event in graph.astream(
                        initial_state,
                        config=config,
                        stream_mode="updates",
                    ):
                        event_count += 1

                        if isinstance(event, dict):
                            # Determine event type from the update content
                            sse_event = _classify_stream_event(event, prev_iteration, prev_decision)
                            prev_iteration = sse_event.data.get("iteration_count", prev_iteration)
                            prev_decision = sse_event.data.get("supervisor_decision", prev_decision)

                            yield _format_sse(sse_event)

                    # Final event: answer_ready or completed
                    final_event = SSEEvent(
                        event_type="answer_ready",
                        data={"thread_id": thread_id, "message": "Processing complete."},
                    )
                    yield _format_sse(final_event)

                except Exception as exc:
                    error_event = SSEEvent(
                        event_type="error",
                        data={
                            "thread_id": thread_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    yield _format_sse(error_event)
                    return

            else:
                # Polling mode: stream state updates for an already-running query
                poll_count = 0
                max_polls = 600  # ~5 minutes at 0.5s intervals
                last_iteration = -1

                while poll_count < max_polls:
                    poll_count += 1

                    # Check in-memory tracker
                    entry = _active_runs.get(thread_id)
                    if entry is not None:
                        status = entry.get("status", "unknown")

                        if status == "completed":
                            result = entry.get("result", {})
                            answer = None
                            if isinstance(result, dict):
                                synthesis = result.get("synthesis_output")
                                if isinstance(synthesis, dict):
                                    answer = synthesis.get("answer")
                                elif hasattr(synthesis, "answer"):
                                    answer = getattr(synthesis, "answer", None)

                            done_event = SSEEvent(
                                event_type="answer_ready",
                                data={
                                    "thread_id": thread_id,
                                    "answer": answer,
                                    "status": "completed",
                                },
                            )
                            yield _format_sse(done_event)
                            return

                        elif status == "failed":
                            err_event = SSEEvent(
                                event_type="error",
                                data={
                                    "thread_id": thread_id,
                                    "error": entry.get("error", "Unknown error"),
                                },
                            )
                            yield _format_sse(err_event)
                            return

                    # Try to read current state from graph
                    try:
                        graph = get_graph()
                        config = make_config(thread_id=thread_id)
                        state = await graph.aget_state(config)

                        if state and state.values:
                            iteration = state.values.get("iteration_count", 0)
                            if iteration != last_iteration:
                                last_iteration = iteration
                                update_event = SSEEvent(
                                    event_type="state_update",
                                    data={
                                        "thread_id": thread_id,
                                        "iteration_count": iteration,
                                        "supervisor_decision": state.values.get(
                                            "supervisor_decision", ""
                                        ),
                                    },
                                )
                                yield _format_sse(update_event)
                    except Exception:
                        pass  # Graph might not be ready yet

                    # Heartbeat
                    if poll_count % 30 == 0:
                        heartbeat = SSEEvent(
                            event_type="heartbeat",
                            data={"thread_id": thread_id},
                        )
                        yield _format_sse(heartbeat)

                    await asyncio.sleep(0.5)

                # Timed out
                timeout_event = SSEEvent(
                    event_type="error",
                    data={
                        "thread_id": thread_id,
                        "error": "Stream timeout: no completion after 5 minutes.",
                    },
                )
                yield _format_sse(timeout_event)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, str]:
        """Return a simple health status."""
        return {"status": "healthy", "service": "masis-api", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _format_sse(event: SSEEvent) -> str:
    """Format an SSEEvent as a standard SSE message string.

    Returns a string in the format::

        event: <event_type>
        data: <json_payload>

    Followed by two newlines (SSE message delimiter).
    """
    payload = {
        "type": event.event_type,
        "data": event.data,
        "timestamp": event.timestamp,
    }
    return f"event: {event.event_type}\ndata: {json.dumps(payload, default=str)}\n\n"


def _classify_stream_event(
    update: Dict[str, Any],
    prev_iteration: int,
    prev_decision: str,
) -> SSEEvent:
    """Classify a graph stream update into a typed SSE event.

    Inspects the keys and values in the update dict to determine the most
    appropriate event type.
    """
    # Check if this is a supervisor update with a new DAG (plan_created)
    if "supervisor" in update:
        sv_data = update["supervisor"]
        if isinstance(sv_data, dict):
            task_dag = sv_data.get("task_dag", [])
            iteration = sv_data.get("iteration_count", prev_iteration)

            if iteration == 1 and isinstance(task_dag, list) and len(task_dag) > 0:
                return SSEEvent(
                    event_type="plan_created",
                    data={
                        "iteration_count": iteration,
                        "tasks_count": len(task_dag),
                        "supervisor_decision": sv_data.get("supervisor_decision", ""),
                    },
                )

            # Check for HITL interrupt
            hitl = sv_data.get("hitl_payload")
            if hitl is not None:
                return SSEEvent(
                    event_type="hitl_required",
                    data={
                        "iteration_count": iteration,
                        "hitl_payload": _safe_serialise(hitl),
                    },
                )

            return SSEEvent(
                event_type="state_update",
                data={
                    "node": "supervisor",
                    "iteration_count": iteration,
                    "supervisor_decision": sv_data.get("supervisor_decision", ""),
                },
            )

    # Check for executor update (task started / completed)
    if "executor" in update:
        ex_data = update["executor"]
        if isinstance(ex_data, dict):
            last_result = ex_data.get("last_task_result")
            if last_result is not None:
                return SSEEvent(
                    event_type="task_completed",
                    data={
                        "node": "executor",
                        "last_task_result": _safe_serialise(last_result),
                    },
                )
            return SSEEvent(
                event_type="task_started",
                data={"node": "executor"},
            )

    # Check for validator update
    if "validator" in update:
        val_data = update["validator"]
        if isinstance(val_data, dict):
            scores = val_data.get("quality_scores", {})
            return SSEEvent(
                event_type="state_update",
                data={
                    "node": "validator",
                    "quality_scores": _safe_serialise(scores),
                },
            )

    # Generic update
    return SSEEvent(
        event_type="state_update",
        data={"keys": list(update.keys()) if isinstance(update, dict) else []},
    )


def _get_task_status(task: Any) -> str:
    """Extract status from a task object (dict or Pydantic model)."""
    if isinstance(task, dict):
        return task.get("status", "pending")
    return getattr(task, "status", "pending")


def _get_task_id(task: Any) -> Optional[str]:
    """Extract task_id from a task object (dict or Pydantic model)."""
    if isinstance(task, dict):
        return task.get("task_id")
    return getattr(task, "task_id", None)


def _safe_serialise(obj: Any) -> Any:
    """Recursively convert an object to JSON-safe Python types.

    Handles Pydantic models, dicts, lists, datetimes, and other common types.
    Falls back to ``str(obj)`` for unrecognised types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {k: _safe_serialise(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    return str(obj)


# ---------------------------------------------------------------------------
# Module-level app instance (for uvicorn: uvicorn masis.api.main:app)
# ---------------------------------------------------------------------------

app = create_app()
