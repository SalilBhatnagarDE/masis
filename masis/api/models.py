"""
masis.api.models
================
Pydantic request and response models for the MASIS FastAPI endpoints (ENG-15).

All models use Pydantic v2 BaseModel with strict validation, default values,
and JSON-serialisation support.

Models
------
QueryRequest    -- Body for POST /masis/query (MF-API-01)
ResumeRequest   -- Body for POST /masis/resume (MF-API-02)
QueryResponse   -- Response for POST /masis/query
StatusResponse  -- Response for GET /masis/status/{thread_id} (MF-API-03)
TraceResponse   -- Response for GET /masis/trace/{thread_id} (MF-API-04)

Architecture reference
----------------------
final_architecture_and_flow.md Section 23.10
engineering_tasks.md ENG-15 M1-M5
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /masis/query (MF-API-01).

    Accepts a natural-language query and optional model overrides that
    replace the default model routing for this specific query session.

    Examples
    --------
    Minimal::

        {"query": "What was TechCorp's Q3 FY26 revenue?"}

    With model overrides::

        {
            "query": "Compare cloud revenue to competitors",
            "model_overrides": {"researcher": "gpt-4.1-nano"}
        }
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The natural-language query to analyse.",
    )
    model_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional per-role model overrides for this query session. "
            "Keys are role names (e.g. 'researcher', 'supervisor_plan', 'skeptic'). "
            "Values are model identifiers (e.g. 'gpt-4.1', 'gpt-4.1-nano')."
        ),
    )


class ResumeRequest(BaseModel):
    """Request body for POST /masis/resume (MF-API-02).

    After a human-in-the-loop interrupt (e.g. ambiguity gate, risk gate, DAG
    approval), the client sends the user's chosen action to resume the graph.

    Examples
    --------
    Expand to web::

        {
            "thread_id": "a1b2c3d4-...",
            "action": "expand_to_web",
            "data": {}
        }

    Accept partial::

        {
            "thread_id": "a1b2c3d4-...",
            "action": "accept_partial",
            "data": {"missing_ok": true}
        }

    Cancel::

        {
            "thread_id": "a1b2c3d4-...",
            "action": "cancel",
            "data": {}
        }
    """

    thread_id: str = Field(
        ...,
        min_length=1,
        description="The thread_id of the paused query session.",
    )
    action: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The resume action chosen by the user.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data payload for the resume action.",
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    """Response for POST /masis/query (MF-API-01).

    Returned immediately after the graph invocation is kicked off in the
    background.  The client uses ``thread_id`` to poll status or connect
    to the SSE stream.
    """

    thread_id: str = Field(
        ...,
        description="Unique identifier for this query session.",
    )
    status: Literal["processing", "queued", "error"] = Field(
        default="processing",
        description="Initial status of the query.",
    )
    message: str = Field(
        default="Query accepted and processing started.",
        description="Human-readable status message.",
    )


class TaskSummary(BaseModel):
    """Compact representation of a single task in the DAG for API responses."""

    task_id: str = Field(default="", description="Unique task identifier.")
    task_type: str = Field(default="", description="Agent type (researcher, skeptic, etc.).")
    status: str = Field(default="pending", description="Current status.")
    query: str = Field(default="", description="Task-specific query.")


class StatusResponse(BaseModel):
    """Response for GET /masis/status/{thread_id} (MF-API-03).

    Provides a lightweight status snapshot suitable for client polling.
    """

    thread_id: str = Field(
        ...,
        description="The thread_id of the query session.",
    )
    status: str = Field(
        default="unknown",
        description=(
            "Current status: processing, paused (HITL), completed, failed, "
            "or unknown if thread_id is not found."
        ),
    )
    iteration_count: int = Field(
        default=0,
        description="Number of supervisor iterations completed so far.",
    )
    tasks_total: int = Field(
        default=0,
        description="Total number of tasks in the DAG.",
    )
    tasks_done: int = Field(
        default=0,
        description="Number of tasks with status 'done'.",
    )
    current_task: Optional[str] = Field(
        default=None,
        description="The task_id of the currently running task, if any.",
    )
    supervisor_decision: str = Field(
        default="",
        description="The last supervisor decision string.",
    )
    is_interrupted: bool = Field(
        default=False,
        description="True if the graph is paused at an interrupt (HITL).",
    )
    next_nodes: List[str] = Field(
        default_factory=list,
        description="The graph node(s) that will execute next.",
    )


class TraceEntry(BaseModel):
    """A single checkpoint entry in the audit trail."""

    checkpoint_id: Optional[str] = Field(default=None, description="Checkpoint identifier.")
    step: Optional[int] = Field(default=None, description="Step number in the graph execution.")
    next_nodes: List[str] = Field(default_factory=list, description="Nodes pending after this step.")
    created_at: Optional[str] = Field(default=None, description="ISO-8601 timestamp.")
    values_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summarised state values (sensitive fields redacted).",
    )


class TraceResponse(BaseModel):
    """Response for GET /masis/trace/{thread_id} (MF-API-04).

    Returns the full audit trail for a query session: DAG, decisions,
    quality scores, and checkpoint history.
    """

    thread_id: str = Field(
        ...,
        description="The thread_id of the query session.",
    )
    total_steps: int = Field(
        default=0,
        description="Total number of checkpoint steps.",
    )
    decision_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of supervisor decisions with mode, reason, cost.",
    )
    quality_scores: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final validation quality scores (faithfulness, citation_accuracy, etc.).",
    )
    task_dag: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The task DAG with final statuses.",
    )
    checkpoints: List[TraceEntry] = Field(
        default_factory=list,
        description="Ordered checkpoint history (most recent first).",
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The final synthesised answer, if available.",
    )


# ---------------------------------------------------------------------------
# SSE Event Models (used internally by the stream endpoint)
# ---------------------------------------------------------------------------

class SSEEvent(BaseModel):
    """Typed server-sent event for GET /masis/stream/{thread_id} (MF-API-05).

    Event types:
        plan_created     -- Supervisor has created the initial task DAG
        task_started     -- Executor has begun dispatching a task
        task_completed   -- A task agent has returned its result
        hitl_required    -- Graph is paused, awaiting human input
        answer_ready     -- Final synthesised answer is available
        error            -- An error occurred during processing
        heartbeat        -- Keep-alive ping (every 15s during processing)
    """

    event_type: Literal[
        "plan_created",
        "task_started",
        "task_completed",
        "hitl_required",
        "answer_ready",
        "error",
        "heartbeat",
        "state_update",
    ] = Field(..., description="The type of SSE event.")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when the event was generated.",
    )
