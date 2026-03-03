"""
masis.nodes.executor
====================
The Executor sits between the Supervisor and the agents. It reads the task list
from the Supervisor and runs them — one at a time or in parallel.

What it does:
    - Reads state["next_tasks"] from the Supervisor
    - Single task   ->  runs dispatch_with_safety() directly
    - Multiple tasks  ->  returns Send() objects so LangGraph runs them in parallel
    - Routes task.type to the right agent function (researcher, skeptic, synthesizer, web_search)
    - Returns a structured error if task type is unknown
    - Wraps every call in an asyncio timeout so nothing hangs
    - Normalizes every result into AgentOutput for consistent handling
    - Gives each agent only the state fields it needs (filtered view)
    - Writes agent evidence to the shared evidence_board via the dedup reducer
    - Updates BudgetTracker after each call
    - Checks per-agent rate limits before dispatching

Public API
----------
execute_dag_tasks(state)                  --  LangGraph node entry point
dispatch_agent(task, filtered_state)      --  routes task.type to the correct agent
dispatch_with_safety(task, state)         --  timeout + exception wrapper
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 0 schema imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        AgentOutput,
        BudgetTracker,
        EvidenceChunk,
        TaskNode,
    )
    from masis.schemas.thresholds import TOOL_LIMITS
except ImportError:
    logger.warning("masis.schemas not found  --  using stub types for executor.py")

    class AgentOutput:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)
        task_id: str = ""
        agent_type: str = ""
        status: str = "success"
        summary: str = ""
        evidence: list = []
        criteria_result: dict = {}
        tokens_used: int = 0
        cost_usd: float = 0.0
        error_detail: str = ""

    class BudgetTracker:  # type: ignore[no-redef]
        remaining: int = 100_000
        total_tokens_used: int = 0
        total_cost_usd: float = 0.0
        api_calls: dict = {}
        start_time: float = 0.0

        def add(self, tokens: int, cost: float, agent_type: str = "") -> "BudgetTracker":
            return self

    class TaskNode:  # type: ignore[no-redef]
        task_id: str = ""
        type: str = "researcher"
        query: str = ""
        dependencies: list = []
        parallel_group: int = 1
        acceptance_criteria: str = ""
        status: str = "pending"

    TOOL_LIMITS: dict = {  # type: ignore[misc]
        "researcher":  {"max_parallel": 3, "max_total": 20, "timeout_s": 60},
        "web_search":  {"max_parallel": 2, "max_total": 10, "timeout_s": 30},
        "skeptic":     {"max_parallel": 1, "max_total": 15, "timeout_s": 150},
        "synthesizer": {"max_parallel": 1, "max_total": 6,  "timeout_s": 150},
    }

# ---------------------------------------------------------------------------
# Optional LangGraph Send import
# ---------------------------------------------------------------------------

try:
    from langgraph.types import Send
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed  --  parallel Send() dispatch unavailable")

    class Send:  # type: ignore[no-redef]
        """Stub for LangGraph Send when langgraph is not installed."""
        def __init__(self, node: str, payload: Dict[str, Any]) -> None:
            self.node = node
            self.payload = payload

# ---------------------------------------------------------------------------
# Agent function imports with stubs
# ---------------------------------------------------------------------------

try:
    from masis.agents.researcher import run_researcher
    _RESEARCHER_AVAILABLE = True
except ImportError:
    _RESEARCHER_AVAILABLE = False
    logger.warning("masis.agents.researcher not found  --  researcher will return stub output")

    async def run_researcher(task: Any, state: Any) -> Any:  # type: ignore[misc]
        return AgentOutput(
            task_id=getattr(task, "task_id", ""),
            agent_type="researcher",
            status="failed",
            summary="Researcher agent not available (import error)",
            error_detail="masis.agents.researcher not installed",
        )

try:
    from masis.agents.skeptic import run_skeptic
    _SKEPTIC_AVAILABLE = True
except ImportError:
    _SKEPTIC_AVAILABLE = False
    logger.warning("masis.agents.skeptic not found  --  skeptic will return stub output")

    async def run_skeptic(task: Any, state: Any) -> Any:  # type: ignore[misc]
        return AgentOutput(
            task_id=getattr(task, "task_id", ""),
            agent_type="skeptic",
            status="failed",
            summary="Skeptic agent not available (import error)",
            error_detail="masis.agents.skeptic not installed",
        )

try:
    from masis.agents.synthesizer import run_synthesizer
    _SYNTHESIZER_AVAILABLE = True
except ImportError:
    _SYNTHESIZER_AVAILABLE = False
    logger.warning("masis.agents.synthesizer not found  --  synthesizer will return stub output")

    async def run_synthesizer(task: Any, state: Any) -> Any:  # type: ignore[misc]
        return AgentOutput(
            task_id=getattr(task, "task_id", ""),
            agent_type="synthesizer",
            status="failed",
            summary="Synthesizer agent not available (import error)",
            error_detail="masis.agents.synthesizer not installed",
        )

try:
    from masis.agents.web_search import run_web_search
    _WEB_SEARCH_AVAILABLE = True
except ImportError:
    _WEB_SEARCH_AVAILABLE = False
    logger.warning("masis.agents.web_search not found  --  web_search will return stub output")

    async def run_web_search(task: Any, state: Any) -> Any:  # type: ignore[misc]
        return AgentOutput(
            task_id=getattr(task, "task_id", ""),
            agent_type="web_search",
            status="failed",
            summary="Web search agent not available (import error)",
            error_detail="masis.agents.web_search not installed",
        )

# ---------------------------------------------------------------------------
# Valid agent type registry (MF-EXE-03)
# ---------------------------------------------------------------------------

VALID_AGENT_TYPES = {"researcher", "web_search", "skeptic", "synthesizer"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def execute_dag_tasks(state: Dict[str, Any]) -> Union[Dict[str, Any], List[Send]]:
    """LangGraph node: execute next DAG task(s) from the Supervisor (MF-EXE-01/02).

    Handles three scenarios:
    1. force_synthesize: Creates an ad-hoc synthesizer task (budget/loop cap hit)
    2. Multi-task: true parallel fan-out via Send() (fallback: sequential)
    3. Single-task: Direct dispatch

    Args:
        state: Current MASISState with ``next_tasks`` populated by the Supervisor.

    Returns:
        A state update dict with results from all executed tasks.
    """
    # ── Force-synthesize shortcut (MF-SUP-13) ───────────────────────────
    # When supervisor decides force_synthesize, skip regular tasks and
    # run the synthesizer immediately with whatever evidence is available.
    # This MUST terminate — bypass rate limits and guarantee a result.
    sup_decision = state.get("supervisor_decision", "")
    if sup_decision == "force_synthesize" or state.get("force_synthesize", False):
        logger.info("force_synthesize detected -- running synthesizer with current evidence")
        synth_task = TaskNode(
            task_id="SYNTH_FORCE",
            type="synthesizer",
            query=state.get("original_query", ""),
            acceptance_criteria="best-effort synthesis with available evidence",
            status="running",
        )

        # Bypass rate limiter — force_synthesize is a terminal operation
        agent_state = _filtered_state(state, synth_task)
        try:
            output = await asyncio.wait_for(
                dispatch_agent(synth_task, agent_state),
                timeout=120.0,
            )
        except Exception as exc:
            logger.warning("force_synthesize dispatch failed: %s -- returning best-effort", exc)
            # Build a best-effort summary from evidence_board
            evidence = state.get("evidence_board", [])
            evidence_text = " ".join(
                getattr(e, "text", str(e))[:200] for e in evidence[:5]
            ) if evidence else "No evidence was collected."
            output = AgentOutput(  # type: ignore[call-arg]
                task_id="SYNTH_FORCE",
                agent_type="synthesizer",
                status="success",
                summary=f"Best-effort synthesis (forced): {evidence_text[:500]}",
                evidence=evidence[:5] if evidence else [],
            )

        update = _build_state_update(output, state)
        dag: List[TaskNode] = list(state.get("task_dag", []))
        for dag_task in dag:
            if dag_task.task_id == synth_task.task_id:
                dag_task.status = "done" if output.status == "success" else "failed"
                dag_task.result_summary = getattr(output, "summary", "")[:500]
                break
        if dag:
            update["task_dag"] = dag

        # Force-synthesis should still pass through Validator so we always
        # expose quality scores and avoid silent, unvalidated exits.
        update["supervisor_decision"] = "ready_for_validation"
        update["force_synthesize"] = False
        update["parallel_batch_mode"] = False
        return update

    next_tasks: List[TaskNode] = state.get("next_tasks", [])

    if not next_tasks:
        logger.warning("execute_dag_tasks called with empty next_tasks -- returning no-op")
        return {"last_task_result": None, "batch_task_results": [], "parallel_batch_mode": False}

    # ── Multi-task: run sequentially within this node (MF-EXE-02) ───────
    if len(next_tasks) > 1:
        enable_parallel_send = (
            str(os.getenv("ENABLE_PARALLEL_SEND", "0")).strip().lower()
            in {"1", "true", "yes"}
        )
        if enable_parallel_send and _LANGGRAPH_AVAILABLE:
            logger.info(
                "Dispatching %d tasks in parallel via Send(): %s",
                len(next_tasks),
                [t.task_id for t in next_tasks],
            )
            return [
                Send(
                    "executor",
                    {
                        "next_tasks": [task],
                        "_fanout_branch": True,
                        "parallel_batch_mode": True,
                    },
                )
                for task in next_tasks
            ]
        if enable_parallel_send and not _LANGGRAPH_AVAILABLE:
            logger.warning(
                "ENABLE_PARALLEL_SEND is set but LangGraph Send is unavailable; "
                "falling back to sequential for %d tasks",
                len(next_tasks),
            )
        else:
            logger.info(
                "Dispatching %d tasks sequentially inside executor wave "
                "(ENABLE_PARALLEL_SEND disabled)",
                len(next_tasks),
            )
        return await _run_tasks_sequentially(next_tasks, state)

    # ── Single-task dispatch (MF-EXE-01) ────────────────────────────────
    task = next_tasks[0]
    logger.info("Dispatching single task %s (%s)", task.task_id, task.type)
    output = await dispatch_with_safety(task, state)
    in_fanout_branch = bool(state.get("_fanout_branch", False))
    update = _build_state_update(
        output,
        state,
        include_budget=not in_fanout_branch,
    )

    # Persist single-task status transitions so scheduler state is durable.
    dag: List[TaskNode] = list(state.get("task_dag", []))
    for dag_task in dag:
        if dag_task.task_id == task.task_id:
            dag_task.status = "done" if output.status == "success" else "failed"
            dag_task.result_summary = getattr(output, "summary", "")[:500]
            break
    if dag and not in_fanout_branch:
        update["task_dag"] = dag

    if in_fanout_branch:
        # Parallel branches should not overwrite shared counters/task_dag.
        # Supervisor aggregates batch outputs in one place.
        update["parallel_batch_mode"] = True

    return update


async def dispatch_agent(task: TaskNode, filtered_state: Dict[str, Any]) -> AgentOutput:
    """Route task.type to the correct agent function (MF-EXE-03/04).

    Args:
        task: The TaskNode to execute.
        filtered_state: A state view containing only the fields this agent needs.

    Returns:
        Normalised AgentOutput from the chosen agent.
    """
    agent_type = getattr(task, "type", "")

    if agent_type not in VALID_AGENT_TYPES:
        logger.error("Unknown agent type: %s (task_id=%s)", agent_type, task.task_id)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task.task_id,
            agent_type=agent_type,
            status="failed",
            summary="unknown_agent_type",
            error_detail=(
                f"'{agent_type}' is not a valid agent type. "
                f"Valid types: {sorted(VALID_AGENT_TYPES)}"
            ),
        )

    logger.debug("dispatch_agent: task_id=%s type=%s", task.task_id, agent_type)

    if agent_type == "researcher":
        raw = await run_researcher(task, filtered_state)
    elif agent_type == "web_search":
        raw = await run_web_search(task, filtered_state)
    elif agent_type == "skeptic":
        raw = await run_skeptic(task, filtered_state)
    elif agent_type == "synthesizer":
        raw = await run_synthesizer(task, filtered_state)
    else:
        # Should be unreachable due to the guard above
        raise ValueError(f"Unhandled agent type: {agent_type}")

    # Normalise raw output to AgentOutput if it is not already one
    return _normalise_output(raw, task)


async def dispatch_with_safety(task: TaskNode, state: Dict[str, Any]) -> AgentOutput:
    """Timeout + exception wrapper around dispatch_agent (MF-EXE-05).

    Steps:
        1. Pre-check rate limit (MF-EXE-10).
        2. Prepare filtered state view (MF-EXE-07).
        3. Wrap in asyncio.wait_for with per-agent timeout.
        4. Catch TimeoutError  ->  return failed AgentOutput.
        5. Catch all other exceptions  ->  return failed AgentOutput.
        6. On success: update budget (MF-EXE-09) and api_call_counts.

    Args:
        task: The TaskNode to execute.
        state: Full MASISState (used for rate-limit check and state filtering).

    Returns:
        AgentOutput  --  never raises.
    """
    agent_type = getattr(task, "type", "")
    task_id = getattr(task, "task_id", "unknown")

    # ── Rate limit pre-check (MF-EXE-10) ────────────────────────────────────
    rate_limit_error = _check_rate_limit(task, state)
    if rate_limit_error:
        logger.warning("Rate limit exceeded for %s (task=%s)", agent_type, task_id)
        return rate_limit_error

    # ── Filter state for the agent (MF-EXE-07) ──────────────────────────────
    agent_state = _filtered_state(state, task)

    # ── Timeout configuration ────────────────────────────────────────────────
    limits = TOOL_LIMITS.get(agent_type, {})
    timeout_s: float = float(limits.get("timeout_s", 30))

    start_ts = time.monotonic()

    try:
        output = await asyncio.wait_for(
            dispatch_agent(task, agent_state),
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - start_ts
        logger.info(
            "Task %s completed in %.2fs (status=%s)", task_id, elapsed, output.status
        )
        return output

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start_ts
        logger.error("Task %s timed out after %.1fs", task_id, elapsed)
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type=agent_type,
            status="timeout",
            summary=f"Task timed out after {timeout_s:.0f}s",
            error_detail=f"asyncio.TimeoutError after {timeout_s:.0f}s",
        )

    except Exception as exc:
        elapsed = time.monotonic() - start_ts
        logger.error(
            "Task %s raised %s after %.2fs: %s",
            task_id, type(exc).__name__, elapsed, exc, exc_info=True,
        )
        return AgentOutput(  # type: ignore[call-arg]
            task_id=task_id,
            agent_type=agent_type,
            status="failed",
            summary=f"Agent error: {type(exc).__name__}",
            error_detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _filtered_state(state: Dict[str, Any], task: TaskNode) -> Dict[str, Any]:
    """Return a filtered state view appropriate for the given agent type (MF-EXE-07).

    Each agent receives only the information it needs:
        - researcher  : original_query, task (query only)
        - web_search  : task.query only
        - skeptic     : original_query, evidence_board, task_dag
        - synthesizer : original_query, evidence_board, critique_notes, task_dag,
                        supervisor_decision (needed for partial mode detection)

    Args:
        state: Full MASISState.
        task: The TaskNode being dispatched (determines which filter to apply).

    Returns:
        Minimal dict for the agent.
    """
    agent_type = getattr(task, "type", "")

    if agent_type == "researcher":
        return {
            "original_query": state.get("original_query", ""),
            "task": task,
        }

    if agent_type == "web_search":
        return {
            "task": task,
        }

    if agent_type == "skeptic":
        return {
            "original_query": state.get("original_query", ""),
            "evidence_board": state.get("evidence_board", []),
            "task_dag": state.get("task_dag", []),
            "task": task,
        }

    if agent_type == "synthesizer":
        return {
            "original_query": state.get("original_query", ""),
            "evidence_board": state.get("evidence_board", []),
            "critique_notes": state.get("critique_notes"),
            "task_dag": state.get("task_dag", []),
            "supervisor_decision": state.get("supervisor_decision", ""),
            "task": task,
        }

    # Fallback  --  all fields except evidence_board (safety)
    return {
        "original_query": state.get("original_query", ""),
        "task": task,
    }


def _check_rate_limit(task: TaskNode, state: Dict[str, Any]) -> Optional[AgentOutput]:
    """Return a rate-limit AgentOutput if the agent type has reached its call cap (MF-EXE-10).

    Args:
        task: The TaskNode about to be dispatched.
        state: Current MASISState (contains api_call_counts).

    Returns:
        AgentOutput with status="rate_limited" if the cap is exceeded, else None.
    """
    agent_type = getattr(task, "type", "")
    limits = TOOL_LIMITS.get(agent_type, {})
    max_total: int = limits.get("max_total", 999)

    api_call_counts: Dict[str, int] = state.get("api_call_counts", {})
    current_count = api_call_counts.get(agent_type, 0)

    if current_count >= max_total:
        return AgentOutput(  # type: ignore[call-arg]
            task_id=getattr(task, "task_id", ""),
            agent_type=agent_type,
            status="rate_limited",
            summary=f"Rate limit reached: {current_count}/{max_total} calls for {agent_type}",
            error_detail=f"max_total={max_total} exceeded for agent type '{agent_type}'",
        )
    return None


def _normalise_output(raw: Any, task: TaskNode) -> AgentOutput:
    """Convert a native agent output to AgentOutput (MF-EXE-06).

    If the raw output is already an AgentOutput, return it directly.
    Otherwise, extract common fields from the agent-specific model.

    Args:
        raw: The raw output from an agent function.
        task: The TaskNode that produced this output.

    Returns:
        Normalised AgentOutput.
    """
    if isinstance(raw, AgentOutput):
        return raw

    task_id = getattr(task, "task_id", getattr(raw, "task_id", ""))
    agent_type = getattr(task, "type", "")

    # Extract criteria dict from agent-specific to_criteria_dict()
    criteria: Dict[str, Any] = {}
    if hasattr(raw, "to_criteria_dict"):
        try:
            criteria = raw.to_criteria_dict()
        except Exception:
            pass

    evidence: List[EvidenceChunk] = getattr(raw, "evidence", [])
    summary: str = getattr(raw, "summary", str(raw)[:300])
    tokens_used: int = getattr(raw, "tokens_used", 0)
    cost_usd: float = getattr(raw, "cost_usd", 0.0)
    status: str = "success" if getattr(raw, "status", "success") not in ("failed", "timeout") else "failed"

    return AgentOutput(  # type: ignore[call-arg]
        task_id=task_id,
        agent_type=agent_type,
        status=status,
        summary=summary,
        evidence=evidence,
        criteria_result=criteria,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        raw_output=raw,
    )


def _build_state_update(
    output: AgentOutput,
    state: Dict[str, Any],
    include_budget: bool = True,
) -> Dict[str, Any]:
    """Build the state update dict after a single successful dispatch (MF-EXE-08/09).

    Writes:
        - last_task_result      ->  the normalised AgentOutput
        - evidence_board        ->  new chunks (merged by the reducer in MASISState)
        - token_budget          ->  updated budget
        - api_call_counts       ->  incremented call counter for this agent type
        - critique_notes        ->  Skeptic raw_output when agent_type == "skeptic"
        - synthesis_output      ->  Synthesizer raw_output when agent_type == "synthesizer"

    Args:
        output: The normalised AgentOutput.
        state: Current MASISState (for budget and call counts).

    Returns:
        Partial state update dict.
    """
    update: Dict[str, Any] = {
        "last_task_result": output,
        "batch_task_results": [output],
        "evidence_board": output.evidence,    # goes through evidence_reducer
    }

    if include_budget:
        # Update budget (MF-EXE-09)
        budget: BudgetTracker = state.get("token_budget", BudgetTracker())
        new_budget = budget.add(output.tokens_used, output.cost_usd, output.agent_type)

        # Update API call counts
        api_counts: Dict[str, int] = dict(state.get("api_call_counts", {}))
        api_counts[output.agent_type] = api_counts.get(output.agent_type, 0) + 1
        update["token_budget"] = new_budget
        update["api_call_counts"] = api_counts

    # Persist typed cross-node artifacts for downstream consumers.
    if output.agent_type == "skeptic":
        update["critique_notes"] = output.raw_output
    elif output.agent_type == "synthesizer":
        update["synthesis_output"] = (
            output.raw_output if output.raw_output is not None else output.summary
        )

    return update


async def _run_tasks_sequentially(
    tasks: List[TaskNode], state: Dict[str, Any]
) -> Dict[str, Any]:
    """Run tasks one by one and accumulate evidence.

    Each completed task is marked 'done' (or 'failed') in the task_dag so
    that downstream tasks (skeptic, synthesizer) see their dependencies met.

    Args:
        tasks: List of TaskNode objects to run.
        state: Current MASISState.

    Returns:
        State update with last_task_result from the last task, and merged evidence.
    """
    all_evidence: List[EvidenceChunk] = []
    last_output: Optional[AgentOutput] = None
    batch_outputs: List[AgentOutput] = []
    current_state = dict(state)
    dag: List[TaskNode] = current_state.get("task_dag", [])

    for task in tasks:
        output = await dispatch_with_safety(task, current_state)
        all_evidence.extend(output.evidence)
        last_output = output
        batch_outputs.append(output)

        # ── Mark this task done/failed in the DAG ─────────────────────────
        for dag_task in dag:
            if dag_task.task_id == task.task_id:
                dag_task.status = "done" if output.status == "success" else "failed"
                dag_task.result_summary = getattr(output, "summary", "")[:500]
                logger.debug(
                    "Marked DAG task %s as %s", task.task_id, dag_task.status,
                )
                break

        # Carry forward counters and typed artifacts from each dispatch.
        step_update = _build_state_update(output, current_state)
        current_state["token_budget"] = step_update.get(
            "token_budget", current_state.get("token_budget")
        )
        current_state["api_call_counts"] = step_update.get(
            "api_call_counts", current_state.get("api_call_counts", {})
        )
        if "critique_notes" in step_update:
            current_state["critique_notes"] = step_update["critique_notes"]
        if "synthesis_output" in step_update:
            current_state["synthesis_output"] = step_update["synthesis_output"]

        # Update running state so subsequent tasks see accumulated evidence
        current_state["evidence_board"] = (
            current_state.get("evidence_board", []) + output.evidence
        )

    if last_output is None:
        return {}

    update = {
        "last_task_result": last_output,
        "batch_task_results": batch_outputs,
        "parallel_batch_mode": False,
        "evidence_board": all_evidence,  # merged via reducer
        "token_budget": current_state.get("token_budget"),
        "api_call_counts": current_state.get("api_call_counts", {}),
        "task_dag": dag,  # persist the updated DAG with done statuses
    }
    if "critique_notes" in current_state:
        update["critique_notes"] = current_state.get("critique_notes")
    if "synthesis_output" in current_state:
        update["synthesis_output"] = current_state.get("synthesis_output")
    return update
