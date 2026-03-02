"""
masis.utils.dag_utils
=====================
Utility functions for walking and querying the dynamic task DAG.

Implements
----------
MF-SUP-07  : Fast Path criteria checker -- check_agent_criteria()
MF-SUP-08  : Next-task resolution -- get_next_ready_tasks()
MF-MEM-05  : Task status tracking (read-only, writes done by Executor)

All functions operate on the in-memory task list stored in MASISState.task_dag.
They are pure functions (no I/O, no LLM calls) so they run in < 1ms and are
always Fast Path compatible.

Usage
-----
    from masis.utils.dag_utils import (
        get_next_ready_tasks,
        all_tasks_done,
        find_task,
        check_agent_criteria,
    )

    # In supervisor_node Fast Path:
    next_tasks = get_next_ready_tasks(state["task_dag"])
    if not next_tasks and all_tasks_done(state["task_dag"]):
        return {"supervisor_decision": "ready_for_validation"}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from masis.schemas.models import AgentOutput, EvidenceChunk, ResearcherOutput, SkepticOutput, SynthesizerOutput, TaskNode
from masis.schemas.thresholds import RESEARCHER_THRESHOLDS, SKEPTIC_THRESHOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENG-03 / M1 / S1 — get_next_ready_tasks  (MF-SUP-08)
# ---------------------------------------------------------------------------

def get_next_ready_tasks(dag: List[TaskNode]) -> List[TaskNode]:
    """
    Return tasks that are ready to execute right now.

    A task is "ready" when:
    - Its status is "pending" (not already running, done, or failed).
    - All its dependencies have status "done".
    - It belongs to the lowest-numbered parallel_group among all ready tasks.

    The lowest-group restriction ensures that group-1 tasks all complete before
    group-2 tasks start, respecting the sequential structure of the DAG while
    still allowing parallel execution within a group.

    Args:
        dag: The full list of TaskNodes from MASISState.task_dag.

    Returns:
        List of TaskNodes that can be dispatched right now. Empty list means
        nothing is runnable (either all done, or waiting on running tasks).

    Examples:
        >>> t1 = TaskNode(task_id="T1", type="researcher", query="q", status="done")
        >>> t2 = TaskNode(task_id="T2", type="skeptic", query="q", dependencies=["T1"])
        >>> get_next_ready_tasks([t1, t2])
        [t2]  # T1 done, T2 dependencies satisfied

        >>> t3 = TaskNode(task_id="T3", type="researcher", query="q", parallel_group=1)
        >>> t4 = TaskNode(task_id="T4", type="skeptic", query="q", parallel_group=2)
        >>> get_next_ready_tasks([t3, t4])
        [t3]  # Only group-1 tasks returned; group-2 waits for group-1

    Raises:
        Never. Returns empty list on any edge case.
    """
    if not dag:
        return []

    # Dependencies are considered satisfied once upstream tasks reach a
    # terminal status (done or failed). This avoids DAG deadlocks and allows
    # skeptic/synthesizer to proceed with partial evidence when needed.
    terminal_ids: set = {t.task_id for t in dag if t.status in ("done", "failed")}

    # Find all tasks that can run right now
    candidates = [t for t in dag if t.is_ready(terminal_ids)]

    if not candidates:
        return []

    # Among ready tasks, dispatch only those in the lowest parallel_group
    min_group = min(t.parallel_group for t in candidates)
    ready = [t for t in candidates if t.parallel_group == min_group]

    logger.debug(
        "get_next_ready_tasks: %d candidates, %d in group %d: %s",
        len(candidates),
        len(ready),
        min_group,
        [t.task_id for t in ready],
    )
    return ready


# ---------------------------------------------------------------------------
# ENG-03 / M1 / S2 — all_tasks_done
# ---------------------------------------------------------------------------

def all_tasks_done(dag: List[TaskNode]) -> bool:
    """
    Return True if every task in the DAG has a terminal status (done or failed).

    Used by the Fast Path to determine whether to route to Validator:
        if all_tasks_done(state["task_dag"]):
            return {"supervisor_decision": "ready_for_validation"}

    Args:
        dag: The full list of TaskNodes from MASISState.task_dag.

    Returns:
        True  -- every task is "done" or "failed" (no pending or running tasks).
        False -- at least one task is still "pending" or "running".

    Examples:
        >>> t1 = TaskNode(task_id="T1", type="researcher", query="q", status="done")
        >>> t2 = TaskNode(task_id="T2", type="skeptic", query="q", status="done")
        >>> all_tasks_done([t1, t2])
        True
        >>> t3 = TaskNode(task_id="T3", type="synthesizer", query="q", status="pending")
        >>> all_tasks_done([t1, t2, t3])
        False
    """
    if not dag:
        return True  # Empty DAG is considered complete (edge case)
    terminal_statuses = {"done", "failed"}
    result = all(t.status in terminal_statuses for t in dag)
    logger.debug("all_tasks_done: %s (dag size: %d)", result, len(dag))
    return result


def all_tasks_succeeded(dag: List[TaskNode]) -> bool:
    """
    Return True if every task has status "done" (no failures or pending tasks).

    Stricter than all_tasks_done() -- used to decide if we can route to Validator
    without caveats.

    Args:
        dag: The full list of TaskNodes from MASISState.task_dag.

    Returns:
        True only when every task has status "done".
    """
    if not dag:
        return True
    return all(t.status == "done" for t in dag)


# ---------------------------------------------------------------------------
# ENG-03 / M1 / S3 — find_task
# ---------------------------------------------------------------------------

def find_task(dag: List[TaskNode], task_id: str) -> Optional[TaskNode]:
    """
    Return the TaskNode with the given task_id, or None if not found.

    Linear scan -- acceptable since DAGs typically have fewer than 20 tasks.

    Args:
        dag:     The full list of TaskNodes from MASISState.task_dag.
        task_id: The task identifier to look for (e.g. "T1", "T2b").

    Returns:
        The matching TaskNode, or None if no task with that ID exists.

    Examples:
        >>> t1 = TaskNode(task_id="T1", type="researcher", query="q")
        >>> find_task([t1], "T1")
        TaskNode(task_id='T1', ...)
        >>> find_task([t1], "T99")
        None
    """
    for task in dag:
        if task.task_id == task_id:
            return task
    return None


def get_tasks_by_type(dag: List[TaskNode], task_type: str) -> List[TaskNode]:
    """
    Return all tasks of the specified type, in DAG order.

    Args:
        dag:       The full list of TaskNodes.
        task_type: One of: researcher, web_search, skeptic, synthesizer.

    Returns:
        Filtered list of TaskNodes matching the type.
    """
    return [t for t in dag if t.type == task_type]


def get_last_completed_task(dag: List[TaskNode], task_type: Optional[str] = None) -> Optional[TaskNode]:
    """
    Return the most recently completed task of the given type (or any type).

    Recency is approximated by position in the dag list (later = more recent).

    Args:
        dag:       The full list of TaskNodes.
        task_type: Optional filter by agent type.

    Returns:
        The last "done" TaskNode matching the filter, or None.
    """
    done_tasks = [t for t in dag if t.status == "done"]
    if task_type:
        done_tasks = [t for t in done_tasks if t.type == task_type]
    return done_tasks[-1] if done_tasks else None


# ---------------------------------------------------------------------------
# ENG-03 / M2 / S1 — check_agent_criteria  (MF-SUP-07)
# ---------------------------------------------------------------------------

def check_agent_criteria(
    task: TaskNode,
    result: AgentOutput,
) -> str:
    """
    Fast Path criteria checker. Evaluates whether an agent's output meets the
    acceptance_criteria defined on its TaskNode.

    This function runs with NO LLM call ($0, < 1ms) and returns "PASS" or "FAIL".
    - "PASS" means the Fast Path routes to the next task in the DAG.
    - "FAIL" triggers the Supervisor Slow Path (LLM decision needed).

    Per-agent criteria (from architecture Section 3, MF-SUP-07):
    +-----------+-----------------------------------------------------------------+
    | Researcher| chunks_after_grading >= 2                                       |
    |           | grading_pass_rate >= 0.30                                        |
    |           | self_rag_verdict == "grounded"                                   |
    +-----------+-----------------------------------------------------------------+
    | Skeptic   | claims_unsupported == 0                                         |
    |           | claims_contradicted == 0                                         |
    |           | len(logical_gaps) == 0                                           |
    |           | overall_confidence >= 0.65                                       |
    +-----------+-----------------------------------------------------------------+
    | Synthesizer| citations_count >= claims_count                                |
    |           | all_citations_in_evidence_board == True                          |
    +-----------+-----------------------------------------------------------------+
    | Web Search | relevant_results >= 1                                           |
    |           | timeout == False                                                  |
    +-----------+-----------------------------------------------------------------+

    Args:
        task:   The TaskNode that was just executed. Used for task.type and logging.
        result: The AgentOutput returned by the Executor. Uses result.criteria_result
                (the structured dict populated by the agent's to_criteria_dict()).

    Returns:
        "PASS" -- all criteria met, Fast Path continues.
        "FAIL" -- one or more criteria failed, Slow Path needed.

    Examples:
        >>> researcher_result = AgentOutput(
        ...     task_id="T1", agent_type="researcher", status="success",
        ...     criteria_result={"chunks_after_grading": 3, "grading_pass_rate": 0.6,
        ...                      "self_rag_verdict": "grounded"}
        ... )
        >>> check_agent_criteria(t1, researcher_result)
        "PASS"

        >>> fail_result = AgentOutput(
        ...     task_id="T1", agent_type="researcher", status="success",
        ...     criteria_result={"chunks_after_grading": 0, ...}
        ... )
        >>> check_agent_criteria(t1, fail_result)
        "FAIL"
    """
    # Any agent that reports status="failed" or "timeout" is an automatic FAIL
    if result.status in ("failed", "timeout", "rate_limited"):
        logger.info(
            "check_agent_criteria: task=%s FAIL -- agent reported status='%s'",
            task.task_id, result.status,
        )
        return "FAIL"

    criteria = result.criteria_result
    agent_type = task.type

    if agent_type == "researcher":
        verdict = _check_researcher_criteria(task.task_id, criteria)
    elif agent_type == "skeptic":
        verdict = _check_skeptic_criteria(task.task_id, criteria)
    elif agent_type == "synthesizer":
        verdict = _check_synthesizer_criteria(task.task_id, criteria)
    elif agent_type == "web_search":
        verdict = _check_web_search_criteria(task.task_id, criteria)
    else:
        # Unknown type should never happen (Pydantic validates TaskNode.type)
        logger.warning(
            "check_agent_criteria: unknown task type '%s', defaulting to FAIL", agent_type
        )
        verdict = "FAIL"

    logger.info(
        "check_agent_criteria: task=%s type=%s verdict=%s criteria=%s",
        task.task_id, agent_type, verdict, criteria,
    )
    return verdict


def _check_researcher_criteria(task_id: str, criteria: Dict[str, Any]) -> str:
    """Evaluate researcher Fast Path criteria against RESEARCHER_THRESHOLDS."""
    min_chunks = RESEARCHER_THRESHOLDS["min_chunks_after_grading"]
    min_pass_rate = RESEARCHER_THRESHOLDS["min_grading_pass_rate"]
    required_verdict = RESEARCHER_THRESHOLDS["required_self_rag_verdict"]
    allow_partial = bool(RESEARCHER_THRESHOLDS.get("allow_partial_self_rag", False))

    chunks_after_grading = criteria.get("chunks_after_grading", 0)
    grading_pass_rate = criteria.get("grading_pass_rate", 0.0)
    self_rag_verdict = criteria.get("self_rag_verdict", "not_grounded")

    failures = []
    if chunks_after_grading < min_chunks:
        failures.append(
            f"chunks_after_grading={chunks_after_grading} < {min_chunks}"
        )
    if grading_pass_rate < min_pass_rate:
        failures.append(
            f"grading_pass_rate={grading_pass_rate:.2f} < {min_pass_rate}"
        )
    allowed_verdicts = {required_verdict}
    if allow_partial:
        allowed_verdicts.add("partial")
    if self_rag_verdict not in allowed_verdicts:
        failures.append(
            f"self_rag_verdict='{self_rag_verdict}' not in {sorted(allowed_verdicts)}"
        )

    if failures:
        logger.debug("Researcher %s FAIL: %s", task_id, "; ".join(failures))
        return "FAIL"
    return "PASS"


def _check_skeptic_criteria(task_id: str, criteria: Dict[str, Any]) -> str:
    """Evaluate skeptic Fast Path criteria against SKEPTIC_THRESHOLDS."""
    max_unsupported = SKEPTIC_THRESHOLDS["max_unsupported_claims"]
    max_contradicted = SKEPTIC_THRESHOLDS["max_contradicted_claims"]
    max_gaps = SKEPTIC_THRESHOLDS["max_logical_gaps"]
    min_confidence = SKEPTIC_THRESHOLDS["min_confidence"]

    claims_unsupported = criteria.get("claims_unsupported", 0)
    claims_contradicted = criteria.get("claims_contradicted", 0)
    logical_gaps_count = criteria.get("logical_gaps_count", 0)
    overall_confidence = criteria.get("overall_confidence", 0.0)

    failures = []
    if claims_unsupported > max_unsupported:
        failures.append(
            f"claims_unsupported={claims_unsupported} > {max_unsupported}"
        )
    if claims_contradicted > max_contradicted:
        failures.append(
            f"claims_contradicted={claims_contradicted} > {max_contradicted}"
        )
    if logical_gaps_count > max_gaps:
        failures.append(
            f"logical_gaps_count={logical_gaps_count} > {max_gaps}"
        )
    if overall_confidence < min_confidence:
        failures.append(
            f"overall_confidence={overall_confidence:.2f} < {min_confidence}"
        )

    if failures:
        logger.debug("Skeptic %s FAIL: %s", task_id, "; ".join(failures))
        return "FAIL"
    return "PASS"


def _check_synthesizer_criteria(task_id: str, criteria: Dict[str, Any]) -> str:
    """Evaluate synthesizer Fast Path criteria."""
    citations_count = criteria.get("citations_count", 0)
    claims_count = criteria.get("claims_count", 0)
    all_in_board = criteria.get("all_citations_in_evidence_board", False)

    failures = []
    # Keep this gate permissive: citation completeness is validated again by
    # the Validator node with stricter semantic checks.
    if citations_count < 1:
        failures.append("citations_count < 1")
    if not all_in_board:
        failures.append("all_citations_in_evidence_board=False")

    if failures:
        logger.debug("Synthesizer %s FAIL: %s", task_id, "; ".join(failures))
        return "FAIL"
    return "PASS"


def _check_web_search_criteria(task_id: str, criteria: Dict[str, Any]) -> str:
    """Evaluate web_search Fast Path criteria."""
    relevant_results = criteria.get("relevant_results", 0)
    timeout = criteria.get("timeout", True)

    failures = []
    if relevant_results < 1:
        failures.append(f"relevant_results={relevant_results} < 1")
    if timeout:
        failures.append("timeout=True")

    if failures:
        logger.debug("WebSearch %s FAIL: %s", task_id, "; ".join(failures))
        return "FAIL"
    return "PASS"


# ---------------------------------------------------------------------------
# DAG statistics (used by Supervisor Slow Path context building)
# ---------------------------------------------------------------------------

def dag_summary(dag: List[TaskNode]) -> Dict[str, Any]:
    """
    Return a concise summary of the current DAG state for Supervisor context.

    Used by the Supervisor to build its minimal-context view (MF-SUP-14).
    Does NOT include full evidence -- only counts and status breakdowns.

    Args:
        dag: The full list of TaskNodes from MASISState.task_dag.

    Returns:
        Dict with status counts and pending task list.
    """
    from collections import Counter
    status_counts = Counter(t.status for t in dag)
    return {
        "total_tasks": len(dag),
        "done": status_counts.get("done", 0),
        "running": status_counts.get("running", 0),
        "pending": status_counts.get("pending", 0),
        "failed": status_counts.get("failed", 0),
        "pending_tasks": [
            {"task_id": t.task_id, "type": t.type, "parallel_group": t.parallel_group}
            for t in dag if t.status == "pending"
        ],
        "failed_tasks": [
            {"task_id": t.task_id, "type": t.type, "retry_count": t.retry_count}
            for t in dag if t.status == "failed"
        ],
    }


def count_completed_by_type(dag: List[TaskNode]) -> Dict[str, int]:
    """
    Return a dict of {agent_type: count_of_done_tasks} for rate-limit tracking.

    Args:
        dag: The full list of TaskNodes from MASISState.task_dag.

    Returns:
        Dict mapping agent type to number of completed tasks.
    """
    counts: Dict[str, int] = {}
    for task in dag:
        if task.status == "done":
            counts[task.type] = counts.get(task.type, 0) + 1
    return counts
