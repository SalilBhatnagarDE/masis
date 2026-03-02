"""
test_dag_utils.py
=================
Unit tests for masis.utils.dag_utils

Covers ENG-03 / M1-M2:
  - get_next_ready_tasks() — parallel group selection, dependency checking (MF-SUP-08)
  - all_tasks_done()       — terminal-status detection
  - all_tasks_succeeded()  — strict done-only check
  - find_task()            — lookup by task_id
  - check_agent_criteria() — Fast Path PASS/FAIL for all four agent types (MF-SUP-07)
  - dag_summary()          — structured state summary for Supervisor context
  - count_completed_by_type()

Run:
    pytest masis/tests/test_dag_utils.py -v
"""

from __future__ import annotations

import pytest

from masis.schemas.models import AgentOutput, TaskNode
from masis.utils.dag_utils import (
    all_tasks_done,
    all_tasks_succeeded,
    check_agent_criteria,
    count_completed_by_type,
    dag_summary,
    find_task,
    get_next_ready_tasks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(
    task_id: str = "T1",
    task_type: str = "researcher",
    deps: list | None = None,
    group: int = 1,
    status: str = "pending",
) -> TaskNode:
    return TaskNode(
        task_id=task_id,
        type=task_type,
        query=f"Query for {task_id}",
        dependencies=deps or [],
        parallel_group=group,
        acceptance_criteria=f"criteria for {task_id}",
        status=status,
    )


def make_result(
    task_id: str = "T1",
    agent_type: str = "researcher",
    status: str = "success",
    criteria: dict | None = None,
) -> AgentOutput:
    return AgentOutput(
        task_id=task_id,
        agent_type=agent_type,
        status=status,
        criteria_result=criteria or {},
    )


def researcher_pass_criteria() -> dict:
    return {
        "chunks_after_grading": 3,
        "grading_pass_rate": 0.60,
        "self_rag_verdict": "grounded",
        "source_diversity": 2,
    }


def researcher_fail_criteria_zero_chunks() -> dict:
    return {
        "chunks_after_grading": 0,
        "grading_pass_rate": 0.0,
        "self_rag_verdict": "not_grounded",
    }


def skeptic_pass_criteria() -> dict:
    return {
        "claims_unsupported": 0,
        "claims_contradicted": 0,
        "logical_gaps_count": 0,
        "overall_confidence": 0.80,
    }


def synthesizer_pass_criteria() -> dict:
    return {
        "citations_count": 3,
        "claims_count": 3,
        "all_citations_in_evidence_board": True,
    }


def web_search_pass_criteria() -> dict:
    return {
        "relevant_results": 3,
        "timeout": False,
    }


# ---------------------------------------------------------------------------
# get_next_ready_tasks()  (MF-SUP-08)
# ---------------------------------------------------------------------------

class TestGetNextReadyTasks:
    def test_empty_dag_returns_empty(self):
        assert get_next_ready_tasks([]) == []

    def test_single_pending_no_deps_returned(self):
        t1 = make_task("T1", deps=[])
        result = get_next_ready_tasks([t1])
        assert len(result) == 1
        assert result[0].task_id == "T1"

    def test_done_task_not_returned(self):
        t1 = make_task("T1", status="done")
        assert get_next_ready_tasks([t1]) == []

    def test_running_task_not_returned(self):
        t1 = make_task("T1", status="running")
        assert get_next_ready_tasks([t1]) == []

    def test_failed_task_not_returned(self):
        t1 = make_task("T1", status="failed")
        assert get_next_ready_tasks([t1]) == []

    def test_dep_satisfied_task_returned(self):
        """When T1=done, T2(deps=[T1]) becomes ready."""
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", deps=["T1"])
        result = get_next_ready_tasks([t1, t2])
        assert len(result) == 1
        assert result[0].task_id == "T2"

    def test_dep_unsatisfied_task_not_returned(self):
        """T2 depends on T1 which is not done, so T2 must NOT be returned.
        But T1 itself IS ready (no deps, pending), so it IS returned."""
        t1 = make_task("T1", status="pending")  # NOT done — but has no deps, so is ready
        t2 = make_task("T2", deps=["T1"])
        result = get_next_ready_tasks([t1, t2])
        result_ids = {t.task_id for t in result}
        assert "T2" not in result_ids, "T2's dep (T1) is not done, so T2 must not be ready"
        assert "T1" in result_ids, "T1 has no deps and is pending, so it IS ready"

    def test_partial_dep_satisfaction_blocks_task(self):
        """T3 depends on both T1 and T2. T1 is done but T2 is not, so T3 is blocked.
        However T2 itself (no deps, pending) IS ready."""
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", status="pending")
        t3 = make_task("T3", deps=["T1", "T2"])
        result = get_next_ready_tasks([t1, t2, t3])
        result_ids = {t.task_id for t in result}
        assert "T3" not in result_ids, "T3's deps not all done — must be blocked"
        assert "T2" in result_ids, "T2 has no deps and is pending, so it IS ready"

    def test_full_dep_satisfaction_unblocks_task(self):
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", status="done")
        t3 = make_task("T3", deps=["T1", "T2"])
        result = get_next_ready_tasks([t1, t2, t3])
        assert len(result) == 1
        assert result[0].task_id == "T3"

    def test_returns_only_lowest_parallel_group(self):
        """Tasks in group 2 should not be returned while group 1 has pending tasks."""
        t1 = make_task("T1", group=1)
        t2 = make_task("T2", group=1)
        t3 = make_task("T3", group=2)
        result = get_next_ready_tasks([t1, t2, t3])
        ids = {t.task_id for t in result}
        assert "T1" in ids
        assert "T2" in ids
        assert "T3" not in ids

    def test_parallel_tasks_in_same_group_all_returned(self):
        """Multiple tasks with the same group number and satisfied deps all returned."""
        t1 = make_task("T1", group=1)
        t2 = make_task("T2", group=1)
        t3 = make_task("T3", group=1)
        result = get_next_ready_tasks([t1, t2, t3])
        assert len(result) == 3

    def test_group2_returned_when_group1_all_done(self):
        t1 = make_task("T1", group=1, status="done")
        t2 = make_task("T2", group=2)
        result = get_next_ready_tasks([t1, t2])
        assert len(result) == 1
        assert result[0].task_id == "T2"

    def test_engineering_tasks_spec_example(self):
        """ENG-03 M1 S1 test: T1=done, T2(deps=[T1]) is returned."""
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", deps=["T1"])
        result = get_next_ready_tasks([t1, t2])
        assert result == [t2]

    def test_all_tasks_terminal_returns_empty(self):
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", status="failed")
        assert get_next_ready_tasks([t1, t2]) == []


# ---------------------------------------------------------------------------
# all_tasks_done()
# ---------------------------------------------------------------------------

class TestAllTasksDone:
    def test_empty_dag_is_done(self):
        assert all_tasks_done([]) is True

    def test_all_done_is_true(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="done")]
        assert all_tasks_done(dag) is True

    def test_all_failed_is_also_terminal(self):
        """Failed tasks count as terminal — the DAG is 'done' in a terminal sense."""
        dag = [make_task("T1", status="failed"), make_task("T2", status="failed")]
        assert all_tasks_done(dag) is True

    def test_mixed_done_and_failed_is_terminal(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="failed")]
        assert all_tasks_done(dag) is True

    def test_one_pending_is_not_done(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="pending")]
        assert all_tasks_done(dag) is False

    def test_one_running_is_not_done(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="running")]
        assert all_tasks_done(dag) is False


# ---------------------------------------------------------------------------
# all_tasks_succeeded()
# ---------------------------------------------------------------------------

class TestAllTasksSucceeded:
    def test_all_done_is_true(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="done")]
        assert all_tasks_succeeded(dag) is True

    def test_failed_task_makes_it_false(self):
        dag = [make_task("T1", status="done"), make_task("T2", status="failed")]
        assert all_tasks_succeeded(dag) is False

    def test_pending_makes_it_false(self):
        dag = [make_task("T1", status="pending")]
        assert all_tasks_succeeded(dag) is False


# ---------------------------------------------------------------------------
# find_task()
# ---------------------------------------------------------------------------

class TestFindTask:
    def test_finds_existing_task(self):
        t1 = make_task("T1")
        t2 = make_task("T2")
        result = find_task([t1, t2], "T1")
        assert result is t1

    def test_returns_none_for_missing_id(self):
        t1 = make_task("T1")
        assert find_task([t1], "T99") is None

    def test_returns_none_on_empty_dag(self):
        assert find_task([], "T1") is None

    def test_finds_task_at_end_of_list(self):
        tasks = [make_task(f"T{i}") for i in range(1, 6)]
        result = find_task(tasks, "T5")
        assert result is not None
        assert result.task_id == "T5"


# ---------------------------------------------------------------------------
# check_agent_criteria()  (MF-SUP-07)
# ---------------------------------------------------------------------------

class TestCheckAgentCriteriaResearcher:
    """ENG-03 M2 S1: Researcher criteria checks."""

    def test_researcher_pass(self):
        """ENG-03 done-when: researcher with 3 chunks, 0.6 pass_rate, grounded → PASS."""
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria=researcher_pass_criteria())
        assert check_agent_criteria(t, result) == "PASS"

    def test_researcher_fail_zero_chunks(self):
        """ENG-03 test: researcher with 0 chunks → FAIL."""
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria=researcher_fail_criteria_zero_chunks())
        assert check_agent_criteria(t, result) == "FAIL"

    def test_researcher_fail_low_pass_rate(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria={
            "chunks_after_grading": 2,
            "grading_pass_rate": 0.10,  # below 0.30 threshold
            "self_rag_verdict": "grounded",
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_researcher_fail_not_grounded(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria={
            "chunks_after_grading": 3,
            "grading_pass_rate": 0.60,
            "self_rag_verdict": "not_grounded",  # fails
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_researcher_fail_partial_verdict(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria={
            "chunks_after_grading": 3,
            "grading_pass_rate": 0.60,
            "self_rag_verdict": "partial",  # not "grounded"
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_researcher_fail_exactly_at_chunk_boundary(self):
        """Exactly 2 chunks should PASS (>= 2)."""
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria={
            "chunks_after_grading": 2,  # exactly at threshold
            "grading_pass_rate": 0.30,  # exactly at threshold
            "self_rag_verdict": "grounded",
        })
        assert check_agent_criteria(t, result) == "PASS"

    def test_researcher_fail_one_chunk_below_threshold(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", criteria={
            "chunks_after_grading": 1,  # below minimum of 2
            "grading_pass_rate": 0.50,
            "self_rag_verdict": "grounded",
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_agent_reported_failed_status_is_auto_fail(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", status="failed",
                             criteria=researcher_pass_criteria())
        assert check_agent_criteria(t, result) == "FAIL"

    def test_timeout_status_is_auto_fail(self):
        t = make_task("T1", task_type="researcher")
        result = make_result("T1", "researcher", status="timeout",
                             criteria=researcher_pass_criteria())
        assert check_agent_criteria(t, result) == "FAIL"


class TestCheckAgentCriteriaSkeptic:
    """Skeptic criteria checks."""

    def test_skeptic_pass(self):
        t = make_task("T2", task_type="skeptic")
        result = make_result("T2", "skeptic", criteria=skeptic_pass_criteria())
        assert check_agent_criteria(t, result) == "PASS"

    def test_skeptic_fail_contradictions(self):
        t = make_task("T2", task_type="skeptic")
        result = make_result("T2", "skeptic", criteria={
            "claims_unsupported": 0,
            "claims_contradicted": 1,  # non-zero → FAIL
            "logical_gaps_count": 0,
            "overall_confidence": 0.80,
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_skeptic_fail_low_confidence(self):
        t = make_task("T2", task_type="skeptic")
        result = make_result("T2", "skeptic", criteria={
            "claims_unsupported": 0,
            "claims_contradicted": 0,
            "logical_gaps_count": 0,
            "overall_confidence": 0.50,  # below 0.65 threshold
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_skeptic_fail_too_many_gaps(self):
        t = make_task("T2", task_type="skeptic")
        result = make_result("T2", "skeptic", criteria={
            "claims_unsupported": 0,
            "claims_contradicted": 0,
            "logical_gaps_count": 4,  # above max_logical_gaps=3
            "overall_confidence": 0.80,
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_skeptic_pass_at_exact_thresholds(self):
        """Exactly at all thresholds should PASS."""
        t = make_task("T2", task_type="skeptic")
        result = make_result("T2", "skeptic", criteria={
            "claims_unsupported": 2,   # max_unsupported=2, so == is allowed
            "claims_contradicted": 0,
            "logical_gaps_count": 3,   # max_logical_gaps=3, so == is allowed
            "overall_confidence": 0.65,  # exactly at min_confidence
        })
        assert check_agent_criteria(t, result) == "PASS"


class TestCheckAgentCriteriaSynthesizer:
    """Synthesizer criteria checks."""

    def test_synthesizer_pass(self):
        t = make_task("T3", task_type="synthesizer")
        result = make_result("T3", "synthesizer", criteria=synthesizer_pass_criteria())
        assert check_agent_criteria(t, result) == "PASS"

    def test_synthesizer_pass_when_citations_present_even_if_fewer_than_claims(self):
        t = make_task("T3", task_type="synthesizer")
        result = make_result("T3", "synthesizer", criteria={
            "citations_count": 2,
            "claims_count": 3,
            "all_citations_in_evidence_board": True,
        })
        assert check_agent_criteria(t, result) == "PASS"

    def test_synthesizer_fail_when_no_citations(self):
        t = make_task("T3", task_type="synthesizer")
        result = make_result("T3", "synthesizer", criteria={
            "citations_count": 0,
            "claims_count": 3,
            "all_citations_in_evidence_board": True,
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_synthesizer_fail_citations_not_in_board(self):
        t = make_task("T3", task_type="synthesizer")
        result = make_result("T3", "synthesizer", criteria={
            "citations_count": 3,
            "claims_count": 3,
            "all_citations_in_evidence_board": False,  # FAIL
        })
        assert check_agent_criteria(t, result) == "FAIL"


class TestCheckAgentCriteriaWebSearch:
    """Web search criteria checks."""

    def test_web_search_pass(self):
        t = make_task("T2", task_type="web_search")
        result = make_result("T2", "web_search", criteria=web_search_pass_criteria())
        assert check_agent_criteria(t, result) == "PASS"

    def test_web_search_fail_no_results(self):
        t = make_task("T2", task_type="web_search")
        result = make_result("T2", "web_search", criteria={
            "relevant_results": 0,
            "timeout": False,
        })
        assert check_agent_criteria(t, result) == "FAIL"

    def test_web_search_fail_timeout(self):
        t = make_task("T2", task_type="web_search")
        result = make_result("T2", "web_search", criteria={
            "relevant_results": 3,
            "timeout": True,  # timeout → FAIL
        })
        assert check_agent_criteria(t, result) == "FAIL"


# ---------------------------------------------------------------------------
# dag_summary()
# ---------------------------------------------------------------------------

class TestDagSummary:
    def test_counts_statuses_correctly(self):
        dag = [
            make_task("T1", status="done"),
            make_task("T2", status="done"),
            make_task("T3", status="running"),
            make_task("T4", status="pending"),
            make_task("T5", status="failed"),
        ]
        summary = dag_summary(dag)
        assert summary["total_tasks"] == 5
        assert summary["done"] == 2
        assert summary["running"] == 1
        assert summary["pending"] == 1
        assert summary["failed"] == 1

    def test_pending_tasks_listed(self):
        t1 = make_task("T1", status="pending")
        summary = dag_summary([t1])
        pending = summary["pending_tasks"]
        assert len(pending) == 1
        assert pending[0]["task_id"] == "T1"

    def test_failed_tasks_listed(self):
        t1 = make_task("T1", status="failed")
        summary = dag_summary([t1])
        failed = summary["failed_tasks"]
        assert len(failed) == 1
        assert failed[0]["task_id"] == "T1"

    def test_empty_dag_summary(self):
        summary = dag_summary([])
        assert summary["total_tasks"] == 0
        assert summary["done"] == 0


# ---------------------------------------------------------------------------
# count_completed_by_type()
# ---------------------------------------------------------------------------

class TestCountCompletedByType:
    def test_counts_by_type(self):
        dag = [
            make_task("T1", task_type="researcher", status="done"),
            make_task("T2", task_type="researcher", status="done"),
            make_task("T3", task_type="skeptic", status="done"),
            make_task("T4", task_type="synthesizer", status="pending"),
        ]
        counts = count_completed_by_type(dag)
        assert counts["researcher"] == 2
        assert counts["skeptic"] == 1
        assert "synthesizer" not in counts  # pending, not done

    def test_empty_dag_returns_empty(self):
        assert count_completed_by_type([]) == {}

