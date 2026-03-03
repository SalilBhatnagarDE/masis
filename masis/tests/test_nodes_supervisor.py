"""
Tests for masis.nodes.supervisor — ENG-04 (MF-SUP-01 through MF-SUP-17).

Coverage
--------
- plan_dag()           : first-turn DAG planning dispatches correctly (M2)
- monitor_and_route()  : Fast Path — budget, iterations, wall clock, repetition,
                          criteria pass/fail, all-done routing (M3)
- supervisor_slow_path(): action handling — retry, modify_dag, escalate,
                          force_synthesize, stop (M4)
- build_supervisor_context(): no evidence_board key (M5)
- decision_log entries: mode, action, cost, latency (M6 MF-SUP-17)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import under test (with try/except for isolated test environments)
# ---------------------------------------------------------------------------

try:
    from masis.schemas.models import (
        AgentOutput,
        BudgetTracker,
        EvidenceChunk,
        ModifyDagSpec,
        RetrySpec,
        SupervisorDecision,
        TaskNode,
        TaskPlan,
    )
    from masis.nodes.supervisor import (
        monitor_and_route,
        plan_dag,
        supervisor_node,
        supervisor_slow_path,
        _build_supervisor_context,
        _fast_decision,
    )
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason=f"Import failed: {locals().get('_IMPORT_ERROR', '')}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_task(task_id: str, ttype: str = "researcher", status: str = "pending", deps: List[str] = None) -> TaskNode:
    """Helper to create a TaskNode with sensible defaults."""
    return TaskNode(
        task_id=task_id,
        type=ttype,
        query=f"Query for {task_id}",
        dependencies=deps or [],
        parallel_group=1,
        acceptance_criteria="≥2 chunks, pass_rate≥0.30, self_rag=grounded",
        status=status,
    )


def make_budget(remaining: int = 100_000, cost: float = 0.0) -> BudgetTracker:
    return BudgetTracker(
        total_tokens_used=max(0, 100_000 - remaining),
        total_cost_usd=cost,
        remaining=remaining,
        start_time=time.time(),
    )


def make_agent_output(task_id: str, status: str = "success", pass_rate: float = 0.6, chunks: int = 3) -> AgentOutput:
    return AgentOutput(
        task_id=task_id,
        agent_type="researcher",
        status=status,
        summary="Test summary",
        evidence=[],
        criteria_result={
            "chunks_after_grading": chunks,
            "grading_pass_rate": pass_rate,
            "self_rag_verdict": "grounded",
        },
    )


def base_state(**kwargs: Any) -> Dict[str, Any]:
    """Return a minimal valid MASISState dict."""
    defaults = {
        "original_query": "What was Q3 revenue?",
        "query_id": "test-query-001",
        "task_dag": [],
        "iteration_count": 1,
        "next_tasks": [],
        "supervisor_decision": "",
        "last_task_result": None,
        "evidence_board": [],
        "critique_notes": None,
        "synthesis_output": None,
        "token_budget": make_budget(),
        "api_call_counts": {},
        "quality_scores": {},
        "start_time": time.time(),
        "validation_round": 0,
        "decision_log": [],
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Tests for plan_dag() — MF-SUP-01, MF-SUP-02, MF-SUP-03
# ---------------------------------------------------------------------------

class TestPlanDag:
    """Tests for first-turn DAG planning (MODE 1)."""

    @pytest.mark.asyncio
    async def test_plan_dag_raises_without_openai(self):
        """plan_dag() raises RuntimeError when LangChain/OpenAI is unavailable."""
        state = base_state(iteration_count=0)
        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", False):
            with patch("masis.nodes.supervisor.ChatOpenAI", None):
                with pytest.raises(RuntimeError, match="langchain_openai"):
                    await plan_dag(state)

    @pytest.mark.asyncio
    async def test_plan_dag_calls_structured_output(self):
        """plan_dag() calls with_structured_output(TaskPlan) on the LLM."""
        mock_tasks = [
            make_task("T1"),
            make_task("T2", ttype="skeptic", deps=["T1"], status="pending"),
            make_task("T3", ttype="synthesizer", deps=["T2"], status="pending"),
        ]
        mock_plan = TaskPlan(tasks=mock_tasks, stop_condition="Revenue figure found")

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_plan)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        state = base_state(iteration_count=0)

        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await plan_dag(state)

        assert result["supervisor_decision"] == "continue"
        assert result["iteration_count"] == 1
        assert len(result["task_dag"]) == 3
        assert "decision_log" in result
        assert len(result["decision_log"]) == 1
        assert result["decision_log"][0]["mode"] == "slow"

    @pytest.mark.asyncio
    async def test_plan_dag_returns_next_ready_tasks(self):
        """plan_dag() returns only the first batch of ready tasks."""
        mock_tasks = [
            make_task("T1"),
            make_task("T2"),
            make_task("T3", ttype="skeptic", deps=["T1", "T2"]),
            make_task("T4", ttype="synthesizer", deps=["T3"]),
        ]
        mock_plan = TaskPlan(tasks=mock_tasks, stop_condition="Done")

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_plan)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        state = base_state(iteration_count=0)
        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await plan_dag(state)

        # Only T1 and T2 (group 1, no dependencies) should be in next_tasks
        next_task_ids = {t.task_id for t in result["next_tasks"]}
        assert "T1" in next_task_ids
        assert "T2" in next_task_ids
        assert "T3" not in next_task_ids


# ---------------------------------------------------------------------------
# Tests for monitor_and_route() — MF-SUP-04 through MF-SUP-08
# ---------------------------------------------------------------------------

class TestMonitorAndRoute:
    """Tests for the Fast Path decision engine."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_forces_synthesize(self):
        """MF-SUP-04: budget.remaining <= 0 → force_synthesize without LLM call."""
        state = base_state(
            iteration_count=2,
            token_budget=make_budget(remaining=0),
        )
        result = await monitor_and_route(state)
        assert result["supervisor_decision"] == "force_synthesize"
        # Must not have called any LLM (fast path)
        log = result.get("decision_log", [])
        assert any(e.get("mode") == "fast" for e in log)

    @pytest.mark.asyncio
    async def test_iteration_limit_forces_synthesize(self):
        """MF-SUP-05: iteration_count >= 15 → force_synthesize."""
        state = base_state(iteration_count=15)
        result = await monitor_and_route(state)
        assert result["supervisor_decision"] == "force_synthesize"

    @pytest.mark.asyncio
    async def test_wall_clock_exceeded_forces_synthesize(self):
        """MF-SUP-16: wall clock > 300s → force_synthesize."""
        budget = BudgetTracker(
            remaining=99_000,
            start_time=time.time() - 400,  # 400s ago — exceeds 300s limit
        )
        state = base_state(iteration_count=2, token_budget=budget)
        result = await monitor_and_route(state)
        assert result["supervisor_decision"] == "force_synthesize"

    @pytest.mark.asyncio
    async def test_good_result_passes_to_next_task(self):
        """MF-SUP-07: passing criteria → continue with next tasks."""
        t1 = make_task("T1", status="running")
        t2 = make_task("T2", ttype="skeptic", deps=["T1"], status="pending")
        t3 = make_task("T3", ttype="synthesizer", deps=["T2"], status="pending")

        last_result = make_agent_output("T1", status="success", pass_rate=0.6, chunks=3)

        state = base_state(
            iteration_count=2,
            task_dag=[t1, t2, t3],
            last_task_result=last_result,
        )

        with patch("masis.nodes.supervisor.check_agent_criteria", return_value="PASS"):
            result = await monitor_and_route(state)

        assert result["supervisor_decision"] == "continue"

    @pytest.mark.asyncio
    async def test_all_tasks_done_routes_to_validation(self):
        """MF-SUP-08: all tasks done → ready_for_validation."""
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", ttype="skeptic", status="done")
        t3 = make_task("T3", ttype="synthesizer", status="done")

        state = base_state(
            iteration_count=4,
            task_dag=[t1, t2, t3],
            last_task_result=None,
        )

        with patch("masis.nodes.supervisor.check_agent_criteria", return_value="PASS"):
            result = await monitor_and_route(state)

        assert result["supervisor_decision"] == "ready_for_validation"

    @pytest.mark.asyncio
    async def test_failed_result_routes_to_slow_path(self):
        """Criteria FAIL → monitor_and_route calls supervisor_slow_path."""
        t1 = make_task("T1", status="running")
        last_result = make_agent_output("T1", status="failed")

        state = base_state(
            iteration_count=2,
            task_dag=[t1],
            last_task_result=last_result,
        )

        # Mock slow path to return a known result
        expected_slow = {"supervisor_decision": "force_synthesize", "iteration_count": 3, "decision_log": []}
        with patch("masis.nodes.supervisor.supervisor_slow_path", AsyncMock(return_value=expected_slow)):
            result = await monitor_and_route(state)

        assert result["supervisor_decision"] == "force_synthesize"

    @pytest.mark.asyncio
    async def test_repetition_detection_forces_synthesize(self):
        """MF-SUP-06: is_repetitive returns True → force_synthesize."""
        state = base_state(iteration_count=3)
        with patch("masis.nodes.supervisor.is_repetitive", return_value=True):
            result = await monitor_and_route(state)
        assert result["supervisor_decision"] == "force_synthesize"


# ---------------------------------------------------------------------------
# Tests for supervisor_slow_path() — MF-SUP-09 through MF-SUP-13
# ---------------------------------------------------------------------------

class TestSupervisorSlowPath:
    """Tests for LLM-based slow path decisions."""

    @pytest.mark.asyncio
    async def test_slow_path_raises_without_openai(self):
        """supervisor_slow_path raises RuntimeError when LangChain unavailable."""
        state = base_state(iteration_count=3)
        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", False):
            with patch("masis.nodes.supervisor.ChatOpenAI", None):
                with pytest.raises(RuntimeError, match="langchain_openai"):
                    await supervisor_slow_path(state)

    @pytest.mark.asyncio
    async def test_slow_path_retry_action(self):
        """MF-SUP-09: retry action resets failing task to pending with new query."""
        t1 = make_task("T1", status="running")
        last_result = make_agent_output("T1", status="failed")

        state = base_state(
            iteration_count=3,
            task_dag=[t1],
            last_task_result=last_result,
        )

        decision = SupervisorDecision(
            action="retry",
            reason="Retry with better query",
            retry_spec=RetrySpec(new_query="Better Q3 revenue query"),
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=decision)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await supervisor_slow_path(state)

        assert result["supervisor_decision"] == "continue"
        updated_dag: List[TaskNode] = result["task_dag"]
        t1_updated = next(t for t in updated_dag if t.task_id == "T1")
        assert t1_updated.status == "running"  # set to pending then picked up by get_next_ready_tasks → running
        assert t1_updated.query.startswith("Better Q3 revenue query")

    @pytest.mark.asyncio
    async def test_slow_path_modify_dag_adds_task(self):
        """MF-SUP-10: modify_dag action adds a new task to the DAG."""
        t1 = make_task("T1", status="done")
        t2 = make_task("T2", ttype="skeptic", status="running", deps=["T1"])

        new_task = make_task("T2b", ttype="web_search", status="pending")
        decision = SupervisorDecision(
            action="modify_dag",
            reason="Add web search for competitor data",
            modify_dag_spec=ModifyDagSpec(add=[new_task], remove=[], update_deps={}),
        )

        state = base_state(
            iteration_count=3,
            task_dag=[t1, t2],
            last_task_result=make_agent_output("T2", status="failed"),
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=decision)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await supervisor_slow_path(state)

        dag_ids = {t.task_id for t in result["task_dag"]}
        assert "T2b" in dag_ids

    @pytest.mark.asyncio
    async def test_slow_path_force_synthesize_action(self):
        """MF-SUP-12: force_synthesize action returns correct decision."""
        decision = SupervisorDecision(
            action="force_synthesize",
            reason="Enough evidence for partial answer",
        )

        state = base_state(iteration_count=8)

        mock_structured_llm = AsyncMock(return_value=decision)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await supervisor_slow_path(state)

        assert result["supervisor_decision"] == "force_synthesize"

    @pytest.mark.asyncio
    async def test_slow_path_stop_action(self):
        """MF-SUP-13: stop action returns failed decision."""
        decision = SupervisorDecision(
            action="stop",
            reason="No evidence found after all retries",
        )

        state = base_state(iteration_count=10)

        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(return_value=decision)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        with patch("masis.nodes.supervisor._OPENAI_AVAILABLE", True):
            with patch("masis.nodes.supervisor.ChatOpenAI", return_value=mock_llm):
                result = await supervisor_slow_path(state)

        assert result["supervisor_decision"] == "failed"


# ---------------------------------------------------------------------------
# Tests for context management — MF-SUP-14
# ---------------------------------------------------------------------------

class TestSupervisorContext:
    """Tests for filtered supervisor context (MF-SUP-14)."""

    def test_build_supervisor_context_excludes_evidence_board(self):
        """MF-SUP-14: supervisor context NEVER contains evidence_board."""
        chunk = EvidenceChunk(
            chunk_id="c1", doc_id="d1", text="Revenue grew 12%", retrieval_score=0.9
        )
        state = base_state(
            evidence_board=[chunk],
            iteration_count=3,
        )
        context = _build_supervisor_context(state)
        assert "evidence_board" not in context
        assert "original_query" in context
        assert "iteration_count" in context

    def test_build_supervisor_context_summary_is_bounded(self):
        """MF-SUP-14: last task summary in context is at most 300 chars."""
        long_summary = "x" * 1000
        last_result = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary=long_summary,
        )
        state = base_state(last_task_result=last_result)
        context = _build_supervisor_context(state)
        assert len(context["last_task_summary"]) <= 400  # Some overhead for prefix text


# ---------------------------------------------------------------------------
# Tests for decision logging — MF-SUP-17
# ---------------------------------------------------------------------------

class TestDecisionLog:
    """Tests for decision audit trail (MF-SUP-17)."""

    @pytest.mark.asyncio
    async def test_fast_path_appends_decision_log_entry(self):
        """MF-SUP-17: every Fast Path call appends to decision_log."""
        state = base_state(iteration_count=15)
        result = await monitor_and_route(state)

        log = result.get("decision_log", [])
        assert len(log) >= 1
        entry = log[-1]
        assert "mode" in entry
        assert "action" in entry
        assert "cost_usd" in entry
        assert "latency_ms" in entry

    @pytest.mark.asyncio
    async def test_fast_path_entry_has_zero_cost(self):
        """MF-SUP-17: Fast Path decisions have cost_usd == 0.0."""
        state = base_state(iteration_count=15)
        result = await monitor_and_route(state)

        log = result.get("decision_log", [])
        fast_entries = [e for e in log if e.get("mode") == "fast"]
        assert all(e["cost_usd"] == 0.0 for e in fast_entries)


# ---------------------------------------------------------------------------
# Tests for supervisor_node() dispatch — MF-SUP-01/02
# ---------------------------------------------------------------------------

class TestSupervisorNode:
    """Tests for the top-level supervisor_node() entry point."""

    @pytest.mark.asyncio
    async def test_supervisor_node_dispatches_plan_on_iteration_0(self):
        """supervisor_node() calls plan_dag when iteration_count == 0."""
        state = base_state(iteration_count=0)
        expected = {"supervisor_decision": "continue", "iteration_count": 1, "decision_log": []}

        with patch("masis.nodes.supervisor.plan_dag", AsyncMock(return_value=expected)) as mock_plan:
            result = await supervisor_node(state)

        mock_plan.assert_called_once()
        assert result["supervisor_decision"] == "continue"

    @pytest.mark.asyncio
    async def test_supervisor_node_dispatches_monitor_on_iteration_gt_0(self):
        """supervisor_node() calls monitor_and_route when iteration_count > 0."""
        state = base_state(iteration_count=2)
        expected = {"supervisor_decision": "force_synthesize", "iteration_count": 3, "decision_log": []}

        with patch("masis.nodes.supervisor.monitor_and_route", AsyncMock(return_value=expected)) as mock_monitor:
            result = await supervisor_node(state)

        mock_monitor.assert_called_once()
        assert result["supervisor_decision"] == "force_synthesize"
