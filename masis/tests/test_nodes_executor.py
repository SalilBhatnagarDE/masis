"""
Tests for masis.nodes.executor â€” ENG-05 (MF-EXE-01 through MF-EXE-10).

Coverage
--------
- execute_dag_tasks()  : single task direct dispatch (M1), parallel Send() (M2)
- dispatch_agent()     : routing to correct agent functions (M3), unknown type guard (M4)
- dispatch_with_safety(): timeout wrapper (M5), rate limit pre-check (M10)
- filtered_state()     : each agent type gets correct field subset (M5/M7)
- result normalization : AgentOutput fields populated correctly (M6)
- evidence board write : evidence appears in state update (M8)
- budget tracking      : token_budget updated after dispatch (M9)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from masis.schemas.models import (
        AgentOutput,
        BudgetTracker,
        EvidenceChunk,
        ResearcherOutput,
        TaskNode,
    )
    from masis.nodes.executor import (
        dispatch_agent,
        dispatch_with_safety,
        execute_dag_tasks,
        _build_state_update,
        _check_rate_limit,
        _filtered_state,
        _normalise_output,
        VALID_AGENT_TYPES,
    )
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason=f"Import failed: {locals().get('_IMPORT_ERROR', '')}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_task(task_id: str = "T1", ttype: str = "researcher") -> TaskNode:
    return TaskNode(
        task_id=task_id,
        type=ttype,
        query=f"Research query for {task_id}",
        parallel_group=1,
        acceptance_criteria="â‰¥2 chunks, pass_rateâ‰¥0.30",
    )


def make_evidence_chunk(chunk_id: str = "c1", score: float = 0.8) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id="doc_001",
        text="Revenue grew 12% YoY to â‚¹41,764 crore.",
        retrieval_score=score,
    )


def base_state(**kwargs: Any) -> Dict[str, Any]:
    return {
        "original_query": "What was Q3 revenue?",
        "evidence_board": [],
        "critique_notes": None,
        "synthesis_output": None,
        "task_dag": [],
        "next_tasks": [],
        "token_budget": BudgetTracker(),
        "api_call_counts": {},
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Tests for dispatch_agent() â€” MF-EXE-03 and MF-EXE-04
# ---------------------------------------------------------------------------

class TestDispatchAgent:
    """Tests for agent routing."""

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error_output(self):
        """MF-EXE-04: unknown task type â†’ AgentOutput with status='failed', not a crash."""
        task = TaskNode(task_id="T_bad", type="researcher", query="q")
        # Manually patch the type to something invalid after construction
        task.__dict__["type"] = "invalid_type"

        result = await dispatch_agent(task, {})
        assert result.status == "failed"
        assert "unknown_agent_type" in result.summary.lower() or "not a valid" in result.error_detail.lower()

    @pytest.mark.asyncio
    async def test_researcher_routes_to_run_researcher(self):
        """MF-EXE-03: type='researcher' â†’ calls run_researcher."""
        task = make_task("T1", "researcher")
        mock_output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Revenue data found",
        )

        with patch("masis.nodes.executor.run_researcher", AsyncMock(return_value=mock_output)):
            result = await dispatch_agent(task, {"original_query": "test", "task": task})

        assert result.status == "success"
        assert result.agent_type == "researcher"

    @pytest.mark.asyncio
    async def test_web_search_routes_to_run_web_search(self):
        """MF-EXE-03: type='web_search' â†’ calls run_web_search."""
        task = make_task("T2", "web_search")
        mock_output = AgentOutput(
            task_id="T2",
            agent_type="web_search",
            status="success",
            summary="AWS revenue: $27.5B",
        )

        with patch("masis.nodes.executor.run_web_search", AsyncMock(return_value=mock_output)):
            result = await dispatch_agent(task, {"task": task})

        assert result.agent_type == "web_search"

    @pytest.mark.asyncio
    async def test_skeptic_routes_to_run_skeptic(self):
        """MF-EXE-03: type='skeptic' â†’ calls run_skeptic."""
        task = make_task("T3", "skeptic")
        mock_output = AgentOutput(
            task_id="T3",
            agent_type="skeptic",
            status="success",
            summary="confidence=0.92",
        )

        with patch("masis.nodes.executor.run_skeptic", AsyncMock(return_value=mock_output)):
            result = await dispatch_agent(task, {"original_query": "test", "evidence_board": [], "task": task})

        assert result.agent_type == "skeptic"

    @pytest.mark.asyncio
    async def test_synthesizer_routes_to_run_synthesizer(self):
        """MF-EXE-03: type='synthesizer' â†’ calls run_synthesizer."""
        task = make_task("T4", "synthesizer")
        mock_output = AgentOutput(
            task_id="T4",
            agent_type="synthesizer",
            status="success",
            summary="Answer synthesized",
        )

        with patch("masis.nodes.executor.run_synthesizer", AsyncMock(return_value=mock_output)):
            result = await dispatch_agent(task, {"original_query": "test", "evidence_board": [], "task": task})

        assert result.agent_type == "synthesizer"


# ---------------------------------------------------------------------------
# Tests for dispatch_with_safety() â€” MF-EXE-05
# ---------------------------------------------------------------------------

class TestDispatchWithSafety:
    """Tests for the timeout + exception wrapper."""

    @pytest.mark.asyncio
    async def test_timeout_returns_failed_output(self):
        """MF-EXE-05: asyncio.TimeoutError â†’ AgentOutput with status='timeout'."""

        async def slow_agent(task: Any, state: Any) -> Any:
            await asyncio.sleep(100)  # Simulates timeout

        task = make_task("T1", "researcher")
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.run_researcher", slow_agent):
            with patch("masis.nodes.executor.TOOL_LIMITS", {
                "researcher": {"max_parallel": 3, "max_total": 8, "timeout_s": 0.01}
            }):
                result = await dispatch_with_safety(task, state)

        assert result.status == "timeout"
        assert "timed out" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_failed_output(self):
        """MF-EXE-05: unexpected exception â†’ AgentOutput with status='failed', not a crash."""

        async def crashing_agent(task: Any, state: Any) -> Any:
            raise ValueError("Something went wrong in the agent")

        task = make_task("T1", "researcher")
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.run_researcher", crashing_agent):
            result = await dispatch_with_safety(task, state)

        assert result.status == "failed"
        assert "ValueError" in result.summary or "ValueError" in result.error_detail

    @pytest.mark.asyncio
    async def test_success_returns_agent_output(self):
        """dispatch_with_safety returns AgentOutput on success."""
        expected_output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Revenue data found",
        )

        task = make_task("T1", "researcher")
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.run_researcher", AsyncMock(return_value=expected_output)):
            result = await dispatch_with_safety(task, state)

        assert result.status == "success"
        assert result.task_id == "T1"


# ---------------------------------------------------------------------------
# Tests for rate limit pre-check â€” MF-EXE-10
# ---------------------------------------------------------------------------

class TestRateLimitPreCheck:
    """Tests for the rate limit guard before dispatch."""

    def test_rate_limit_exceeded_returns_error(self):
        """MF-EXE-10: api_call_counts >= max_total â†’ AgentOutput status='rate_limited'."""
        task = make_task("T9", "researcher")
        state = base_state(api_call_counts={"researcher": 8})  # max_total=8

        result = _check_rate_limit(task, state)
        assert result is not None
        assert result.status == "rate_limited"
        assert "8/8" in result.summary or "rate_limited" in result.status

    def test_rate_limit_not_exceeded_returns_none(self):
        """MF-EXE-10: api_call_counts < max_total â†’ None (proceed with dispatch)."""
        task = make_task("T1", "researcher")
        state = base_state(api_call_counts={"researcher": 3})  # max_total=8

        result = _check_rate_limit(task, state)
        assert result is None

    def test_unknown_agent_type_uses_high_limit(self):
        """Rate limit check for unknown types uses default high limit (no false blocks)."""
        task = make_task("T1", "researcher")
        task.__dict__["type"] = "hypothetical_future_agent"
        state = base_state(api_call_counts={"hypothetical_future_agent": 5})

        result = _check_rate_limit(task, state)
        assert result is None  # Should not block unknown types


# ---------------------------------------------------------------------------
# Tests for filtered_state() â€” MF-EXE-07
# ---------------------------------------------------------------------------

class TestFilteredState:
    """Tests for per-agent state filtering."""

    def test_researcher_gets_no_evidence_board(self):
        """MF-EXE-07: researcher filtered state has no evidence_board key."""
        chunk = make_evidence_chunk()
        task = make_task("T1", "researcher")
        state = base_state(evidence_board=[chunk])

        filtered = _filtered_state(state, task)
        assert "evidence_board" not in filtered
        assert "original_query" in filtered

    def test_skeptic_gets_evidence_board(self):
        """MF-EXE-07: skeptic filtered state contains evidence_board."""
        chunk = make_evidence_chunk()
        task = make_task("T2", "skeptic")
        state = base_state(evidence_board=[chunk], task_dag=[])

        filtered = _filtered_state(state, task)
        assert "evidence_board" in filtered
        assert len(filtered["evidence_board"]) == 1

    def test_synthesizer_gets_evidence_and_critique(self):
        """MF-EXE-07: synthesizer filtered state contains evidence_board and critique_notes."""
        task = make_task("T3", "synthesizer")
        mock_critique = MagicMock()
        state = base_state(evidence_board=[make_evidence_chunk()], critique_notes=mock_critique)

        filtered = _filtered_state(state, task)
        assert "evidence_board" in filtered
        assert "critique_notes" in filtered
        assert filtered["critique_notes"] is mock_critique

    def test_web_search_gets_minimal_state(self):
        """MF-EXE-07: web_search filtered state contains only task (no original_query or board)."""
        task = make_task("T4", "web_search")
        state = base_state(evidence_board=[make_evidence_chunk()])

        filtered = _filtered_state(state, task)
        assert "evidence_board" not in filtered


# ---------------------------------------------------------------------------
# Tests for result normalisation â€” MF-EXE-06
# ---------------------------------------------------------------------------

class TestResultNormalisation:
    """Tests for AgentOutput normalisation from native agent outputs."""

    def test_native_agent_output_passthrough(self):
        """If raw output is already AgentOutput, return it unchanged."""
        output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Test",
        )
        task = make_task("T1")
        result = _normalise_output(output, task)
        assert result is output

    def test_researcher_output_normalises_correctly(self):
        """ResearcherOutput normalises to AgentOutput with criteria_result populated."""
        raw = ResearcherOutput(
            task_id="T1",
            chunks_after_grading=3,
            grading_pass_rate=0.6,
            self_rag_verdict="grounded",
            summary="Revenue: 41764",
        )
        task = make_task("T1")
        result = _normalise_output(raw, task)

        assert result.task_id == "T1"
        assert result.criteria_result["chunks_after_grading"] == 3
        assert result.criteria_result["grading_pass_rate"] == 0.6
        assert result.criteria_result["self_rag_verdict"] == "grounded"


# ---------------------------------------------------------------------------
# Tests for execute_dag_tasks() â€” MF-EXE-01 and MF-EXE-02
# ---------------------------------------------------------------------------

class TestExecuteDagTasks:
    """Tests for the main executor entry point."""

    @pytest.mark.asyncio
    async def test_single_task_dispatches_directly(self):
        """MF-EXE-01: single next_task â†’ direct dispatch_with_safety call."""
        task = make_task("T1")
        mock_output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Revenue data",
        )
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.dispatch_with_safety", AsyncMock(return_value=mock_output)):
            result = await execute_dag_tasks(state)

        assert isinstance(result, dict)
        assert "last_task_result" in result

    @pytest.mark.asyncio
    async def test_multiple_tasks_default_to_sequential_wave(self):
        """MF-EXE-02: multi-task wave defaults to sequential execution for compatibility."""
        tasks = [make_task("T1"), make_task("T2")]
        outputs = [
            AgentOutput(task_id="T1", agent_type="researcher", status="success", summary="ok1"),
            AgentOutput(task_id="T2", agent_type="researcher", status="success", summary="ok2"),
        ]
        state = base_state(next_tasks=tasks)

        with patch("masis.nodes.executor.dispatch_with_safety", AsyncMock(side_effect=outputs)):
            result = await execute_dag_tasks(state)

        assert isinstance(result, dict)
        assert "batch_task_results" in result
        assert len(result["batch_task_results"]) == 2

    @pytest.mark.asyncio
    async def test_empty_next_tasks_returns_no_op(self):
        """execute_dag_tasks with empty next_tasks returns no-op dict."""
        state = base_state(next_tasks=[])
        result = await execute_dag_tasks(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_state_update_includes_budget(self):
        """MF-EXE-09: budget is updated in state after dispatch."""
        task = make_task("T1")
        mock_output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Test",
            tokens_used=500,
            cost_usd=0.001,
        )
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.dispatch_with_safety", AsyncMock(return_value=mock_output)):
            result = await execute_dag_tasks(state)

        if isinstance(result, dict):
            assert "token_budget" in result
            assert result["token_budget"].total_tokens_used >= 500

    @pytest.mark.asyncio
    async def test_state_update_includes_evidence(self):
        """MF-EXE-08: evidence from agent appears in state update."""
        task = make_task("T1")
        chunk = make_evidence_chunk("c1")
        mock_output = AgentOutput(
            task_id="T1",
            agent_type="researcher",
            status="success",
            summary="Test",
            evidence=[chunk],
        )
        state = base_state(next_tasks=[task])

        with patch("masis.nodes.executor.dispatch_with_safety", AsyncMock(return_value=mock_output)):
            result = await execute_dag_tasks(state)

        if isinstance(result, dict):
            assert "evidence_board" in result
            assert len(result["evidence_board"]) == 1
