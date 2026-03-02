"""
Tests for masis.graph — ENG-14 (Graph Assembly & Routing).

Coverage
--------
- route_supervisor()  : conditional edge routing (M1 S2)
- route_validator()   : conditional edge routing (M1 S3)
- build_workflow()    : graph topology — 3 nodes, correct edges (M1 S1)
- compile_workflow()  : checkpointer wiring (M2 S1)
- runner helpers      : generate_thread_id, make_config, _build_initial_state (M2/M3)
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    END = "__end__"
    _LANGGRAPH_AVAILABLE = False

try:
    from masis.graph.edges import (
        NODE_EXECUTOR,
        NODE_SUPERVISOR,
        NODE_VALIDATOR,
        DECISION_CONTINUE,
        DECISION_READY_FOR_VALIDATION,
        DECISION_FORCE_SYNTHESIZE,
        DECISION_HITL_PAUSE,
        DECISION_FAILED,
        route_supervisor,
        route_validator,
    )
    from masis.graph.runner import (
        generate_thread_id,
        make_config,
        reset_graph,
        _build_initial_state,
    )
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)

pytestmark = pytest.mark.skipif(
    not _IMPORTS_OK,
    reason=f"Import failed: {locals().get('_IMPORT_ERROR', '')}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_state(**kwargs: Any) -> Dict[str, Any]:
    """Return a minimal state dict with sensible defaults."""
    defaults = {
        "supervisor_decision": "",
        "validation_pass": False,
        "validation_round": 0,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Tests for route_supervisor() — ENG-14 M1 S2
# ---------------------------------------------------------------------------

class TestRouteSupervisor:
    """Tests for the supervisor conditional edge function."""

    def test_continue_routes_to_executor(self):
        """'continue' → executor."""
        state = make_state(supervisor_decision=DECISION_CONTINUE)
        assert route_supervisor(state) == NODE_EXECUTOR

    def test_ready_for_validation_routes_to_validator(self):
        """'ready_for_validation' → validator."""
        state = make_state(supervisor_decision=DECISION_READY_FOR_VALIDATION)
        assert route_supervisor(state) == NODE_VALIDATOR

    def test_force_synthesize_routes_to_executor(self):
        """'force_synthesize' → executor (synthesizer dispatched via state flag)."""
        state = make_state(supervisor_decision=DECISION_FORCE_SYNTHESIZE)
        assert route_supervisor(state) == NODE_EXECUTOR

    def test_hitl_pause_routes_to_end(self):
        """'hitl_pause' → END."""
        state = make_state(supervisor_decision=DECISION_HITL_PAUSE)
        assert route_supervisor(state) == END

    def test_failed_routes_to_end(self):
        """'failed' → END."""
        state = make_state(supervisor_decision=DECISION_FAILED)
        assert route_supervisor(state) == END

    def test_unknown_decision_routes_to_end(self):
        """Any unknown decision value → END (safe fallback)."""
        state = make_state(supervisor_decision="totally_unknown")
        assert route_supervisor(state) == END

    def test_empty_decision_routes_to_end(self):
        """Empty string decision → END (treated as 'failed')."""
        state = make_state(supervisor_decision="")
        assert route_supervisor(state) == END

    def test_missing_decision_defaults_to_end(self):
        """Missing supervisor_decision key → defaults to 'failed' → END."""
        state = {}
        assert route_supervisor(state) == END


# ---------------------------------------------------------------------------
# Tests for route_validator() — ENG-14 M1 S3
# ---------------------------------------------------------------------------

class TestRouteValidator:
    """Tests for the validator conditional edge function."""

    def test_pass_routes_to_end(self):
        """validation_pass=True → END (answer accepted)."""
        state = make_state(validation_pass=True)
        assert route_validator(state) == END

    def test_fail_routes_to_supervisor(self):
        """validation_pass=False → supervisor (revision needed)."""
        state = make_state(validation_pass=False)
        assert route_validator(state) == NODE_SUPERVISOR

    def test_missing_validation_pass_routes_to_supervisor(self):
        """Missing validation_pass defaults to False → supervisor."""
        state = {}
        assert route_validator(state) == NODE_SUPERVISOR

    def test_pass_with_round_number(self):
        """Pass on round 3 still routes to END."""
        state = make_state(validation_pass=True, validation_round=3)
        assert route_validator(state) == END


# ---------------------------------------------------------------------------
# Tests for constants
# ---------------------------------------------------------------------------

class TestEdgeConstants:
    """Verify critical string constants used in routing."""

    def test_node_executor_value(self):
        assert NODE_EXECUTOR == "executor"

    def test_node_supervisor_value(self):
        assert NODE_SUPERVISOR == "supervisor"

    def test_node_validator_value(self):
        assert NODE_VALIDATOR == "validator"

    def test_decision_continue_value(self):
        assert DECISION_CONTINUE == "continue"

    def test_decision_ready_for_validation_value(self):
        assert DECISION_READY_FOR_VALIDATION == "ready_for_validation"

    def test_decision_force_synthesize_value(self):
        assert DECISION_FORCE_SYNTHESIZE == "force_synthesize"

    def test_decision_hitl_pause_value(self):
        assert DECISION_HITL_PAUSE == "hitl_pause"

    def test_decision_failed_value(self):
        assert DECISION_FAILED == "failed"


# ---------------------------------------------------------------------------
# Tests for build_workflow() — ENG-14 M1 S1
# ---------------------------------------------------------------------------

class TestBuildWorkflow:
    """Tests for StateGraph construction."""

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_workflow_builds_without_error(self):
        """build_workflow() returns a StateGraph without crashing."""
        from masis.graph.workflow import build_workflow
        wf = build_workflow()
        assert wf is not None

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_workflow_has_three_nodes(self):
        """Graph should contain supervisor, executor, and validator nodes."""
        from masis.graph.workflow import build_workflow
        wf = build_workflow()
        node_names = set(wf.nodes.keys())
        assert "supervisor" in node_names
        assert "executor" in node_names
        assert "validator" in node_names

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_workflow_compiles_with_in_memory_saver(self):
        """Graph compiles successfully with InMemorySaver."""
        from langgraph.checkpoint.memory import InMemorySaver
        from masis.graph.workflow import build_workflow
        wf = build_workflow()
        compiled = wf.compile(checkpointer=InMemorySaver())
        assert compiled is not None


# ---------------------------------------------------------------------------
# Tests for compile_workflow() — ENG-14 M2 S1
# ---------------------------------------------------------------------------

class TestCompileWorkflow:
    """Tests for compile_workflow() function."""

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_compile_with_explicit_checkpointer(self):
        """compile_workflow() accepts an explicit checkpointer."""
        from langgraph.checkpoint.memory import InMemorySaver
        from masis.graph.workflow import compile_workflow
        compiled = compile_workflow(checkpointer=InMemorySaver())
        assert compiled is not None

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_compile_falls_back_to_in_memory(self):
        """compile_workflow() without Postgres uses InMemorySaver."""
        from masis.graph.workflow import compile_workflow
        with patch.dict("os.environ", {}, clear=True):
            compiled = compile_workflow()
        assert compiled is not None


# ---------------------------------------------------------------------------
# Tests for runner helpers — ENG-14 M2/M3
# ---------------------------------------------------------------------------

class TestRunnerHelpers:
    """Tests for graph runner utility functions."""

    def test_generate_thread_id_is_uuid(self):
        """generate_thread_id() returns a valid UUID string."""
        tid = generate_thread_id()
        assert isinstance(tid, str)
        assert len(tid) == 36  # UUID format: 8-4-4-4-12
        assert tid.count("-") == 4

    def test_generate_thread_id_unique(self):
        """Each call returns a unique thread_id."""
        ids = {generate_thread_id() for _ in range(100)}
        assert len(ids) == 100

    def test_make_config_auto_generates_thread_id(self):
        """make_config() without args generates a config with thread_id."""
        config = make_config()
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert len(config["configurable"]["thread_id"]) == 36

    def test_make_config_uses_provided_thread_id(self):
        """make_config(thread_id=...) uses the provided ID."""
        config = make_config(thread_id="my-custom-thread")
        assert config["configurable"]["thread_id"] == "my-custom-thread"

    def test_make_config_merges_extra(self):
        """make_config(extra=...) merges extra keys into configurable."""
        config = make_config(thread_id="t1", extra={"recursion_limit": 50})
        assert config["configurable"]["recursion_limit"] == 50

    def test_build_initial_state_has_required_keys(self):
        """_build_initial_state() produces a dict with all MASISState fields."""
        state = _build_initial_state("What was Q3 revenue?")
        required_keys = [
            "original_query",
            "query_id",
            "task_dag",
            "iteration_count",
            "next_tasks",
            "supervisor_decision",
            "evidence_board",
            "decision_log",
            "token_budget",
            "api_call_counts",
            "start_time",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_build_initial_state_query_preserved(self):
        """Original query is stored in the state."""
        state = _build_initial_state("How is AI impacting operations?")
        assert state["original_query"] == "How is AI impacting operations?"

    def test_build_initial_state_iteration_starts_at_zero(self):
        """Iteration count starts at 0 (triggers plan_dag)."""
        state = _build_initial_state("test")
        assert state["iteration_count"] == 0

    def test_build_initial_state_empty_dag(self):
        """Task DAG starts empty (supervisor will populate it)."""
        state = _build_initial_state("test")
        assert state["task_dag"] == []

    def test_build_initial_state_empty_evidence(self):
        """Evidence board starts empty."""
        state = _build_initial_state("test")
        assert state["evidence_board"] == []


# ---------------------------------------------------------------------------
# Tests for graph singleton lifecycle
# ---------------------------------------------------------------------------

class TestGraphSingleton:
    """Tests for compile_graph / get_graph / reset_graph lifecycle."""

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_reset_graph_clears_singleton(self):
        """reset_graph() clears the cached graph."""
        from masis.graph.runner import reset_graph, _compiled_graph
        reset_graph()
        # After reset, module-level _compiled_graph should be None
        from masis.graph import runner as runner_mod
        assert runner_mod._compiled_graph is None

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_compile_graph_returns_same_instance(self):
        """compile_graph() returns cached instance on second call."""
        from masis.graph.runner import compile_graph, reset_graph
        reset_graph()
        g1 = compile_graph()
        g2 = compile_graph()
        assert g1 is g2

    @pytest.mark.skipif(not _LANGGRAPH_AVAILABLE, reason="langgraph not installed")
    def test_compile_graph_force_recompiles(self):
        """compile_graph(force=True) creates a new instance."""
        from masis.graph.runner import compile_graph, reset_graph
        reset_graph()
        g1 = compile_graph()
        g2 = compile_graph(force=True)
        assert g1 is not g2
        # Cleanup
        reset_graph()
