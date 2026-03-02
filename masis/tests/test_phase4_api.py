"""
Tests for Phase 4: API & Observability (ENG-15, ENG-16).

Covers:
- API request/response models (models.py)
- Tracing callback factory (tracing.py)
- Prometheus metrics (metrics.py)
- FastAPI endpoint wiring (main.py) via TestClient
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test: API Models (masis.api.models)
# ---------------------------------------------------------------------------


class TestQueryRequest:
    """Tests for QueryRequest Pydantic model."""

    def test_minimal_query(self):
        from masis.api.models import QueryRequest
        req = QueryRequest(query="What was Q3 revenue?")
        assert req.query == "What was Q3 revenue?"
        assert req.model_overrides is None

    def test_query_with_overrides(self):
        from masis.api.models import QueryRequest
        req = QueryRequest(
            query="Compare revenue",
            model_overrides={"researcher": "gpt-4.1-nano"},
        )
        assert req.model_overrides == {"researcher": "gpt-4.1-nano"}

    def test_empty_query_rejected(self):
        from masis.api.models import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_max_length(self):
        from masis.api.models import QueryRequest
        long_query = "a" * 5000
        req = QueryRequest(query=long_query)
        assert len(req.query) == 5000

    def test_query_over_max_rejected(self):
        from masis.api.models import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(query="a" * 5001)


class TestResumeRequest:
    """Tests for ResumeRequest Pydantic model."""

    def test_basic_resume(self):
        from masis.api.models import ResumeRequest
        req = ResumeRequest(
            thread_id="abc-123",
            action="expand_to_web",
        )
        assert req.thread_id == "abc-123"
        assert req.action == "expand_to_web"
        assert req.data == {}

    def test_resume_with_data(self):
        from masis.api.models import ResumeRequest
        req = ResumeRequest(
            thread_id="abc-123",
            action="accept_partial",
            data={"missing_ok": True},
        )
        assert req.data["missing_ok"] is True

    def test_empty_thread_id_rejected(self):
        from masis.api.models import ResumeRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ResumeRequest(thread_id="", action="cancel")


class TestResponseModels:
    """Tests for QueryResponse, StatusResponse, TraceResponse."""

    def test_query_response_defaults(self):
        from masis.api.models import QueryResponse
        resp = QueryResponse(thread_id="test-123")
        assert resp.status == "processing"
        assert "accepted" in resp.message.lower()

    def test_status_response_defaults(self):
        from masis.api.models import StatusResponse
        resp = StatusResponse(thread_id="test-123")
        assert resp.status == "unknown"
        assert resp.iteration_count == 0
        assert resp.tasks_done == 0
        assert resp.is_interrupted is False

    def test_trace_response_defaults(self):
        from masis.api.models import TraceResponse
        resp = TraceResponse(thread_id="test-123")
        assert resp.total_steps == 0
        assert resp.decision_log == []
        assert resp.quality_scores == {}
        assert resp.checkpoints == []

    def test_sse_event(self):
        from masis.api.models import SSEEvent
        ev = SSEEvent(event_type="plan_created", data={"tasks_count": 3})
        assert ev.event_type == "plan_created"
        assert ev.data["tasks_count"] == 3
        assert ev.timestamp.endswith("+00:00")


# ---------------------------------------------------------------------------
# Test: Tracing (masis.infra.tracing)
# ---------------------------------------------------------------------------


class TestNoOpTracingCallback:
    """Tests for NoOpTracingCallback."""

    def test_no_op_methods_do_not_raise(self):
        from masis.infra.tracing import NoOpTracingCallback
        cb = NoOpTracingCallback()
        cb.on_llm_start({}, ["prompt"])
        cb.on_llm_end(None)
        cb.on_llm_error(ValueError("test"))
        cb.on_chain_start({}, {})
        cb.on_chain_end({})
        cb.on_chain_error(ValueError("test"))
        cb.on_tool_start({}, "input")
        cb.on_tool_end("output")
        cb.on_tool_error(ValueError("test"))

    def test_repr(self):
        from masis.infra.tracing import NoOpTracingCallback
        cb = NoOpTracingCallback()
        assert "NoOp" in repr(cb)


class TestGetTracingCallback:
    """Tests for get_tracing_callback factory."""

    def test_no_env_returns_noop(self):
        from masis.infra.tracing import get_tracing_callback, NoOpTracingCallback
        # Clear any tracing env vars
        env_backup = {}
        for key in ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGSMITH_API_KEY"]:
            env_backup[key] = os.environ.pop(key, None)
        try:
            cb = get_tracing_callback()
            assert isinstance(cb, NoOpTracingCallback)
        finally:
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val

    def test_with_langfuse_env_attempts_import(self):
        from masis.infra.tracing import get_tracing_callback
        env_backup = {}
        for key in ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGSMITH_API_KEY"]:
            env_backup[key] = os.environ.pop(key, None)

        os.environ["LANGFUSE_SECRET_KEY"] = "test-secret"
        os.environ["LANGFUSE_PUBLIC_KEY"] = "test-public"
        try:
            cb = get_tracing_callback(session_name="test-session")
            # Will be NoOp if langfuse is not installed, or LangfuseHandler if it is
            assert cb is not None
        finally:
            for key in ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]:
                os.environ.pop(key, None)
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


class TestGraphInvocationSpan:
    """Tests for GraphInvocationSpan and trace_graph_invocation."""

    def test_span_attributes(self):
        from masis.infra.tracing import GraphInvocationSpan
        span = GraphInvocationSpan(thread_id="t-1", query="test query")
        span.set_attribute("status", "completed")
        assert span.attributes["status"] == "completed"
        assert span.thread_id == "t-1"
        assert span.query == "test query"

    def test_span_duration(self):
        from masis.infra.tracing import GraphInvocationSpan
        span = GraphInvocationSpan(thread_id="t-1", query="test")
        span.start_time = time.time() - 2.0
        span.end_time = time.time()
        assert span.duration_seconds >= 1.9

    def test_context_manager_records_timing(self):
        from masis.infra.tracing import trace_graph_invocation
        with trace_graph_invocation("t-1", "test query") as span:
            time.sleep(0.01)
            span.set_attribute("custom", "value")
        assert span.duration_seconds > 0
        assert span.attributes["custom"] == "value"
        assert span.end_time > span.start_time

    def test_context_manager_records_error(self):
        from masis.infra.tracing import trace_graph_invocation
        with pytest.raises(ValueError):
            with trace_graph_invocation("t-err", "bad query") as span:
                raise ValueError("test error")
        assert span.attributes.get("error") is True
        assert "ValueError" in span.attributes.get("error.type", "")

    def test_long_query_truncated(self):
        from masis.infra.tracing import GraphInvocationSpan
        long_q = "x" * 500
        span = GraphInvocationSpan(thread_id="t-1", query=long_q)
        assert len(span.query) == 200


# ---------------------------------------------------------------------------
# Test: Metrics (masis.infra.metrics)
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for Prometheus metrics module."""

    def test_record_query_metrics(self):
        from masis.infra.metrics import record_query_metrics
        summary = record_query_metrics(
            latency_seconds=5.2,
            cost_usd=0.03,
            agent_type_counts={"researcher": 2, "skeptic": 1},
            fast_path_decisions=3,
            total_decisions=5,
        )
        assert summary["latency_seconds"] == 5.2
        assert summary["cost_usd"] == 0.03
        assert summary["fast_path_ratio"] == 0.6
        assert summary["error_type"] is None

    def test_record_query_metrics_with_error(self):
        from masis.infra.metrics import record_query_metrics
        summary = record_query_metrics(
            latency_seconds=1.0,
            cost_usd=0.0,
            agent_type_counts={},
            error_type="TimeoutError",
        )
        assert summary["error_type"] == "TimeoutError"

    def test_extract_metrics_from_result(self):
        from masis.infra.metrics import extract_metrics_from_result
        result = {
            "token_budget": {"total_cost_usd": 0.15, "total_tokens_used": 50000},
            "api_call_counts": {"researcher": 3, "skeptic": 1},
            "decision_log": [
                {"mode": "fast", "decision": "continue"},
                {"mode": "fast", "decision": "continue"},
                {"mode": "slow", "decision": "retry"},
            ],
            "iteration_count": 3,
        }
        start = time.time() - 10.0
        metrics = extract_metrics_from_result(result, start)
        assert metrics["latency_seconds"] >= 9.0
        assert metrics["cost_usd"] == 0.15
        assert metrics["agent_type_counts"] == {"researcher": 3, "skeptic": 1}
        assert metrics["fast_path_decisions"] == 2
        assert metrics["total_decisions"] == 3

    def test_extract_metrics_empty_result(self):
        from masis.infra.metrics import extract_metrics_from_result
        metrics = extract_metrics_from_result({}, time.time())
        assert metrics["cost_usd"] == 0.0
        assert metrics["agent_type_counts"] == {}
        assert metrics["fast_path_decisions"] == 0

    def test_noop_metrics_dont_crash(self):
        from masis.infra.metrics import _NoOpMetric
        m = _NoOpMetric()
        m.observe(1.0)
        m.inc(1.0)
        m.set(0.5)
        m.labels(agent_type="test").inc()
        assert repr(m).startswith("_NoOpMetric")

    def test_get_metrics_app_returns_callable(self):
        from masis.infra.metrics import get_metrics_app
        app = get_metrics_app()
        assert callable(app)


# ---------------------------------------------------------------------------
# Test: FastAPI app creation (masis.api.main)
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for FastAPI app factory and endpoint wiring."""

    def test_create_app_returns_fastapi(self):
        from masis.api.main import create_app
        app = create_app()
        assert app.title == "MASIS API"
        assert app.version == "2.0.0"

    def test_app_has_required_routes(self):
        from masis.api.main import create_app
        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/masis/query" in route_paths
        assert "/masis/resume" in route_paths
        assert "/masis/status/{thread_id}" in route_paths
        assert "/masis/trace/{thread_id}" in route_paths
        assert "/masis/stream/{thread_id}" in route_paths
        assert "/health" in route_paths

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from masis.api.main import create_app
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_query_endpoint_returns_thread_id(self):
        from fastapi.testclient import TestClient
        from masis.api.main import create_app

        app = create_app()
        client = TestClient(app)

        # Mock the graph so we don't actually run the pipeline
        with patch("masis.api.main.ainvoke_graph", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = {
                "iteration_count": 1,
                "token_budget": {"total_cost_usd": 0.01},
                "api_call_counts": {},
                "decision_log": [],
            }
            resp = client.post(
                "/masis/query",
                json={"query": "What was Q3 revenue?"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert data["status"] == "processing"
        assert len(data["thread_id"]) > 10  # UUID length

    def test_query_endpoint_rejects_empty(self):
        from fastapi.testclient import TestClient
        from masis.api.main import create_app
        app = create_app()
        client = TestClient(app)
        resp = client.post("/masis/query", json={"query": ""})
        assert resp.status_code == 422  # Validation error

    def test_status_endpoint_unknown_thread(self):
        from fastapi.testclient import TestClient
        from masis.api.main import create_app

        app = create_app()
        client = TestClient(app)

        # Mock get_graph to return a mock graph
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {}
        mock_state.next = []
        mock_graph.aget_state = AsyncMock(return_value=mock_state)

        with patch("masis.api.main.get_graph", return_value=mock_graph):
            resp = client.get("/masis/status/nonexistent-thread")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unknown"

    def test_trace_endpoint_empty(self):
        from fastapi.testclient import TestClient
        from masis.api.main import create_app

        app = create_app()
        client = TestClient(app)

        mock_graph = MagicMock()
        with patch("masis.api.main.get_graph", return_value=mock_graph):
            with patch(
                "masis.api.main.get_state_history_for_thread_async",
                new_callable=AsyncMock,
                return_value=[],
            ):
                resp = client.get("/masis/trace/nonexistent-thread")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_steps"] == 0
        assert data["decision_log"] == []


class TestHelperFunctions:
    """Tests for internal helper functions in main.py."""

    def test_safe_serialise_primitives(self):
        from masis.api.main import _safe_serialise
        assert _safe_serialise(None) is None
        assert _safe_serialise("hello") == "hello"
        assert _safe_serialise(42) == 42
        assert _safe_serialise(3.14) == 3.14
        assert _safe_serialise(True) is True

    def test_safe_serialise_nested(self):
        from masis.api.main import _safe_serialise
        result = _safe_serialise({"a": [1, {"b": 2}]})
        assert result == {"a": [1, {"b": 2}]}

    def test_safe_serialise_pydantic(self):
        from masis.api.main import _safe_serialise
        from masis.api.models import QueryResponse
        resp = QueryResponse(thread_id="t-1")
        result = _safe_serialise(resp)
        assert isinstance(result, dict)
        assert result["thread_id"] == "t-1"

    def test_format_sse(self):
        from masis.api.main import _format_sse
        from masis.api.models import SSEEvent
        ev = SSEEvent(event_type="heartbeat", data={"thread_id": "t-1"})
        formatted = _format_sse(ev)
        assert formatted.startswith("event: heartbeat\n")
        assert "data:" in formatted
        assert formatted.endswith("\n\n")
        # Parse the data line
        data_line = [l for l in formatted.split("\n") if l.startswith("data:")][0]
        payload = json.loads(data_line[len("data: "):])
        assert payload["type"] == "heartbeat"

    def test_classify_stream_event_supervisor(self):
        from masis.api.main import _classify_stream_event
        update = {
            "supervisor": {
                "task_dag": [{"task_id": "T1"}, {"task_id": "T2"}],
                "iteration_count": 1,
                "supervisor_decision": "dispatch",
            }
        }
        event = _classify_stream_event(update, -1, "")
        assert event.event_type == "plan_created"
        assert event.data["tasks_count"] == 2

    def test_classify_stream_event_executor(self):
        from masis.api.main import _classify_stream_event
        update = {
            "executor": {
                "last_task_result": {"task_id": "T1", "status": "done"},
            }
        }
        event = _classify_stream_event(update, 1, "dispatch")
        assert event.event_type == "task_completed"

    def test_classify_stream_event_hitl(self):
        from masis.api.main import _classify_stream_event
        update = {
            "supervisor": {
                "hitl_payload": {"type": "ambiguity", "options": ["A", "B"]},
                "iteration_count": 2,
            }
        }
        event = _classify_stream_event(update, 1, "")
        assert event.event_type == "hitl_required"

    def test_get_task_status_dict(self):
        from masis.api.main import _get_task_status
        assert _get_task_status({"status": "done"}) == "done"
        assert _get_task_status({}) == "pending"

    def test_get_task_id_dict(self):
        from masis.api.main import _get_task_id
        assert _get_task_id({"task_id": "T1"}) == "T1"
        assert _get_task_id({}) is None


# ---------------------------------------------------------------------------
# Test: Integration -- infra __init__ exports
# ---------------------------------------------------------------------------


class TestInfraExports:
    """Verify that the updated infra __init__ exports the new modules."""

    def test_tracing_exports(self):
        from masis.infra import get_tracing_callback, trace_graph_invocation
        from masis.infra import NoOpTracingCallback, GraphInvocationSpan
        assert callable(get_tracing_callback)
        assert callable(trace_graph_invocation)

    def test_metrics_exports(self):
        from masis.infra import (
            query_latency_seconds,
            cost_per_query_usd,
            fast_path_ratio,
            agent_calls_total,
            errors_total,
            active_queries,
            record_query_metrics,
            extract_metrics_from_result,
            get_metrics_app,
            MetricsMiddleware,
        )
        assert callable(record_query_metrics)
        assert callable(get_metrics_app)


# ---------------------------------------------------------------------------
# Test: API __init__ exports
# ---------------------------------------------------------------------------


class TestApiExports:
    """Verify that the api __init__ exports work correctly."""

    def test_model_exports(self):
        from masis.api import (
            QueryRequest,
            QueryResponse,
            ResumeRequest,
            StatusResponse,
            TraceResponse,
        )
        assert QueryRequest is not None
        assert QueryResponse is not None

    def test_create_app_export(self):
        from masis.api import create_app
        assert callable(create_app)
