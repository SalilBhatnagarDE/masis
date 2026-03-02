"""
masis.infra
===========
Phase 2 infrastructure modules for MASIS (Multi-Agent Supervisor-Monitored
Intelligent System).

Modules
-------
persistence     -- PostgresSaver / InMemorySaver checkpoint management (ENG-11)
circuit_breaker -- Three-state circuit breaker with failure thresholds (ENG-12)
safety          -- Per-role circuit breakers, content sanitizer, rate-limit
                   guards wired to model fallback chains (ENG-12)
hitl            -- Human-in-the-Loop integration: ambiguity detection, DAG
                   approval interrupt, risk gate, resume handler (ENG-13)
tracing         -- Langfuse/LangSmith/no-op tracing callback factory (ENG-16)
metrics         -- Prometheus counters/histograms for observability (ENG-16)

All modules use forward-compatible imports from masis.schemas.models and
masis.config.model_routing with try/except ImportError stubs so that they
remain runnable even when Phase 0 deliverables are not yet present in the
Python path.

MF-IDs covered
--------------
MF-MEM-08   Checkpoint persistence (PostgresSaver / InMemorySaver)
MF-HITL-01  Pre-supervisor ambiguity gate
MF-HITL-02  DAG approval pause
MF-HITL-03  Mid-execution evidence pause (wired via Supervisor escalate)
MF-HITL-04  Risk gate pause
MF-HITL-05  Resume with action handler
MF-HITL-06  Graceful partial result
MF-HITL-07  Cancel support
MF-SAFE-01  3-layer loop prevention (documented; circuit breaker is Layer 0)
MF-SAFE-02  3-state circuit breaker (CLOSED / OPEN / HALF_OPEN)
MF-SAFE-03  Model fallback chains per role
MF-SAFE-04  Content sanitizer (wired through safety guards)
MF-SAFE-05  Rate limiting per agent type
MF-SAFE-06  Budget enforcement helpers
MF-SAFE-07  Graceful degradation (never crash; always return best-effort)
MF-SAFE-08  Drift detection helpers
"""

from masis.infra.circuit_breaker import CircuitBreaker, CircuitState
from masis.infra.persistence import get_checkpointer, CheckpointerContext
from masis.infra.safety import (
    ROLE_BREAKERS,
    TOOL_LIMITS,
    BUDGET_LIMITS,
    content_sanitizer,
    check_rate_limit,
    check_budget,
    call_with_fallback,
)
from masis.infra.hitl import (
    AmbiguityClassification,
    ambiguity_detector,
    dag_approval_interrupt,
    risk_gate,
    handle_resume,
    build_partial_result,
    build_cancel_result,
)
from masis.infra.tracing import (
    get_tracing_callback,
    trace_graph_invocation,
    NoOpTracingCallback,
    GraphInvocationSpan,
)
from masis.infra.metrics import (
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

__all__ = [
    # circuit_breaker
    "CircuitBreaker",
    "CircuitState",
    # persistence
    "get_checkpointer",
    "CheckpointerContext",
    # safety
    "ROLE_BREAKERS",
    "TOOL_LIMITS",
    "BUDGET_LIMITS",
    "content_sanitizer",
    "check_rate_limit",
    "check_budget",
    "call_with_fallback",
    # hitl
    "AmbiguityClassification",
    "ambiguity_detector",
    "dag_approval_interrupt",
    "risk_gate",
    "handle_resume",
    "build_partial_result",
    "build_cancel_result",
    # tracing (ENG-16)
    "get_tracing_callback",
    "trace_graph_invocation",
    "NoOpTracingCallback",
    "GraphInvocationSpan",
    # metrics (ENG-16)
    "query_latency_seconds",
    "cost_per_query_usd",
    "fast_path_ratio",
    "agent_calls_total",
    "errors_total",
    "active_queries",
    "record_query_metrics",
    "extract_metrics_from_result",
    "get_metrics_app",
    "MetricsMiddleware",
]
