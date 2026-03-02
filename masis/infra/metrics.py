"""
masis.infra.metrics
====================
Prometheus metrics and instrumentation helpers for MASIS (ENG-16 M2).

Defines four core metric instruments aligned with the spec:

    query_latency_seconds   -- Histogram of end-to-end query latency
    cost_per_query_usd      -- Histogram of per-query cost in USD
    fast_path_ratio         -- Gauge tracking the ratio of Fast Path decisions
    agent_calls_total       -- Counter of agent dispatches, labelled by agent_type

Also provides:

    record_query_metrics()  -- convenience function to observe all metrics at once
    get_metrics_app()       -- returns a Starlette app that serves /metrics
    MetricsMiddleware       -- ASGI middleware that auto-instruments HTTP requests

When ``prometheus_client`` is not installed, all metric objects are replaced by
lightweight no-op stubs so the rest of MASIS can import this module unconditionally.

MF-IDs
------
MF-API-07  Prometheus metrics (query_latency, cost, agent_call_count, fast_path_ratio)
MF-API-08  Model routing config (cross-ref; metrics tag the model used)

Architecture reference
----------------------
final_architecture_and_flow.md Section 23.10
engineering_tasks.md ENG-16 M2
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric definitions (with graceful fallback)
# ---------------------------------------------------------------------------

_PROMETHEUS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    from prometheus_client import make_asgi_app as _make_prom_asgi_app
    from prometheus_client import REGISTRY

    _PROMETHEUS_AVAILABLE = True
    logger.debug("metrics: prometheus_client is available")
except ImportError:
    logger.info(
        "metrics: prometheus_client is not installed. "
        "All metrics will be no-ops. pip install prometheus-client to enable."
    )

# ---------------------------------------------------------------------------
# Metric bucket definitions (matching the spec exactly)
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
_COST_BUCKETS = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00)

# ---------------------------------------------------------------------------
# No-op metric stubs
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """Stub that silently accepts any metric operation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._value: float = 0.0

    def observe(self, amount: float = 0.0, **kwargs: Any) -> None:
        self._value = amount

    def inc(self, amount: float = 1.0, **kwargs: Any) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0, **kwargs: Any) -> None:
        self._value -= amount

    def set(self, value: float = 0.0, **kwargs: Any) -> None:
        self._value = value

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
        return self

    def info(self, val: Dict[str, str], **kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"_NoOpMetric(value={self._value})"


# ---------------------------------------------------------------------------
# Create metric instances
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    query_latency_seconds: Any = Histogram(
        "masis_query_latency_seconds",
        "End-to-end query latency in seconds",
        buckets=_LATENCY_BUCKETS,
    )
    cost_per_query_usd: Any = Histogram(
        "masis_cost_per_query_usd",
        "Total cost per query in USD",
        buckets=_COST_BUCKETS,
    )
    fast_path_ratio: Any = Gauge(
        "masis_fast_path_ratio",
        "Ratio of Fast Path decisions vs total supervisor decisions",
    )
    agent_calls_total: Any = Counter(
        "masis_agent_calls_total",
        "Total number of agent dispatch calls",
        ["agent_type"],
    )
    errors_total: Any = Counter(
        "masis_errors_total",
        "Total errors by error type",
        ["error_type"],
    )
    active_queries: Any = Gauge(
        "masis_active_queries",
        "Number of currently running graph invocations",
    )
    masis_info: Any = Info(
        "masis_build",
        "MASIS build and version information",
    )
    # Set build info once at import time
    masis_info.info({"version": "0.1.0", "phase": "4"})
else:
    query_latency_seconds = _NoOpMetric()
    cost_per_query_usd = _NoOpMetric()
    fast_path_ratio = _NoOpMetric()
    agent_calls_total = _NoOpMetric()
    errors_total = _NoOpMetric()
    active_queries = _NoOpMetric()
    masis_info = _NoOpMetric()


# ---------------------------------------------------------------------------
# Convenience: record all metrics for a completed query
# ---------------------------------------------------------------------------

def record_query_metrics(
    latency_seconds: float,
    cost_usd: float,
    agent_type_counts: Dict[str, int],
    fast_path_decisions: int = 0,
    total_decisions: int = 1,
    error_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Observe all core metrics for a completed MASIS query.

    This is the primary instrumentation call made by the API layer after a
    graph invocation completes (or fails).

    Parameters
    ----------
    latency_seconds : float
        Wall-clock time for the full query in seconds.
    cost_usd : float
        Total cost of the query in US dollars.
    agent_type_counts : dict
        Mapping of agent type names to the number of times they were called
        during this query (e.g. ``{"researcher": 3, "skeptic": 1}``).
    fast_path_decisions : int
        Number of supervisor decisions resolved by the Fast Path.
    total_decisions : int
        Total supervisor decisions (fast + slow). Used to compute ratio.
    error_type : str, optional
        If the query failed, the error classification string.

    Returns
    -------
    dict
        A summary of the recorded metric values (useful for logging/testing).
    """
    query_latency_seconds.observe(latency_seconds)
    cost_per_query_usd.observe(cost_usd)

    ratio = fast_path_decisions / max(total_decisions, 1)
    fast_path_ratio.set(ratio)

    for agent_type, count in agent_type_counts.items():
        for _ in range(count):
            agent_calls_total.labels(agent_type=agent_type).inc()

    if error_type is not None:
        errors_total.labels(error_type=error_type).inc()

    summary = {
        "latency_seconds": latency_seconds,
        "cost_usd": cost_usd,
        "agent_type_counts": agent_type_counts,
        "fast_path_ratio": ratio,
        "error_type": error_type,
    }
    logger.info("record_query_metrics: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Extract metrics from a completed MASISState result
# ---------------------------------------------------------------------------

def extract_metrics_from_result(
    result: Dict[str, Any],
    start_time: float,
) -> Dict[str, Any]:
    """Extract metric values from a completed graph invocation result.

    Reads the standard MASISState fields (token_budget, api_call_counts,
    decision_log, iteration_count) and computes the values needed by
    ``record_query_metrics``.

    Parameters
    ----------
    result : dict
        The final MASISState dict returned by ``ainvoke_graph``.
    start_time : float
        The ``time.time()`` recorded when the query started.

    Returns
    -------
    dict
        A dict with keys: latency_seconds, cost_usd, agent_type_counts,
        fast_path_decisions, total_decisions.
    """
    latency = time.time() - start_time

    budget = result.get("token_budget", {})
    cost_usd = budget.get("total_cost_usd", 0.0) if isinstance(budget, dict) else 0.0

    api_counts = result.get("api_call_counts", {})
    agent_type_counts: Dict[str, int] = {}
    if isinstance(api_counts, dict):
        for agent_type, count_val in api_counts.items():
            agent_type_counts[agent_type] = int(count_val) if isinstance(count_val, (int, float)) else 0

    decision_log = result.get("decision_log", [])
    fast_path_decisions = 0
    total_decisions = 0
    if isinstance(decision_log, list):
        for entry in decision_log:
            if isinstance(entry, dict):
                total_decisions += 1
                mode = entry.get("mode", "")
                if mode == "fast":
                    fast_path_decisions += 1

    return {
        "latency_seconds": latency,
        "cost_usd": cost_usd,
        "agent_type_counts": agent_type_counts,
        "fast_path_decisions": fast_path_decisions,
        "total_decisions": max(total_decisions, 1),
    }


# ---------------------------------------------------------------------------
# /metrics endpoint (Starlette ASGI sub-app for Prometheus scraping)
# ---------------------------------------------------------------------------

def get_metrics_app() -> Any:
    """Return a Starlette ASGI application that serves Prometheus metrics.

    Mount this under your FastAPI app at ``/metrics``::

        from masis.infra.metrics import get_metrics_app
        app.mount("/metrics", get_metrics_app())

    If ``prometheus_client`` is not installed, returns a minimal ASGI app
    that responds with 503 and a helpful message.

    Returns
    -------
    ASGIApp
        An ASGI application suitable for mounting in FastAPI/Starlette.
    """
    if _PROMETHEUS_AVAILABLE:
        return _make_prom_asgi_app()

    # Fallback: return a minimal app that explains metrics are disabled
    async def _no_metrics_app(scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] == "http":
            body = b"prometheus_client not installed. Metrics are disabled."
            await send({
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    [b"content-type", b"text/plain"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })

    return _no_metrics_app


# ---------------------------------------------------------------------------
# ASGI Middleware for HTTP request instrumentation
# ---------------------------------------------------------------------------

# Request-level histogram (separate from query_latency which measures graph invocations)
if _PROMETHEUS_AVAILABLE:
    _http_request_duration: Any = Histogram(
        "masis_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint", "status_code"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _http_requests_total: Any = Counter(
        "masis_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
else:
    _http_request_duration = _NoOpMetric()
    _http_requests_total = _NoOpMetric()


class MetricsMiddleware:
    """ASGI middleware that records HTTP request duration and counts.

    Attach to the FastAPI app::

        from masis.infra.metrics import MetricsMiddleware
        app.add_middleware(MetricsMiddleware)

    Records:
        - ``masis_http_request_duration_seconds`` (histogram)
        - ``masis_http_requests_total`` (counter)

    Both are labelled by HTTP method, endpoint path, and response status code.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable,
        send: Callable,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        status_code = "500"  # default until we capture the actual response
        start = time.time()

        async def _send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = str(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.time() - start
            _http_request_duration.labels(
                method=method, endpoint=path, status_code=status_code,
            ).observe(duration)
            _http_requests_total.labels(
                method=method, endpoint=path, status_code=status_code,
            ).inc()
